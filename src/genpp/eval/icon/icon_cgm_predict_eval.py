#!/usr/bin/env python
"""
Predict and evaluate with Flow Matching, CGM, or Engression models on ICON data.

This script loads a trained model from a WandB run, runs predictions on the
specified data split, computes evaluation metrics (CRPS, Energy Score,
Variogram Score), and logs results to WandB and local files.

Unlike cgm_predict_eval.py (which uses WeatherBench2 data), this script handles
the ICON dataset structure where:
  - Ground truth is loaded directly from per-date .pt rea tensor files via dataset.samples
  - Both predictions and ground truth are rescaled to the original space before scoring
  - Leadtimes are extracted from the dataset sample tuples
  - Predictions are saved as NetCDF with proper dimension labels

Usage:
    python icon_predict_eval.py --run-path feik/genpp/abc123 --split val
    python icon_predict_eval.py --run-path feik/genpp/abc123 feik/genpp/def456 --split val test
    python icon_predict_eval.py --run-path feik/genpp/abc123 --split test --skip-variogram
    python icon_predict_eval.py --run-path feik/genpp/abc123 --batch-size 32 -v
"""

import argparse
import importlib
import inspect

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import reduce
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.eval.utils import (
    log_scores,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import (
    EnergyScore,
    EnsembleCRPS,
    MultiScaleEnergyScore,
    MultiScalePatchwiseEnergyScore,
    PatchwiseEnergyScore,
    VariogramScore,
)

# Grid definition from target_grid.txt (rotated lat/lon grid)
GRID_XSIZE = 260
GRID_YSIZE = 240
GRID_XFIRST = -16.64
GRID_YFIRST = -15.0
GRID_XINC = 0.13
GRID_YINC = 0.13


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict and evaluate with Flow Matching, CGM, or Engression models on ICON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        type=str,
        nargs="+",
        required=True,
        help="WandB run path(s) (e.g., 'feik/genpp/abc123' or multiple separated by spaces)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Dataset split(s) to evaluate (e.g., --split val test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--skip-variogram",
        action="store_true",
        help="Skip variogram score computation (faster)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )
    parser.add_argument(
        "--force-repredict",
        action="store_true",
        help="Force re-running the model forward pass even if saved predictions exist",
    )
    parser.add_argument(
        "--leadtimes",
        type=int,
        nargs="+",
        default=None,
        help="Subset of leadtimes (in hours) to evaluate (e.g., --leadtimes 6 12 24). If not set, all leadtimes are used.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=40,
        help="Number of ensemble samples to predict (default 40 for ICON)",
    )
    return parser.parse_args()


def filter_samples_by_leadtimes(samples: list, leadtimes_hours: list[int]) -> list:
    """Filter dataset samples to only include specified leadtimes.

    Args:
        samples: List of (fc_path, rea_path, init_date, leadtime) tuples.
        leadtimes_hours: List of leadtime values in hours to keep.

    Returns:
        Filtered list of samples.
    """
    lt_set = {np.timedelta64(h, "h") for h in leadtimes_hours}
    return [s for s in samples if s[3] in lt_set]


def log_msg(msg: str, verbose: bool) -> None:
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)


def get_split_config(split: str) -> dict:
    """Get configuration for the specified split.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        Dict with setup_stage, dataloader_method, and dataset_attr
    """
    config = {
        "train": {
            "setup_stage": "fit",
            "dataloader_method": "train_dataloader",
            "dataset_attr": "train_dataset",
        },
        "val": {
            "setup_stage": "validate",
            "dataloader_method": "val_dataloader",
            "dataset_attr": "val_dataset",
        },
        "test": {
            "setup_stage": "test",
            "dataloader_method": "test_dataloader",
            "dataset_attr": "test_dataset",
        },
    }
    return config[split]


def _rescale_y(y: torch.Tensor, reverse_modules: list) -> torch.Tensor:
    """Rescale normalized y values back to original space.

    Applies the reverse affine transform per variable channel.

    Args:
        y: Normalized tensor with shape (..., c, x, y).
        reverse_modules: List of ReverseAffineTransform modules, one per channel.

    Returns:
        Rescaled tensor in original space.
    """
    y_rescaled = y.clone()
    for i, mod in enumerate(reverse_modules):
        y_rescaled[..., i, :, :] = y_rescaled[..., i, :, :] * mod.scale + mod.mean
    return y_rescaled


def _rescale_y_batched(
    y: torch.Tensor, reverse_modules: list, batch_size: int = 64
) -> torch.Tensor:
    """Rescale normalized y values back to original space in CPU batches.

    Processes the tensor in chunks along dim-0 to avoid OOM when the full
    tensor is too large to clone at once.

    Args:
        y: Normalized tensor with shape (N, ..., c, x, y) on CPU.
        reverse_modules: List of ReverseAffineTransform modules, one per channel.
        batch_size: Number of samples to rescale at a time.

    Returns:
        Rescaled tensor (CPU) in original space, same shape as input.
    """
    n = y.shape[0]
    out = torch.empty_like(y)
    for start in trange(0, n, batch_size, desc="Rescaling predictions"):
        end = min(start + batch_size, n)
        chunk = y[start:end].clone()
        for i, mod in enumerate(reverse_modules):
            chunk[..., i, :, :] = chunk[..., i, :, :] * mod.scale + mod.mean
        out[start:end] = chunk
    return out


def _build_grid_coords() -> tuple[np.ndarray, np.ndarray]:
    """Build rotated longitude/latitude coordinate arrays from the grid definition.

    Returns:
        Tuple of (rlon, rlat) coordinate arrays.
    """
    rlon = np.arange(GRID_XSIZE) * GRID_XINC + GRID_XFIRST
    rlat = np.arange(GRID_YSIZE) * GRID_YINC + GRID_YFIRST
    return rlon, rlat


def _load_ground_truth_from_samples(
    samples: list,
    verbose: bool = False,
) -> torch.Tensor:
    """Load raw ground truth tensors directly from dataset sample rea_paths.

    This avoids the overhead of a DataLoader and guarantees identical ordering
    to the predictions since we iterate samples in sequential order.

    Args:
        samples: List of (fc_path, rea_path, init_date, leadtime) tuples.
        verbose: Whether to show progress bar.

    Returns:
        Tensor of shape [N, c, x, y] with raw (unscaled) ground truth.
    """
    y_list = []
    for sample in tqdm(
        samples, desc="Loading raw ground truth from rea files", disable=not verbose
    ):
        rea_path = sample[1]
        rea = torch.load(rea_path, weights_only=True)  # shape [c, x, y]
        y_list.append(rea)
    return torch.stack(y_list, dim=0)  # shape [N, c, x, y]


def _predictions_to_xarray(
    predictions: torch.Tensor,
    samples: list,
    y_select_variables: list[str],
) -> xr.Dataset:
    """Convert prediction tensor to an xarray Dataset with proper coordinates.

    Args:
        predictions: Tensor of shape [N, n_samples, c, x, y] in original space.
        samples: List of (fc_path, rea_path, init_date, leadtime) tuples.
        y_select_variables: List of target variable names.

    Returns:
        xr.Dataset with predictions and coordinate labels.
    """
    rlon, rlat = _build_grid_coords()

    # Extract metadata from samples
    init_dates = np.array([s[2] for s in samples])  # np.datetime64
    leadtimes = np.array([s[3] for s in samples])  # np.timedelta64
    valid_times = init_dates + leadtimes

    n_samples = predictions.shape[1]

    pred_np = predictions.cpu().numpy()

    ds = xr.Dataset(
        {
            "prediction": xr.DataArray(
                data=pred_np,
                dims=["time", "sample", "variable", "rlon", "rlat"],
                coords={
                    "time": valid_times,
                    "sample": np.arange(n_samples),
                    "variable": y_select_variables,
                    "rlon": rlon,
                    "rlat": rlat,
                },
            ),
        },
        attrs={
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_longitude": -170.0,
            "grid_north_pole_latitude": 40.0,
            "description": "Ensemble predictions on ICON rotated lat/lon grid",
        },
    )
    # Add init_date and leadtime as non-dimension coordinates
    ds = ds.assign_coords(
        init_date=("time", init_dates),
        leadtime=("time", leadtimes),
    )

    return ds


def _crps_maps_to_xarray(
    crps_maps: torch.Tensor,
    samples: list,
    y_select_variables: list[str],
) -> xr.Dataset:
    """Convert per-sample CRPS maps to init_time/lead_time gridded xarray.

    Args:
        crps_maps: Tensor of shape [N, c, x, y] with per-location CRPS values.
        samples: List of (fc_path, rea_path, init_date, leadtime) tuples.
        y_select_variables: List of target variable names.

    Returns:
        xr.Dataset with one data variable ``crps`` and dimensions
        [init_time, lead_time, variable, rlon, rlat].

    Raises:
        ValueError: If duplicate (init_time, lead_time) pairs are found.
    """
    init_times = np.array([s[2] for s in samples])
    lead_times = np.array([s[3] for s in samples])

    pair_index = pd.MultiIndex.from_arrays(
        [init_times, lead_times], names=["init_time", "lead_time"]
    )
    dup_mask = pair_index.duplicated(keep=False)
    if dup_mask.any():
        dup_pairs = pair_index[dup_mask].unique()
        preview = ", ".join([f"({i}, {l})" for i, l in dup_pairs[:5]])  # noqa: E741
        raise ValueError(
            "Duplicate (init_time, lead_time) pairs found in ICON samples. "
            f"Expected unique pairs but found {len(dup_pairs)} duplicates. "
            f"Examples: {preview}"
        )

    unique_init = np.sort(np.unique(init_times))
    unique_lead = np.sort(np.unique(lead_times))

    init_to_idx = {val: i for i, val in enumerate(unique_init)}
    lead_to_idx = {val: i for i, val in enumerate(unique_lead)}

    crps_np = crps_maps.cpu().numpy().astype(np.float32)
    n_var, n_x, n_y = crps_np.shape[1:]

    crps_grid = np.full(
        (len(unique_init), len(unique_lead), n_var, n_x, n_y),
        np.nan,
        dtype=np.float32,
    )

    for sample_idx, (init_time, lead_time) in enumerate(zip(init_times, lead_times)):
        i = init_to_idx[init_time]
        j = lead_to_idx[lead_time]
        crps_grid[i, j] = crps_np[sample_idx]

    rlon, rlat = _build_grid_coords()

    ds = xr.Dataset(
        {
            "crps": xr.DataArray(
                data=crps_grid,
                dims=["init_time", "lead_time", "variable", "rlon", "rlat"],
                coords={
                    "init_time": unique_init,
                    "lead_time": unique_lead,
                    "variable": y_select_variables,
                    "rlon": rlon,
                    "rlat": rlat,
                },
            )
        },
        attrs={
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_longitude": -170.0,
            "grid_north_pole_latitude": 40.0,
            "description": "Per-location CRPS maps on ICON rotated lat/lon grid",
        },
    )

    valid_time = unique_init[:, None] + unique_lead[None, :]
    ds = ds.assign_coords(valid_time=(("init_time", "lead_time"), valid_time))
    return ds


def compute_icon_scores_per_leadtime(
    prediction_timedeltas: np.ndarray,
    crpss: torch.Tensor,
    ess_per_var: torch.Tensor,
    ess_complete: torch.Tensor,
    pess_per_var: torch.Tensor,
    pess_complete: torch.Tensor,
    msess_per_var: torch.Tensor,
    msess_complete: torch.Tensor,
    mspess_per_var: torch.Tensor,
    mspess_complete: torch.Tensor,
    y_select_variables: list[str],
    vss_per_var: torch.Tensor | None = None,
    vss_complete: torch.Tensor | None = None,
    method: str | None = None,
) -> dict:
    """Compute scores per lead time for the ICON dataset.

    Uses dynamic variable names from y_select_variables instead of hardcoded
    weatherbench variable names.

    Args:
        prediction_timedeltas: Array of prediction timedeltas (np.timedelta64).
        crpss: CRPS scores with shape (time, feature, x, y).
        ess_per_var: Energy scores per variable with shape (time, feature).
        ess_complete: Energy scores complete with shape (time,).
        pess_per_var: Patchwise energy scores per variable with shape (time, feature).
        pess_complete: Patchwise energy scores complete with shape (time,).
        msess_per_var: Multi-scale energy scores per variable with shape (time, feature).
        msess_complete: Multi-scale energy scores complete with shape (time,).
        mspess_per_var: Multi-scale patchwise energy scores per variable with shape (time, feature).
        mspess_complete: Multi-scale patchwise energy scores complete with shape (time,).
        y_select_variables: List of target variable names.
        vss_per_var: Variogram scores per variable with shape (time, feature).
        vss_complete: Variogram scores complete with shape (time,).
        method: Method name for the scores.

    Returns:
        dict: Dictionary with scores per lead time.
    """
    td = np.unique(prediction_timedeltas)
    td_str = [f"{t / np.timedelta64(1, 'h'):.0f}h" for t in td]

    # Build score keys dynamically from variable names
    scores_delta: dict = {
        method: {
            "CRPS_combined": {},
            "EnergyScore_combined": {},
            "PatchwiseEnergyScore_combined": {},
            "MultiScaleEnergyScore_combined": {},
            "MultiScalePatchwiseEnergyScore_combined": {},
        }
    }
    for var_name in y_select_variables:
        scores_delta[method][f"CRPS_{var_name}"] = {}
        scores_delta[method][f"EnergyScore_{var_name}"] = {}
        scores_delta[method][f"PatchwiseEnergyScore_{var_name}"] = {}
        scores_delta[method][f"MultiScaleEnergyScore_{var_name}"] = {}
        scores_delta[method][f"MultiScalePatchwiseEnergyScore_{var_name}"] = {}

    if vss_per_var is not None and vss_complete is not None:
        scores_delta[method]["VariogramScore_combined"] = {}
        for var_name in y_select_variables:
            scores_delta[method][f"VariogramScore_{var_name}"] = {}

    for delta, delta_str in tqdm(zip(td, td_str), total=len(td), desc="Processing leadtimes"):
        mask = prediction_timedeltas == delta
        crpss_delta = crpss[mask]
        ess_per_var_delta = ess_per_var[mask]
        ess_complete_delta = ess_complete[mask]
        pess_per_var_delta = pess_per_var[mask]
        pess_complete_delta = pess_complete[mask]
        msess_per_var_delta = msess_per_var[mask]
        msess_complete_delta = msess_complete[mask]
        mspess_per_var_delta = mspess_per_var[mask]
        mspess_complete_delta = mspess_complete[mask]

        scores_delta[method]["CRPS_combined"][delta_str] = reduce(
            crpss_delta, "t f x y -> 1", "mean"
        ).item()
        for vi, var_name in enumerate(y_select_variables):
            scores_delta[method][f"CRPS_{var_name}"][delta_str] = reduce(
                crpss_delta, "t f x y -> f", "mean"
            )[vi].item()

        scores_delta[method]["EnergyScore_combined"][delta_str] = ess_complete_delta.mean(
            dim=0
        ).item()
        for vi, var_name in enumerate(y_select_variables):
            scores_delta[method][f"EnergyScore_{var_name}"][delta_str] = ess_per_var_delta.mean(
                dim=0
            )[vi].item()

        scores_delta[method]["PatchwiseEnergyScore_combined"][delta_str] = pess_complete_delta.mean(
            dim=0
        ).item()
        for vi, var_name in enumerate(y_select_variables):
            scores_delta[method][f"PatchwiseEnergyScore_{var_name}"][delta_str] = (
                pess_per_var_delta.mean(dim=0)[vi].item()
            )

        scores_delta[method]["MultiScaleEnergyScore_combined"][delta_str] = (
            msess_complete_delta.mean(dim=0).item()
        )
        for vi, var_name in enumerate(y_select_variables):
            scores_delta[method][f"MultiScaleEnergyScore_{var_name}"][delta_str] = (
                msess_per_var_delta.mean(dim=0)[vi].item()
            )

        scores_delta[method]["MultiScalePatchwiseEnergyScore_combined"][delta_str] = (
            mspess_complete_delta.mean(dim=0).item()
        )
        for vi, var_name in enumerate(y_select_variables):
            scores_delta[method][f"MultiScalePatchwiseEnergyScore_{var_name}"][delta_str] = (
                mspess_per_var_delta.mean(dim=0)[vi].item()
            )

        if vss_per_var is not None and vss_complete is not None:
            vss_per_var_delta = vss_per_var[mask]
            vss_complete_delta = vss_complete[mask]
            scores_delta[method]["VariogramScore_combined"][delta_str] = vss_complete_delta.mean(
                dim=0
            ).item()
            for vi, var_name in enumerate(y_select_variables):
                scores_delta[method][f"VariogramScore_{var_name}"][delta_str] = (
                    vss_per_var_delta.mean(dim=0)[vi].item()
                )

    if method is None:
        return scores_delta[None]
    return scores_delta


def evaluate_split(
    split: str,
    *,
    model,
    trainer: L.Trainer,
    datamodule,
    cfg: DictConfig,
    score_file,
    model_dir,
    skip_variogram: bool,
    save_predictions: bool,
    force_repredict: bool,
    verbose: bool,
) -> dict:
    """Run prediction and evaluation for a single data split on ICON data.

    Ground truth is loaded directly from the rea tensor files referenced in
    dataset.samples. Predictions are rescaled to the original space before
    computing evaluation scores.

    Returns:
        Dict of scores per leadtime for this split.
    """
    split_config = get_split_config(split)
    dataset = getattr(datamodule, split_config["dataset_attr"])
    samples = dataset.samples
    y_select_variables = list(cfg.data.y_select_variables)

    # Check for cached predictions (.nc format)
    predictions_path = model_dir / f"{split}_predictions.nc"
    use_cached = predictions_path.exists() and not force_repredict

    if use_cached:
        log_msg(f"Loading cached predictions from {predictions_path}...", verbose)
        log_msg("Using cached predictions; skipping model forward pass.", verbose)
        ds = xr.open_dataset(predictions_path)
        predictions_rescaled = torch.from_numpy(ds["prediction"].values)
        ds.close()

        # Load ground truth directly from rea files in dataset.samples
        log_msg("Loading raw ground truth from dataset samples...", verbose)
        y_original = _load_ground_truth_from_samples(samples, verbose=verbose)
    else:
        log_msg(
            f"No cached predictions found at {predictions_path} (or force_repredict set); running model forward pass.",
            verbose,
        )
        # Create a dedicated eval dataloader with shuffle=False to ensure
        # predictions align with ground truth ordering.
        eval_dataloader = DataLoader(
            dataset,
            batch_size=datamodule.val_batch_size or datamodule.batch_size,
            shuffle=False,
            num_workers=datamodule.num_workers,
            pin_memory=datamodule.pin_memory,
            persistent_workers=datamodule.persistent_workers
            if datamodule.num_workers > 0
            else False,
        )

        # Load raw ground truth directly from rea files in dataset.samples
        # (same ordering as eval_dataloader since both use sequential access)
        log_msg("Loading raw ground truth from dataset samples...", verbose)
        y_original = _load_ground_truth_from_samples(samples, verbose=verbose)

        # Run predictions using the eval dataloader (sequential, no shuffle)
        log_msg(f"Running predictions on {split} split...", verbose)
        pred_list = trainer.predict(model, eval_dataloader, return_predictions=True)
        predictions = torch.cat(pred_list, dim=0)  # shape: [N, n_samples, c, x, y] # type: ignore

        # Rescale predictions to original space (ground truth is already in original space)
        # Process in batches on CPU to avoid OOM (full tensor can be >11 GB).
        log_msg("Rescaling predictions...", verbose)
        log_msg("Predictions shape: " + str(predictions.shape), verbose)
        reverse_modules = datamodule.y_reverseModules
        predictions_rescaled = _rescale_y_batched(predictions, reverse_modules, batch_size=64)

        del predictions
        torch.cuda.empty_cache()

    # Extract leadtimes from dataset samples
    timedeltas = np.array([sample[3] for sample in samples])

    # Compute scores (loop over time steps to avoid OOM from pairwise expansions)
    log_msg("Computing evaluation scores...", verbose)
    crps_ens = EnsembleCRPS().cuda()
    es = EnergyScore(clamp=False).cuda()
    pes = PatchwiseEnergyScore(clamp=False).cuda()
    mses = MultiScaleEnergyScore(clamp=False).cuda()
    mspes = MultiScalePatchwiseEnergyScore(clamp=False).cuda()
    # Use chunked variogram score on GPU to avoid OOM while maintaining speed
    vs = VariogramScore(p=0.5, chunk_size=256).cuda()

    n_times = predictions_rescaled.shape[0]
    crps_list, es_pv_list, es_full_list = [], [], []
    pes_pv_list, pes_full_list = [], []
    mses_pv_list, mses_full_list = [], []
    mspes_pv_list, mspes_full_list = [], []
    vs_pv_list, vs_full_list = [], []

    for i in trange(n_times, desc="Computing scores"):
        pred_i = predictions_rescaled[i : i + 1].cuda()  # [1, n_samples, c, x, y]
        y_i = y_original[i : i + 1].cuda()  # [1, c, x, y]
        with torch.no_grad():
            crps_list.append(crps_ens(pred_i, y_i).cpu())
            es_pv_list.append(es(pred_i, y_i, mode="per_var").cpu())
            es_full_list.append(es(pred_i, y_i, mode="complete").cpu())
            pes_pv_list.append(pes(pred_i, y_i, mode="per_var").cpu())
            pes_full_list.append(pes(pred_i, y_i, mode="complete").cpu())
            mses_pv_list.append(mses(pred_i, y_i, mode="per_var").cpu())
            mses_full_list.append(mses(pred_i, y_i, mode="complete").cpu())
            mspes_pv_list.append(mspes(pred_i, y_i, mode="per_var").cpu())
            mspes_full_list.append(mspes(pred_i, y_i, mode="complete").cpu())
            if not skip_variogram:
                # Chunked variogram score computation on GPU avoids OOM
                vs_pv_list.append(vs(pred_i, y_i, mode="per_var").cpu())
                vs_full_list.append(vs(pred_i, y_i, mode="complete").cpu())

    crps_per_margin = torch.cat(crps_list, dim=0)

    crps_maps_path = model_dir / f"{split}_crps_maps.nc"
    crps_maps_ds = _crps_maps_to_xarray(
        crps_maps=crps_per_margin,
        samples=samples,
        y_select_variables=y_select_variables,
    )
    crps_maps_ds.to_netcdf(crps_maps_path)
    log_msg(
        f"CRPS maps saved to {crps_maps_path} with shape {crps_maps_ds['crps'].shape}",
        verbose,
    )

    energy_score_per_var_u = torch.cat(es_pv_list, dim=0)
    energy_score_full_u = torch.cat(es_full_list, dim=0)
    patchwise_energy_score_per_var_u = torch.cat(pes_pv_list, dim=0)
    patchwise_energy_score_full_u = torch.cat(pes_full_list, dim=0)
    multiscale_energy_score_per_var_u = torch.cat(mses_pv_list, dim=0)
    multiscale_energy_score_full_u = torch.cat(mses_full_list, dim=0)
    multiscale_patchwise_energy_score_per_var_u = torch.cat(mspes_pv_list, dim=0)
    multiscale_patchwise_energy_score_full_u = torch.cat(mspes_full_list, dim=0)
    variogram_score_per_var_u = torch.cat(vs_pv_list, dim=0) if not skip_variogram else None
    variogram_score_full_u = torch.cat(vs_full_list, dim=0) if not skip_variogram else None

    # Reduce scores
    log_msg("Reducing scores...", verbose)
    crps_per_var = reduce(crps_per_margin, "t d x y -> d", reduction="mean")
    crps_full = reduce(crps_per_margin, "t d x y -> 1", "mean")
    energy_score_per_var = reduce(energy_score_per_var_u, "t d -> d", "mean")
    energy_score_full = reduce(energy_score_full_u, "t -> 1", "mean")
    patchwise_energy_score_per_var = reduce(patchwise_energy_score_per_var_u, "t d -> d", "mean")
    patchwise_energy_score_full = reduce(patchwise_energy_score_full_u, "t -> 1", "mean")
    multiscale_energy_score_per_var = reduce(multiscale_energy_score_per_var_u, "t d -> d", "mean")
    multiscale_energy_score_full = reduce(multiscale_energy_score_full_u, "t -> 1", "mean")
    multiscale_patchwise_energy_score_per_var = reduce(
        multiscale_patchwise_energy_score_per_var_u, "t d -> d", "mean"
    )
    multiscale_patchwise_energy_score_full = reduce(
        multiscale_patchwise_energy_score_full_u, "t -> 1", "mean"
    )

    if not skip_variogram:
        variogram_score_per_var = reduce(variogram_score_per_var_u, "t d -> d", "mean")
        variogram_score_full = reduce(variogram_score_full_u, "t -> 1", "mean")

    # Log scores to file
    log_msg(f"Logging scores to {score_file}...", verbose)
    model_class = cfg.model._target_.split(".")[-1]

    log_scores(
        file=score_file,
        model=model_class,
        metric="CRPS",
        variables=datamodule.y_select_variables,
        scores=crps_per_var,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="CRPS",
        variables=["combined"],
        scores=crps_full,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="EnergyScore",
        variables=datamodule.y_select_variables,
        scores=energy_score_per_var,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="EnergyScore",
        variables=["combined"],
        scores=energy_score_full,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="PatchwiseEnergyScore",
        variables=datamodule.y_select_variables,
        scores=patchwise_energy_score_per_var,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="PatchwiseEnergyScore",
        variables=["combined"],
        scores=patchwise_energy_score_full,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="MultiScaleEnergyScore",
        variables=datamodule.y_select_variables,
        scores=multiscale_energy_score_per_var,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="MultiScaleEnergyScore",
        variables=["combined"],
        scores=multiscale_energy_score_full,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="MultiScalePatchwiseEnergyScore",
        variables=datamodule.y_select_variables,
        scores=multiscale_patchwise_energy_score_per_var,
    )
    log_scores(
        file=score_file,
        model=model_class,
        metric="MultiScalePatchwiseEnergyScore",
        variables=["combined"],
        scores=multiscale_patchwise_energy_score_full,
    )

    if not skip_variogram:
        log_scores(
            file=score_file,
            model=model_class,
            metric="VariogramScore",
            variables=datamodule.y_select_variables,
            scores=variogram_score_per_var,  # type: ignore
        )
        log_scores(
            file=score_file,
            model=model_class,
            metric="VariogramScore",
            variables=["combined"],
            scores=variogram_score_full,  # type: ignore
        )

    # Compute scores per leadtime
    log_msg("Computing scores per leadtime...", verbose)
    scores = compute_icon_scores_per_leadtime(
        timedeltas,
        crps_per_margin,
        energy_score_per_var_u,
        energy_score_full_u,
        patchwise_energy_score_per_var_u,
        patchwise_energy_score_full_u,
        multiscale_energy_score_per_var_u,
        multiscale_energy_score_full_u,
        multiscale_patchwise_energy_score_per_var_u,
        multiscale_patchwise_energy_score_full_u,
        y_select_variables=y_select_variables,
        vss_per_var=variogram_score_per_var_u if not skip_variogram else None,
        vss_complete=variogram_score_full_u if not skip_variogram else None,
        method=None,
    )

    # Save predictions as NetCDF if requested (and they were freshly computed)
    if save_predictions and not use_cached:
        log_msg("Saving predictions as NetCDF...", verbose)
        ds = _predictions_to_xarray(predictions_rescaled, samples, y_select_variables)
        ds.to_netcdf(predictions_path)
        log_msg(f"Predictions saved to {predictions_path}", verbose)

    return scores


def process_run(run_path: str, args: argparse.Namespace) -> None:
    """Process a single WandB run: load model, predict, evaluate, and log results."""
    log_msg(f"\n{'#' * 60}\nProcessing run: {run_path}\n{'#' * 60}", args.verbose)

    # Parse run path
    model_id = run_path.split("/")[-1]
    output_dir = BASE_DIR.parent.parent / "outputs"

    log_msg(f"Looking for model with ID: {model_id}", args.verbose)

    # Find model directory
    model_dirs = list(output_dir.rglob(f"*{model_id}*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directory found for run ID '{model_id}' in {output_dir}")
    model_dir = model_dirs[0].parent.parent.parent

    # Find checkpoint
    checkpoints = list(model_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint (.ckpt) found in {model_dir}")
    model_checkpoint = checkpoints[0]

    score_file = model_dir / "scores.csv"

    log_msg(f"Model directory: {model_dir}", args.verbose)
    log_msg(f"Checkpoint: {model_checkpoint}", args.verbose)

    # Load Hydra config
    log_msg("Loading Hydra config...", args.verbose)
    with hydra.initialize_config_dir(config_dir=str(model_dir / ".hydra"), version_base=None):
        cfg: DictConfig = hydra.compose(config_name="config")

    # Configure batch sizes for evaluation
    if hasattr(cfg.data.module, "val_batch_size"):
        cfg.data.module.val_batch_size = args.batch_size
    if hasattr(cfg.data.module, "test_batch_size"):
        cfg.data.module.test_batch_size = args.batch_size

    # Setup datamodule
    log_msg("Setting up data module...", args.verbose)
    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()

    # ICON's setup() creates all splits regardless of stage, so one call suffices
    splits = args.split
    datamodule.setup(stage="fit")

    # Filter samples by leadtime if requested
    if args.leadtimes is not None:
        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            ds = getattr(datamodule, attr, None)
            if ds is not None:
                orig_len = len(ds.samples)
                ds.samples = filter_samples_by_leadtimes(ds.samples, args.leadtimes)
                log_msg(
                    f"Filtered {attr}: {orig_len} \u2192 {len(ds.samples)} samples "
                    f"(leadtimes={args.leadtimes}h)",
                    args.verbose,
                )

    # Load model
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    class_path = cfg.model._target_
    module_path, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    # Build model_kwargs dynamically from the model's __init__ signature and config.
    # Supply a default EnergyScore for loss_fn since it is not used during
    # inference but some models (e.g. CNNChenNoiseModel) require it as a
    # positional argument.  Using a plain EnergyScore avoids state_dict
    # mismatches that would occur with the original (possibly kernel-based) loss.
    sig = inspect.signature(ModelClass.__init__)
    model_kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name == "loss_fn":
            model_kwargs["loss_fn"] = EnergyScore()
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param_name not in cfg.model or OmegaConf.is_missing(cfg.model, param_name):
            if param.default is inspect.Parameter.empty:
                model_kwargs[param_name] = None
            continue

        value = cfg.model[param_name]

        if isinstance(value, DictConfig) and "_target_" in value:
            value = hydra.utils.instantiate(value)
        elif isinstance(value, ListConfig):
            value = tuple(value)

        model_kwargs[param_name] = value

    # ICON has 40 ensemble members, so we want to predict 40 samples for scoring
    model_kwargs["n_samples"] = 40

    # Load checkpoint state dict to inspect td_scaling buffer sizes before
    # constructing the model. FixedTDScaling registers lead_times/lookup_table
    # buffers with a default size that may differ from the checkpoint (which was
    # fitted on the actual training data). We resize the buffers to match before
    # loading the state dict so that load_state_dict does not raise a size
    # mismatch error.
    try:
        ckpt = torch.load(model_checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    ckpt_state_dict = ckpt.get("state_dict", ckpt)

    try:
        model_kwargs["n_samples_predict"] = 40
        model = ModelClass(**model_kwargs)
    except TypeError:
        model_kwargs.pop("n_samples_predict", None)
        model = ModelClass(**model_kwargs)

    # Resize internal_td_scaling buffers to match checkpoint shapes
    if hasattr(model, "internal_td_scaling"):
        for buf_name in ("lead_times", "lookup_table"):
            key = f"internal_td_scaling.{buf_name}"
            if key in ckpt_state_dict and hasattr(model.internal_td_scaling, buf_name):
                log_msg(
                    f"Resizing {key}: {getattr(model.internal_td_scaling, buf_name).shape}"
                    f" -> {ckpt_state_dict[key].shape}",
                    args.verbose,
                )
                model.internal_td_scaling.register_buffer(
                    buf_name, torch.zeros_like(ckpt_state_dict[key])
                )

    model.load_state_dict(ckpt_state_dict, strict=False)

    # Fix internal_td_scaling metadata if needed
    if hasattr(model, "internal_td_scaling"):
        td_scaling = model.internal_td_scaling
        if not getattr(td_scaling, "is_fitted", False):
            log_msg("Setting internal_td_scaling.is_fitted = True", args.verbose)
            td_scaling.is_fitted = True  # type: ignore
        if not getattr(td_scaling, "n_vars", False):
            log_msg("Setting internal_td_scaling.n_vars = 2", args.verbose)
            td_scaling.n_vars = 2  # type: ignore

    # Create trainer
    trainer = L.Trainer(logger=False, accelerator="gpu", devices="auto", enable_progress_bar=True)

    # Evaluate each requested split
    full_scores = {}
    for split in splits:
        log_msg(f"\n{'=' * 60}\nEvaluating split: {split}\n{'=' * 60}", args.verbose)
        scores = evaluate_split(
            split,
            model=model,
            trainer=trainer,
            datamodule=datamodule,
            cfg=cfg,
            score_file=score_file,
            model_dir=model_dir,
            skip_variogram=args.skip_variogram,
            save_predictions=args.save_predictions,
            force_repredict=args.force_repredict,
            verbose=args.verbose,
        )
        full_scores[split] = scores

    # Update WandB run with all split scores
    log_msg("Updating WandB run...", args.verbose)
    update_wandb_run(run_path, full_scores)

    # Save scores DataFrame
    records = []
    for dataset, metrics in full_scores.items():
        for metric_name, horizons in metrics.items():
            for horizon, value in horizons.items():
                records.append(
                    (f"{model.__class__.__name__}", dataset, metric_name, horizon, value)
                )
    df = pd.DataFrame(records, columns=["method", "dataset", "metric", "horizon", "value"])
    save_scores_df(df=df, run_path=run_path)

    log_msg(f"Done with run: {run_path}", args.verbose)


def main() -> None:
    """Main entry point for the ICON prediction script."""
    args = parse_args()

    # Register Hydra resolvers
    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    for run_path in args.run_path:
        process_run(run_path, args)

    log_msg("All runs completed!", args.verbose)


if __name__ == "__main__":
    main()
