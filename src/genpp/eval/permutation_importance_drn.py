#!/usr/bin/env python
"""
Permutation-based feature importance for DRN/EMOS distributional regression models.

These models predict distribution parameters (mu, sigma) rather than samples
directly. To evaluate with Energy Score, predictions must first be converted to
forecast samples via Ensemble Copula Coupling (ECC) or Gaussian Copula Approach
(GCA).

The script follows the same importance formula as ``permutation_importance.py``::

    importance = (ES_permuted - ES_baseline) / ES_baseline

Usage:
    python -m genpp.eval.permutation_importance_drn --run-path feik/genpp/m5y9kwlh --split val -v
    python -m genpp.eval.permutation_importance_drn --run-path feik/genpp/m5y9kwlh --split val --copula ecc --n-repeats 3
    python -m genpp.eval.permutation_importance_drn --run-path feik/genpp/m5y9kwlh --split val --channels 0 1 2 --device 0 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from genpp.configs import register_resolvers
from genpp.data.weatherbench2.fast_dataset_simple import TransformTensorDataset
from genpp.eval import ModelEntry
from genpp.eval.copulas_eval import (
    do_ecc,
    do_gca,
    get_split_predictions_and_obs,
    predictions_to_dataarray,
    stack_predictions,
    transform_to_latent_gaussian,
)
from genpp.eval.permutation_importance import (
    _build_x_transform,
    _get_channel_info,
    log_msg,
    parse_device,
)
from genpp.models.scores import EnergyScore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Permutation-based feature importance for DRN/EMOS models using copula postprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="WandB run path (e.g., 'feik/genpp/m5y9kwlh')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--copula",
        type=str,
        default="ecc",
        choices=["ecc", "gca"],
        help="Copula method to use for generating forecast samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device index or comma-separated list (e.g., '0' or '0,1,2')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of permutation repeats per channel (for robust estimates)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of channel indices to permute (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <model_dir>/permutation_importance_drn.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def _compute_copula_energy_score(
    pred_samples_xr,
    y_obs_xr,
    device: str = "cpu",
) -> float:
    """Compute mean energy score from xarray copula predictions and observations.

    Loops over predictions one-by-one to avoid OOM from pairwise expansions.

    Args:
        pred_samples_xr: xarray DataArray with dims ``(prediction, sample, feature, lon, lat)``.
        y_obs_xr: xarray DataArray with dims ``(prediction, feature, lon, lat)``.
        device: device for computation.

    Returns:
        Scalar mean energy score.
    """
    es_fn = EnergyScore(clamp=False).to(device)
    es_list: list[torch.Tensor] = []

    pred_samples_xr = pred_samples_xr.transpose(
        "prediction", "sample", "feature", "longitude", "latitude"
    )
    predictions = y_obs_xr.prediction

    for p in predictions:
        obs = y_obs_xr.sel(prediction=p).to_numpy()
        pred = pred_samples_xr.sel(prediction=p).to_numpy()

        obs_t = torch.tensor(obs[None, ...], dtype=torch.float32, device=device)
        pred_t = torch.tensor(pred[None, ...], dtype=torch.float32, device=device)

        with torch.no_grad():
            es_list.append(es_fn(pred_t, obs_t, mode="complete").cpu())

    es_all = torch.cat(es_list, dim=0)
    return es_all.mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — sequential orchestration script
    """Entry point for DRN/EMOS permutation importance evaluation."""
    args = parse_args()

    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    # ----- locate model artefacts via ModelEntry -----
    model_entry = ModelEntry(id=args.run_path)
    model_dir = model_entry.model_dir
    model_checkpoint = model_entry.model_checkpoint

    log_msg(f"Model directory: {model_dir}", args.verbose)
    log_msg(f"Checkpoint:      {model_checkpoint}", args.verbose)

    # ----- load Hydra config -----
    log_msg("Loading Hydra config...", args.verbose)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(model_dir / ".hydra"), version_base=None):
        cfg: DictConfig = hydra.compose(config_name="config")

    cfg.data.module.dataloader_config.train.shuffle = False
    cfg.data.module.dataloader_config.val.shuffle = False
    cfg.data.module.dataloader_config.val.batch_size = args.batch_size
    cfg.data.module.dataloader_config.test.shuffle = False

    # ----- setup data -----
    log_msg("Setting up data module...", args.verbose)
    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")

    # Grab the original x_transform from the instantiated datamodule
    original_x_transform = datamodule.dataset_config.train.x_transform
    y_transform = datamodule.dataset_config.train.y_transform

    # ----- load model -----
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    model = model_entry.model

    devices = parse_device(args.device)
    device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
    trainer = L.Trainer(logger=False, accelerator="gpu", devices=devices)

    # ----- GCA requires Sigma from train split -----
    Sigma = None
    if args.copula == "gca":
        log_msg("Computing GCA covariance matrix from train split...", args.verbose)
        train_preds_xr, y_train, _ = get_split_predictions_and_obs(
            "train", model, trainer, datamodule, cfg, args.verbose
        )
        latent = transform_to_latent_gaussian(y_train, train_preds_xr)
        flat = latent.stack(space=("feature", "longitude", "latitude"))
        Sigma = np.cov(flat.values, rowvar=False)
        del latent, flat, train_preds_xr, y_train

    # ----- load cached tensors + metadata for channel info -----
    all_tensors = torch.load(datamodule.tensor_path)
    cache_metadata = datamodule.cache_metadata
    feature_metadata = cache_metadata["feature_metadata"]
    split = args.split

    x_tensor = all_tensors[split]["x"]
    y_tensor = all_tensors[split]["y"]
    td_tensor = all_tensors[split]["prediction_timedelta"]

    channel_info = _get_channel_info(cache_metadata)
    n_channels = x_tensor.shape[1]  # feature dim

    channels_to_permute = args.channels if args.channels is not None else list(range(n_channels))

    dl_kwargs = OmegaConf.to_container(
        getattr(cfg.data.module.dataloader_config, split), resolve=True
    )
    dl_kwargs["shuffle"] = False  # type: ignore

    # ----- baseline (no permutation) -----
    log_msg(
        f"Computing baseline energy score (no permutation, copula={args.copula})...", args.verbose
    )
    predictions_xr, y_obs, prediction_index = get_split_predictions_and_obs(
        split, model, trainer, datamodule, cfg, args.verbose
    )

    if args.copula == "ecc":
        baseline_samples = do_ecc(predictions_xr, prediction_index)
    else:
        baseline_samples = do_gca(Sigma, predictions_xr, y_obs.shape)

    baseline_es = _compute_copula_energy_score(baseline_samples, y_obs, device=str(device))
    log_msg(f"Baseline energy score: {baseline_es:.6f}", args.verbose)
    del baseline_samples

    # ----- permutation loop -----
    results: list[dict[str, Any]] = []

    for ch_idx in channels_to_permute:
        ch_name = (
            channel_info[ch_idx]["name"] if ch_idx < len(channel_info) else f"channel_{ch_idx}"
        )
        ch_cat = channel_info[ch_idx]["category"] if ch_idx < len(channel_info) else "unknown"
        log_msg(f"\nPermuting channel {ch_idx} ({ch_name})...", args.verbose)

        repeat_scores: list[float] = []
        for r in range(args.n_repeats):
            torch.manual_seed(args.seed + ch_idx * args.n_repeats + r)
            perm_transform = _build_x_transform(original_x_transform, ch_idx, seed=None)

            # Build a permuted dataset and dataloader
            perm_dataset = TransformTensorDataset(
                x_tensor,
                y_tensor,
                td_tensor,
                feature_metadata=feature_metadata,
                x_transform=perm_transform,
                y_transform=y_transform,
            )
            perm_dl = torch.utils.data.DataLoader(perm_dataset, **dl_kwargs)  # type: ignore

            # Run model predictions on the permuted data
            perm_predictions = trainer.predict(model, perm_dl, return_predictions=True)

            # Convert predictions to xarray (reuse copulas_eval helpers)
            stacked = stack_predictions(perm_predictions)
            perm_preds_xr = predictions_to_dataarray(y_obs, stacked)

            # Apply copula postprocessing
            if args.copula == "ecc":
                perm_samples = do_ecc(perm_preds_xr, prediction_index)
            else:
                perm_samples = do_gca(Sigma, perm_preds_xr, y_obs.shape)

            es_permuted = _compute_copula_energy_score(perm_samples, y_obs, device=str(device))
            repeat_scores.append(es_permuted)
            log_msg(f"  repeat {r}: ES={es_permuted:.6f}", args.verbose)
            del perm_samples, perm_preds_xr, stacked

        mean_es = float(np.mean(repeat_scores))
        std_es = float(np.std(repeat_scores)) if len(repeat_scores) > 1 else 0.0
        importance = (mean_es - baseline_es) / baseline_es if baseline_es != 0 else np.nan

        results.append(
            {
                "channel_index": ch_idx,
                "channel_name": ch_name,
                "category": ch_cat,
                "baseline_es": baseline_es,
                "permuted_es": mean_es,
                "importance": importance,
                "importance_std": std_es / abs(baseline_es) if baseline_es != 0 else np.nan,
            }
        )

        log_msg(
            f"  → importance = {importance:.4f} "
            f"(ES baseline={baseline_es:.6f}, permuted={mean_es:.6f})",
            args.verbose,
        )

    # ----- write results (append if file exists) -----
    output_path = Path(args.output) if args.output else model_dir / "permutation_importance_drn.csv"
    fieldnames = [
        "channel_index",
        "channel_name",
        "category",
        "baseline_es",
        "permuted_es",
        "importance",
        "importance_std",
    ]
    write_header = not output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # ----- print summary -----
    sorted_results = sorted(results, key=lambda r: r["importance"], reverse=True)
    print(f"\n--- Feature Importance (sorted by importance, copula={args.copula}) ---")
    print(f"{'Idx':>4}  {'Channel':<50}  {'Category':<15}  {'Importance':>10}")
    print("-" * 85)
    for r in sorted_results:
        print(
            f"{r['channel_index']:>4}  {r['channel_name']:<50}  "
            f"{r['category']:<15}  {r['importance']:>10.4f}"
        )


if __name__ == "__main__":
    main()
