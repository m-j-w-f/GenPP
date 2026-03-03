#!/usr/bin/env python
"""
ECC postprocessing evaluation for EMOS/DRN models on ICON data.

Loads EMOS/DRN models, runs predictions to obtain distribution parameters
(mu, sigma), generates quantile samples, applies Ensemble Copula Coupling
(ECC) using the raw ICON ensemble forecasts, and evaluates with CRPS,
Energy Score, and Variogram Score.

No GCA is applied for ICON (only ECC).

Usage:
    python icon_copulas_eval.py --run-path feik/genpp/abc123
    python icon_copulas_eval.py --run-path feik/genpp/abc123 feik/genpp/def456 --split val test
    python icon_copulas_eval.py --run-path feik/genpp/abc123 --split test --skip-variogram
    python icon_copulas_eval.py --run-path feik/genpp/abc123 --split test --save-predictions
    python icon_copulas_eval.py --run-path feik/genpp/abc123 --batch-size 32 -v
"""

import argparse
import importlib
import inspect
import pickle
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import reduce
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.stats import norm, truncnorm
from tqdm import trange

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.data.icon import DATA_DIR
from genpp.eval.icon.icon_cgm_predict_eval import (
    _load_ground_truth_from_samples,
    _predictions_to_xarray,
    _rescale_y,
    compute_icon_scores_per_leadtime,
    filter_samples_by_leadtimes,
    get_split_config,
)
from genpp.eval.icon.icon_raw_ensemble import load_ensemble_tensor
from genpp.eval.utils import (
    log_scores,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import EnergyScore, EnsembleCRPS, VariogramScore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ECC postprocessing evaluation for EMOS/DRN on ICON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        type=str,
        nargs="+",
        required=True,
        help="WandB run path(s) (e.g., 'feik/genpp/abc123')",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Dataset split(s) to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--n-quantile-samples",
        type=int,
        default=40,
        help="Number of quantile samples to generate (should match ensemble size)",
    )
    parser.add_argument(
        "--skip-variogram",
        action="store_true",
        help="Skip variogram score computation (faster)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save ECC predictions to file",
    )
    parser.add_argument(
        "--force-repredict",
        action="store_true",
        help="Force re-running model predictions even if cached",
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
    return parser.parse_args()


def log_msg(msg: str, verbose: bool) -> None:
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def stack_predictions(predictions: list) -> list[dict[str, torch.Tensor]]:
    """Stack batched predictions from trainer.predict() into per-variable dicts.

    The model returns a list of dicts (one per variable) for each batch.
    This function concatenates across batches.

    Args:
        predictions: List of batches, each batch is a list of dicts with
            keys like 'mu' and 'sigma', each value of shape [B, h, w].

    Returns:
        List of dicts (one per variable) with concatenated tensors [N, h, w].
    """
    n_dicts = len(predictions[0])
    merged: list[dict[str, list[torch.Tensor]]] = [defaultdict(list) for _ in range(n_dicts)]
    for batch in predictions:
        for i, d in enumerate(batch):
            for k, v in d.items():
                merged[i][k].append(v)
    return [{k: torch.cat(vs, dim=0) for k, vs in m.items()} for m in merged]


# ---------------------------------------------------------------------------
# Quantile sampling from predicted distributions
# ---------------------------------------------------------------------------


def quantile_samples_icon(
    stacked_preds: list[dict[str, torch.Tensor]],
    y_select_variables: list[str],
    M: int = 40,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Generate quantile samples from the predicted marginal distributions.

    ICON uses PredictiveCombinedDistribution which returns a Normal distribution
    for the first variable (T_2M) and a TruncatedNormal (lower=0) for the second
    (VMAX_10M).

    Processes samples in chunks to avoid OOM on large datasets (ICON grids are
    240x260 with ~1800 time steps, so the full [N, M, n_vars, h, w] array would
    require ~37 GB for M=40).

    Args:
        stacked_preds: List of per-variable dicts with keys 'mu' and 'sigma',
            each of shape [N, h, w].
        y_select_variables: List of target variable names (e.g., ['T_2M...', 'VMAX_10M...']).
        M: Number of quantile samples to generate.
        chunk_size: Number of samples to process at a time.

    Returns:
        Tensor of shape [N, M, n_vars, h, w] with quantile samples.
    """
    qs = np.linspace(1 / (M + 1), M / (M + 1), M)
    N = stacked_preds[0]["mu"].shape[0]
    n_vars = len(y_select_variables)
    h, w = stacked_preds[0]["mu"].shape[1], stacked_preds[0]["mu"].shape[2]

    result = torch.empty(N, M, n_vars, h, w, dtype=torch.float32)

    for start in trange(0, N, chunk_size, desc="Generating quantile samples"):
        end = min(start + chunk_size, N)

        for var_idx, var_name in enumerate(y_select_variables):
            mu = stacked_preds[var_idx]["mu"][start:end].cpu().numpy()  # [chunk, h, w]
            sigma = stacked_preds[var_idx]["sigma"][start:end].cpu().numpy()

            if "VMAX" in var_name or "wind" in var_name.lower():
                a = (0 - mu) / sigma
                b = np.inf
                samples = truncnorm.ppf(
                    qs[:, None, None, None],
                    a[None, ...],
                    b,
                    loc=mu[None, ...],
                    scale=sigma[None, ...],
                )  # [M, chunk, h, w]
            else:
                samples = norm.ppf(
                    qs[:, None, None, None],
                    loc=mu[None, ...],
                    scale=sigma[None, ...],
                )  # [M, chunk, h, w]

            # [M, chunk, h, w] -> [chunk, M, h, w]
            result[start:end, :, var_idx, :, :] = torch.from_numpy(
                samples.transpose(1, 0, 2, 3)
            ).float()

    return result


# ---------------------------------------------------------------------------
# Ensemble loading for ECC
# ---------------------------------------------------------------------------


def load_ensemble_for_sample(
    init_date: np.datetime64,
    leadtime: np.timedelta64,
    ens_dir: Path,
) -> torch.Tensor | None:
    """Load raw ensemble forecast for a specific init_date and leadtime.

    Uses the proven ``load_ensemble_tensor`` from ``raw_ensemble.py`` which
    handles variable stacking (T_2M, VMAX_10M) and dimension transposing.

    Args:
        init_date: Forecast initialization date.
        leadtime: Lead time as np.timedelta64.
        ens_dir: Path to the ensemble NetCDF directory.

    Returns:
        Tensor of shape [n_members, n_vars, x, y] or None if file not found.
    """
    init_str = pd.Timestamp(init_date).strftime("%Y%m%d%H")
    leadtime_hours = int(leadtime / np.timedelta64(1, "h"))
    ens_path = ens_dir / f"ens_{init_str}_{leadtime_hours}.nc"

    if not ens_path.exists():
        return None

    # load_ensemble_tensor already transposes to [members, n_vars, x, y]
    return load_ensemble_tensor(ens_path)


# ---------------------------------------------------------------------------
# ECC postprocessing
# ---------------------------------------------------------------------------


def do_ecc_icon(
    quantile_preds: torch.Tensor,
    dataset_samples: list,
    ens_dir: Path,
    verbose: bool = False,
) -> torch.Tensor:
    """Apply Ensemble Copula Coupling (ECC) for ICON data.

    For each sample, loads the corresponding ensemble forecast, ranks the
    ensemble members, and reorders the quantile samples according to the
    ensemble rank structure.

    Args:
        quantile_preds: Quantile samples [N, M, n_vars, h, w].
        dataset_samples: List of (fc_path, rea_path, init_date, leadtime) tuples.
        ens_dir: Path to the ensemble NetCDF directory.
        verbose: Whether to print verbose output.

    Returns:
        Reordered samples [N, M, n_vars, h, w].
    """
    N, M, n_vars, h, w = quantile_preds.shape
    result = quantile_preds  # Modify in-place to avoid doubling memory

    rng = np.random.default_rng(seed=420)
    n_missing = 0

    for i in trange(N, desc="ECC reordering"):
        _, _, init_date, leadtime = dataset_samples[i]

        # Load ensemble for this sample
        ensemble = load_ensemble_for_sample(init_date, leadtime, ens_dir)
        if ensemble is None:
            n_missing += 1
            continue

        n_members = ensemble.shape[0]

        # Validate spatial dimensions match on first loaded sample
        if i == 0 or (n_missing > 0 and n_missing == i):
            ens_h, ens_w = ensemble.shape[2], ensemble.shape[3]
            if ens_h != h or ens_w != w:
                raise ValueError(
                    f"Spatial dimension mismatch: predictions ({h}, {w}) "
                    f"vs ensemble ({ens_h}, {ens_w})"
                )
            if verbose:
                print(
                    f"  Ensemble shape: {ensemble.shape} "
                    f"(members={n_members}, vars={ensemble.shape[1]}, "
                    f"h={ens_h}, w={ens_w})"
                )

        # Add small noise to break ties, then rank
        noise = rng.uniform(low=-1e-8, high=1e-8, size=ensemble.shape)
        ensemble_noised = ensemble.numpy() + noise

        # Rank within ensemble members dimension (axis=0): rank 0..n_members-1
        # Use argsort of argsort to get ranks
        ranks = np.argsort(np.argsort(ensemble_noised, axis=0), axis=0)  # [n_members, n_vars, y, x]

        # Sort quantile samples along sample dimension for each spatial location
        sorted_preds = torch.sort(quantile_preds[i], dim=0).values  # [M, n_vars, h, w]

        # Reorder: use the ensemble ranks to pick from sorted predictions
        # Only use the first M ranks (or first n_members if M > n_members)
        n_use = min(M, n_members)
        rank_indices = torch.from_numpy(ranks[:n_use]).long()  # [n_use, n_vars, h, w]

        # Clamp rank indices to valid range [0, M-1]
        rank_indices = rank_indices.clamp(0, M - 1)

        # Gather: for each spatial location and variable, reorder samples
        reordered = torch.gather(sorted_preds, 0, rank_indices)  # [n_use, n_vars, h, w]

        # If we have fewer ensemble members than quantile samples, pad with remaining
        if n_use < M:
            result[i, :n_use] = reordered
        else:
            result[i] = reordered[:M]

    if n_missing > 0:
        print(
            f"Warning: {n_missing}/{N} samples had no ensemble data. "
            "Using original quantile ordering for those."
        )

    return result


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_ecc_scores(
    ecc_preds: torch.Tensor,
    y_rescaled: torch.Tensor,
    timedeltas: np.ndarray,
    y_select_variables: list[str],
    *,
    score_file: Path,
    model_class: str,
    skip_variogram: bool,
) -> dict:
    """Compute CRPS, Energy Score, and Variogram Score for ECC predictions.

    Args:
        ecc_preds: ECC predictions [N, M, n_vars, h, w].
        y_rescaled: Rescaled ground truth [N, n_vars, h, w].
        timedeltas: Array of lead times [N].
        y_select_variables: List of target variable names.
        score_file: Path to the scores CSV file.
        model_class: Name of the model class.
        skip_variogram: Whether to skip variogram score.

    Returns:
        Dict of scores per leadtime.
    """
    crps_fn = EnsembleCRPS().cuda()
    es_fn = EnergyScore(clamp=False).cuda()
    vs_fn = VariogramScore(p=0.5, chunk_size=256).cuda()

    n_times = ecc_preds.shape[0]
    crps_list, es_pv_list, es_full_list = [], [], []
    vs_pv_list, vs_full_list = [], []

    for i in trange(n_times, desc="Computing ECC scores"):
        pred_i = ecc_preds[i : i + 1].cuda()  # [1, M, n_vars, h, w]
        y_i = y_rescaled[i : i + 1].cuda()  # [1, n_vars, h, w]

        with torch.no_grad():
            crps_list.append(crps_fn(pred_i, y_i).cpu())
            es_pv_list.append(es_fn(pred_i, y_i, mode="per_var").cpu())
            es_full_list.append(es_fn(pred_i, y_i, mode="complete").cpu())
            if not skip_variogram:
                vs_pv_list.append(vs_fn(pred_i, y_i, mode="per_var").cpu())
                vs_full_list.append(vs_fn(pred_i, y_i, mode="complete").cpu())

        # Free GPU memory periodically
        del pred_i, y_i
        if i % 100 == 0:
            torch.cuda.empty_cache()

    crps_per_margin = torch.cat(crps_list, dim=0)
    energy_score_per_var_u = torch.cat(es_pv_list, dim=0)
    energy_score_full_u = torch.cat(es_full_list, dim=0)
    variogram_pv_u = torch.cat(vs_pv_list, dim=0) if not skip_variogram else None
    variogram_full_u = torch.cat(vs_full_list, dim=0) if not skip_variogram else None

    # Reduce scores
    method_label = f"{model_class}+ECC"
    crps_per_var = reduce(crps_per_margin, "t d x y -> d", "mean")
    crps_full = reduce(crps_per_margin, "t d x y -> 1", "mean")
    es_per_var = reduce(energy_score_per_var_u, "t d -> d", "mean")
    es_full = reduce(energy_score_full_u, "t -> 1", "mean")

    print(f"[{method_label}] Mean CRPS per var: {crps_per_var}")
    print(f"[{method_label}] Mean CRPS combined: {crps_full}")
    print(f"[{method_label}] Mean ES per var: {es_per_var}")
    print(f"[{method_label}] Mean ES combined: {es_full}")

    # Log scores
    log_scores(
        file=score_file,
        model=method_label,
        metric="CRPS",
        variables=y_select_variables,
        scores=crps_per_var,
    )
    log_scores(
        file=score_file, model=method_label, metric="CRPS", variables=["combined"], scores=crps_full
    )
    log_scores(
        file=score_file,
        model=method_label,
        metric="EnergyScore",
        variables=y_select_variables,
        scores=es_per_var,
    )
    log_scores(
        file=score_file,
        model=method_label,
        metric="EnergyScore",
        variables=["combined"],
        scores=es_full,
    )

    if not skip_variogram:
        vs_per_var = reduce(variogram_pv_u, "t d -> d", "mean")  # type: ignore
        vs_full = reduce(variogram_full_u, "t -> 1", "mean")  # type: ignore
        print(f"[{method_label}] Mean VS per var: {vs_per_var}")
        print(f"[{method_label}] Mean VS combined: {vs_full}")
        log_scores(
            file=score_file,
            model=method_label,
            metric="VariogramScore",
            variables=y_select_variables,
            scores=vs_per_var,  # type: ignore
        )
        log_scores(
            file=score_file,
            model=method_label,
            metric="VariogramScore",
            variables=["combined"],
            scores=vs_full,  # type: ignore
        )

    # Per-leadtime scores
    scores = compute_icon_scores_per_leadtime(
        timedeltas,
        crps_per_margin,
        energy_score_per_var_u,
        energy_score_full_u,
        y_select_variables=y_select_variables,
        vss_per_var=variogram_pv_u,
        vss_complete=variogram_full_u,
        method="ECC",
    )

    return scores


# ---------------------------------------------------------------------------
# Split evaluation
# ---------------------------------------------------------------------------


def evaluate_split(
    split: str,
    *,
    model,
    trainer: L.Trainer,
    datamodule,
    cfg: DictConfig,
    score_file: Path,
    model_dir: Path,
    ens_dir: Path,
    n_quantile_samples: int,
    skip_variogram: bool,
    save_predictions: bool,
    force_repredict: bool,
    verbose: bool,
) -> dict:
    """Run ECC evaluation for a single data split.

    Steps:
        1. Run model to get distribution parameters (mu, sigma per variable)
        2. Generate quantile samples from the predicted distributions
        3. Apply ECC using raw ensemble forecasts
        4. Rescale predictions and ground truth to original space
        5. Compute evaluation scores

    Returns:
        Dict of scores per leadtime.
    """
    split_config = get_split_config(split)

    # Get dataloader and dataset
    dataloader_method = getattr(datamodule, split_config["dataloader_method"])
    dataloader = dataloader_method()
    dataset = getattr(datamodule, split_config["dataset_attr"])
    y_select_variables = list(cfg.data.y_select_variables)

    # Check for cached predictions (.nc format)
    predictions_path = model_dir / f"{split}_predictions_ecc.nc"
    use_cached = predictions_path.exists() and not force_repredict

    if use_cached:
        log_msg(f"Loading cached ECC predictions from {predictions_path}...", verbose)
        ds = xr.open_dataset(predictions_path)
        ecc_preds_rescaled = torch.from_numpy(ds["prediction"].values)
        ds.close()
    else:
        # Step 1: Run model forward pass
        log_msg(f"Running predictions on {split} split...", verbose)
        raw_predictions = trainer.predict(model, dataloader, return_predictions=True)
        stacked_preds = stack_predictions(raw_predictions)  # type: ignore
        del raw_predictions

        # Step 2: Generate quantile samples (chunked to avoid OOM)
        log_msg("Generating quantile samples from predicted distributions...", verbose)
        quantile_preds = quantile_samples_icon(
            stacked_preds, y_select_variables, M=n_quantile_samples
        )
        del stacked_preds
        log_msg(f"Quantile samples shape: {quantile_preds.shape}", verbose)

        # Step 3: Apply ECC
        log_msg("Applying ECC postprocessing...", verbose)
        ecc_preds = do_ecc_icon(quantile_preds, dataset.samples, ens_dir, verbose=verbose)
        del quantile_preds

        # Step 4: Rescale predictions to original space
        log_msg("Rescaling ECC predictions to original space...", verbose)
        reverse_modules = datamodule.y_reverseModules
        ecc_preds_rescaled = _rescale_y(ecc_preds, reverse_modules)
        del ecc_preds

        # Save predictions as NetCDF if requested
        if save_predictions:
            log_msg(f"Saving ECC predictions to {predictions_path}...", verbose)
            ds = _predictions_to_xarray(ecc_preds_rescaled, dataset.samples, y_select_variables)
            ds.to_netcdf(predictions_path)
            log_msg(f"ECC predictions saved to {predictions_path}", verbose)

    # Load ground truth directly from rea files in dataset.samples
    # (already in original space, no rescaling needed)
    log_msg("Loading raw ground truth from dataset samples...", verbose)
    y_rescaled = _load_ground_truth_from_samples(dataset.samples, verbose=verbose)

    # Extract leadtimes from dataset samples
    timedeltas = np.array([sample[3] for sample in dataset.samples])

    # Step 5: Compute scores
    log_msg("Computing evaluation scores...", verbose)
    model_class = cfg.model._target_.split(".")[-1]
    scores = compute_ecc_scores(
        ecc_preds_rescaled,
        y_rescaled,
        timedeltas,
        y_select_variables,
        score_file=score_file,
        model_class=model_class,
        skip_variogram=skip_variogram,
    )

    return scores


# ---------------------------------------------------------------------------
# Run processing
# ---------------------------------------------------------------------------


def process_run(run_path: str, args: argparse.Namespace) -> None:
    """Process a single WandB run: load EMOS/DRN model, apply ECC, evaluate."""
    log_msg(f"\n{'#' * 60}\nProcessing run: {run_path}\n{'#' * 60}", args.verbose)

    # Parse run path
    model_id = run_path.split("/")[-1]
    output_dir = BASE_DIR.parent.parent / "outputs"

    log_msg(f"Looking for model with ID: {model_id}", args.verbose)

    # Find model directory
    model_dirs = list(output_dir.rglob(f"*{model_id}*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directory found for run ID '{model_id}' in {output_dir}")
    # Navigate up from matched path to the run root directory:
    # outputs/<MODEL>/<DATE>/logs/genpp/<MODEL_ID> -> outputs/<MODEL>/<DATE>
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
    datamodule.setup(stage="fit")

    # Filter samples by leadtime if requested
    if args.leadtimes is not None:
        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            ds = getattr(datamodule, attr, None)
            if ds is not None:
                orig_len = len(ds.samples)
                ds.samples = filter_samples_by_leadtimes(ds.samples, args.leadtimes)
                log_msg(
                    f"Filtered {attr}: {orig_len} → {len(ds.samples)} samples "
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
    sig = inspect.signature(ModelClass.__init__)
    model_kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
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

    try:
        model = ModelClass.load_from_checkpoint(model_checkpoint, strict=False, **model_kwargs)  # type: ignore
    except (pickle.UnpicklingError, TypeError):
        model = ModelClass.load_from_checkpoint(
            model_checkpoint,
            weights_only=False,
            strict=False,
            **model_kwargs,  # type: ignore
        )

    # Create trainer
    trainer = L.Trainer(logger=False, accelerator="gpu", devices="auto", enable_progress_bar=True)

    # Determine ensemble directory.
    # DATA_DIR may be overridden via GENPP_DATA_DIR env var to a temp directory
    # that only contains tensors, not the raw ensemble NC files. Fall back to
    # the source location if the ens/ subdirectory is not found.
    ens_dir = DATA_DIR / "ens"
    if not ens_dir.exists():
        ens_dir_fallback = BASE_DIR / "data" / "icon" / "data" / "ens"
        log_msg(
            f"Ensemble directory not found at {ens_dir}, falling back to {ens_dir_fallback}",
            args.verbose,
        )
        ens_dir = ens_dir_fallback
    log_msg(f"Ensemble directory: {ens_dir}", args.verbose)

    # Evaluate each requested split
    full_scores: dict = {}
    for split in args.split:
        log_msg(f"\n{'=' * 60}\nEvaluating split: {split}\n{'=' * 60}", args.verbose)
        scores = evaluate_split(
            split,
            model=model,
            trainer=trainer,
            datamodule=datamodule,
            cfg=cfg,
            score_file=score_file,
            model_dir=model_dir,
            ens_dir=ens_dir,
            n_quantile_samples=args.n_quantile_samples,
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
    model_class = cfg.model._target_.split(".")[-1]
    records = []
    for dataset, metrics in full_scores.items():
        for method_key, method_scores in metrics.items():
            for metric_name, horizons in method_scores.items():
                for horizon, value in horizons.items():
                    records.append(
                        (f"{model_class}+{method_key}", dataset, metric_name, horizon, value)
                    )
    df = pd.DataFrame(records, columns=["method", "dataset", "metric", "horizon", "value"])
    save_scores_df(df=df, run_path=run_path)

    log_msg(f"Done with run: {run_path}", args.verbose)


def main() -> None:
    """Main entry point for ICON ECC evaluation."""
    args = parse_args()

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
