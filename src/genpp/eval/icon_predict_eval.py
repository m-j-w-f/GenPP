#!/usr/bin/env python
"""
Predict and evaluate with Flow Matching, CGM, or Engression models on ICON data.

This script loads a trained model from a WandB run, runs predictions on the
specified data split, computes evaluation metrics (CRPS, Energy Score,
Variogram Score), and logs results to WandB and local files.

Unlike cgm_predict_eval.py (which uses WeatherBench2 data), this script handles
the ICON dataset structure where:
  - Ground truth is loaded from per-date .pt tensor files via the dataloader
  - Both predictions and ground truth are rescaled to the original space before scoring
  - Leadtimes are extracted from the dataset sample tuples

Usage:
    python icon_predict_eval.py --run-path feik/genpp/abc123 --split val
    python icon_predict_eval.py --run-path feik/genpp/abc123 feik/genpp/def456 --split val test
    python icon_predict_eval.py --run-path feik/genpp/abc123 --split test --skip-variogram
    python icon_predict_eval.py --run-path feik/genpp/abc123 --batch-size 32 -v
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pickle

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from einops import reduce
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm, trange

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.eval.utils import (
    log_scores,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import EnergyScore, EnsembleCRPS, VariogramScore


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
        default=16,
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


def compute_icon_scores_per_leadtime(
    prediction_timedeltas: np.ndarray,
    crpss: torch.Tensor,
    ess_per_var: torch.Tensor,
    ess_complete: torch.Tensor,
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
    scores_delta: dict = {method: {"CRPS_combined": {}, "EnergyScore_combined": {}}}
    for var_name in y_select_variables:
        scores_delta[method][f"CRPS_{var_name}"] = {}
        scores_delta[method][f"EnergyScore_{var_name}"] = {}

    if vss_per_var is not None and vss_complete is not None:
        scores_delta[method]["VariogramScore_combined"] = {}
        for var_name in y_select_variables:
            scores_delta[method][f"VariogramScore_{var_name}"] = {}

    for delta, delta_str in tqdm(zip(td, td_str), total=len(td), desc="Processing leadtimes"):
        mask = prediction_timedeltas == delta
        crpss_delta = crpss[mask]
        ess_per_var_delta = ess_per_var[mask]
        ess_complete_delta = ess_complete[mask]

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

    Ground truth (y) is loaded from the dataloader in scaled form, then both
    predictions and ground truth are rescaled back to the original space before
    computing evaluation scores.

    Returns:
        Dict of scores per leadtime for this split.
    """
    split_config = get_split_config(split)

    # Check for cached predictions
    predictions_path = model_dir / f"{split}_predictions.pt"
    gt_path = model_dir / f"{split}_ground_truth.pt"
    use_cached = predictions_path.exists() and not force_repredict

    if use_cached:
        log_msg(f"Loading cached predictions from {predictions_path}...", verbose)
        predictions_rescaled = torch.load(predictions_path, weights_only=True).cuda()

        # Load ground truth if it exists, otherwise collect it
        if gt_path.exists():
            log_msg(f"Loading cached ground truth from {gt_path}...", verbose)
            y_rescaled = torch.load(gt_path, weights_only=True).cuda()
        else:
            # Collect ground truth from the dataloader
            log_msg("Collecting ground truth from dataloader...", verbose)
            dataloader_method = getattr(datamodule, split_config["dataloader_method"])
            dataloader = dataloader_method()
            y_list = []
            for batch in tqdm(dataloader, desc="Collecting ground truth"):
                y_list.append(batch["y"])  # shape per batch: [B, c, x, y]
            y_scaled = torch.cat(y_list, dim=0)  # shape: [N, c, x, y]

            # Rescale ground truth to original space
            reverse_modules = datamodule.y_reverseModules
            y_rescaled = _rescale_y(y_scaled, reverse_modules).cuda()
            del y_scaled
            torch.cuda.empty_cache()

            # Save ground truth (only once, not per model)
            log_msg(f"Saving ground truth to {gt_path}...", verbose)
            torch.save(y_rescaled.cpu(), gt_path)
            y_rescaled = y_rescaled.cuda()
            log_msg(f"Ground truth saved to {gt_path}", verbose)
    else:
        # Get dataloader for the specified split
        dataloader_method = getattr(datamodule, split_config["dataloader_method"])
        dataloader = dataloader_method()

        # Run predictions
        log_msg(f"Running predictions on {split} split...", verbose)
        pred_list = trainer.predict(model, dataloader, return_predictions=True)
        predictions = torch.cat(pred_list, dim=0)  # shape: [N, n_samples, c, x, y] # type: ignore

        # Collect ground truth from the dataloader
        log_msg("Collecting ground truth from dataloader...", verbose)
        y_list = []
        for batch in tqdm(dataloader, desc="Collecting ground truth"):
            y_list.append(batch["y"])  # shape per batch: [B, c, x, y]
        y_scaled = torch.cat(y_list, dim=0)  # shape: [N, c, x, y]

        # Rescale both predictions and ground truth to original space
        log_msg("Rescaling predictions and ground truth...", verbose)
        reverse_modules = datamodule.y_reverseModules

        predictions_rescaled = _rescale_y(predictions, reverse_modules).cuda()
        y_rescaled = _rescale_y(y_scaled, reverse_modules).cuda()

        # Free memory from the original scaled tensors
        del predictions, y_scaled
        torch.cuda.empty_cache()

    # Extract leadtimes from dataset samples
    dataset = getattr(datamodule, split_config["dataset_attr"])
    timedeltas = np.array([sample[3] for sample in dataset.samples])

    # Compute scores (loop over time steps to avoid OOM from pairwise expansions)
    log_msg("Computing evaluation scores...", verbose)
    crps_ens = EnsembleCRPS().cuda()
    es = EnergyScore(clamp=False).cuda()
    vs = VariogramScore(p=0.5).cuda()

    n_times = predictions_rescaled.shape[0]
    crps_list, es_pv_list, es_full_list = [], [], []
    vs_pv_list, vs_full_list = [], []

    for i in trange(n_times, desc="Computing scores"):
        pred_i = predictions_rescaled[i : i + 1]  # [1, n_samples, c, x, y]
        y_i = y_rescaled[i : i + 1]  # [1, c, x, y]
        with torch.no_grad():
            crps_list.append(crps_ens(pred_i, y_i).cpu())
            es_pv_list.append(es(pred_i, y_i, mode="per_var").cpu())
            es_full_list.append(es(pred_i, y_i, mode="complete").cpu())
            if not skip_variogram:
                vs_pv_list.append(vs(pred_i, y_i, mode="per_var").cpu())
                vs_full_list.append(vs(pred_i, y_i, mode="complete").cpu())

        # Clear GPU cache periodically to avoid OOM
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    crps_per_margin = torch.cat(crps_list, dim=0)
    energy_score_per_var_u = torch.cat(es_pv_list, dim=0)
    energy_score_full_u = torch.cat(es_full_list, dim=0)
    variogram_score_per_var_u = torch.cat(vs_pv_list, dim=0) if not skip_variogram else None
    variogram_score_full_u = torch.cat(vs_full_list, dim=0) if not skip_variogram else None

    # Reduce scores
    log_msg("Reducing scores...", verbose)
    crps_per_var = reduce(crps_per_margin, "t d x y -> d", reduction="mean")
    crps_full = reduce(crps_per_margin, "t d x y -> 1", "mean")
    energy_score_per_var = reduce(energy_score_per_var_u, "t d -> d", "mean")
    energy_score_full = reduce(energy_score_full_u, "t -> 1", "mean")

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
        y_select_variables=list(cfg.data.y_select_variables),
        vss_per_var=variogram_score_per_var_u if not skip_variogram else None,
        vss_complete=variogram_score_full_u if not skip_variogram else None,
        method=None,
    )

    # Save predictions if requested (and they were freshly computed)
    if save_predictions and not use_cached:
        log_msg("Saving predictions...", verbose)
        torch.save(predictions_rescaled.cpu(), predictions_path)
        log_msg(f"Predictions saved to {predictions_path}", verbose)

    # Save ground truth only if it doesn't exist yet (shared across all models)
    if not use_cached and not gt_path.exists():
        log_msg("Saving ground truth...", verbose)
        torch.save(y_rescaled.cpu(), gt_path)
        log_msg(f"Ground truth saved to {gt_path}", verbose)

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

    model_kwargs["n_samples"] = 50

    try:
        model = ModelClass.load_from_checkpoint(model_checkpoint, **model_kwargs)  # type: ignore
    except (pickle.UnpicklingError, TypeError):
        model = ModelClass.load_from_checkpoint(
            model_checkpoint,
            weights_only=False,
            **model_kwargs,  # type: ignore
        )

    # Fix internal_td_scaling if needed
    if hasattr(model, "internal_td_scaling"):
        td_scaling = model.internal_td_scaling
        if not getattr(td_scaling, "is_fitted", False):
            log_msg("Setting internal_td_scaling.is_fitted = True", args.verbose)
            td_scaling.is_fitted = True  # type: ignore
        if not getattr(td_scaling, "n_vars", False):
            log_msg("Setting internal_td_scaling.n_vars = 2", args.verbose)
            td_scaling.n_vars = 2  # type: ignore

    # Create trainer
    trainer = L.Trainer(
        logger=False, accelerator="gpu", devices="auto", enable_progress_bar=True
    )

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
