#!/usr/bin/env python
"""
Predict and evaluate with Flow Matching, CGM, or Engression models.

This script loads a trained model from a WandB run, runs predictions on the
specified data split, computes evaluation metrics (CRPS, Energy Score,
Variogram Score), and logs results to WandB and local files.

Usage:
    python cgm_predict_eval.py --run-path feik/genpp/abc123 --split val
    python cgm_predict_eval.py --run-path feik/genpp/abc123 feik/genpp/def456 --split val test
    python cgm_predict_eval.py --run-path feik/genpp/abc123 --split test --skip-variogram
    python cgm_predict_eval.py --run-path feik/genpp/abc123 --device 0,1 --batch-size 32 -v
    python cgm_predict_eval.py --run-path feik/genpp/hbuy7eio --split val --device 0,1 --batch-size 32 --skip-variogram --save-predictions -v
"""

import argparse
import importlib
import inspect
import json
import pickle
import shlex

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange, reduce
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb
from genpp import BASE_DIR
from genpp.configs import add_y_kwargs, del_key, register_resolvers
from genpp.data.weatherbench2 import OBSERVATIONS_FLAT_PATH
from genpp.eval.utils import (
    compute_scores_per_leadtime,
    load_predictions_dataarray,
    log_scores,
    save_predictions_dataarray,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import EnergyScore, EnsembleCRPS, VariogramScore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict and evaluate with Flow Matching, CGM, or Engression models.",
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
        "--device",
        type=str,
        default="0",
        help="GPU device index or comma-separated list (e.g., '0' or '0,1,2')",
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
        "--old-config",
        action="store_true",
        help="Apply legacy config fixes for older models",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to Zarr file",
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


def parse_device(device_str: str) -> int | list[int]:
    """Parse device string to int or list of ints.

    Args:
        device_str: Device specification, e.g., "0" or "0,1,2"

    Returns:
        Single int or list of ints for device indices
    """
    if "," in device_str:
        return [int(d.strip()) for d in device_str.split(",")]
    return [int(device_str)]


def log_msg(msg: str, verbose: bool) -> None:
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)


def get_split_config(split: str) -> dict:
    """Get configuration for the specified split.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        Dict with setup_stage, dataloader_method, and metadata_key
    """
    config = {
        "train": {
            "setup_stage": "fit",
            "dataloader_method": "train_dataloader",
            "metadata_key": "train",
        },
        "val": {
            "setup_stage": "validate",
            "dataloader_method": "val_dataloader",
            "metadata_key": "val",
        },
        "test": {
            "setup_stage": "test",
            "dataloader_method": "test_dataloader",
            "metadata_key": "test",
        },
    }
    return config[split]


def get_original_command(run_path: str) -> str | None:
    """Fetch the original command from W&B metadata.

    Reads the run's wandb-metadata.json and reconstructs the command
    from the captured program and args.
    """
    try:
        api = wandb.Api()
        run = api.run(run_path)
        meta_file = run.file("wandb-metadata.json").download(replace=True)
        with open(meta_file.name) as f:
            meta = json.load(f)

        program = meta.get("program")
        args = meta.get("args") or []

        parts = ([program] if program else []) + [str(a) for a in args]
        cmd = shlex.join(parts)
        return cmd or None
    except Exception:
        return None


def store_original_command_config(run_path: str, cmd: str, verbose: bool = False) -> None:
    """Store the preserved original command in the run config.

    Only writes to config (not summary or notes). Safe no-op if unchanged.
    """
    try:
        api = wandb.Api()
        run = api.run(run_path)
        existing = run.config.get("original_command")
        if existing != cmd:
            run.config["original_command"] = cmd
            run.update()
            log_msg("Preserved original W&B command in config", verbose)
    except Exception as e:
        log_msg(f"Failed to store original command: {e}", verbose)


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
    """Run prediction and evaluation for a single data split.

    Returns:
        Dict of scores per leadtime for this split.
    """
    split_config = get_split_config(split)

    # Check for cached predictions
    predictions_path = model_dir / f"{split}_predictions.zarr"
    use_cached = predictions_path.exists() and not force_repredict

    if use_cached:
        log_msg(f"Loading cached predictions from {predictions_path}...", verbose)
        cached_da = load_predictions_dataarray(predictions_path)
        predictions_rescaled = torch.from_numpy(cached_da.values).cuda()
    else:
        # Get dataloader for the specified split
        dataloader_method = getattr(datamodule, split_config["dataloader_method"])
        dataloader = dataloader_method()

        # Run predictions
        log_msg(f"Running predictions on {split} split...", verbose)
        pred_list = trainer.predict(model, dataloader, return_predictions=True)
        predictions = torch.cat(pred_list, dim=0)  # type: ignore

        # Rescale predictions
        log_msg("Rescaling predictions...", verbose)
        reverse_transform = datamodule.y_reverseModules[0]
        mean = rearrange(reverse_transform.mean, "f -> 1 1 f 1 1")
        scale = rearrange(reverse_transform.scale, "f -> 1 1 f 1 1")
        predictions_rescaled = (predictions * scale + mean).cuda()

    # Load ground truth observations
    log_msg("Loading ground truth observations...", verbose)
    metadata_key = split_config["metadata_key"]
    init_times = datamodule.cache_metadata["feature_metadata"]["time"][metadata_key]
    timedeltas = datamodule.cache_metadata["feature_metadata"]["prediction_timedelta"][metadata_key]
    target_times = init_times + timedeltas
    prediction_index = pd.MultiIndex.from_arrays(
        [init_times, timedeltas], names=["time", "prediction_timedelta"]
    )

    y_obs = (
        xr.open_dataset(OBSERVATIONS_FLAT_PATH)
        .sel(time=target_times)
        .to_dataarray("feature")
        .transpose("time", "feature", "longitude", "latitude")
        .rename({"time": "prediction_time"})
        .assign_coords(prediction=("prediction_time", prediction_index))
        .swap_dims({"prediction_time": "prediction"})
    )
    feature_order = list(cfg.data.y_select_variables)
    y_obs = y_obs.sel(feature=feature_order)
    y_t = torch.from_numpy(y_obs.values).to(predictions_rescaled)

    # Compute scores (loop over time steps to avoid OOM from pairwise expansions)
    log_msg("Computing evaluation scores...", verbose)
    crps_ens = EnsembleCRPS().cuda()
    es = EnergyScore(clamp=False).cuda()
    vs = VariogramScore(p=0.5).cuda()

    n_times = predictions_rescaled.shape[0]
    crps_list, es_pv_list, es_full_list = [], [], []
    vs_pv_list, vs_full_list = [], []

    for i in range(n_times):
        pred_i = predictions_rescaled[i : i + 1]
        y_i = y_t[i : i + 1]
        with torch.no_grad():
            crps_list.append(crps_ens(pred_i, y_i).cpu())
            es_pv_list.append(es(pred_i, y_i, mode="per_var").cpu())
            es_full_list.append(es(pred_i, y_i, mode="complete").cpu())
            if not skip_variogram:
                vs_pv_list.append(vs(pred_i, y_i, mode="per_var").cpu())
                vs_full_list.append(vs(pred_i, y_i, mode="complete").cpu())

    crps_per_margin = torch.cat(crps_list, dim=0)
    energy_score_per_var_u = torch.cat(es_pv_list, dim=0)
    energy_score_full_u = torch.cat(es_full_list, dim=0)
    variogram_score_per_var_u = torch.cat(vs_pv_list, dim=0) if not skip_variogram else None
    variogram_score_full_u = torch.cat(vs_full_list, dim=0) if not skip_variogram else None

    # Reduce scores
    log_msg("Reducing scores...", verbose)
    crps_per_var = reduce(crps_per_margin, "t d h w -> d", reduction="mean")
    crps_full = reduce(crps_per_margin, "t d h w -> 1", "mean")
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
    scores = compute_scores_per_leadtime(
        timedeltas,
        crps_per_margin,
        energy_score_per_var_u,
        energy_score_full_u,
        variogram_score_per_var_u if not skip_variogram else None,
        variogram_score_full_u if not skip_variogram else None,
        method=None,
    )

    # Save predictions if requested (and they were freshly computed)
    if save_predictions and not use_cached:
        log_msg("Saving predictions...", verbose)

        n_samples = predictions_rescaled.shape[1]
        N = np.arange(n_samples)
        res = xr.DataArray(
            predictions_rescaled.cpu().numpy(),
            coords={
                "prediction": y_obs.prediction,
                "sample": N,
                "feature": y_obs.feature,
                "longitude": y_obs.longitude,
                "latitude": y_obs.latitude,
            },
            dims=("prediction", "sample", "feature", "longitude", "latitude"),
        )

        save_predictions_dataarray(predictions=res, save_path=predictions_path, overwrite=True)
        log_msg(f"Predictions saved to {predictions_path}", verbose)

    return scores


def process_run(run_path: str, args: argparse.Namespace) -> None:
    """Process a single WandB run: load model, predict, evaluate, and log results."""
    log_msg(f"\n{'#' * 60}\nProcessing run: {run_path}\n{'#' * 60}", args.verbose)

    # Capture the original W&B command before any updates
    old_cmd = get_original_command(run_path)

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

    # Configure dataloaders - disable shuffle, set batch size
    cfg.data.module.dataloader_config.train.shuffle = False
    cfg.data.module.dataloader_config.val.shuffle = False
    cfg.data.module.dataloader_config.val.batch_size = args.batch_size
    cfg.data.module.dataloader_config.test.shuffle = False

    # Apply old config fixes if needed
    if args.old_config:
        log_msg("Applying legacy config fixes...", args.verbose)
        add_y_kwargs(
            cfg,
            y_kwargs={
                "batch_dims": {},
                "input_dims": {"feature": 2, "longitude": 37, "latitude": 31},
            },
        )
        del_key(cfg.data.module.dataset_config.train.x_kwargs.batch_dims, "time")
        del_key(
            cfg.data.module.dataset_config.train.x_kwargs.batch_dims,
            "prediction_timedelta",
        )

    # Setup datamodule
    log_msg("Setting up data module...", args.verbose)
    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()

    # Setup the appropriate stage(s) based on requested splits
    splits = args.split
    needs_fit = "train" in splits
    needs_val_test = any(s in splits for s in ("val", "test"))

    if needs_fit:
        datamodule.setup(stage="fit")
    if needs_val_test:
        datamodule.setup(stage="validate")
        datamodule.setup(stage="test")

    # Load model
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    class_path = cfg.model._target_
    module_path, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    # Build model_kwargs dynamically from the model's __init__ signature and config.
    # This is needed for old checkpoints that don't have hyperparameters saved.
    sig = inspect.signature(ModelClass.__init__)
    model_kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param_name not in cfg.model or OmegaConf.is_missing(cfg.model, param_name):
            # For required params not in config, default to None (e.g. rescaler
            # comes from the datamodule, not the model config).
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
    devices = parse_device(args.device)
    trainer = L.Trainer(logger=False, accelerator="gpu", devices=devices)

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

    # Restore the original command into run config (only config)
    if old_cmd:
        store_original_command_config(run_path, old_cmd, args.verbose)

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
    """Main entry point for the prediction script."""
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
