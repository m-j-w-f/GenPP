#!/usr/bin/env python
"""
Predict and evaluate with Flow Matching, CGM, or Engression models.

This script loads a trained model from a WandB run, runs predictions on the
specified data split, computes evaluation metrics (CRPS, Energy Score,
Variogram Score), and logs results to WandB and local files.

Usage:
    python fm_predict.py --run-path feik/genpp/abc123 --split val
    python fm_predict.py --run-path feik/genpp/abc123 --split test --skip-variogram
    python fm_predict.py --run-path feik/genpp/abc123 --device 0,1 --batch-size 32 -v
    python fm_predict.py --run-path feik/genpp/hbuy7eio --split val --device 0,1 --batch-size 32 --skip-variogram --save-predictions -v
"""

from __future__ import annotations

import argparse
import importlib
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
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from genpp import BASE_DIR
from genpp.configs import add_y_kwargs, del_key, register_resolvers
from genpp.data import OBSERVATIONS_FLAT_PATH
from genpp.eval.utils import (
    compute_scores_per_leadtime,
    log_scores,
    save_predictions_dataarray,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.cgm.chen import CNNChenModel
from genpp.models.loss import EnergyScore, EnsembleCRPS, VariogramScore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict and evaluate with Flow Matching, CGM, or Engression models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="WandB run path (e.g., 'feik/genpp/abc123')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
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


def main() -> None:
    """Main entry point for the prediction script."""
    args = parse_args()

    # Capture the original W&B command before any updates
    old_cmd = get_original_command(args.run_path)

    # Register Hydra resolvers
    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    # Parse run path
    model_id = args.run_path.split("/")[-1]
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

    # Get split configuration
    split_config = get_split_config(args.split)

    # Setup the appropriate stage(s)
    if args.split == "train":
        datamodule.setup(stage="fit")
    else:
        datamodule.setup(stage="validate")
        datamodule.setup(stage="test")

    # Load model
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    class_path = cfg.model._target_
    module_path, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    if ModelClass is CNNChenModel:
        try:
            model = ModelClass.load_from_checkpoint(
                model_checkpoint,
                final_activation=hydra.utils.instantiate(cfg.model.final_activation),
                loss_fn=hydra.utils.instantiate(cfg.model.loss_fn),
                n_samples=50,
            )
        except pickle.UnpicklingError:
            model = ModelClass.load_from_checkpoint(
                model_checkpoint,
                final_activation=hydra.utils.instantiate(cfg.model.final_activation),
                loss_fn=hydra.utils.instantiate(cfg.model.loss_fn),
                n_samples=50,
                weights_only=False,
            )
    else:
        try:
            model = ModelClass.load_from_checkpoint(model_checkpoint)
        except pickle.UnpicklingError:
            model = ModelClass.load_from_checkpoint(model_checkpoint, weights_only=False)

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

    # Get dataloader for the specified split
    dataloader_method = getattr(datamodule, split_config["dataloader_method"])
    dataloader = dataloader_method()

    # Run predictions
    log_msg(f"Running predictions on {args.split} split...", args.verbose)
    pred_list = trainer.predict(model, dataloader, return_predictions=True)
    predictions = torch.cat(pred_list, dim=0)  # type: ignore

    # Rescale predictions
    log_msg("Rescaling predictions...", args.verbose)
    reverse_transform = datamodule.y_reverseModules[0]
    mean = rearrange(reverse_transform.mean, "f -> 1 1 f 1 1")
    scale = rearrange(reverse_transform.scale, "f -> 1 1 f 1 1")
    predictions_rescaled = predictions * scale + mean

    # Load ground truth observations
    log_msg("Loading ground truth observations...", args.verbose)
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

    # Compute scores
    log_msg("Computing evaluation scores...", args.verbose)
    crps_ens = EnsembleCRPS()
    es = EnergyScore(clamp=False)
    vs = VariogramScore(p=0.5)

    crps_per_margin = crps_ens(predictions_rescaled, y_t)

    # Per-variable scores
    x_spatial = rearrange(predictions_rescaled, "t n d lat lon -> t d n (lat lon)")
    y_spatial = rearrange(y_t, "t d lat lon -> t d (lat lon)")
    energy_score_per_var_u = es(x_spatial, y_spatial)

    variogram_score_per_var_u = None
    if not args.skip_variogram:
        log_msg("Computing per-variable variogram scores...", args.verbose)
        vss = []
        for x_i, y_i in tqdm(
            zip(x_spatial, y_spatial),
            total=predictions_rescaled.shape[0],
            desc="Variogram (per-var)",
        ):
            vss.append(vs(x_i, y_i))
        variogram_score_per_var_u = torch.stack(vss)

    # Full (combined) scores
    x_full = rearrange(predictions_rescaled, "t n d lat lon -> t n (d lat lon)")
    y_full = rearrange(y_t, "t d lat lon -> t (d lat lon)")
    energy_score_full_u = es(x_full, y_full)

    variogram_score_full_u = None
    if not args.skip_variogram:
        log_msg("Computing full variogram scores...", args.verbose)
        vss = []
        for x_i, y_i in tqdm(
            zip(x_full, y_full),
            total=predictions_rescaled.shape[0],
            desc="Variogram (full)",
        ):
            vss.append(vs(x_i, y_i))
        variogram_score_full_u = torch.stack(vss)

    # Reduce scores
    log_msg("Reducing scores...", args.verbose)
    crps_per_var = reduce(crps_per_margin, "t d h w -> d", reduction="mean")
    crps_full = reduce(crps_per_margin, "t d h w -> 1", "mean")
    energy_score_per_var = reduce(energy_score_per_var_u, "t d -> d", "mean")
    energy_score_full = reduce(energy_score_full_u, "t -> 1", "mean")

    if not args.skip_variogram:
        variogram_score_per_var = reduce(variogram_score_per_var_u, "t d -> d", "mean")
        variogram_score_full = reduce(variogram_score_full_u, "t -> 1", "mean")

    # Log scores to file
    log_msg(f"Logging scores to {score_file}...", args.verbose)
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

    if not args.skip_variogram:
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
    log_msg("Computing scores per leadtime...", args.verbose)
    scores = compute_scores_per_leadtime(
        timedeltas,
        crps_per_margin,
        energy_score_per_var_u,
        energy_score_full_u,
        variogram_score_per_var_u if not args.skip_variogram else None,
        variogram_score_full_u if not args.skip_variogram else None,
        method=None,
    )

    # Update WandB run
    log_msg("Updating WandB run...", args.verbose)
    full_scores = {args.split: scores}
    update_wandb_run(args.run_path, full_scores)

    # Restore the original command into run config (only config)
    if old_cmd:
        store_original_command_config(args.run_path, old_cmd, args.verbose)

    # Save scores DataFrame
    records = []
    for dataset, metrics in full_scores.items():
        for metric_name, horizons in metrics.items():
            for horizon, value in horizons.items():
                records.append(
                    (f"{model.__class__.__name__}", dataset, metric_name, horizon, value)
                )
    df = pd.DataFrame(records, columns=["method", "dataset", "metric", "horizon", "value"])
    save_scores_df(df=df, run_path=args.run_path)

    # Save predictions if requested
    if args.save_predictions:
        log_msg("Saving predictions...", args.verbose)

        if hasattr(model, "n_samples_train"):
            n_samples = model.n_samples_train
        elif hasattr(model, "n_samples"):
            n_samples = model.n_samples
        else:
            raise ValueError(
                "Model has no attribute 'n_samples' or 'n_samples_train'. "
                "Cannot determine number of samples for saving predictions."
            )

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

        predictions_path = model_dir / f"{args.split}_predictions.zarr"
        save_predictions_dataarray(predictions=res, save_path=predictions_path, overwrite=True)
        log_msg(f"Predictions saved to {predictions_path}", args.verbose)

    log_msg("Done!", args.verbose)


if __name__ == "__main__":
    main()
