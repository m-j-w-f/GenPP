#!/usr/bin/env python
"""
GPU step: Run EMOS/DRN model predictions and save distribution parameters.

Loads the model, runs forward pass on the specified split(s), and saves
the resulting distribution parameters (mu, sigma per variable) along with
ground truth and metadata to disk. These files are then consumed by
icon_copulas_score.py on a CPU node.

Saved files (in model_dir):
    {split}_dist_params.pt  — dict with keys:
        'params': list of per-variable dicts {'mu': [N, h, w], 'sigma': [N, h, w]}
        'y_select_variables': list of target variable names
        'samples': list of (fc_path, rea_path, init_date, leadtime) tuples
        'reverse_modules': list of dicts {'mean': float, 'scale': float}
        'model_class': str
        'dims': '(N, h, w) per variable; samples dim matches N'
    {split}_ground_truth.pt — rescaled ground truth [N, n_vars, h, w]

Usage:
    python icon_copulas_predict.py --run-path feik/genpp/abc123 --split test
    python icon_copulas_predict.py --run-path feik/genpp/abc123 --split val test --batch-size 8
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pickle
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.eval.icon_copulas_eval import stack_predictions
from genpp.eval.icon_predict_eval import (
    _rescale_y,
    get_split_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU step: run EMOS/DRN predictions and save distribution parameters.",
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
        "--force-repredict",
        action="store_true",
        help="Force re-running model predictions even if cached",
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


def predict_split(
    split: str,
    *,
    model,
    trainer: L.Trainer,
    datamodule,
    cfg: DictConfig,
    model_dir: Path,
    force_repredict: bool,
    verbose: bool,
) -> None:
    """Run model predictions for a single split and save results.

    Saves:
        {split}_dist_params.pt — distribution parameters + metadata
        {split}_ground_truth.pt — rescaled ground truth
    """
    split_config = get_split_config(split)

    dataloader_method = getattr(datamodule, split_config["dataloader_method"])
    dataloader = dataloader_method()
    dataset = getattr(datamodule, split_config["dataset_attr"])
    y_select_variables = list(cfg.data.y_select_variables)

    params_path = model_dir / f"{split}_dist_params.pt"
    gt_path = model_dir / f"{split}_ground_truth.pt"

    # --- Distribution parameters ---
    if params_path.exists() and not force_repredict:
        log_msg(f"Cached distribution params found at {params_path}, skipping prediction.", verbose)
    else:
        log_msg(f"Running predictions on {split} split...", verbose)
        raw_predictions = trainer.predict(model, dataloader, return_predictions=True)
        stacked_preds = stack_predictions(raw_predictions)  # type: ignore
        del raw_predictions

        # Serialize reverse module parameters (mean/scale are scalar tensors)
        reverse_modules = datamodule.y_reverseModules
        reverse_info = [
            {"mean": float(mod.mean), "scale": float(mod.scale)} for mod in reverse_modules
        ]

        save_payload = {
            "params": [{k: v.cpu() for k, v in d.items()} for d in stacked_preds],
            "y_select_variables": y_select_variables,
            "samples": dataset.samples,
            "reverse_modules": reverse_info,
            "model_class": cfg.model._target_.split(".")[-1],
            "dims": "(N, h, w) per variable; samples dim matches N",
        }

        log_msg(f"Saving distribution params to {params_path}...", verbose)
        torch.save(save_payload, params_path)
        del stacked_preds
        log_msg(f"Saved {params_path}", verbose)

    # --- Ground truth ---
    if gt_path.exists():
        log_msg(f"Ground truth already exists at {gt_path}, skipping.", verbose)
    else:
        log_msg("Collecting and rescaling ground truth...", verbose)
        y_list = []
        for batch in tqdm(dataloader, desc="Collecting ground truth"):
            y_list.append(batch["y"])
        y_scaled = torch.cat(y_list, dim=0)

        reverse_modules = datamodule.y_reverseModules
        y_rescaled = _rescale_y(y_scaled, reverse_modules)
        del y_scaled

        log_msg(f"Saving ground truth to {gt_path}...", verbose)
        torch.save(y_rescaled.cpu(), gt_path)
        log_msg(f"Saved {gt_path}", verbose)


def process_run(run_path: str, args: argparse.Namespace) -> None:
    """Process a single WandB run: load model, predict, save."""
    log_msg(f"\n{'#' * 60}\nProcessing run: {run_path}\n{'#' * 60}", args.verbose)

    model_id = run_path.split("/")[-1]
    output_dir = BASE_DIR.parent.parent / "outputs"

    log_msg(f"Looking for model with ID: {model_id}", args.verbose)

    model_dirs = list(output_dir.rglob(f"*{model_id}*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directory found for run ID '{model_id}' in {output_dir}")
    model_dir = model_dirs[0].parent.parent.parent

    checkpoints = list(model_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint (.ckpt) found in {model_dir}")
    model_checkpoint = checkpoints[0]

    log_msg(f"Model directory: {model_dir}", args.verbose)
    log_msg(f"Checkpoint: {model_checkpoint}", args.verbose)

    # Load Hydra config
    log_msg("Loading Hydra config...", args.verbose)
    with hydra.initialize_config_dir(config_dir=str(model_dir / ".hydra"), version_base=None):
        cfg: DictConfig = hydra.compose(config_name="config")

    if hasattr(cfg.data.module, "val_batch_size"):
        cfg.data.module.val_batch_size = args.batch_size
    if hasattr(cfg.data.module, "test_batch_size"):
        cfg.data.module.test_batch_size = args.batch_size

    # Setup datamodule
    log_msg("Setting up data module...", args.verbose)
    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Load model
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    class_path = cfg.model._target_
    module_path, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

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

    trainer = L.Trainer(logger=False, accelerator="gpu", devices="auto", enable_progress_bar=True)

    for split in args.split:
        log_msg(f"\n{'=' * 60}\nPredicting split: {split}\n{'=' * 60}", args.verbose)
        predict_split(
            split,
            model=model,
            trainer=trainer,
            datamodule=datamodule,
            cfg=cfg,
            model_dir=model_dir,
            force_repredict=args.force_repredict,
            verbose=args.verbose,
        )

    log_msg(f"Done with run: {run_path}", args.verbose)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    for run_path in args.run_path:
        process_run(run_path, args)

    log_msg("All predictions saved!", args.verbose)


if __name__ == "__main__":
    main()
