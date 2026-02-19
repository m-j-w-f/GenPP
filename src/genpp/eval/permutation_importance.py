#!/usr/bin/env python
"""
Permutation-based feature importance for trained generative postprocessing models.

This script loads a trained model from a WandB run, computes a baseline energy
score, then permutes each input channel one at a time and re-evaluates.  The
relative change ``(ES_permuted - ES_baseline) / ES_baseline`` quantifies the
importance of each channel.

Usage:
    python -m genpp.eval.permutation_importance --run-path feik/genpp/abc123 --split val
    python -m genpp.eval.permutation_importance --run-path feik/genpp/abc123 --split val --n-repeats 5
    python -m genpp.eval.permutation_importance --run-path feik/genpp/abc123 --split val --channels 0 1 2
    python -m genpp.eval.permutation_importance --run-path feik/genpp/abc123 --split val --device 0 --batch-size 32 -v
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import pickle
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import torch
from einops import reduce
from omegaconf import DictConfig, ListConfig, OmegaConf

from genpp import BASE_DIR
from genpp.configs import register_resolvers
from genpp.data.weatherbench2.fast_dataset_simple import TransformTensorDataset
from genpp.models.scores import EnergyScore
from genpp.preproc.transforms import PermuteChannel, Pipe


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Permutation-based feature importance using energy score.",
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
        help="Dataset split to evaluate on",
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
        "--n-repeats",
        type=int,
        default=1,
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
        help="Output CSV path (default: <model_dir>/permutation_importance.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def parse_device(device_str: str) -> list[int]:
    """Parse device string to list of ints."""
    if "," in device_str:
        return [int(d.strip()) for d in device_str.split(",")]
    return [int(device_str)]


def log_msg(msg: str, verbose: bool) -> None:
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)


def _get_split_config(split: str) -> dict[str, str]:
    """Get configuration for the specified split."""
    return {
        "train": {"setup_stage": "fit", "dataloader_method": "train_dataloader", "metadata_key": "train"},
        "val": {"setup_stage": "validate", "dataloader_method": "val_dataloader", "metadata_key": "val"},
        "test": {"setup_stage": "test", "dataloader_method": "test_dataloader", "metadata_key": "test"},
    }[split]


def _load_model(cfg: DictConfig, model_checkpoint: Path, verbose: bool = False):
    """Load a trained model from a checkpoint using the Hydra config.

    Reuses the dynamic-kwargs approach from cgm_predict_eval.py.
    """
    class_path = cfg.model._target_
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    sig = inspect.signature(ModelClass.__init__)
    model_kwargs: dict[str, Any] = {}
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
        model = ModelClass.load_from_checkpoint(model_checkpoint, **model_kwargs)
    except (pickle.UnpicklingError, TypeError):
        model = ModelClass.load_from_checkpoint(
            model_checkpoint, weights_only=False, **model_kwargs
        )

    # Fix internal_td_scaling if needed
    if hasattr(model, "internal_td_scaling"):
        td_scaling = model.internal_td_scaling
        if not getattr(td_scaling, "is_fitted", False):
            log_msg("Setting internal_td_scaling.is_fitted = True", verbose)
            td_scaling.is_fitted = True
        if not getattr(td_scaling, "n_vars", False):
            log_msg("Setting internal_td_scaling.n_vars = 2", verbose)
            td_scaling.n_vars = 2

    return model


def _build_x_transform(
    existing_transform: Any,
    channel_index: int,
    seed: int | None = None,
) -> Pipe:
    """Wrap an existing x_transform (or ``None``) with a :class:`PermuteChannel`.

    If *existing_transform* is already a :class:`Pipe`, prepend the permutation;
    otherwise, create a new :class:`Pipe`.
    """
    permute = PermuteChannel(channel_index=channel_index, seed=seed)
    if existing_transform is None:
        return Pipe([permute])
    if isinstance(existing_transform, Pipe):
        return Pipe([permute] + list(existing_transform.transforms))
    # Single transform (e.g. Pad)
    return Pipe([permute, existing_transform])


def _compute_energy_score(
    predictions: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Compute mean energy score (complete mode) over all time steps.

    Loops over samples to avoid OOM from pairwise expansions.

    Args:
        predictions: ``[n_times, n_samples, features, lon, lat]``
        y: ``[n_times, features, lon, lat]``
        device: device for computation

    Returns:
        Scalar mean energy score.
    """
    es = EnergyScore(clamp=False).to(device)
    es_list: list[torch.Tensor] = []
    n_times = predictions.shape[0]
    for i in range(n_times):
        pred_i = predictions[i : i + 1].to(device)
        y_i = y[i : i + 1].to(device)
        with torch.no_grad():
            es_list.append(es(pred_i, y_i, mode="complete").cpu())
    es_all = torch.cat(es_list, dim=0)
    return reduce(es_all, "t -> 1", "mean").item()


def _get_channel_info(cache_metadata: dict) -> list[dict[str, Any]]:
    """Build a list of ``{index, name, category}`` for every x-feature channel."""
    fm = cache_metadata["feature_metadata"]
    x_vars = cache_metadata.get("x_variables", [])

    channels: list[dict[str, Any]] = []
    for idx, name in enumerate(x_vars):
        if idx in fm.get("all_var_mean_indices", []):
            cat = "all_var_mean"
        elif idx in fm.get("all_var_std_indices", []):
            cat = "all_var_std"
        elif idx in fm.get("meta_var_indices", []):
            cat = "meta_var"
        elif fm.get("pixel_idx_index") is not None and idx in fm["pixel_idx_index"]:
            cat = "pixel_idx"
        else:
            cat = "unknown"
        channels.append({"index": idx, "name": name, "category": cat})
    return channels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 — sequential orchestration script
    """Entry point for permutation importance evaluation."""
    args = parse_args()

    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    # ----- locate model artefacts -----
    model_id = args.run_path.split("/")[-1]
    output_dir = BASE_DIR.parent.parent / "outputs"

    model_dirs = list(output_dir.rglob(f"*{model_id}*"))
    if not model_dirs:
        raise FileNotFoundError(
            f"No model directory found for run ID '{model_id}' in {output_dir}"
        )
    model_dir = model_dirs[0].parent.parent.parent

    checkpoints = list(model_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint (.ckpt) found in {model_dir}")
    model_checkpoint = checkpoints[0]

    log_msg(f"Model directory: {model_dir}", args.verbose)
    log_msg(f"Checkpoint:      {model_checkpoint}", args.verbose)

    # ----- load Hydra config -----
    log_msg("Loading Hydra config...", args.verbose)
    with hydra.initialize_config_dir(
        config_dir=str(model_dir / ".hydra"), version_base=None
    ):
        cfg: DictConfig = hydra.compose(config_name="config")

    cfg.data.module.dataloader_config.train.shuffle = False
    cfg.data.module.dataloader_config.val.shuffle = False
    cfg.data.module.dataloader_config.val.batch_size = args.batch_size
    cfg.data.module.dataloader_config.test.shuffle = False

    # ----- setup data -----
    log_msg("Setting up data module...", args.verbose)
    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")

    # Grab the original x_transform so we can wrap it later
    original_x_transform = cfg.data.module.dataset_config.train.x_transform
    y_transform = cfg.data.module.dataset_config.train.y_transform

    # ----- load model -----
    log_msg(f"Loading model from {model_checkpoint}...", args.verbose)
    model = _load_model(cfg, model_checkpoint, verbose=args.verbose)

    devices = parse_device(args.device)
    trainer = L.Trainer(logger=False, accelerator="gpu", devices=devices)

    # ----- load cached tensors + metadata -----
    all_tensors = torch.load(datamodule.tensor_path)
    cache_metadata = datamodule.cache_metadata
    feature_metadata = cache_metadata["feature_metadata"]
    split = args.split

    x_tensor = all_tensors[split]["x"]
    y_tensor = all_tensors[split]["y"]
    td_tensor = all_tensors[split]["prediction_timedelta"]

    channel_info = _get_channel_info(cache_metadata)
    n_channels = x_tensor.shape[1]  # feature dim

    channels_to_permute = (
        args.channels if args.channels is not None else list(range(n_channels))
    )

    # ----- baseline (no permutation) -----
    log_msg("Computing baseline energy score (no permutation)...", args.verbose)
    # Rebuild the dataset with the original transform to get baseline predictions
    baseline_dataset = TransformTensorDataset(
        x_tensor, y_tensor, td_tensor,
        feature_metadata=feature_metadata,
        x_transform=original_x_transform,
        y_transform=y_transform,
    )
    split_config = _get_split_config(split)
    dl_kwargs = OmegaConf.to_container(
        getattr(cfg.data.module.dataloader_config, split), resolve=True
    )
    dl_kwargs["shuffle"] = False  # type: ignore
    baseline_dl = torch.utils.data.DataLoader(baseline_dataset, **dl_kwargs)  # type: ignore

    baseline_preds = torch.cat(
        trainer.predict(model, baseline_dl, return_predictions=True), dim=0  # type: ignore
    )
    baseline_es = _compute_energy_score(baseline_preds, y_tensor, device="cuda")
    log_msg(f"Baseline energy score: {baseline_es:.6f}", args.verbose)

    # ----- permutation loop -----
    results: list[dict[str, Any]] = []

    for ch_idx in channels_to_permute:
        ch_name = channel_info[ch_idx]["name"] if ch_idx < len(channel_info) else f"channel_{ch_idx}"
        ch_cat = channel_info[ch_idx]["category"] if ch_idx < len(channel_info) else "unknown"
        log_msg(f"\nPermuting channel {ch_idx} ({ch_name})...", args.verbose)

        repeat_scores: list[float] = []
        for r in range(args.n_repeats):
            # Seed global RNG for reproducibility across runs;
            # each sample still gets an independent random permutation
            # because PermuteChannel(seed=None) creates an unseeded generator.
            torch.manual_seed(args.seed + ch_idx * args.n_repeats + r)
            perm_transform = _build_x_transform(original_x_transform, ch_idx, seed=None)

            perm_dataset = TransformTensorDataset(
                x_tensor, y_tensor, td_tensor,
                feature_metadata=feature_metadata,
                x_transform=perm_transform,
                y_transform=y_transform,
            )
            perm_dl = torch.utils.data.DataLoader(perm_dataset, **dl_kwargs)  # type: ignore

            perm_preds = torch.cat(
                trainer.predict(model, perm_dl, return_predictions=True), dim=0  # type: ignore
            )
            es_permuted = _compute_energy_score(perm_preds, y_tensor, device="cuda")
            repeat_scores.append(es_permuted)
            log_msg(f"  repeat {r}: ES={es_permuted:.6f}", args.verbose)

        mean_es = float(np.mean(repeat_scores))
        std_es = float(np.std(repeat_scores)) if len(repeat_scores) > 1 else 0.0
        importance = (mean_es - baseline_es) / baseline_es if baseline_es != 0 else np.nan

        results.append({
            "channel_index": ch_idx,
            "channel_name": ch_name,
            "category": ch_cat,
            "baseline_es": baseline_es,
            "permuted_es": mean_es,
            "importance": importance,
            "importance_std": std_es / abs(baseline_es) if baseline_es != 0 else np.nan,
        })

        log_msg(
            f"  → importance = {importance:.4f} "
            f"(ES baseline={baseline_es:.6f}, permuted={mean_es:.6f})",
            args.verbose,
        )

    # ----- write results -----
    output_path = Path(args.output) if args.output else model_dir / "permutation_importance.csv"
    fieldnames = [
        "channel_index", "channel_name", "category",
        "baseline_es", "permuted_es", "importance", "importance_std",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # ----- print summary -----
    sorted_results = sorted(results, key=lambda r: r["importance"], reverse=True)
    print("\n--- Feature Importance (sorted by importance) ---")
    print(f"{'Idx':>4}  {'Channel':<50}  {'Category':<15}  {'Importance':>10}")
    print("-" * 85)
    for r in sorted_results:
        print(
            f"{r['channel_index']:>4}  {r['channel_name']:<50}  "
            f"{r['category']:<15}  {r['importance']:>10.4f}"
        )


if __name__ == "__main__":
    main()
