#!/usr/bin/env python
"""
Parallel launcher for DRN/EMOS permutation-based feature importance.

Discovers all input channels from the cached dataset metadata, then launches
one ``permutation_importance_drn.py`` subprocess per channel in parallel.
Each subprocess writes its result to a unique temporary CSV to avoid write
collisions.  Once all subprocesses finish, the results are merged into a
single sorted output CSV.

Usage:
    python -m genpp.eval.launch_permutation_importance_drn --run-path feik/genpp/m5y9kwlh -v
    python -m genpp.eval.launch_permutation_importance_drn --run-path feik/genpp/m5y9kwlh --max-parallel 4 --device 0 -v
    python -m genpp.eval.launch_permutation_importance_drn --run-path feik/genpp/m5y9kwlh --copula gca --split val
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra

from genpp.configs import register_resolvers
from genpp.eval import ModelEntry


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel launcher for DRN/EMOS permutation feature importance.",
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
        help="GPU device index (e.g., '0')",
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
        help="Number of permutation repeats per channel",
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
        help="Optional subset of channel indices (default: all)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of parallel subprocesses (default: all channels at once)",
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


def _discover_channels(run_path: str, split: str, verbose: bool) -> tuple[list[int], Path]:
    """Discover all input channels from cached dataset metadata.

    Returns:
        A tuple of (channel_indices, model_dir).
    """
    model_entry = ModelEntry(id=run_path)
    model_dir = model_entry.model_dir

    if verbose:
        print(f"Model directory: {model_dir}")

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(model_dir / ".hydra"), version_base=None):
        cfg = hydra.compose(config_name="config")

    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()

    setup_stages = {"train": "fit", "val": "validate", "test": "test"}
    datamodule.setup(stage=setup_stages[split])

    all_tensors = torch.load(datamodule.tensor_path)
    x_tensor = all_tensors[split]["x"]
    n_channels = x_tensor.shape[1]

    if verbose:
        print(f"Discovered {n_channels} channels in {split} split")

    return list(range(n_channels)), model_dir


def _launch_channel(
    channel_idx: int,
    args: argparse.Namespace,
    output_path: Path,
) -> subprocess.Popen:
    """Launch a single permutation_importance_drn subprocess for one channel."""
    cmd = [
        sys.executable,
        "-m",
        "genpp.eval.permutation_importance_drn",
        "--run-path",
        args.run_path,
        "--split",
        args.split,
        "--copula",
        args.copula,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--n-repeats",
        str(args.n_repeats),
        "--seed",
        str(args.seed),
        "--channels",
        str(channel_idx),
        "--output",
        str(output_path),
    ]
    if args.verbose:
        cmd.append("-v")

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


FIELDNAMES = [
    "channel_index",
    "channel_name",
    "category",
    "baseline_es",
    "permuted_es",
    "importance",
    "importance_std",
]


def _merge_results(per_channel_files: dict[int, Path], output_path: Path) -> list[dict]:
    """Merge per-channel CSV results into a single sorted output file.

    Reads each per-channel CSV, collects all rows, sorts by importance
    (descending), and writes the merged CSV.

    Returns:
        The merged list of result dicts for printing the summary.
    """
    all_rows: list[dict] = []
    for ch_idx in sorted(per_channel_files.keys()):
        csv_path = per_channel_files[ch_idx]
        if not csv_path.exists():
            print(f"  WARNING: no output for channel {ch_idx} ({csv_path})")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    # Sort by importance descending
    all_rows.sort(key=lambda r: float(r.get("importance", 0)), reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


def main() -> None:
    """Entry point for the parallel launcher."""
    args = parse_args()

    try:
        register_resolvers()
    except Exception:
        pass

    # ----- discover channels -----
    channels, model_dir = _discover_channels(args.run_path, args.split, args.verbose)
    if args.channels is not None:
        channels = [ch for ch in args.channels if ch in channels]

    output_path = Path(args.output) if args.output else model_dir / "permutation_importance_drn.csv"

    if args.verbose:
        print(f"Channels to permute: {channels}")
        print(f"Output path: {output_path}")
        max_p = args.max_parallel or len(channels)
        print(f"Max parallel subprocesses: {max_p}")

    # ----- launch subprocesses in batches -----
    # Each channel writes to a unique temporary file to avoid collisions
    tmp_dir = tempfile.mkdtemp(prefix="perm_importance_drn_")
    per_channel_files: dict[int, Path] = {}

    max_parallel = args.max_parallel or len(channels)
    pending: list[tuple[int, subprocess.Popen, Path]] = []
    channel_queue = list(channels)

    while channel_queue or pending:
        # Fill up to max_parallel
        while channel_queue and len(pending) < max_parallel:
            ch_idx = channel_queue.pop(0)
            ch_output = Path(tmp_dir) / f"channel_{ch_idx}.csv"
            per_channel_files[ch_idx] = ch_output
            proc = _launch_channel(ch_idx, args, ch_output)
            pending.append((ch_idx, proc, ch_output))
            if args.verbose:
                print(f"  Launched channel {ch_idx} (PID {proc.pid})")

        # Wait for at least one to finish
        if pending:
            still_running = []
            for ch_idx, proc, ch_output in pending:
                ret = proc.poll()
                if ret is not None:
                    stdout_data = proc.stdout.read().decode() if proc.stdout else ""
                    stderr_data = proc.stderr.read().decode() if proc.stderr else ""
                    if ret == 0:
                        if args.verbose:
                            print(f"  ✓ Channel {ch_idx} finished (exit 0)")
                            if stdout_data.strip():
                                # Print only the last few relevant lines
                                lines = stdout_data.strip().split("\n")
                                for line in lines[-3:]:
                                    print(f"    {line}")
                    else:
                        print(f"  ✗ Channel {ch_idx} FAILED (exit {ret})")
                        if stderr_data.strip():
                            for line in stderr_data.strip().split("\n")[-10:]:
                                print(f"    {line}")
                else:
                    still_running.append((ch_idx, proc, ch_output))
            pending = still_running

            # Brief sleep to avoid busy-waiting
            if pending:
                time.sleep(2)

    # ----- merge results -----
    if args.verbose:
        print(f"\nMerging results from {len(per_channel_files)} channels...")

    all_rows = _merge_results(per_channel_files, output_path)

    print(f"\nResults saved to {output_path}")

    # ----- print summary -----
    print(f"\n--- Feature Importance (sorted by importance, copula={args.copula}) ---")
    print(f"{'Idx':>4}  {'Channel':<50}  {'Category':<15}  {'Importance':>10}")
    print("-" * 85)
    for r in all_rows:
        print(
            f"{r['channel_index']:>4}  {r['channel_name']:<50}  "
            f"{r['category']:<15}  {float(r['importance']):>10.4f}"
        )

    # ----- cleanup temp files -----
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
