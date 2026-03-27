"""Helpers for launching and merging DRN permutation-importance runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "channel_index",
    "channel_name",
    "category",
    "baseline_es",
    "permuted_es",
    "importance",
    "importance_std",
]


def _importance_key(row: dict[str, Any]) -> float:
    """Return a sortable numeric importance, falling back to -inf on parse errors."""
    try:
        return float(row.get("importance", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _merge_results(channel_files: dict[int, Path], output_path: Path) -> list[dict[str, Any]]:
    """Merge per-channel CSV outputs into one sorted CSV.

    Args:
        channel_files: Mapping from channel index to CSV path.
        output_path: Target merged CSV path.

    Returns:
        List of merged rows sorted by descending importance.
    """
    merged_rows: list[dict[str, Any]] = []

    for channel_idx in sorted(channel_files):
        csv_path = channel_files[channel_idx]
        if not csv_path.exists():
            print(f"WARNING: Missing channel result file for channel {channel_idx}: {csv_path}")
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            merged_rows.extend(reader)

    merged_rows.sort(key=_importance_key, reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged_rows)

    return merged_rows


__all__ = ["FIELDNAMES", "_merge_results"]
