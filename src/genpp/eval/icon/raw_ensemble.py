"""Compute Energy Score and CRPS for raw ICON ensemble outputs.

For each ensemble forecast file, this script loads the corresponding reanalysis
file, computes:
    - Energy score (combined and per variable)
    - Mean CRPS (combined and per variable)

Results are saved to ``<output-dir>/raw_ensemble_scores_YYYYMM.csv``.
The script is designed to run per month and iterate all lead times.
"""

import argparse
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
import xarray as xr

from genpp.data.icon import DATA_DIR
from genpp.models.scores import EnergyScore, EnsembleCRPS

VARIABLES = ["T_2M", "VMAX_10M"]
DEFAULT_LEADTIMES = [24, 48, 72, 96, 120]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Energy Score and CRPS for raw ICON ensemble forecasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--year", type=int, required=True, help="Year (YYYY)")
    parser.add_argument("--month", type=int, required=True, help="Month (MM)")
    parser.add_argument("--day", type=int, help="Optional day filter (DD) to process a single day")
    parser.add_argument(
        "--leadtime",
        dest="leadtimes",
        action="append",
        type=int,
        help="Lead time (hours). Can be provided multiple times. Defaults to all standard lead times.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory containing 'ens' and 'rea' folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store score CSVs. Defaults to <data-dir>/scores/raw_ensemble",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly CSV instead of appending/skipping computed entries.",
    )
    return parser.parse_args()


def list_ensemble_files(
    ens_dir: Path, year: int, month: int, leadtimes: Iterable[int], day: int | None
) -> list[Path]:
    """Return ensemble NetCDF files for the requested month/day and lead times."""
    files: list[Path] = []
    for lt in sorted(set(leadtimes)):
        pattern = f"ens_{year:04d}{month:02d}"
        pattern += f"{day:02d}" if day is not None else ""
        pattern += f"*_{lt}.nc"
        files.extend(sorted(ens_dir.glob(pattern)))
    return sorted(files)


def _stack_variables(ds: xr.Dataset) -> xr.DataArray:
    """Extract and stack T_2M and VMAX_10M into a single DataArray."""
    t2m = ds["T_2M"].squeeze(drop=True)
    vmax = ds["VMAX_10M"].squeeze(drop=True)

    stacked = xr.concat(
        [
            t2m.assign_coords(variable="T_2M").expand_dims("variable"),
            vmax.assign_coords(variable="VMAX_10M").expand_dims("variable"),
        ],
        dim="variable",
    )
    return stacked


def load_ensemble_tensor(path: Path) -> torch.Tensor:
    """Load ensemble NetCDF into a torch tensor of shape [members, 2, y, x]."""
    with xr.open_dataset(path) as ds:
        stacked = _stack_variables(ds)
        stacked = stacked.transpose("time", "variable", "y", "x")
        return torch.from_numpy(stacked.values).float()


def load_reanalysis_tensor(path: Path) -> torch.Tensor:
    """Load reanalysis NetCDF into a torch tensor of shape [2, y, x]."""
    with xr.open_dataset(path) as ds:
        stacked = _stack_variables(ds)
        stacked = stacked.transpose("variable", "y", "x")
        return torch.from_numpy(stacked.values).float()


def compute_scores(ensemble: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """Compute combined/per-variable energy score and CRPS."""
    if ensemble.dim() != 4:
        raise ValueError(f"Ensemble tensor must be 4D [members, 2, y, x], got {ensemble.shape}")
    if truth.dim() != 3:
        raise ValueError(f"Truth tensor must be 3D [2, y, x], got {truth.shape}")

    ensemble = ensemble.float().contiguous()
    truth = truth.float().contiguous()

    ensemble_b = ensemble.unsqueeze(0)  # [1, n, c, y, x]
    truth_b = truth.unsqueeze(0)  # [1, c, y, x]

    crps_model = EnsembleCRPS(n_axis=-4)
    crps_map = crps_model(ensemble_b, truth_b).squeeze(0)
    crps_per_var = crps_map.mean(dim=(-1, -2))
    crps_mean = crps_map.mean()

    es_model = EnergyScore(beta=1.0, clamp=False, unbiased=False)
    es_combined = es_model(ensemble_b, truth_b, mode="complete")
    es_per_var = es_model(ensemble_b, truth_b, mode="per_var").squeeze()
    es_per_var_flat = es_per_var.view(-1)

    return {
        "energy_score": float(es_combined.item()),
        "energy_score_T_2M": float(es_per_var_flat[0].item()),
        "energy_score_VMAX_10M": float(es_per_var_flat[1].item()),
        "crps_mean": float(crps_mean.item()),
        "crps_T_2M": float(crps_per_var[0].item()),
        "crps_VMAX_10M": float(crps_per_var[1].item()),
    }


def main() -> None:
    args = parse_args()
    leadtimes = args.leadtimes if args.leadtimes else DEFAULT_LEADTIMES

    data_dir = args.data_dir
    ens_dir = data_dir / "ens"
    rea_dir = data_dir / "rea"
    output_dir = args.output_dir or data_dir / "scores" / "raw_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"raw_ensemble_scores_{args.year:04d}{args.month:02d}.csv"

    existing_df = None
    processed_keys: set[tuple[str, int]] = set()
    if output_file.exists() and not args.overwrite:
        existing_df = pd.read_csv(output_file)
        if not existing_df.empty:
            processed_keys = set(
                zip(existing_df["init_time"].astype(str), existing_df["leadtime_hours"].astype(int))
            )

    ens_files = list_ensemble_files(ens_dir, args.year, args.month, leadtimes, args.day)
    if not ens_files:
        print("No ensemble files found for the requested period.")
        return

    results: list[dict[str, float | str | int]] = []

    for ens_path in ens_files:
        parts = ens_path.stem.split("_")
        if len(parts) < 3:
            continue

        init_str = parts[1]
        leadtime = int(parts[2])

        if args.day and int(init_str[6:8]) != args.day:
            continue

        key = (init_str, leadtime)
        if key in processed_keys:
            continue

        try:
            init_time = datetime.strptime(init_str, "%Y%m%d%H")
        except ValueError:
            continue

        valid_time = init_time + timedelta(hours=leadtime)
        rea_path = rea_dir / f"rea_{valid_time:%Y%m%d}.nc"
        if not rea_path.exists():
            print(f"Skipping {ens_path.name}: missing reanalysis file {rea_path.name}")
            continue

        try:
            ensemble = load_ensemble_tensor(ens_path)
            truth = load_reanalysis_tensor(rea_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load data for {ens_path.name}: {exc}")
            continue

        try:
            scores = compute_scores(ensemble, truth)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Failed to compute scores for {ens_path.name} "
                f"(ensemble shape {ensemble.shape}, truth shape {truth.shape}): {exc}"
            )
            continue
        results.append(
            {
                "init_time": init_str,
                "valid_date": valid_time.strftime("%Y%m%d"),
                "leadtime_hours": leadtime,
                "n_members": int(ensemble.shape[0]),
                **scores,
            }
        )

    if not results:
        print("No scores computed.")
        return

    new_df = pd.DataFrame(results)
    if existing_df is not None and not args.overwrite:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["init_time", "leadtime_hours"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["init_time", "leadtime_hours"]).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    print(f"Saved {len(new_df)} new entries to {output_file}")


if __name__ == "__main__":
    main()
