"""Evaluate baseline ICON ensemble predictions against reanalysis on GPU.

This script mirrors the ICON evaluation flow used in icon_cgm_predict_eval.py,
but assumes predictions already exist as a NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import trange

from genpp.data.icon import DATA_DIR
from genpp.eval.utils import save_scores_df, update_wandb_run
from genpp.models.scores import (
    EnergyScore,
    EnsembleCRPS,
    MultiScaleEnergyScore,
    MultiScalePatchwiseEnergyScore,
    PatchwiseEnergyScore,
    VariogramScore,
)

DEFAULT_PREDICTIONS_PATH = Path("/hpc/uhome/extmfeik/GenPP/outputs/BASELINE/test_predictions.nc")
DEFAULT_OUTPUT_DIR = Path("/hpc/uhome/extmfeik/GenPP/outputs/BASELINE")
BUILD_SCRIPT_PATH = Path(
    "/hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/build_baseline_test_predictions.py"
)
WANDB_RUN_PATH = "feik/genpp/2x8upzec"


# Keep compatibility with icon_copulas_eval.py imports.
def load_ensemble_tensor(path: Path) -> torch.Tensor:
    """Load one ensemble forecast file into tensor shape [members, variable, x, y]."""
    with xr.open_dataset(path) as ds:
        t2m = ds["T_2M"].squeeze(drop=True)
        vmax = ds["VMAX_10M"].squeeze(drop=True)
        stacked = xr.concat(
            [
                t2m.assign_coords(variable="T_2M").expand_dims("variable"),
                vmax.assign_coords(variable="VMAX_10M").expand_dims("variable"),
            ],
            dim="variable",
        )
        stacked = stacked.transpose("time", "variable", "x", "y")
        return torch.from_numpy(stacked.values).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate precomputed baseline predictions on ICON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to precomputed baseline predictions NetCDF.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory containing a rea folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for scores.csv and test_crps_maps.nc.",
    )
    parser.add_argument(
        "--leadtimes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of leadtimes (hours) to evaluate.",
    )
    parser.add_argument(
        "--skip-variogram",
        action="store_true",
        help="Skip variogram score computation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Scoring device.",
    )
    return parser.parse_args()


def _find_coord(ds: xr.Dataset, primary: str, fallback: str) -> xr.DataArray:
    if primary in ds.coords:
        return ds.coords[primary]
    if primary in ds.variables:
        return ds[primary]
    if fallback in ds.coords:
        return ds.coords[fallback]
    if fallback in ds.variables:
        return ds[fallback]
    raise ValueError(f"Missing {primary}/{fallback} in predictions dataset")


def _to_timedelta64_h(lead_vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(lead_vals)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype("timedelta64[h]")

    result = []
    for item in arr:
        try:
            td = pd.to_timedelta(item)
            hours = int(td.total_seconds() // 3600)
        except Exception:  # noqa: BLE001
            hours = int(item)
        result.append(np.timedelta64(hours, "h"))
    return np.asarray(result, dtype="timedelta64[h]")


def _map_reanalysis_var_name(var_name: str, available: set[str]) -> str:
    candidates = [
        var_name,
        var_name.split("+")[0],
    ]

    if var_name.startswith("T_2M"):
        candidates.extend(["T_2M", "T_2M+height_2.0"])
    if var_name.startswith("VMAX_10M") or var_name.startswith("VMAX_10.0"):
        candidates.extend(["VMAX_10M", "VMAX_10.0", "VMAX_10M+height_2_10.0"])

    for cand in candidates:
        if cand in available:
            return cand

    raise KeyError(
        f"Could not map prediction variable '{var_name}' to reanalysis variables {sorted(available)}"
    )


def load_reanalysis_tensor(path: Path, variable_names: list[str]) -> torch.Tensor:
    """Load reanalysis tensor as [variable, x, y] ordered by prediction variables."""
    with xr.open_dataset(path) as ds:
        available = set(ds.data_vars)
        arrays = []
        for pred_var in variable_names:
            rea_var = _map_reanalysis_var_name(pred_var, available)  # type: ignore
            arrays.append(
                ds[rea_var]
                .squeeze(drop=True)
                .assign_coords(variable=pred_var)
                .expand_dims("variable")
            )
        stacked = xr.concat(arrays, dim="variable")
        stacked = stacked.transpose("variable", "x", "y")
        return torch.from_numpy(stacked.values).float()


def _build_crps_maps_dataset(
    crps_per_margin: np.ndarray,
    init_times: np.ndarray,
    lead_times: np.ndarray,
    variable_names: list[str],
    sx_name: str,
    sy_name: str,
    sx_coord: np.ndarray,
    sy_coord: np.ndarray,
) -> xr.Dataset:
    pair_index = pd.MultiIndex.from_arrays(
        [init_times, lead_times], names=["init_time", "lead_time"]
    )
    dup_mask = pair_index.duplicated(keep=False)
    if dup_mask.any():
        dup_pairs = pair_index[dup_mask].unique()
        preview = ", ".join([f"({i}, {l})" for i, l in dup_pairs[:5]])  # noqa: E741
        raise ValueError(
            "Duplicate (init_time, lead_time) pairs found in baseline predictions. "
            f"Examples: {preview}"
        )

    unique_init = np.sort(np.unique(init_times))
    unique_lead = np.sort(np.unique(lead_times))

    init_to_idx = {v: i for i, v in enumerate(unique_init)}
    lead_to_idx = {v: i for i, v in enumerate(unique_lead)}

    n_var, n_x, n_y = crps_per_margin.shape[1:]
    crps_grid = np.full(
        (len(unique_init), len(unique_lead), n_var, n_x, n_y),
        np.nan,
        dtype=np.float32,
    )

    for i, (init_t, lead_t) in enumerate(zip(init_times, lead_times)):
        ii = init_to_idx[init_t]
        jj = lead_to_idx[lead_t]
        crps_grid[ii, jj] = crps_per_margin[i]

    ds = xr.Dataset(
        {
            "crps": xr.DataArray(
                data=crps_grid,
                dims=["init_time", "lead_time", "variable", sx_name, sy_name],
                coords={
                    "init_time": unique_init,
                    "lead_time": unique_lead,
                    "variable": variable_names,
                    sx_name: sx_coord,
                    sy_name: sy_coord,
                },
            )
        }
    )
    ds = ds.assign_coords(
        valid_time=(("init_time", "lead_time"), unique_init[:, None] + unique_lead[None, :])
    )
    return ds


def _compute_scores_per_leadtime(
    leadtimes: np.ndarray,
    crps_per_margin: np.ndarray,
    es_per_var: np.ndarray,
    es_full: np.ndarray,
    pes_per_var: np.ndarray,
    pes_full: np.ndarray,
    mses_per_var: np.ndarray,
    mses_full: np.ndarray,
    mspes_per_var: np.ndarray,
    mspes_full: np.ndarray,
    variable_names: list[str],
    vs_per_var: np.ndarray | None = None,
    vs_full: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    uniq = np.unique(leadtimes)

    for lead in uniq:
        lead_str = f"{lead / np.timedelta64(1, 'h'):.0f}h"
        mask = leadtimes == lead

        out.setdefault("CRPS_combined", {})[lead_str] = float(np.mean(crps_per_margin[mask]))
        crps_var = np.mean(crps_per_margin[mask], axis=(0, 2, 3))
        for vi, var_name in enumerate(variable_names):
            out.setdefault(f"CRPS_{var_name}", {})[lead_str] = float(crps_var[vi])

        out.setdefault("EnergyScore_combined", {})[lead_str] = float(np.mean(es_full[mask]))
        out.setdefault("PatchwiseEnergyScore_combined", {})[lead_str] = float(
            np.mean(pes_full[mask])
        )
        out.setdefault("MultiScaleEnergyScore_combined", {})[lead_str] = float(
            np.mean(mses_full[mask])
        )
        out.setdefault("MultiScalePatchwiseEnergyScore_combined", {})[lead_str] = float(
            np.mean(mspes_full[mask])
        )

        for vi, var_name in enumerate(variable_names):
            out.setdefault(f"EnergyScore_{var_name}", {})[lead_str] = float(
                np.mean(es_per_var[mask, vi])
            )
            out.setdefault(f"PatchwiseEnergyScore_{var_name}", {})[lead_str] = float(
                np.mean(pes_per_var[mask, vi])
            )
            out.setdefault(f"MultiScaleEnergyScore_{var_name}", {})[lead_str] = float(
                np.mean(mses_per_var[mask, vi])
            )
            out.setdefault(f"MultiScalePatchwiseEnergyScore_{var_name}", {})[lead_str] = float(
                np.mean(mspes_per_var[mask, vi])
            )

        if vs_per_var is not None and vs_full is not None:
            out.setdefault("VariogramScore_combined", {})[lead_str] = float(np.mean(vs_full[mask]))
            for vi, var_name in enumerate(variable_names):
                out.setdefault(f"VariogramScore_{var_name}", {})[lead_str] = float(
                    np.mean(vs_per_var[mask, vi])
                )

    return out


def _overall_metric_rows(
    variable_names: list[str],
    crps_per_margin: np.ndarray,
    es_per_var: np.ndarray,
    es_full: np.ndarray,
    pes_per_var: np.ndarray,
    pes_full: np.ndarray,
    mses_per_var: np.ndarray,
    mses_full: np.ndarray,
    mspes_per_var: np.ndarray,
    mspes_full: np.ndarray,
    vs_per_var: np.ndarray | None = None,
    vs_full: np.ndarray | None = None,
) -> dict[str, float]:
    rows = {
        "CRPS_combined": float(np.mean(crps_per_margin)),
        "EnergyScore_combined": float(np.mean(es_full)),
        "PatchwiseEnergyScore_combined": float(np.mean(pes_full)),
        "MultiScaleEnergyScore_combined": float(np.mean(mses_full)),
        "MultiScalePatchwiseEnergyScore_combined": float(np.mean(mspes_full)),
    }

    crps_var = np.mean(crps_per_margin, axis=(0, 2, 3))
    for vi, var_name in enumerate(variable_names):
        rows[f"CRPS_{var_name}"] = float(crps_var[vi])
        rows[f"EnergyScore_{var_name}"] = float(np.mean(es_per_var[:, vi]))
        rows[f"PatchwiseEnergyScore_{var_name}"] = float(np.mean(pes_per_var[:, vi]))
        rows[f"MultiScaleEnergyScore_{var_name}"] = float(np.mean(mses_per_var[:, vi]))
        rows[f"MultiScalePatchwiseEnergyScore_{var_name}"] = float(np.mean(mspes_per_var[:, vi]))

    if vs_per_var is not None and vs_full is not None:
        rows["VariogramScore_combined"] = float(np.mean(vs_full))
        for vi, var_name in enumerate(variable_names):
            rows[f"VariogramScore_{var_name}"] = float(np.mean(vs_per_var[:, vi]))

    return rows


def main() -> None:
    args = parse_args()

    if not args.predictions_path.exists():
        raise FileNotFoundError(
            f"Baseline predictions not found at: {args.predictions_path}\n"
            f"Please run: {BUILD_SCRIPT_PATH}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Please run on a GPU node.")

    device = torch.device(args.device)
    rea_dir = args.data_dir / "rea"
    if not rea_dir.exists():
        raise FileNotFoundError(f"Reanalysis directory not found: {rea_dir}")

    with xr.open_dataset(args.predictions_path) as ds:
        if "prediction" not in ds.data_vars:
            raise ValueError(f"Missing 'prediction' variable in {args.predictions_path}")

        pred_da = ds["prediction"]
        dims = list(pred_da.dims)
        required = {"time", "sample", "variable"}
        if not required.issubset(set(dims)):
            raise ValueError(f"Prediction dims must include {required}, got {dims}")

        spatial_dims = [d for d in dims if d not in required]
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected exactly 2 spatial dims, got {spatial_dims}")

        sx_name, sy_name = spatial_dims
        pred_da = pred_da.transpose("time", "sample", "variable", sx_name, sy_name)

        init_da = _find_coord(ds, "init_date", "init_time")
        lead_da = _find_coord(ds, "leadtime", "lead_time")

        init_vals = np.asarray(init_da.values).astype("datetime64[ns]")
        lead_vals = _to_timedelta64_h(np.asarray(lead_da.values))
        var_names = [str(v) for v in np.asarray(pred_da.coords["variable"].values)]
        sx_coord = np.asarray(pred_da.coords[sx_name].values)
        sy_coord = np.asarray(pred_da.coords[sy_name].values)

        if len(init_vals) != pred_da.sizes["time"] or len(lead_vals) != pred_da.sizes["time"]:
            raise ValueError("init_date/leadtime length does not match prediction time dimension")

        if args.leadtimes is not None:
            wanted = {np.timedelta64(int(h), "h") for h in args.leadtimes}
            keep_idx = [i for i, td in enumerate(lead_vals) if td in wanted]
            if not keep_idx:
                raise ValueError(f"No samples found for requested leadtimes: {args.leadtimes}")
            pred_da = pred_da.isel(time=keep_idx)
            init_vals = init_vals[keep_idx]
            lead_vals = lead_vals[keep_idx]

        sort_idx = np.lexsort((lead_vals.astype("timedelta64[h]").astype(np.int64), init_vals))
        pred_da = pred_da.isel(time=sort_idx)
        init_vals = init_vals[sort_idx]
        lead_vals = lead_vals[sort_idx]

        n_times = pred_da.sizes["time"]
        n_var = pred_da.sizes["variable"]
        n_x = pred_da.sizes[sx_name]
        n_y = pred_da.sizes[sy_name]

        crps_ens = EnsembleCRPS().to(device)
        es = EnergyScore(clamp=False).to(device)
        pes = PatchwiseEnergyScore(clamp=False).to(device)
        mses = MultiScaleEnergyScore(clamp=False).to(device)
        mspes = MultiScalePatchwiseEnergyScore(clamp=False).to(device)
        vs = VariogramScore(p=0.5, chunk_size=256).to(device) if not args.skip_variogram else None

        crps_per_margin = np.empty((n_times, n_var, n_x, n_y), dtype=np.float32)
        es_per_var = np.empty((n_times, n_var), dtype=np.float32)
        es_full = np.empty((n_times,), dtype=np.float32)
        pes_per_var = np.empty((n_times, n_var), dtype=np.float32)
        pes_full = np.empty((n_times,), dtype=np.float32)
        mses_per_var = np.empty((n_times, n_var), dtype=np.float32)
        mses_full = np.empty((n_times,), dtype=np.float32)
        mspes_per_var = np.empty((n_times, n_var), dtype=np.float32)
        mspes_full = np.empty((n_times,), dtype=np.float32)
        vs_per_var = np.empty((n_times, n_var), dtype=np.float32) if vs is not None else None
        vs_full = np.empty((n_times,), dtype=np.float32) if vs is not None else None

        for i in trange(n_times, desc="Computing scores"):
            init_time = pd.Timestamp(init_vals[i]).to_pydatetime()
            lead_h = int(lead_vals[i].astype("timedelta64[h]").astype(np.int64))
            valid_time = init_time + pd.to_timedelta(lead_h, unit="h")
            rea_path = rea_dir / f"rea_{valid_time:%Y%m%d%H}.nc"
            if not rea_path.exists():
                raise FileNotFoundError(f"Missing reanalysis file required for scoring: {rea_path}")

            pred_np = pred_da.isel(time=i).values.astype(np.float32, copy=False)
            truth = load_reanalysis_tensor(rea_path, var_names)

            pred_i = torch.from_numpy(pred_np).unsqueeze(0).to(device=device)
            truth_i = truth.unsqueeze(0).to(device=device)

            with torch.no_grad():
                crps_map = (
                    crps_ens(pred_i, truth_i).squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                crps_per_margin[i] = crps_map

                es_per_var[i] = (
                    es(pred_i, truth_i, mode="per_var")
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                es_full[i] = float(es(pred_i, truth_i, mode="complete").item())

                pes_per_var[i] = (
                    pes(pred_i, truth_i, mode="per_var")
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pes_full[i] = float(pes(pred_i, truth_i, mode="complete").item())

                mses_per_var[i] = (
                    mses(pred_i, truth_i, mode="per_var")
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                mses_full[i] = float(mses(pred_i, truth_i, mode="complete").item())

                mspes_per_var[i] = (
                    mspes(pred_i, truth_i, mode="per_var")
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                mspes_full[i] = float(mspes(pred_i, truth_i, mode="complete").item())

                if vs is not None and vs_per_var is not None and vs_full is not None:
                    vs_per_var[i] = (
                        vs(pred_i, truth_i, mode="per_var")
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    vs_full[i] = float(vs(pred_i, truth_i, mode="complete").item())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_long = []
    method = "RawEnsembleBaseline"
    dataset = "test"

    overall = _overall_metric_rows(
        variable_names=var_names,
        crps_per_margin=crps_per_margin,
        es_per_var=es_per_var,
        es_full=es_full,
        pes_per_var=pes_per_var,
        pes_full=pes_full,
        mses_per_var=mses_per_var,
        mses_full=mses_full,
        mspes_per_var=mspes_per_var,
        mspes_full=mspes_full,
        vs_per_var=vs_per_var,
        vs_full=vs_full,
    )
    for metric, value in overall.items():
        scores_long.append((method, dataset, metric, "all", float(value)))

    by_lead = _compute_scores_per_leadtime(
        leadtimes=lead_vals,
        crps_per_margin=crps_per_margin,
        es_per_var=es_per_var,
        es_full=es_full,
        pes_per_var=pes_per_var,
        pes_full=pes_full,
        mses_per_var=mses_per_var,
        mses_full=mses_full,
        mspes_per_var=mspes_per_var,
        mspes_full=mspes_full,
        variable_names=var_names,
        vs_per_var=vs_per_var,
        vs_full=vs_full,
    )

    for metric, horizons in by_lead.items():
        for horizon, value in horizons.items():
            scores_long.append((method, dataset, metric, horizon, float(value)))

    scores_df = pd.DataFrame(
        scores_long, columns=["method", "dataset", "metric", "horizon", "value"]
    )
    scores_df = scores_df.sort_values(["dataset", "metric", "horizon"]).reset_index(drop=True)

    scores_path = output_dir / "scores.csv"
    if scores_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Scores file already exists: {scores_path}. Use --overwrite to replace it."
        )
    scores_df.to_csv(scores_path, index=False)

    full_scores = {"test": {}}
    all_metrics = _overall_metric_rows(
        variable_names=var_names,
        crps_per_margin=crps_per_margin,
        es_per_var=es_per_var,
        es_full=es_full,
        pes_per_var=pes_per_var,
        pes_full=pes_full,
        mses_per_var=mses_per_var,
        mses_full=mses_full,
        mspes_per_var=mspes_per_var,
        mspes_full=mspes_full,
        vs_per_var=vs_per_var,
        vs_full=vs_full,
    )
    for metric_name, horizons in by_lead.items():
        full_scores["test"][metric_name] = dict(horizons)
    for metric_name, value in all_metrics.items():
        full_scores["test"].setdefault(metric_name, {})["all"] = float(value)

    update_wandb_run(WANDB_RUN_PATH, full_scores)
    save_scores_df(df=scores_df, run_path=WANDB_RUN_PATH)

    crps_ds = _build_crps_maps_dataset(
        crps_per_margin=crps_per_margin,
        init_times=init_vals,
        lead_times=lead_vals,
        variable_names=var_names,
        sx_name=sx_name,  # type: ignore
        sy_name=sy_name,  # type: ignore
        sx_coord=sx_coord,
        sy_coord=sy_coord,
    )

    crps_path = output_dir / "test_crps_maps.nc"
    if crps_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"CRPS map file already exists: {crps_path}. Use --overwrite to replace it."
        )
    crps_ds.to_netcdf(crps_path)

    print(f"Saved scores to {scores_path}")
    print(f"Saved CRPS maps to {crps_path}")


if __name__ == "__main__":
    main()
