#!/usr/bin/env python
"""Plot PIT histograms for ICON baseline and best models."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from tqdm.auto import tqdm

from genpp import BASE_DIR
from genpp.eval.icon import baseline, best_models


DEFAULT_OUTPUTS_ROOT = BASE_DIR.parent.parent / "outputs"
DEFAULT_OBS_DIR = BASE_DIR / "data" / "icon" / "data" / "rea"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUTS_ROOT / "results" / "icon" / "pit"

N_PIT_BINS = 40
PIT_BINS = (np.arange(N_PIT_BINS + 1) - 0.5) / N_PIT_BINS

TITLE_FS = 14
LABEL_FS = 12
TICK_FS = 10
COL_W = 4.0
ROW_H = 2.2

VAR_DISPLAY = {
    "T_2M": "2m Temperature (K)",
    "VMAX_10M": "10m Wind Speed (m/s)",
}

COLOR_MAPPING = {
    "RAW": "black",
    "EMOS": "#E65100",
    "DRN": "#7E57C2",
    "LNGM (ES, IND)": "#A5D6A7",
    "LNGM (PES, IND)": "#4CAF50",
    "LNGM (MSES, IND)": "#2E7D32",
    "ENGRESSION (ES, IND)": "#BBDEFB",
    "ENGRESSION (PES, IND)": "#42A5F5",
    "ENGRESSION (MSES, IND)": "#1565C0",
    "FM_UNET (IND)": "#00CED1",
    "FM_UNET (DIR)": "#008B8B",
    "FM_UViT (IND)": "#F83E82",
    "FM_UViT (DIR)": "#C2185B",
}


MODEL_PLOT_ORDER = [
    "EMOS",
    "DRN",
    "LNGM (ES, IND)",
    "LNGM (PES, IND)",
    "LNGM (MSES, IND)",
    "ENGRESSION (ES, IND)",
    "ENGRESSION (PES, IND)",
    "ENGRESSION (MSES, IND)",
    "FM_UNET (IND)",
    "FM_UNET (DIR)",
    "FM_UViT (IND)",
    "FM_UViT (DIR)",
]


@dataclass(frozen=True)
class ModelTarget:
    key: str
    label: str
    prediction_path: Path
    color: str


class ObservationStore:
    """Load and cache flattened observation arrays by valid time and variable."""

    def __init__(self, obs_dir: Path) -> None:
        self.obs_dir = obs_dir
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _to_timestamp_str(valid_time: np.datetime64) -> str:
        ts = pd.Timestamp(valid_time)
        return ts.strftime("%Y%m%d%H")

    @staticmethod
    def _candidate_obs_vars(pred_var: str) -> list[str]:
        base = pred_var.split("+")[0]
        candidates = [pred_var, base]
        if base == "T_2M":
            candidates.extend(["T_2M+height_2.0", "T_2M"])
        if base in {"VMAX_10M", "VMAX_10.0"}:
            candidates.extend(["VMAX_10M", "VMAX_10.0", "VMAX_10M+height_2_10.0"])
        return candidates

    @staticmethod
    def _pick_obs_var_name(pred_var: str, available: list[str]) -> str:
        candidates = ObservationStore._candidate_obs_vars(pred_var)
        available_set = set(available)
        for cand in candidates:
            if cand in available_set:
                return cand

        lower_map = {name.lower(): name for name in available}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]

        raise KeyError(
            f"No observation variable match for prediction variable '{pred_var}'. "
            f"Available observation vars: {sorted(available)}"
        )

    def compare_prediction_observation_names(
        self, pred_vars: list[str], sample_valid_time: np.datetime64
    ) -> dict[str, str]:
        ts = self._to_timestamp_str(sample_valid_time)
        obs_path = self.obs_dir / f"rea_{ts}.nc"
        if not obs_path.exists():
            raise FileNotFoundError(f"Observation file not found for preflight check: {obs_path}")

        with xr.open_dataset(obs_path) as ds:
            available = list(ds.data_vars)

        mapping: dict[str, str] = {}
        for pv in pred_vars:
            mapping[pv] = self._pick_obs_var_name(pv, available)
        return mapping

    def get_flat(
        self, valid_time: np.datetime64, pred_var: str, expected_shape: tuple[int, int]
    ) -> np.ndarray:
        ts = self._to_timestamp_str(valid_time)
        cache_key = (ts, pred_var)
        if cache_key in self._cache:
            return self._cache[cache_key]

        obs_path = self.obs_dir / f"rea_{ts}.nc"
        if not obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {obs_path}")

        with xr.open_dataset(obs_path) as ds:
            obs_var = self._pick_obs_var_name(pred_var, list(ds.data_vars))
            arr = ds[obs_var].squeeze(drop=True)

            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D observation field after squeeze for {obs_var} in {obs_path}, got shape {arr.shape}"
                )

            values = np.asarray(arr.values, dtype=np.float32)

        if tuple(values.shape) == expected_shape:
            pass
        elif tuple(values.shape[::-1]) == expected_shape:
            values = values.T
        else:
            raise ValueError(
                f"Observation shape mismatch for {obs_path.name}/{pred_var}: "
                f"obs={values.shape}, expected={expected_shape}"
            )

        flat = values.reshape(-1)
        self._cache[cache_key] = flat
        return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ICON PIT histograms for baseline and best models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT / "BASELINE" / "test_predictions.nc",
    )
    parser.add_argument(
        "--pit-cache-dir", type=Path, default=None, help="Directory to store cached PIT values"
    )
    parser.add_argument(
        "--force-recompute-pit",
        action="store_true",
        help="Ignore existing PIT cache files and recompute",
    )
    parser.add_argument("--bins", type=int, default=N_PIT_BINS)
    parser.add_argument(
        "--chunk-size", type=int, default=8, help="Number of time steps to process per chunk"
    )
    parser.add_argument(
        "--max-times", type=int, default=None, help="Optional limit on number of prediction times"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-ecc", action="store_true", default=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Only run discovery and schema checks"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def pick_prediction_file(model_dir: Path, prefer_ecc: bool = True) -> Path | None:
    preferred = (
        ["test_predictions_ecc.nc", "test_predictions.nc"]
        if prefer_ecc
        else ["test_predictions.nc", "test_predictions_ecc.nc"]
    )

    for name in preferred:
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    any_nc = sorted(model_dir.glob("test_predictions*.nc"))
    return any_nc[0] if any_nc else None


def _display_label(group: str, tag: str | None) -> str:
    if group == "baseline":
        return "RAW"
    if group == "emos":
        return "EMOS"
    if group == "drn":
        return "DRN"
    if group == "chen":
        mapping = {
            "ind_es": "LNGM (ES, IND)",
            "ind_pes": "LNGM (PES, IND)",
            "ind_mses": "LNGM (MSES, IND)",
        }
        return mapping.get(tag or "", f"LNGM ({tag})")
    if group == "engression":
        mapping = {
            "ind_es": "ENGRESSION (ES, IND)",
            "ind_pes": "ENGRESSION (PES, IND)",
            "ind_mses": "ENGRESSION (MSES, IND)",
        }
        return mapping.get(tag or "", f"ENGRESSION ({tag})")
    if group == "fm":
        mapping = {
            "ind_unet": "FM_UNET (IND)",
            "dir_unet": "FM_UNET (DIR)",
            "ind_uvit": "FM_UViT (IND)",
            "dir_uvit": "FM_UViT (DIR)",
        }
        return mapping.get(tag or "", f"FM ({tag})")
    return f"{group.upper()} ({tag})" if tag else group.upper()


def _display_color(label: str) -> str:
    return COLOR_MAPPING.get(label, "#607D8B")


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_paths(cache_dir: Path, target_key: str, pred_var: str) -> tuple[Path, Path]:
    stem = f"{_sanitize_component(target_key)}__{_sanitize_component(pred_var)}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _load_cached_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_cached_meta(meta_path: Path, meta: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _cached_pit_valid(meta: dict | None, expected: dict) -> bool:
    if meta is None:
        return False

    for key in ("prediction_path", "pred_var", "n_times_used", "spatial_shape", "n_valid"):
        if key not in meta:
            return False

    return (
        meta["prediction_path"] == expected["prediction_path"]
        and meta["pred_var"] == expected["pred_var"]
        and int(meta["n_times_used"]) == int(expected["n_times_used"])
        and list(meta["spatial_shape"]) == list(expected["spatial_shape"])
        and int(meta["n_valid"]) >= 0
    )


def compute_pit(
    ensemble: np.ndarray, observation: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Compute randomized PIT values.

    ensemble shape: [n_points, n_members]
    observation shape: [n_points]
    """
    n_members = ensemble.shape[1]
    n_below = np.sum(ensemble < observation[:, None], axis=1)
    n_equal = np.sum(ensemble == observation[:, None], axis=1)
    u = rng.random(observation.shape[0])
    pit = (n_below + u * n_equal) / n_members
    return np.clip(pit, 0.0, 1.0)


def robust_hist_ylim(densities: list[np.ndarray], margin: float = 0.4) -> float:
    if not densities:
        return 1.5
    heights = np.concatenate(densities)
    heights = heights[np.isfinite(heights)]
    if heights.size == 0:
        return 1.5
    return float(np.percentile(heights, 95) * (1.0 + margin))


def discover_targets(args: argparse.Namespace) -> list[ModelTarget]:
    targets: list[ModelTarget] = []

    if args.baseline_path.exists():
        label = _display_label("baseline", baseline.tag)
        targets.append(
            ModelTarget(
                key="baseline",
                label=label,
                prediction_path=args.baseline_path,
                color=_display_color(label),
            )
        )
    else:
        print(f"Warning: baseline prediction file not found: {args.baseline_path}")

    for group_name, entries in best_models:
        for entry in entries:
            pred_path = pick_prediction_file(entry.model_dir, prefer_ecc=args.prefer_ecc)
            if pred_path is None:
                print(
                    f"Warning: no test_predictions*.nc file found for {group_name}/{entry.tag} in {entry.model_dir}"
                )
                continue

            label = _display_label(group_name, entry.tag)
            targets.append(
                ModelTarget(
                    key=f"{group_name}:{entry.tag or 'standard'}",
                    label=label,
                    prediction_path=pred_path,
                    color=_display_color(label),
                )
            )

    return targets


def _prediction_metadata(da: xr.DataArray) -> tuple[np.ndarray, list[str], tuple[str, str]]:
    dims = set(da.dims)
    required = {"time", "sample", "variable"}
    if not required.issubset(dims):
        raise ValueError(f"Prediction dims must contain {required}, got {da.dims}")

    if "rlon" in da.dims and "rlat" in da.dims:
        spatial_dims = ("rlon", "rlat")
    elif "x" in da.dims and "y" in da.dims:
        spatial_dims = ("x", "y")
    else:
        unknown = [d for d in da.dims if d not in {"time", "sample", "variable"}]
        if len(unknown) != 2:
            raise ValueError(f"Could not infer spatial dimensions from {da.dims}")
        spatial_dims = (unknown[0], unknown[1])

    pred_vars = [str(v) for v in da.coords["variable"].values]

    if "time" in da.coords:
        valid_times = np.asarray(da.coords["time"].values)
    elif "init_date" in da.coords and "leadtime" in da.coords:
        valid_times = np.asarray(da.coords["init_date"].values) + np.asarray(
            da.coords["leadtime"].values
        )
    else:
        raise ValueError(
            "Prediction data has neither time coordinate nor init_date+leadtime coordinates"
        )

    return valid_times, pred_vars, spatial_dims


def accumulate_histograms(
    target: ModelTarget,
    obs_store: ObservationStore,
    bins: np.ndarray,
    chunk_size: int,
    rng: np.random.Generator,
    max_times: int | None,
    verbose: bool,
    cache_dir: Path,
    force_recompute_pit: bool,
) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}

    with xr.open_dataset(target.prediction_path) as ds:
        if "prediction" not in ds.data_vars:
            raise ValueError(f"Missing 'prediction' variable in {target.prediction_path}")

        pred_da = ds["prediction"]
        valid_times, pred_vars, spatial_dims = _prediction_metadata(pred_da)

        n_times = pred_da.sizes["time"]
        if max_times is not None:
            n_times = min(n_times, max_times)

        spatial_shape = (pred_da.sizes[spatial_dims[0]], pred_da.sizes[spatial_dims[1]])
        spatial_size = spatial_shape[0] * spatial_shape[1]

        log(
            f"[{target.key}] using {target.prediction_path.name} with {n_times} times, "
            f"{pred_da.sizes['sample']} members, variables={pred_vars}",
            verbose,
        )

        for pred_var in pred_vars:
            cache_npy, cache_meta = _cache_paths(cache_dir, target.key, pred_var)
            expected = {
                "prediction_path": str(target.prediction_path.resolve()),
                "pred_var": pred_var,
                "n_times_used": int(n_times),
                "spatial_shape": [int(spatial_shape[0]), int(spatial_shape[1])],
            }

            meta = _load_cached_meta(cache_meta)
            if not force_recompute_pit and cache_npy.exists() and _cached_pit_valid(meta, expected):
                n_valid = int(meta["n_valid"])
                mmap = np.load(cache_npy, mmap_mode="r")
                if n_valid > mmap.shape[0]:
                    raise ValueError(
                        f"Invalid PIT cache length in {cache_meta}: n_valid={n_valid}, file_size={mmap.shape[0]}"
                    )

                hist, _ = np.histogram(mmap[:n_valid], bins=bins)
                out[pred_var] = (hist.astype(np.float64), n_valid)
                log(
                    f"[{target.key}/{pred_var}] loaded PIT cache: {cache_npy} (n={n_valid})",
                    verbose,
                )
                continue

            counts = np.zeros(len(bins) - 1, dtype=np.float64)
            max_cache_size = int(n_times) * int(spatial_size)
            mmap = np.lib.format.open_memmap(
                cache_npy,
                mode="w+",
                dtype=np.float32,
                shape=(max_cache_size,),
            )
            write_idx = 0

            chunk_iter = range(0, n_times, chunk_size)
            progress = tqdm(
                chunk_iter,
                desc=f"PIT {target.label} [{pred_var}]",
                unit="chunk",
                leave=False,
            )

            for start in progress:
                stop = min(start + chunk_size, n_times)
                arr = (
                    pred_da.sel(variable=pred_var)
                    .isel(time=slice(start, stop))
                    .transpose("time", "sample", spatial_dims[0], spatial_dims[1])
                    .values
                )

                for ti in range(arr.shape[0]):
                    expected_shape = (arr.shape[2], arr.shape[3])
                    obs_flat = obs_store.get_flat(valid_times[start + ti], pred_var, expected_shape)

                    ens = arr[ti].reshape(arr.shape[1], -1).T
                    pit = compute_pit(ens, obs_flat, rng)
                    pit = pit[np.isfinite(pit)]
                    if pit.size == 0:
                        continue

                    end_idx = write_idx + pit.size
                    if end_idx > mmap.shape[0]:
                        raise RuntimeError(
                            f"PIT cache overflow for {target.key}/{pred_var}: "
                            f"attempted={end_idx}, capacity={mmap.shape[0]}"
                        )

                    mmap[write_idx:end_idx] = pit.astype(np.float32, copy=False)
                    write_idx = end_idx

                    hist, _ = np.histogram(pit, bins=bins)
                    counts += hist

                progress.set_postfix(times=f"{stop}/{n_times}")

            mmap.flush()
            meta_out = {
                **expected,
                "n_valid": int(write_idx),
                "cache_file": str(cache_npy),
            }
            _write_cached_meta(cache_meta, meta_out)
            log(f"[{target.key}/{pred_var}] wrote PIT cache: {cache_npy} (n={write_idx})", verbose)

            out[pred_var] = (counts, int(write_idx))

    return out


def plot_histograms(
    targets: list[ModelTarget],
    histograms: dict[str, dict[str, tuple[np.ndarray, int]]],
    bins: np.ndarray,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    variables = sorted({v for model_data in histograms.values() for v in model_data.keys()})

    bin_width = np.diff(bins)
    centers = (bins[:-1] + bins[1:]) / 2

    n_cols = 4
    ordered_labels = {label: idx for idx, label in enumerate(MODEL_PLOT_ORDER)}
    model_targets = sorted(
        [t for t in targets if t.key != "baseline"],
        key=lambda t: (ordered_labels.get(t.label, len(ordered_labels)), t.label),
    )

    for var in variables:
        all_densities: list[np.ndarray] = []
        per_model_density: dict[str, np.ndarray] = {}

        for target in targets:
            if var not in histograms[target.key]:
                continue
            counts, n_total = histograms[target.key][var]
            if n_total == 0:
                density = np.zeros_like(counts, dtype=np.float64)
            else:
                density = counts / (n_total * bin_width)
            per_model_density[target.key] = density
            all_densities.append(density)

        y_max = robust_hist_ylim(all_densities)

        n_model_rows = max(1, int(np.ceil(len(model_targets) / n_cols)))
        n_rows = 1 + n_model_rows
        fig = plt.figure(figsize=(COL_W * n_cols, ROW_H * n_rows), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, n_cols)

        baseline_target = next((t for t in targets if t.key == "baseline"), None)
        if baseline_target is not None and baseline_target.key in per_model_density:
            ax_raw = fig.add_subplot(gs[0, 1:3])
            d = per_model_density[baseline_target.key]
            ax_raw.bar(
                centers,
                d,
                width=bin_width,
                color=baseline_target.color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_raw.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
            ax_raw.set_ylim(0.0, y_max)
            ax_raw.set_title(baseline_target.label, fontsize=TITLE_FS)
            ax_raw.set_ylabel("Density", fontsize=LABEL_FS)
            ax_raw.tick_params(labelsize=TICK_FS)

        for idx, target in enumerate(model_targets):
            if target.key not in per_model_density:
                continue
            row = 1 + idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            d = per_model_density[target.key]
            ax.bar(
                centers,
                d,
                width=bin_width,
                color=target.color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.0, y_max)
            ax.set_title(target.label, fontsize=TITLE_FS)
            if col == 0:
                ax.set_ylabel("Density", fontsize=LABEL_FS)
            if row == n_rows - 1:
                ax.set_xlabel("PIT value", fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)

        var_key = var.split("+")[0]
        fig.suptitle(VAR_DISPLAY.get(var_key, var), fontsize=TITLE_FS + 2)

        out_pdf = results_dir / f"pit_histograms_{var_key}.pdf"
        out_png = results_dir / f"pit_histograms_{var_key}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    sns.set_theme(style="whitegrid")

    bins = (np.arange(args.bins + 1) - 0.5) / args.bins
    rng = np.random.default_rng(args.seed)

    if not args.obs_dir.exists():
        raise FileNotFoundError(f"Observation directory does not exist: {args.obs_dir}")

    targets = discover_targets(args)
    if not targets:
        raise RuntimeError("No prediction targets discovered.")

    pit_cache_dir = (
        args.pit_cache_dir if args.pit_cache_dir is not None else (args.results_dir / "pit_cache")
    )
    pit_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Discovered PIT targets:")
    for t in targets:
        print(f"  - {t.key}: {t.prediction_path}")

    print(f"PIT cache dir: {pit_cache_dir}")
    if args.force_recompute_pit:
        print("Force recompute PIT: enabled")

    # Preflight schema + variable-name comparison (required before full compute).
    obs_store = ObservationStore(args.obs_dir)
    with xr.open_dataset(targets[0].prediction_path) as ds:
        pred_da = ds["prediction"]
        valid_times, pred_vars, spatial_dims = _prediction_metadata(pred_da)

    print("\nPreflight schema check:")
    print(f"  prediction file: {targets[0].prediction_path}")
    print(f"  prediction dims: {pred_da.dims}")
    print(f"  prediction vars: {pred_vars}")
    print(f"  spatial dims: {spatial_dims}")

    name_map = obs_store.compare_prediction_observation_names(pred_vars, valid_times[0])
    print("  observation variable mapping:")
    for k, v in name_map.items():
        print(f"    {k} -> {v}")

    if args.dry_run:
        print("\nDry run complete. No PIT histograms were generated.")
        return

    histograms: dict[str, dict[str, tuple[np.ndarray, int]]] = {}

    for target in targets:
        histograms[target.key] = accumulate_histograms(
            target=target,
            obs_store=obs_store,
            bins=bins,
            chunk_size=args.chunk_size,
            rng=rng,
            max_times=args.max_times,
            verbose=args.verbose,
            cache_dir=pit_cache_dir,
            force_recompute_pit=args.force_recompute_pit,
        )

    plot_histograms(
        targets=targets,
        histograms=histograms,
        bins=bins,
        results_dir=args.results_dir,
    )

    print(f"Saved PIT histograms to {args.results_dir}")


if __name__ == "__main__":
    main()
