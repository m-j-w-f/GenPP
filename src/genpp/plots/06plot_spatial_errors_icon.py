import gc
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

# Use a non-interactive backend for script execution stability.
matplotlib.use("Agg", force=True)
import itertools

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from genpp.eval.icon import baseline, best_models
from genpp.plots import RESULTS_DIR


def pick_crps_map_file(model_dir: Path) -> Path | None:
    exact = model_dir / "test_crps_maps.nc"
    if exact.exists():
        return exact
    found = sorted(model_dir.rglob("test_crps_maps.nc"))
    return found[0] if found else None


def get_variable_dim(dims: tuple[str, ...]) -> str:
    for dim in ("variable", "feature"):
        if dim in dims:
            return dim
    raise ValueError(f"Could not find variable dimension in {dims}")


def get_spatial_dims(dims: tuple[str, ...]) -> tuple[str, str]:
    pairs = [
        ("rlon", "rlat"),
        ("x", "y"),
        ("lon", "lat"),
        ("longitude", "latitude"),
    ]
    for x_name, y_name in pairs:
        if x_name in dims and y_name in dims:
            return x_name, y_name
    raise ValueError(f"Could not infer spatial dims from {dims}")


def _to_str_list(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in np.asarray(values).ravel():
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def _pick_main_data_var(ds: Dataset) -> str:
    if "crps" in ds.variables:
        return "crps"

    # Fallback: choose the non-coordinate variable with the most dimensions.
    candidates = []
    for name, var in ds.variables.items():
        if name in ds.dimensions:
            continue
        candidates.append((var.ndim, name))

    if not candidates:
        raise ValueError("No suitable data variable found in NetCDF file")

    candidates.sort(reverse=True)
    return candidates[0][1]


def load_mean_crps_fields(
    path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load CRPS map and average over all non-variable, non-spatial dims.

    Uses pure netCDF4 + NumPy streaming reduction (single-threaded).
    """
    with Dataset(path, mode="r") as ds:
        data_var_name = _pick_main_data_var(ds)
        var = ds.variables[data_var_name]
        dims = tuple(var.dimensions)

        var_dim = get_variable_dim(dims)
        x_dim, y_dim = get_spatial_dims(dims)

        x_coord = np.asarray(ds.variables[x_dim][:])
        y_coord = np.asarray(ds.variables[y_dim][:])

        var_axis = dims.index(var_dim)
        x_axis = dims.index(x_dim)
        y_axis = dims.index(y_dim)

        if var_dim in ds.variables:
            var_names = _to_str_list(np.asarray(ds.variables[var_dim][:]))
        else:
            var_names = [str(i) for i in range(var.shape[var_axis])]

        reduce_dims = [d for d in dims if d not in {var_dim, x_dim, y_dim}]
        kept_dims = [d for d in dims if d in {var_dim, x_dim, y_dim}]
        perm = [kept_dims.index(var_dim), kept_dims.index(x_dim), kept_dims.index(y_dim)]

        n_var = var.shape[var_axis]
        n_x = var.shape[x_axis]
        n_y = var.shape[y_axis]
        sum_arr = np.zeros((n_var, n_x, n_y), dtype=np.float64)
        cnt_arr = np.zeros((n_var, n_x, n_y), dtype=np.float64)

        if reduce_dims:
            reduce_sizes = [var.shape[dims.index(d)] for d in reduce_dims]
            for idx_tuple in itertools.product(*(range(sz) for sz in reduce_sizes)):
                sel = [slice(None)] * var.ndim
                for d, idx in zip(reduce_dims, idx_tuple):
                    sel[dims.index(d)] = idx

                slab_raw = var[tuple(sel)]
                if np.ma.isMaskedArray(slab_raw):
                    slab = slab_raw.filled(np.nan).astype(np.float64, copy=False)
                else:
                    slab = np.asarray(slab_raw, dtype=np.float64)

                slab = np.transpose(slab, axes=perm)
                sum_arr += np.where(np.isfinite(slab), slab, 0.0)
                cnt_arr += np.isfinite(slab)
        else:
            slab_raw = var[:]
            if np.ma.isMaskedArray(slab_raw):
                slab = slab_raw.filled(np.nan).astype(np.float64, copy=False)
            else:
                slab = np.asarray(slab_raw, dtype=np.float64)

            slab = np.transpose(slab, axes=perm)
            sum_arr += np.where(np.isfinite(slab), slab, 0.0)
            cnt_arr += np.isfinite(slab)

        reduced_np = np.divide(
            sum_arr,
            cnt_arr,
            out=np.full_like(sum_arr, np.nan, dtype=np.float64),
            where=cnt_arr > 0,
        ).astype(np.float32, copy=False)

        if np.isinf(reduced_np).any():
            inf_count = int(np.isinf(reduced_np).sum())
            raise ValueError(f"Found {inf_count} infinite values in reduced CRPS map: {path}")

    out: dict[str, np.ndarray] = {}
    for i, var_name in enumerate(var_names):
        out[var_name] = reduced_np[i]

    return out, x_coord, y_coord


def _var_label(var_name: str) -> dict[str, float | str]:
    var_settings = {
        "2m_temperature": {
            "cmap": "magma",
            "label": "CRPS 2m temperature",
            "vmax_cap": 1.8,
            "over_color": "cyan",
        },
        "10m_wind_speed": {
            "cmap": "viridis",
            "label": "CRPS 10m wind speed",
            "vmax_cap": 1.1,
            "over_color": "hotpink",
        },
    }

    if var_name == "T_2M+height_2.0" or "T_2M" in var_name:
        return var_settings["2m_temperature"]
    if var_name == "VMAX_10M+height_2_10.0" or "VMAX_10M" in var_name or "wind" in var_name.lower():
        return var_settings["10m_wind_speed"]

    return {
        "cmap": "cividis",
        "label": f"CRPS {var_name}",
        "vmax_cap": float("inf"),
        "over_color": "white",
    }


def _pick_var_in_fields(fields: dict[str, np.ndarray], target_var: str) -> np.ndarray:
    if target_var in fields:
        return fields[target_var]

    target_key = target_var.split("+")[0]
    for candidate, values in fields.items():
        if candidate.split("+")[0] == target_key:
            return values

    raise KeyError(f"Variable {target_var} not found in {list(fields.keys())}")


# Build CRPS path index without loading full maps.
model_crps_paths: dict[str, dict[str, Path]] = {}
for model_name, model_entries in best_models:
    if not model_entries:
        continue
    model_crps_paths[model_name] = {}
    for entry in model_entries:
        variant_key = entry.tag or "standard"
        crps_path = pick_crps_map_file(entry.model_dir)
        if crps_path is None:
            print(f"  x {model_name}/{variant_key}: no test_crps_maps.nc found")
            continue
        model_crps_paths[model_name][variant_key] = crps_path
        print(f"  check {model_name}/{variant_key}: {crps_path.name}")

print("\nAvailable models / variants:")
for mname, variants in model_crps_paths.items():
    print(f"  {mname}: {list(variants.keys())}")

baseline_crps_path = pick_crps_map_file(baseline.model_dir)
if baseline_crps_path is None:
    print("  x baseline: no test_crps_maps.nc found")
else:
    print(f"  check baseline: {baseline_crps_path.name}")

# Cache each file once; values are already reduced to [var, x, y], so memory is small.
# Sanity check: print one scalar mean per model/variant across variable+spatial dims.
cache: dict[Path, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
for mkey, variants in model_crps_paths.items():
    for vname, crps_path in variants.items():
        if crps_path in cache:
            continue
        fields, xcoord, ycoord = load_mean_crps_fields(crps_path)
        cache[crps_path] = (fields, xcoord, ycoord)
        field_stack = np.stack(list(fields.values()), axis=0)
        model_mean = float(np.nanmean(field_stack))
        model_min = float(np.nanmin(field_stack))
        model_max = float(np.nanmax(field_stack))
        print(
            f"  sanity {mkey}/{vname}: mean CRPS={model_mean:.6f}, "
            f"min={model_min:.6f}, max={model_max:.6f}"
        )

baseline_cache_key: Path | None = None
if baseline_crps_path is not None:
    if baseline_crps_path not in cache:
        fields, xcoord, ycoord = load_mean_crps_fields(baseline_crps_path)
        cache[baseline_crps_path] = (fields, xcoord, ycoord)
    baseline_fields, _, _ = cache[baseline_crps_path]
    baseline_stack = np.stack(list(baseline_fields.values()), axis=0)
    baseline_mean = float(np.nanmean(baseline_stack))
    baseline_min = float(np.nanmin(baseline_stack))
    baseline_max = float(np.nanmax(baseline_stack))
    print(
        f"  sanity baseline: mean CRPS={baseline_mean:.6f}, "
        f"min={baseline_min:.6f}, max={baseline_max:.6f}"
    )
    baseline_cache_key = baseline_crps_path

if not cache:
    raise RuntimeError("No CRPS map files could be loaded")

reference_path = next(iter(cache.keys()))
ref_fields, _, _ = cache[reference_path]
crps_vars = list(ref_fields.keys())
print("\nCRPS variables:", crps_vars)

# Keep the same panel logic as 03: fixed model columns, rows are model variants.
MODEL_COLUMNS = [
    ("emos", "EMOS", None),
    ("drn", "DRN", None),
    ("chen", "LNGM (IND)", "ind_"),
    ("engression", "ENGRESSION (IND)", "ind_"),
    ("fm", "FM", None),
]

use_rotated_coords = True
rotated_crs = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=40.0)
plot_proj = rotated_crs if use_rotated_coords else ccrs.PlateCarree()
DRAW_MAP_FEATURES = True

results_dir = RESULTS_DIR / "results" / "icon" / "spatial_errors"
results_dir.mkdir(parents=True, exist_ok=True)

for var_name in crps_vars:
    cfg = _var_label(var_name)
    cmap = plt.get_cmap(str(cfg["cmap"])).copy()
    cmap.set_over(str(cfg["over_color"]))

    col_variants: list[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = []
    col_titles: list[str] = []

    for mkey, display, prefix in MODEL_COLUMNS:
        entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        variants = model_crps_paths.get(mkey, {})

        for vname, crps_path in variants.items():
            if prefix is not None and not vname.startswith(prefix):
                continue
            tag = vname[len(prefix) :] if prefix and vname.startswith(prefix) else vname
            try:
                fields, xcoord, ycoord = cache[crps_path]
                arr = _pick_var_in_fields(fields, var_name)
            except Exception as exc:
                print(f"  x {mkey}/{vname}/{var_name}: {exc}")
                continue
            entries.append((tag, arr, xcoord, ycoord))

        col_variants.append(entries)
        col_titles.append(display)

    n_rows = max((len(v) for v in col_variants), default=1) or 1
    n_cols = 1 + len(MODEL_COLUMNS)

    all_vals = []
    for entries in col_variants:
        for _, arr, _, _ in entries:
            all_vals.append(arr.ravel())
    baseline_entry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if baseline_cache_key is not None:
        try:
            b_fields, b_xcoord, b_ycoord = cache[baseline_cache_key]
            b_arr = _pick_var_in_fields(b_fields, var_name)
            baseline_entry = (b_arr, b_xcoord, b_ycoord)
            all_vals.append(b_arr.ravel())
        except Exception as exc:
            print(f"  x baseline/{var_name}: {exc}")
    if not all_vals:
        print(f"  x {var_name}: no data to plot")
        continue

    subplot_kw = {"projection": plot_proj} if plot_proj is not None else {}
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 4.5 * n_rows),
        subplot_kw=subplot_kw,
    )
    fig.subplots_adjust(wspace=0.2, hspace=-0.7)
    axes = np.atleast_2d(axes)

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]

            if col == 0:
                if row == 0 and baseline_entry is not None:
                    b_arr, b_xcoord, b_ycoord = baseline_entry
                    if use_rotated_coords:
                        pcm = ax.pcolormesh(
                            b_xcoord,
                            b_ycoord,
                            b_arr.T,
                            cmap=cmap,
                            transform=rotated_crs,
                            shading="auto",
                            vmax=float(cfg["vmax_cap"]),
                            rasterized=True,
                        )
                        if DRAW_MAP_FEATURES:
                            ax.coastlines(resolution="50m")
                            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    else:
                        pcm = ax.pcolormesh(
                            b_xcoord,
                            b_ycoord,
                            b_arr.T,
                            cmap=cmap,
                            shading="auto",
                            vmax=float(cfg["vmax_cap"]),
                            rasterized=True,
                        )
                    ax.set_title("Baseline", fontsize=14)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_visible(False)
            else:
                entries = col_variants[col - 1]

                if row < len(entries):
                    tag, arr, xcoord, ycoord = entries[row]
                    if use_rotated_coords:
                        pcm = ax.pcolormesh(
                            xcoord,
                            ycoord,
                            arr.T,
                            cmap=cmap,
                            transform=rotated_crs,
                            shading="auto",
                            vmax=float(cfg["vmax_cap"]),
                            rasterized=True,
                        )
                        if DRAW_MAP_FEATURES:
                            ax.coastlines(resolution="50m")
                            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    else:
                        pcm = ax.pcolormesh(
                            xcoord,
                            ycoord,
                            arr.T,
                            cmap=cmap,
                            shading="auto",
                            vmax=float(cfg["vmax_cap"]),
                            rasterized=True,
                        )

                    if row == 0:
                        ax.set_title(col_titles[col - 1], fontsize=14)
                    if tag and tag != "standard":
                        ax.annotate(
                            tag.upper(),
                            xy=(1.02, 0.5),
                            xycoords="axes fraction",
                            fontsize=14,
                            va="center",
                            ha="left",
                            rotation=-90,
                            annotation_clip=False,
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_visible(False)

    cbar = fig.colorbar(
        pcm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        label=cfg["label"],
        shrink=0.6,
        pad=0.02,
        extend="max",
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cfg["label"], fontsize=14)

    fig.suptitle("Mean CRPS (averaged over all non-spatial dimensions)", fontsize=16, y=0.85)

    safe_var = var_name.replace(" ", "_").replace("+", "_").replace("/", "_")
    out_file = results_dir / f"icon_mean_crps_{safe_var}.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved: {out_file}")
    fig.clear()
    plt.close(fig)
    gc.collect()
