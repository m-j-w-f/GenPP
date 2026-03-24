import gc
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

# Use a non-interactive backend for script execution to avoid GUI-related native crashes.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from genpp.data.icon import DATA_DIR
from genpp.eval.icon import best_models
from genpp.plots import RESULTS_DIR


def open_prediction_dataarray(pred_path: Path) -> xr.DataArray:
    try:
        da = xr.open_dataarray(pred_path)
    except ValueError:
        ds = xr.open_dataset(pred_path)
        if "prediction" in ds.data_vars:
            da = ds["prediction"]
        else:
            da = ds[list(ds.data_vars)[0]]
    return da


def get_prediction_dim(da: xr.DataArray) -> str:
    if "prediction" in da.dims:
        return "prediction"
    if "time" in da.dims:
        return "time"
    raise ValueError(f"Could not find prediction dimension in {da.dims}")


def get_variable_dim(da: xr.DataArray) -> str:
    for dim in ("variable", "feature"):
        if dim in da.dims:
            return dim
    raise ValueError(f"Could not find variable dimension in {da.dims}")


def pick_prediction_file(model_dir: Path) -> Path | None:
    preferred = [
        "test_predictions_ecc.nc",
        "test_predictions.nc",
    ]
    for name in preferred:
        exact = model_dir / name
        if exact.exists():
            return exact

    files = sorted(model_dir.rglob("test_predictions*.nc"))
    if files:
        return files[0]
    return None


# Build prediction path index without loading data
model_prediction_paths: dict[str, dict[str, Path]] = {}
for model_name, model_entries in best_models:
    if not model_entries:
        continue
    model_prediction_paths[model_name] = {}
    for entry in model_entries:
        variant_key = entry.tag or "standard"
        pred_path = pick_prediction_file(entry.model_dir)
        if pred_path is None:
            print(f"  x {model_name}/{variant_key}: no test_predictions*.nc found")
            continue
        model_prediction_paths[model_name][variant_key] = pred_path
        print(f"  check {model_name}/{variant_key}: {pred_path.name}")

print("\nAvailable models / variants:")
for mname, variants in model_prediction_paths.items():
    print(f"  {mname}: {list(variants.keys())}")

# Use first available path as reference to define prediction index and coordinates
reference_path = None
for variants in model_prediction_paths.values():
    if variants:
        reference_path = next(iter(variants.values()))
        break
if reference_path is None:
    raise RuntimeError("No ICON prediction files discovered")

ref_da = open_prediction_dataarray(reference_path)
pred_dim = get_prediction_dim(ref_da)
var_dim = get_variable_dim(ref_da)

N_PRED = ref_da.sizes[pred_dim]
SELECTED_PRED_IDX = 24
if SELECTED_PRED_IDX >= N_PRED:
    print(f"Selected index {SELECTED_PRED_IDX} out of range for N={N_PRED}; using 0")
    SELECTED_PRED_IDX = 0

ref_sel = ref_da.isel({pred_dim: SELECTED_PRED_IDX})
if "time" in ref_sel.coords:
    sel_valid = pd.Timestamp(ref_sel["time"].item())
else:
    sel_valid = pd.Timestamp(ref_da[pred_dim].values[SELECTED_PRED_IDX])

sel_init = (
    pd.Timestamp(ref_sel.coords["init_date"].item()) if "init_date" in ref_sel.coords else None
)
sel_lead = ref_sel.coords["leadtime"].item() if "leadtime" in ref_sel.coords else None
sel_lead_h = int(sel_lead / np.timedelta64(1, "h")) if sel_lead is not None else None

fc_vars = [str(v) for v in ref_da[var_dim].values]
rlon_name = "rlon" if "rlon" in ref_sel.coords else "x"
rlat_name = "rlat" if "rlat" in ref_sel.coords else "y"

# Load only the matching observation tensor for this valid time
valid_tag = sel_valid.strftime("%Y%m%d%H")
rea_tensor_path = DATA_DIR / "tensors" / "rea" / f"rea_{valid_tag}.pt"
if not rea_tensor_path.exists():
    raise FileNotFoundError(f"Observation tensor not found: {rea_tensor_path}")
obs_tensor = torch.load(rea_tensor_path, weights_only=True).cpu().numpy()

print(f"\nSelected prediction idx: {SELECTED_PRED_IDX}/{N_PRED - 1}")
if sel_init is not None and sel_lead_h is not None:
    print(f"Init {sel_init:%Y-%m-%d %HZ} +{sel_lead_h}h -> valid {sel_valid:%Y-%m-%d %HZ}")
else:
    print(f"Valid time: {sel_valid:%Y-%m-%d %HZ}")
print(f"Loaded observation tensor: {obs_tensor.shape} from {rea_tensor_path.name}")

# Cell-15-style layout for ICON: observation + fixed model columns
MODEL_COLUMNS = [
    ("emos", "EMOS", None),
    ("drn", "DRN", None),
    ("chen", "LNGM (IND)", "ind_"),
    ("engression", "ENGRESSION (IND)", "ind_"),
    ("fm", "FM", None),
]

VAR_SETTINGS = {
    "T_2M+height_2.0": {"cmap": "magma", "label": "2m temperature (K)"},
    "VMAX_10M+height_2_10.0": {"cmap": "viridis", "label": "10m wind speed (m/s)"},
}


def _var_label(var_name: str) -> dict[str, str]:
    if var_name in VAR_SETTINGS:
        return VAR_SETTINGS[var_name]
    if "T_2M" in var_name:
        return {"cmap": "magma", "label": "2m temperature"}
    if "VMAX_10M" in var_name or "wind" in var_name.lower():
        return {"cmap": "viridis", "label": "10m wind speed"}
    return {"cmap": "cividis", "label": var_name}


def _pick_var_in_da(da: xr.DataArray, target_var: str, var_dim_name: str) -> xr.DataArray:
    available = [str(v) for v in da[var_dim_name].values]
    if target_var in available:
        return da.sel({var_dim_name: target_var})

    target_key = target_var.split("+")[0]
    for candidate in available:
        if candidate.split("+")[0] == target_key:
            return da.sel({var_dim_name: candidate})

    raise KeyError(f"Variable {target_var} not found in {available}")


def load_model_field(
    path: Path, pred_idx: int, var_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use a context manager so NetCDF file handles are released every call.
    with xr.open_dataset(path) as ds:
        if "prediction" in ds.data_vars:
            da = ds["prediction"]
        else:
            da = ds[list(ds.data_vars)[0]]

        p_dim = get_prediction_dim(da)
        v_dim = get_variable_dim(da)

        sel = da.isel({p_dim: pred_idx})
        for sample_dim in ("sample", "number", "member"):
            if sample_dim in sel.dims:
                sel = sel.isel({sample_dim: 0})
                break

        sel = _pick_var_in_da(sel, var_name, v_dim).load()

    x_name = "rlon" if "rlon" in sel.coords else ("x" if "x" in sel.coords else sel.dims[0])
    y_name = "rlat" if "rlat" in sel.coords else ("y" if "y" in sel.coords else sel.dims[1])

    x_coord = np.asarray(sel[x_name].values)
    y_coord = np.asarray(sel[y_name].values)
    return np.asarray(sel.values), x_coord, y_coord


def _obs_channel_index(var_name: str, var_idx: int) -> int:
    if var_idx < obs_tensor.shape[0]:
        return var_idx

    if "T_2M" in var_name or "temperature" in var_name.lower():
        return 0
    if "VMAX_10M" in var_name or "wind" in var_name.lower():
        return 1
    raise IndexError(f"Could not map observation channel for variable {var_name}")


use_rotated_coords = True  # rlon_name == "rlon" and rlat_name == "rlat"
rotated_crs = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=40.0)
plot_proj = rotated_crs if use_rotated_coords else ccrs.PlateCarree()
DRAW_MAP_FEATURES = True

results_dir = RESULTS_DIR / "results" / "icon" / "samples"
results_dir.mkdir(parents=True, exist_ok=True)

# Freeze coordinate arrays once to avoid repeated xarray access in the plotting loop.
ref_lon = np.asarray(ref_sel[rlon_name].values) if rlon_name in ref_sel.coords else None
ref_lat = np.asarray(ref_sel[rlat_name].values) if rlat_name in ref_sel.coords else None

# Close lazy xarray handle from the reference file once metadata is extracted.
try:
    ref_da.close()
except Exception:
    pass

for var_idx, var_name in enumerate(fc_vars):
    cfg = _var_label(var_name)

    obs_idx = _obs_channel_index(var_name, var_idx)
    obs_v = obs_tensor[obs_idx]
    obs_lon = ref_lon if ref_lon is not None else np.arange(obs_v.shape[0])
    obs_lat = ref_lat if ref_lat is not None else np.arange(obs_v.shape[1])

    col_variants: list[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = []
    col_titles: list[str] = []

    for mkey, display, prefix in MODEL_COLUMNS:
        entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        variants = model_prediction_paths.get(mkey, {})

        for vname, pred_path in variants.items():
            if prefix is not None and not vname.startswith(prefix):
                continue
            tag = vname[len(prefix) :] if prefix and vname.startswith(prefix) else vname
            try:
                field, xcoord, ycoord = load_model_field(pred_path, SELECTED_PRED_IDX, var_name)
            except Exception as exc:
                print(f"  x {mkey}/{vname}/{var_name}: {exc}")
                continue
            entries.append((tag, field, xcoord, ycoord))

        col_variants.append(entries)
        col_titles.append(display)

    n_rows = max((len(v) for v in col_variants), default=1) or 1
    n_cols = 1 + len(MODEL_COLUMNS)

    all_vals = [obs_v.ravel()]
    for entries in col_variants:
        for _, arr, _, _ in entries:
            all_vals.append(arr.ravel())
    joined = np.concatenate(all_vals)
    vmin, vmax = float(np.nanmin(joined)), float(np.nanmax(joined))

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
                if row == 0:
                    if use_rotated_coords:
                        pcm = ax.pcolormesh(
                            obs_lon,
                            obs_lat,
                            obs_v.T,
                            cmap=cfg["cmap"],
                            vmin=vmin,
                            vmax=vmax,
                            transform=rotated_crs,
                            shading="auto",
                            rasterized=True,
                        )
                        if DRAW_MAP_FEATURES:
                            ax.coastlines(resolution="50m")
                            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    else:
                        pcm = ax.pcolormesh(
                            obs_lon,
                            obs_lat,
                            obs_v.T,
                            cmap=cfg["cmap"],
                            vmin=vmin,
                            vmax=vmax,
                            shading="auto",
                            rasterized=True,
                        )
                    ax.set_title("Observation", fontsize=14)
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
                            cmap=cfg["cmap"],
                            vmin=vmin,
                            vmax=vmax,
                            transform=rotated_crs,
                            shading="auto",
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
                            cmap=cfg["cmap"],
                            vmin=vmin,
                            vmax=vmax,
                            shading="auto",
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
        pcm,  # type: ignore
        ax=axes.ravel().tolist(),
        orientation="vertical",
        label=cfg["label"],
        shrink=0.6,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cfg["label"], fontsize=14)

    title_time = (
        f"init {sel_init:%Y-%m-%d %HZ} +{sel_lead_h}h"
        if (sel_init is not None and sel_lead_h is not None)
        else f"valid {sel_valid:%Y-%m-%d %HZ}"
    )
    fig.suptitle(title_time, fontsize=16, y=0.85)

    safe_var = var_name.replace(" ", "_").replace("+", "_").replace("/", "_")
    lead_suffix = sel_lead_h if sel_lead_h is not None else "na"
    out_file = results_dir / f"icon_all_models_{safe_var}_leadtime_{lead_suffix}.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved: {out_file}")
    fig.clear()
    plt.close(fig)
    gc.collect()
