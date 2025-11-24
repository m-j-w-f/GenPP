#!/usr/bin/env python3
"""Web-based forecast comparison dashboard.

This script creates an interactive Panel dashboard that compares post-processing forecasts
to the verifying analysis, individual IFS ensemble members, and the ensemble mean.

Usage:
    python plot_forecasts_web.py [--port PORT] [--host HOST]

    Or use panel serve:
    panel serve plot_forecasts_web.py --show --port 5006

Requirements:
    - panel >= 0.12.1
    - hvplot >= 0.12.1
    - bokeh
    - xarray
    - numpy
    - pandas

The dashboard will open automatically in your default web browser.
"""

from __future__ import annotations

import argparse  # noqa: F401
from collections import OrderedDict  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

import hvplot  # noqa: F401
import hvplot.xarray  # noqa: F401 -> register xarray accessor
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import panel as pn  # noqa: F401
import xarray as xr  # noqa: F401
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper  # noqa: F401
from bokeh.palettes import Inferno256, Turbo256  # noqa: F401
from bokeh.plotting import figure  # noqa: F401

from genpp import BASE_DIR  # noqa: F401
from genpp.data import FC_VARS, OBSERVATIONS_FLAT_PATH, VAL_PREDICTIONS  # noqa: F401
from genpp.eval.utils import load_predictions_dataarray  # noqa: F401

# Initialize extensions
print("Initializing extensions...")
hvplot.extension("bokeh")  # pyright: ignore[reportCallIssue]
pn.extension()
print("Extensions initialized.")

# ============================================================================
# Configuration
# ============================================================================

# Model configurations with run IDs
models: dict[str, dict[str, Any]] = {
    "emos": {"id": "k32mygar"},
    "drn": {"id": "hn0gdrqm"},
    "chen": {"id": "qbuvhf5p"},
    "fm": {"id": "blkpcik8"},
}

OUTPUT_DIR = BASE_DIR.parent.parent / "outputs"

# Plot dimensions adjusted for Germany (height > width, roughly 4:5 aspect ratio)
PLOT_WIDTH = 400
PLOT_HEIGHT = 480
COLORBAR_WIDTH = 120
COLORBAR_HEIGHT = PLOT_HEIGHT

# Color palettes for different variables
TEMPERATURE_PALETTE = Turbo256  # Diverging colormap for temperature
WIND_PALETTE = Inferno256  # Sequential colormap for wind speed

# ============================================================================
# Data Loading
# ============================================================================


def load_model_predictions():
    """Load validation predictions for all configured models."""
    print("\nLoading model predictions...")
    for model_name, model_info in models.items():
        try:
            model_dir = list(OUTPUT_DIR.rglob(f"*{model_info['id']}*"))[0].parent.parent.parent
            model_info["val_predictions"] = {
                f.name: f for f in model_dir.rglob("val_predictions*.zarr")
            }
            print(
                f"  ✓ Found {len(model_info['val_predictions'])} prediction file(s) for '{model_name}'"
            )
        except IndexError:
            print(
                f"  ✗ Warning: Could not find predictions for model '{model_name}' with ID {model_info['id']}"
            )
            model_info["val_predictions"] = {}


load_model_predictions()
print("Model predictions loaded.")

# ============================================================================
# Coordinate Setup
# ============================================================================

print("\nSetting up prediction coordinates...")
prediction_index = VAL_PREDICTIONS
init_times = prediction_index.get_level_values("time")
lead_offsets = prediction_index.get_level_values("prediction_timedelta")
lead_hours = np.asarray(lead_offsets / np.timedelta64(1, "h"), dtype=int)
valid_times = init_times + lead_offsets
prediction_labels = [
    f"{init:%Y-%m-%d %HZ} +{int(hours):02d}h" for init, hours in zip(init_times, lead_hours)
]

COMMON_PREDICTION_COORDS = {
    "init_time": ("prediction", init_times),
    "leadtime_hours": ("prediction", lead_hours),
    "valid_time": ("prediction", valid_times),
    "prediction_label": ("prediction", prediction_labels),
}
print(f"  Prepared {len(prediction_index)} prediction time steps")

# ============================================================================
# Array Preparation Utilities
# ============================================================================


def ensure_member_dim(arr: xr.DataArray) -> xr.DataArray:
    """Ensure array has a 'member' dimension."""
    if "sample" in arr.dims:
        arr = arr.rename({"sample": "member"})
    elif "number" in arr.dims:
        arr = arr.rename({"number": "member"})
    elif "member" not in arr.dims:
        arr = arr.expand_dims(member=[0])
    return arr


def prepare_array(
    arr: xr.DataArray,
    *,
    member_labels: list[str] | None = None,
    member_label_prefix: str | None = None,
) -> xr.DataArray:
    """Prepare array with consistent dimensions and coordinates."""
    arr = ensure_member_dim(arr)
    if "feature" in arr.dims:
        arr = arr.rename({"feature": "variable"})

    if member_labels is not None:
        arr = arr.assign_coords(member=("member", member_labels))
    elif member_label_prefix is not None:
        arr = arr.assign_coords(
            member=(
                "member",
                [f"{member_label_prefix}_{idx:02d}" for idx in range(arr.sizes["member"])],
            ),
        )
    elif "member" not in arr.coords or len(arr.coords["member"]) != arr.sizes["member"]:
        arr = arr.assign_coords(
            member=("member", [f"member_{idx:02d}" for idx in range(arr.sizes["member"])])
        )

    arr = arr.assign_coords(**COMMON_PREDICTION_COORDS)  # type: ignore
    return arr.transpose(
        "member",
        "prediction",
        "variable",
        "latitude",
        "longitude",
        missing_dims="ignore",
    )


# ============================================================================
# Load Forecast Sources
# ============================================================================

print("\nLoading forecast sources...")
forecast_sources: OrderedDict[str, xr.DataArray] = OrderedDict()

# Load observations
print("  Loading observations...")
obs_dataset = xr.open_dataset(OBSERVATIONS_FLAT_PATH)
obs_valid_times = init_times + lead_offsets
obs = (
    obs_dataset.sel(time=obs_valid_times)
    .to_dataarray("feature")
    .sel(feature=FC_VARS)
    .transpose("time", "feature", "longitude", "latitude")
    .rename({"time": "prediction_time"})
    .assign_coords(prediction=("prediction_time", VAL_PREDICTIONS))
    .swap_dims({"prediction_time": "prediction"})
)
forecast_sources["ground_truth"] = prepare_array(obs, member_labels=["obs"])
print("    ✓ Observations loaded")

# Load IFS ensemble mean
print("  Loading IFS ensemble mean...")
weatherbench_dir = BASE_DIR / "data" / "weatherbench2"
ens_mean = (
    xr.open_dataset(weatherbench_dir / "ens_flat_agg.zarr")
    .sel(statistic="mean")[FC_VARS]
    .stack(prediction=("time", "prediction_timedelta"))
    .sel(prediction=VAL_PREDICTIONS)
    .to_dataarray("feature")
    .transpose("prediction", "feature", "longitude", "latitude")
)
forecast_sources["ifs_mean"] = prepare_array(ens_mean, member_labels=["mean"])
print("    ✓ IFS ensemble mean loaded")

# Load IFS ensemble members
print("  Loading IFS ensemble members...")
ens_members = (
    xr.open_dataset(weatherbench_dir / "ifs_ens.zarr")[FC_VARS]
    .stack(prediction=("time", "prediction_timedelta"))
    .sel(prediction=VAL_PREDICTIONS)
    .to_dataarray("feature")
    .transpose("prediction", "number", "feature", "longitude", "latitude")
)
forecast_sources["ifs_member"] = prepare_array(ens_members, member_label_prefix="ens")
print(f"    ✓ IFS ensemble members loaded ({ens_members.sizes['number']} members)")
print("Forecast sources loaded.")

# ============================================================================
# Load Model Forecasts
# ============================================================================


def _variant_key_from_filename(file_name: str) -> str:
    """Extract variant key from prediction filename."""
    stem = Path(file_name).stem
    suffix = stem[len("val_predictions") :].strip("_") if stem.startswith("val_predictions") else ""
    return suffix or "standard"


def _variant_display_name(variant_key: str) -> str:
    """Convert variant key to display name."""
    return "Standard" if variant_key == "standard" else variant_key.upper()


model_forecasts: dict[str, OrderedDict[str, xr.DataArray]] = {}
model_variant_labels: dict[str, OrderedDict[str, str]] = {}

print("\nLoading model forecasts...")
for model_name, model_info in models.items():
    print(f"  Loading '{model_name}' variants...")
    variant_arrays: OrderedDict[str, xr.DataArray] = OrderedDict()
    variant_labels: OrderedDict[str, str] = OrderedDict()
    for file_name, val_path in sorted(model_info["val_predictions"].items()):
        variant_key = _variant_key_from_filename(file_name)
        variant_display = _variant_display_name(variant_key)
        preds = load_predictions_dataarray(val_path).sel(prediction=VAL_PREDICTIONS)
        member_prefix = f"{model_name}_{variant_key}" if variant_key != "standard" else model_name
        variant_arrays[variant_key] = prepare_array(preds, member_label_prefix=member_prefix)
        variant_labels[variant_key] = variant_display
        print(f"    ✓ Loaded variant '{variant_key}'")
    model_forecasts[model_name] = variant_arrays
    model_variant_labels[model_name] = variant_labels
print("Model forecasts loaded.")

# ============================================================================
# Widget Setup
# ============================================================================

print("\nSetting up widgets...")
prediction_options = OrderedDict(
    (label, pred) for label, pred in zip(prediction_labels, prediction_index)
)
variable_options = list(FC_VARS)
model_options = list(models.keys())


def _variant_options_for_model(model_name: str) -> OrderedDict[str, str]:
    """Get variant options for a specific model."""
    labels = model_variant_labels[model_name]
    return OrderedDict((label, key) for key, label in labels.items())


def _member_labels_for_model(model_name: str, variant_key: str) -> list[str]:
    """Get member labels for a specific model variant."""
    da = model_forecasts[model_name][variant_key]
    return list(map(str, da.coords["member"].values))


# Create main control widgets
variable_select = pn.widgets.Select(name="Variable", options=variable_options, width=250)
prediction_select = pn.widgets.Select(name="Init + Lead", options=prediction_options, width=250)

# Create IFS member slider
ensemble_member_slider = pn.widgets.DiscreteSlider(
    name="IFS member",
    options=list(map(str, forecast_sources["ifs_member"].coords["member"].values)),
    width=250,
)
ensemble_member_slider.value = ensemble_member_slider.options[0]  # type: ignore

# Create model-specific widgets
model_variant_widgets: dict[str, pn.widgets.Select] = {}
model_member_sliders: dict[str, pn.widgets.DiscreteSlider] = {}

for model_name in model_options:
    variant_options = _variant_options_for_model(model_name)
    variant_select = pn.widgets.Select(
        name=f"{model_name.upper()} variant",
        options=variant_options,
        width=200,
    )
    variant_select.value = next(iter(variant_options.values()))

    member_labels = _member_labels_for_model(model_name, variant_select.value)
    member_slider = pn.widgets.DiscreteSlider(
        name=f"{model_name.upper()} member",
        options=member_labels,
        width=200,
    )
    member_slider.value = member_labels[0]

    def _update_member_slider(event, *, _model_name=model_name, slider=member_slider):
        new_labels = _member_labels_for_model(_model_name, event.new)
        next_value = slider.value if slider.value in new_labels else new_labels[0]
        slider.param.update(options=new_labels, value=next_value)

    variant_select.param.watch(_update_member_slider, "value")
    model_variant_widgets[model_name] = variant_select
    model_member_sliders[model_name] = member_slider
print("Widgets configured.")

# ============================================================================
# Color Palette Selection
# ============================================================================


def _get_color_palette(variable_name: str) -> list:
    """Get appropriate color palette for the variable."""
    if "temperature" in variable_name.lower():
        return TEMPERATURE_PALETTE  # type: ignore
    else:
        return WIND_PALETTE  # type: ignore


# ============================================================================
# Data Selection and Color Limit Computation
# ============================================================================


def _selection_arrays(
    variable_name: str,
    prediction_value: Any,
    ensemble_member: str,
    variant_values: list[str],
    member_values: list[str],
) -> list[xr.DataArray]:
    """Get all selected data arrays for current widget values."""
    arrays: list[xr.DataArray] = [
        forecast_sources["ground_truth"].sel(
            member="obs", variable=variable_name, prediction=prediction_value
        ),
        forecast_sources["ifs_mean"].sel(
            member="mean", variable=variable_name, prediction=prediction_value
        ),
        forecast_sources["ifs_member"].sel(
            member=ensemble_member, variable=variable_name, prediction=prediction_value
        ),
    ]
    for model_name, variant_key, member_label in zip(model_options, variant_values, member_values):
        arrays.append(
            model_forecasts[model_name][variant_key].sel(
                member=member_label,
                variable=variable_name,
                prediction=prediction_value,
            )
        )
    return arrays


def _compute_color_limits(
    variable_name: str,
    prediction_value: Any,
    ensemble_member: str,
    variant_values: list[str],
    member_values: list[str],
) -> tuple[float, float]:
    """Compute shared color limits across all displayed forecasts."""
    arrays = _selection_arrays(
        variable_name, prediction_value, ensemble_member, variant_values, member_values
    )
    mins: list[float] = []
    maxs: list[float] = []
    for arr in arrays:
        data = arr.values
        with np.errstate(all="ignore"):
            arr_min = float(np.nanmin(data))
            arr_max = float(np.nanmax(data))
        if np.isfinite(arr_min):
            mins.append(arr_min)
        if np.isfinite(arr_max):
            maxs.append(arr_max)
    if not mins or not maxs:
        return (0.0, 1.0)
    low = min(mins)
    high = max(maxs)

    # For temperature, center the colormap symmetrically around midpoint
    if "temperature" in variable_name.lower():
        midpoint = (low + high) / 2
        max_range = max(abs(low - midpoint), abs(high - midpoint))
        low = midpoint - max_range
        high = midpoint + max_range

    if np.isclose(low, high):
        high = float(np.nextafter(high, high + 1.0))
    return (float(low), float(high))


def _current_variant_values() -> list[str]:
    """Get current variant values for all models."""
    return [model_variant_widgets[name].value for name in model_options]  # type: ignore


def _current_member_values() -> list[str]:
    """Get current member values for all models."""
    return [model_member_sliders[name].value for name in model_options]  # type: ignore


def _current_color_limits() -> tuple[float, float]:
    """Get current color limits based on all widget values."""
    return _compute_color_limits(
        variable_select.value,  # type: ignore
        prediction_select.value,
        ensemble_member_slider.value,  # type: ignore
        _current_variant_values(),
        _current_member_values(),
    )


# ============================================================================
# Rendering Functions
# ============================================================================


def _format_title(label: str, da: xr.DataArray) -> str:
    """Format plot title with forecast metadata."""
    init_ts = pd.Timestamp(da.coords["init_time"].item())
    lead = int(da.coords["leadtime_hours"].item())
    valid_ts = pd.Timestamp(da.coords["valid_time"].item())
    return f"{label}\ninit {init_ts:%Y-%m-%d %HZ} +{lead:02d}h (valid {valid_ts:%Y-%m-%d %HZ})"


def _render_image(
    label: str,
    da: xr.DataArray,
    *,
    color_limits: tuple[float, float],
    color_palette: list,
):
    """Render a single forecast image."""
    return da.hvplot.image(
        x="longitude",
        y="latitude",
        cmap=color_palette,
        clim=color_limits,
        colorbar=False,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title=_format_title(label, da),
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
    ).opts(framewise=True, tools=["hover", "pan", "wheel_zoom", "reset"])


def _variant_display(model_name: str, variant_key: str) -> str:
    """Get display name for model variant."""
    return model_variant_labels[model_name][variant_key]


# Define color dependencies for reactive updates
COLOR_DEPENDENCIES = [
    variable_select,
    prediction_select,
    ensemble_member_slider,
    *[model_variant_widgets[name] for name in model_options],
    *[model_member_sliders[name] for name in model_options],
]

# ============================================================================
# Panel Creation Functions
# ============================================================================


def _make_postproc_panel(model_name: str):
    """Create a panel for a post-processing model."""
    variant_widget = model_variant_widgets[model_name]
    member_slider = model_member_sliders[model_name]

    @pn.depends(*COLOR_DEPENDENCIES)  # type: ignore
    def _panel(*_):
        variant_key = variant_widget.value
        member_label = member_slider.value
        selection = model_forecasts[model_name][variant_key].sel(  # type: ignore
            member=member_label,
            variable=variable_select.value,
            prediction=prediction_select.value,
        )
        label = (
            f"{model_name.upper()} {_variant_display(model_name, variant_key)}\n({member_label})"  # type: ignore
        )
        palette = _get_color_palette(variable_select.value)  # type: ignore
        return _render_image(
            label, selection, color_limits=_current_color_limits(), color_palette=palette
        )

    return _panel


postproc_panels = {model_name: _make_postproc_panel(model_name) for model_name in model_options}


@pn.depends(*COLOR_DEPENDENCIES)  # type: ignore
def ground_truth_panel(*_):
    """Create ground truth observation panel."""
    selection = forecast_sources["ground_truth"].sel(
        member="obs",
        variable=variable_select.value,
        prediction=prediction_select.value,
    )
    palette = _get_color_palette(variable_select.value)  # type: ignore
    return _render_image(
        "Ground Truth", selection, color_limits=_current_color_limits(), color_palette=palette
    )


@pn.depends(*COLOR_DEPENDENCIES)  # type: ignore
def ensemble_member_panel(*_):
    """Create IFS ensemble member panel."""
    selection = forecast_sources["ifs_member"].sel(
        member=ensemble_member_slider.value,
        variable=variable_select.value,
        prediction=prediction_select.value,
    )
    label = f"IFS Member\n{ensemble_member_slider.value}"
    palette = _get_color_palette(variable_select.value)  # type: ignore
    return _render_image(
        label, selection, color_limits=_current_color_limits(), color_palette=palette
    )


@pn.depends(*COLOR_DEPENDENCIES)  # type: ignore
def ensemble_mean_panel(*_):
    """Create IFS ensemble mean panel."""
    selection = forecast_sources["ifs_mean"].sel(
        member="mean",
        variable=variable_select.value,
        prediction=prediction_select.value,
    )
    palette = _get_color_palette(variable_select.value)  # type: ignore
    return _render_image(
        "IFS Mean", selection, color_limits=_current_color_limits(), color_palette=palette
    )


@pn.depends(*COLOR_DEPENDENCIES)  # type: ignore
def shared_colorbar(*_):
    """Create shared colorbar for all plots."""
    low, high = _current_color_limits()
    palette = _get_color_palette(variable_select.value)  # type: ignore
    mapper = LinearColorMapper(palette=palette, low=low, high=high)
    colorbar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        title=variable_select.value,
        title_text_font_size="11pt",
        major_label_text_font_size="10pt",
    )
    fig = figure(height=COLORBAR_HEIGHT, width=COLORBAR_WIDTH, toolbar_location=None, min_border=10)
    fig.add_layout(colorbar, "right")
    fig.outline_line_color = None
    fig.axis.visible = False
    fig.grid.visible = False
    return fig


# ============================================================================
# Dashboard Layout
# ============================================================================

print("\nBuilding dashboard layout...")
# Create model columns
postproc_columns = [
    pn.Column(
        pn.pane.Markdown(f"### {model_name.upper()}", align="center"),
        model_variant_widgets[model_name],
        model_member_sliders[model_name],
        postproc_panels[model_name],
        sizing_mode="fixed",
    )
    for model_name in model_options
]

# Create reference row (ground truth, IFS member, IFS mean, colorbar)
# Add spacers to align plots with the IFS member slider
reference_row = pn.Row(
    pn.Column(
        pn.pane.Markdown("### Ground Truth", align="center"),
        pn.Spacer(height=45),  # Spacer to align with slider
        ground_truth_panel,
        sizing_mode="fixed",
    ),
    pn.Column(
        pn.pane.Markdown("### IFS Member", align="center"),
        ensemble_member_slider,
        ensemble_member_panel,
        sizing_mode="fixed",
    ),
    pn.Column(
        pn.pane.Markdown("### IFS Mean", align="center"),
        pn.Spacer(height=45),  # Spacer to align with slider
        ensemble_mean_panel,
        sizing_mode="fixed",
    ),
    pn.Column(
        pn.pane.Markdown("### Colorbar", align="center"),
        pn.Spacer(height=45),  # Spacer to align with slider
        shared_colorbar,
        sizing_mode="fixed",
    ),
    sizing_mode="stretch_width",
)

# Main dashboard
comparison_dashboard = pn.Column(
    pn.pane.Markdown(
        "# Forecast Comparison Dashboard\n\n"
        "**Controls:** Select the variable, initialization time, model variants, "
        "and ensemble members to compare forecast maps. "
        "All plots share the same color scale for direct comparison.",
        sizing_mode="stretch_width",
    ),
    pn.layout.Divider(),
    pn.Row(
        pn.Column(
            pn.pane.Markdown("### Global Controls", align="center"),
            variable_select,
            prediction_select,
            sizing_mode="fixed",
        ),
    ),
    pn.layout.Divider(),
    pn.pane.Markdown("### Post-Processing Models", sizing_mode="stretch_width"),
    pn.Row(*postproc_columns, sizing_mode="stretch_width"),
    pn.layout.Divider(),
    pn.pane.Markdown("### Reference Forecasts", sizing_mode="stretch_width"),
    reference_row,
    sizing_mode="stretch_width",
)

# ============================================================================
# Servable Application
# ============================================================================

# Make the dashboard servable
comparison_dashboard.servable(title="Forecast Comparison Dashboard")
print("Dashboard ready!")
print("=" * 60)


# ============================================================================
# Command-line Interface
# ============================================================================


def main():
    """Run the dashboard with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Web-based forecast comparison dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve the dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to serve the dashboard on",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't automatically open browser",
    )

    args = parser.parse_args()

    # Serve the dashboard
    print(f"Starting forecast comparison dashboard on {args.host}:{args.port}")
    print(f"Open your browser to: http://{args.host}:{args.port}/")

    comparison_dashboard.show(port=args.port, threaded=False, open=not args.no_show)


if __name__ == "__main__":
    main()
