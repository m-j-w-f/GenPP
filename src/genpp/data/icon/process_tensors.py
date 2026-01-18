#!/usr/bin/env python
"""
Standalone script for processing ICON forecast and reanalysis data into tensors.

This script extracts the _get_fc_tensors and _get_rea_tensors functions from
dataset.py to run them as standalone jobs, enabling parallel processing via
NQSV job submissions.

Usage:
    For forecast tensors:
        JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py

    For reanalysis tensors:
        JOB_TYPE=rea YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py

Environment Variables:
    JOB_TYPE: Either 'fc' or 'rea' to specify which tensor type to process
    YEAR: Year in format YYYY (e.g., '2021')
    MONTH: Month in format MM (e.g., '01')
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

# Import constants from the package
from genpp.data import MetadataVars
from genpp.data.icon import (
    AXIS_ORDER,
    DATA_DIR,
    LEVELS_TO_FLATTEN,
    VARS_GRID_28,
    VARS_TO_DROP,
)
from genpp.data.utils import flatten_levels

# Variable selection as specified in the comment
X_SELECT_VARIABLES = [
    "ALB_RAD",
    "ASOB_S",
    "ASOB_T",
    "ATHB_S",
    "ATHB_T",
    "CLCH+plev_0.0",
    "CLCL+plev_2_80000.0",
    "CLCM+plev_3_40000.0",
    "CLCT",
    "HBAS_CON",
    "HTOP_CON",
    "PMSL",
    "RAIN_CON",
    "RAIN_GSP",
    "SNOW_CON",
    "SNOW_GSP",
    "SOBS_RAD",
    "TD_2M+height_2.0",
    "THBS_RAD",
    "TMAX_2M+height_2.0",
    "TMIN_2M+height_2.0",
    "TOT_PREC",
    "TQC",
    "TQI",
    "TQV",
    "T_2M+height_2.0",
    "T_G",
    "U_10M+height_2_10.0",
    "VMAX_10M+height_2_10.0",
    "V_10M+height_2_10.0",
    "W_SNOW",
    "W_SO+depth_0.0",
    "W_SO+depth_0.01",
    "W_SO+depth_0.03",
    "W_SO+depth_0.09",
    "W_SO+depth_0.27",
    "W_SO+depth_0.81",
    "W_SO+depth_2.43",
    "W_SO+depth_7.29",
    "Z0",
]

Y_SELECT_VARIABLES = ["T_2M+height_2.0", "VMAX_10M+height_2_10.0"]

# Compute auxiliary variables (x variables without y variables)
X_SELECT_VARIABLES_WO_Y = [var for var in X_SELECT_VARIABLES if var not in Y_SELECT_VARIABLES]

# Tensor output directories
FC_TENSOR_DIR = DATA_DIR / "tensors" / "fc"
META_TENSOR_DIR = DATA_DIR / "tensors" / "meta"
REA_TENSOR_DIR = DATA_DIR / "tensors" / "rea"


def _add_sincos_doy(da: xr.DataArray) -> xr.DataArray:
    """Add sinusoidal day-of-year features."""
    doy = da.time.dt.dayofyear
    sin_time = np.sin(doy * 2 * np.pi / 365).astype(np.float32)
    cos_time = np.cos(doy * 2 * np.pi / 365).astype(np.float32)
    transformed_time = xr.concat([sin_time, cos_time], dim="feature", coords="minimal")
    transformed_time["feature"] = [
        MetadataVars.SIN_PREDICTION_TIME.value,
        MetadataVars.COS_PREDICTION_TIME.value,
    ]
    transformed_time = transformed_time.expand_dims(
        {
            "x": da.x,
            "y": da.y,
        }
    )
    return transformed_time


def _add_xy(da: xr.DataArray) -> xr.DataArray:
    """Add normalized x/y coordinate features."""
    # normalize x per-axis (min-max) and expand to 2D feature map
    x = da.x
    x_mean = float(x.mean())
    x_std = float(x.std())
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    x_grid = x_norm.expand_dims({"y": da.y, "feature": [MetadataVars.LONGITUDE.value]})
    x_grid = x_grid.transpose("feature", "x", "y")

    # normalize y per-axis (min-max) and expand to 2D feature map
    y = da.y
    y_mean = float(y.mean())
    y_std = float(y.std())
    y_norm = ((y - y_mean) / y_std).astype(np.float32)
    y_grid = y_norm.expand_dims({"x": da.x, "feature": [MetadataVars.LATITUDE.value]})
    y_grid = y_grid.transpose("feature", "x", "y")

    return xr.concat([x_grid, y_grid], dim="feature", coords="minimal")


def get_metadata_features(da: xr.DataArray) -> xr.DataArray:
    """Get metadata features including day-of-year and coordinates."""
    sincos_doy = _add_sincos_doy(da)
    xy_grid = _add_xy(da)
    return xr.concat([sincos_doy, xy_grid], dim="feature", coords="minimal").transpose(*AXIS_ORDER)


def _get_fc_tensors(ens_nc_paths: list[Path]) -> None:
    """Build and store forecast tensors from ensemble NetCDF paths.

    Args:
        ens_nc_paths (list[Path]): Paths to ensmean NetCDF files to process.

    Returns:
        None: Writes forecast and metadata tensors to disk.
    """
    # Ensure output directories exist
    FC_TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    META_TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    # Skip entries with already materialized tensors
    filtered_paths: list[Path] = []
    for ens_path in ens_nc_paths:
        time_leadtime = "_".join(ens_path.stem.split("_")[1:])
        fc_path = FC_TENSOR_DIR / f"fc_{time_leadtime}.pt"
        meta_path = META_TENSOR_DIR / f"meta_{time_leadtime}.pt"
        if fc_path.exists() and meta_path.exists():
            continue
        filtered_paths.append(ens_path)
    ens_nc_paths = filtered_paths

    # Build matching ensstd paths for remaining inputs
    std_nc_paths = [Path(str(p).replace("ensmean", "ensstd")) for p in ens_nc_paths]
    # Process mean/std pairs together
    for paths in tqdm(
        zip(ens_nc_paths, std_nc_paths), desc="Generating FC Tensors", total=len(ens_nc_paths)
    ):
        datasets = []
        time_leadtime = "_".join(paths[0].stem.split("_")[1:])
        for path in paths:
            ds = xr.open_dataset(path).drop_vars(VARS_TO_DROP)
            for level in LEVELS_TO_FLATTEN:
                try:
                    ds = flatten_levels(ds, level)
                except KeyError:
                    pass
            da = ds[VARS_GRID_28].to_dataarray("feature").squeeze().transpose(*AXIS_ORDER)
            datasets.append(da)
        da_stacked = xr.concat(datasets, dim="aggregation")
        da_stacked.coords["aggregation"] = ["mean", "std"]

        # Select the predicted vars (y_select_vars) and use only aggr dim mean
        predicted_vars = da_stacked.sel(aggregation="mean", feature=Y_SELECT_VARIABLES)
        predicted_vars = torch.from_numpy(predicted_vars.values)

        # Select the auxiliary vars (all vars in x_select_vars) and kick out the
        aux_vars_mean = da_stacked.sel(aggregation="mean", feature=X_SELECT_VARIABLES_WO_Y)
        aux_vars_std = da_stacked.sel(aggregation="std", feature=X_SELECT_VARIABLES)
        aux_vars = xr.concat([aux_vars_mean, aux_vars_std], dim="feature", coords="different")
        aux_vars = torch.from_numpy(aux_vars.values)

        fc_tensors = {
            "predicted_vars": predicted_vars,  # [c0,x,y]
            "auxiliary_vars": aux_vars,  # [c1,x,y]
        }
        fc_path = FC_TENSOR_DIR / f"fc_{time_leadtime}.pt"
        torch.save(fc_tensors, fc_path)

        meta = get_metadata_features(da_stacked)
        meta_path = META_TENSOR_DIR / f"meta_{time_leadtime}.pt"
        meta_tensor = torch.from_numpy(meta.values)
        # Meta tensors have shape [c, x, y]
        torch.save(meta_tensor, meta_path)


def _get_rea_tensors(rea_nc_paths: list[Path]) -> None:
    """Build and store reanalysis tensors from NetCDF paths, skipping existing outputs.

    Args:
        rea_nc_paths (list[Path]): Paths to reanalysis NetCDF files to process.

    Returns:
        None: Writes reanalysis tensors to disk.
    """
    # Ensure output directory exists
    REA_TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    # Skip entries with already materialized tensors
    filtered_paths: list[Path] = []
    for rea_path in rea_nc_paths:
        date = rea_path.stem.split("_")[-1]
        tens_path = REA_TENSOR_DIR / f"rea_{date}.pt"
        if tens_path.exists():
            continue
        filtered_paths.append(rea_path)
    rea_nc_paths = filtered_paths

    for rea_path in tqdm(rea_nc_paths, desc="Generating REA Tensors"):
        date = rea_path.stem.split("_")[-1]
        rea = xr.open_dataset(rea_path)
        rea = rea.drop_vars("rotated_pole")
        for dim in ["height", "height_2"]:
            try:
                rea = flatten_levels(rea, level_dim=dim)
            except KeyError:
                # Some files may not have all dimensions (e.g., early rea files missing height_2)
                pass
        rea = (
            rea.to_dataarray("feature").sel(feature=Y_SELECT_VARIABLES).transpose(..., *AXIS_ORDER)
        )
        tens_path = REA_TENSOR_DIR / f"rea_{date}.pt"
        tens = torch.from_numpy(rea.values).squeeze()
        # Rea has shape [c, x, y]
        torch.save(tens, tens_path)


def filter_paths_by_month(paths: list[Path], year: str, month: str) -> list[Path]:
    """Filter file paths to only include those from a specific year-month.

    Args:
        paths: List of file paths where filename contains date like YYYYMMDDHH
        year: String in format 'YYYY'
        month: String in format 'MM'

    Returns:
        Filtered list of paths
    """
    prefix = f"{year}{month}"
    filtered = []
    for path in paths:
        # Extract date from filename (format varies by file type)
        stem = path.stem
        # For ensmean/ensstd files: ens_YYYYMMDDHH_XXX.nc
        # For rea files: rea_YYYYMMDDHH.nc
        parts = stem.split("_")
        for part in parts:
            if len(part) >= 6 and part[:6] == prefix:
                filtered.append(path)
                break
    return filtered


def main():
    """Main entry point for the script."""
    # Get job type, year and month from environment variables
    job_type = os.environ.get("JOB_TYPE", "").lower()
    year = os.environ.get("YEAR", "")
    month = os.environ.get("MONTH", "")

    if job_type not in ["fc", "rea"]:
        print(f"Error: JOB_TYPE must be 'fc' or 'rea', got '{job_type}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    if not year or len(year) != 4 or not year.isdigit():
        print(f"Error: YEAR must be in format YYYY, got '{year}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    if not month or len(month) != 2 or not month.isdigit():
        print(f"Error: MONTH must be in format MM, got '{month}'")
        print("Usage: JOB_TYPE=fc YEAR=2021 MONTH=01 pixi run -e nb python process_tensors.py")
        sys.exit(1)

    print(f"Processing {job_type.upper()} tensors for {year}-{month}")

    if job_type == "fc":
        # Process forecast tensors
        ens_nc_paths = sorted(list((DATA_DIR / "ensmean").glob("*.nc")))
        if not ens_nc_paths:
            print(f"Error: No ensmean files found in {DATA_DIR / 'ensmean'}")
            sys.exit(1)

        # Filter to only this month
        ens_nc_paths = filter_paths_by_month(ens_nc_paths, year, month)
        print(f"Found {len(ens_nc_paths)} ensmean files for {year}-{month}")

        if ens_nc_paths:
            _get_fc_tensors(ens_nc_paths)
            print(f"Completed processing {len(ens_nc_paths)} FC tensors for {year}-{month}")
        else:
            print(f"No files found for {year}-{month}")

    elif job_type == "rea":
        # Process reanalysis tensors
        rea_nc_paths = sorted(list((DATA_DIR / "rea").glob("*.nc")))
        if not rea_nc_paths:
            print(f"Error: No rea files found in {DATA_DIR / 'rea'}")
            sys.exit(1)

        # Filter to only this month
        rea_nc_paths = filter_paths_by_month(rea_nc_paths, year, month)
        print(f"Found {len(rea_nc_paths)} rea files for {year}-{month}")

        if rea_nc_paths:
            _get_rea_tensors(rea_nc_paths)
            print(f"Completed processing {len(rea_nc_paths)} REA tensors for {year}-{month}")
        else:
            print(f"No files found for {year}-{month}")


if __name__ == "__main__":
    main()
