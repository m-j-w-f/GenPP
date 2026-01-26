"""
This script preprocesses ensemble forecast and observation datasets by:
1. Removing missing days and aligning prediction times with observations.
2. Flattening multi-dimensional data for easier processing.
3. Computing mean and standard deviation across ensemble members.
4. Saving the processed datasets to NetCDF files for efficient storage and future use.

Outputs:
- flat_obs_preproc.nc: Preprocessed observation data.
- flat_ens_preproc_agg.nc: Preprocessed ensemble data with aggregated statistics.
"""

import shutil
from pathlib import Path
from warnings import warn

import xarray as xr
from dask.distributed import Client

from genpp.data.weatherbench2 import (
    FC_VARS,
    FORECAST_ENS_FLAT_AGG_NAME,
    FORECAST_ENS_PATH,
    MISSING_DAYS,
    OBSERVATIONS_FLAT_NAME,
    OBSERVATIONS_PATH,
    OUTPUT_DIR,
)
from genpp.data.weatherbench2.utils import flatten_levels


def main(base_dir: Path = OUTPUT_DIR) -> None:
    client = Client()
    print("Dask dashboard link:", client.dashboard_link)

    ens = xr.open_zarr(FORECAST_ENS_PATH, consolidated=True)
    ens = ens.drop_sel(time=MISSING_DAYS)
    flat_ens = flatten_levels(ens)
    mean_ens = flat_ens.mean(dim="number")
    std_ens = flat_ens.std(dim="number", ddof=1)
    flat_ens_aggr = xr.concat([mean_ens, std_ens], dim="statistic").assign_coords(
        statistic=["mean", "std"]
    )
    if (ENS_OUTPUT := base_dir / FORECAST_ENS_FLAT_AGG_NAME).exists():
        warn(f"Output path {ENS_OUTPUT} exists and will be removed.")
        shutil.rmtree(ENS_OUTPUT)
    flat_ens_aggr.to_zarr(ENS_OUTPUT, consolidated=True)

    obs = xr.open_zarr(OBSERVATIONS_PATH, consolidated=True)
    obs = obs[FC_VARS]
    flat_obs = flatten_levels(obs)
    if (OBS_OUTPUT := base_dir / OBSERVATIONS_FLAT_NAME).exists():
        warn(f"Output path {OBS_OUTPUT} exists and will be removed.")
        shutil.rmtree(OBS_OUTPUT)
    flat_obs.to_zarr(OBS_OUTPUT, consolidated=True)


if __name__ == "__main__":
    main()
