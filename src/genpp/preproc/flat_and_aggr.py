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
# TODO fatten the aggregations and append _mean and _std to the variable names, this way we can add Metadata to the variables and still use the same code for the MapDataset
# We only need to keep track of the length of the variables to slice them correctly later

from pathlib import Path

import xarray as xr
from dask.distributed import Client

from genpp.data import (
    FC_VARS,
    FORECAST_ENS_FLAT_AGG_NAME,
    FORECAST_ENS_NAME,
    MISSING_DAYS,
    OBSERVATIONS_FLAT_NAME,
    OBSERVATIONS_NAME,
    OUTPUT_DIR,
)
from genpp.data.utils import flatten_levels, get_time_intersection


def main(base_dir: Path = OUTPUT_DIR) -> None:
    client = Client()
    print("Dask dashboard link:", client.dashboard_link)

    ens = xr.open_dataset(
        base_dir / FORECAST_ENS_NAME,
        chunks={
            "time": "auto",
            "number": -1,
            "latitude": -1,
            "longitude": -1,
            "level": -1,
        },
    )
    obs = xr.open_dataset(
        base_dir / OBSERVATIONS_NAME,
        chunks="auto",
    )
    # Cut out the missing days first, since they are in time, not prediction_time
    ens = ens.sel(time=~ens.time.isin(MISSING_DAYS))
    ens = ens.assign_coords(prediction_time=ens.time + ens.prediction_timedelta).swap_dims(
        {"time": "prediction_time"}
    )

    times = get_time_intersection(ens, obs)

    ens = ens.sel(prediction_time=times)
    obs = obs.sel(time=times)
    flat_ens = flatten_levels(ens)
    flat_ens = flat_ens.transpose("prediction_time", ...)

    obs = obs[FC_VARS]

    flat_obs = flatten_levels(obs)
    flat_obs = flat_obs.transpose("time", "variable", "longitude", "latitude")

    # Compute mean and std across the 'number' dimension (ensemble members) and save to file
    mean_ens = flat_ens.mean(dim="number")
    std_ens = flat_ens.std(dim="number", ddof=1)

    # Create new variable coordinates with _mean and _std suffixes
    mean_vars = [f"{var}_mean" for var in mean_ens.coords["variable"].values]
    std_vars = [f"{var}_std" for var in std_ens.coords["variable"].values]

    # Assign new variable coordinates
    mean_ens = mean_ens.assign_coords(variable=mean_vars)
    std_ens = std_ens.assign_coords(variable=std_vars)

    # Concatenate along variable dimension
    flat_ens_aggr = xr.concat([mean_ens, std_ens], dim="variable")
    flat_ens_aggr = flat_ens_aggr.transpose("prediction_time", "variable", "longitude", "latitude")
    # Save to disk for later use and faster loading
    flat_obs.to_netcdf(base_dir / OBSERVATIONS_FLAT_NAME, mode="w", format="NETCDF4")

    flat_ens_aggr.to_netcdf(base_dir / FORECAST_ENS_FLAT_AGG_NAME, mode="w", format="NETCDF4")


if __name__ == "__main__":
    main()
