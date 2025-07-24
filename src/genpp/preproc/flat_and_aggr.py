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

import xarray as xr
from dask.distributed import Client
from pandas import Index

from genpp.data import (
    FC_VARS,
    FORECAST_ENS_PATH,
    MISSING_DAYS,
    OBSERVATIONS_PATH,
    OUTPUT_DIR,
)
from genpp.data.utils import flatten_levels, get_time_intersection

if __name__ == "__main__":
    client = Client()
    print("Dask dashboard link:", client.dashboard_link)

    ens = xr.open_dataset(
        FORECAST_ENS_PATH,
        chunks={
            "time": "auto",
            "number": -1,
            "latitude": -1,
            "longitude": -1,
            "level": -1,
        },
    )
    obs = xr.open_dataset(
        OBSERVATIONS_PATH,
        chunks="auto",
    )
    # Cut out the missing days first, since they are in time, not prediction_time
    ens = ens.sel(time=~ens.time.isin(MISSING_DAYS))
    ens = ens.assign_coords(
        prediction_time=ens.time + ens.prediction_timedelta
    ).swap_dims({"time": "prediction_time"})

    times = get_time_intersection(ens, obs)

    ens = ens.sel(prediction_time=times)
    obs = obs.sel(time=times)
    flat_ens = flatten_levels(ens)
    flat_ens = flat_ens.transpose("prediction_time", ...)

    obs = obs[FC_VARS]

    flat_obs = flatten_levels(obs)
    flat_obs = flat_obs.transpose("time", "latitude", "longitude", "variable")

    # Compute mean and std across the 'number' dimension (ensemble members) and save to file
    mean_ens = flat_ens.mean(dim="number")
    std_ens = flat_ens.std(dim="number", ddof=1)
    idx = Index(["mean", "std"], name="aggregate")
    flat_ens_aggr = xr.concat([mean_ens, std_ens], dim=idx)
    flat_ens_aggr = flat_ens_aggr.transpose(
        "prediction_time", "latitude", "longitude", "aggregate", "variable"
    )

    # Save to disk for later use and faster loading
    flat_obs.to_netcdf(OUTPUT_DIR / "flat_obs_preproc.nc", mode="w", format="NETCDF4")

    flat_ens_aggr.to_netcdf(
        OUTPUT_DIR / "flat_ens_preproc_agg.nc", mode="w", format="NETCDF4"
    )
