"""
Slice the IFS ensemble Zarr dataset in parallel with Dask,
write a local NetCDF, then upload it to GCS.

This script is designed to run on a gcloud compute instance.
"""

import os
import subprocess

import xarray as xr
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, LocalCluster

from genpp.data import (
    FORECAST_ENS_SLICE,
    FORECAST_ENS_URL,
    FORECAST_SLICE,
    FORECAST_URL,
    OBSERVATIONS_SLICE,
    OBSERVATIONS_URL,
)


def main():
    # Use 8 processes, each with 2 threads, ~12 GB per worker
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=2,
        memory_limit="12GB",
        processes=True,
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)
    print(client)

    for dataset_url, slice_dict, local_nc, gcs_dest in [
        (
            FORECAST_URL,
            FORECAST_SLICE,
            "hres_slice_l.nc",
            "gs://slice-data-output/hres_slice_l.nc",
        ),
        (
            FORECAST_ENS_URL,
            FORECAST_ENS_SLICE,
            "ifs_ens_slice_l.nc",
            "gs://slice-data-output/ifs_ens_slice_l.nc",
        ),
        (
            OBSERVATIONS_URL,
            OBSERVATIONS_SLICE,
            "hres_t0_slice_l.nc",
            "gs://slice-data-output/hres_t0_slice_l.nc",
        ),
    ]:
        # Open the Zarr store lazily
        ds = xr.open_zarr(dataset_url, decode_timedelta=True)

        # Select the region and forecast lead time
        ds_sliced = ds.sel(slice_dict)

        if slice_dict.get("prediction_timedelta") is not None:
            ds_sliced = ds_sliced.chunk({"prediction_timedelta": 20, "time": 8}).unify_chunks()
        else:
            ds_sliced = ds_sliced.chunk({"time": 20}).unify_chunks()

        # Remove existing local output if present
        if os.path.exists(local_nc):
            os.remove(local_nc)

        # Write to local NetCDF in parallel
        print("Writing local NetCDF with Dask...")
        with ProgressBar():
            ds_sliced.to_netcdf(local_nc, engine="netcdf4")
        print(f"Local NetCDF written to {local_nc}.")

        print(f"Uploading {local_nc} to {gcs_dest} ...")
        subprocess.check_call(["gsutil", "-m", "cp", local_nc, gcs_dest])
        print("Upload complete.")


if __name__ == "__main__":
    main()
