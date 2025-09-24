"""
Slice the IFS ensemble Zarr dataset in parallel with Dask,
write a local NetCDF, then upload it to GCS.

This script is designed to run on a gcloud compute instance.
"""

import os
import shutil
import subprocess

import xarray as xr
from dask.distributed import Client, LocalCluster

from genpp.data import (
    FORECAST_ENS_SLICE,
    FORECAST_ENS_URL,
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

    for dataset_url, slice_dict, local_zarr, gcs_dest in [
        (
            FORECAST_ENS_URL,
            FORECAST_ENS_SLICE,
            "ifs_ens_slice.zarr",
            "gs://slice-data-output/ifs_ens_slice.zarr",
        ),
        (
            OBSERVATIONS_URL,
            OBSERVATIONS_SLICE,
            "hres_t0_slice.zarr",
            "gs://slice-data-output/hres_t0_slice.zarr",
        ),
    ]:
        # Open the Zarr store lazily
        ds = xr.open_zarr(dataset_url, decode_timedelta=True)

        # Select the region and forecast lead time
        ds_sliced = ds.sel(slice_dict)

        print(f"Dataset size: {ds_sliced.nbytes / (1024**3):.2f} GB")

        # Slightly better chunking for writing
        ds_sliced = ds_sliced.chunk("auto")

        # Remove existing local output if present
        if os.path.exists(local_zarr):
            shutil.rmtree(local_zarr)

        # Write to local Zarr
        print("Writing local NetCDF with Dask...")
        ds_sliced.to_zarr(local_zarr, mode="w", compute=True, consolidated=True)
        print(f"Local Zarr written to {local_zarr}.")

        print(f"Uploading {local_zarr} to {gcs_dest} ...")
        subprocess.check_call(["gsutil", "-m", "cp", "-r", local_zarr, gcs_dest])
        print("Upload complete.")


if __name__ == "__main__":
    main()
