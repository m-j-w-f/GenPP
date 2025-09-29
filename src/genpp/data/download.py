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

    for dataset_url, slice_dict, local_path, gcs_dest in [
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
        time_chunk_size = 37

        new_chunks = {
            "time": time_chunk_size,
            "latitude": ds_sliced.sizes["latitude"],  # Unchunked
            "longitude": ds_sliced.sizes["longitude"],  # Unchunked
        }
        # For variables that also have a 'level' dimension
        if "level" in ds_sliced.sizes:
            new_chunks["level"] = ds_sliced.sizes["level"]  # Unchunked
        if "number" in ds_sliced.sizes:
            new_chunks["number"] = ds_sliced.sizes["number"]  # Unchunked
        if "prediction_timedelta" in ds_sliced.sizes:
            new_chunks["prediction_timedelta"] = ds_sliced.sizes[
                "prediction_timedelta"
            ]  # Unchunked
        ds_rechunked = ds.chunk(new_chunks)

        # Remove existing local output if present
        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        # Write to local NetCDF
        print("Writing local NetCDF with Dask...")
        ds_rechunked.to_netcdf(local_path, format="NETCDF4")
        print(f"Local NetCDF written to {local_path}.")

        print(f"Uploading {local_path} to {gcs_dest} ...")
        subprocess.check_call(["gsutil", "-m", "cp", "-r", local_path, gcs_dest])
        print("Upload complete.")


if __name__ == "__main__":
    main()
