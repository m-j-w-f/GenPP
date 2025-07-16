import dask
import dask.config
import gcsfs
import xarray as xr
from dask.distributed import Client, LocalCluster, progress

from genpp.data import (
    FC_VARS,
    FORECAST,
    FORECAST_ENS,
    LATITUDE_SLICE,
    LEVEL,
    LONGITUDE_SLICE,
    OBSERVATIONS,
    OUTPUT_DIR,
    PREDICTION_TIMEDELTA,
    TIME_SLICE,
)


def main() -> None:
    dataset_url = FORECAST_ENS
    output_path = OUTPUT_DIR / "hres_t0.nc"

    # Configure Dask to maximize throughput
    dask.config.set(
        {
            "array.chunk-size": "128MiB",
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.9,
            "distributed.comm.timeouts.connect": 100,
            "distributed.comm.timeouts.tcp": 100,
        }
    )

    # Set up client with more resources for better throughput
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="4GB",
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    fs = gcsfs.GCSFileSystem(
        retries=10,
        timeout=120,
    )

    ds = xr.open_zarr(fs.get_mapper(dataset_url), consolidated=True)

    print("Downloading spatial subset (Germany)...")
    if dataset_url == FORECAST_ENS:
        ds_subset = ds.sel(
            latitude=LATITUDE_SLICE,
            longitude=LONGITUDE_SLICE,
            prediction_timedelta=PREDICTION_TIMEDELTA,
            level=LEVEL,
        )
    elif dataset_url == OBSERVATIONS:
        ds_subset = ds[FC_VARS].sel(latitude=LATITUDE_SLICE, longitude=LONGITUDE_SLICE)
    elif dataset_url == FORECAST:
        ds_subset = ds.sel(
            latitude=LATITUDE_SLICE,
            longitude=LONGITUDE_SLICE,
            time=TIME_SLICE,
            level=LEVEL,
        )
    else:
        raise ValueError(f"Unknown dataset URL: {dataset_url}")

    ds_subset = ds_subset.chunk(
        {
            "time": "auto",
            "latitude": "auto",
            "longitude": "auto",
        }
    )

    # Start download with progress monitoring
    print("Beginning download...")
    future = client.persist(ds_subset)
    progress(future)

    # Save result
    result = future.compute()  # type: ignore
    result.to_netcdf(path=output_path, mode="w", format="NETCDF4")


if __name__ == "__main__":
    main()
