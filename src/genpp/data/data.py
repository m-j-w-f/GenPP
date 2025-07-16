"""
This file will be used to store a local copy of the weatherbench2 dataset.
The dataset will be downloaded from the Google Cloud Storage bucket and stored locally as a Zarr file.

To make filesizes manageable, the dataset is sliced to only contain germany, the 48h forecast and the levels 500, 700 and 850.
The dataset is also sliced to only contain the time from 2018-01-01 to 2022-12-31.
The choice of levels is based on the fact that these are the levels used in the IFS ensemble dataset.
The dataset is stored in the `OUTPUT_DIR` directory.
"""

import xarray as xr
from dask.diagnostics.progress import ProgressBar

from genpp.data import (
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
from genpp.data.utils import print_info

print(f"Output directory: {OUTPUT_DIR}")

for dataset_url in [FORECAST, FORECAST_ENS, OBSERVATIONS]:
    dataset_name = dataset_url.split("/")[-2]
    output_path = OUTPUT_DIR / dataset_name
    print(f"Processing dataset: {dataset_url}")

    ds = xr.open_zarr(dataset_url, decode_timedelta=True)

    # Lazy loading and slicing the dataset
    ds_sliced = ds.sel(
        time=TIME_SLICE,
        latitude=LATITUDE_SLICE,  # Germany
        longitude=LONGITUDE_SLICE,  # Germany
        prediction_timedelta=PREDICTION_TIMEDELTA,
        level=LEVEL,
    )

    print_info(ds_sliced)

    with ProgressBar():
        ds_sliced.to_netcdf(
            path=output_path.with_suffix(".nc"), mode="w", format="NETCDF4"
        )

    print(f"Completed: {dataset_name}\n{'=' * 50}")
