from datetime import timedelta

import numpy as np

from genpp import BASE_DIR

FORECAST_URL = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
FORECAST_ENS_URL = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"  # This is from 2018-2022, not 2016-2022
OBSERVATIONS_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
OUTPUT_DIR = BASE_DIR / "data" / "weatherbench2"

FORECAST_PATH = OUTPUT_DIR / "hres.nc"
FORECAST_ENS_PATH = OUTPUT_DIR / "ifs_ens.nc"
OBSERVATIONS_PATH = OUTPUT_DIR / "hres_t0.nc"

TIME_SLICE = slice("2018-01-01", "2022-12-31")
LATITUDE_SLICE = slice(47.3, 55.1)
LONGITUDE_SLICE = slice(5.9, 15.0)
PREDICTION_TIMEDELTA = timedelta(hours=48)
LEVEL = [500, 700, 850]  # Level of the FORECAST_ENS data

# For this date, the predictions in the ensemble data are missing.
# TODO: this will need a fix once we are using multiple prediction times.
MISSING_DAYS = [np.datetime64("2019-10-17T00:00:00.000000000")]

FORECAST_SLICE = {
    "time": TIME_SLICE,
    "latitude": LATITUDE_SLICE,
    "longitude": LONGITUDE_SLICE,
    "level": LEVEL,
    "prediction_timedelta": PREDICTION_TIMEDELTA,
}

FORECAST_ENS_SLICE = {
    "time": TIME_SLICE,
    "latitude": LATITUDE_SLICE,
    "longitude": LONGITUDE_SLICE,
    "level": LEVEL,
    "prediction_timedelta": PREDICTION_TIMEDELTA,
}

OBSERVATIONS_SLICE = {
    "time": TIME_SLICE,
    "latitude": LATITUDE_SLICE,
    "longitude": LONGITUDE_SLICE,
    "level": LEVEL,
}

FC_VARS = [
    "2m_temperature",
    "10m_wind_speed",
]
