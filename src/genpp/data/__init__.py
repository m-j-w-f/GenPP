from datetime import timedelta
from enum import Enum

import numpy as np

from genpp import BASE_DIR

FORECAST_URL = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
FORECAST_ENS_URL = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"  # This is from 2018-2022, not 2016-2022
OBSERVATIONS_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
OUTPUT_DIR = BASE_DIR / "data" / "weatherbench2"

FORECAST_NAME = "hres.zarr"
FORECAST_ENS_NAME = "ifs_ens.zarr"
OBSERVATIONS_NAME = "hres_t0.zarr"

FORECAST_PATH = OUTPUT_DIR / FORECAST_NAME
FORECAST_ENS_PATH = OUTPUT_DIR / FORECAST_ENS_NAME
OBSERVATIONS_PATH = OUTPUT_DIR / OBSERVATIONS_NAME

FORECAST_ENS_FLAT_AGG_NAME = "ens_flat_agg.zarr"
OBSERVATIONS_FLAT_NAME = "obs_flat.zarr"

FORECAST_ENS_FLAT_AGG_PATH = OUTPUT_DIR / FORECAST_ENS_FLAT_AGG_NAME
OBSERVATIONS_FLAT_PATH = OUTPUT_DIR / OBSERVATIONS_FLAT_NAME

FORECAST_ENS_FLAT_AGG_PREPROC_NAME = "ens_flat_agg_preproc.zarr"
OBSERVATIONS_FLAT_PREPROC_NAME = "obs_flat_preproc.zarr"

FORECAST_ENS_FLAT_AGG_PREPROC_PATH = OUTPUT_DIR / FORECAST_ENS_FLAT_AGG_PREPROC_NAME
OBSERVATIONS_FLAT_PREPROC_PATH = OUTPUT_DIR / OBSERVATIONS_FLAT_PREPROC_NAME

TIME_SLICE = slice("2018-01-01", "2022-12-31")
LATITUDE_SLICE = slice(47.3, 55.1)
LONGITUDE_SLICE = slice(5.9, 15.0)
PREDICTION_TIMEDELTA = [timedelta(days=d + 1) for d in range(0, 5)]  # 1-5 day forecasts
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

ALL_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "2m_temperature",
    "geopotential_lev500",
    "geopotential_lev700",
    "geopotential_lev850",
    "mean_sea_level_pressure",
    "relative_humidity_lev500",
    "relative_humidity_lev700",
    "relative_humidity_lev850",
    "specific_humidity_lev500",
    "specific_humidity_lev700",
    "specific_humidity_lev850",
    "temperature_lev500",
    "temperature_lev700",
    "temperature_lev850",
    "total_precipitation",
    "total_precipitation_24hr",
    "total_precipitation_6hr",
    "u_component_of_wind_lev500",
    "u_component_of_wind_lev700",
    "u_component_of_wind_lev850",
    "v_component_of_wind_lev500",
    "v_component_of_wind_lev700",
    "v_component_of_wind_lev850",
    "wind_speed_lev500",
    "wind_speed_lev700",
    "wind_speed_lev850",
]


class MetadataVars(Enum):
    PIXEL_IDX = "pixel_idx"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SIN_PREDICTION_TIME = "sin_prediction_time"
    COS_PREDICTION_TIME = "cos_prediction_time"
