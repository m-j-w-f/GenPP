from datetime import timedelta

from genpp import BASE_DIR

FORECAST = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
FORECAST_ENS = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"  # This is from 2018-2022, not 2016-2022
OBSERVATIONS = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
OUTPUT_DIR = BASE_DIR / "data" / "weatherbench2"

TIME_SLICE = slice("2018-01-01", "2022-12-31")
LATITUDE_SLICE = slice(47.3, 55.1)
LONGITUDE_SLICE = slice(5.9, 15.0)
PREDICTION_TIMEDELTA = timedelta(hours=48)
LEVEL = [500, 700, 850]  # Level of the FORECAST_ENS data

FC_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "2m_temperature",
]
