"""Backward compatibility module for weatherbench2 data.

This module re-exports all weatherbench2 data components for backward compatibility.
New code should import from genpp.data.weatherbench2 directly.
"""
import warnings
from genpp.data.weatherbench2 import (
    FORECAST_URL,
    FORECAST_ENS_URL,
    OBSERVATIONS_URL,
    OUTPUT_DIR,
    FORECAST_ENS_NAME,
    OBSERVATIONS_NAME,
    FORECAST_ENS_PATH,
    OBSERVATIONS_PATH,
    FORECAST_ENS_FLAT_AGG_NAME,
    OBSERVATIONS_FLAT_NAME,
    FORECAST_ENS_FLAT_AGG_PATH,
    OBSERVATIONS_FLAT_PATH,
    FORECAST_ENS_FLAT_AGG_PREPROC_NAME,
    OBSERVATIONS_FLAT_PREPROC_NAME,
    FORECAST_ENS_FLAT_AGG_PREPROC_PATH,
    OBSERVATIONS_FLAT_PREPROC_PATH,
    TIME_SLICE,
    LATITUDE_SLICE,
    LONGITUDE_SLICE,
    PREDICTION_TIMEDELTA,
    LEVEL,
    MISSING_DAYS,
    FORECAST_SLICE,
    FORECAST_ENS_SLICE,
    OBSERVATIONS_SLICE,
    FC_VARS,
    ALL_VARS,
    MetadataVars,
    TRAIN_PREDICTIONS,
    VAL_PREDICTIONS,
    TEST_PREDICTIONS,
    _get_MapDataset,
)

# Issue deprecation warning when importing from genpp.data
warnings.warn(
    "Importing from 'genpp.data' is deprecated. "
    "Please use 'genpp.data.weatherbench2' or 'genpp.data.icon' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FORECAST_URL",
    "FORECAST_ENS_URL",
    "OBSERVATIONS_URL",
    "OUTPUT_DIR",
    "FORECAST_ENS_NAME",
    "OBSERVATIONS_NAME",
    "FORECAST_ENS_PATH",
    "OBSERVATIONS_PATH",
    "FORECAST_ENS_FLAT_AGG_NAME",
    "OBSERVATIONS_FLAT_NAME",
    "FORECAST_ENS_FLAT_AGG_PATH",
    "OBSERVATIONS_FLAT_PATH",
    "FORECAST_ENS_FLAT_AGG_PREPROC_NAME",
    "OBSERVATIONS_FLAT_PREPROC_NAME",
    "FORECAST_ENS_FLAT_AGG_PREPROC_PATH",
    "OBSERVATIONS_FLAT_PREPROC_PATH",
    "TIME_SLICE",
    "LATITUDE_SLICE",
    "LONGITUDE_SLICE",
    "PREDICTION_TIMEDELTA",
    "LEVEL",
    "MISSING_DAYS",
    "FORECAST_SLICE",
    "FORECAST_ENS_SLICE",
    "OBSERVATIONS_SLICE",
    "FC_VARS",
    "ALL_VARS",
    "MetadataVars",
    "TRAIN_PREDICTIONS",
    "VAL_PREDICTIONS",
    "TEST_PREDICTIONS",
    "_get_MapDataset",
]
