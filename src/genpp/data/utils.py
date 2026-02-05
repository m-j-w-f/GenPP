from enum import Enum
from warnings import warn

import pandas as pd
import xarray as xr


class MetadataVars(Enum):
    """Metadata variable names used across datasets."""
    PIXEL_IDX = "pixel_idx"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SIN_PREDICTION_TIME = "sin_prediction_time"
    COS_PREDICTION_TIME = "cos_prediction_time"


def flatten_levels(ds: xr.Dataset, level_dim: str = "level", interleave=True) -> xr.Dataset:
    """Flattens the level dimension of an xarray Dataset by creating separate features for each level.
    Do not call .compute() on large datasets, as this will likely lead to memory issues.

    Args:
        ds (xr.Dataset): The input dataset.
        level_dim (str, optional): The name of the level dimension. Defaults to "level".
        interleave (bool, optional): If True, the new features are interleaved by feature name
            (e.g., var1_level1, var1_level2, var2_level1, var2_level2).
            If False, they are interleaved by level
            (e.g., var1_level1, var2_level1, var1_level2, var2_level2).
            Defaults to True.

    Returns:
        xr.Dataset: The flattened dataset with separate features for each level.
    """
    # Create new dataset
    ds_flat = xr.Dataset()

    # Copy coordinates (except level)
    for coord in ds.coords:
        if coord != level_dim:
            ds_flat.coords[coord] = ds.coords[coord]

    # Get features with and without the level dimension
    features_with_level = [name for name, var in ds.data_vars.items() if level_dim in var.dims]
    features_without_level = [name for name in ds.data_vars if name not in features_with_level]
    levels = ds[level_dim].values

    if interleave:
        # Interleave by feature: var1_level1, var1_level2, var2_level1, var2_level2
        for feat_name in features_with_level:
            feat_data = ds[feat_name]
            for level in levels:
                new_name = f"{feat_name}+{level_dim}_{level}"
                ds_flat[new_name] = feat_data.sel({level_dim: level}).reset_coords(
                    level_dim, drop=True
                )
    else:
        # Interleave by level: var1_level1, var2_level1, var1_level2, var2_level2
        for level in levels:
            for feat_name in features_with_level:
                feat_data = ds[feat_name]
                new_name = f"{feat_name}+{level_dim}_{level}"
                ds_flat[new_name] = feat_data.sel({level_dim: level}).reset_coords(
                    level_dim, drop=True
                )

    # Keep features without level dimension as-is
    for feat_name in features_without_level:
        ds_flat[feat_name] = ds[feat_name]

    # Handle empty dataset case
    if not ds_flat.data_vars:
        raise ValueError("The input dataset has no data variables.")

    return ds_flat


def get_time_intersection(
    ds1: xr.Dataset | xr.DataArray,
    ds2: xr.Dataset | xr.DataArray,
    time_dim1: str = "prediction_time",
    time_dim2: str = "time",
) -> pd.Index:
    """Get the intersection of the prediction time in the ensemble dataset and the observation time.

    Args:
        ds1 (xr.Dataset): The first dataset with a time coordinate.
        ds2 (xr.Dataset): The second dataset with a time coordinate.
        time_dim1 (str, optional): The name of the time dimension in the first dataset. Defaults to "prediction_time".
        time_dim2 (str, optional): The name of the time dimension in the second dataset. Defaults to "time".

    Returns:
        pd.Index: The intersection of the two time coordinates.
    """
    # Emit deprecation warning
    warn(
        "get_time_intersection is deprecated and will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ds1[time_dim1].to_index().intersection(ds2[time_dim2].to_index())
