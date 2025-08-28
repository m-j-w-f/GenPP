import pandas as pd
import xarray as xr


def flatten_levels(ds: xr.Dataset, level_dim: str = "level") -> xr.DataArray:
    """Flattens the level dimension of an xarray Dataset by creating separate features for each level.
    Do not call .compute() on large datasets, as this will likely lead to memory issues.

    Args:
        ds (xr.Dataset): The input dataset.
        level_dim (str, optional): The name of the level dimension. Defaults to "level".

    Returns:
        xr.DataArray: The flattened dataset with separate features for each level.
    """
    # Create new dataset
    ds_flat = xr.Dataset()

    # Copy coordinates (except level)
    for coord in ds.coords:
        if coord != level_dim:
            ds_flat.coords[coord] = ds.coords[coord]

    # Process each feature
    for feat_name, feat_data in ds.data_vars.items():
        if level_dim in feat_data.dims:
            # Create separate features for each level
            for level in feat_data[level_dim].values:
                new_name = f"{feat_name}_lev{level}"
                ds_flat[new_name] = feat_data.sel({level_dim: level}).reset_coords(
                    level_dim, drop=True
                )
        else:
            # Keep features without level as-is
            ds_flat[feat_name] = feat_data

    # Handle empty dataset case
    if len(ds_flat.data_vars) == 0:
        return xr.DataArray(data=[], coords={"feature": []}, dims=["feature"])
    return ds_flat.to_array().rename({"variable": "feature"})


def get_time_intersection(
    ds1: xr.Dataset | xr.DataArray,
    ds2: xr.Dataset | xr.DataArray,
    time_dim1: str = "prediction_time",
    time_dim2: str = "time",
) -> pd.Index:
    """Get the intersection of the prediction time in the ensemble dataset and the observation time.

    Args:
        ens (xr.Dataset): The ensemble dataset with a 'prediction_time' coordinate.
        obs (xr.Dataset): The observation dataset with a 'time' coordinate.
        time_dim1 (str, optional): The name of the time dimension in the first dataset. Defaults to "prediction_time".
        time_dim2 (str, optional): The name of the time dimension in the second dataset

    Returns:
        pd.Index: The intersection of the two time coordinates.
    """
    return ds1[time_dim1].to_index().intersection(ds2[time_dim2].to_index())
