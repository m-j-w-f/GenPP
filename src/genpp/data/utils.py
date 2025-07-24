import pandas as pd
import xarray as xr


def flatten_levels(ds: xr.Dataset, level_dim: str = "level") -> xr.DataArray:
    """Flattens the level dimension of an xarray Dataset by creating separate variables for each level.
    Do not call .compute() on large datasets, as this will likely lead to memory issues.

    Args:
        ds (xr.Dataset): The input dataset.
        level_dim (str, optional): The name of the level dimension. Defaults to "level".

    Returns:
        xr.DataArray: The flattened dataset with separate variables for each level.
    """
    # Create new dataset
    ds_flat = xr.Dataset()

    # Copy coordinates (except level)
    for coord in ds.coords:
        if coord != level_dim:
            ds_flat.coords[coord] = ds.coords[coord]

    # Process each variable
    for var_name, var_data in ds.data_vars.items():
        if level_dim in var_data.dims:
            # Create separate variables for each level
            for level in var_data.level.values:
                new_name = f"{var_name}_lev{level}"
                ds_flat[new_name] = var_data.sel(level=level).reset_coords(
                    level_dim, drop=True
                )
        else:
            # Keep variables without level as-is
            ds_flat[var_name] = var_data
    return ds_flat.to_array()


def get_time_intersection(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
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
