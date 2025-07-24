from typing import Callable

import xarray as xr
import xbatcher
from xbatcher.loaders.torch import MapDataset, to_tensor


def get_MapDataset(
    x_ds: xr.DataArray,
    y_ds: xr.DataArray,
    x_kwargs: dict,
    y_kwargs: dict,
    batch_size: int = 8,
    x_transform: Callable = to_tensor,
    y_transform: Callable = to_tensor,
) -> MapDataset:
    """Creates a MapDataset for the given xarray Datasets.

    Args:
        x_ds (xr.Dataset): dataset containing the input data.
        y_ds (xr.Dataset): dataset containing the target data.
        x_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for input data.
        y_kwargs (dict): keyword arguments for the xbatcher.BatchGenerator for target data.
        batch_size (int, optional): number of samples per batch. Defaults to 8.
        x_transform (Callable, optional): transform function for input data. Defaults to to_tensor.
        y_transform (Callable, optional): transform function for target data. Defaults to to_tensor.

    Returns:
        MapDataset: A PyTorch-compatible MapDataset that generates batches from the input datasets
               using xbatcher BatchGenerators with the specified transforms applied.
    """
    x_kwargs["input_dims"]["prediction_time"] = batch_size
    y_kwargs["input_dims"]["time"] = batch_size

    x_gen = xbatcher.BatchGenerator(
        x_ds,
        **x_kwargs,
    )
    y_gen = xbatcher.BatchGenerator(
        y_ds,
        **y_kwargs,
    )

    map_ds = MapDataset(
        X_generator=x_gen,
        y_generator=y_gen,
        transform=x_transform,
        target_transform=y_transform,
    )
    return map_ds
