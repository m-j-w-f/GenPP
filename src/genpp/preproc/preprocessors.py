import warnings
from abc import ABC, abstractmethod
from typing import List, Type

import dask  # noqa: F401
import dask.array as da
import torch
import xarray as xr
from xarray.core.types import Dims

from genpp.data import MetadataVars


class Preprocessor(ABC):
    """Abstract base class for preprocessing data.
    Preprocessors should be able to preprocess the xarray DataArray to return a xarray DataArray.
    The Inverse transform should be applied to a torch.Tensor to return a xarray DataArray or a torch.Tensor.
    """

    @abstractmethod
    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Preprocess the input data using the defined transforms.

        Args:
            data: The input data to preprocess.

        Returns:
            xr.DataArray: The preprocessed data.
        """
        pass

    @abstractmethod
    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data.

        Args:
            data: The input data to fit the preprocessor on.
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> xr.DataArray | torch.Tensor:
        """Inverse transform the preprocessed data back to its original form.

        Args:
            data: The preprocessed data to inverse transform.

        Returns:
            xr.DataArray | torch.Tensor: The data in its original form.
        """
        pass


class StandardScalerPreprocessor(Preprocessor):
    """A preprocessor that standardizes the data by removing the mean and scaling to unit variance."""

    def __init__(self, dim: Dims):
        self.dim = dim
        self.is_fitted = False

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data by calculating the mean and standard deviation."""
        # TODO change mean computation as discussed
        self.mean = data.mean(dim=self.dim).compute()
        self.std = data.std(dim=self.dim, ddof=1).compute()
        self.is_fitted = True

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize the input data."""
        result = (data - self.mean) / self.std
        # Preserve attributes from the original data
        result.attrs = data.attrs
        return result

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse standardization of the preprocessed data."""
        raise NotImplementedError(
            "TODO Inverse transform is not implemented for StandardScalerPreprocessor."
        )


class AddMetadataPreprocessor(Preprocessor):
    """A preprocessor that adds metadata to the data."""

    def __init__(self, meta_features: List[MetadataVars] | Type[MetadataVars]) -> None:
        self.meta_features = meta_features
        self.num_meta_features = len(meta_features)  # type: ignore

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data. This preprocessor does not require fitting."""
        pass

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Add metadata to the input data."""
        # TODO this function could need some optimization, dask is complaining about the graph size.
        if not isinstance(data, xr.DataArray):
            raise TypeError("Input data must be an xarray DataArray.")

        meta_vars = []

        try:
            times = data.prediction_time
        except KeyError:
            warnings.warn(
                "The input data does not have a 'prediction_time' dimension. "
                "Metadata variables that depend on 'prediction_time' will now depend on 'time'."
            )
            times = data.time

        times_doy = times.dt.dayofyear

        if MetadataVars.SIN_PREDICTION_TIME in self.meta_features:  # type: ignore
            transformed_times = da.sin(times_doy * 2 * da.pi / 365).astype(da.float32)
            # Use dask's broadcast_to instead of einops
            time_grid = transformed_times.expand_dims(
                {
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                    "variable": [MetadataVars.SIN_PREDICTION_TIME.value],
                }
            )
            meta_vars.append(time_grid)

        if MetadataVars.COS_PREDICTION_TIME in self.meta_features:  # type: ignore
            transformed_times = da.cos(times_doy * 2 * da.pi / 365).astype(da.float32)
            time_grid = transformed_times.expand_dims(
                {
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                    "variable": [MetadataVars.COS_PREDICTION_TIME.value],
                }
            )
            meta_vars.append(time_grid)

        if MetadataVars.LATITUDE in self.meta_features:  # type: ignore
            lat_grid = data.latitude.expand_dims(
                {
                    "prediction_time": data.prediction_time,
                    "longitude": data.longitude,
                    "variable": [MetadataVars.LATITUDE.value],
                }
            )
            meta_vars.append(lat_grid)

        if MetadataVars.LONGITUDE in self.meta_features:  # type: ignore
            lon_grid = data.longitude.expand_dims(
                {
                    "prediction_time": data.prediction_time,
                    "latitude": data.latitude,
                    "variable": [MetadataVars.LONGITUDE.value],
                }
            )
            meta_vars.append(lon_grid)

        if MetadataVars.PIXEL_IDX in self.meta_features:  # type: ignore
            pixel_idx = da.arange(
                data.latitude.size * data.longitude.size, chunks="auto", dtype=da.float32
            )
            pixel_idx_reshaped = pixel_idx.reshape(data.latitude.size, data.longitude.size)
            pixel_idx_xr = xr.DataArray(
                pixel_idx_reshaped,
                dims={
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                },
            )
            pixel_idx_grid = pixel_idx_xr.expand_dims(
                {
                    "prediction_time": data.prediction_time,
                    "variable": [MetadataVars.PIXEL_IDX.value],
                }
            )
            meta_vars.append(pixel_idx_grid)

        # Concatenate all metadata variables along the last dimension using dask
        if meta_vars:
            meta_xr = xr.concat(meta_vars, dim="variable", coords="minimal").transpose(
                "prediction_time", "latitude", "longitude", "variable"
            )
            return xr.concat([data, meta_xr], dim="variable")
        else:
            return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Remove metadata from the preprocessed data."""
        return data[..., : -self.num_meta_features]  # Remove the last 'meta_features' dimensions
