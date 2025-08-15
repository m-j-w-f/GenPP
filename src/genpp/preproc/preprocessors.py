import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable

import dask  # noqa: F401
import dask.array as da
import numpy as np
import torch
import xarray as xr
from xarray.core.types import Dims

from genpp.data import MetadataVars


class Preprocessor(ABC, Hashable):
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

    def __hash__(self):
        """Hash based on class name and internal state."""
        state = getattr(self, "__dict__", {})
        state_str = json.dumps(state, sort_keys=True, default=str)
        combined = f"{self.__class__.__module__}.{self.__class__.__name__}:{state_str}"
        return int(hashlib.md5(combined.encode()).hexdigest(), 16)

    def __eq__(self, other):
        """Equality based on class and state."""
        return isinstance(other, self.__class__) and getattr(self, "__dict__", {}) == getattr(
            other, "__dict__", {}
        )


class StandardScalerPreprocessor(Preprocessor):
    """A preprocessor that standardizes the data by removing the mean and scaling to unit variance."""

    def __init__(self, dim: Dims, features: list[str] | None = None):
        """Initialize the StandardScalerPreprocessor.

        Args:
            dim: Dimension(s) along which to compute statistics for scaling.
            features: List of feature names to scale. If None, scale all features.
                      features not in this list will be kept unchanged.
        """
        self.dim = dim
        self.features = features
        self.is_fitted = False

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data by calculating the mean and standard deviation."""
        if self.features is not None:
            # Only fit on specified features
            if "feature" not in data.dims:
                raise ValueError("DataArray must have 'feature' dimension for fitting.")
            self.mean = data.sel(feature=self.features).mean(dim=self.dim).compute()
            self.std = data.sel(feature=self.features).std(dim=self.dim, ddof=1).compute()
            # Store this for the reverse transform
            self._changed_var_idx = np.where(np.isin(data.feature, self.features))[0]
        else:
            # Fit on all data
            self.mean = data.mean(dim=self.dim).compute()
            self.std = data.std(dim=self.dim, ddof=1).compute()
        self.is_fitted = True

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize the input data."""
        # Preserve attributes from the original data
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing data.")

        if self.features is not None:
            # Only scale specified features, keep others unchanged
            res_scaled = (data - self.mean) / self.std
            data.loc[dict(feature=self.features)] = res_scaled
        else:
            data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse standardization of the preprocessed data."""
        raise NotImplementedError(
            "TODO Inverse transform is not implemented for StandardScalerPreprocessor."
        )


class MinMaxScalerPreprocessor(Preprocessor):
    """A preprocessor that scales the data to a given range."""

    def __init__(self, dim: Dims, feature_range=(0, 1), features: list[str] | None = None):
        """Initialize the MinMaxScalerPreprocessor.

        Args:
            dim: Dimension(s) along which to compute statistics for scaling.
            feature_range: Target range for scaling, default (0, 1).
            features: List of feature names to scale. If None, scale all features.
                      features not in this list will be kept unchanged.
        """
        self.dim = dim
        self.feature_range = feature_range
        self.features = features
        self.is_fitted = False

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data by calculating the min and max values."""
        if self.features is not None:
            # Only fit on specified features
            if "feature" not in data.dims:
                raise ValueError("DataArray must have 'feature' dimension for fitting.")
            # Select only the specified features
            self.data_min = data.sel(feature=self.features).min(dim=self.dim).compute()
            self.data_max = data.sel(feature=self.features).max(dim=self.dim).compute()
            # Store this for the reverse transform
            self._changed_var_idx = np.where(np.isin(data.feature, self.features))[0]
        else:
            self.data_min = data.min(dim=self.dim).compute()
            self.data_max = data.max(dim=self.dim).compute()
        self.is_fitted = True

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Scale the input data to the specified feature range."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing data.")

        if self.features is not None:
            scaled_res = (data - self.data_min) / (self.data_max - self.data_min)
            if self.feature_range != (0, 1):
                scaled_res = (
                    scaled_res * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0]
                )
            data.loc[dict(feature=self.features)] = scaled_res
        else:
            # Scale all data (original behavior)
            scaled_data = (data - self.data_min) / (self.data_max - self.data_min)
            if self.feature_range != (0, 1):
                scaled_data = (
                    scaled_data * (self.feature_range[1] - self.feature_range[0])
                    + self.feature_range[0]
                )
            data = scaled_data
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse scaling of the preprocessed data."""
        raise NotImplementedError(
            "TODO Inverse transform is not implemented for MinMaxScalerPreprocessor."
        )


class AddMetadataPreprocessor(Preprocessor):
    """A preprocessor that adds metadata to the data."""

    def __init__(self, meta_features: list[MetadataVars] | type[MetadataVars]) -> None:
        self.meta_features = meta_features
        self.num_meta_features = len(meta_features)  # type: ignore

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data. This preprocessor does not require fitting."""
        pass

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Add metadata to the input data."""
        if not isinstance(data, xr.DataArray):
            raise TypeError("Input data must be an xarray DataArray.")

        meta_vars = []

        try:
            times = data.prediction_time
        except KeyError:
            warnings.warn(
                "The input data does not have a 'prediction_time' dimension. "
                "Metadata features that depend on 'prediction_time' will now depend on 'time'."
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
                    "feature": [MetadataVars.SIN_PREDICTION_TIME.value],
                }
            )
            meta_vars.append(time_grid)

        if MetadataVars.COS_PREDICTION_TIME in self.meta_features:  # type: ignore
            transformed_times = da.cos(times_doy * 2 * da.pi / 365).astype(da.float32)
            time_grid = transformed_times.expand_dims(
                {
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                    "feature": [MetadataVars.COS_PREDICTION_TIME.value],
                }
            )
            meta_vars.append(time_grid)

        if MetadataVars.LATITUDE in self.meta_features:  # type: ignore
            lat_grid = data.latitude.expand_dims(
                {
                    "prediction_time": data.prediction_time,
                    "longitude": data.longitude,
                    "feature": [MetadataVars.LATITUDE.value],
                }
            )
            meta_vars.append(lat_grid)

        if MetadataVars.LONGITUDE in self.meta_features:  # type: ignore
            lon_grid = data.longitude.expand_dims(
                {
                    "prediction_time": data.prediction_time,
                    "latitude": data.latitude,
                    "feature": [MetadataVars.LONGITUDE.value],
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
                    "feature": [MetadataVars.PIXEL_IDX.value],
                }
            )
            meta_vars.append(pixel_idx_grid)

        # Concatenate all metadata features along the last dimension using dask
        if meta_vars:
            meta_xr = xr.concat(meta_vars, dim="feature", coords="minimal").transpose(
                "prediction_time", "latitude", "longitude", "feature"
            )
            return xr.concat([data, meta_xr], dim="feature")
        else:
            return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Remove metadata from the preprocessed data."""
        return data[..., : -self.num_meta_features]  # Remove the last 'meta_features' dimensions
