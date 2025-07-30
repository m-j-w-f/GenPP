from abc import ABC, abstractmethod

import torch
import xarray as xr


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

    def __init__(self, dim: str):
        self.dim = dim
        self.is_fitted = False

    def fit(self, data: xr.DataArray) -> None:
        """Fit the preprocessor to the data by calculating the mean and standard deviation."""
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
