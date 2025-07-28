import warnings
from typing import Any

import torch
import xarray as xr


class StandardScaler:
    def __init__(self, dim: list[str] | str) -> None:
        self.dim = dim

    def fit(self, data: xr.DataArray) -> None:
        mean_da = data.mean(dim=self.dim)
        scale_da = data.std(dim=self.dim, ddof=1)

        self.mean = torch.Tensor(mean_da.values)
        self.scale = torch.Tensor(scale_da.values)

        # Store the dimension info for proper broadcasting during transform
        self._mean_da = mean_da
        self._scale_da = scale_da

        # Check for zero standard deviation and warn
        if torch.any(self.scale == 0):
            warnings.warn(
                "Standard deviation is zero for one or more features. "
                "This will result in division by zero and produce inf/nan values during transformation.",
                RuntimeWarning,
            )

    def transform(self, data: xr.DataArray) -> torch.Tensor:
        # Use the stored DataArrays for proper broadcasting
        normalized = (data - self._mean_da) / self._scale_da
        return torch.Tensor(normalized.values)

    def fit_transform(self, data: xr.DataArray) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.transform(*args, **kwargs)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.scale + self.mean
