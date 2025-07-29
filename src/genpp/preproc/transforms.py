import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, override

import torch
import torch.nn.functional as F
import xarray as xr
from einops import rearrange


class Transform(ABC):
    """Abstract base class for data transformations."""

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input data.

        Args:
            data: The input data to transform.

        Returns:
            torch.Tensor: The transformed data.
        """
        pass

    def __call__(self, data: xr.DataArray | torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input data.

        Args:
            data: The input data to transform.

        Returns:
            torch.Tensor: The transformed data.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data.values)
        return self.transform(data)


class StandardScaler(Transform):
    # TODO since this transform is pixel-wise, it does not work when the input is cropped.
    # However me might be able to get the lat and lon dimensions from the xarray data and cut the mean and scale accordingly.
    # However this information is in the __call__ method so we have to override it.
    def __init__(self, dim: str) -> None:
        self.dim = dim

    def fit(self, data: xr.DataArray) -> None:
        mean_da = data.mean(dim=self.dim)
        scale_da = data.std(dim=self.dim, ddof=1)

        self.mean = torch.tensor(mean_da.values)
        self.scale = torch.tensor(scale_da.values)

        del mean_da, scale_da  # Free memory, sometimes you have to do this explicitly

        # Check for zero standard deviation and warn
        if torch.any(self.scale == 0):
            warnings.warn(
                "Standard deviation is zero for one or more features. "
                "This will result in division by zero and produce inf/nan values during transformation.",
                RuntimeWarning,
            )

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        normalized = (data - self.mean) / self.scale
        return normalized

    @override
    def __call__(self, data: xr.DataArray | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "This is a TODO. The StandardScaler is pixel-wise and does not work with cropped data. "
        )

    def fit_transform(self, data: xr.DataArray) -> torch.Tensor:
        self.fit(data)
        return self(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.scale + self.mean


class Pad(Transform):
    """
    A callable class to pad a n-D tensor of shape (B, H, W, ...)
    on the H and W dimensions to a target size. It is assume that H and W stay constant.

    Args:
        target_shape (Optional[Tuple[int, int]]): The target (H, W) dimensions.
            Defaults to (32, 48).
        mode (str): The padding mode to use. Accepts "constant", "reflect",
            "replicate", or "circular". Defaults to "reflect".
    """

    def __init__(self, target_shape: Tuple[int, int] = (32, 48), mode: str = "reflect"):
        if target_shape is not None:
            assert len(target_shape) == 2, "target_shape must be a tuple of (height, width)"
        self.target_shape = target_shape
        self.mode = mode
        self.is_fitted = False

    def fit(self, data: torch.Tensor) -> None:
        """This save the data shape to reconstruct the correct dims later as pytorch does not support non-constant padding for 2D, 3D, 4D and 5D tensors.

        Args:
            data (torch.Tensor): The input data tensor.
        """
        self.is_fitted = True
        b, h, w, *c = data.shape
        extra_dims_names = [f"c{i}" for i in range(len(c))]
        self.extra_dims_dict = {name: size for name, size in zip(extra_dims_names, c)}
        left_pattern = f"(b {' '.join(extra_dims_names)}) h w"
        right_pattern = f"b h w {' '.join(extra_dims_names)}"
        self.pattern = f"{left_pattern} -> {right_pattern}"

        # Calculate padding for width (last dimension)
        target_h, target_w = self.target_shape
        pad_w = target_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Calculate padding for height (second to last dimension)
        pad_h = target_h - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        self.padding = (pad_left, pad_right, pad_top, pad_bottom)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor's spatial dimensions (H, W).

        Args:
            data (torch.Tensor): The input tensor with shape
                (batch, height, width, channels1, channels2).

        Returns:
            torch.Tensor: The padded tensor with shape
                (batch, target_height, target_width, channels1, channels2).
        """
        if self.target_shape is None:
            return data

        b, h, w, *c = data.shape
        if not self.is_fitted:
            self.fit(data)

        target_h, target_w = self.target_shape

        if h >= target_h and w >= target_w:
            raise ValueError(
                f"Input tensor dimensions ({h}, {w}) are larger than target shape {self.target_shape}. "
                f"Consider using a cropping operation or increasing the target size."
            )

        if self.mode == "constant":  # constant padding works for any dimension
            data_reshaped = rearrange(data, "b h w ... -> b ... h w")
            data_padded = F.pad(data_reshaped, self.padding, mode=self.mode)
            output = rearrange(data_padded, "b ... h w -> b h w ...")
            return output
        else:
            # torch.nn.functional.pad works on the last dimensions of a tensor.
            data_reshaped = rearrange(data, "b h w ... -> (b ...) h w")
            data_padded = F.pad(data_reshaped, self.padding, mode=self.mode)
            output = rearrange(data_padded, self.pattern, b=b, **self.extra_dims_dict)

            return output


class Pipe(Transform):
    """A pipeline that chains multiple transforms together.

    Args:
        transforms (List[Transform]): A list of transforms to apply in sequence.
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in the pipeline sequentially.

        Args:
            data: The input data to transform.

        Returns:
            torch.Tensor: The transformed data after applying all transforms.
        """
        result = data
        for transform in self.transforms:
            result = transform(result)
        return result
