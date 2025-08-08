from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn.functional as F
import xarray as xr
from einops import rearrange


class Transform(ABC):
    """Abstract base class for data transformations.
    These transformations can be supplied to the xbatcher.MapDataset
    to apply on the fly to the data."""

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


class Pad(Transform):
    """
    A callable class to pad a n-D tensor of shape (B, H, W, ...)
    on the H and W dimensions to a target size. It is assume that H and W stay constant.

    Args:
        padding (Tuple[int, int, int, int]): the sizes to pad on the left, right, top, and bottom.
        mode (str): The padding mode to use. Accepts "constant", "reflect",
            "replicate", or "circular". Defaults to "reflect".
    """

    def __init__(self, padding: Tuple[int, int, int, int], mode: str = "reflect"):
        self.padding = padding
        self.mode = mode

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor's spatial dimensions (H, W).

        Args:
            data (torch.Tensor): The input tensor with shape
                (batch, height, width, channels).

        Returns:
            torch.Tensor: The padded tensor with shape
                (batch, target_height, target_width, channels).
        """
        b, h, w, c = data.shape

        if self.mode == "constant":  # constant padding works for any dimension
            data_reshaped = rearrange(data, "b h w ... -> b ... h w")
            data_padded = F.pad(data_reshaped, self.padding, mode=self.mode)
            output = rearrange(data_padded, "b ... h w -> b h w ...")
            return output
        else:
            # torch.nn.functional.pad works on the last dimensions of a tensor.
            data_reshaped = rearrange(data, "b h w c -> (b c) h w")
            data_padded = F.pad(data_reshaped, self.padding, mode=self.mode)
            output = rearrange(data_padded, "(b c) h w -> b h w c", b=b, c=c)

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
