from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import ListConfig, OmegaConf


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
    Wrapper around torch.nn.functional.pad to pad the spatial dimensions of a tensor.

    Args:
        padding (Tuple[int, int, int, int]): padding size (pad_lat_left, pad_lat_right, pad_lon_top, pad_lon_bottom).
        mode (str): The padding mode to use. Accepts "constant", "reflect",
            "replicate", or "circular". Defaults to "reflect".
    """

    def __init__(self, padding: Sequence[int], mode: str = "reflect"):
        if isinstance(padding, ListConfig):
            self.padding: Sequence[int] = OmegaConf.to_object(padding)  # type: ignore
        else:
            self.padding = padding
        self.mode = mode

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor's spatial dimensions (H, W).

        Args:
            data (torch.Tensor): The input tensor with shape
                (batch, channels, lon, lat).

        Returns:
            torch.Tensor: The padded tensor with shape
                (batch, target_height, target_width, channels).
        """
        return F.pad(data, self.padding)


class Pipe(Transform):
    """A pipeline that chains multiple transforms together.

    Args:
        transforms (List[Transform]): A list of transforms to apply in sequence.
    """

    def __init__(self, transforms: list[Transform]) -> None:
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
