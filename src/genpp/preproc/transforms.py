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


class PermuteChannel(Transform):
    """Permute the spatial values of a single channel for feature importance analysis.

    Shuffles the flattened spatial grid of the specified channel, breaking spatial
    structure while preserving the marginal distribution. Useful for permutation-based
    feature importance assessment.

    Args:
        channel_index (int): Index of the feature channel to permute.
        seed (int | None): Optional random seed for reproducibility.
    """

    def __init__(self, channel_index: int, seed: int | None = None):
        self.channel_index = channel_index
        self.seed = seed

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Permute spatial dimensions of the specified channel.

        For 4D inputs, each batch element receives an independent random
        permutation. When a seed is set, the generator is seeded once and
        successive batch elements consume sequential random states, so results
        are reproducible but differ across the batch dimension.

        Args:
            data: Tensor with shape (feature, longitude, latitude) or
                  (batch, feature, longitude, latitude).

        Returns:
            torch.Tensor: Tensor with the specified channel's spatial values shuffled.
        """
        result = data.clone()
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)

        if data.ndim == 3:
            # (feature, lon, lat)
            channel = result[self.channel_index]
            flat = channel.flatten()
            perm = torch.randperm(flat.numel(), generator=generator)
            result[self.channel_index] = flat[perm].reshape(channel.shape)
        elif data.ndim == 4:
            # (batch, feature, lon, lat)
            for i in range(result.shape[0]):
                channel = result[i, self.channel_index]
                flat = channel.flatten()
                perm = torch.randperm(flat.numel(), generator=generator)
                result[i, self.channel_index] = flat[perm].reshape(channel.shape)
        else:
            raise ValueError(
                f"Expected 3D (feature, lon, lat) or 4D (batch, feature, lon, lat) tensor, "
                f"got {data.ndim}D."
            )
        return result


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
