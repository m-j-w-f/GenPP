"""Custom neural network layers for the GenPP project."""

from itertools import batched
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LocallyConnected2D(nn.Module):
    """A custom layer that applies a separate linear transformation for each (height, width) location.

    Args:
        height (int): Height of the input feature map.
        width (int): Width of the input feature map.
        in_features (int): Number of input features per location.
        out_features (int): Number of output features per location.
    """

    def __init__(self, height: int, width: int, in_features: int, out_features: int) -> None:
        super(LocallyConnected2D, self).__init__()
        self.height = height
        self.width = width
        self.in_features = in_features
        self.out_features = out_features

        # Create a weight tensor for all spatial locations
        self.weight = nn.Parameter(torch.randn(height, width, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features, height, width))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LocallyConnected2D layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_features, height, width].

        Returns:
            Tensor: Output tensor of shape [batch_size, out_features, height, width].
        """
        # Perform the linear transformation for all spatial locations in parallel
        out = torch.einsum("bchw,hwco->bohw", x, self.weight) + self.bias
        return out


class UNet(nn.Module):
    """A simple UNet model."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super(UNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = dict(kernel_size=3, padding=1, padding_mode="replicate")
        up_kwargs = {"kernel_size": 3, "stride": 2, "padding": 1}

        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(self.in_features, 32, **kwargs)  # type: ignore
        self.conv2 = nn.Conv2d(32, 64, **kwargs)  # type: ignore
        self.conv3 = nn.Conv2d(64, 64, **kwargs)  # type: ignore

        self.upconv1 = nn.ConvTranspose2d(64, 64, **up_kwargs)  # type: ignore
        self.upconv2 = nn.ConvTranspose2d(
            128,
            32,
            **up_kwargs,  # type: ignore
        )  # 64 from upconv1 + 64 from conv2
        self.upconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=1
        )  # 32 from upconv2 + 32 from conv1

        self.predict = nn.Conv2d(
            32, self.out_features, kernel_size=3, padding=1, padding_mode="replicate"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for each class for each pixel in the input image.

        Args:
            x (torch.Tensor): Input image tensor. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits for each class for each pixel in the input image. Shape (B, 6, H, W).
        """
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))

        x2_up = F.relu(self.upconv1(x3, output_size=x2.shape))
        x2_up_cat = torch.cat([x2, x2_up], dim=1)

        x1_up = F.relu(self.upconv2(x2_up_cat, output_size=x1.shape))
        x1_up_cat = torch.cat([x1, x1_up], dim=1)

        x = F.relu(self.upconv3(x1_up_cat, output_size=x.shape))
        x = self.predict(x)
        return x


class CropND(nn.Module):
    """A simple cropping layer that crops the input tensor to a specified size.

    Args:
        padding (Sequence[int]): The padding to remove from the input tensor. The first two values correspond to the last dimension an so on.
        The padding is specified as (pad_lat_left, pad_lat_right, pad_lon_top, pad_lon_bottom).
    """

    def __init__(self, padding: Sequence[int]) -> None:
        super(CropND, self).__init__()
        if len(padding) % 2 != 0:
            raise ValueError("Padding sequence must have even length (pairs of left/right padding)")

        self.padding = padding

        self.spatial_slices = []
        for stop, start in batched(reversed(self.padding), 2):
            self.spatial_slices.append(slice(start, -stop))

    def forward(self, x: Tensor) -> Tensor:
        """Crops the input tensor to the target size.

        Args:
            x (Tensor): Input tensor of shape [..., height, width, channels].

        Returns:
            Tensor: Cropped tensor of shape [..., target_height, target_width, channels].
        """
        cropped = x[..., *self.spatial_slices]
        return cropped


class FinalActivation(nn.Module):
    """A final activation layer that applies a specified activation function.
    Splits the input tensor into multiple parts and applies the activation function to each part.
    The split is done along the last dimension.

    Args:
        activation (str): The activation function to use. Options are 'relu', 'sigmoid', 'tanh', 'softmax'.
    """

    def __init__(self, activations: List[torch.nn.Module]) -> None:
        super(FinalActivation, self).__init__()
        self.activations = activations
        self.len = len(activations)

    def __repr__(self) -> str:
        return f"FinalActivation(activations={self.activations})"

    def forward(self, x: Tensor) -> Tensor:
        x = torch.stack([act(x[:, :, i, ...]) for i, act in enumerate(self.activations)], dim=2)
        return x
