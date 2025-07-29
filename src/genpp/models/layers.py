"""Custom neural network layers for the GenPP project."""

from typing import Tuple

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
        self.bias = nn.Parameter(torch.zeros(height, width, out_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LocallyConnected2D layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_features, height, width].

        Returns:
            Tensor: Output tensor of shape [batch_size, out_features, height, width].
        """
        # Perform the linear transformation for all spatial locations in parallel
        out = torch.einsum("bhwc,hwco->bhwo", x, self.weight) + self.bias
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


class Crop2D(nn.Module):
    """A simple cropping layer that crops the input tensor to a specified size.

    Args:
        target_size (Tuple[int, int, int, int]): The padding which was added to the input tensor.
        The padding is specified as (top, bottom, left, right).
    """

    def __init__(self, padding: Tuple[int, int, int, int]) -> None:
        super(Crop2D, self).__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Crops the input tensor to the target size.

        Args:
            x (Tensor): Input tensor of shape [..., height, width, channels].

        Returns:
            Tensor: Cropped tensor of shape [..., target_height, target_width, channels].
        """
        cropped = x[..., self.padding[0] : -self.padding[1], self.padding[2] : -self.padding[3], :]
        return cropped
