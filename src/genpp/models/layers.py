"""Custom neural network layers for the GenPP project."""

import math
from collections.abc import Sequence
from itertools import batched

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        if len(padding) % 2 != 0:
            raise ValueError("Padding sequence must have even length (pairs of left/right padding)")

        self.padding = padding

        self.spatial_slices = []
        for stop, start in batched(reversed(self.padding), 2):
            stop = None if stop == 0 else -stop
            self.spatial_slices.append(slice(start, stop))

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

    def __init__(self, activations: list[torch.nn.Module], split_dim: int = 2) -> None:
        super().__init__()
        self.activations = activations
        self.split_dim = split_dim

    def __repr__(self) -> str:
        return f"FinalActivation(activations={self.activations})"

    def forward(self, x: Tensor) -> Tensor:
        # shape of the input tensor is [b, n, c, ...]
        x_list = torch.split(x, 1, dim=self.split_dim)
        x = torch.cat([act(x) for act, x in zip(self.activations, x_list)], dim=self.split_dim)
        return x


class ReverseAffineTransform(nn.Module):
    """A layer that reverses the affine transformation of the input tensor.
    This should be applied to the output of the model, so that we can compare the crps in the original space.

    Args:
        mean (torch.Tensor): The mean tensor used for standardization.
        std (torch.Tensor): The standard deviation tensor used for standardization.
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, mu: Tensor, sigma: Tensor | None) -> tuple[Tensor, Tensor] | Tensor:
        """Reverses the standardization of the input tensor.

        Args:
            mu (Tensor): Input tensor of shape [b, c, ...].
            sigma (Tensor): Optional input tensor of shape [b, c, ...].


        Returns:
            Tensor: Tensor with the same shape as the input, but with standardization reversed.
        """
        mu = mu * self.scale + self.mean  # type: ignore
        if sigma is None:
            return mu
        sigma = sigma * self.scale  # type: ignore
        return mu, sigma


class PixelEmbedder(nn.Module):
    """A layer that embeds the input tensor into a higher-dimensional space.

    Args:
        in_dim (int): The dimensionality of the input space.
        out_dim (int): The dimensionality of the output space.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Embeds the input tensor.

        Args:
            x (Tensor): Input tensor of shape [b, c, h, w]. Where the last channel carries the indexes to be embedded and appended.

        Returns:
            Tensor: Embedded tensor of shape [b, c + embedding_dim - 1, h, w].
        """
        pixel_idx = x[:, -1].long()
        x = x[:, :-1]
        emb = self.emb(pixel_idx)
        emb = rearrange(emb, "b h w c -> b c h w")
        x = torch.cat([x, emb], dim=1)

        return x


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): [...], typically [bs]
        Returns:
            torch.Tensor: [..., dim]
        """
        t = rearrange(t, "... -> (...) 1")  # [..., 1]
        freqs = t * self.weights * 2 * math.pi  # [..., half_dim]
        sin_embed = torch.sin(freqs)  # [..., half_dim]
        cos_embed = torch.cos(freqs)  # [..., half_dim]
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # [..., dim]


def _get_scale_td(td: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """Get the scale tensor based on the the time delta.

    Args:
        td (torch.Tensor): The time delta input tensor of shape [batch_size].
        betas (torch.Tensor | None): The beta parameters of shape [n_vars, 2] or None.

    Raises:
        ValueError: If betas is None. This is placed inside this function to make the model code cleaner.

    Returns:
        torch.Tensor: The scale tensor of shape [b, n_vars, 1, 1].
    """
    if betas is None:
        raise ValueError(
            "scale_variance_td is not fitted yet. Please run the 'fit_scale_variance_td' callback first."
        )
    # Ensure betas is on the same device as td
    betas = betas.to(td)
    intercepts_scale_variance_td = betas[:, 0]  # Shape [n_vars]
    betas_scale_variance_td = betas[:, 1]  # Shape [n_vars]
    scale = rearrange(intercepts_scale_variance_td, "c -> 1 c") + rearrange(
        betas_scale_variance_td, "c -> 1 c"
    ) * rearrange(td, "b -> b 1")  # Shape [b, n_vars]
    scale = rearrange(scale, "b c -> b c 1 1")  # Shape [b, n_vars, 1, 1]
    return scale
