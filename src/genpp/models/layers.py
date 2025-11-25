"""Custom neural network layers for the GenPP project."""

import math
from collections.abc import Sequence
from itertools import batched

import torch
import torch.nn as nn
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
    """A customizable UNet model.

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        channels (Sequence[int]): Number of channels at each encoder level.
            The decoder mirrors this structure. Default is (32, 64, 64) for backward compatibility.
        kernel_size (int): Kernel size for convolutions. Default is 3.
        padding_mode (str): Padding mode for convolutions. Default is "replicate".
        activation (nn.Module): Activation function to use. Default is nn.ReLU().
        use_batchnorm (bool): Whether to use batch normalization. Default is False.
        pool_type (str): Type of pooling to use ("max" or "avg"). Default is "max".

    Example:
        >>> # Default structure (backward compatible)
        >>> unet = UNet(in_features=3, out_features=1)
        >>> # Deeper network with more channels
        >>> unet = UNet(in_features=3, out_features=1, channels=(64, 128, 256, 512))
        >>> # Shallower network
        >>> unet = UNet(in_features=3, out_features=1, channels=(32, 64))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: nn.Module = nn.ReLU(),
        use_batchnorm: bool = False,
        pool_type: str = "max",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = list(channels)
        self.depth = len(channels)
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        if self.depth < 2:
            raise ValueError(
                "UNet requires at least 2 levels (channels must have at least 2 elements)"
            )

        padding = kernel_size // 2
        conv_kwargs = dict(kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        up_kwargs = dict(kernel_size=kernel_size, stride=2, padding=padding)

        # Pooling layer
        if pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(2)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use 'max' or 'avg'.")

        # Build encoder
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList() if use_batchnorm else None
        encoder_channels = [in_features] + self.channels
        for i in range(self.depth):
            self.encoders.append(
                nn.Conv2d(encoder_channels[i], encoder_channels[i + 1], **conv_kwargs)  # type: ignore
            )
            if use_batchnorm:
                self.encoder_norms.append(nn.BatchNorm2d(encoder_channels[i + 1]))  # type: ignore

        # Build decoder (mirrors encoder, but with skip connections)
        self.decoders = nn.ModuleList()
        self.decoder_norms = nn.ModuleList() if use_batchnorm else None

        # Decoder channels go from deepest to shallowest
        # reversed_channels = [channels[-1], channels[-2], ..., channels[0]]
        reversed_channels = list(reversed(self.channels))

        for i in range(self.depth - 1):
            if i == 0:
                # First decoder: input from bottleneck
                in_ch = reversed_channels[0]
            else:
                # Subsequent decoders: input is concat of skip + prev decoder output
                in_ch = reversed_channels[i] * 2
            out_ch = reversed_channels[i + 1]

            self.decoders.append(
                nn.ConvTranspose2d(in_ch, out_ch, **up_kwargs)  # type: ignore
            )
            if use_batchnorm:
                self.decoder_norms.append(nn.BatchNorm2d(out_ch))  # type: ignore

        # Final convolution after last skip connection
        # Input: skip from first encoder + last decoder output = channels[0] + channels[0]
        self.final_conv = nn.ConvTranspose2d(
            self.channels[0] * 2,
            self.channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.final_norm = nn.BatchNorm2d(self.channels[0]) if use_batchnorm else None

        # Prediction head
        self.predict = nn.Conv2d(
            self.channels[0],
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def _apply_encoder_block(self, x: Tensor, encoder: nn.Module, norm: nn.Module | None) -> Tensor:
        """Apply encoder block with optional batch norm and activation."""
        x = encoder(x)
        if norm is not None:
            x = norm(x)
        return self.activation(x)

    def _apply_decoder_block(
        self, x: Tensor, decoder: nn.Module, norm: nn.Module | None, output_size: torch.Size
    ) -> Tensor:
        """Apply decoder block with optional batch norm and activation."""
        x = decoder(x, output_size=output_size)
        if norm is not None:
            x = norm(x)
        return self.activation(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            x (torch.Tensor): Input image tensor. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape (B, out_features, H, W).
        """
        # Encoder path - store outputs for skip connections
        skips = []
        encoder_norms = self.encoder_norms or [None] * self.depth
        for i, (encoder, norm) in enumerate(zip(self.encoders, encoder_norms)):
            x = self._apply_encoder_block(x, encoder, norm)
            skips.append(x)
            if i < self.depth - 1:  # Don't pool after the last encoder
                x = self.pool(x)

        # Save first encoder output shape for final_conv
        first_encoder_shape = skips[0].shape

        # Decoder path
        # x is the bottleneck (last encoder output = skips[-1])
        # Pop the bottleneck since we already have it as x
        skips.pop()

        # Process decoder blocks, popping skip connections in reverse order
        decoder_norms = self.decoder_norms or [None] * (self.depth - 1)
        for decoder, norm in zip(self.decoders, decoder_norms):
            skip = skips.pop()
            x = self._apply_decoder_block(x, decoder, norm, skip.shape)
            x = torch.cat([skip, x], dim=1)

        # Final convolution with first encoder output shape
        x = self.final_conv(x, output_size=first_encoder_shape)
        if self.final_norm is not None:
            x = self.final_norm(x)
        x = self.activation(x)

        # Prediction
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
    This is essentially a wrapper around nn.Embedding.

    Args:
        in_dim (int): The dimensionality of the input space.
        out_dim (int): The dimensionality of the output space.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, pixel_idx: Tensor) -> Tensor:
        """Embeds the input tensor.

        Args:
            pixel_idx (Tensor): Input tensor of shape [b, 1, h, w].

        Returns:
            Tensor: Embedded tensor of shape [b, embedding_dim, h, w].
        """
        emb = self.emb(pixel_idx)
        emb = rearrange(emb, "b 1 h w c -> b c h w")
        return emb


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
        NOTE: when t is only a number (i.e. of shape []) then this number will first be transformed
              to a tensor of shape [1] before being passed to this function.
              The result will then have shape [1, dim] instead of the expected shape [dim].
              However this works perfectly well as the first dimension in used to broadcast later.
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
