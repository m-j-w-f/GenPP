"""CNN-based Engression model for grid-based weather forecast post-processing.

This module implements a UNet-style stochastic neural network for engression,
adapted for spatial weather data on a grid.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import DictConfig

from genpp.models.engression.base import (
    EngressionModel,
    StochasticBackbone,
    StochasticLayer2D,
    StochasticResBlock2D,
)
from genpp.models.layers import PixelEmbedder


class StochasticEncoder(nn.Module):
    """Stochastic encoder block for UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        noise_channels (int): Number of noise channels. Defaults to 32.
        num_layers (int): Number of stochastic layers. Defaults to 2.
        use_resblock (bool): Whether to use residual blocks. Defaults to False.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_channels: int = 32,
        num_layers: int = 2,
        use_resblock: bool = False,
        kernel_size: int = 3,
        add_bn: bool = True,
    ) -> None:
        super().__init__()
        self.noise_channels = noise_channels

        layers: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(num_layers):
            if use_resblock and current_channels == out_channels:
                layers.append(
                    StochasticResBlock2D(
                        channels=current_channels,
                        noise_channels=noise_channels,
                        kernel_size=kernel_size,
                        add_bn=add_bn,
                    )
                )
            else:
                layers.append(
                    StochasticLayer2D(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        noise_channels=noise_channels,
                        kernel_size=kernel_size,
                        add_bn=add_bn,
                        activation=nn.ReLU(),
                    )
                )
                current_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (downsampled output, skip connection).
        """
        out = self.layers(x)
        return self.pool(out), out


class StochasticDecoder(nn.Module):
    """Stochastic decoder block for UNet.

    Args:
        in_channels (int): Number of input channels from previous layer.
        skip_channels (int): Number of channels in skip connection.
        out_channels (int): Number of output channels.
        noise_channels (int): Number of noise channels. Defaults to 32.
        num_layers (int): Number of stochastic layers. Defaults to 2.
        use_resblock (bool): Whether to use residual blocks. Defaults to False.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        noise_channels: int = 32,
        num_layers: int = 2,
        use_resblock: bool = False,
        kernel_size: int = 3,
        add_bn: bool = True,
    ) -> None:
        super().__init__()
        self.noise_channels = noise_channels

        # Upsample layer
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

        layers: list[nn.Module] = []
        current_channels = in_channels + skip_channels  # After concatenation with skip

        for _ in range(num_layers):
            if use_resblock and current_channels == out_channels:
                layers.append(
                    StochasticResBlock2D(
                        channels=current_channels,
                        noise_channels=noise_channels,
                        kernel_size=kernel_size,
                        add_bn=add_bn,
                    )
                )
            else:
                layers.append(
                    StochasticLayer2D(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        noise_channels=noise_channels,
                        kernel_size=kernel_size,
                        add_bn=add_bn,
                        activation=nn.ReLU(),
                    )
                )
                current_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width].
            skip (torch.Tensor): Skip connection tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.layers(x)


class StochasticUNet(StochasticBackbone):
    """Stochastic UNet backbone for engression.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (Sequence[int]): Channel dimensions at each level. Defaults to (32, 64, 128).
        noise_channels (int): Number of noise channels per layer. Defaults to 32.
        num_layers_per_block (int): Number of layers per encoder/decoder block. Defaults to 2.
        use_resblock (bool): Whether to use residual blocks. Defaults to False.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (32, 64, 128),
        noise_channels: int = 32,
        num_layers_per_block: int = 2,
        use_resblock: bool = False,
        kernel_size: int = 3,
        add_bn: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.noise_channels = noise_channels

        # Initial convolution
        self.init_conv = StochasticLayer2D(
            in_channels=in_channels,
            out_channels=channels[0],
            noise_channels=noise_channels,
            kernel_size=kernel_size,
            add_bn=add_bn,
            activation=nn.ReLU(),
        )

        # Encoders
        self.encoders = nn.ModuleList()
        encoder_channels = channels
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                StochasticEncoder(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    noise_channels=noise_channels,
                    num_layers=num_layers_per_block,
                    use_resblock=use_resblock,
                    kernel_size=kernel_size,
                    add_bn=add_bn,
                )
            )

        # Bottleneck
        self.bottleneck = StochasticResBlock2D(
            channels=channels[-1],
            noise_channels=noise_channels,
            kernel_size=kernel_size,
            add_bn=add_bn,
        )

        # Decoders
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(channels))
        for i in range(len(decoder_channels) - 1):
            # Input to decoder comes from previous decoder/bottleneck
            # Skip connection comes from corresponding encoder level
            # Skip channels = encoder output at that level = decoder_channels[i] (in reversed order)
            # But the first decoder gets skip from last encoder, which has channels[-1] = decoder_channels[0]
            # After that, decoder[i] gets skip from encoder with decoder_channels[i] channels
            skip_ch = decoder_channels[i]  # Skip from encoder at this level
            self.decoders.append(
                StochasticDecoder(
                    in_channels=decoder_channels[i],
                    skip_channels=skip_ch,
                    out_channels=decoder_channels[i + 1],
                    noise_channels=noise_channels,
                    num_layers=num_layers_per_block,
                    use_resblock=use_resblock,
                    kernel_size=kernel_size,
                    add_bn=add_bn,
                )
            )

        # Output convolution (deterministic)
        self.out_conv = nn.Conv2d(
            channels[0], out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stochastic UNet.

        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, height, width].
        """
        # Initial conv
        x = self.init_conv(x)

        # Encoder path
        skips = [x]
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        # skips[0] is from init_conv (not used for decoders in standard UNet)
        # skips[1:] are from encoders in order [encoder[0] output, encoder[1] output, ...]
        # We need them in reverse order for decoders
        encoder_skips = list(reversed(skips[1:]))  # [encoder[-1] output, ..., encoder[0] output]
        for decoder, skip in zip(self.decoders, encoder_skips):
            x = decoder(x, skip)

        # Output
        return self.out_conv(x)

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Generate multiple samples for the same input.

        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width].
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples [batch, n_samples, out_channels, height, width].
        """
        batch_size = x.shape[0]

        # Repeat input for each sample
        x_repeated = repeat(x, "b c h w -> (n b) c h w", n=n_samples)

        # Forward pass (noise is injected inside)
        out = self.forward(x_repeated)

        # Reshape back
        out = rearrange(out, "(n b) c h w -> b n c h w", n=n_samples, b=batch_size)
        return out


class CNNEngressionModel(EngressionModel):
    """CNN-based Engression model using a stochastic UNet backbone.

    This model combines the engression approach with a UNet architecture
    for grid-based weather forecast post-processing.

    Args:
        in_channels (int): Number of input channels (predicted + auxiliary + meta).
        out_channels (int): Number of output channels.
        height (int): Height of the input grid.
        width (int): Width of the input grid.
        embedding_dim (int): Dimension of pixel embeddings. Defaults to 5.
        channels (Sequence[int]): UNet channel dimensions. Defaults to (32, 64, 128).
        noise_channels (int): Number of noise channels per layer. Defaults to 32.
        num_layers_per_block (int): Number of layers per encoder/decoder block. Defaults to 2.
        use_resblock (bool): Whether to use residual blocks. Defaults to False.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        add_bn (bool): Whether to add batch normalization. Defaults to True.
        n_samples (int): Number of samples to generate. Defaults to 50.
        padding (Sequence[int]): Padding values for cropping output.
        optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory.
        lr_scheduler (DictConfig): Learning rate scheduler config.
        internal_td_scaling (str): TD scaling strategy.
        use_rescaler (bool): Whether to use rescaling modules.
        rescaler (Sequence[nn.Module | None] | nn.Module | None): Rescaling modules.
        loss_fn (nn.Module | None): Loss function. Defaults to EnergyScore.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        embedding_dim: int,
        channels: Sequence[int],
        noise_channels: int,
        num_layers_per_block: int,
        use_resblock: bool,
        kernel_size: int,
        add_bn: bool,
        n_samples: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        # Create pixel embedder
        self.use_embedding = embedding_dim > 0
        if self.use_embedding:
            self.pixel_embedder = PixelEmbedder(
                num_embeddings=height * width, embedding_dim=embedding_dim
            )
            total_in_channels = in_channels + embedding_dim
        else:
            self.pixel_embedder = None
            total_in_channels = in_channels

        # Create stochastic UNet backbone
        backbone = StochasticUNet(
            in_channels=total_in_channels,
            out_channels=out_channels,
            channels=channels,
            noise_channels=noise_channels,
            num_layers_per_block=num_layers_per_block,
            use_resblock=use_resblock,
            kernel_size=kernel_size,
            add_bn=add_bn,
        )

        super().__init__(
            backbone=backbone,
            n_samples=n_samples,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            internal_td_scaling=internal_td_scaling,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
            loss_fn=loss_fn,
        )

        # Store additional parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim

    def prepare_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare input for the backbone by concatenating all features.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary with:
                - predicted_vars: [batch, pred_channels, height, width]
                - auxiliary_vars: [batch, aux_channels, height, width]
                - meta_vars: [batch, meta_channels, height, width]
                - pixel_idx: [batch, 1, height, width]

        Returns:
            torch.Tensor: Concatenated input tensor.
        """
        inputs = [
            x["predicted_vars"],
            x["auxiliary_vars"],
            x["meta_vars"],
        ]

        if self.use_embedding:
            pixel_emb = self.pixel_embedder(x["pixel_idx"])
            inputs.append(pixel_emb)

        return torch.cat(inputs, dim=1)
