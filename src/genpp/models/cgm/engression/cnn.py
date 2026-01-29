"""CNN-based Engression model for grid-based weather forecast post-processing.

This module implements a UNet-style stochastic neural network for engression,
adapted for spatial weather data on a grid.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.cgm.engression.base import (
    BaseEngressionDirectModel,
    BaseEngressionNoiseModel,
    StochasticBackbone,
    StochasticLayer2D,
    StochasticResBlock2D,
)
from genpp.models.layers import FourierEncoder, PixelEmbedder
from genpp.models.scores import EnergyScore


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
        channels: Sequence[int] = (32, 64),
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
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder in self.decoders:
            x = decoder(x, skips.pop())

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


class CNNEngressionNoiseModel(BaseEngressionNoiseModel):
    """CNN-based Engression model using a stochastic UNet backbone with noise prediction.

    This model combines the engression approach with a UNet architecture
    for grid-based weather forecast post-processing. It predicts deviations
    from the NWP forecast with internal TD scaling.
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
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        loss_fn: nn.Module = EnergyScore(),
        n_samples: int | None = None,
        n_samples_train: int | None = None,
        n_samples_predict: int | None = None,
    ) -> None:
        """
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
            padding (Sequence[int]): Padding values for cropping output.
            optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory.
            lr_scheduler (DictConfig): Learning rate scheduler config.
            internal_td_scaling (str): TD scaling strategy.
            use_rescaler (bool): Whether to use rescaling modules.
            rescaler (Sequence[nn.Module | None] | nn.Module | None): Rescaling modules.
            loss_fn (nn.Module | None): Loss function. Defaults to EnergyScore.
            n_samples (int | None): Number of samples to generate. Defaults to None.
            n_samples_train (int | None): Number of samples during training. If None, defaults to n_samples.
            n_samples_predict (int | None): Number of samples during prediction. If None, defaults to n_samples.
        """
        self.save_hyperparameters()
        # Calculate total input channels
        use_embedding = embedding_dim > 0
        if use_embedding:
            total_in_channels = in_channels + embedding_dim
        else:
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
            height=height,
            width=width,
            out_channels=out_channels,
            backbone=backbone,
            n_samples=n_samples,
            n_samples_train=n_samples_train,
            n_samples_predict=n_samples_predict,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            internal_td_scaling=internal_td_scaling,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
            loss_fn=loss_fn,
        )

        # Create pixel embedder after super().__init__() call
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.pixel_embedder = PixelEmbedder(
                num_embeddings=height * width, embedding_dim=embedding_dim
            )
        else:
            self.pixel_embedder = None

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
            pixel_emb = self.pixel_embedder(x["pixel_idx"])  # type: ignore
            inputs.append(pixel_emb)
        inputs_concat = torch.cat(inputs, dim=1)
        return inputs_concat


class CNNEngressionDirectModel(BaseEngressionDirectModel):
    """CNN-based Engression model with direct prediction and timedelta encoding.

    This model directly predicts target values without using internal TD scaling.
    The timedelta is encoded using a FourierEncoder and added to the model input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        embedding_dim: int,
        td_embedding_dim: int = 8,
        channels: Sequence[int] = (32, 64, 128),
        noise_channels: int = 32,
        num_layers_per_block: int = 2,
        use_resblock: bool = False,
        kernel_size: int = 3,
        add_bn: bool = True,
        padding: Sequence[int] = (0, 0, 0, 0),
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: DictConfig | None = None,
        use_rescaler: bool = False,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        loss_fn: nn.Module = EnergyScore(),
        n_samples: int | None = None,
        n_samples_train: int | None = None,
        n_samples_predict: int | None = None,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels (predicted + auxiliary + meta).
            out_channels (int): Number of output channels.
            height (int): Height of the input grid.
            width (int): Width of the input grid.
            embedding_dim (int): Dimension of pixel embeddings. Defaults to 5.
            td_embedding_dim (int): Dimension of timedelta encoding. Defaults to 8.
            channels (Sequence[int]): UNet channel dimensions. Defaults to (32, 64, 128).
            noise_channels (int): Number of noise channels per layer. Defaults to 32.
            num_layers_per_block (int): Number of layers per encoder/decoder block. Defaults to 2.
            use_resblock (bool): Whether to use residual blocks. Defaults to False.
            kernel_size (int): Kernel size for convolutions. Defaults to 3.
            add_bn (bool): Whether to add batch normalization. Defaults to True.
            padding (Sequence[int]): Padding values for cropping output.
            optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory.
            lr_scheduler (DictConfig): Learning rate scheduler config.
            use_rescaler (bool): Whether to use rescaling modules.
            rescaler (Sequence[nn.Module | None] | nn.Module | None): Rescaling modules.
            loss_fn (nn.Module | None): Loss function. Defaults to EnergyScore.
            n_samples (int | None): Number of samples to generate. Defaults to None.
            n_samples_train (int | None): Number of samples during training. If None, defaults to n_samples.
            n_samples_predict (int | None): Number of samples during prediction. If None, defaults
        """
        self.save_hyperparameters()
        # Check required parameters
        if optimizer is None:
            raise ValueError("optimizer is required for CNNEngressionDirectModel")
        if lr_scheduler is None:
            raise ValueError("lr_scheduler is required for CNNEngressionDirectModel")

        # Calculate dimensions (don't create modules yet)
        if td_embedding_dim > 0:
            td_embedding_dim_value = td_embedding_dim
        elif td_embedding_dim == 0:
            td_embedding_dim_value = 1
        else:
            raise ValueError("td_embedding_dim must be >= 0")

        # Calculate total input channels
        use_embedding_value = embedding_dim > 0
        if use_embedding_value:
            total_in_channels = in_channels + embedding_dim + td_embedding_dim_value
        else:
            total_in_channels = in_channels + td_embedding_dim_value

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

        # Call super().__init__() before assigning any module attributes
        super().__init__(
            height=height,
            width=width,
            out_channels=out_channels,
            backbone=backbone,
            padding=padding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_rescaler=use_rescaler,
            rescaler=rescaler,
            loss_fn=loss_fn,
            n_samples=n_samples,
            n_samples_train=n_samples_train,
            n_samples_predict=n_samples_predict,
        )

        # NOW assign instance variables and module attributes after super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim
        self.use_embedding = use_embedding_value
        self.td_embedding_dim = td_embedding_dim_value

        # Create timedelta encoder module
        if td_embedding_dim > 0:
            self.td_encoder = FourierEncoder(dim=td_embedding_dim)
        elif td_embedding_dim == 0:
            self.td_encoder = Rearrange("b -> b 1")

        # Create pixel embedder module
        if self.use_embedding:
            self.pixel_embedder = PixelEmbedder(
                num_embeddings=height * width, embedding_dim=self.embedding_dim
            )
        else:
            self.pixel_embedder = None

    def prepare_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare input for the backbone by concatenating all feature tensors.

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
            pixel_emb = self.pixel_embedder(x["pixel_idx"])  # type: ignore
            inputs.append(pixel_emb)

        inputs_concat = torch.cat(inputs, dim=1)
        return inputs_concat

    def forward(self, x: dict[str, torch.Tensor], td: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Forward pass through the model with direct prediction.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary.
            td (torch.Tensor): Time delta tensor.
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Output tensor of shape [batch, n_samples, out_features, height, width].
        """
        # Prepare base input
        backbone_input = self.prepare_input(x)

        # Encode timedelta and expand spatially
        enc_timedelta = self.td_encoder(td)  # [batch, td_encoding_dim]
        *_, h, w = x["predicted_vars"].shape
        enc_timedelta = enc_timedelta[..., None, None].expand(
            -1, -1, h, w
        )  # [batch, td_encoding_dim, height, width]

        # Concatenate with timedelta encoding
        backbone_input = torch.cat([backbone_input, enc_timedelta], dim=1)

        # Generate samples using the stochastic backbone
        samples = self.backbone.sample(backbone_input, n_samples)

        # Residual Connection
        means = x["predicted_vars"].unsqueeze(1)  # [batch, 1, out_channels, height, width]
        res = means + samples

        # Crop padding
        res = self.crop(res)
        return res


# Backwards compatibility alias
CNNEngressionModel = CNNEngressionNoiseModel
