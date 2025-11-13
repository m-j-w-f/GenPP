from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from genpp.models.fm.base import FlowMatchingModel
from genpp.models.fm.helpers import ConditionalVectorField
from genpp.models.layers import FourierEncoder, PixelEmbedder


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, channels_conditioning: int, time_embed_dim: int, depth: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # Converts [bs, time_embed_dim] -> [bs, channels]
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels),
        )
        self.conditioning_adapter = nn.Sequential(  # the context adapter should put context in the same shape as x in terms of c, h, w
            nn.Conv2d(channels_conditioning, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            # Add downsampling layers based on depth
            *[
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                )
                for _ in range(depth)
            ],
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The sampled noise [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            conditioning (torch.Tensor): The conditioning tensor [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c, h, w]
        """
        res = x.clone()  # [bs, c, h, w]

        # Initial conv block
        x = self.block1(x)  # [bs, c, h, w]

        # Add time embedding
        t_embed = self.time_adapter(t_embed)
        t_embed = rearrange(t_embed, "bs c -> bs c 1 1")
        x = x + t_embed

        # Add conditioning
        conditioning_embed = self.conditioning_adapter(conditioning)  # [bs, c, h, w]
        x = x + conditioning_embed

        # Second conv block
        x = self.block2(x)  # [bs, c, h, w]

        # Add back residual
        x = x + res  # [bs, c, h, w]

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        channels_conditioning: int,
        num_residual_layers: int,
        t_embed_dim: int,
        depth: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(
                    channels=channels_in,
                    channels_conditioning=channels_conditioning,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c_in, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            conditioning (torch.Tensor): Conditioning tensor [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c_out, h // 2, w // 2]
        """
        # Pass through residual blocks: [bs, c_in, h, w] -> [bs, c_in, h, w]
        for block in self.res_blocks:
            x = block(x, t_embed, conditioning)

        # Downsample: [bs, c_in, h, w] -> [bs, c_out, h // 2, w // 2]
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        t_embed_dim: int,
        channels_conditioning: int,
        depth: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(
                    channels=channels,
                    channels_conditioning=channels_conditioning,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            conditioning (torch.Tensor): Conditioning tensor [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c, h, w]
        """
        # Pass through residual blocks: [bs, c, h, w] -> [bs, c, h, w]
        for block in self.res_blocks:
            x = block(x, t_embed, conditioning)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        channels_conditioning: int,
        num_residual_layers: int,
        t_embed_dim: int,
        depth: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(
                    channels=channels_out,
                    channels_conditioning=channels_conditioning,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            conditioning (torch.Tensor): [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c_out, 2 * h, 2 * w]
        """
        # Upsample: [bs, c_in, h, w] -> [bs, c_out, 2 * h, 2 * w]
        x = self.upsample(x)

        # Pass through residual blocks: [bs, c_out, h, w] -> [bs, c_out, 2 * h, 2 * w]
        for block in self.res_blocks:
            x = block(x, t_embed, conditioning)

        return x


class _FMUNet(ConditionalVectorField):
    def __init__(
        self,
        channels: Sequence[int],
        num_residual_layers: int,
        t_embed_dim: int,
        embedding_dim: int,
        height: int,
        width: int,
        channels_conditioning: int,
        channels_x: int = 2,
    ):
        super().__init__()
        # Initial convolution: [bs, 2, 32, 32] -> [bs, c_0, 32, 32]
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels_x, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Embed the Pixel IDX
        # Note that conditioning now has the channels channels_conditioning + pixel_embed_dim
        self.conditioning_embedder = PixelEmbedder(
            num_embeddings=height * width, embedding_dim=embedding_dim
        )
        # Adjust channels_conditioning to account for pixel embedding
        channels_conditioning += embedding_dim

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        depth = 0
        for depth, (curr_c, next_c) in enumerate(zip(channels[:-1], channels[1:])):
            encoders.append(
                Encoder(
                    channels_in=curr_c,
                    channels_out=next_c,
                    channels_conditioning=channels_conditioning,
                    num_residual_layers=num_residual_layers,
                    t_embed_dim=t_embed_dim,
                    depth=depth,
                )
            )
            decoders.append(
                Decoder(
                    channels_in=next_c,
                    channels_out=curr_c,
                    channels_conditioning=channels_conditioning,
                    num_residual_layers=num_residual_layers,
                    t_embed_dim=t_embed_dim,
                    depth=depth,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(
            channels=channels[-1],
            num_residual_layers=num_residual_layers,
            t_embed_dim=t_embed_dim,
            channels_conditioning=channels_conditioning,
            depth=depth + 1,
        )

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], channels_x, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: dict[str, torch.Tensor]):
        """
        Args:
            x (torch.Tensor): [bs, 2, 48, 32]
            t (torch.Tensor): [bs, 1, 1, 1]
            conditioning (torch.Tensor): [bs, 50, 48, 32]
        Returns:
            torch.Tensor: u_t^theta(x|conditioning) [bs, 2, 48, 32]
        """
        # Embed t and conditioning
        t_embed = self.time_embedder(t)  # [bs, time_embed_dim]
        conditioning_embed = torch.cat(
            [
                conditioning["predicted_vars"],
                conditioning["auxiliary_vars"],
                conditioning["meta_vars"],
                self.conditioning_embedder(conditioning["pixel_idx"]),
            ],
            dim=1,
        )  # [bs, c_conditioning, 48, 32] most likely c_conditioning = 2 + 56 + 4 + 5 = 67

        # Initial convolution
        x = self.init_conv(x)  # [bs, c_0, 48, 32]

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(
                x, t_embed, conditioning_embed
            )  # [bs, c_i, h, w] -> [bs, c_{i+1}, h // 2, w //2]
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, conditioning_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()  # [bs, c_i, h, w]
            x = x + res
            x = decoder(
                x, t_embed, conditioning_embed
            )  # [bs, c_i, h, w] -> [bs, c_{i-1}, 2 * h, 2 * w]

        # Final convolution
        x = self.final_conv(x)  # [bs, 1, 48, 32]

        return x


def FMUNet(
    channels: list[int],
    num_residual_layers: int,
    t_embed_dim: int,
    embedding_dim: int,
    height: int,
    width: int,
    channels_conditioning: int,
    channels_x: int,
    n_samples: int,
    solver_iter: int,
    padding: Sequence[int],
    optimizer: Callable[..., torch.optim.Optimizer],
    lr_scheduler: DictConfig,
    use_rescaler: bool,
    rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
) -> FlowMatchingModel:
    """
    Factory function to create a FlowMatchingModel with a UNet backbone.
    
    NOTE that in this class the naming convention is different than in the other classes:
    - x_1 is the target (i.e. the ground truth forecasts, for which we want to generate samples that are similar to)
    - the conditioning is the nwp forecast(s)
    - td is the lead time (between 0 and 1) for which the prediction is made

    How the prediction works:
    - Instead of generating samples similar to the ground truth directly, we want to sample the deviation (x_1 - nwp_fc)
    - Then these sampled deviations are added to the nwp forecasts to get the final samples
    - Also the deviations are scaled according to the lead time. The scaling factor is learned via linear regression.
    """
    backbone = _FMUNet(
        channels=channels,
        num_residual_layers=num_residual_layers,
        t_embed_dim=t_embed_dim,
        channels_conditioning=channels_conditioning,
        channels_x=channels_x,
        height=height,
        width=width,
        embedding_dim=embedding_dim,
    )
    
    return FlowMatchingModel(
        backbone=backbone,
        n_samples=n_samples,
        solver_iter=solver_iter,
        padding=padding,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_rescaler=use_rescaler,
        rescaler=rescaler,
    )
