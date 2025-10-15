from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from omegaconf import DictConfig

from genpp.models.layers import CropND, FourierEncoder, PixelEmbedder, _get_scale_td
from genpp.models.utils import BaseModule


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, channels_y: int, time_embed_dim: int, depth: int):
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
        self.y_adapter = (
            nn.Sequential(  # the y adapter should put y in the same shape as x in terms of c, h, w
                nn.Conv2d(channels_y, channels, kernel_size=1),
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
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The sampled noise [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            y (torch.Tensor): The conditioning tensor [bs, cy, h, w]
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

        # Add y (conditioning)
        y_embed = self.y_adapter(y)  # [bs, c, h, w]
        x = x + y_embed

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
        channels_y: int,
        num_residual_layers: int,
        t_embed_dim: int,
        depth: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(
                    channels=channels_in,
                    channels_y=channels_y,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c_in, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            y (torch.Tensor): Conditioning tensor [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c_out, h // 2, w // 2]
        """
        # Pass through residual blocks: [bs, c_in, h, w] -> [bs, c_in, h, w]
        for block in self.res_blocks:
            x = block(x, t_embed, y)

        # Downsample: [bs, c_in, h, w] -> [bs, c_out, h // 2, w // 2]
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(
        self, channels: int, num_residual_layers: int, t_embed_dim: int, channels_y: int, depth: int
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(
                    channels=channels,
                    channels_y=channels_y,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            y (torch.Tensor): Conditioning tensor [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c, h, w]
        """
        # Pass through residual blocks: [bs, c, h, w] -> [bs, c, h, w]
        for block in self.res_blocks:
            x = block(x, t_embed, y)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        channels_y: int,
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
                    channels_y=channels_y,
                    time_embed_dim=t_embed_dim,
                    depth=depth,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t_embed (torch.Tensor): [bs, t_embed_dim]
            y (torch.Tensor): [bs, cy, h, w]
        Returns:
            torch.Tensor: [bs, c_out, 2 * h, 2 * w]
        """
        # Upsample: [bs, c_in, h, w] -> [bs, c_out, 2 * h, 2 * w]
        x = self.upsample(x)

        # Pass through residual blocks: [bs, c_out, h, w] -> [bs, c_out, 2 * h, 2 * w]
        for block in self.res_blocks:
            x = block(x, t_embed, y)

        return x


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [bs, c, h, w]
            t (torch.Tensor): [bs, 1, 1, 1]
            y (dict[str, torch.Tensor]): [bs,...]
        Returns:
            torch.Tensor: u_t^theta(x|y) [bs, c, h, w]
        """
        pass


class _FMUNet(ConditionalVectorField):
    def __init__(
        self,
        channels: Sequence[int],
        num_residual_layers: int,
        t_embed_dim: int,
        embedding_dim: int,
        height: int,
        width: int,
        channels_y: int,
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
        # Note that y now has the channels channels_y + pixel_embed_dim
        self.y_embedder = PixelEmbedder(num_embeddings=height * width, embedding_dim=embedding_dim)
        # Adjust channels_y to account for pixel embedding
        channels_y += embedding_dim

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        depth = 0
        for depth, (curr_c, next_c) in enumerate(zip(channels[:-1], channels[1:])):
            encoders.append(
                Encoder(
                    channels_in=curr_c,
                    channels_out=next_c,
                    channels_y=channels_y,
                    num_residual_layers=num_residual_layers,
                    t_embed_dim=t_embed_dim,
                    depth=depth,
                )
            )
            decoders.append(
                Decoder(
                    channels_in=next_c,
                    channels_out=curr_c,
                    channels_y=channels_y,
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
            channels_y=channels_y,
            depth=depth + 1,
        )

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], channels_x, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: dict[str, torch.Tensor]):
        """
        Args:
            x (torch.Tensor): [bs, 2, 48, 32]
            t (torch.Tensor): [bs, 1, 1, 1]
            y (torch.Tensor): [bs, 50, 48, 32]
        Returns:
            torch.Tensor: u_t^theta(x|y) [bs, 2, 48, 32]
        """
        # Embed t and y
        t_embed = self.time_embedder(t)  # [bs, time_embed_dim]
        y_embed = torch.cat(
            [
                y["predicted_vars"],
                y["auxiliary_vars"],
                y["meta_vars"],
                self.y_embedder(y["pixel_idx"]),
            ],
            dim=1,
        )  # [bs, c_y, 48, 32] most likely c_y = 2 + 56 + 4 + 5 = 67

        # Initial convolution
        x = self.init_conv(x)  # [bs, c_0, 48, 32]

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)  # [bs, c_i, h, w] -> [bs, c_{i+1}, h // 2, w //2]
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()  # [bs, c_i, h, w]
            x = x + res
            x = decoder(x, t_embed, y_embed)  # [bs, c_i, h, w] -> [bs, c_{i-1}, 2 * h, 2 * w]

        # Final convolution
        x = self.final_conv(x)  # [bs, 1, 48, 32]

        return x


class FMUNet(BaseModule):
    """
    NOTE that in this class the naming convention is different than in the other classes:
    - x_1 is the target (i.e. the ground truth forecasts, for which we want to generate samples that are similar to)
    - y is the conditioning (i.e. the nwp forecasts)
    - td is the lead time (between 0 and 1) for which the prediction is made

    How the prediction works:
    - Instead of generating samples similar to the ground truth directly, we want to sample the deviation (x_1 - y)
    - Then these sampled deviations are added to the nwp forecasts y to get the final samples
    - Also the deviations are scaled according to the lead time. The scaling factor is learned via linear regression.
    """

    def __init__(
        self,
        channels: list[int],
        num_residual_layers: int,
        t_embed_dim: int,
        embedding_dim: int,
        height: int,
        width: int,
        channels_y: int,
        channels_x: int,
        n_samples: int,
        solver_iter: int,
        padding: Sequence[int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        use_rescaler: bool,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ):
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        # Loss FN is a nn.module so it does not need to be saved explicitly
        self.save_hyperparameters()
        if use_rescaler:
            # TODO implement rescaling
            if isinstance(rescaler, Sequence):
                filtered = [m for m in rescaler if m is not None]
                self.rescaler = nn.ModuleList(filtered) if filtered else None
        self.model = _FMUNet(
            channels=channels,
            num_residual_layers=num_residual_layers,
            t_embed_dim=t_embed_dim,
            channels_y=channels_y,
            channels_x=channels_x,
            height=height,
            width=width,
            embedding_dim=embedding_dim,
        )
        self.n_samples = n_samples
        self.padding = padding
        self.crop = CropND(padding=padding) if padding else nn.Identity()
        self.path = CondOTProbPath()
        self.solver = ODESolver(self.model)
        self.step_size = 1 / solver_iter

        self.register_buffer("scale_variance_td", None)  # To be fitted via callback

        self.num_predicted_vars = 2  # TODO in the PR for the improved dataloader fix this

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: dict[str, torch.Tensor]):
        """
        Args:
            x (torch.Tensor): the (noisy) input [bs, 2, 48, 32]
            t (torch.Tensor): the timestep [bs, 1, 1, 1]
            y (dict[str, torch.Tensor]): the conditioning dict with tensors of shape [bs, ...]
        Returns:
            torch.Tensor: [bs, 2, 48, 32], h and w dim might be cropped
        """
        res = self.model(x, t, y)
        return self.crop(res)

    def _calc_loss(self, batch) -> torch.Tensor:
        # Sample Data (X_0,X_1) ~ π(X_0,X_1) = N(X_0|0,I)q(X_1)
        nwp_fc, ground_truth, td = batch["x"], batch["y"], batch["timedelta"]
        # We want to predict the errors of the NWP forecasts
        # Now x_1 contains the errors
        x_1 = ground_truth - nwp_fc["predicted_vars"]
        # x_1 should always have roughly the same magnitude, independent of the lead time
        scale = _get_scale_td(
            td=td,
            betas=self.scale_variance_td,  # type: ignore
        )  # Shape [b, n_vars, 1, 1]
        # Now x_1 contains the scaled errors, the model has to learn only one scale
        # NOTE the predicted noise needs to be scaled back during inference
        # to get the actual noise that is added to the NWP forecasts
        x_1 = x_1 / scale

        # Sample x_0 ~ N(0,I)
        x_0 = torch.randn_like(x_1).to(x_1)

        # Sample a random timestep
        t = torch.rand(x_1.size(0)).to(x_1)

        # Get the probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        u_t_theta = self.model(x=path_sample.x_t, t=path_sample.t, y=nwp_fc)
        u_t_ref = path_sample.dx_t

        # Calc the l2 loss
        loss = torch.pow(u_t_theta - u_t_ref, 2)
        return loss

    def training_step(self, batch) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = loss.mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch) -> torch.Tensor:
        y, x_1, td = batch["x"], batch["y"], batch["timedelta"]

        ens_mean = rearrange(y["predicted_vars"], "b c h w -> b 1 c h w")

        # repeat shapes to be able to generate 50 different samples
        for k, v in y.items():
            y[k] = repeat(v, "b ... -> (n_samples b) ...", n_samples=self.n_samples)

        # Sample n_samples * batch size random images. Keep the other dimensions as x_1
        x_init = torch.randn(y["predicted_vars"].size(0), *x_1.shape[1:]).to(x_1)

        sol = self.solver.sample(
            x_init=x_init,
            y=y,
            method="midpoint",
            step_size=self.step_size,
        )
        # Sol now contains the deviations that need to be added to the nwp forecasts
        sol = rearrange(sol, "(n_samples b) ... -> b n_samples ...", n_samples=self.n_samples)

        # Calculate the scale factor based on the lead time
        scale = _get_scale_td(
            td=td,
            betas=self.scale_variance_td,  # type: ignore
        )  # Shape [b, n_vars, 1, 1]
        # Rescale the deviations according to the lead time (inverse of what was done during training)
        sol = sol * rearrange(scale, "b n_vars 1 1 -> b 1 n_vars 1 1")  # type: ignore
        res = ens_mean + sol  # Add the nwp forecasts to the deviations to get the final samples
        res_cropped = self.crop(res)
        return res_cropped

    def validation_step(self, batch) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = reduce(loss, "b c h w -> c", "mean")  # How good are we per channel
        for i, lo in enumerate(loss):
            self.log(
                f"val_loss_var_{i}",
                lo,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss = torch.mean(loss)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch) -> torch.Tensor:
        loss = self._calc_loss(batch)
        loss = reduce(loss, "b c h w -> c", "mean")  # How good are we per channel
        for i, lo in enumerate(loss):
            self.log(
                f"test_loss_var_{i}",
                lo,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss = torch.mean(loss)
        self.log("test_loss", loss)
        return loss
