from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from tqdm import tqdm

from genpp.models.layers import CropND
from genpp.models.utils import BaseModule


class AutoEncoder(BaseModule):
    def __init__(
        self,
        in_channels: int,
        padding: tuple[int, int, int, int],
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        latent_dim: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.save_hyperparameters(ignore=["args", "kwargs"])
        print(f"Ignored args: {args}, kwargs: {kwargs}")
        self.in_channels = in_channels
        self.padding = list(padding) if not isinstance(padding, list) else padding
        self.crop = CropND(padding)
        self.height = 40  # Expected height after padding
        self.width = 40  # Expected width after padding
        self.loss_l1 = F.l1_loss
        self.grad_loss = GradientDifferenceLoss()
        self.loss = lambda recon, orig: self.loss_l1(recon, orig) + 0.2 * self.grad_loss(
            recon, orig
        )
        self.stats_computed = False

        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, in_channels, H, W)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 256, 1, 1)
            nn.Flatten(),  # -> (B, 256)
            nn.Linear(256, latent_dim),  # -> (B, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.height // 8 * self.width // 8),
            Rearrange("b (c h w) -> b c h w", c=256, h=self.height // 8, w=self.width // 8),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 10, 10)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 20, 20)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, in_channels, kernel_size=4, stride=2, padding=1
            ),  # -> (B, in_channels, 40, 40)
        )

    def setup(self, stage: str):
        """Compute normalization statistics from training data."""
        if stage == "fit" and not self.stats_computed:
            print("Computing channel statistics from training data...")

            train_dataloader = self.trainer.datamodule.train_dataloader()  # type: ignore

            channel_sum = torch.zeros(self.in_channels)
            channel_sum_sq = torch.zeros(self.in_channels)
            n_pixels = 0

            # Iterate through a subset of training data (or all if dataset is small)
            pbar = tqdm(train_dataloader, desc="Calculating stats", leave=False)
            for batch in pbar:
                nwp = batch[0]
                # Compute statistics per channel
                channel_sum += reduce(nwp, "b c h w -> c", "sum")
                channel_sum_sq += reduce(nwp**2, "b c h w -> c", "sum")
                n_pixels += nwp.shape[0] * nwp.shape[2] * nwp.shape[3]

            # Calculate mean and std
            means = channel_sum / n_pixels
            stds = torch.sqrt(channel_sum_sq / n_pixels - means**2)
            # Avoid division by zero
            stds = torch.clamp(stds, min=1e-6)

            self.register_buffer("channel_means", rearrange(means, "c -> c 1 1"))
            self.register_buffer("channel_stds", rearrange(stds, "c -> c 1 1"))

            self.stats_computed = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using channel-wise mean and std."""
        return (x - self.channel_means) / self.channel_stds  # type: ignore

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output back to original scale."""
        return x * self.channel_stds + self.channel_means  # type: ignore

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        x = self.normalize(x)
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = self.encoder(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        x = self.decoder(z)
        x = self.crop(x)
        x = self.denormalize(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_recon = self(x)

        # MSE loss only on valid pixels
        loss = self.loss(x_recon, x)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_recon = self(x)

        # MSE loss only on valid pixels
        loss = self.loss(x_recon, x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return the latent representations

        Args:
            batch (list[torch.Tensor]): The input batch of tensors.
            batch_idx (int): The index of the batch.

        Returns:
            Any: The latent representations of the input batch.
        """
        x = batch[0]
        return self.encode(x)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # If buffer exists in checkpoint, load it
        if "channel_means" in checkpoint["state_dict"]:
            print("Loading channel_means from checkpoint")
            self.register_buffer("channel_means", checkpoint["state_dict"]["channel_means"])
        if "channel_stds" in checkpoint["state_dict"]:
            print("Loading channel_stds from checkpoint")
            self.register_buffer("channel_stds", checkpoint["state_dict"]["channel_stds"])


class GradientDifferenceLoss(nn.Module):
    def __init__(self, loss_type="l1"):
        super().__init__()
        self.loss_type = loss_type

        # Sobel filters for x and y gradients
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        # Shape for conv2d weight: (out_channels, in_channels, kH, kW)
        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)

    def forward(self, pred, target):
        # If multi-channel, apply per channel
        grad_loss = 0.0
        for c in range(pred.shape[1]):
            gx_pred = F.conv2d(pred[:, c : c + 1], self.sobel_x.to(pred.device), padding=1)
            gy_pred = F.conv2d(pred[:, c : c + 1], self.sobel_y.to(pred.device), padding=1)
            gx_true = F.conv2d(target[:, c : c + 1], self.sobel_x.to(target.device), padding=1)
            gy_true = F.conv2d(target[:, c : c + 1], self.sobel_y.to(target.device), padding=1)

            grad_mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)
            grad_mag_true = torch.sqrt(gx_true**2 + gy_true**2 + 1e-6)

            if self.loss_type == "l1":
                grad_loss += torch.mean(torch.abs(grad_mag_true - grad_mag_pred))
            else:
                grad_loss += torch.mean((grad_mag_true - grad_mag_pred) ** 2)

        return grad_loss / pred.shape[1]
