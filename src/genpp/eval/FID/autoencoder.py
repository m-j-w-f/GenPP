import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpp.models.layers import CropND


class AutoEncoder(L.LightningModule):
    def __init__(self, in_channels: int, padding: tuple[int, int, int, int], latent_dim: int = 128):
        super().__init__()
        self.save_hyperparameters()
        self.padding = padding
        self.crop = CropND(padding)

        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, in_channels, H, W)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
        )

        # Latent space
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (B, 256, 1, 1)
            nn.Flatten(),  # -> (B, 256)
            nn.Linear(256, latent_dim),  # -> (B, latent_dim)
        )

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 5 * 5)  # -> (B, 256*5*5)

        self.decoder = nn.Sequential(
            # Input: (B, 256, 5, 5)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 10, 10)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 20, 20)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, in_channels, kernel_size=4, stride=2, padding=1
            ),  # -> (B, in_channels, 40, 40)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        x = self.encoder(x)
        x = self.encoder_fc(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        x = self.decoder_fc(z)
        x = x.view(-1, 256, 5, 5)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)

        # Crop both input and reconstruction to only consider valid pixels
        x_cropped = self.crop(x)
        x_recon_cropped = self.crop(x_recon)

        # MSE loss only on valid pixels
        loss = F.mse_loss(x_recon_cropped, x_cropped)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)

        # Crop both input and reconstruction to only consider valid pixels
        x_cropped = self.crop(x)
        x_recon_cropped = self.crop(x_recon)

        # MSE loss only on valid pixels
        loss = F.mse_loss(x_recon_cropped, x_cropped)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
