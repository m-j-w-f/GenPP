from collections.abc import Callable, Mapping
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig

from genpp.models.distributions import PredictiveDistribution
from genpp.models.utils import instantiate_partial_scheduler


class DRNModel(L.LightningModule):
    """
    Distributional Regression Network (DRN) model for probabilistic forecasting.
    Rasp and Lerch (2018)

    This model predicts mu and sigma for a Gaussian distribution per grid point per variable.
    """

    def __init__(
        self,
        in_features: int,
        out_distribution: PredictiveDistribution,
        hidden_channels: list[int],
        height: int,
        width: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        embedding_dim: int = 5,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        # Pixel index is removed if favor for embedding
        self.in_features = in_features + embedding_dim - 1
        self.out_distribution = out_distribution
        self.out_features = out_distribution.n_params
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        self.optimizer_partial = optimizer
        self.lr_scheduler_partial = lr_scheduler
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.embedding = nn.Embedding(
            num_embeddings=self.height * self.width, embedding_dim=embedding_dim
        )

        layers = []
        prev_dim = self.in_features

        # Hidden layers
        for hidden_dim in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_dim, hidden_dim, kernel_size=1),
                    nn.ELU(),
                ]
            )
            if normalize:
                layers.extend(
                    [
                        Rearrange("b c h w -> b h w c"),
                        nn.LayerNorm(hidden_dim),
                        Rearrange("b h w c -> b c h w"),
                    ]
                )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Conv2d(prev_dim, self.out_features, kernel_size=1))
        layers.append(self.out_distribution.final_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, var, lon, lat].

        Returns:
            torch.Tensor: Output tensor.
        """
        x, pixel_idx = x[:, :-1], x[:, -1]  # Last variable is the pixel index
        embedding = self.embedding(pixel_idx.int())
        embedding = rearrange(embedding, "b h w e -> b e h w")
        x = torch.cat([x, embedding], dim=1)
        x = self.network(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        x, y = batch
        res = self.forward(x)
        loss = self.out_distribution.compute_loss(res, y)
        loss = torch.mean(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        x, y = batch
        res = self.forward(x)
        loss = self.out_distribution.compute_loss(res, y)
        loss = reduce(loss, "b c h w -> c", "mean")
        # Log the loss for each variable separately
        for i, l_value in enumerate(loss):
            self.log(
                f"val_loss_var_{i}",
                l_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        # Log the overall loss
        loss = torch.mean(loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        x, y = batch
        res = self.forward(x)
        loss = self.out_distribution.compute_loss(res, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Instantiate the optimizer and scheduler from the config
        self.optimizer = self.optimizer_partial(self.parameters())
        self.lr_scheduler_partial = instantiate_partial_scheduler(
            self.lr_scheduler_partial, self.optimizer
        )

        return {  # type: ignore
            "optimizer": self.optimizer,
            "lr_scheduler": {**self.lr_scheduler_partial},
        }
