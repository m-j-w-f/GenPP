from collections.abc import Callable, Mapping
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig

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
        out_features: int,
        hidden_channels: list[int],
        height: int,
        width: int,
        loss_fn: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        embedding_dim: int = 5,
        normalize: bool = False,
        final_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        self.loss_fn = loss_fn
        self.optimizer_partial = optimizer
        self.lr_scheduler_partial = lr_scheduler
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.final_activation = final_activation

        self.embedding = nn.Embedding(
            num_embeddings=self.height * self.width, embedding_dim=embedding_dim
        )

        layers = []
        prev_dim = in_features

        # Hidden layers
        for hidden_dim in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_dim, hidden_dim, kernel_size=1),
                    nn.ELU(),
                ]
            )
            if normalize:
                layers.append(
                    [
                        Rearrange("b c h w -> b h w c"),
                        nn.LayerNorm(hidden_dim),
                        Rearrange("b h w c -> b c h w"),
                    ]
                )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Conv2d(prev_dim, out_features, kernel_size=1))
        layers.append(self.final_activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, lat, lon, var].

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.network(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        x, y = batch
        res = self.forward(x)
        loss = self.loss_fn(res, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        x, y = batch
        res = self.forward(x)
        loss = self.loss_fn(res, y, avg="variable")
        # Log the loss for each variable separately
        for i in range(self.out_features):
            self.log(
                f"val_loss_var_{i}",
                loss[i],
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
        loss = self.loss_fn(res, y)
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
