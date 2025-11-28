from abc import ABC
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from einops import reduce
from omegaconf import DictConfig

from genpp.models.base_module import BaseModule
from genpp.models.distributionalRegression.distributions import PredictiveDistribution


class DistributionRegression(BaseModule, ABC):
    def __init__(
        self,
        out_distribution: Callable[..., PredictiveDistribution],
        height: int,
        width: int,
        embedding_dim: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        if isinstance(rescaler, Sequence):
            filtered = [m for m in rescaler if m is not None]
            self.rescaler = nn.ModuleList(filtered) if filtered else None
        self.out_distribution = out_distribution(rescaler=rescaler)
        self.out_features = self.out_distribution.n_params
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim
        self.optimizer_partial = optimizer
        self.lr_scheduler_partial = lr_scheduler

    def training_step(self, batch) -> torch.Tensor:
        x, y, time_delta = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, time_delta)
        loss = self.out_distribution.compute_loss(res, y)
        loss = torch.mean(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        x, y, time_delta = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, time_delta)
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

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        x, y, time_delta = batch["x"], batch["y"], batch["timedelta"]
        res = self.forward(x, time_delta)
        loss_u = self.out_distribution.compute_loss(res, y)
        loss = reduce(loss_u, "b c h w -> c", "mean")
        # Log the loss for each variable separately
        for i, l_value in enumerate(loss):
            self.log(
                f"test_loss_var_{i}",
                l_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss_mean = reduce(loss_u, "b c h w -> 1", "mean")
        self.log(
            "test_loss",
            loss_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss_mean

    def predict_step(self, batch) -> torch.Tensor:
        x, time_delta = batch["x"], batch["timedelta"]
        res = self.forward(x, time_delta)
        return res
