from abc import ABC
from collections.abc import Callable
from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import reduce
from omegaconf import DictConfig

from genpp.models.distributionalRegression.distributions import PredictiveDistribution
from genpp.models.utils import BaseModule


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
        x, y = batch
        res = self.forward(x)
        loss = self.out_distribution.compute_loss(res, y)
        loss = torch.mean(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch) -> torch.Tensor:
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

    def test_step(self, batch) -> torch.Tensor:
        x, y = batch
        res = self.forward(x)
        loss = self.out_distribution.compute_loss(res, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
