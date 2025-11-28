from abc import ABC
from collections.abc import Callable

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig


def _instantiate_partial_scheduler(
    partial_scheduler: DictConfig, optimizer: torch.optim.Optimizer
) -> DictConfig:
    # This is ugly but works because the lr_scheduler_partial is a DictConfig
    if (
        partial_scheduler.scheduler.func is not torch.optim.lr_scheduler.ChainedScheduler
    ):  # Just a single scheduler
        partial_scheduler.scheduler = partial_scheduler.scheduler(optimizer)
    else:  # We need to instantiate the chained scheduler with the optimizer
        # It gets even uglier
        inner_schedulers = [
            p(optimizer) for p in partial_scheduler.scheduler.keywords["schedulers"]
        ]
        # Overwrite the inner schedulers with the instantiated ones by calling the func directly
        partial_scheduler.scheduler = partial_scheduler.scheduler.func(
            *partial_scheduler.scheduler.args,
            schedulers=inner_schedulers,
            **{k: v for k, v in partial_scheduler.scheduler.keywords.items() if k != "schedulers"},
        )

    return partial_scheduler


class BaseModule(L.LightningModule, ABC):
    """Base Module for all Lightning Modules in GenPP.
    This class handles the optimizer and learning rate scheduler configuration.
    """

    def __init__(
        self, optimizer: Callable[..., torch.optim.Optimizer], lr_scheduler: DictConfig
    ) -> None:
        super().__init__()
        self.optimizer_partial = optimizer
        self.lr_scheduler_partial = lr_scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Instantiate the optimizer and scheduler from the config
        self.optimizer = self.optimizer_partial(self.parameters())
        self.lr_scheduler_partial = _instantiate_partial_scheduler(
            self.lr_scheduler_partial, self.optimizer
        )

        return {  # type: ignore
            "optimizer": self.optimizer,
            "lr_scheduler": {**self.lr_scheduler_partial},
        }
