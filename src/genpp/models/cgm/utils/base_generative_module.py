from abc import ABC
from collections.abc import Callable

import torch
from omegaconf import DictConfig

from genpp.models.base_module import BaseModule


class BaseGenerativeModule(BaseModule, ABC):
    """Base class for generative models.

    This class provides common functionality for generative models including:
    - n_samples: Number of ensemble samples to generate during prediction

    Args:
        optimizer (Callable[..., torch.optim.Optimizer]): Factory function to create the optimizer.
        lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
        n_samples (int): Number of ensemble samples to generate during prediction.
    """

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        n_samples: int,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.n_samples = n_samples
