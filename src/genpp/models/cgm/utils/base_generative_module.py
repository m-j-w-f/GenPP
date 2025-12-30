from abc import ABC
from collections.abc import Callable

import torch
from omegaconf import DictConfig

from genpp.models.base_module import BaseModule


class BaseGenerativeModule(BaseModule, ABC):
    """Base class for generative models.

    This class provides common functionality for generative models including:
    - n_samples_train: Number of ensemble samples to generate during training
    - n_samples_predict: Number of ensemble samples to generate during prediction

    Args:
        optimizer (Callable[..., torch.optim.Optimizer]): Factory function to create the optimizer.
        lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
        n_samples (int, optional): Number of ensemble samples for both training and prediction.
            Used for backwards compatibility. If provided, sets both n_samples_train and n_samples_predict.
        n_samples_train (int, optional): Number of samples to generate during training.
            If not provided, uses n_samples.
        n_samples_predict (int, optional): Number of samples to generate during prediction.
            If not provided, uses n_samples.
    """

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        n_samples: int | None = None,
        n_samples_train: int | None = None,
        n_samples_predict: int | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)

        self.n_samples_train: int
        self.n_samples_predict: int

        if n_samples_train is not None:
            self.n_samples_train = n_samples_train
        elif n_samples is not None:
            self.n_samples_train = n_samples
        else:
            raise ValueError("n_samples_train must be specified either directly or via n_samples.")

        if n_samples_predict is not None:
            self.n_samples_predict = n_samples_predict
        elif n_samples is not None:
            self.n_samples_predict = n_samples
        else:
            raise ValueError(
                "n_samples_predict must be specified either directly or via n_samples."
            )
