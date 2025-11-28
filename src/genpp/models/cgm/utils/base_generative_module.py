from abc import ABC
from collections.abc import Callable

import torch
from omegaconf import DictConfig

from genpp.models.base_module import BaseModule
from genpp.models.cgm.utils.td_scaling import (
    BaseInternalTDScaling,
    FixedTDScaling,
    LearnedTDScaling,
    LinearAbsTDScaling,
)


class BaseGenerativeModule(BaseModule, ABC):
    """Base class for generative models that use internal TD scaling.

    This class unifies common functionality between FlowMatchingModel and BaseChenModel,
    including:
    - internal_td_scaling: Scaling strategy to normalize (predicted) deviations based on lead time
    - n_samples: Number of ensemble samples to generate during prediction

    Args:
        optimizer (Callable[..., torch.optim.Optimizer]): Factory function to create the optimizer.
        lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
        internal_td_scaling (str): Scaling strategy to normalize (predicted) deviations based on
            lead time, ensuring the model learns a single scale across different forecast horizons.
            Can be "abs", "std", "learned", or "linear_abs".
        n_samples (int): Number of ensemble samples to generate during prediction.
    """

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        internal_td_scaling: str,
        n_samples: int,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.n_samples = n_samples
        if internal_td_scaling == "abs":
            self.internal_td_scaling: BaseInternalTDScaling = FixedTDScaling(mode="abs")
        elif internal_td_scaling == "std":
            self.internal_td_scaling = FixedTDScaling(mode="std")
        elif internal_td_scaling == "learned":
            self.internal_td_scaling = LearnedTDScaling()
        elif internal_td_scaling == "linear_abs":
            self.internal_td_scaling = LinearAbsTDScaling()
        else:
            raise ValueError(f"Invalid internal_td_scaling: {internal_td_scaling}")

    def setup(self, stage: str | None = None) -> None:
        """Fit the internal TD scaling model during the 'fit' stage.

        Args:
            stage (str | None): The stage of setup, e.g., "fit"
        """
        if stage == "fit":
            self.internal_td_scaling.fit(self)
