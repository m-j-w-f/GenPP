from collections.abc import Callable, Sequence
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from genpp.models.distributionalRegression.distributions import (
    PredictiveDistribution,
    maybe_list_dist_param_dict,
)
from genpp.models.distributionalRegression.meta import DistributionRegression


class EMOS(DistributionRegression):
    """EMOS Model
    NOTE that we can use unscaled inputs and outputs since the model is very simple

    Args:
        DistributionRegression (_type_): _description_
    """

    def __init__(
        self,
        out_distribution: Callable[..., PredictiveDistribution],
        height: int,
        width: int,
        embedding_dim: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        use_rescaler: bool = False,  # NOTE difference to drn
        n_lead_times: int = 5,
        **kwargs: Any,
    ) -> None:
        self.save_hyperparameters()
        super().__init__(
            out_distribution=out_distribution,
            height=height,
            width=width,
            embedding_dim=embedding_dim,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rescaler=rescaler if use_rescaler else None,
        )
        if kwargs:
            warn(f"Ignoring additional arguments: {kwargs}")
        if self.embedding_dim != 0:
            raise ValueError("EMOS model does not support embedding_dim != 0")
        # Number of features that are predicted. Each distribution has 2 params (mean and std) which is double the number of features
        self.n_vars = self.out_distribution.n_params // 2
        self.n_lead_times = n_lead_times
        self.register_buffer("lead_times", torch.linspace(0, 1, n_lead_times + 1)[1:])  # Exclude 0

        self.weight_mean = nn.Parameter(torch.ones(self.n_vars, height, width))
        self.bias_mean = nn.Parameter(torch.zeros_like(self.weight_mean))
        self.weight_std = nn.Parameter(torch.randn(self.n_lead_times, self.n_vars, height, width))
        self.bias_std = nn.Parameter(torch.zeros_like(self.weight_std))

    def forward(self, x: torch.Tensor, time_delta: torch.Tensor) -> maybe_list_dist_param_dict:
        """Forward pass for the EMOS model.

        Args:
            x (torch.Tensor): Input tensor containing the variables. Should contain the means and standard deviations of the respective variables.
            The ordering of the channels is expected to be mean_var0, mean_var1, ..., std_var0, std_var1, ...
            time_delta (torch.Tensor): Lead time (between 0 and 1) for which the prediction is made. There are n_lead_times different lead times.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The means and standard deviations predicted by the model.
        """
        means, stds = torch.chunk(x, 2, dim=1)  # Both have shape [b, n_vars, h, w]
        means = means * self.weight_mean + self.bias_mean
        # Select the appropriate std parameters based on lead time
        distances = torch.abs(
            self.lead_times.unsqueeze(0) - time_delta.unsqueeze(1)  # type: ignore
        )  # [b, n_lead_times]
        indices = torch.argmin(distances, dim=1)  # [b]
        weight_std = self.weight_std[indices]  # [b, n_vars, h, w]
        bias_std = self.bias_std[indices]  # [b, n_vars, h, w]
        stds = stds * weight_std + bias_std

        # Interleave the means and stds so that we have mean_var0, std_var0, mean_var1, std_var1, ...
        x = torch.stack([means, stds], dim=2)  # Shape [b, n_vars, 2, h, w]
        x = rearrange(x, "b c two h w -> b (c two) h w")
        # TODO check what this function here expects (in which order)
        x = self.out_distribution.final_activation(x)
        return x  # type: ignore
