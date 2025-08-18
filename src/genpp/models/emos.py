from collections.abc import Callable

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from genpp.models.distributions import PredictiveDistribution, maybe_list_dist_param_dict
from genpp.models.meta import DistributionRegression


class EMOS(DistributionRegression):
    def __init__(
        self,
        out_distribution: PredictiveDistribution,
        height: int,
        width: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        rescalers: list[nn.Module | None] | nn.Module | None = None,
    ) -> None:
        super().__init__(
            out_distribution=out_distribution,
            height=height,
            width=width,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rescalers=rescalers,
        )
        # Number of features that are predicted. Each distribution has 2 params (mean and std) which is double the number of features
        self.n_vars = self.out_distribution.n_params // 2

        self.weight_mean = nn.Parameter(torch.empty(self.n_vars, height, width))
        self.bias_mean = nn.Parameter(torch.empty(self.n_vars, height, width))
        self.weight_std = nn.Parameter(torch.empty(self.n_vars, height, width))
        self.bias_std = nn.Parameter(torch.empty(self.n_vars, height, width))

    def forward(self, x: torch.Tensor) -> maybe_list_dist_param_dict:
        """Forward pass for the EMOS model.

        Args:
            x (torch.Tensor): Input tensor containing the variables. Should contain the means and standard deviations of the respective variables.
            The ordering of the channels is expected to be mean_var0, mean_var1, ..., std_var0, std_var1, ...

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The means and standard deviations predicted by the model.
        """
        means, stds = torch.chunk(x, 2, dim=1)  # Both have shape [b, n_vars, h, w]
        means = torch.einsum("b c h w, c h w -> b c h w", means, self.weight_mean) + self.bias_mean
        stds = torch.einsum("b c h w, c h w -> b c h w", stds, self.weight_std) + self.bias_std

        # Interleave the means and stds so that we have mean_var0, std_var0, mean_var1, std_var1, ...
        x = torch.stack([means, stds], dim=2)  # Shape [b, n_vars, 2, h, w]
        x = rearrange(x, "b c two h w -> b (c two) h w")

        x = self.out_distribution.final_activation(x)
        return x  # type: ignore
