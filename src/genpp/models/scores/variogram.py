"""Variogram score implementation."""

import torch
import torch.nn as nn
from einops import rearrange, reduce

from .kernels.reshaping import _flatten_per_mode


class VariogramScore(nn.Module):
    """Computes the variogram score between predicted and true values.

    The variogram score evaluates how well the predicted samples capture
    the spatial correlation structure of the true values.

    Args:
        p (float): The power parameter for the variogram. Default: 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Core variogram score computation on flattened inputs.

        Args:
            x (torch.Tensor): Predicted values with shape [b, n_samples, c, h, w].
            y (torch.Tensor): True values with shape [b, c, h, w].

        Returns:
            torch.Tensor: Variogram score with shape [..., ].
        """
        x = _flatten_per_mode(x, mode=mode)  # [..., n_samples, var]
        y = _flatten_per_mode(y.unsqueeze(1), mode=mode)  # [..., 1 var]

        y_diff = rearrange(y, "... 1 var -> ... var 1") - y
        y_diff = torch.abs(y_diff) ** self.p  # [..., var, var]

        x_diff = rearrange(x, "... n var -> ... n var 1") - rearrange(x, "... n var -> ... n 1 var")
        x_diff = torch.abs(x_diff) ** self.p  # [..., n_samples, var, var]
        x_diff = reduce(x_diff, "... n var1 var2 -> ... var1 var2", "mean")

        total_diff = torch.pow(y_diff - x_diff, 2)  # [..., var, var]

        res = reduce(total_diff, "... var1 var2 -> ...", "sum")
        return res
