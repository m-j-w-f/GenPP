import lightning as L
import torch
import torch.nn as nn


class EMOS(L.LightningModule):
    def __init__(self, n_vars: int, height: int, width: int, loss_fn: nn.Module) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.weight_mean = nn.Parameter(torch.empty(n_vars, height, width))
        self.bias_mean = nn.Parameter(torch.empty(n_vars, height, width))
        self.weight_std = nn.Parameter(torch.empty(n_vars, height, width))
        self.bias_std = nn.Parameter(torch.empty(n_vars, height, width))
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vars = torch.split(x, self.n_vars, dim=1)
        assert len(vars) == 2, "Input tensor must be split into two parts: mean and std"
        means = (
            torch.einsum("b c h w, c h w -> b c h w", vars[0], self.weight_mean) + self.bias_mean
        )
        stds = torch.einsum("b c h w, c h w -> b c h w", vars[1], self.weight_std) + self.bias_std
        return means, stds
