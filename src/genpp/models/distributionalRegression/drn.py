from collections.abc import Callable, Sequence
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.distributionalRegression.distributions import PredictiveDistribution
from genpp.models.distributionalRegression.meta import DistributionRegression
from genpp.models.layers import FourierEncoder


class DRNModel(DistributionRegression):
    """Distributional Regression Network (DRN) model for probabilistic forecasting.
    Rasp and Lerch (2018)

    This model predicts mu and sigma for a Gaussian distribution per grid point per variable.

    Args:
        in_features (int): Number of input features (excluding pixel index).
        out_distribution (Callable[..., PredictiveDistribution]): Function that returns
            the predictive distribution used for output.
        hidden_channels (list[int]): List of hidden layer dimensions for the network.
        height (int): Height dimension of the spatial grid.
        width (int): Width dimension of the spatial grid.
        optimizer (Callable[..., torch.optim.Optimizer]): Optimizer factory function.
        lr_scheduler (DictConfig): Learning rate scheduler configuration.
        embedding_dim (int, optional): Dimension of the pixel position embedding.
            Defaults to 5.
        normalize (bool, optional): Whether to apply layer normalization after each
            hidden layer. Defaults to False.
        rescaler (Sequence[nn.Module | None] | nn.Module, optional): Rescaling
            module(s) for output normalization. Defaults to None.
        use_rescaler (bool, optional): Whether to use the rescaler. Defaults to True.
            This is here as the rescaling module(s) are always passed to the model.
        **kwargs: Any additional keyword arguments. These are ignored and are only here
            for compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_distribution: Callable[..., PredictiveDistribution],
        hidden_channels: Sequence[int],
        height: int,
        width: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        embedding_dim: int = 5,
        td_embedding_dim: int = 4,
        normalize: bool = False,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        use_rescaler: bool = True,  # NOTE difference to emos
        variable_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            out_distribution=out_distribution,
            height=height,
            width=width,
            embedding_dim=embedding_dim,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rescaler=rescaler if use_rescaler else None,
            variable_names=variable_names,
        )
        self.save_hyperparameters()
        if kwargs:
            warn(f"Ignoring additional arguments: {kwargs}")

        self.in_features = in_features + embedding_dim + td_embedding_dim
        self.hidden_channels = hidden_channels
        self.normalize = normalize
        self.td_embedding = FourierEncoder(dim=td_embedding_dim)
        self.space_embedding = nn.Embedding(
            num_embeddings=self.height * self.width, embedding_dim=embedding_dim
        )

        layers = []
        prev_dim = self.in_features

        # Hidden layers
        for hidden_dim in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_dim, hidden_dim, kernel_size=1),
                    nn.ELU(),
                ]
            )
            if normalize:
                layers.extend(
                    [
                        Rearrange("b c h w -> b h w c"),
                        nn.LayerNorm(hidden_dim),
                        Rearrange("b h w c -> b c h w"),
                    ]
                )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Conv2d(prev_dim, self.out_features, kernel_size=1))
        layers.append(self.out_distribution.final_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: dict[str, torch.Tensor], time_delta: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, var, lon, lat].
            time_delta (torch.Tensor): Time delta tensor of shape [batch_size].

        Returns:
            torch.Tensor: Output tensor.
        """
        x_full = torch.cat([x["all_vars_mean"], x["all_vars_std"], x["meta_vars"]], dim=1)
        pixel_idx = x["pixel_idx"]
        space_embedding = self.space_embedding(pixel_idx)
        space_embedding = rearrange(space_embedding, "b 1 h w e -> b e h w")
        td_embedding = self.td_embedding(time_delta)  # [b, td_embedding_dim]
        td_embedding = repeat(td_embedding, "b e -> b e h w", h=self.height, w=self.width)
        x_full = torch.cat([x_full, space_embedding, td_embedding], dim=1)
        res = self.network(x_full)
        return res
