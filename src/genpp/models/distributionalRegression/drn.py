from collections.abc import Callable
from typing import Any
from collections.abc import Sequence
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.distributionalRegression.distributions import PredictiveDistribution
from genpp.models.distributionalRegression.meta import DistributionRegression


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
        hidden_channels: list[int],
        height: int,
        width: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        embedding_dim: int = 5,
        normalize: bool = False,
        rescaler: Sequence[nn.Module | None] | nn.Module | None = None,
        use_rescaler: bool = True,  # NOTE difference to emos
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
        )
        if kwargs:
            warn(f"Ignoring additional arguments: {kwargs}")
        # Pixel index is removed if favor for embedding
        self.in_features = in_features + embedding_dim - 1
        self.hidden_channels = hidden_channels
        self.normalize = normalize

        self.embedding = nn.Embedding(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, var, lon, lat].

        Returns:
            torch.Tensor: Output tensor.
        """
        x, pixel_idx = x[:, :-1], x[:, -1]  # Last variable is the pixel index
        embedding = self.embedding(pixel_idx.int())
        embedding = rearrange(embedding, "b h w e -> b e h w")
        x = torch.cat([x, embedding], dim=1)
        x = self.network(x)
        return x
