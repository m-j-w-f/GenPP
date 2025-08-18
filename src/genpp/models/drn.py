from collections.abc import Callable

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from genpp.models.distributions import PredictiveDistribution
from genpp.models.meta import DistributionRegression


class DRNModel(DistributionRegression):
    """
    Distributional Regression Network (DRN) model for probabilistic forecasting.
    Rasp and Lerch (2018)

    This model predicts mu and sigma for a Gaussian distribution per grid point per variable.
    """

    def __init__(
        self,
        in_features: int,
        out_distribution: PredictiveDistribution,
        hidden_channels: list[int],
        height: int,
        width: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: DictConfig,
        embedding_dim: int = 5,
        normalize: bool = False,
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
        # Pixel index is removed if favor for embedding
        self.in_features = in_features + embedding_dim - 1
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
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
