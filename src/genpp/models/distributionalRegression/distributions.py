from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from genpp.models.scores import CRPS_Normal, CRPS_TruncatedNormal

dist_param_dict = dict[str, torch.Tensor]
dist_param_dicts = list[dist_param_dict]
maybe_list_dist_param_dict = dist_param_dict | dist_param_dicts


class PredictiveDistribution(ABC):
    def __init__(self, rescaler: Sequence[nn.Module | None] | nn.Module | None) -> None:
        self.n_params: int
        self.rescaler = rescaler

    @property
    @abstractmethod
    def final_activation(self) -> nn.Module:
        pass

    @abstractmethod
    def compute_loss(self, targets: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass


class PredictiveNormalDistribution(PredictiveDistribution):
    def __init__(self, rescaler: nn.Module | None = None) -> None:
        self.loss_fn = CRPS_Normal()
        self.n_params = 2  # mean and standard deviation
        self.rescaler = rescaler  # BUG this is none here and should not be none
        self._final_activation = self._create_final_activation_module()

    def _create_final_activation_module(self) -> nn.Module:
        rescaler = self.rescaler  # From outer scope

        class NormalActivation(nn.Module):
            def forward(self, x: torch.Tensor) -> dist_param_dict:
                mu = x[:, 0]
                sigma = x[:, 1]
                # Apply final activation
                sigma = torch.nn.functional.softplus(sigma) + 1e-9  # Must be strictly positive
                if rescaler is not None:
                    mu, sigma = rescaler(mu, sigma)  # type: ignore
                return {"mu": mu, "sigma": sigma}

        return NormalActivation()

    @property
    def final_activation(self) -> nn.Module:
        return self._final_activation

    def compute_loss(self, mu_sigma: dist_param_dict, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(**mu_sigma, y=y)


class PredictiveTruncatedNormalDistribution(PredictiveDistribution):
    def __init__(self, rescaler: nn.Module | None = None) -> None:
        self.loss_fn = CRPS_TruncatedNormal()
        self.n_params = 2  # mean and standard deviation
        self.rescaler = rescaler
        self._final_activation = self._create_final_activation_module()

    def _create_final_activation_module(self) -> nn.Module:
        rescaler = self.rescaler  # From outer scope

        class TruncatedNormalActivation(nn.Module):
            def forward(self, x: torch.Tensor) -> dist_param_dict:
                mu = x[:, 0]
                sigma = x[:, 1]
                # Apply final activation
                # TODO this assumes that wind speed will always be min max transformed
                mu = torch.nn.functional.softplus(mu)  # Must be positive
                sigma = torch.nn.functional.softplus(sigma) + 1e-9  # Must be strictly positive
                if rescaler is not None:
                    mu, sigma = rescaler(mu, sigma)  # type: ignore
                return {
                    "mu": mu,
                    "sigma": sigma,
                }

        return TruncatedNormalActivation()

    @property
    def final_activation(self) -> nn.Module:
        return self._final_activation

    def compute_loss(self, mu_sigma: dist_param_dict, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(**mu_sigma, y=y)


class PredictiveCombinedDistribution(PredictiveDistribution):
    def __init__(self, rescaler: Sequence[nn.Module | None] | None = None) -> None:
        dists = [PredictiveNormalDistribution, PredictiveTruncatedNormalDistribution]
        if rescaler is None:
            self.rescaler = [None] * len(dists)
        else:
            self.rescaler = rescaler
            assert len(dists) == len(self.rescaler)
        self.dists = [
            dist(rescaler=r)
            for dist, r in zip(
                dists,
                self.rescaler,
            )
        ]
        self.n_params = sum(dist.n_params for dist in self.dists)
        self.split = [dist.n_params for dist in self.dists]
        self._final_activation = self._create_final_activation_module()

    def _create_final_activation_module(self) -> nn.Module:
        class CombinedActivation(nn.Module):
            def __init__(self, dists, split):
                super().__init__()
                self.dists = dists
                self.split = split

            def forward(self, x: torch.Tensor) -> dist_param_dicts:
                param_groups = torch.split(
                    x, self.split, dim=1
                )  # Param groups for each distribution
                # Returns a list with {param_dict for dist1, param_dict for dist2, ...}
                return [
                    dist.final_activation(params)
                    for (dist, params) in zip(self.dists, param_groups)
                ]

        return CombinedActivation(self.dists, self.split)

    @property
    def final_activation(self) -> nn.Module:
        return self._final_activation

    def compute_loss(self, param_dict_list: dist_param_dicts, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss for each distribution.

        Args:
            param_dict_list (dist_param_dicts): A list of parameter dictionaries for each distribution.
            Each parameter should have shape [b, 1, h, w]
            y (torch.Tensor): The target values. Shape [b, 1, h, w]

        Returns:
            torch.Tensor: The computed loss values. Shape [b, n_dists, h, w].
        """
        y_split = torch.split(y, 1, dim=1)
        return torch.cat(
            [
                dist.compute_loss(params, y=y_s.squeeze()).unsqueeze(1)
                for dist, params, y_s in zip(self.dists, param_dict_list, y_split)
            ],
            dim=1,
        )
