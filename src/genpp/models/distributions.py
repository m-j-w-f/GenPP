from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from genpp.models.loss import CRPS_Normal, CRPS_TruncatedNormal

dist_param_dict = dict[str, torch.Tensor]
dist_param_dicts = list[dist_param_dict]
maybe_list_dist_param_dict = dist_param_dict | dist_param_dicts


class PredictiveDistribution(ABC):
    def __init__(self) -> None:
        self.n_params: int
        self.rescaler: list[nn.Module | None] | nn.Module | None

    @property
    @abstractmethod
    def final_activation(self) -> nn.Module:
        pass

    @abstractmethod
    def compute_loss(self, targets: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass


class NormalDistribution(PredictiveDistribution):
    def __init__(self, rescale: nn.Module | None = None) -> None:
        self.loss_fn = CRPS_Normal()
        self.n_params = 2  # mean and standard deviation
        self.rescaler = rescale
        self._final_activation = self._create_final_activation_module()

    def _create_final_activation_module(self) -> nn.Module:
        rescaler = self.rescaler  # From outer scope

        class NormalActivation(nn.Module):
            def forward(self, x: torch.Tensor) -> dist_param_dict:
                mu = x[:, 0]
                sigma = x[:, 1]
                # Apply final activation
                sigma = torch.nn.functional.sigmoid(sigma) + 1e-9  # Must be strictly positive
                if rescaler is not None:
                    mu, sigma = rescaler(mu, sigma)  # type: ignore
                return {"mu": mu, "sigma": sigma}

        return NormalActivation()

    @property
    def final_activation(self) -> nn.Module:
        return self._final_activation

    def compute_loss(self, mu_sigma: dist_param_dict, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(**mu_sigma, y=y)


class TruncatedNormalDistribution(PredictiveDistribution):
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
                mu = torch.nn.functional.sigmoid(mu)  # Must be positive
                sigma = torch.nn.functional.sigmoid(sigma) + 1e-9  # Must be strictly positive
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


class CombinedPredictiveDistribution(PredictiveDistribution):
    def __init__(self, rescalers: list[nn.Module | None] | None = None) -> None:
        self.dists = [NormalDistribution(), TruncatedNormalDistribution()]
        self.n_params = sum(dist.n_params for dist in self.dists)
        self.split = [dist.n_params for dist in self.dists]
        if rescalers is None:
            self.rescalers = [None] * len(self.dists)
        else:
            self.rescalers = rescalers
        # Each distribution should have the correct rescaler
        assert len(self.dists) == len(self.rescalers)
        for dist, rescaler in zip(self.dists, self.rescalers):
            dist.rescaler = rescaler
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
