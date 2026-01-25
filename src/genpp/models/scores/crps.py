"""CRPS (Continuous Ranked Probability Score) for parametric distributions.

These classes compute CRPS assuming the predictions follow a parametric
distribution (Normal or Truncated Normal). For sample-based CRPS,
see ensemble_crps.py.
"""

import math

import torch
import torch.nn as nn


class CRPS_Normal(nn.Module):
    """CRPS for Normal Distribution.

    Computes the Continuous Ranked Probability Score (CRPS) assuming
    normally distributed predictions.

    Source: Höhlein et. al (2024) Postprocessing of Ensemble Weather Forecasts Using
    Permutation-Invariant Neural Networks
    https://github.com/khoehlein/Permutation-invariant-Postprocessing/blob/main/model/loss/losses.py
    """

    def __init__(self):
        super().__init__()
        self._inv_sqrt_pi = 1 / torch.sqrt(torch.tensor(math.pi))
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the CRPS assuming normally distributed data.

        Args:
            mu (torch.Tensor): Tensor of mean. Shape [b, 1, h, w]
            sigma (torch.Tensor): Tensor of standard deviation. Shape [b, 1, h, w]
            y (torch.Tensor): Observed data. Shape [b, 1, h, w]

        Returns:
            torch.Tensor: CRPS value of shape [b, 1, h, w]
        """
        z_red = (y - mu) / sigma

        cdf = self.dist.cdf(z_red)
        pdf = torch.exp(self.dist.log_prob(z_red))
        crps = sigma * (z_red * (2.0 * cdf - 1.0) + 2.0 * pdf - self._inv_sqrt_pi)
        return crps


class CRPS_TruncatedNormal(nn.Module):
    """CRPS for Truncated Normal Distribution.

    Computes the Continuous Ranked Probability Score (CRPS) for predictions
    following a truncated normal distribution.

    Source: Thorarinsdottir, Gneiting (2010) Probabilistic Forecasts of Wind Speed:
    Ensemble Model Output Statistics by using Heteroscedastic Censored Regression
    https://academic.oup.com/jrsssa/article/173/2/371/7077664

    Implementation based on:
    A. Jordan, F. Krüger, S. Lerch (2019) Evaluating Probabilistic Forecasts with scoringRules
    https://www.jstatsoft.org/article/view/v090i12
    and
    https://github.com/frazane/scoringrules/blob/e3739338e42393ff4487c4c760e194378a32546e/scoringrules/core/crps/_closed.py#L329

    This method is reimplemented here so that the backward pass does not return nans.
    """

    def __init__(
        self, lower: torch.Tensor | float = 0.0, upper: torch.Tensor | float = float("inf")
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        # Might need to convert to primitive types if not moved to correct device
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self.inv_sqrt_pi = 1 / torch.sqrt(torch.tensor(math.pi))
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def _norm_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the probability density function (PDF) of the standard normal distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: PDF values.
        """
        return torch.exp(self.dist.log_prob(x))

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the CRPS for Truncated Normal Distribution.

        Args:
            mu (torch.Tensor): Mean tensor, must be positive.
            sigma (torch.Tensor): Standard deviation tensor, must be strictly positive.
            y (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: CRPS value.
        """
        loc = (y - mu) / sigma
        l = (self.lower - mu) / sigma  # noqa: E741
        u = (self.upper - mu) / sigma if self.upper != float("inf") else torch.tensor(float("inf"))
        z = torch.min(torch.max(loc, l), u)

        F_u = self.dist.cdf(u) if self.upper != float("inf") else torch.tensor(1.0)
        F_l = self.dist.cdf(l)
        F_z = self.dist.cdf(z)
        F_u2 = self.dist.cdf(u * self.sqrt_2) if self.upper != float("inf") else torch.tensor(1.0)
        F_l2 = self.dist.cdf(l * self.sqrt_2)
        f_z = self._norm_pdf(z)

        c = 1 / (F_u - F_l)

        s1 = torch.abs(loc - z)
        s2 = c * z * (2 * F_z - (F_u + F_l))
        s3 = c * 2 * f_z
        s4 = c**2 * (F_u2 - F_l2) * self.inv_sqrt_pi
        res = sigma * (s1 + s2 + s3 - s4)
        return res


class EnsembleCRPS(nn.Module):
    """Sample-based CRPS computed from a finite ensemble.

    This implementation is the same as the previous ``ensemble_crps.py`` module
    but lives here so all CRPS-related implementations are colocated.

    Args:
        n_axis (int): Axis along which the samples are arranged. Default: -4.
    """

    def __init__(self, n_axis: int = -4) -> None:
        super().__init__()
        self.n_axis = n_axis

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the CRPS based on a finite ensemble.

        Args:
            x (torch.Tensor): Predictions of shape [..., n, c, h, w] where n is
                the number of ensemble members.
            y (torch.Tensor): Target values of shape [..., c, h, w].

        Returns:
            torch.Tensor: CRPS values of shape [..., c, h, w].
        """
        y = y.unsqueeze(self.n_axis)
        dxy = torch.abs(x - y).mean(self.n_axis)
        dxx = torch.abs(x.unsqueeze(self.n_axis) - x.unsqueeze(self.n_axis - 1)).mean(
            dim=[self.n_axis, self.n_axis - 1]
        )
        return dxy - 0.5 * dxx
