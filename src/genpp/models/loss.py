import math

import torch
import torch.nn as nn
from einops import rearrange, reduce


class EnergyScore(nn.Module):
    """Computes the energy score between predicted and true values.

    Args:
        beta (float): The beta parameter for the energy score.
        clamp (bool): Whether to clamp the values to avoid numerical issues.
    """

    def __init__(self, beta: float = 1.0, clamp: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.clamp = clamp
        if clamp:
            self.eps = 1e-8
            self.max_value = 1e10

    def l2_beta_norm(self, diff: torch.Tensor) -> torch.Tensor:
        sq_diff = (diff) ** 2
        sq_diff_sum = reduce(sq_diff, "... spatial -> ... 1", reduction="sum")
        if self.clamp:
            sq_diff_sum = torch.clamp(sq_diff_sum, min=self.eps, max=self.max_value)
        reduced = reduce(torch.sqrt(sq_diff_sum) ** self.beta, "... n 1 -> ...", reduction="mean")
        return reduced

    def forward(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Computes the energy score between the predicted and true values.

        Args:
            x (torch.Tensor): The predicted values with shape [..., n_samples, variables].
            y (torch.Tensor): The true values with shape [..., variables].
            Additional args and kwargs are ignored.

        Returns:
            torch.Tensor: The computed energy score with shape [..., ].
        """
        # For correct broadcasting
        y = rearrange(y, "... var -> ... 1 var")

        # Calculate first term: E[||y_pred - y_true||]
        es_12 = self.l2_beta_norm(x - y)

        # Calculate second term: E[||y_pred_i - y_pred_j||] for i != j
        G = torch.matmul(
            x, rearrange(x, "... n spatial -> ... spatial n")
        )  # [..., n_samples, n_samples]

        # Extract diagonal elements (||y_pred_i||^2)
        d = rearrange(torch.diagonal(G, dim1=-2, dim2=-1), "... n -> ... n 1")

        # Compute pairwise distances: ||y_pred_i||^2 + ||y_pred_j||^2 - 2 * y_pred_i^T * y_pred_j
        distances_22 = d + rearrange(d, "... n 1 -> ... 1 n") - 2 * G
        if self.clamp:
            # Clamp distances to avoid numerical issues
            distances_22 = torch.clamp(distances_22, min=self.eps, max=self.max_value)

        # Sum over all pairs (including diagonal, but we'll account for that)
        # NOTE: in the engression codebase the diagonal elements are excluded, here we don't
        #       this is inline with the library scoringRules in R
        es_22 = reduce(torch.sqrt(distances_22), "... n1 n2 -> ...", reduction="mean")  # [..., ]
        es = es_12 - 0.5 * es_22
        return es


class PatchwiseEnergyScore(EnergyScore):
    def __init__(
        self,
        beta: float = 1.0,
        clamp: bool = True,
        patch_size: int = 3,
        height: int = 37,
        width: int = 31,
        normalize: bool = True,
    ) -> None:
        """
        Initialize a PatchwiseEnergyScore.

        Args:
            beta (float, optional): Exponent used in the L2-beta norm inside the energy
                score computations. Default: 1.0.
            clamp (bool, optional): If True, clamp small/large values to avoid numerical
                instabilities. Default: True.
            patch_size (int, optional): Side length of the square patch used to compute
                the patchwise energy score. Default: 3.
            height (int, optional): Height of the input image. Default: 37.
            width (int, optional): Width of the input image. Default: 31.
            normalize (bool, optional): Whether to scale the computed per-patch energy by
                1.0 / (patch_size ** beta) to keep magnitudes comparable across patch sizes.
                Default: True.

        Notes:
            - In 'complete' mode, the forward expects:
                x: [batch, n_samples, c*h*w]
                y: [batch, c*h*w]
              The inputs are reshaped to [batch, c, h, w] and patches are extracted via
              unfold; the result is averaged over patches to produce [batch].
            - In 'per_var' mode, the forward expects:
                x: [batch, c, n_samples, h*w]
                y: [batch, c, h*w]
              Patches are computed per channel and averaged over patches to produce [batch, c].
            - Pads are chosen so every pixel participates equally using padding derived
              from patch_size.
        """
        super().__init__(beta=beta, clamp=clamp)
        self.patch_size = (patch_size, patch_size)
        self.height = height
        self.width = width
        self.padding = tuple(
            [(k - 1) // 2 for k in self.patch_size]
        )  # ensure that every pixel is covered the same number of times
        if normalize:
            self.normalization_factor = 1.0 / (patch_size**beta)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "complete") -> torch.Tensor:
        """Computes the spatial energy score between predicted and true values.

        Args:
            x (torch.Tensor): Predicted values with shape [batch, n_samples, (variables, height, width)] in case of mode "complete".
                If mode is "per_var", shape is [batch, variables, n_samples, (height, width)].
            y (torch.Tensor): True values with shape [batch, (variables, height, width)].
                If mode is "per_var", shape is [batch, variables, (height, width)].
            mode (str): Mode of input data, either "complete" or "per_var".
        Note:
            In this implementations c refers to the number of channels/variables.
        Returns:
            torch.Tensor: The computed energy score with shape [batch, num_patches] in case of mode "complete".
                If mode is "per_var", shape is [batch, variables].
        """
        if mode == "complete":
            # F.unfold only works with tensors of shape [b, c, *]
            *B, N, _ = x.shape
            x = rearrange(x, "b n (c h w) -> (b n) c h w", h=self.height, w=self.width)
            x_patchwise = torch.nn.functional.unfold(
                x, kernel_size=self.patch_size, stride=1
            )  # [(b, n), patchsize, num_patches]
            x_patchwise = rearrange(
                x_patchwise, "(b n) patchsize num_patches -> b num_patches n patchsize", n=N
            )  # [b, num_patches, n, patchsize]

            y = rearrange(y, "... (c h w) -> (...) c h w", h=self.height, w=self.width)
            y_patchwise = torch.nn.functional.unfold(
                y, kernel_size=self.patch_size, stride=1
            )  # [B, patchsize, num_patches]
            y_patchwise = rearrange(
                y_patchwise, "b patchsize num_patches -> b num_patches patchsize"
            )  # [b, num_patches, patchsize]
            es = super().forward(x_patchwise, y_patchwise)  # [b, num_patches]
            if self.normalization_factor is not None:
                es = es * self.normalization_factor
            return reduce(es, "b num_patches -> b", "mean")  # [b]
        elif mode == "per_var":
            *B, C, N, _ = x.shape
            x = rearrange(x, "b c n (h w) -> (b c n) 1 h w", h=self.height, w=self.width)
            x_patchwise = torch.nn.functional.unfold(
                x, kernel_size=self.patch_size, stride=1
            )  # [(b, c, n), patchsize, num_patches]
            x_patchwise = rearrange(
                x_patchwise,
                "(b c n) patchsize num_patches -> b c num_patches n patchsize",
                c=C,
                n=N,
            )

            y = rearrange(y, "b c (h w) -> (b c) 1 h w", h=self.height, w=self.width)
            y_patchwise = torch.nn.functional.unfold(
                y, kernel_size=self.patch_size, stride=1
            )  # [(b, c), patchsize, num_patches]
            y_patchwise = rearrange(
                y_patchwise, "(b c) patchsize num_patches -> b c num_patches patchsize", c=C
            )
            es = super().forward(x_patchwise, y_patchwise)  # [b, c, num_patches]
            if self.normalization_factor is not None:
                es = es * self.normalization_factor
            return reduce(es, "b c num_patches -> b c", "mean")  # [b, c]

        else:
            raise ValueError(f"Mode {mode} not recognized. Use 'complete' or 'per_var'.")


class VariogramScore(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the variogram score between predicted and true values.

        Args:
            x (torch.Tensor): Predicted values with shape [..., n_samples, variables].
            y (torch.Tensor): True values with shape [..., variables].

        Returns:
            torch.Tensor: The computed variogram score with shape [..., ].
        """
        y_diff = rearrange(y, "... var -> ... var 1") - rearrange(y, "... var -> ... 1 var")
        y_diff = torch.abs(y_diff) ** self.p  # [b, d, margin, margin]

        x_diff = rearrange(x, "... n var -> ... n var 1") - rearrange(x, "... n var -> ... n 1 var")
        x_diff = torch.abs(x_diff) ** self.p  # [b, n_samples, margin, margin]
        x_diff = reduce(x_diff, "... n var1 var2 -> ... var1 var2", "mean")

        total_diff = torch.pow(y_diff - x_diff, 2)  # [b, d, margin, margin]

        res = reduce(total_diff, "... var1 var2 -> ...", "sum")
        return res


class CRPS_Normal(nn.Module):
    """Source: Höhlein et. al (2024) Postprocessing of Ensemble Weather Forecasts Using
    Permutation-Invariant Neural Networks
    https://github.com/khoehlein/Permutation-invariant-Postprocessing/blob/main/model/loss/losses.py
    """

    def __init__(self):
        super().__init__()
        self._inv_sqrt_pi = 1 / torch.sqrt(torch.tensor(math.pi))
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed data.

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
    Source: Thorarinsdottir, Gneiting (2010) Probabilistic Forecasts of Wind Speed: Ensemble Model Output Statistics by using Heteroscedastic Censored Regression
    https://academic.oup.com/jrsssa/article/173/2/371/7077664
    Implementation based on
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
    def __init__(self, n_axis: int = -4) -> None:
        super().__init__()
        self.n_axis = n_axis

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the CRPS based on an finite ensemble.

        Args:
            x (torch.Tensor): predictions of shape [..., n, d, h, w]
            y (torch.Tensor): target values of shape [..., d, h, w]

        Returns:
            torch.Tensor: CRPS values of shape [..., d, h, w]
        """
        y = y.unsqueeze(self.n_axis)
        dxy = torch.abs(x - y).mean(self.n_axis)
        dxx = torch.abs(x.unsqueeze(self.n_axis) - x.unsqueeze(self.n_axis - 1)).mean(
            dim=[self.n_axis, self.n_axis - 1]
        )
        return dxy - 0.5 * dxx
