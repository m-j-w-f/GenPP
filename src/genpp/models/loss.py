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
        # Note: x will be referred to as y_pred and y as y_true in comments below
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


class RBFScore(nn.Module):
    """Radial Basis Function (RBF) score for multivariate predictions.

    Args:
        lengthscale (float): Lengthscale parameter for the RBF kernel.
    """

    def __init__(self, lengthscale: float = 1.0) -> None:
        super().__init__()
        self.lengthscale = lengthscale

    def _rbf_kernel(self, diff: torch.Tensor) -> torch.Tensor:
        sq_diff = (diff) ** 2
        sq_diff_sum = torch.sum(sq_diff, dim=-1)
        rbf = torch.exp(-0.5 * sq_diff_sum / (self.lengthscale**2))
        return rbf

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the RBF score between predicted and true values.

        Args:
            x (torch.Tensor): Predicted values with shape [..., n_samples, variables].
            y (torch.Tensor): True values with shape [..., variables].
        """
        diff = x - rearrange(y, "... var -> ... 1 var")
        rbf_xy = self._rbf_kernel(diff)  # shape [..., n_samples]
        term1 = reduce(rbf_xy, "... n-> ...", reduction="mean")

        diff_xx = rearrange(x, "... n var -> ... n 1 var") - rearrange(
            x, "... n var -> ... 1 n var"
        )  # shape [..., n_samples, n_samples, variables]
        rbf_xx = self._rbf_kernel(diff_xx)  # shape [..., n_samples, n_samples]
        term2 = reduce(rbf_xx, "... n1 n2 -> ...", reduction="mean")
        rbf_score = term1 - 0.5 * term2
        return rbf_score


class PatchwiseMixin:
    def __init__(self, patch_size, height, width) -> None:
        self.patch_size = (patch_size, patch_size)
        self.height = height
        self.width = width
        self.padding = tuple(
            [k // 2 for k in self.patch_size] * 2
        )  # ensure that every pixel is covered the same number of times

    def _pad_and_unfold(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, self.padding, mode="reflect")
        x_patchwise = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=1
        )  # [B, patchsize, num_patches]
        return x_patchwise

    def patchify(
        self, x: torch.Tensor, y: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits x and y into patches

        Args:
            x (torch.Tensor): Predictions Tensor of shape [batch, n_samples, (variables, height, width)] in case of mode "complete".
                If mode is "per_var", shape is [batch, variables, n_samples, (height, width)].
            y (torch.Tensor): Target Values Tensor of shape [batch, (variables, height, width)].
                If mode is "per_var", shape is [batch, variables, (height, width)].
            mode (str): "complete" or "per_var"

        Raises:
            ValueError: If mode is not recognized.

        Returns:
            torch.Tensor: Patchified tensors x and y. Of shapes:
                - In "complete" mode:
                    x: [batch, num_patches, n_samples, patchsize]
                    y: [batch, num_patches, patchsize]
                - In "per_var" mode:
                    x: [batch, variables, num_patches, n_samples, patchsize]
                    y: [batch, variables, num_patches, patchsize]
        """
        if mode == "complete":
            # F.unfold only works with tensors of shape [b, c, *]
            *B, N, _ = x.shape
            x = rearrange(x, "b n (c h w) -> (b n) c h w", h=self.height, w=self.width)
            x_patchwise = self._pad_and_unfold(x)  # [(b n), patchsize, num_patches]
            x_patchwise = rearrange(
                x_patchwise, "(b n) patchsize num_patches -> b num_patches n patchsize", n=N
            )  # [b, num_patches, n, patchsize]

            y = rearrange(y, "... (c h w) -> (...) c h w", h=self.height, w=self.width)
            y_patchwise = self._pad_and_unfold(y)
            y_patchwise = rearrange(
                y_patchwise, "b patchsize num_patches -> b num_patches patchsize"
            )  # [b, num_patches, patchsize]
            return x_patchwise, y_patchwise
        elif mode == "per_var":
            *B, C, N, _ = x.shape
            x = rearrange(x, "b c n (h w) -> (b c n) 1 h w", h=self.height, w=self.width)
            x_patchwise = self._pad_and_unfold(x)  # [(b c n), patchsize, num_patches]
            x_patchwise = rearrange(
                x_patchwise,
                "(b c n) patchsize num_patches -> b c num_patches n patchsize",
                c=C,
                n=N,
            )

            y = rearrange(y, "b c (h w) -> (b c) 1 h w", h=self.height, w=self.width)
            y_patchwise = self._pad_and_unfold(y)  # [(b c), patchsize, num_patches]
            y_patchwise = rearrange(
                y_patchwise, "(b c) patchsize num_patches -> b c num_patches patchsize", c=C
            )
            return x_patchwise, y_patchwise

        else:
            raise ValueError(f"Mode {mode} not recognized. Use 'complete' or 'per_var'.")


class PatchwiseEnergyScore(EnergyScore, PatchwiseMixin):
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
        """
        EnergyScore.__init__(self, beta=beta, clamp=clamp)
        PatchwiseMixin.__init__(self, patch_size=patch_size, height=height, width=width)
        if normalize:
            self.normalization_factor = 1.0 / (patch_size**beta)
        else:
            self.normalization_factor = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "complete") -> torch.Tensor:
        x_patchwise, y_patchwise = self.patchify(x, y, mode)
        es = super().forward(x_patchwise, y_patchwise)
        if self.normalization_factor is not None:
            es = es * self.normalization_factor
        return reduce(es, "... num_patches -> ...", "mean")


class PatchwiseRBFScore(RBFScore, PatchwiseMixin):
    def __init__(
        self,
        lengthscale: float | None = None,
        patch_size: int = 3,
        height: int = 37,
        width: int = 31,
    ) -> None:
        """Patch-based RBF score with optional per-variable mode.

        Args:
            lengthscale (float | None, optional): RBF kernel lengthscale. If None, it is set to
                ``patch_size ** 2`` so that larger patches default to broader kernels. Default: None.
            patch_size (int, optional): Square patch side length used for unfolding inputs. Default: 3.
            height (int, optional): Height of the input grid before unfolding. Default: 37.
            width (int, optional): Width of the input grid before unfolding. Default: 31.

        Modes:
            - "complete": forward expects ``x`` shaped [batch, n_samples, c*h*w] and ``y`` shaped
              [batch, c*h*w]. Patches are extracted across all variables jointly.
            - "per_var": forward expects ``x`` shaped [batch, variables, n_samples, h*w] and ``y``
              shaped [batch, variables, h*w]. Patches are extracted separately per variable.
        """
        if lengthscale is None:
            lengthscale = patch_size**2
        RBFScore.__init__(self, lengthscale=lengthscale)
        PatchwiseMixin.__init__(self, patch_size=patch_size, height=height, width=width)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "complete") -> torch.Tensor:
        x_patchwise, y_patchwise = self.patchify(x, y, mode)
        s = super().forward(x_patchwise, y_patchwise)
        return reduce(s, "... num_patches -> ...", "mean")


class MultiPatchwiseRBFScore(nn.Module):
    def __init__(
        self,
        patch_sizes: list[int] = [3, 5, 7],
        height: int = 37,
        width: int = 31,
    ) -> None:
        super().__init__()
        self.patch_sizes = patch_sizes
        self.height = height
        self.width = width
        self.scores = nn.ModuleList(
            [
                PatchwiseRBFScore(
                    lengthscale=patch_size**2,
                    patch_size=patch_size,
                    height=height,
                    width=width,
                )
                for patch_size in self.patch_sizes
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "complete") -> torch.Tensor:
        score = self.scores[0].forward(x, y, mode=mode)
        for score_module in self.scores[1:] if len(self.scores) > 1 else []:
            score += score_module.forward(x, y, mode=mode)
        avg_score = score / len(self.scores)
        return avg_score
