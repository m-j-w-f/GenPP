from collections.abc import Sequence

import torch
from einops import rearrange

from .base import CondNegDefKernel
from .reshaping import (
    _apply_blur_kernels,
    _flatten_per_mode,
    _patchwise_flatten_per_mode,
    _precompute_blur_kernels,
)


def _rbf_kernel(diff: torch.Tensor, lengthscales: torch.Tensor) -> torch.Tensor:
    """Compute RBF similarity values for a set of differences.

    The returned value is ``exp(-||diff||^2 / (d * 2 * l^2))`` averaged across provided
    lengthscales, where ``d`` is the variable dimension used to normalize the
    squared distance in this implementation.

    Args:
        diff (torch.Tensor): Difference tensor of shape ``[..., variables]``.
        lengthscales (torch.Tensor): Lengthscale(s) for the RBF kernel with shape ``[L]``.

    Returns:
        torch.Tensor: RBF values with shape ``[...]`` (averaged across lengthscales).
    """
    d = diff.shape[-1]
    sq_diff_sum = torch.sum(diff**2, dim=-1) * 1 / d  # normalize by dimension
    rbf = torch.exp(-sq_diff_sum.unsqueeze(-1) / (2 * lengthscales.to(sq_diff_sum) ** 2))
    rbf = rbf.mean(dim=-1)  # average over lengthscales if multiple are provided
    return rbf


def _sanitize_tensor(t: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
    """Convert ``t`` to a :class:`torch.Tensor`.

    Args:
        t (torch.Tensor | float | Sequence[float]): Input value(s) to sanitize.

    Returns:
        torch.Tensor: Tensor representation of ``t``.

    Raises:
        TypeError: If ``t`` is not a tensor, number, or sequence of numbers.
    """
    if isinstance(t, torch.Tensor):
        return t
    elif isinstance(t, float) or isinstance(t, int):
        return torch.tensor(t)
    elif isinstance(t, Sequence):
        return torch.tensor(t)
    else:
        raise TypeError("Input must be a torch.Tensor, float, int, or a sequence of floats.")


class RBF_Kernel(CondNegDefKernel):
    """Radial Basis Function (RBF) kernel wrapper.

    This class wraps an RBF similarity computation and returns the *negative*
    of the RBF similarity per pair, which yields a conditionally negative
    definite value suitable for kernel-based scoring rules.

    Args:
        lengthscales (float | list[float] | torch.Tensor): Lengthscale(s) used
            when computing the RBF similarity.
    """

    def __init__(self, lengthscales: torch.Tensor | list[float] | float = 1.0) -> None:
        super().__init__()
        self.lengthscales = _sanitize_tensor(lengthscales)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute pairwise negative RBF kernel values between ``x`` and ``y``.

        Args:
            x (torch.Tensor): Tensor of shape ``[b, n_samples_x, c, h, w]``.
            y (torch.Tensor): Tensor of shape ``[b, n_samples_y, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"``.

        Returns:
            torch.Tensor: Negative RBF kernel values with shape ``[..., n_samples_x, n_samples_y]``.
        """
        x_flat = _flatten_per_mode(x, mode)  # shape [..., n_samples_x, variables]
        y_flat = _flatten_per_mode(y, mode)  # shape [..., n_samples_y, variables]

        diff = rearrange(x_flat, "... n_x var -> ... n_x 1 var") - rearrange(
            y_flat, "... n_y var -> ... 1 n_y var"
        )  # shape [..., n_samples_x, n_samples_y, variables]

        rbf_values = _rbf_kernel(diff, self.lengthscales)

        return -rbf_values  # shape [..., n_samples_x, n_samples_y]


class PatchwiseRBF_Kernel(CondNegDefKernel):
    """Patchwise RBF kernel that computes RBF similarity patch-wise and negates.

    Args:
        lengthscales (float | list[float] | torch.Tensor): Lengthscale(s) for
            computing RBF similarity within each patch.
        patch_size (int): Patch side length used when extracting local patches.
    """

    def __init__(
        self, lengthscales: torch.Tensor | list[float] | float = 1.0, patch_size: int = 3
    ) -> None:
        super().__init__()
        self.lengthscales = _sanitize_tensor(lengthscales)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute negative RBF kernel values by comparing patches.

        Args:
            x (torch.Tensor): Tensor of shape ``[b, n_samples_x, c, h, w]``.
            y (torch.Tensor): Tensor of shape ``[b, n_samples_y, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"``.

        Returns:
            torch.Tensor: Negative patchwise RBF kernel values with shape
                ``[..., n_samples_x, n_samples_y]``.
        """
        x_flat = _patchwise_flatten_per_mode(
            x, mode, self.patch_size
        )  # shape [..., n_samples_x, num_patches, patchsize]
        y_flat = _patchwise_flatten_per_mode(
            y, mode, self.patch_size
        )  # shape [..., n_samples_y, num_patches, patchsize]

        diff = rearrange(x_flat, "... n_x p ps -> ... n_x 1 p ps") - rearrange(
            y_flat, "... n_y p ps -> ... 1 n_y p ps"
        )  # shape [..., n_samples_x, n_samples_y, num_patches, patchsize]

        rbf_values = _rbf_kernel(
            diff, self.lengthscales
        )  # shape [..., n_samples_x, n_samples_y, num_patches]

        mean_rbf = rbf_values.mean(dim=-1)  # shape [..., n_samples_x, n_samples_y]

        return -mean_rbf  # shape [..., n_samples_x, n_samples_y]


class MultiScaleRBF_Kernel(CondNegDefKernel):
    """Multi-scale Radial Basis Function (RBF) conditionally negative definite kernel used in kernel scores.

    The kernel combines RBF kernels computed at multiple blur scales and averages them (optionally weighted).
    """

    def __init__(
        self,
        lengthscales: torch.Tensor | list[float] | float = 1.0,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        # Include original (unblurred) scale in addition to blur scales
        self.n_scales = len(blur_kernel_sizes) + 1  # +1 for the original scale
        if scale_weights is None:
            scale_weights = [1.0] * self.n_scales

        # Weights per scale
        self.scale_weights: torch.Tensor
        self.register_buffer("scale_weights", torch.tensor(scale_weights))

        # Weights for blurring kernels
        weights, pad_size = _precompute_blur_kernels(
            kernel_sizes=blur_kernel_sizes,
        )
        self.blur_weights: torch.Tensor
        self.register_buffer("blur_weights", weights)
        self.pad_size = pad_size

        self._inner_kernel = RBF_Kernel(lengthscales=lengthscales)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        x_blurred = _apply_blur_kernels(
            x, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]
        y_blurred = _apply_blur_kernels(
            y, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]

        # Include the original scale as the first scale
        x_all_scales = torch.cat(
            [x.unsqueeze(2), x_blurred], dim=2
        )  # [b, n_samples, n_scales+1, c, h, w]
        y_all_scales = torch.cat(
            [y.unsqueeze(2), y_blurred], dim=2
        )  # [b, n_samples, n_scales+1, c, h, w]

        # Merge blur scales into batch dimension for kernel computation
        x_all_scales = rearrange(
            x_all_scales, "b n_samples n_scales c h w -> (b n_scales) n_samples c h w"
        )
        y_all_scales = rearrange(
            y_all_scales, "b n_samples n_scales c h w -> (b n_scales) n_samples c h w"
        )

        # Get RBF Kernel
        res_per_scale = self._inner_kernel(x_all_scales, y_all_scales, mode)

        # Reshape back to separate scales
        res_per_scale = rearrange(
            res_per_scale,
            "(b n_scales) ... -> b ... n_scales",
            n_scales=self.n_scales,
        )
        # Weighted average across scales
        weighted_res = (res_per_scale * self.scale_weights).sum(dim=-1) / (self.scale_weights.sum())
        return weighted_res


class MultiScalePatchwiseRBF_Kernel(CondNegDefKernel):
    """Multi-scale Radial Basis Function (RBF) conditionally negative definite kernel used in kernel scores.

    The kernel is defined as the average of patchwise RBF kernels computed across multiple blur scales.
    Blurring creates multiple scales and results are averaged (optionally weighted) across scales.
    """

    def __init__(
        self,
        lengthscales: torch.Tensor | list[float] | float = 1.0,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
        patch_size: int = 3,
    ) -> None:
        super().__init__()
        # Include original (unblurred) scale in addition to blur scales
        self.n_scales = len(blur_kernel_sizes) + 1  # +1 for the original scale
        if scale_weights is None:
            scale_weights = [1.0] * self.n_scales

        # Weights per scale
        self.scale_weights: torch.Tensor
        self.register_buffer("scale_weights", torch.tensor(scale_weights))

        # Weights for blurring kernels
        weights, pad_size = _precompute_blur_kernels(
            kernel_sizes=blur_kernel_sizes,
        )
        self.blur_weights: torch.Tensor
        self.register_buffer("blur_weights", weights)
        self.pad_size = pad_size

        self._inner_kernel = PatchwiseRBF_Kernel(lengthscales=lengthscales, patch_size=patch_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        x_blurred = _apply_blur_kernels(
            x, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]
        y_blurred = _apply_blur_kernels(
            y, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]

        # Include the original scale as the first scale
        x_all_scales = torch.cat(
            [x.unsqueeze(2), x_blurred], dim=2
        )  # [b, n_samples, n_scales+1, c, h, w]
        y_all_scales = torch.cat(
            [y.unsqueeze(2), y_blurred], dim=2
        )  # [b, n_samples, n_scales+1, c, h, w]

        # Merge blur scales into batch dimension for kernel computation
        x_all_scales = rearrange(
            x_all_scales, "b n_samples n_scales c h w -> (b n_scales) n_samples c h w"
        )
        y_all_scales = rearrange(
            y_all_scales, "b n_samples n_scales c h w -> (b n_scales) n_samples c h w"
        )

        # Get Patchwise RBF Kernel
        res_per_scale = self._inner_kernel(x_all_scales, y_all_scales, mode)

        # Reshape back to separate scales
        res_per_scale = rearrange(
            res_per_scale,
            "(b n_scales) ... -> b ... n_scales",
            n_scales=self.n_scales,
        )
        # Weighted average across scales
        weighted_res = (res_per_scale * self.scale_weights).sum(dim=-1) / (self.scale_weights.sum())
        return weighted_res
