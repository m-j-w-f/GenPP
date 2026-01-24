"""RBF-based kernel scores implementations.

This module provides RBFScore, PatchwiseRBFScore, MultiScaleRBFScore and
MultiPatchwiseRBFScore which mirror the Energy score classes but use RBF-based
conditionally negative definite kernels.
"""

import torch

from .base import KernelScore
from .kernels.rbf import (
    MultiScalePatchwiseRBF_Kernel,
    MultiScaleRBF_Kernel,
    PatchwiseRBF_Kernel,
    RBF_Kernel,
)


class RBFScore(KernelScore):
    """RBF-based kernel score.

    Wraps :class:`RBF_Kernel` and computes the kernel score using RBF-based
    conditional negative-definite values (i.e., negative RBF similarity).

    Args:
        lengthscales (float | list[float] | torch.Tensor): Lengthscale(s) for the RBF kernel.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self, lengthscales: torch.Tensor | float | list[float] = 1.0, unbiased: bool = True
    ) -> None:
        kernel = RBF_Kernel(lengthscales=lengthscales)
        super().__init__(kernel=kernel, unbiased=unbiased)


class PatchwiseRBFScore(KernelScore):
    """Patchwise RBF score that compares local patches using RBF similarity.

    Args:
        lengthscales (float | list[float] | torch.Tensor): Lengthscale(s) for the RBF kernel.
        patch_size (int): Patch side length used when extracting local patches.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self,
        lengthscales: torch.Tensor | list[float] | float = 1.0,
        patch_size: int = 3,
        unbiased: bool = True,
    ) -> None:
        kernel = PatchwiseRBF_Kernel(lengthscales=lengthscales, patch_size=patch_size)
        super().__init__(kernel=kernel, unbiased=unbiased)


class MultiScaleRBFScore(KernelScore):
    """Multi-scale RBF score combining blurred scales.

    Combines an unblurred scale and multiple blurred scales (via separable
    Gaussian kernels) and averages results, optionally using ``scale_weights``.

    Args:
        lengthscales (float | list[float] | torch.Tensor): Base lengthscale(s) for the RBF kernel.
        blur_kernel_sizes (list[int]): Blur kernel sizes (e.g., ``[3, 5, 7]``). A single
            size of ``[3]`` yields similar behavior to :class:`PatchwiseRBFScore`.
        scale_weights (list[float] | None): Optional per-scale weights.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self,
        lengthscales: torch.Tensor | list[float] | float = 1.0,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
        unbiased: bool = True,
    ) -> None:
        kernel = MultiScaleRBF_Kernel(
            lengthscales=lengthscales,
            blur_kernel_sizes=blur_kernel_sizes,
            scale_weights=scale_weights,
        )
        super().__init__(kernel=kernel, unbiased=unbiased)


class MultiScalePatchwiseRBFScore(KernelScore):
    """Multi-scale patchwise RBF score.

    Uses :class:`MultiScalePatchwiseRBF_Kernel` to compute patchwise comparisons
    across multiple blurred scales and averages results.

    Args:
        lengthscales (float | list[float]): Lengthscale(s) either applied uniformly
            or specified per (patch, scale) as needed by the underlying kernel.
        blur_kernel_sizes (list[int]): List of blur kernel sizes (e.g., ``[3,5,7]``).
        patch_size (int): Patch side length to extract.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self,
        lengthscales: torch.Tensor | list[float] | float = 1.0,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        patch_size: int = 3,
        unbiased: bool = True,
    ) -> None:
        kernel = MultiScalePatchwiseRBF_Kernel(
            lengthscales=lengthscales,
            blur_kernel_sizes=blur_kernel_sizes,
            scale_weights=None,
            patch_size=patch_size,
        )
        super().__init__(kernel=kernel, unbiased=unbiased)
