"""Energy score implementations."""

from .base import KernelScore
from .kernels.l2 import (
    L2_Beta_Kernel,
    MultiScaleL2_Beta_Kernel,
    MultiScalePatchwiseL2_Beta_Kernel,
    PatchwiseL2_Beta_Kernel,
)


class EnergyScore(KernelScore):
    """Energy score using powered L2 distances.

    This score wraps :class:`L2_Beta_Kernel` which computes ``||x - y||_2**beta``
    (optionally patchwise or multi-scale variants are available in this module).

    Args:
        beta (float): Exponent for the L2 distance (``0 < beta <= 2``).
        clamp (bool): Whether to clamp squared distances to avoid numerical issues.
        normalize (bool): Whether to normalize per-variable sums in the underlying
            kernel (use mean instead of sum), making the score invariant to
            variable/patch size.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self, beta: float = 1.0, clamp: bool = True, unbiased: bool = True, normalize: bool = False
    ) -> None:
        kernel = L2_Beta_Kernel(beta=beta, clamp=clamp, normalize=normalize)
        super().__init__(kernel=kernel, unbiased=unbiased)


class PatchwiseEnergyScore(KernelScore):
    """Energy score computed from local patches.

    Args:
        beta (float): Exponent for the L2 distance.
        clamp (bool): Whether to clamp squared distances.
        patch_size (int): Patch side length used for local comparisons.
        normalize (bool): Whether to normalize per-patch sums by patch area.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self,
        beta: float = 1.0,
        clamp: bool = True,
        patch_size: int = 3,
        normalize=True,
        unbiased: bool = True,
    ) -> None:
        kernel = PatchwiseL2_Beta_Kernel(
            beta=beta, clamp=clamp, patch_size=patch_size, normalize=normalize
        )
        super().__init__(kernel=kernel, unbiased=unbiased)


class MultiScaleEnergyScore(KernelScore):
    """Multi-scale Energy score combining blurred scales.

    Args:
        beta (float): Exponent for the L2 distance.
        clamp (bool): Whether to clamp squared distances.
        blur_kernel_sizes (list[int]): List of blur kernel sizes (e.g., [3, 5, 7]).
        scale_weights (list[float] | None): Optional per-scale weights.
        normalize (bool): Whether inner kernels should normalize per-variable sums
            (use mean instead of sum), to make scores robust to variable/patch size.
        unbiased (bool): Whether to use the unbiased estimator for the self-term.
    """

    def __init__(
        self,
        beta: float = 1.0,
        clamp: bool = True,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
        normalize: bool = False,
        unbiased: bool = True,
    ) -> None:
        kernel = MultiScaleL2_Beta_Kernel(
            beta=beta,
            clamp=clamp,
            blur_kernel_sizes=blur_kernel_sizes,
            scale_weights=scale_weights,
            normalize=normalize,
        )
        super().__init__(kernel=kernel, unbiased=unbiased)


class MultiScalePatchwiseEnergyScore(KernelScore):
    """Multi-scale patchwise Energy kernel score using the Multi-scale L2-based conditionally negative definite kernel.

    Combines patchwise kernels computed at multiple scales by downscaling predictions
    and targets using interpolation, then averaging scores across all scales.
    Args:
        beta (float): The beta parameter for the L2 norm.
        clamp (bool): Whether to clamp values to avoid numerical issues.
        blur_kernel_sizes (list[int]): List of kernel sizes for blurring at different scales.
        scale_weights (list[float] | None): Optional weights for each scale.
        patch_size (int): Size of the patches to extract.
        normalize (bool): Whether the inner patchwise kernel should normalize per-patch
            sums by patch area (use mean instead of sum).
    """

    def __init__(
        self,
        beta: float = 1.0,
        clamp: bool = True,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
        patch_size: int = 3,
        normalize: bool = False,
        unbiased: bool = True,
    ) -> None:
        kernel = MultiScalePatchwiseL2_Beta_Kernel(
            beta=beta,
            clamp=clamp,
            blur_kernel_sizes=blur_kernel_sizes,
            scale_weights=scale_weights,
            patch_size=patch_size,
            normalize=normalize,
        )
        super().__init__(kernel=kernel, unbiased=unbiased)
