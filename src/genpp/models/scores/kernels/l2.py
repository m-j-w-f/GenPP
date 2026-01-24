import torch
from einops import rearrange

from .base import CondNegDefKernel
from .reshaping import (
    _apply_blur_kernels,
    _flatten_per_mode,
    _patchwise_flatten_per_mode,
    _precompute_blur_kernels,
)


class L2_Beta_Kernel(CondNegDefKernel):
    """L2-based kernel returning element-wise Euclidean distance to the power ``beta``.

    This implementation returns ``||x - y||_2**beta`` (non-negative). The function
    ``f(x, y) = ||x - y||_2**beta`` for ``0 < beta <= 2`` is conditionally negative
    definite, which makes it suitable for energy-score style losses.

    Args:
        beta (float): Exponent for the L2 norm (``0 < beta <= 2``).
        clamp (bool): Whether to clamp squared distances to avoid numerical issues.
        normalize (bool): Whether to normalize the per-variable squared-sum by the
            number of variables (uses mean instead of sum). When True, the kernel
            becomes invariant to the number of variables or patch size.
    """

    def __init__(self, beta: float = 1.0, clamp: bool = True, normalize: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.clamp = clamp
        self.normalize = normalize
        if clamp:
            self.eps = 1e-8
            self.max_value = 1e10

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute pairwise ``||x - y||_2**beta`` between flattened representations.

        Args:
            x (torch.Tensor): Tensor of shape ``[b, n_samples_x, c, h, w]``.
            y (torch.Tensor): Tensor of shape ``[b, n_samples_y, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"``.

        Returns:
            torch.Tensor: Pairwise powered L2 distances with shape ``[..., n_samples_x, n_samples_y]``.
        """
        x_flat = _flatten_per_mode(x, mode)  # shape [..., n_samples_x, variables]
        y_flat = _flatten_per_mode(y, mode)  # shape [..., n_samples_y, variables]

        diff = rearrange(x_flat, "... n_x var -> ... n_x 1 var") - rearrange(
            y_flat, "... n_y var -> ... 1 n_y var"
        )  # shape [..., n_samples_x, n_samples_y, variables]

        if self.normalize:
            sq_diff_sum = (
                torch.mean(diff**2, dim=-1)  # Mean over variables to account for dimension size
            )  # shape [..., n_samples_x, n_samples_y]
        else:
            sq_diff_sum = torch.sum(diff**2, dim=-1)  # shape [..., n_samples_x, n_samples_y]

        if self.clamp:
            sq_diff_sum = torch.clamp(sq_diff_sum, min=self.eps, max=self.max_value)

        return torch.sqrt(sq_diff_sum) ** self.beta  # shape [..., n_samples_x, n_samples_y]


class PatchwiseL2_Beta_Kernel(CondNegDefKernel):
    """Patchwise L2 kernel computing mean powered-L2 over local patches.

    The returned value is the average (over patches) of ``||patch_x - patch_y||_2**beta``.

    Args:
        beta (float): Exponent for the L2 norm.
        clamp (bool): Whether to clamp squared distances.
        patch_size (int): Patch side length used when extracting local patches.
        normalize (bool): Whether to normalize per-patch sums by the number of pixels in the patch.
    """

    def __init__(
        self, beta: float = 1.0, clamp: bool = True, patch_size: int = 3, normalize: bool = False
    ) -> None:
        super().__init__()
        self.beta = beta
        self.clamp = clamp
        self.patch_size = patch_size
        self.normalize = normalize
        if clamp:
            self.eps = 1e-8
            self.max_value = 1e10

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute patchwise powered-L2 distances and average across patches.

        Args:
            x (torch.Tensor): Tensor of shape ``[b, n_samples_x, c, h, w]``.
            y (torch.Tensor): Tensor of shape ``[b, n_samples_y, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"``.

        Returns:
            torch.Tensor: Patchwise averaged powered-L2 distances with shape
                ``[..., n_samples_x, n_samples_y]``.
        """
        x_flat = _patchwise_flatten_per_mode(
            x, mode, self.patch_size
        )  # shape [..., n_samples_x, num_patches, patchsize]
        y_flat = _patchwise_flatten_per_mode(
            y, mode, self.patch_size
        )  # shape [..., n_samples_y, num_patches, patchsize]

        diff = rearrange(
            x_flat, "... n_x n_patches patchsize -> ... n_x 1 n_patches patchsize"
        ) - rearrange(
            y_flat, "... n_y n_patches patchsize -> ... 1 n_y n_patches patchsize"
        )  # shape [..., n_samples_x, n_samples_y, num_patches, patchsize]

        if self.normalize:
            sq_diff_sum = (
                torch.mean(diff**2, dim=-1)  # Mean over patchsize to account for dimension size
            )  # shape [..., n_samples_x, n_samples_y, num_patches]
        else:
            sq_diff_sum = torch.sum(
                diff**2, dim=-1
            )  # shape [..., n_samples_x, n_samples_y, num_patches]

        if self.clamp:
            sq_diff_sum = torch.clamp(sq_diff_sum, min=self.eps, max=self.max_value)

        reduced = (torch.sqrt(sq_diff_sum) ** self.beta).mean(
            dim=-1
        )  # shape [..., n_samples_x, n_samples_y]
        return reduced


class MultiScaleL2_Beta_Kernel(CondNegDefKernel):
    """Multi-scale Energy kernel score using multiple L2-based conditionally negative definite kernels.

    Args:
        beta (float): The beta parameter for the L2 norm.
        clamp (bool): Whether to clamp values to avoid numerical issues.
        blur_kernel_sizes (list[int]): List of blur kernel sizes used to form blurred
            downscaled versions of the inputs (one scale per kernel size).
        scale_weights (list[float] | None): Optional weights applied to each scale
            (including the original unblurred scale). If None, equal weights are used.
        normalize (bool): Whether the inner L2 kernels should normalize per-variable
            sums (use mean instead of sum) to make scale comparisons invariant to
            variable count/patch size.
    """

    def __init__(
        self,
        beta: float = 1.0,
        clamp: bool = True,
        blur_kernel_sizes: list[int] = [3, 5, 7],
        scale_weights: list[float] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.n_scales = len(blur_kernel_sizes) + 1  # +1 for the original scale
        if scale_weights is None:
            scale_weights = [1.0] * self.n_scales
        assert len(scale_weights) == self.n_scales, (
            "Length of scale_weights must be number of scales + 1 as the unscaled score is included."
        )
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

        self._inner_kernel = L2_Beta_Kernel(beta=beta, clamp=clamp, normalize=normalize)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        x_blurred = _apply_blur_kernels(
            x, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]
        y_blurred = _apply_blur_kernels(
            y, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]

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
        # Get L2 Beta Kernel
        res_per_scale = self._inner_kernel(
            x_all_scales, y_all_scales, mode
        )  # [(b n_scales), (c)] dimension c exists only in "per_var" mode

        # Reshape back to separate scales
        res_per_scale = rearrange(
            res_per_scale,
            "(b n_scales) ... -> b ... n_scales",
            n_scales=self.n_scales,
        )
        # Weighted average across scales
        weighted_res = (res_per_scale * self.scale_weights).sum(dim=-1) / (self.scale_weights.sum())
        return weighted_res


class MultiScalePatchwiseL2_Beta_Kernel(CondNegDefKernel):
    """Multi-scale Energy kernel score using multiple L2-based conditionally negative definite kernels.

    Args:
        beta (float): The beta parameter for the L2 norm.
        clamp (bool): Whether to clamp values to avoid numerical issues.
        blur_kernel_sizes (list[int]): List of blur kernel sizes used for blurring at
            different scales.
        scale_weights (list[float] | None): Optional weights for each scale (including the
            original scale). If None, equal weights are used.
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
    ) -> None:
        super().__init__()
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

        self._inner_kernel = PatchwiseL2_Beta_Kernel(
            beta=beta, clamp=clamp, patch_size=patch_size, normalize=normalize
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        x_blurred = _apply_blur_kernels(
            x, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]
        y_blurred = _apply_blur_kernels(
            y, self.blur_weights, self.pad_size
        )  # [b, n_samples, n_scales, c, h, w]

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

        # Get Patchwise L2 Beta Kernel
        res_per_scale = self._inner_kernel(
            x_all_scales, y_all_scales, mode
        )  # [(b n_scales), (c)] dimension c exists only in "per_var" mode

        # Reshape back to separate scales
        res_per_scale = rearrange(
            res_per_scale,
            "(b n_scales) ... -> b ... n_scales",
            n_scales=self.n_scales,
        )
        # Weighted average across scales
        weighted_res = (res_per_scale * self.scale_weights).sum(dim=-1) / (self.scale_weights.sum())
        return weighted_res
