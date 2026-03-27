import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _flatten_per_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Helper function to flatten tensors based on the specified mode.

    Args:
        x (torch.Tensor): Input tensor with shape [b, n_samples, c, h, w].
        mode (str): Mode of flattening, either "complete" or "per_var".

    Returns:
        torch.Tensor: Flattened tensor.
            - "complete" mode: shape [b, n_samples, c*h*w]
            - "per_var" mode: shape [b, c, n_samples, h*w]
    """
    if mode == "complete":
        # Reshape: x [b, n, c, h, w] -> [b, n, c*h*w], y [b, c, h, w] -> [b, c*h*w]
        x_flat = rearrange(x, "b n c h w -> b n (c h w)")
        return x_flat
    elif mode == "per_var":
        # Reshape: x [b, n, c, h, w] -> [b, c, n, h*w], y [b, c, h, w] -> [b, c, h*w]
        x_flat = rearrange(x, "b n c h w -> b c n (h w)")
        return x_flat
    else:
        raise ValueError(f"Mode {mode} not recognized. Use 'complete' or 'per_var'.")


def _patchwise_flatten_per_mode(x: torch.Tensor, mode: str, patch_size: int) -> torch.Tensor:
    """Extract patches and flatten ``x`` according to ``mode``.

    Args:
        x (torch.Tensor): Input tensor with shape ``[b, n_samples, c, h, w]``.
        mode (str): Flattening mode. Either ``"complete"`` or ``"per_var"``.
        patch_size (int): Patch side length to extract (odd preferred for symmetric padding).

    Returns:
        torch.Tensor: Patchwise representation:
            - ``"complete"``: ``[b, n_samples, num_patches, patchsize]``
            - ``"per_var"``: ``[b, c, n_samples, num_patches, patchsize]``

    Raises:
        ValueError: If ``mode`` is not supported.
    """
    B, N, C, H, W = x.shape
    patch = (patch_size, patch_size)
    padding = (patch_size - 1) // 2

    if mode == "complete":
        x = rearrange(x, "b n c h w -> (b n) c h w")
    elif mode == "per_var":
        x = rearrange(x, "b n c h w -> (b c n) 1 h w")
    else:
        raise ValueError(f"Mode {mode} not recognized. Use 'complete' or 'per_var'.")

    x_padded = nn.functional.pad(x, (padding, padding, padding, padding), mode="reflect")

    x_patchwise = torch.nn.functional.unfold(
        x_padded, kernel_size=patch, stride=1
    )  # [B, patchsize, num_patches]

    if mode == "complete":
        x_patchwise = rearrange(
            x_patchwise,
            "(b n) patchsize num_patches -> b n num_patches patchsize",
            n=N,
            b=B,
        )  # [b, n, num_patches, patchsize]
        return x_patchwise
    elif mode == "per_var":
        x_patchwise = rearrange(
            x_patchwise,
            "(b c n) patchsize num_patches -> b c n num_patches patchsize",
            b=B,
            c=C,
            n=N,
        )  # [b, c, n, num_patches, patchsize]
        return x_patchwise


def _precompute_blur_kernels(
    kernel_sizes: list[int],
    sigmas: list[float] | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Precompute separable 2D Gaussian blur kernels for a set of sizes.

    The function returns kernels stacked along the first dimension and the padding
    size required for the largest kernel.

    Args:
        kernel_sizes (list[int]): Kernel sizes to compute (typically odd integers).
        sigmas (list[float] | None): Optional standard deviations per kernel. If
            not provided sensible defaults are computed.
        device (torch.device): Device for the returned tensors.
        dtype (torch.dtype): Data type for the kernels.

    Returns:
        Tuple[torch.Tensor, int]: ``(weights, pad_size)`` where ``weights`` has
            shape ``[N, 1, max_k, max_k]`` and ``pad_size`` is ``max_k // 2``.
    """
    N = len(kernel_sizes)
    max_k = max(kernel_sizes)
    pad_size = max_k // 2

    if sigmas is None:
        sigmas = [0.3 * ((k - 1) * 0.5 - 1) + 0.8 for k in kernel_sizes]

    k_sizes = torch.tensor(kernel_sizes, device=device)
    s_vals = torch.tensor(sigmas, device=device, dtype=dtype)
    coords = torch.arange(max_k, device=device, dtype=dtype) - (max_k - 1) / 2
    grid = coords.view(1, -1).expand(N, -1)

    mask = grid.abs() <= (k_sizes.view(-1, 1) - 1) / 2
    pdf = torch.exp(-0.5 * (grid / s_vals.view(-1, 1)).pow(2)) * mask
    kernel1d = pdf / pdf.sum(dim=-1, keepdim=True)

    # Shape: [N, 1, max_k, max_k]
    weights = (kernel1d.unsqueeze(-1) * kernel1d.unsqueeze(-2)).unsqueeze(1)

    return weights, pad_size


def _apply_blur_kernels(x: torch.Tensor, weights: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Apply precomputed blur kernels to ``x`` and return multiple blurred scales.

    Args:
        x (torch.Tensor): Input with shape ``[..., C, H, W]``. Leading dimensions
            are treated as batch-like.
        weights (torch.Tensor): Blur kernels of shape ``[N, 1, K, K]`` returned by
            :func:`_precompute_blur_kernels`.
        pad_size (int): Padding size to apply before convolution.

    Returns:
        torch.Tensor: Tensor reshaped to ``[..., N, C, H, W]`` where ``N`` is the
            number of blur scales; channels are treated independently.
    """
    original_shape = x.shape
    batch_dims = original_shape[:-3]
    C, H, W = original_shape[-3:]
    N = weights.shape[0]

    # 1. Collapse all leading dimensions AND channels into batch
    # [..., C, H, W] -> [Total_Batch * C, 1, H, W]
    # This treats every channel as an independent grayscale image
    x_flat = x.reshape(-1, 1, H, W)

    # 2. Reflection Padding
    x_padded = F.pad(x_flat, [pad_size] * 4, mode="reflect")

    # 3. Convolution
    # Input: [B*C, 1, H, W], Weights: [N, 1, K, K]
    # Output: [B*C, N, H, W] (Each channel now has N blurred versions)
    output = F.conv2d(x_padded, weights)

    # 4. Reshape and Reorder
    # [B*C, N, H, W] -> [..., C, N, H, W]
    output = output.view(*batch_dims, C, N, H, W)

    # Move N to the front of C: [..., N, C, H, W]
    output = output.transpose(-4, -3)

    return output
