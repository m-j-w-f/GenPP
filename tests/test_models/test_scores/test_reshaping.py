import pytest
import torch

from genpp.models.scores.kernels.reshaping import (
    _apply_blur_kernels,
    _precompute_blur_kernels,
)

# Skip tests if torchvision v2 gaussian_blur isn't available
gaussian_blur = pytest.importorskip(
    "torchvision.transforms.v2.functional", reason="needs torchvision v2"
).gaussian_blur


def _device_and_dtype():
    # prioritize CPU for tests; allow running on CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    return device, dtype


@pytest.mark.unit
def test_apply_blur_kernels_matches_torchvision_single_scale():
    device, dtype = _device_and_dtype()
    torch.manual_seed(0)

    B, C, H, W = 2, 3, 32, 32
    x = torch.rand(B, C, H, W, device=device, dtype=dtype)

    kernel_sizes = [3, 5, 7]
    sigmas = [0.5, 1.0, 1.5]

    weights, pad_size = _precompute_blur_kernels(kernel_sizes, sigmas, device=device, dtype=dtype)

    out = _apply_blur_kernels(x, weights, pad_size)
    # out shape: [B, N, C, H, W]
    assert out.shape == (B, len(kernel_sizes), C, H, W)

    for i, k in enumerate(kernel_sizes):
        tv = gaussian_blur(x, kernel_size=k, sigma=sigmas[i])
        torch.testing.assert_close(out[:, i], tv, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_apply_blur_kernels_matches_torchvision_with_batch_dims():
    device, dtype = _device_and_dtype()
    torch.manual_seed(1)

    B, N, C, H, W = 2, 4, 3, 24, 24
    x = torch.rand(B, N, C, H, W, device=device, dtype=dtype)

    kernel_sizes = [3, 5]
    sigmas = [0.7, 1.2]

    weights, pad_size = _precompute_blur_kernels(kernel_sizes, sigmas, device=device, dtype=dtype)

    out = _apply_blur_kernels(x, weights, pad_size)
    # out shape: [B, N, Nscales, C, H, W]
    assert out.shape == (B, N, len(kernel_sizes), C, H, W)

    for b in range(B):
        for n in range(N):
            img = x[b, n]
            for i, k in enumerate(kernel_sizes):
                tv = gaussian_blur(img, kernel_size=k, sigma=sigmas[i])
                torch.testing.assert_close(out[b, n, i], tv, rtol=1e-5, atol=1e-6)
