import torch
from einops import rearrange

from genpp.models.scores import EnergyScore as ES_NEW


def manual_unbiased_complete(x, y):
    x_flat = rearrange(x, "b n c h w -> b n (c h w)")
    y_flat = rearrange(y, "b c h w -> b (c h w)")
    term1 = torch.norm(x_flat - y_flat.unsqueeze(1), dim=-1).mean(dim=-1)
    d = torch.norm(x_flat.unsqueeze(2) - x_flat.unsqueeze(1), dim=-1)  # [b, n, n]
    n = d.shape[-1]
    off_diag_mean = (d.sum(dim=(-2, -1)) - d.diagonal(dim1=-2, dim2=-1).sum(dim=-1)) / (n * (n - 1))
    return term1 - 0.5 * off_diag_mean


def manual_unbiased_per_var(x, y):
    x_flat = rearrange(x, "b n c h w -> b c n (h w)")
    y_flat = rearrange(y, "b c h w -> b c (h w)")
    term1 = torch.norm(x_flat - y_flat.unsqueeze(2), dim=-1).mean(dim=2)  # mean over n -> [b, c]
    d = torch.norm(x_flat.unsqueeze(3) - x_flat.unsqueeze(2), dim=-1)  # [b, c, n, n]
    n = d.shape[-1]
    off_diag_mean = (d.sum(dim=(-2, -1)) - d.diagonal(dim1=-2, dim2=-1).sum(dim=-1)) / (n * (n - 1))
    return term1 - 0.5 * off_diag_mean


def test_kernel_unbiased_complete():
    torch.manual_seed(0)
    b, n, c, h, w = 4, 10, 2, 8, 9
    y = torch.randn(b, c, h, w)
    x = torch.randn(b, n, c, h, w)

    es_new = ES_NEW(unbiased=True)

    res_new = es_new(x, y, mode="complete")
    expected = manual_unbiased_complete(x, y)

    assert torch.allclose(res_new, expected, rtol=1e-6, atol=1e-6)


def test_kernel_unbiased_per_var():
    torch.manual_seed(1)
    b, n, c, h, w = 3, 8, 3, 6, 7
    y = torch.randn(b, c, h, w)
    x = torch.randn(b, n, c, h, w)

    es_new = ES_NEW(unbiased=True)

    res_new = es_new(x, y, mode="per_var")
    expected = manual_unbiased_per_var(x, y)

    assert torch.allclose(res_new, expected, rtol=1e-6, atol=1e-6)
