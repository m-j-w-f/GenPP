from abc import ABC

import torch
from torch import nn

from .kernels.base import CondNegDefKernel


class KernelScore(nn.Module, ABC):
    """Base class for kernel-based scores.

    The :class:`KernelScore` wraps a conditionally negative definite kernel and
    computes the corresponding score from samples and observations.

    Args:
        kernel (CondNegDefKernel): Kernel instance used to compute pairwise values.
        unbiased (bool): If True, use an unbiased estimator for the self-term
            (averaging only off-diagonal entries). Default is ``True``.
    """

    def __init__(self, kernel: CondNegDefKernel, unbiased: bool = True) -> None:
        super().__init__()
        self.kernel = kernel
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute the kernel score between sample ensembles ``x`` and observations ``y``.

        The implementation follows the standard kernel score formula:
        ``score = mean_x k(x, y) - 0.5 * mean_{i,j} k(x_i, x_j)`` with an
        option to use an unbiased estimator for the second term.

        Args:
            x (torch.Tensor): Samples with shape ``[b, n_samples, c, h, w]``.
            y (torch.Tensor): Observations with shape ``[b, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"`` to control
                reshaping within kernels.

        Returns:
            torch.Tensor: Score tensor. The exact output shape depends on ``mode``
                (e.g., ``[b]`` for complete mode or ``[b, c]`` for per_var).
        """
        kxy = self.kernel(x, y.unsqueeze(1), mode)  # shape [..., n_samples_x, n_samples_y]
        term1 = kxy.squeeze(-1).mean(dim=-1)

        kxx = self.kernel(x, x, mode)  # shape [..., n_samples_x, n_samples_x]
        if self.unbiased:
            n = kxx.shape[-1]

            sum_all = kxx.sum(dim=(-2, -1))
            sum_diag = kxx.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

            # Use the unbiased off-diagonal estimator: average over i != j
            term2 = (sum_all - sum_diag) / (n * (n - 1))
        else:
            term2 = kxx.mean(dim=(-2, -1))

        score = term1 - 0.5 * term2
        return score
