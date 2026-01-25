from abc import ABC, abstractmethod

import torch
from torch import nn


class CondNegDefKernel(nn.Module, ABC):
    """Base class for conditionally negative definite kernels.

    Kernel implementations compute pairwise values between two sample sets and are
    used by kernel-based scoring rules.

    Note:
        Subclasses should implement :meth:`forward` and return a tensor of shape
        ``[..., n_samples_x, n_samples_y]``.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute pairwise kernel values between ``x`` and ``y``.

        Args:
            x (torch.Tensor): Tensor of shape ``[b, n_samples_x, c, h, w]``.
            y (torch.Tensor): Tensor of shape ``[b, n_samples_y, c, h, w]``.
            mode (str): Either ``"complete"`` or ``"per_var"`` indicating how spatial
                and channel dimensions are flattened prior to pairwise computation.

        Returns:
            torch.Tensor: Kernel values with shape ``[..., n_samples_x, n_samples_y]``.

        Raises:
            ValueError: If ``mode`` is not recognized by the kernel implementation.
        """
        pass
