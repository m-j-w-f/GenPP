"""Variogram score implementation."""

import torch
import torch.nn as nn
from einops import rearrange, reduce

from .kernels.reshaping import _flatten_per_mode


class VariogramScore(nn.Module):
    """Computes the variogram score between predicted and true values.

    The variogram score evaluates how well the predicted samples capture
    the spatial correlation structure of the true values.

    Args:
        p (float): The power parameter for the variogram. Default: 0.5.
        chunk_size (int): Number of spatial locations to process at once to avoid OOM.
            Default: 256. Set to None to disable chunking (original behavior).
    """

    def __init__(self, p: float = 0.5, chunk_size: int | None = 256) -> None:
        super().__init__()
        self.p = p
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        """Core variogram score computation on flattened inputs.

        Args:
            x (torch.Tensor): Predicted values with shape [b, n_samples, c, h, w].
            y (torch.Tensor): True values with shape [b, c, h, w].

        Returns:
            torch.Tensor: Variogram score with shape [..., ].
        """
        x = _flatten_per_mode(x, mode=mode)  # [..., n_samples, var]
        y = _flatten_per_mode(y.unsqueeze(1), mode=mode)  # [..., 1, var]

        # If chunk_size is None or var is small enough, use original implementation
        var_size = y.shape[-1]
        if self.chunk_size is None or var_size <= self.chunk_size:
            return self._compute_full(x, y)
        else:
            return self._compute_chunked(x, y)

    def _compute_full(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Original full matrix computation (faster but memory-intensive)."""
        y_diff = rearrange(y, "... 1 var -> ... var 1") - y
        y_diff = torch.abs(y_diff) ** self.p  # [..., var, var]

        x_diff = rearrange(x, "... n var -> ... n var 1") - rearrange(x, "... n var -> ... n 1 var")
        x_diff = torch.abs(x_diff) ** self.p  # [..., n_samples, var, var]
        x_diff = reduce(x_diff, "... n var1 var2 -> ... var1 var2", "mean")

        total_diff = torch.pow(y_diff - x_diff, 2)  # [..., var, var]

        res = reduce(total_diff, "... var1 var2 -> ...", "sum")
        return res

    def _compute_chunked(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Memory-efficient chunked computation (slower but avoids OOM).

        Computes pairwise differences in chunks to avoid creating full [var, var] matrices.
        """
        # Get dimensions
        *batch_dims, n_samples, var_size = x.shape
        batch_shape = batch_dims if batch_dims else [1]

        # Reshape for easier processing
        x_flat = x.reshape(-1, n_samples, var_size)  # [batch, n_samples, var]
        y_flat = y.reshape(-1, 1, var_size)  # [batch, 1, var]
        batch_size = x_flat.shape[0]

        # Initialize accumulator for the sum of squared differences
        total_sum = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        # Process in chunks
        chunk_size = self.chunk_size
        num_chunks = (var_size + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, var_size)

            # Get chunk of y values for first dimension
            y_chunk_i = y_flat[:, :, start_i:end_i]  # [batch, 1, chunk]
            x_chunk_i = x_flat[:, :, start_i:end_i]  # [batch, n_samples, chunk]

            for j in range(num_chunks):
                start_j = j * chunk_size
                end_j = min((j + 1) * chunk_size, var_size)

                # Get chunk of y values for second dimension
                y_chunk_j = y_flat[:, :, start_j:end_j]  # [batch, 1, chunk]
                x_chunk_j = x_flat[:, :, start_j:end_j]  # [batch, n_samples, chunk]

                # Compute y_diff for this chunk: [batch, chunk_i, chunk_j]
                y_diff_chunk = torch.abs(
                    y_chunk_i.unsqueeze(-1) - y_chunk_j.unsqueeze(-2)
                ) ** self.p

                # Compute x_diff for this chunk: [batch, n_samples, chunk_i, chunk_j]
                x_diff_chunk = torch.abs(
                    x_chunk_i.unsqueeze(-1) - x_chunk_j.unsqueeze(-2)
                ) ** self.p
                # Average over samples
                x_diff_chunk = x_diff_chunk.mean(dim=1)  # [batch, chunk_i, chunk_j]

                # Compute squared difference and sum
                diff_sq = torch.pow(y_diff_chunk.squeeze(1) - x_diff_chunk, 2)
                total_sum += diff_sq.sum(dim=(-2, -1))

                # Free memory
                del y_diff_chunk, x_diff_chunk, diff_sq

        # Reshape back to original batch dimensions
        if batch_dims:
            total_sum = total_sum.reshape(*batch_dims)
        else:
            total_sum = total_sum.squeeze(0)

        return total_sum
