import torch
import torch.nn as nn
from einops import rearrange, reduce


class EnergyScore(nn.Module):
    """Computes the energy score between predicted and true values.

    Args:
        beta (float): The beta parameter for the energy score.
        clamp (bool): Whether to clamp the values to avoid numerical issues.
    """

    def __init__(self, beta: float = 1.0, clamp: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.clamp = clamp
        if clamp:
            self.eps = 1e-8
            self.max_value = 1e10

    def l2_beta_norm(self, diff: torch.Tensor) -> torch.Tensor:
        sq_diff = (diff) ** 2
        sq_diff_sum = reduce(sq_diff, "... spatial -> ... 1", reduction="sum")
        if self.clamp:
            sq_diff_sum = torch.clamp(sq_diff_sum, min=self.eps, max=self.max_value)
        reduced = reduce(torch.sqrt(sq_diff_sum) ** self.beta, "b d ... -> b d", reduction="mean")
        return reduced

    def forward(self, x: torch.Tensor, y: torch.Tensor, avg: str | None = "mean") -> torch.Tensor:
        """Computes the energy score between the predicted and true values.

        Args:
            x (torch.Tensor): The predicted values with shape [batch_size, n_samples, out_features, lon, lat].
            y (torch.Tensor): The true values with shape [batch_size, out_features, lon, lat].

        Returns:
            torch.Tensor: The computed energy score with shape [out_features].
        """

        batch_size, n_samples, lat, lon, out_features = x.shape

        # Reshape tensors for easier computation
        # x: [batch_size, out_features, n_samples, lat * lon]
        # y: [batch_size, out_features, 1, lat * lon]
        x_reshaped = rearrange(x, "b n d lon lat -> b d n (lon lat)")
        y_reshaped = rearrange(y, "b d lon lat -> b d 1 (lon lat)")

        # Calculate first term: E[||y_pred - y_true||]
        es_12 = self.l2_beta_norm(x_reshaped - y_reshaped)

        # Calculate second term: E[||y_pred_i - y_pred_j||] for i != j
        G = torch.matmul(
            x_reshaped, rearrange(x_reshaped, "b d n spatial -> b d spatial n")
        )  # [batch_size, out_features, n_samples, n_samples]

        # Extract diagonal elements (||y_pred_i||^2)
        d = rearrange(torch.diagonal(G, dim1=-2, dim2=-1), "b d n -> b d n 1")

        # Compute pairwise distances: ||y_pred_i||^2 + ||y_pred_j||^2 - 2 * y_pred_i^T * y_pred_j
        distances_22 = d + rearrange(d, "b d n 1 -> b d 1 n") - 2 * G
        if self.clamp:
            # Clamp distances to avoid numerical issues
            distances_22 = torch.clamp(distances_22, min=self.eps, max=self.max_value)

        # Sum over all pairs (including diagonal, but we'll account for that)
        es_22 = reduce(
            torch.sqrt(distances_22), "b d n1 n2 -> b d", reduction="mean"
        )  # [batch_size, out_features]
        es = es_12 - 0.5 * es_22
        if avg == "mean":
            return torch.mean(es)
        if avg == "variable":
            return torch.mean(es, dim=0)
        return es
