import torch


def fid(
    features1: torch.Tensor | None = None,
    features2: torch.Tensor | None = None,
    mu1: torch.Tensor | None = None,
    mu2: torch.Tensor | None = None,
    sigma1: torch.Tensor | None = None,
    sigma2: torch.Tensor | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute the Fréchet Inception Distance (FID) between two distributions.

    This function can be called in multiple ways:
    1. With feature tensors (features1, features2): mean and covariance will be computed
    2. With precomputed statistics (mu1, mu2, sigma1, sigma2): directly use provided values
    3. Mixed mode: features for one distribution and precomputed statistics for the other

    Args:
        features1 (torch.Tensor | None): Feature tensor of shape [n_samples, vector_size]
        features2 (torch.Tensor | None): Feature tensor of shape [m_samples, vector_size]
        mu1 (torch.Tensor | None): Mean vector of the first distribution, shape [vector_size]
        mu2 (torch.Tensor | None): Mean vector of the second distribution, shape [vector_size]
        sigma1 (torch.Tensor | None): Covariance matrix of the first distribution, shape [vector_size, vector_size]
        sigma2 (torch.Tensor | None): Covariance matrix of the second distribution, shape [vector_size, vector_size]
        eps (float): Small value for numerical stability (default: 0.0)

    Returns:
        fid_score (torch.Tensor): the FID score between the two distributions

    Raises:
        ValueError: If insufficient arguments are provided for either distribution

    Examples:
        >>> # Using feature tensors
        >>> features1 = torch.randn(100, 2048)
        >>> features2 = torch.randn(100, 2048)
        >>> score = fid(features1=features1, features2=features2)

        >>> # Using precomputed statistics
        >>> mu1, mu2 = torch.randn(2048), torch.randn(2048)
        >>> sigma1, sigma2 = torch.eye(2048), torch.eye(2048)
        >>> score = fid(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)

        >>> # Mixed mode: features for first, precomputed for second
        >>> score = fid(features1=features1, mu2=mu2, sigma2=sigma2)

        >>> # Mixed mode: precomputed for first, features for second
        >>> score = fid(mu1=mu1, sigma1=sigma1, features2=features2)
    """
    # Process first distribution
    if features1 is not None:
        if mu1 is not None or sigma1 is not None:
            raise ValueError(
                "For distribution 1, provide either features1 or (mu1, sigma1), not both."
            )
        # Compute statistics from features
        mu1 = torch.mean(features1, dim=0)
        sigma1 = torch.cov(features1.T)
    elif mu1 is not None and sigma1 is not None:
        # Use precomputed statistics
        pass
    else:
        raise ValueError("For distribution 1, provide either features1 or both (mu1, sigma1).")

    # Process second distribution
    if features2 is not None:
        if mu2 is not None or sigma2 is not None:
            raise ValueError(
                "For distribution 2, provide either features2 or (mu2, sigma2), not both."
            )
        # Compute statistics from features
        mu2 = torch.mean(features2, dim=0)
        sigma2 = torch.cov(features2.T)
    elif mu2 is not None and sigma2 is not None:
        # Use precomputed statistics
        pass
    else:
        raise ValueError("For distribution 2, provide either features2 or both (mu2, sigma2).")

    # Ensure all tensors are on the same device and dtype
    mu2 = mu2.to(mu1.device, mu1.dtype)
    sigma1 = sigma1.to(mu1.device, mu1.dtype)
    sigma2 = sigma2.to(mu1.device, mu1.dtype)

    return _fid(mu1, mu2, sigma1, sigma2, eps)


def _fid(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute the Fréchet Inception Distance (FID) between two distributions.

    Args:
        mu1 (torch.Tensor): Mean vector of the first distribution, shape [vector_size]
        mu2 (torch.Tensor): Mean vector of the second distribution, shape [vector_size]
        sigma1 (torch.Tensor): Covariance matrix of the first distribution, shape [vector_size, vector_size]
        sigma2 (torch.Tensor): Covariance matrix of the second distribution, shape [vector_size, vector_size]
        eps (float): Small value for numerical stability (default: 0.0)

    Returns:
        fid_score (torch.Tensor): the FID score between the two distributions

    The FID is computed as:
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    where mu1, mu2 are the means and sigma1, sigma2 are the covariance matrices.
    """
    # Compute the squared L2 norm of the difference between means
    mean_diff = torch.sum((mu1 - mu2) ** 2)

    # Compute sqrt(sigma1 @ sigma2) using eigendecomposition
    # Add small diagonal term for numerical stability
    product = sigma1 @ sigma2
    product = product + eps * torch.eye(product.shape[0]).to(product)

    # Note: product should be positive semidefinite, so eigh is appropriate
    eigvals, _ = torch.linalg.eigh(product)

    # The sum of eigenvalues is the trace of the matrix, so no need to compute full matrix sqrt
    # Clamp negative eigenvalues to zero for numerical stability
    trace_sqrt = torch.clamp(eigvals, min=0).sqrt().sum()

    # Compute FID score
    trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * trace_sqrt
    fid_score = mean_diff + trace_term

    return fid_score
