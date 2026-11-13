"""
moment_matching.py — E5: Weinberger MM-cLVM alignment loss.

Aligns z_c marginals between background (B) and salient (S) sets
via matching first and second moments.
"""
import torch


def moment_matching_loss(
    z_c_b: torch.Tensor,
    z_c_s: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    z_c_b : (N_B, D) — z_c samples from background set
    z_c_s : (N_S, D) — z_c samples from salient set

    Returns
    -------
    Scalar: ‖μ_B − μ_S‖² + ‖Σ_B − Σ_S‖_F²
    """
    mu_b, mu_s = z_c_b.mean(0), z_c_s.mean(0)
    mean_loss = (mu_b - mu_s).pow(2).sum()

    # Unbiased covariance estimates
    cov_b = _cov(z_c_b)
    cov_s = _cov(z_c_s)
    cov_loss = (cov_b - cov_s).pow(2).sum()

    return mean_loss + cov_loss


def _cov(x: torch.Tensor) -> torch.Tensor:
    """(N, D) → (D, D) unbiased sample covariance."""
    n = x.shape[0]
    x_c = x - x.mean(0, keepdim=True)
    return (x_c.T @ x_c) / max(n - 1, 1)
