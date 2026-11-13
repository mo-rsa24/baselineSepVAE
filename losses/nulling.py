"""
nulling.py — R3: nulling loss (push disease heads to zero on normal samples)
             and orthogonality loss (push disease heads apart).
"""
import torch
import torch.nn.functional as F


def nulling_loss(mu_disease: torch.Tensor, is_normal: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    mu_disease : (B, D) or (B, D, H, W) — disease-head posterior mean
    is_normal  : (B,) float — 1.0 for normal samples, 0.0 otherwise

    Returns
    -------
    Scalar loss: ‖μ_disease[normal]‖² (mean over normal samples, 0 if none).
    """
    if mu_disease.dim() > 2:
        # spatial latent → GAP first
        mu = mu_disease.mean(dim=(-2, -1))  # (B, D)
    else:
        mu = mu_disease

    mask = is_normal.bool()
    if mask.sum() == 0:
        return mu.new_tensor(0.0)
    return mu[mask].pow(2).mean()


def orthogonality_loss(mu_d1: torch.Tensor, mu_d2: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between disease-head means across the batch.
    Parameters may be spatial (B,D,H,W) or vector (B,D).
    """
    if mu_d1.dim() > 2:
        v1 = mu_d1.mean(dim=(-2, -1))
        v2 = mu_d2.mean(dim=(-2, -1))
    else:
        v1, v2 = mu_d1, mu_d2

    cos = F.cosine_similarity(v1, v2, dim=-1)  # (B,)
    return cos.abs().mean()
