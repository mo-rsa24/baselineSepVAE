"""
base.py — Shared loss primitives used by all experiments.
"""
import torch
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return mu
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) per sample, summed over latent dims."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (..., D)
    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=-1).mean()  # scalar


def mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)
