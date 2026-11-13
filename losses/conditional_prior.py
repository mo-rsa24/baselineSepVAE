"""
conditional_prior.py — E6: Zou label-conditioned prior p(z_s|y).

For y=normal:  p(z_s|y) = N(0, I)
For y=disease: p(z_s|y) = N(μ_y, I)  where μ_y is learned per class.

This is the key ingredient missing from superdiff-ldm's SepVAE,
which uses nulling loss as an approximation instead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelConditionedPrior(nn.Module):
    """
    Learns one mean vector μ_y per disease class.
    Normal class is always μ=0 (no parameter needed).

    Parameters
    ----------
    n_disease_classes : int — number of disease classes (not counting normal)
    z_dim             : int — dimensionality of z_s
    """

    def __init__(self, n_disease_classes: int, z_dim: int):
        super().__init__()
        # Index 0 = normal (no parameter — always 0)
        # Index i = disease class i  (i = 1..n_disease_classes)
        self.mu_table = nn.Embedding(n_disease_classes + 1, z_dim)
        nn.init.zeros_(self.mu_table.weight)  # start all at 0

    def get_prior_mu(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,) long in {0=normal, 1=d1, 2=d2, …}"""
        return self.mu_table(labels)  # (B, z_dim)


def kl_to_label_prior(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
) -> torch.Tensor:
    """
    KL(q(z_s|x) ∥ N(μ_p, I)) — closed form.

    Parameters
    ----------
    mu_q, logvar_q : (B, D) — encoder posterior
    mu_p           : (B, D) — label-conditioned prior mean

    Returns
    -------
    Scalar: mean over batch of sum over dims.
    """
    # KL = 0.5 * (σ² + (μ_q − μ_p)² − 1 − log σ²)
    kl = 0.5 * (logvar_q.exp() + (mu_q - mu_p).pow(2) - 1 - logvar_q)
    return kl.sum(dim=-1).mean()
