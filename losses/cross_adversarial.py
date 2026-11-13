"""
cross_adversarial.py — R5b: cross-head adversarial discriminator.

Trains a small discriminator to predict z_d2 from z_d1 (and vice versa).
Encoder loss tries to fool the discriminator → pushes information out of wrong head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossHeadDiscriminator(nn.Module):
    """Predicts one disease head's GAP embedding from the other's."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def cross_adv_discriminator_loss(
    disc: CrossHeadDiscriminator,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator step: minimise MSE(disc(source), target.detach()).
    source, target: (B, D) GAP'd latent vectors.
    """
    pred = disc(source.detach())
    return F.mse_loss(pred, target.detach())


def cross_adv_encoder_loss(
    disc: CrossHeadDiscriminator,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Encoder step: maximise MSE(disc(source), target) → fool discriminator.
    Gradients flow through source; disc weights are frozen externally.
    """
    pred = disc(source)
    return -F.mse_loss(pred, target.detach())
