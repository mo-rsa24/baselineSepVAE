"""
disease_clf_loss.py — R11: disease discriminability classifier loss.

Small per-head classifier trained cooperatively with the encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseClassifier(nn.Module):
    """GAP(z_disease) → scalar disease-presence logit."""

    def __init__(self, z_ch: int, hidden: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_ch, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_mu: torch.Tensor) -> torch.Tensor:
        """
        z_mu: (B, C) vector or (B, C, H, W) spatial — GAP applied here.
        Returns (B,) logits.
        """
        if z_mu.dim() > 2:
            z_mu = z_mu.mean(dim=(-2, -1))
        return self.fc(z_mu).squeeze(-1)


def stable_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def clf_bce_loss(
    classifier: DiseaseClassifier,
    z_mu: torch.Tensor,
    has_disease: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    classifier  : DiseaseClassifier
    z_mu        : (B, C) or (B, C, H, W) — disease head posterior mean
    has_disease : (B,) float — 1.0 if disease present, 0.0 if not

    Returns
    -------
    Scalar BCE loss.
    """
    logits = classifier(z_mu)
    return stable_bce(logits, has_disease)
