"""
contrastive.py — R5a: margin contrastive loss between disease heads.

Pushes z_d1 and z_d2 embeddings apart in the latent space.
"""
import torch
import torch.nn.functional as F


def margin_contrastive_loss(
    mu_d1: torch.Tensor,
    mu_d2: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Cosine-distance contrastive loss.
    Maximises distance between z_d1 and z_d2 per sample.

    Parameters
    ----------
    mu_d1, mu_d2 : (B, D) — disease head means (GAP'd if spatial)
    margin       : minimum desired cosine distance

    Returns
    -------
    Scalar: mean(max(0, margin - distance(d1, d2)))
    """
    if mu_d1.dim() > 2:
        mu_d1 = mu_d1.mean(dim=(-2, -1))
        mu_d2 = mu_d2.mean(dim=(-2, -1))

    cos_sim = F.cosine_similarity(mu_d1, mu_d2, dim=-1)   # (B,)
    dist    = 1.0 - cos_sim                                 # in [0, 2]
    loss    = F.relu(margin - dist).mean()
    return loss
