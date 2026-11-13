"""
shared.py — Reusable building blocks for all encoder/decoder architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Convolutional blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → SiLU (encoder building block, stride=2 → halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, groups: int = 1):
        super().__init__()
        n_groups = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """ConvTranspose2d → GroupNorm → SiLU (decoder building block, stride=2 → doubles dims)."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 1):
        super().__init__()
        n_groups = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder head: outputs (mu, logvar) at current spatial resolution
# ---------------------------------------------------------------------------

class ConvHead(nn.Module):
    """1×1 conv split into mu and logvar outputs."""

    def __init__(self, in_ch: int, z_ch: int):
        super().__init__()
        self.mu_conv     = nn.Conv2d(in_ch, z_ch, 1)
        self.logvar_conv = nn.Conv2d(in_ch, z_ch, 1)

    def forward(self, feat: torch.Tensor):
        mu     = self.mu_conv(feat)
        logvar = self.logvar_conv(feat).clamp(-10, 2)
        return mu, logvar


# ---------------------------------------------------------------------------
# Label-guided attention head (A3 / R7 analog)
# ---------------------------------------------------------------------------

class LabelAttentionHead(nn.Module):
    """
    Produces a spatial attention map conditioned on a disease label embedding.
    Returns (mu, logvar, attn_map) where attn_map is (B, H, W).

    Parameters
    ----------
    feat_ch   : number of input feature channels
    z_ch      : disease latent channels
    query_dim : internal projection dim for the label embedding
    """

    def __init__(self, feat_ch: int, z_ch: int, query_dim: int = 64, n_labels: int = 2):
        super().__init__()
        self.label_embed = nn.Embedding(n_labels, query_dim)
        self.query_proj  = nn.Linear(query_dim, feat_ch)
        self.head        = ConvHead(feat_ch, z_ch)

    def forward(self, feat: torch.Tensor, label: torch.Tensor):
        """
        feat  : (B, C, H, W)
        label : (B,) long — 0=absent, 1=present
        """
        q = self.query_proj(self.label_embed(label))        # (B, C)
        q = q[:, :, None, None]                             # (B, C, 1, 1)
        attn_logits = (feat * q).sum(dim=1)                 # (B, H, W)
        attn_map    = F.softmax(attn_logits.view(feat.shape[0], -1), dim=-1)
        attn_map    = attn_map.view_as(attn_logits)
        # Scale so uniform map ≈ 1.0
        H, W = feat.shape[-2:]
        attn_map    = attn_map * (H * W)

        attended = feat * attn_map.unsqueeze(1)
        mu, logvar = self.head(attended)
        return mu, logvar, attn_map
