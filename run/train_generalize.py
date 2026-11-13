"""
train_generalize.py — Unified d0–d5 Generalisation Suite
==========================================================
Demonstrates that the d0–d5 SepVAE framework is a generalisable abstraction
of Weakly-Supervised Compositional Disentanglement (WSCD), not a CXR-specific
construction.

The 2×2 factorial domain design:

    ┌──────────────────────────┬─────────────────────────┐
    │                          │ Independent (A1 ✓)      │  Correlated (A1 ✗)     │
    ├──────────────────────────┼─────────────────────────┼──────────────────────┤
    │ Spatial (A2 ✓)           │ Domain A — StampedMNIST │ Domain B — CorrStamps │
    │ No spatial (A2 ✗)        │ Domain C — TextureFact. │ Domain D — CorrTexts  │
    └──────────────────────────┴─────────────────────────┴──────────────────────┘

    Domain A = positive control (all assumptions met)
    Domain B = A1 violated: spurious correlation ≈ 30%
    Domain C = A2 violated: global Fourier textures, no bbox
    Domain D = A1 + A2 violated: documented failure mode

Stage-to-loss mapping (mirrors d0–d5 in baselineSepVAE):

    d0 — rec + KL                 (base reconstruction)
    d1 — d0 + null + spatial gate (routing + spatial grounding, A/B only)
    d2 — d1 + MI discriminator    (orthogonalisation)
    d3 — d2 + disease clf         (discriminability pressure)
    d4 — d3 + orthogonality loss  (Barlow-style cross-correlation)
    d5 — d4 + contrastive         (prototype separation, not yet active)

Architecture (reuses SynDisSepVAE from train_syndis_sepvae.py):
    Encoder: 3-layer CNN (64→32→16→8) → common head + 2 disease heads
    Decoder: 3-layer transposed CNN (8→16→32→64)
    Optional: SpatialGateHead (1×1 conv → sigmoid) on disease heads for D1+

Usage
-----
    # Domain A, stage d5 (full suite)
    python -m run.train_generalize --domain A --stage d5 --epochs 100 --wandb

    # Domain C, stage d2 (MI only, no spatial prior)
    python -m run.train_generalize --domain C --stage d2

    # Run all 24 combinations via launcher
    bash launchers/single_runs/generalize/run_all_domains_d0_d5.sh

Output
------
    runs_generalize/<domain>_<stage>_<exp_name>/
        checkpoints/
        diagnostics/   (recon grids, composition grids, latent PCAs)
    results/domain_comparison_table.csv   (one row per run, all metrics)
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ─── Reuse SynDis model and loss utilities (domain-agnostic) ─────────────────
from run.train_syndis_sepvae import (
    SynDisSepVAE, SynDisEncoder, SynDisDecoder, DiseaseClassifier,
    kl_loss, nulling_loss, orthogonality_loss, stable_bce,
    encode_dataset, evaluate_disentanglement,
)

# ─── Dataset label constants (same values across all domains) ─────────────────
from datasets.SynDis import (
    LABEL_NORMAL, LABEL_DISEASE1 as LABEL_FACTOR1,
    LABEL_DISEASE2 as LABEL_FACTOR2, LABEL_BOTH,
)

# ─── Domain-specific region constants ─────────────────────────────────────────
from datasets.SplatteredMNIST import (
    SLOT_F1_ROWS, SLOT_F1_COLS, SLOT_F2_ROWS, SLOT_F2_COLS,
    BBOX_F1 as SPLAT_BBOX_F1, BBOX_F2 as SPLAT_BBOX_F2,
)
import datasets.CelebAFactor as _celeba_mod


# =============================================================================
# Optional spatial gate (D1 for domains A and B)
# =============================================================================

class SpatialGateHead(nn.Module):
    """
    Lightweight spatial attention gate for D1.
    Applies a learned 8×8 mask to the encoder feature map before each
    disease head, supervised by BCE against the known bbox mask.

    Architecture: Conv2d(64→16, 1×1) → ReLU → Conv2d(16→1, 1×1) → Sigmoid
    """
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.gate(feature_map)   # (B, 1, H_lat, W_lat)


class GeneralizeSepVAE(SynDisSepVAE):
    """
    Extends SynDisSepVAE with optional SpatialGateHead modules on disease heads.
    When use_spatial_gate=True, the encoder feature map is spatially gated
    before each disease head (mimicking bbox cross-attention from D1 in CXR).

    in_channels : int
        1 for greyscale (Domains A/B), 3 for RGB (Domains C/D/E/F).
        If >1, replaces the first encoder conv and the last decoder conv.
    """
    def __init__(self, z_ch_common: int = 4, z_ch_disease: int = 2,
                 use_spatial_gate: bool = False, in_channels: int = 1):
        super().__init__(z_ch_common, z_ch_disease)
        self.use_spatial_gate = use_spatial_gate
        if in_channels != 1:
            # Patch encoder input conv
            self.encoder.backbone[0] = nn.Conv2d(
                in_channels, 16, 3, stride=2, padding=1)
            # Patch decoder output conv (net[-2] = Conv2d(16, 1, 1); net[-1] = Tanh)
            self.decoder.net[-2] = nn.Conv2d(16, in_channels, 1)
        if use_spatial_gate:
            self.gate_d1 = SpatialGateHead(64)
            self.gate_d2 = SpatialGateHead(64)

    def forward(self, x: torch.Tensor):
        h = self.encoder.backbone(x)   # (B, 64, 8, 8)

        mu_c, lv_c = self.encoder.common_head(h)

        if self.use_spatial_gate:
            attn_d1 = self.gate_d1(h)             # (B, 1, 8, 8)
            attn_d2 = self.gate_d2(h)
            mu_d1, lv_d1 = self.encoder.disease1_head(h * attn_d1)
            mu_d2, lv_d2 = self.encoder.disease2_head(h * attn_d2)
        else:
            attn_d1 = attn_d2 = None
            mu_d1, lv_d1 = self.encoder.disease1_head(h)
            mu_d2, lv_d2 = self.encoder.disease2_head(h)

        z_c  = self.reparameterise(mu_c,  lv_c)
        z_d1 = self.reparameterise(mu_d1, lv_d1)
        z_d2 = self.reparameterise(mu_d2, lv_d2)

        x_rec = self.decoder(z_c, z_d1, z_d2)

        return x_rec, {
            "mu_c":   mu_c,  "lv_c":  lv_c,
            "mu_d1": mu_d1,  "lv_d1": lv_d1,
            "mu_d2": mu_d2,  "lv_d2": lv_d2,
            "z_c":   z_c,    "z_d1":  z_d1,   "z_d2": z_d2,
            "attn_d1": attn_d1, "attn_d2": attn_d2,
        }


# =============================================================================
# MI discriminator (D2: FactorVAE-style)
# =============================================================================

class FactorVAEDiscriminator(nn.Module):
    """
    Small MLP that distinguishes joint q(z_d1, z_d2) from product q(z_d1)×q(z_d2).
    Inputs are GAP'd (globally average-pooled) latent vectors.
    """
    def __init__(self, z1_dim: int = 4, z2_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z1_dim + z2_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),           nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, z1_gap: torch.Tensor, z2_gap: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z1_gap, z2_gap], dim=1)).squeeze(-1)   # (B,)


def mi_loss_vae(disc: FactorVAEDiscriminator,
                z_d1_gap: torch.Tensor, z_d2_gap: torch.Tensor) -> torch.Tensor:
    """
    Encoder-side MI loss: push z_d1 and z_d2 toward product of marginals.
    Encoder wants to fool the discriminator → maximise its output on joint.
    Gradient on encoder: `loss = mean(logit_joint)` (higher logit = more confused disc).
    """
    return disc(z_d1_gap, z_d2_gap).mean()


def mi_loss_disc(disc: FactorVAEDiscriminator,
                 z_d1_gap: torch.Tensor, z_d2_gap: torch.Tensor) -> torch.Tensor:
    """
    Discriminator-side MI loss: learn to distinguish joint from permuted marginals.
    """
    B    = z_d1_gap.shape[0]
    perm = torch.randperm(B, device=z_d1_gap.device)
    logit_joint = disc(z_d1_gap,          z_d2_gap)
    logit_perm  = disc(z_d1_gap, z_d2_gap[perm])
    loss_real   = F.binary_cross_entropy_with_logits(
                      logit_joint, torch.ones_like(logit_joint))
    loss_fake   = F.binary_cross_entropy_with_logits(
                      logit_perm,  torch.zeros_like(logit_perm))
    return (loss_real + loss_fake) * 0.5


def disc_accuracy(disc: FactorVAEDiscriminator,
                  z_d1_gap: torch.Tensor, z_d2_gap: torch.Tensor) -> float:
    """Fraction of correct discriminator predictions (should → 0.5 when disentangled)."""
    with torch.no_grad():
        B    = z_d1_gap.shape[0]
        perm = torch.randperm(B, device=z_d1_gap.device)
        lj   = disc(z_d1_gap, z_d2_gap).sigmoid()
        lp   = disc(z_d1_gap, z_d2_gap[perm]).sigmoid()
        n_correct = ((lj > 0.5).sum() + (lp < 0.5).sum()).item()
        return n_correct / (2 * B)


# =============================================================================
# Spatial gate loss (D1, domains A and B only)
# =============================================================================

def _bbox_to_latent_mask(bbox_norm: tuple, lat_h: int, lat_w: int) -> np.ndarray:
    """Convert a normalised [x_min, y_min, x_max, y_max] bbox to a binary
    latent-resolution mask. x → columns, y → rows."""
    x_min, y_min, x_max, y_max = bbox_norm
    r_min = int(y_min * lat_h)
    r_max = max(r_min + 1, int(y_max * lat_h))
    c_min = int(x_min * lat_w)
    c_max = max(c_min + 1, int(x_max * lat_w))
    mask  = np.zeros((lat_h, lat_w), dtype=np.float32)
    mask[r_min:r_max, c_min:c_max] = 1.0
    return mask


def spatial_gate_loss(attn_d1: torch.Tensor, attn_d2: torch.Tensor,
                      mask_d1: torch.Tensor, mask_d2: torch.Tensor) -> torch.Tensor:
    """
    BCE loss: gate should activate (≈1) inside bbox, suppress (≈0) outside.

    attn_d{1,2}: (B, 1, H_lat, W_lat)
    mask_d{1,2}: (1, 1, H_lat, W_lat) precomputed binary masks
    """
    L1 = F.binary_cross_entropy(attn_d1, mask_d1.expand_as(attn_d1))
    L2 = F.binary_cross_entropy(attn_d2, mask_d2.expand_as(attn_d2))
    return (L1 + L2) * 0.5


# =============================================================================
# Norm ratio metric
# =============================================================================

@torch.no_grad()
def compute_norm_ratio(model, val_loader, device) -> Tuple[float, float]:
    """
    Compute ||mu_d1||_F (active) / ||mu_d1||_F (inactive) for both heads.
    Should be >> 1 when nulling is working correctly.
    """
    all_z_d1, all_z_d2, all_labels = [], [], []
    for x, y in val_loader:
        x = x.to(device)
        _, lat = model(x)
        all_z_d1.append(lat["mu_d1"].flatten(1).cpu().numpy())
        all_z_d2.append(lat["mu_d2"].flatten(1).cpu().numpy())
        all_labels.append(y.numpy())

    z_d1    = np.concatenate(all_z_d1)
    z_d2    = np.concatenate(all_z_d2)
    labels  = np.concatenate(all_labels)

    mask_f1_active   = (labels == LABEL_FACTOR1) | (labels == LABEL_BOTH)
    mask_f1_inactive = labels == LABEL_NORMAL
    mask_f2_active   = (labels == LABEL_FACTOR2) | (labels == LABEL_BOTH)
    mask_f2_inactive = labels == LABEL_NORMAL

    def ratio(z, act_mask, inact_mask):
        norm_act   = np.linalg.norm(z[act_mask],   axis=1).mean() + 1e-8
        norm_inact = np.linalg.norm(z[inact_mask], axis=1).mean() + 1e-8
        return float(norm_act / norm_inact)

    r1 = ratio(z_d1, mask_f1_active, mask_f1_inactive)
    r2 = ratio(z_d2, mask_f2_active, mask_f2_inactive)
    return r1, r2


# =============================================================================
# Domain-specific composition evaluators
# =============================================================================

@torch.no_grad()
def evaluate_composition_stamps(model, val_ds, device, n: int = 50) -> dict:
    """
    Composition test for Domains A and B (stamp datasets).
    Decodes z_c (normal) + z_d1 (factor1 source) + z_d2 (factor2 source).
    Measures pixel-level signal increase in known stamp regions.
    """
    model.eval()

    def gather_mu(label, key, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        _, lat = model(xs)
        return lat[key]

    mu_c  = gather_mu(LABEL_NORMAL,  "mu_c",  n)
    mu_d1 = gather_mu(LABEL_FACTOR1, "mu_d1", n)
    mu_d2 = gather_mu(LABEL_FACTOR2, "mu_d2", n)
    n_actual = min(len(mu_c), len(mu_d1), len(mu_d2))
    mu_c, mu_d1, mu_d2 = mu_c[:n_actual], mu_d1[:n_actual], mu_d2[:n_actual]

    x_comp = model.decoder(mu_c, mu_d1, mu_d2).cpu().numpy()
    x_comp = (x_comp + 1) / 2   # → [0, 1]

    z_null  = torch.zeros_like(mu_d1)
    x_null  = model.decoder(mu_c, z_null, z_null).cpu().numpy()
    x_null  = (x_null + 1) / 2

    r1, c1 = SLOT_F1_ROWS, SLOT_F1_COLS   # factor1 slot region
    r2, c2 = SLOT_F2_ROWS, SLOT_F2_COLS   # factor2 slot region

    d1_delta = float(x_comp[:, 0, r1[0]:r1[1], c1[0]:c1[1]].mean()
                     - x_null[:, 0, r1[0]:r1[1], c1[0]:c1[1]].mean())
    d2_delta = float(x_comp[:, 0, r2[0]:r2[1], c2[0]:c2[1]].mean()
                     - x_null[:, 0, r2[0]:r2[1], c2[0]:c2[1]].mean())

    return {
        "f1_delta":          d1_delta,
        "f2_delta":          d2_delta,
        "composition_score": d1_delta + d2_delta,
    }


@torch.no_grad()
def _orientation_proxy(imgs_np: np.ndarray) -> float:
    """
    Mean absolute principal-axis angle for a batch of greyscale images.
    imgs_np: (B, H, W) in [0, 1].  Returns mean angle in [0, π/2].
    Large = rotated shape (high orientation factor).
    """
    B, H, W = imgs_np.shape
    yg, xg  = np.mgrid[0:H, 0:W].astype(np.float64)
    angles   = []
    for b in range(B):
        m     = imgs_np[b].astype(np.float64)
        total = m.sum() + 1e-8
        cx    = (m * xg).sum() / total
        cy    = (m * yg).sum() / total
        mxx   = (m * (xg - cx) ** 2).sum() / total
        myy   = (m * (yg - cy) ** 2).sum() / total
        mxy   = (m * (xg - cx) * (yg - cy)).sum() / total
        angle = 0.5 * abs(np.arctan2(2 * mxy, mxx - myy + 1e-10))
        angles.append(angle)
    return float(np.mean(angles))


def evaluate_composition_dsprites(model, val_ds, device, n: int = 50) -> dict:
    """
    Composition test for Domains C and D (dSprites).

    Factor 1 = large scale  → composed image should have higher mean pixel value.
    Factor 2 = high orient  → composed image should have larger principal axis angle.

    Metrics
    -------
    f1_delta : mean(x_comp) - mean(x_null)   (>0 ⟹ larger shape decoded)
    f2_delta : orient_proxy(x_comp) - orient_proxy(x_null)  (>0 ⟹ more rotated)
    composition_score : f1_delta + f2_delta
    """
    model.eval()

    def gather_mu(label, key, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        _, lat = model(xs)
        return lat[key]

    mu_c  = gather_mu(LABEL_NORMAL,  "mu_c",  n)
    mu_d1 = gather_mu(LABEL_FACTOR1, "mu_d1", n)
    mu_d2 = gather_mu(LABEL_FACTOR2, "mu_d2", n)
    n_act = min(len(mu_c), len(mu_d1), len(mu_d2))
    mu_c, mu_d1, mu_d2 = mu_c[:n_act], mu_d1[:n_act], mu_d2[:n_act]

    x_comp = ((model.decoder(mu_c, mu_d1, mu_d2).cpu().numpy() + 1) / 2)

    z_null = torch.zeros_like(mu_d1)
    x_null = ((model.decoder(mu_c, z_null, z_null).cpu().numpy() + 1) / 2)

    # Squeeze channel dim for greyscale dSprites (B,1,H,W) → (B,H,W)
    x_comp_g = x_comp[:, 0] if x_comp.shape[1] == 1 else x_comp.mean(1)
    x_null_g = x_null[:, 0] if x_null.shape[1] == 1 else x_null.mean(1)

    f1_delta = float(x_comp_g.mean() - x_null_g.mean())
    f2_delta = _orientation_proxy(x_comp_g) - _orientation_proxy(x_null_g)

    return {
        "f1_delta":          f1_delta,
        "f2_delta":          f2_delta,
        "composition_score": f1_delta + f2_delta,
    }


@torch.no_grad()
def evaluate_composition_spatial_rgb(model, val_ds, device, n: int = 50) -> dict:
    """
    Composition test for Domains E and F (CelebA).
    Pixel region delta using known attribute bboxes.
    Uses luminance (mean of 3 channels) so works for both grey and RGB.
    """
    model.eval()

    def gather_mu(label, key, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        _, lat = model(xs)
        return lat[key]

    mu_c  = gather_mu(LABEL_NORMAL,  "mu_c",  n)
    mu_d1 = gather_mu(LABEL_FACTOR1, "mu_d1", n)
    mu_d2 = gather_mu(LABEL_FACTOR2, "mu_d2", n)
    n_actual = min(len(mu_c), len(mu_d1), len(mu_d2))
    mu_c, mu_d1, mu_d2 = mu_c[:n_actual], mu_d1[:n_actual], mu_d2[:n_actual]

    x_comp = model.decoder(mu_c, mu_d1, mu_d2).cpu().numpy()
    x_comp = (x_comp + 1) / 2   # (B, C, H, W)
    lum_comp = x_comp.mean(axis=1)   # (B, H, W) luminance

    z_null = torch.zeros_like(mu_d1)
    x_null = model.decoder(mu_c, z_null, z_null).cpu().numpy()
    x_null = (x_null + 1) / 2
    lum_null = x_null.mean(axis=1)

    H = W = lum_comp.shape[-1]
    # CelebA bbox for glasses and bald regions
    x1f1, y1f1, x2f1, y2f1 = _celeba_mod.BBOX_F1
    x1f2, y1f2, x2f2, y2f2 = _celeba_mod.BBOX_F2
    r1 = (int(y1f1 * H), int(y2f1 * H))
    c1 = (int(x1f1 * W), int(x2f1 * W))
    r2 = (int(y1f2 * H), int(y2f2 * H))
    c2 = (int(x1f2 * W), int(x2f2 * W))

    d1_delta = float(lum_comp[:, r1[0]:r1[1], c1[0]:c1[1]].mean()
                     - lum_null[:, r1[0]:r1[1], c1[0]:c1[1]].mean())
    d2_delta = float(lum_comp[:, r2[0]:r2[1], c2[0]:c2[1]].mean()
                     - lum_null[:, r2[0]:r2[1], c2[0]:c2[1]].mean())

    return {
        "f1_delta":          d1_delta,
        "f2_delta":          d2_delta,
        "composition_score": d1_delta + d2_delta,
    }


# =============================================================================
# Visualisation helpers (adapted from train_syndis_sepvae.py)
# =============================================================================

def _img_for_display(tensor_chw):
    """Convert (C, H, W) tensor in [-1, 1] to numpy for imshow."""
    arr = np.clip((tensor_chw.cpu().numpy() + 1) / 2, 0, 1)
    if arr.shape[0] == 1:
        return arr[0], "gray"          # (H, W), greyscale
    return arr.transpose(1, 2, 0), None  # (H, W, 3), RGB


@torch.no_grad()
def save_recon_grid(model, val_ds, device, path: Path, n_per_class: int = 4):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    class_labels = [LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2]
    class_names  = ["Normal", "Factor-1", "Factor-2"]

    fig, axes = plt.subplots(len(class_labels) * 2, n_per_class,
                             figsize=(n_per_class * 1.8, len(class_labels) * 4))
    for row, (cls, name) in enumerate(zip(class_labels, class_names)):
        idxs = np.where(val_ds.labels == cls)[0][:n_per_class]
        xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        x_rec, _ = model(xs)
        for col in range(len(idxs)):
            orig_arr, cmap = _img_for_display(xs[col])
            rec_arr,  _    = _img_for_display(x_rec[col])
            axes[row * 2,     col].imshow(orig_arr, cmap=cmap, vmin=0, vmax=1)
            axes[row * 2 + 1, col].imshow(rec_arr,  cmap=cmap, vmin=0, vmax=1)
            axes[row * 2,     col].axis("off")
            axes[row * 2 + 1, col].axis("off")
        axes[row * 2,     0].set_ylabel(f"{name}\n(orig)", fontsize=7)
        axes[row * 2 + 1, 0].set_ylabel("recon", fontsize=7)
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_composition_grid(model, val_ds, device, path: Path, n: int = 4):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    has_both = (val_ds.labels == LABEL_BOTH).sum() > 0
    n_cols   = 5 if has_both else 4

    def get_xs(label, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        return torch.stack([val_ds[i][0] for i in idxs]).to(device)

    xs_n  = get_xs(LABEL_NORMAL,  n)
    xs_d1 = get_xs(LABEL_FACTOR1, n)
    xs_d2 = get_xs(LABEL_FACTOR2, n)
    xs_b  = get_xs(LABEL_BOTH, n) if has_both else None

    _, lat_n  = model(xs_n)
    _, lat_d1 = model(xs_d1)
    _, lat_d2 = model(xs_d2)
    x_comp = model.decoder(lat_n["mu_c"], lat_d1["mu_d1"], lat_d2["mu_d2"])

    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 1.8, n * 1.8))
    titles = ["Normal (z_c)", "F1 source (z_d1)", "F2 source (z_d2)", "Composed", "GT Both"]
    for i in range(n):
        srcs = [xs_n[i], xs_d1[i], xs_d2[i], x_comp[i]]
        if has_both and xs_b is not None:
            srcs.append(xs_b[i])
        for j, img_t in enumerate(srcs):
            img_arr, cmap = _img_for_display(img_t)
            axes[i, j].imshow(img_arr, cmap=cmap, vmin=0, vmax=1)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=7)
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_pca_scatter(model, val_ds, device, path: Path):
    """PCA scatter of z_d1 and z_d2 GAP vectors, coloured by label."""
    try:
        from sklearn.decomposition import PCA
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    z_d1, z_d2, _, labels = encode_dataset(model, DataLoader(val_ds, batch_size=256), device)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {LABEL_NORMAL: "grey", LABEL_FACTOR1: "steelblue",
              LABEL_FACTOR2: "tomato", LABEL_BOTH: "purple"}
    cnames = {LABEL_NORMAL: "Normal", LABEL_FACTOR1: "F1", LABEL_FACTOR2: "F2", LABEL_BOTH: "Both"}

    for ax, (z, title) in zip(axes, [(z_d1, "z_d1"), (z_d2, "z_d2")]):
        emb = PCA(n_components=2).fit_transform(z) if z.shape[1] > 2 else z
        for lbl, col in colors.items():
            m = labels == lbl
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=col, label=cnames[lbl], s=8, alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, markerscale=2)
        ax.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Stage configuration
# =============================================================================

STAGE_WEIGHTS = {
    # (weight_kl, weight_null, weight_mi, weight_clf, weight_orth)
    "d0": (1e-3, 0.0,  0.0,  0.0,  0.0),
    "d1": (1e-3, 0.2,  0.0,  0.0,  0.0),
    "d2": (1e-3, 0.2,  0.05, 0.0,  0.0),
    "d3": (1e-3, 0.2,  0.05, 1.0,  0.0),
    "d4": (1e-3, 0.2,  0.05, 1.0,  0.1),
    "d5": (1e-3, 0.2,  0.05, 1.0,  0.1),   # d5: same losses; contrastive placeholder
}


# =============================================================================
# Dataset builder
# =============================================================================

def build_dataset(args):
    """
    Returns (train_ds, val_ds, has_bbox, in_channels, bbox_f1, bbox_f2).

    has_bbox=True → spatial gate supervision active in D1+.
    in_channels    → 1 for greyscale (A/B), 3 for RGB (C/D/E/F).
    bbox_f1/f2     → normalised bbox tuples for spatial gate mask.
    """
    kw = dict(n_normal=1500, n_factor1=750, n_factor2=750, n_both=0,
              img_size=64, global_seed=args.seed, mode='flat')
    kw_val = dict(n_normal=300, n_factor1=150, n_factor2=150, n_both=150,
                  img_size=64, global_seed=args.seed + 999, mode='flat')
    mnist_kw     = dict(mnist_root=args.mnist_root)
    mnist_kw_val = dict(mnist_root=args.mnist_root)

    if args.domain == "A":
        from datasets.SplatteredMNIST import SplatteredMNISTDataset, BBOX_F1, BBOX_F2
        return (SplatteredMNISTDataset(**kw, **mnist_kw),
                SplatteredMNISTDataset(**kw_val, **mnist_kw_val),
                True, 1, BBOX_F1, BBOX_F2)

    if args.domain == "B":
        from datasets.CorrelatedSplatteredMNIST import CorrelatedSplatteredMNISTDataset
        from datasets.SplatteredMNIST import BBOX_F1, BBOX_F2
        cr = args.correlation_rate
        return (CorrelatedSplatteredMNISTDataset(**kw, **mnist_kw, correlation_rate=cr),
                CorrelatedSplatteredMNISTDataset(**kw_val, **mnist_kw_val, correlation_rate=0.0),
                True, 1, BBOX_F1, BBOX_F2)

    if args.domain == "C":
        from datasets.DSprites import DSpritesDataset, BBOX_F1, BBOX_F2
        dsprites_kw = dict(dsprites_root=args.dsprites_root)
        return (DSpritesDataset(**kw, **dsprites_kw),
                DSpritesDataset(**kw_val, **dsprites_kw),
                False, 1, BBOX_F1, BBOX_F2)

    if args.domain == "D":
        from datasets.CorrelatedDSprites import CorrelatedDSpritesDataset
        from datasets.DSprites import BBOX_F1, BBOX_F2
        dsprites_kw = dict(dsprites_root=args.dsprites_root)
        cr = args.correlation_rate
        return (CorrelatedDSpritesDataset(**kw, **dsprites_kw, correlation_rate=cr),
                CorrelatedDSpritesDataset(**kw_val, **dsprites_kw, correlation_rate=0.0),
                False, 1, BBOX_F1, BBOX_F2)

    if args.domain == "E":
        from datasets.CelebAFactor import CelebAFactorDataset, BBOX_F1, BBOX_F2
        return (CelebAFactorDataset(celeba_root=args.celeba_root, split="train",
                                    balanced=True, global_seed=args.seed),
                CelebAFactorDataset(celeba_root=args.celeba_root, split="valid",
                                    balanced=True, global_seed=args.seed + 999),
                True, 3, BBOX_F1, BBOX_F2)

    if args.domain == "F":
        from datasets.CelebAFactor import CelebAFactorDataset, BBOX_F1, BBOX_F2
        return (CelebAFactorDataset(celeba_root=args.celeba_root, split="train",
                                    balanced=False, global_seed=args.seed),
                CelebAFactorDataset(celeba_root=args.celeba_root, split="valid",
                                    balanced=False, global_seed=args.seed + 999),
                True, 3, BBOX_F1, BBOX_F2)

    raise ValueError(f"Unknown domain {args.domain!r}. Choose from A, B, C, D, E, F.")


# =============================================================================
# CSV result logger
# =============================================================================

def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Gate evaluation
# =============================================================================

def evaluate_gates(probe_f1, probe_f2, cross_head, comp_score, domain):
    """
    Returns (G1, G2, G3) as booleans.
    G3 threshold: pixel-delta (A/B/E/F) or channel-mean-delta (C/D).
    """
    G1 = (probe_f1 > 0.75) and (probe_f2 > 0.75)
    G2 = cross_head < 0.60
    G3 = comp_score > 0.05   # same threshold across all domains
    return G1, G2, G3


# =============================================================================
# Main training loop
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Domain: {args.domain}  Stage: {args.stage}  Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds, val_ds, has_bbox, in_channels, bbox_f1, bbox_f2 = build_dataset(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Stage config ──────────────────────────────────────────────────────
    use_spatial = has_bbox and (args.stage not in ("d0",))
    w_kl, w_null, w_mi, w_clf, w_orth = STAGE_WEIGHTS[args.stage]
    # Allow CLI overrides
    w_kl   = args.weight_kl   if args.weight_kl   >= 0 else w_kl
    w_null = args.weight_null if args.weight_null  >= 0 else w_null
    w_mi   = args.weight_mi   if args.weight_mi    >= 0 else w_mi
    w_clf  = args.weight_clf  if args.weight_clf   >= 0 else w_clf
    w_orth = args.weight_orth if args.weight_orth  >= 0 else w_orth
    w_spatial = args.weight_spatial if args.weight_spatial >= 0 else (0.5 if use_spatial else 0.0)

    print(f"Weights: kl={w_kl} null={w_null} mi={w_mi} clf={w_clf} "
          f"orth={w_orth} spatial={w_spatial}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = GeneralizeSepVAE(z_ch_common=4, z_ch_disease=2,
                             use_spatial_gate=use_spatial,
                             in_channels=in_channels).to(device)

    clf_d1 = DiseaseClassifier(z_ch=2).to(device)
    clf_d2 = DiseaseClassifier(z_ch=2).to(device)
    mi_disc = FactorVAEDiscriminator(z1_dim=2, z2_dim=2, hidden=64).to(device)

    vae_params  = list(model.parameters())
    opt_vae     = torch.optim.Adam(vae_params, lr=args.lr)
    opt_clfs    = torch.optim.Adam(list(clf_d1.parameters()) +
                                   list(clf_d2.parameters()), lr=args.lr)
    opt_mi_disc = torch.optim.Adam(mi_disc.parameters(), lr=args.lr * 0.5)

    # ── Precompute bbox masks for spatial gate ─────────────────────────────
    lat_h = lat_w = 8  # encoder output at 64×64 input
    if use_spatial:
        mask_d1 = torch.tensor(
            _bbox_to_latent_mask(bbox_f1, lat_h, lat_w)
        )[None, None].to(device)   # (1, 1, 8, 8)
        mask_d2 = torch.tensor(
            _bbox_to_latent_mask(bbox_f2, lat_h, lat_w)
        )[None, None].to(device)
    else:
        mask_d1 = mask_d2 = None

    # ── Output directories ─────────────────────────────────────────────────
    run_name = f"{args.domain}_{args.stage}_{args.exp_name}"
    out_dir  = Path(args.output_root) / run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────
    _wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project,
                       name=run_name, config=vars(args))
            _wandb = True
        except Exception as e:
            print(f"W&B init failed: {e}")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train(); clf_d1.train(); clf_d2.train(); mi_disc.train()
        t0 = time.time()
        acc = {k: 0.0 for k in ["total", "rec", "kl", "null", "orth",
                                  "clf", "mi_vae", "mi_disc", "spatial"]}
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            has_d1 = ((y == LABEL_FACTOR1) | (y == LABEL_BOTH)).float()
            has_d2 = ((y == LABEL_FACTOR2) | (y == LABEL_BOTH)).float()

            # ── VAE step ──────────────────────────────────────────────────
            opt_vae.zero_grad()
            x_rec, lat = model(x)

            L_rec  = F.mse_loss(x_rec, x)
            L_kl   = (kl_loss(lat["mu_c"],  lat["lv_c"])
                    + kl_loss(lat["mu_d1"], lat["lv_d1"])
                    + kl_loss(lat["mu_d2"], lat["lv_d2"]))
            L_null = (nulling_loss(lat["mu_d1"], has_d1) +
                      nulling_loss(lat["mu_d2"], has_d2))
            L_orth = orthogonality_loss(lat["mu_d1"], lat["mu_d2"])

            # Classifier cooperative loss (encoder grad)
            L_clf_enc = (stable_bce(clf_d1(lat["mu_d1"]), has_d1) +
                         stable_bce(clf_d2(lat["mu_d2"]), has_d2))

            # MI encoder loss (discriminator frozen for encoder step)
            z_d1_gap = lat["mu_d1"].mean(dim=[2, 3])
            z_d2_gap = lat["mu_d2"].mean(dim=[2, 3])
            with torch.no_grad():
                pass  # disc params contribute gradient in next block
            L_mi_vae = mi_loss_vae(mi_disc, z_d1_gap.detach(), z_d2_gap.detach()) \
                       if w_mi > 0 else x.new_zeros(1).squeeze()
            # Re-enable grad through z for actual encoder update
            if w_mi > 0:
                L_mi_vae = mi_loss_vae(mi_disc, z_d1_gap, z_d2_gap)

            # Spatial gate loss
            L_spatial = x.new_zeros(1).squeeze()
            if use_spatial and lat["attn_d1"] is not None:
                L_spatial = spatial_gate_loss(
                    lat["attn_d1"], lat["attn_d2"], mask_d1, mask_d2)

            L_total = (L_rec
                       + w_kl      * L_kl
                       + w_null    * L_null
                       + w_orth    * L_orth
                       + w_clf     * L_clf_enc
                       + w_mi      * L_mi_vae
                       + w_spatial * L_spatial)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, 1.0)
            opt_vae.step()

            # ── Classifier step ────────────────────────────────────────────
            if w_clf > 0:
                opt_clfs.zero_grad()
                L_clf = (stable_bce(clf_d1(lat["mu_d1"].detach()), has_d1) +
                         stable_bce(clf_d2(lat["mu_d2"].detach()), has_d2))
                L_clf.backward()
                opt_clfs.step()

            # ── MI discriminator step ──────────────────────────────────────
            L_mi_d = x.new_zeros(1).squeeze()
            if w_mi > 0:
                opt_mi_disc.zero_grad()
                L_mi_d = mi_loss_disc(mi_disc,
                                      lat["mu_d1"].mean(dim=[2, 3]).detach(),
                                      lat["mu_d2"].mean(dim=[2, 3]).detach())
                L_mi_d.backward()
                opt_mi_disc.step()

            # Accumulate
            acc["total"]   += L_total.item()
            acc["rec"]     += L_rec.item()
            acc["kl"]      += L_kl.item()
            acc["null"]    += L_null.item()
            acc["orth"]    += L_orth.item()
            acc["clf"]     += L_clf_enc.item()
            acc["mi_vae"]  += L_mi_vae.item()
            acc["mi_disc"] += L_mi_d.item()
            acc["spatial"] += L_spatial.item()
            n_batches += 1

        avgs = {k: v / n_batches for k, v in acc.items()}
        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"total={avgs['total']:.4f}  rec={avgs['rec']:.4f}  "
              f"kl={avgs['kl']:.3f}  null={avgs['null']:.4f}  "
              f"mi_disc={avgs['mi_disc']:.4f}  ({elapsed:.1f}s)")

        # ── Evaluation ──────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()

            # Val reconstruction
            with torch.no_grad():
                val_rec, val_n = 0.0, 0
                for xv, _ in val_loader:
                    xv = xv.to(device)
                    xrv, _ = model(xv)
                    val_rec += F.mse_loss(xrv, xv).item() * xv.shape[0]
                    val_n   += xv.shape[0]
                val_rec /= val_n

            # Disentanglement metrics
            dis = evaluate_disentanglement(model, val_loader, device)
            auc_f1      = dis.get("auc_d1_d1", float("nan"))
            auc_f2      = dis.get("auc_d2_d2", float("nan"))
            cross_score = (dis.get("auc_d1_d2", 0.5) + dis.get("auc_d2_d1", 0.5)) / 2

            # Norm ratio
            norm_r1, norm_r2 = compute_norm_ratio(model, val_loader, device)

            # MI discriminator accuracy
            z_d1_all, z_d2_all = [], []
            for xv, _ in val_loader:
                xv = xv.to(device)
                _, lat = model(xv)
                z_d1_all.append(lat["mu_d1"].mean(dim=[2, 3]))
                z_d2_all.append(lat["mu_d2"].mean(dim=[2, 3]))
            z_d1_cat = torch.cat(z_d1_all)
            z_d2_cat = torch.cat(z_d2_all)
            mi_acc = disc_accuracy(mi_disc, z_d1_cat, z_d2_cat)

            # Composition test
            if args.domain in ("A", "B"):
                comp = evaluate_composition_stamps(model, val_ds, device)
            elif args.domain in ("C", "D"):
                comp = evaluate_composition_dsprites(model, val_ds, device)
            else:  # E, F
                comp = evaluate_composition_spatial_rgb(model, val_ds, device)
            comp_score = comp["composition_score"]

            # Gates
            G1, G2, G3 = evaluate_gates(auc_f1, auc_f2, cross_score,
                                         comp_score, args.domain)

            print(f"  [Val]  rec={val_rec:.4f}  "
                  f"AUC f1={auc_f1:.3f}  f2={auc_f2:.3f}  "
                  f"cross={cross_score:.3f}  norm_r1={norm_r1:.1f}  "
                  f"mi_acc={mi_acc:.3f}  comp={comp_score:+.4f}")
            print(f"  [Gate] G1={'✓' if G1 else '✗'}  "
                  f"G2={'✓' if G2 else '✗'}  "
                  f"G3={'✓' if G3 else '✗'}")

            # Save visualisations
            save_recon_grid(model, val_ds, device,
                            out_dir / "diagnostics" / f"recon_{epoch:04d}.png")
            save_composition_grid(model, val_ds, device,
                                  out_dir / "diagnostics" / f"compose_{epoch:04d}.png")
            if epoch == args.epochs:
                save_pca_scatter(model, val_ds, device,
                                 out_dir / "diagnostics" / "pca_final.png")

            # Append to comparison CSV (final epoch only)
            if epoch == args.epochs:
                csv_row = {
                    "domain":          args.domain,
                    "stage":           args.stage,
                    "exp_name":        args.exp_name,
                    "epochs":          args.epochs,
                    "seed":            args.seed,
                    "correlation":     getattr(args, "correlation_rate", 0.0),
                    "probe_auc_f1":    round(auc_f1,      4),
                    "probe_auc_f2":    round(auc_f2,      4),
                    "cross_head_score":round(cross_score, 4),
                    "norm_ratio_f1":   round(norm_r1,     3),
                    "norm_ratio_f2":   round(norm_r2,     3),
                    "composition_score": round(comp_score, 5),
                    "mi_disc_acc":     round(mi_acc,      4),
                    "val_rec":         round(val_rec,     5),
                    "G1": int(G1), "G2": int(G2), "G3": int(G3),
                }
                csv_path = Path(args.results_csv)
                _append_csv(csv_path, csv_row)
                print(f"  [CSV]  row appended → {csv_path}")

            if _wandb:
                import wandb
                wandb.log({
                    **{f"train/{k}": v for k, v in avgs.items()},
                    "val/rec": val_rec,
                    **{f"val/{k}": v for k, v in dis.items()},
                    "val/cross_head_score": cross_score,
                    "val/norm_ratio_f1": norm_r1,
                    "val/norm_ratio_f2": norm_r2,
                    "val/mi_disc_acc": mi_acc,
                    **{f"composition/{k}": v for k, v in comp.items()},
                    "gates/G1": int(G1), "gates/G2": int(G2), "gates/G3": int(G3),
                    "epoch": epoch,
                })

        # ── Checkpoint ──────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "clf_d1": clf_d1.state_dict(), "clf_d2": clf_d2.state_dict(),
                "mi_disc": mi_disc.state_dict(),
                "opt_vae": opt_vae.state_dict(), "args": vars(args),
            }, out_dir / "checkpoints" / f"ckpt_epoch{epoch:04d}.pt")

    if _wandb:
        import wandb; wandb.finish()

    print(f"\nDone. Outputs in {out_dir}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="d0-d5 Generalisation Suite: train SepVAE on toy domains")

    # Domain & stage
    p.add_argument("--domain", type=str, default="A",
                   choices=["A", "B", "C", "D", "E", "F"],
                   help="A=SplatteredMNIST, B=CorrelatedSplatteredMNIST, "
                        "C=dSprites(scale×orient), D=CorrelatedDSprites, "
                        "E=CelebA-balanced, F=CelebA-natural")
    p.add_argument("--stage", type=str, default="d5",
                   choices=["d0", "d1", "d2", "d3", "d4", "d5"],
                   help="d0=base, d1=+null+spatial, d2=+MI, d3=+clf, "
                        "d4=+orth, d5=+all")
    p.add_argument("--correlation_rate", type=float, default=0.30,
                   help="Spurious co-occurrence rate for domains B and D")

    # Data roots
    p.add_argument("--mnist_root",    type=str, default="~/.cache/mnist",
                   help="MNIST download/cache directory (domains A-B)")
    p.add_argument("--dsprites_root", type=str, default="~/.cache/dsprites",
                   help="dSprites download/cache directory (domains C-D)")
    p.add_argument("--celeba_root",   type=str, default="~/.cache/celeba",
                   help="CelebA download/cache directory (domains E-F)")

    # Training
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int,   default=42)

    # Loss weight overrides (-1 = use stage default)
    p.add_argument("--weight_kl",      type=float, default=-1)
    p.add_argument("--weight_null",    type=float, default=-1)
    p.add_argument("--weight_mi",      type=float, default=-1)
    p.add_argument("--weight_clf",     type=float, default=-1)
    p.add_argument("--weight_orth",    type=float, default=-1)
    p.add_argument("--weight_spatial", type=float, default=-1)

    # Logging
    p.add_argument("--eval_every",   type=int, default=10)
    p.add_argument("--save_every",   type=int, default=50)
    p.add_argument("--exp_name",     type=str, default="run")
    p.add_argument("--output_root",  type=str, default="runs_generalize")
    p.add_argument("--results_csv",  type=str,
                   default="results/domain_comparison_table.csv")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project", type=str, default="sepvae-generalize")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
