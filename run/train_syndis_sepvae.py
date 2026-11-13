"""
train_syndis_sepvae.py — SepVAE proof-of-concept on the SynDis synthetic dataset
==================================================================================
Stripped-down PyTorch re-implementation of the SepVAE architecture and loss
portfolio, designed to run in minutes on CPU (or seconds on a GPU) to validate
the disentanglement hypothesis before committing to full CXR runs.

Architecture
------------
    Encoder:  3-layer CNN → shared feature map (B, 64, 8, 8)
              ↳ common head   → mu_c,  logvar_c   (B, 4, 8, 8)
              ↳ disease-1 head → mu_d1, logvar_d1  (B, 2, 8, 8)
              ↳ disease-2 head → mu_d2, logvar_d2  (B, 2, 8, 8)
    Decoder:  concat(z_c, z_d1, z_d2) → 3-layer transposed CNN → (B, 1, 64, 64)

Losses
------
    L_rec     : MSE reconstruction
    L_kl      : KL divergence (all three heads)
    L_null    : Nulling — drive z_disease toward 0 when disease is absent
    L_orth    : Orthogonality — cosine similarity between z_d1 and z_d2 GAP
    L_clf     : Disease discriminability (R11) — cooperative BCE classifier

Key metrics (computed every --eval_every epochs)
-------------------------------------------------
    AUC d1→d1  : linear probe on z_d1_GAP predicts disease-1 label
    AUC d2→d2  : linear probe on z_d2_GAP predicts disease-2 label
    AUC d1→d2  : cross-probe (should be ~0.5 if disentangled)
    AUC d2→d1  : cross-probe (should be ~0.5 if disentangled)
    specificity_ratio : AUC(same) / AUC(cross) — target > 2.0

Composition test (requires n_both > 0 in val set)
--------------------------------------------------
    Compose z_common (from normal image) + z_d1 (from disease-1 image)
                                         + z_d2 (from disease-2 image)
    Decode → measure pixel signal in blob and stripe regions.
    Score > 0 means composed output contains both disease signals.

Usage
-----
    # Quick smoke test (~1 min on CPU)
    python run/train_syndis_sepvae.py --epochs 30

    # Full run with all losses
    python run/train_syndis_sepvae.py --epochs 100 --weight_null 0.2 --weight_orth 0.1 \\
        --weight_clf 1.0 --wandb

    # Ablation: no nulling, no orthogonality (vanilla multi-head VAE)
    python run/train_syndis_sepvae.py --epochs 100 --weight_null 0 --weight_orth 0 \\
        --weight_clf 0 --exp_name syndis_ablation
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when running as `python run/train_syndis_sepvae.py`
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

from datasets.SynDis import (
    SynDisDataset,
    BLOB_ROWS, BLOB_COLS, STRIPE_ROWS, STRIPE_COLS,
    LABEL_NORMAL, LABEL_DISEASE1, LABEL_DISEASE2, LABEL_BOTH,
)


# =============================================================================
# Model
# =============================================================================

class _ConvHead(nn.Module):
    """1×1 conv → mu + logvar, shared spatial resolution."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mu_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.lv_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, h):
        return self.mu_conv(h), self.lv_conv(h)


class SynDisEncoder(nn.Module):
    def __init__(self, z_ch_common: int = 4, z_ch_disease: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.ReLU(),  # 64→32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32→16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16→ 8
        )
        self.common_head   = _ConvHead(64, z_ch_common)
        self.disease1_head = _ConvHead(64, z_ch_disease)
        self.disease2_head = _ConvHead(64, z_ch_disease)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "common":   self.common_head(h),    # (mu, logvar) each (B, 4, 8, 8)
            "disease1": self.disease1_head(h),  # (mu, logvar) each (B, 2, 8, 8)
            "disease2": self.disease2_head(h),  # (mu, logvar) each (B, 2, 8, 8)
        }


class SynDisDecoder(nn.Module):
    def __init__(self, z_ch_total: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch_total, 64, 4, stride=2, padding=1), nn.ReLU(),  # 8→16
            nn.ConvTranspose2d(64,         32, 4, stride=2, padding=1), nn.ReLU(),  # 16→32
            nn.ConvTranspose2d(32,         16, 4, stride=2, padding=1), nn.ReLU(),  # 32→64
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),  # output in [-1, 1], matches input convention
        )

    def forward(self, z_common, z_d1, z_d2):
        z = torch.cat([z_common, z_d1, z_d2], dim=1)
        return self.net(z)


class SynDisSepVAE(nn.Module):
    def __init__(self, z_ch_common: int = 4, z_ch_disease: int = 2):
        super().__init__()
        self.encoder = SynDisEncoder(z_ch_common, z_ch_disease)
        self.decoder = SynDisDecoder(z_ch_common + 2 * z_ch_disease)
        self.z_ch_d  = z_ch_disease

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        enc = self.encoder(x)
        mu_c,  lv_c  = enc["common"]
        mu_d1, lv_d1 = enc["disease1"]
        mu_d2, lv_d2 = enc["disease2"]

        z_c  = self.reparameterise(mu_c,  lv_c)
        z_d1 = self.reparameterise(mu_d1, lv_d1)
        z_d2 = self.reparameterise(mu_d2, lv_d2)

        x_rec = self.decoder(z_c, z_d1, z_d2)

        return x_rec, {
            "mu_c": mu_c, "lv_c": lv_c,
            "mu_d1": mu_d1, "lv_d1": lv_d1,
            "mu_d2": mu_d2, "lv_d2": lv_d2,
            "z_c": z_c, "z_d1": z_d1, "z_d2": z_d2,
        }


# =============================================================================
# Disease classifier (R11)
# =============================================================================

class DiseaseClassifier(nn.Module):
    """Tiny cooperative MLP: GAP(z_disease) → binary logit."""
    def __init__(self, z_ch: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_ch, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_mu):
        z_gap = z_mu.mean(dim=[2, 3])   # (B, C)
        return self.net(z_gap).squeeze(-1)   # (B,)


# =============================================================================
# Loss functions
# =============================================================================

def kl_loss(mu, logvar):
    """Per-image KL averaged over batch."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2, 3]).mean()


def nulling_loss(mu_disease, disease_present: torch.Tensor, sigma: float = 0.1):
    """
    When disease is absent, push z_disease mu toward 0.
    disease_present: (B,) float 0/1.
    """
    absent = (1.0 - disease_present).bool()
    if absent.sum() == 0:
        return mu_disease.new_zeros(1).squeeze()
    return mu_disease[absent].pow(2).mean() / (sigma ** 2)


def orthogonality_loss(mu_d1, mu_d2):
    """Penalise cosine similarity between z_d1 and z_d2 (GAP'd)."""
    g1 = mu_d1.mean(dim=[2, 3])   # (B, C)
    g2 = mu_d2.mean(dim=[2, 3])
    cos = F.cosine_similarity(g1, g2, dim=1)   # (B,)
    return cos.abs().mean()


def stable_bce(logit, target):
    """Numerically stable binary cross-entropy (matches JAX implementation)."""
    return (logit.clamp(min=0) - logit * target + (1 + (-logit.abs()).exp()).log()).mean()


def clf_loss(clf: DiseaseClassifier, mu_disease, has_disease: torch.Tensor):
    """Cooperative disease discriminability loss (R11)."""
    logit = clf(mu_disease)
    return stable_bce(logit, has_disease)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def encode_dataset(model: SynDisSepVAE, loader: DataLoader, device: torch.device):
    """Return numpy arrays of GAP'd z_d1, z_d2, labels for an entire split."""
    all_z_d1, all_z_d2, all_z_c, all_labels = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        _, lat = model(x)
        all_z_d1.append(lat["mu_d1"].mean(dim=[2, 3]).cpu().numpy())
        all_z_d2.append(lat["mu_d2"].mean(dim=[2, 3]).cpu().numpy())
        all_z_c.append(lat["mu_c"].mean(dim=[2, 3]).cpu().numpy())
        all_labels.append(y.numpy())
    return (
        np.concatenate(all_z_d1),
        np.concatenate(all_z_d2),
        np.concatenate(all_z_c),
        np.concatenate(all_labels),
    )


def auc_probe(z_feats: np.ndarray, binary_labels: np.ndarray) -> float:
    """Fit logistic regression and return AUC-ROC."""
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return float("nan")
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(z_feats, binary_labels)
    prob = clf.predict_proba(z_feats)[:, 1]
    return roc_auc_score(binary_labels, prob)


def evaluate_disentanglement(model, val_loader, device):
    """
    Compute AUC of linear probes on z_d1 and z_d2.

    Returns dict with:
        auc_d1_d1, auc_d2_d2  — on-diagonal (should be high)
        auc_d1_d2, auc_d2_d1  — off-diagonal (should be ~0.5)
        specificity_ratio      — geometric mean of (auc_dd / auc_cross)
    """
    z_d1, z_d2, _, labels = encode_dataset(model, val_loader, device)

    has_d1 = ((labels == LABEL_DISEASE1) | (labels == LABEL_BOTH)).astype(int)
    has_d2 = ((labels == LABEL_DISEASE2) | (labels == LABEL_BOTH)).astype(int)

    # Filter to images where the probe makes sense:
    #   d1 probe: images with label 0 or 1 only (no comorbid for clean binary eval)
    mask_for_d1 = (labels == LABEL_NORMAL) | (labels == LABEL_DISEASE1)
    mask_for_d2 = (labels == LABEL_NORMAL) | (labels == LABEL_DISEASE2)

    auc_d1_d1 = auc_probe(z_d1[mask_for_d1], has_d1[mask_for_d1])
    auc_d2_d2 = auc_probe(z_d2[mask_for_d2], has_d2[mask_for_d2])
    auc_d1_d2 = auc_probe(z_d1[mask_for_d2], has_d2[mask_for_d2])   # z_d1 predicting d2
    auc_d2_d1 = auc_probe(z_d2[mask_for_d1], has_d1[mask_for_d1])   # z_d2 predicting d1

    # Specificity ratio: how much better is the on-diagonal probe vs. cross?
    def spec_ratio(on_diag, cross):
        if math.isnan(on_diag) or math.isnan(cross) or cross <= 0.5:
            return float("nan")
        return on_diag / cross

    return {
        "auc_d1_d1": auc_d1_d1,
        "auc_d2_d2": auc_d2_d2,
        "auc_d1_d2": auc_d1_d2,
        "auc_d2_d1": auc_d2_d1,
        "spec_d1":   spec_ratio(auc_d1_d1, auc_d1_d2),
        "spec_d2":   spec_ratio(auc_d2_d2, auc_d2_d1),
    }


@torch.no_grad()
def evaluate_composition(model: SynDisSepVAE, val_ds: SynDisDataset,
                          device: torch.device, n: int = 50) -> dict:
    """
    Composition test: does decode(z_c_normal, z_d1_disease1, z_d2_disease2)
    produce an image that contains both disease signals?

    Score: mean pixel value increase over normal in blob region AND stripe region.
    Returns composition_score (higher = better composition).
    """
    model.eval()

    def gather_latent(label, key, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        xs = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        _, lat = model(xs)
        return lat[key]   # (N, C, H, W)

    mu_c  = gather_latent(LABEL_NORMAL,   "mu_c",  n)
    mu_d1 = gather_latent(LABEL_DISEASE1, "mu_d1", n)
    mu_d2 = gather_latent(LABEL_DISEASE2, "mu_d2", n)

    n_actual = min(len(mu_c), len(mu_d1), len(mu_d2))
    mu_c  = mu_c[:n_actual]
    mu_d1 = mu_d1[:n_actual]
    mu_d2 = mu_d2[:n_actual]

    # Compose: pair each normal with a disease-1 and disease-2 source
    x_composed = model.decoder(mu_c, mu_d1, mu_d2).cpu().numpy()   # (N, 1, 64, 64)
    x_composed = (x_composed + 1) / 2                               # → [0, 1]

    # Baseline: normal images encoded and decoded with null disease latents
    z_null = torch.zeros_like(mu_d1)
    x_normal_dec = model.decoder(mu_c, z_null, z_null).cpu().numpy()
    x_normal_dec = (x_normal_dec + 1) / 2

    # Region means
    br, bc = BLOB_ROWS,   BLOB_COLS
    sr, sc = STRIPE_ROWS, STRIPE_COLS

    blob_comp   = x_composed  [:, 0, br[0]:br[1], bc[0]:bc[1]].mean()
    stripe_comp = x_composed  [:, 0, sr[0]:sr[1], sc[0]:sc[1]].mean()
    blob_norm   = x_normal_dec[:, 0, br[0]:br[1], bc[0]:bc[1]].mean()
    stripe_norm = x_normal_dec[:, 0, sr[0]:sr[1], sc[0]:sc[1]].mean()

    blob_delta   = float(blob_comp   - blob_norm)
    stripe_delta = float(stripe_comp - stripe_norm)
    # Positive score = composed image has MORE signal in both disease regions
    composition_score = blob_delta + stripe_delta

    return {
        "blob_delta":        blob_delta,
        "stripe_delta":      stripe_delta,
        "composition_score": composition_score,
    }


# =============================================================================
# Visualisation helpers
# =============================================================================

@torch.no_grad()
def save_recon_grid(model: SynDisSepVAE, val_ds: SynDisDataset,
                    device: torch.device, path: Path, n_per_class: int = 4):
    """Save originals / reconstructions grid (3 classes × n_per_class)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    class_labels = [LABEL_NORMAL, LABEL_DISEASE1, LABEL_DISEASE2]
    class_names  = ["Normal", "Disease-1 (blob)", "Disease-2 (stripe)"]

    fig, axes = plt.subplots(len(class_labels) * 2, n_per_class,
                             figsize=(n_per_class * 1.8, len(class_labels) * 4))
    for row, (cls, name) in enumerate(zip(class_labels, class_names)):
        idxs = np.where(val_ds.labels == cls)[0][:n_per_class]
        xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)
        x_rec, _ = model(xs)
        for col in range(len(idxs)):
            orig = (xs[col, 0].cpu().numpy()    + 1) / 2
            rec  = (x_rec[col, 0].cpu().numpy() + 1) / 2
            axes[row * 2,     col].imshow(orig, cmap="gray", vmin=0, vmax=1)
            axes[row * 2 + 1, col].imshow(rec,  cmap="gray", vmin=0, vmax=1)
            axes[row * 2,     col].axis("off")
            axes[row * 2 + 1, col].axis("off")
        axes[row * 2, 0].set_ylabel(f"{name}\n(orig)", fontsize=7)
        axes[row * 2 + 1, 0].set_ylabel("recon", fontsize=7)

    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_composition_grid(model: SynDisSepVAE, val_ds: SynDisDataset,
                           device: torch.device, path: Path, n: int = 4):
    """
    Save composition grid:
    Columns: [normal source] [disease-1 source] [disease-2 source]
             [composed output] [gt_both (if available)]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    has_both = (val_ds.labels == LABEL_BOTH).sum() > 0
    n_cols = 5 if has_both else 4

    def get_xs(label, count):
        idxs = np.where(val_ds.labels == label)[0][:count]
        return torch.stack([val_ds[i][0] for i in idxs]).to(device)

    xs_normal = get_xs(LABEL_NORMAL,   n)
    xs_d1     = get_xs(LABEL_DISEASE1, n)
    xs_d2     = get_xs(LABEL_DISEASE2, n)
    xs_both   = get_xs(LABEL_BOTH,     n) if has_both else None

    _, lat_n  = model(xs_normal)
    _, lat_d1 = model(xs_d1)
    _, lat_d2 = model(xs_d2)

    x_composed = model.decoder(lat_n["mu_c"], lat_d1["mu_d1"], lat_d2["mu_d2"])

    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 1.8, n * 1.8))
    col_titles = ["Normal (z_c)", "Disease-1\n(z_d1 src)", "Disease-2\n(z_d2 src)",
                  "Composed", "GT Both"]

    for i in range(n):
        imgs = [xs_normal[i, 0], xs_d1[i, 0], xs_d2[i, 0], x_composed[i, 0]]
        if has_both and xs_both is not None:
            imgs.append(xs_both[i, 0])
        for j, img_t in enumerate(imgs):
            img = (img_t.cpu().numpy() + 1) / 2
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=7)

    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = SynDisDataset(n_normal=1500, n_disease1=750, n_disease2=750,
                             n_both=0,   global_seed=42)
    val_ds   = SynDisDataset(n_normal=300,  n_disease1=150, n_disease2=150,
                             n_both=150, global_seed=99)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model + classifiers
    # ------------------------------------------------------------------
    model = SynDisSepVAE(z_ch_common=4, z_ch_disease=2).to(device)

    clf_d1 = DiseaseClassifier(z_ch=2).to(device)
    clf_d2 = DiseaseClassifier(z_ch=2).to(device)

    vae_params = list(model.parameters())
    optimizer  = torch.optim.Adam(vae_params, lr=args.lr)
    optimizer_clfs = torch.optim.Adam(
        list(clf_d1.parameters()) + list(clf_d2.parameters()), lr=args.lr
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_root) / args.exp_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(exist_ok=True)

    # Optional W&B
    _wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.exp_name,
                       config=vars(args))
            _wandb = True
        except Exception as e:
            print(f"W&B init failed: {e}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        clf_d1.train()
        clf_d2.train()

        t0 = time.time()
        loss_accum = {k: 0.0 for k in
                      ["total", "rec", "kl", "null", "orth", "clf"]}
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            has_d1 = ((y == LABEL_DISEASE1) | (y == LABEL_BOTH)).float()
            has_d2 = ((y == LABEL_DISEASE2) | (y == LABEL_BOTH)).float()

            # ----------------------------------------------------------
            # VAE step (classifiers frozen → gradient flows to encoder)
            # ----------------------------------------------------------
            optimizer.zero_grad()

            x_rec, lat = model(x)

            L_rec  = F.mse_loss(x_rec, x)
            L_kl   = (kl_loss(lat["mu_c"],  lat["lv_c"])
                    + kl_loss(lat["mu_d1"], lat["lv_d1"])
                    + kl_loss(lat["mu_d2"], lat["lv_d2"]))
            L_null = (nulling_loss(lat["mu_d1"], has_d1) +
                      nulling_loss(lat["mu_d2"], has_d2))
            L_orth = orthogonality_loss(lat["mu_d1"], lat["mu_d2"])

            # Classifier loss with frozen classifier params
            with torch.no_grad():
                clf_d1_params_frozen = {k: v.detach() for k, v in clf_d1.named_parameters()}
                clf_d2_params_frozen = {k: v.detach() for k, v in clf_d2.named_parameters()}

            L_clf_d1 = stable_bce(clf_d1(lat["mu_d1"].detach()), has_d1)
            L_clf_d2 = stable_bce(clf_d2(lat["mu_d2"].detach()), has_d2)

            # Classifier contribution to ENCODER loss (cooperative: same BCE,
            # but computed through mu_d so gradient reaches encoder)
            L_clf_enc = (stable_bce(clf_d1(lat["mu_d1"]), has_d1) +
                         stable_bce(clf_d2(lat["mu_d2"]), has_d2))

            L_total = (L_rec
                       + args.weight_kl   * L_kl
                       + args.weight_null * L_null
                       + args.weight_orth * L_orth
                       + args.weight_clf  * L_clf_enc)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, 1.0)
            optimizer.step()

            # ----------------------------------------------------------
            # Classifier step (encoder frozen)
            # ----------------------------------------------------------
            if args.weight_clf > 0:
                optimizer_clfs.zero_grad()
                mu_d1_det = lat["mu_d1"].detach()
                mu_d2_det = lat["mu_d2"].detach()
                L_clf = (stable_bce(clf_d1(mu_d1_det), has_d1) +
                         stable_bce(clf_d2(mu_d2_det), has_d2))
                L_clf.backward()
                optimizer_clfs.step()

            # Accumulate
            loss_accum["total"] += L_total.item()
            loss_accum["rec"]   += L_rec.item()
            loss_accum["kl"]    += L_kl.item()
            loss_accum["null"]  += L_null.item()
            loss_accum["orth"]  += L_orth.item()
            loss_accum["clf"]   += (L_clf_d1.item() + L_clf_d2.item()) / 2
            n_batches += 1

        # Average losses
        avgs = {k: v / n_batches for k, v in loss_accum.items()}
        elapsed = time.time() - t0

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"total={avgs['total']:.4f}  rec={avgs['rec']:.4f}  "
              f"kl={avgs['kl']:.3f}  null={avgs['null']:.4f}  "
              f"orth={avgs['orth']:.4f}  clf={avgs['clf']:.4f}  "
              f"({elapsed:.1f}s)")

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_rec_total, val_n = 0.0, 0
                for xv, _ in val_loader:
                    xv   = xv.to(device)
                    xrv, _ = model(xv)
                    val_rec_total += F.mse_loss(xrv, xv).item() * xv.shape[0]
                    val_n         += xv.shape[0]
                val_rec = val_rec_total / val_n

            dis = evaluate_disentanglement(model, val_loader, device)

            print(f"  [Val]  rec={val_rec:.4f}  "
                  f"AUC d1→d1={dis['auc_d1_d1']:.3f}  d2→d2={dis['auc_d2_d2']:.3f}  "
                  f"d1→d2={dis['auc_d1_d2']:.3f}  d2→d1={dis['auc_d2_d1']:.3f}  "
                  f"spec_d1={dis['spec_d1']:.2f}  spec_d2={dis['spec_d2']:.2f}")

            comp = evaluate_composition(model, val_ds, device)
            print(f"  [Comp] blob_Δ={comp['blob_delta']:+.4f}  "
                  f"stripe_Δ={comp['stripe_delta']:+.4f}  "
                  f"score={comp['composition_score']:+.4f}")

            # Save grids
            save_recon_grid(model, val_ds, device,
                            out_dir / "diagnostics" / f"recon_epoch{epoch:04d}.png")
            save_composition_grid(model, val_ds, device,
                                  out_dir / "diagnostics" / f"compose_epoch{epoch:04d}.png")

            if _wandb:
                import wandb
                log = {**{f"train/{k}": v for k, v in avgs.items()},
                       "val/rec": val_rec,
                       **{f"val/{k}": v for k, v in dis.items()},
                       **{f"composition/{k}": v for k, v in comp.items()},
                       "epoch": epoch}
                wandb.log(log)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "clf_d1": clf_d1.state_dict(),
                "clf_d2": clf_d2.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "checkpoints" / f"checkpoint_epoch{epoch:04d}.pt")

    if _wandb:
        import wandb
        wandb.finish()

    print(f"\nDone. Outputs in {out_dir}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SepVAE on SynDis synthetic dataset")

    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)

    # Loss weights
    p.add_argument("--weight_kl",   type=float, default=1e-3,
                   help="KL divergence weight")
    p.add_argument("--weight_null", type=float, default=0.2,
                   help="Nulling loss weight (R2)")
    p.add_argument("--weight_orth", type=float, default=0.1,
                   help="Orthogonality loss weight (R3)")
    p.add_argument("--weight_clf",  type=float, default=1.0,
                   help="Disease discriminability loss weight (R11)")

    # Logging
    p.add_argument("--eval_every",  type=int,   default=10)
    p.add_argument("--save_every",  type=int,   default=25)
    p.add_argument("--exp_name",    type=str,   default="syndis_sepvae")
    p.add_argument("--output_root", type=str,   default="runs_syndis")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--wandb_project", type=str, default="sepvae-syndis")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
