"""
trainer_base.py — Shared utilities: eval loop, metric computation, grid saving.

All experiment trainers import from here so the metric protocol is identical.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def auc_probe(z: np.ndarray, y: np.ndarray) -> float:
    """Fit a LogisticRegression probe and return AUC."""
    if y.sum() == 0 or (1 - y).sum() == 0:
        return 0.5
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(z, y)
    prob = clf.predict_proba(z)[:, 1]
    return float(roc_auc_score(y, prob))


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    region_blob:   tuple = ((20, 50), (10, 38)),
    region_stripe: tuple = ((48, 64), (0,  64)),
) -> Dict[str, float]:
    """
    Runs one pass over loader and returns the unified metric dict.

    Expected model signature: model(x) → (x_hat, enc_dict)
    enc_dict must contain: mu_c, mu_d1, mu_d2 (plus lv_* variants).
    """
    model.eval()
    all_mu_c, all_mu_d1, all_mu_d2 = [], [], []
    all_labels, all_x, all_xhat = [], [], []
    total_mse = total_kl = n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            x_hat, enc = model(x)

            # Reconstruction MSE
            total_mse += nn.functional.mse_loss(x_hat, x, reduction='sum').item()

            # KL (uses mu + logvar from enc dict)
            for key in ('c', 'd1', 'd2'):
                mu  = enc[f'mu_{key}']
                lv  = enc[f'lv_{key}']
                kl  = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
                total_kl += kl.sum().item()

            # Collect GAP'd latents
            def gap(t):
                return t.mean(dim=(-2, -1)).cpu().numpy() if t.dim() > 2 else t.cpu().numpy()

            all_mu_c.append(gap(enc['mu_c']))
            all_mu_d1.append(gap(enc['mu_d1']))
            all_mu_d2.append(gap(enc['mu_d2']))
            all_labels.append(y.cpu().numpy())
            all_x.append(x.cpu())
            all_xhat.append(x_hat.cpu())
            n_samples += x.shape[0]

    mu_c  = np.concatenate(all_mu_c)
    mu_d1 = np.concatenate(all_mu_d1)
    mu_d2 = np.concatenate(all_mu_d2)
    labels = np.concatenate(all_labels)

    has_d1 = ((labels == 1) | (labels == 3)).astype(float)
    has_d2 = ((labels == 2) | (labels == 3)).astype(float)

    auc_d1_d1 = auc_probe(mu_d1, has_d1)
    auc_d2_d2 = auc_probe(mu_d2, has_d2)
    auc_d1_d2 = auc_probe(mu_d1, has_d2)   # leakage: d1 predicts d2
    auc_d2_d1 = auc_probe(mu_d2, has_d1)   # leakage: d2 predicts d1
    auc_c_d1  = auc_probe(mu_c,  has_d1)
    auc_c_d2  = auc_probe(mu_c,  has_d2)

    spec_d1 = auc_d1_d1 / max(auc_d2_d1, 1e-6)
    spec_d2 = auc_d2_d2 / max(auc_d1_d2, 1e-6)

    n_dims_per_sample = sum(
        enc[f'mu_{k}'].numel() / enc[f'mu_{k}'].shape[0]
        for k in ('c', 'd1', 'd2')
    )

    metrics = {
        'rec_mse':      total_mse / n_samples,
        'kl_per_dim':   total_kl / (n_samples * n_dims_per_sample),
        'auc_d1_d1':    auc_d1_d1,
        'auc_d2_d2':    auc_d2_d2,
        'auc_d1_d2':    auc_d1_d2,
        'auc_d2_d1':    auc_d2_d1,
        'auc_c_d1':     auc_c_d1,
        'auc_c_d2':     auc_c_d2,
        'spec_ratio_d1': spec_d1,
        'spec_ratio_d2': spec_d2,
    }

    # Composition metrics — need comorbid samples
    comp_metrics = _composition_metrics(
        model, labels, all_x, all_xhat, device, region_blob, region_stripe
    )
    metrics.update(comp_metrics)

    model.train()
    return metrics


def _composition_metrics(model, labels, all_x, all_xhat, device, region_blob, region_stripe):
    """
    decode(z_c_normal, z_d1_disease1, z_d2_disease2) then measure signal in
    expected spatial regions.
    """
    x_all = torch.cat(all_x)
    x_hat_all = torch.cat(all_xhat)

    idx_normal = np.where(labels == 0)[0]
    idx_d1     = np.where(labels == 1)[0]
    idx_d2     = np.where(labels == 2)[0]

    if len(idx_normal) == 0 or len(idx_d1) == 0 or len(idx_d2) == 0:
        return {'comp_blob_delta': 0.0, 'comp_stripe_delta': 0.0,
                'comp_score': 0.0}

    n = min(len(idx_normal), len(idx_d1), len(idx_d2), 64)
    idx_n  = idx_normal[:n]
    idx_d1 = idx_d1[:n]
    idx_d2 = idx_d2[:n]

    model.eval()
    with torch.no_grad():
        _, enc_n  = model(x_all[idx_n].to(device))
        _, enc_d1 = model(x_all[idx_d1].to(device))
        _, enc_d2 = model(x_all[idx_d2].to(device))

        # Compose: common from normal, disease heads from respective disease images
        from losses.base import reparameterize
        z_c  = reparameterize(enc_n['mu_c'],   enc_n['lv_c'])
        z_d1 = reparameterize(enc_d1['mu_d1'], enc_d1['lv_d1'])
        z_d2 = reparameterize(enc_d2['mu_d2'], enc_d2['lv_d2'])
        x_comp = model.decoder(z_c, z_d1, z_d2).cpu().numpy()

    # Normal reconstructions for baseline
    x_norm_hat = x_hat_all[idx_n].numpy()

    (br0, br1), (bc0, bc1) = region_blob
    (sr0, sr1), (sc0, sc1) = region_stripe

    blob_comp   = x_comp[:, 0, br0:br1, bc0:bc1].mean()
    blob_normal = x_norm_hat[:, 0, br0:br1, bc0:bc1].mean()
    stripe_comp   = x_comp[:, 0, sr0:sr1, sc0:sc1].mean()
    stripe_normal = x_norm_hat[:, 0, sr0:sr1, sc0:sc1].mean()

    blob_delta   = float(blob_comp   - blob_normal)
    stripe_delta = float(stripe_comp - stripe_normal)
    comp_score   = (blob_delta + stripe_delta) / 2.0

    return {
        'comp_blob_delta':   blob_delta,
        'comp_stripe_delta': stripe_delta,
        'comp_score':        comp_score,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_recon_grid(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
    out_path: str,
    n_cols: int = 8,
):
    """Save a 2-row grid: top = original, bottom = reconstruction."""
    x, y = batch[0][:n_cols].to(device), batch[1][:n_cols]
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x)
    model.train()

    imgs = torch.cat([x.cpu(), x_hat.cpu()], dim=0)   # (2*n_cols, 1, H, W)
    imgs = (imgs.squeeze(1).numpy() + 1) / 2           # to [0,1]

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.5, 3))
    for i in range(n_cols):
        axes[0, i].imshow(imgs[i],           cmap='gray', vmin=0, vmax=1)
        axes[1, i].imshow(imgs[n_cols + i],  cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('original', fontsize=8)
    axes[1, 0].set_ylabel('recon',    fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint / results helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, out_dir: str, epoch: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f'metrics_ep{epoch:04d}.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    # Always overwrite latest
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return path
