"""
Binary SepVAE training script — V1 (CheSS backbone) and V2 (ResNet-50 scratch).

z = [z_common(16ch), z_cardio(16ch)]  @ 16×16 spatial (V2) / 64×64 (V1)

Architecture (V2):
  Encoder: ResNet-50 from scratch + CBAM + self-attn at layer3 bottleneck
           + two learnable Layer4 branches (bg → z_common, tg → z_cardio)
           + optional BboxCrossAttnHead (D1+)
  Nulling: hard-zero z_cardio for Normal images before decoding
  Decoder: progressive bilinear upsampling 16→256 with SE-gated ResBlocks

Loss stack (three separate optimizers):
  VAE optimizer:
    L_rec (MSE) + KL_common + KL_cardio + κ·L_mi + λ·L_bbox + γ·L_perceptual
    + α·L_gan (hinge generator vs PatchGAN) + τ·L_tv (total variation)
  FactorVAE discriminator optimizer:
    L_disc = BCE(D(z_c, z_ca), joint=1) + BCE(D(z_c, z_ca[perm]), marginal=0)
  PatchGAN discriminator optimizer (D5):
    L_patch = hinge(D_patch(x_real), D_patch(x_rec_stale))
"""

import os
# Must be set before JAX/XLA initialises — suppresses C++ WARNING-level logs
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',    '3')
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('GLOG_minloglevel',         '3')

import argparse
import json
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, msgpack_restore, from_state_dict
import torch
from torch.utils.data import DataLoader, Subset

from datasets.VinBigData import VinBigDataPairDataset, jax_pair_collate_fn
from losses.sep_vae_losses import (
    SepVAELossConfig, sepvae_loss,
    FactorDiscriminator, factor_disc_loss,
)
from losses.lpips_gan import NLayerDiscriminator, hinge_d_loss, vanilla_d_loss

try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False

# ── Diagnostic helpers (imported lazily at first use to avoid import-time cost)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.plot_training_scenarios import make_scenario_overlay as _make_scenario_overlay
    from scripts.eval_counterfactual import run_counterfactual_eval as _run_counterfactual_eval
    from scripts.make_scaffolding import run_scaffolding as _run_scaffolding
    from scripts.make_traversal import run_traversal as _run_traversal
    from scripts.make_traversal import run_composition as _run_composition
    _DIAG = True
except Exception as _diag_err:
    _make_scenario_overlay  = None
    _run_counterfactual_eval = None
    _run_scaffolding         = None
    _run_traversal           = None
    _run_composition         = None
    _DIAG = False
    print(f"[warn] Diagnostic scripts not loaded: {_diag_err}")


def parse_args():
    p = argparse.ArgumentParser("Binary SepVAE trainer (Normal vs. Cardiomegaly)")

    # Data
    p.add_argument("--dicom_dir",  type=str, default="/datasets/mmolefe/vinbigdata/train")
    p.add_argument("--csv_path",   type=str, default="/datasets/mmolefe/vinbigdata/train.csv")
    p.add_argument("--use_cache",  action="store_true",
                   help="Load pre-cached .npy files instead of raw DICOMs.")
    p.add_argument("--deterministic_data", action=argparse.BooleanOptionalAction, default=True,
                   help="Deterministic pair construction, loader seeding, and eval subset selection.")
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--exclude_cross_disease_overlap", action="store_true")

    # Model version
    p.add_argument("--model_version",      type=str, default="v2", choices=["v1", "v2"],
                   help="v1=CheSS backbone; v2=ResNet-50 from scratch + CBAM + SE decoder")
    p.add_argument("--use_bbox_cross_attn", action="store_true",
                   help="Enable BboxCrossAttnHead (D1+). D0 uses learned query only.")
    p.add_argument("--attn_heads",          type=int, default=4,
                   help="Number of self-attention heads in ResNet-50 bottleneck (V2).")
    p.add_argument("--decoder_res_blocks",  type=int, default=2,
                   help="ResBlockSE count per decoder level (V2). Default=2 (D0–D1). "
                        "Set to 3 for D2+ — Flax names blocks by loop index so "
                        "existing ResBlockSE_0/1 weights restore cleanly from a "
                        "D1 checkpoint; only ResBlockSE_2 is freshly initialised.")
    p.add_argument("--perceptual_only",     action="store_true",
                   help="Use CheSS as frozen perceptual loss extractor only (D4). "
                        "Does NOT inject CheSS weights into the encoder.")

    # Model dims
    p.add_argument("--z_channels_common",  type=int, default=16)
    p.add_argument("--z_channels_disease", type=int, default=16)
    p.add_argument("--attn_query_dim",     type=int, default=256)

    # CheSS weights (V1 encoder backbone, or V2 perceptual loss with --perceptual_only)
    p.add_argument("--chess_checkpoint", type=str,
                   default="/datasets/mmolefe/chess/pretrained_weights.pth.tar")
    p.add_argument("--chess_converted",  type=str, default=None)

    # Loss weights
    p.add_argument("--weight_rec",           type=float, default=1.0)
    p.add_argument("--weight_kl_common",     type=float, default=1e-4)
    p.add_argument("--weight_kl_disease",    type=float, default=5e-5)
    p.add_argument("--weight_mi_factor",     type=float, default=1.0)
    p.add_argument("--weight_bbox_attn",     type=float, default=0.0)
    p.add_argument("--weight_cardio_supcon", type=float, default=0.05)
    p.add_argument("--weight_perceptual",    type=float, default=0.0)
    p.add_argument("--weight_gan",           type=float, default=0.0,
                   help="PatchGAN hinge generator loss weight (D5+). 0=disabled.")
    p.add_argument("--weight_tv",            type=float, default=0.0,
                   help="Total variation loss weight — suppresses stripe artifacts (D5+).")
    p.add_argument("--weight_masked_rec",    type=float, default=0.0,
                   help="Masked anatomy recon weight: outside-bbox MSE with z_cardio=0. "
                        "Forces z_common not to encode cardiac shape. 0=disabled.")
    p.add_argument("--weight_ctr_reg",       type=float, default=0.0,
                   help="CTR regression weight: L1 loss forcing z_cardio to predict "
                        "cardiac-to-thoracic ratio from CheXmask. 0=disabled. (D5+)")
    p.add_argument("--chexmask_csv",         type=str,   default=None,
                   help="Path to CheXmask VinDr-CXR_preprocessed.csv for mask supervision. "
                        "If None, mask supervision is disabled. (D5+)")
    p.add_argument("--heart_out_zc",         action="store_true",
                   help="Heart-out z_c: zero the cardiac region from h_shared before the "
                        "z_c (bg) encoder branch so z_c cannot encode cardiac features. "
                        "Requires --chexmask_csv. z_d branch still sees the full image. "
                        "Combined with decoder mask gate (Cardiomegaly-only), this forces "
                        "strict heart/non-heart disentanglement without input-level masking.")
    p.add_argument("--heart_in_zd",          action="store_true",
                   help="Heart-in z_d: show ONLY the cardiac region (complement of "
                        "heart_out_zc) to the z_d (tg) encoder branch. Replaces the "
                        "cross-attention disease head with a plain ConvHeadGN — no "
                        "attention is needed when the mask structurally bounds z_d to "
                        "the cardiac spatial support. Together with heart_out_zc this "
                        "gives a full spatial factorisation: z_c↔background, "
                        "z_d↔cardiac silhouette, which is the prerequisite for valid "
                        "CFG composition in the downstream LDM. Requires --chexmask_csv.")
    p.add_argument("--gan_start_step",       type=int,   default=5000,
                   help="Steps from the START OF THIS RUN before PatchGAN activates. "
                        "Counted from phase_start_global_step (not absolute global_step), "
                        "so resuming D5 from a D4 checkpoint still gives a proper warm-up.")
    p.add_argument("--disc_r1_penalty",      type=float, default=10.0,
                   help="R1 gradient penalty weight for PatchGAN discriminator. "
                        "Penalises ||∇_x D(x_real)||^2 — prevents discriminator from "
                        "overfitting and dominating the generator. Standard value: 10.0. "
                        "Set 0 to disable.")
    p.add_argument("--lr_patch_disc",        type=float, default=1e-4,
                   help="Learning rate for PatchGAN discriminator optimizer.")
    p.add_argument("--sigma_inactive",       type=float, default=0.1)
    p.add_argument("--supcon_temperature",   type=float, default=0.1)
    p.add_argument("--kl_warmup_epochs",     type=int,   default=0)
    p.add_argument("--kl_free_bits",         type=float, default=0.0,
                   help="Per-dim KL floor in nats: clamp KL(dim) >= value before summing. "
                        "Prevents posterior collapse and bounds step-to-step KL variance. "
                        "0 = disabled (legacy). Recommended: 0.5 for 16x16x16 latents.")
    p.add_argument("--bbox_query_mix",       type=float, default=0.7,
                   help="Blend weight for bbox-guided query vs learned fallback query.")
    p.add_argument("--bbox_dropout_prob",    type=float, default=0.3,
                   help="Drop bbox guidance from the query path on positive samples only.")

    # Optimizers
    p.add_argument("--lr_vae",       type=float, default=1e-4)
    p.add_argument("--lr_backbone",  type=float, default=1e-5,
                   help="V1 only: CheSS trunk LR (lower than lr_vae).")
    p.add_argument("--lr_disc",      type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    # Training
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--eval_num_workers", type=int, default=0)
    p.add_argument("--seed",         type=int, default=0)

    # Logging & checkpoints
    p.add_argument("--output_root",          type=str, default="runs_sepvae")
    p.add_argument("--exp_name",             type=str, default="sepvae")
    p.add_argument("--log_every",            type=int, default=100)
    p.add_argument("--save_every",           type=int, default=5)
    p.add_argument("--sample_every",         type=int, default=5)
    p.add_argument("--n_samples_per_class",  type=int, default=4)
    p.add_argument("--manifold_every",       type=int, default=-1,
                   help="Manifold plot cadence (-1=match sample_every, 0=disable)")
    p.add_argument("--manifold_max_samples", type=int, default=600)
    p.add_argument("--eval_subset_size",     type=int, default=1024)
    p.add_argument("--manifold_method",      type=str, default="pca",
                   choices=["pca", "tsne", "both"])
    p.add_argument("--manifold_bbox_mode",   type=str, default="both",
                   choices=["bbox_free", "bbox_guided", "both"])

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Resume
    p.add_argument("--resume", type=str, default=None)

    # W&B
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb_project",  type=str, default="baseline-sepvae")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_id",   type=str, default=None)

    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def seed_data_worker(worker_id, base_seed):
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2 ** 32))
    torch.manual_seed(worker_seed)


# ============================================================================
# Visualisation helpers
# ============================================================================

def make_recon_grid(x_input, x_rec, labels, n_per_class=4):
    """Two-row grid: originals (top) | reconstructions (bottom), grouped by class."""
    from torchvision.utils import make_grid
    from PIL import Image

    x_01      = (np.array(x_input) + 1.0) / 2.0
    x_rec_np  = np.clip(np.array(x_rec), 0.0, 1.0)
    labels_np = np.array(labels)

    originals, reconstructions = [], []
    for cls_id in [0, 1]:
        idxs = np.where(labels_np == cls_id)[0][:n_per_class]
        for idx in idxs:
            originals.append(x_01[idx])
            reconstructions.append(x_rec_np[idx])

    all_imgs = originals + reconstructions
    imgs_np  = np.transpose(np.stack(all_imgs, axis=0), (0, 3, 1, 2))
    grid     = make_grid(torch.tensor(imgs_np).clamp(0, 1), nrow=len(originals), padding=2)
    return Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def make_attention_grid(x_input, attn_maps, labels, x_rec=None,
                        heart_masks=None, n_per_class=4, bboxes_cardio=None):
    """
    N×3 panel — one row per sample, three columns:
      Col 0: GT CXR (grayscale)
      Col 1: GT CXR + GT segmentation mask (CheXmask heart mask, cyan)
      Col 2: Reconstruction + predicted segmentation mask (attention map, plasma)

    Row ordering: Normal samples first, then Cardiomegaly.
    N = n_normal + n_cardio (up to n_per_class each class).

    When heart_masks is None (no CheXmask), col 1 falls back to plain GT CXR.
    When x_rec is None, col 2 uses the input CXR instead of reconstruction.

    Global colour anchor on attention ensures Normal row shows near-zero heat
    and Cardiomegaly row shows strong activation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    x_01      = (np.array(x_input) + 1.0) / 2.0          # (2B, H, W, 1) in [0,1]
    attn_ca   = np.array(attn_maps['cardiomegaly'])        # (2B, H_lat, W_lat)
    labels_np = np.array(labels)
    x_rec_np  = np.array(x_rec) if x_rec is not None else x_01  # already [0,1]
    hm_np     = np.array(heart_masks) if heart_masks is not None else None

    # Global colour anchor — prevents normal row being artificially saturated
    global_vmax = float(np.percentile(attn_ca, 99)) if attn_ca.size > 0 else 1.0
    global_vmax = max(global_vmax, 1e-6)

    # Collect sample indices: Normal first, Cardiomegaly second
    norm_idxs = list(np.where(labels_np == 0)[0][:n_per_class])
    card_idxs = list(np.where(labels_np == 1)[0][:n_per_class])
    all_idxs  = norm_idxs + card_idxs
    n_rows    = len(all_idxs)
    if n_rows == 0:
        # fallback: empty image
        from PIL import Image as _PIL
        return _PIL.new('RGB', (300, 100), color=(200, 200, 200))

    col_headers = ['GT CXR', 'GT CXR + GT mask', 'Recon + pred mask']

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(3 * 3.2, n_rows * 2.6),
        squeeze=False,
    )

    # Column header titles on first row only
    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, fontsize=8, pad=3)

    for row_i, idx in enumerate(all_idxs):
        img_in   = x_01[idx, :, :, 0]      # (H, W) in [0,1]
        img_out  = x_rec_np[idx, :, :, 0]  # (H, W) in [0,1]
        H, W     = img_in.shape
        attn_raw = attn_ca[idx]             # (H_lat, W_lat)
        label_str = 'Normal' if idx in norm_idxs else 'Cardio'

        # Upsample attention to full image resolution
        attn_up = np.array(
            jax.image.resize(attn_raw[..., None], (H, W, 1), method='bilinear')[:, :, 0]
        )
        # Locally normalise for display (globally anchored)
        attn_disp = np.clip(attn_up / global_vmax, 0.0, 1.0)

        # Col 0: GT CXR
        axes[row_i, 0].imshow(img_in, cmap='gray', vmin=0, vmax=1,
                               interpolation='lanczos')
        axes[row_i, 0].set_ylabel(label_str, fontsize=7, labelpad=3)
        axes[row_i, 0].axis('off')

        # Col 1: GT CXR + GT segmentation mask
        axes[row_i, 1].imshow(img_in, cmap='gray', vmin=0, vmax=1,
                               interpolation='lanczos')
        if hm_np is not None:
            hm = hm_np[idx].astype(np.float32)   # (S, S)
            hm_up = np.array(
                jax.image.resize(hm[..., None], (H, W, 1), method='nearest')[:, :, 0]
            )
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[hm_up > 0.5] = [0.0, 0.9, 0.9, 0.45]  # cyan fill
            axes[row_i, 1].imshow(rgba, extent=(0, W, H, 0))
            axes[row_i, 1].contour(hm_up, levels=[0.5], colors=['cyan'],
                                   linewidths=[1.0], extent=(0, W, 0, H))
        axes[row_i, 1].axis('off')

        # Col 2: Reconstruction + predicted segmentation mask (attention)
        axes[row_i, 2].imshow(img_out, cmap='gray', vmin=0, vmax=1,
                               interpolation='lanczos')
        axes[row_i, 2].imshow(attn_disp, cmap='plasma', alpha=0.45,
                               vmin=0, vmax=1,
                               extent=(0, W, H, 0), interpolation='bilinear')
        axes[row_i, 2].axis('off')

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_kl_heatmap(model, params, x_normal, x_cardio):
    """
    2-class per-channel KL heatmap (no scipy / 3-class dependency).

    Layout: 2 subplots — Common head | Cardiomegaly head.
      Rows (Y): Normal / Cardiomegaly.
      Cols (X): latent channel index.
      Colour:   mean KL averaged over batch + spatial dims.

    Ideal: Cardio head shows HIGH KL for Cardiomegaly row, near-zero for Normal.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    variables    = {'params': params}
    head_keys    = ['common', 'cardiomegaly']
    head_titles  = {'common': 'Common head', 'cardiomegaly': 'Cardiomegaly head'}
    class_inputs = {'Normal': x_normal, 'Cardiomegaly': x_cardio}
    class_order  = ['Normal', 'Cardiomegaly']

    kl_by_head_class = {h: {} for h in head_keys}
    for cls_name, x in class_inputs.items():
        ld = model.apply(variables, x, method=model.encode)
        for head in head_keys:
            mu, logvar = ld[head]
            kl_elem = 0.5 * (jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar)
            reduce_axes = tuple(range(kl_elem.ndim - 1))  # all dims except channel
            kl_by_head_class[head][cls_name] = np.array(jnp.mean(kl_elem, axis=reduce_axes))

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={'wspace': 0.35})
    for ax, head in zip(axes, head_keys):
        mat  = np.stack([kl_by_head_class[head][cls] for cls in class_order], axis=0)  # (2, C)
        vmax = float(np.percentile(mat, 98)) if mat.max() > 0 else 1.0
        im   = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0.0, vmax=vmax,
                         interpolation='nearest')
        ax.set_title(head_titles[head], fontsize=10, pad=4)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f'ch{i}' for i in range(mat.shape[1])], fontsize=6)
        ax.set_yticks(range(2))
        ax.set_yticklabels(class_order, fontsize=9)
        ax.set_xlabel('Channel', fontsize=8)
        for r in range(2):
            for c in range(mat.shape[1]):
                val     = mat[r, c]
                txt_col = 'white' if val > 0.65 * vmax else 'black'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.0, color=txt_col, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('KL (nats)', fontsize=7)

    fig.suptitle(
        'Per-Channel KL Divergence by Class\n'
        'Ideal: Cardio head HIGH only for Cardiomegaly row',
        fontsize=8,
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def save_latent_manifold_plot(
    model,
    vae_params,
    vae_batch_stats,
    loader,
    save_path,
    max_samples=600,
    method="pca",
    use_bbox_cross_attn=False,
    bbox_mode="both",
):
    """PCA/t-SNE scatter of common + cardio heads with deterministic eval modes."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    if use_bbox_cross_attn:
        modes = ['bbox_free', 'bbox_guided'] if bbox_mode == 'both' else [bbox_mode]
    else:
        modes = ['bbox_free']

    def encode_latents(x, bbox=None, has_bbox=None, heart_mask=None):
        kwargs = {}
        if use_bbox_cross_attn:
            kwargs['bbox']     = bbox
            kwargs['has_bbox'] = has_bbox
        if heart_mask is not None:
            kwargs['heart_mask'] = heart_mask
        return model.apply(variables, x, method=model.encode, **kwargs)

    per_mode = {
        mode: {'common': [], 'cardio': [], 'labels': []}
        for mode in modes
    }

    for batch_torch in loader:
        current_total = len(per_mode[modes[0]]['labels'])
        if current_total >= max_samples:
            break

        x_norm = jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy())
        x_cardio = jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy())
        bbox_cardio = jnp.array(batch_torch['bbox_disease1'].numpy())

        bbox_zero_norm = jnp.zeros((x_norm.shape[0], 4), dtype=jnp.float32)
        has_bbox_zero_norm = jnp.zeros((x_norm.shape[0],), dtype=jnp.float32)
        bbox_zero_cardio = jnp.zeros_like(bbox_cardio)
        has_bbox_zero_cardio = jnp.zeros((x_cardio.shape[0],), dtype=jnp.float32)
        has_bbox_guided_cardio = (
            (bbox_cardio[:, 2] - bbox_cardio[:, 0]) > 1e-4
        ).astype(jnp.float32)

        # heart_mask lives in the second half of the 2B batch tensor
        hm_jax_cardio = None
        hm_jax_norm   = None
        if 'heart_mask' in batch_torch:
            hm_full  = jnp.array(batch_torch['heart_mask'].numpy())  # (2B, S, S)
            B_half   = hm_full.shape[0] // 2
            hm_jax_norm   = hm_full[:B_half]   # (B, S, S) — normal half
            hm_jax_cardio = hm_full[B_half:]   # (B, S, S) — cardio half

        for mode in modes:
            ld_norm = encode_latents(
                x_norm,
                bbox=bbox_zero_norm,
                has_bbox=has_bbox_zero_norm,
                heart_mask=hm_jax_norm,
            )
            if mode == 'bbox_guided':
                ld_cardio = encode_latents(
                    x_cardio,
                    bbox=bbox_cardio,
                    has_bbox=has_bbox_guided_cardio,
                    heart_mask=hm_jax_cardio,
                )
            else:
                ld_cardio = encode_latents(
                    x_cardio,
                    bbox=bbox_zero_cardio,
                    has_bbox=has_bbox_zero_cardio,
                    heart_mask=hm_jax_cardio,
                )

            remaining = max_samples - len(per_mode[mode]['labels'])
            if remaining <= 0:
                continue

            z_c_norm = np.array(jnp.mean(ld_norm['common'][0], axis=(1, 2)))
            z_ca_norm = np.array(jnp.mean(ld_norm['cardiomegaly'][0], axis=(1, 2)))
            z_c_cardio = np.array(jnp.mean(ld_cardio['common'][0], axis=(1, 2)))
            z_ca_cardio = np.array(jnp.mean(ld_cardio['cardiomegaly'][0], axis=(1, 2)))

            take_norm = min(z_c_norm.shape[0], remaining // 2 if remaining > 1 else remaining)
            take_cardio = min(z_c_cardio.shape[0], remaining - take_norm)

            if take_norm > 0:
                per_mode[mode]['common'].extend(z_c_norm[:take_norm])
                per_mode[mode]['cardio'].extend(z_ca_norm[:take_norm])
                per_mode[mode]['labels'].extend([0] * take_norm)
            if take_cardio > 0:
                per_mode[mode]['common'].extend(z_c_cardio[:take_cardio])
                per_mode[mode]['cardio'].extend(z_ca_cardio[:take_cardio])
                per_mode[mode]['labels'].extend([1] * take_cardio)

    if len(per_mode[modes[0]]['labels']) < 2:
        return {}

    methods = ['pca', 'tsne'] if method == 'both' else [method]
    metrics = {}
    colors  = ['blue', 'green']
    names   = ['Normal', 'Cardiomegaly']
    set_titles = {
        'all_heads': 'All heads (common + cardio)',
        'disease_only': 'Cardio head only',
    }

    fig, axes = plt.subplots(
        2,
        max(len(methods) * len(modes), 1),
        figsize=(7 * max(len(methods) * len(modes), 1), 10),
        squeeze=False,
    )

    for mode_idx, mode in enumerate(modes):
        lc = np.array(per_mode[mode]['common'])
        lcard = np.array(per_mode[mode]['cardio'])
        lbls = np.array(per_mode[mode]['labels'])

        cardio_norm = np.linalg.norm(lcard, axis=1)
        inactive_mask = lbls == 0
        active_mask = lbls == 1
        inactive_norm = float(cardio_norm[inactive_mask].mean()) if inactive_mask.any() else float('nan')
        active_norm = float(cardio_norm[active_mask].mean()) if active_mask.any() else float('nan')
        ratio = float(active_norm / max(inactive_norm, 1e-6)) if np.isfinite(active_norm) and np.isfinite(inactive_norm) else float('nan')

        metrics[f'z_cardio_norm_inactive_{mode}'] = inactive_norm
        metrics[f'z_cardio_norm_active_{mode}'] = active_norm
        metrics[f'z_cardio_norm_ratio_{mode}'] = ratio

        feature_sets = {
            'all_heads': np.concatenate([lc, lcard], axis=1),
            'disease_only': lcard,
        }

        for row_idx, (set_name, feats) in enumerate(feature_sets.items()):
            for method_idx, m in enumerate(methods):
                col_idx = mode_idx * len(methods) + method_idx
                ax = axes[row_idx][col_idx]
                reducer = (
                    TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=min(30, max(5, len(feats) // 4)),
                    )
                    if m == 'tsne'
                    else PCA(n_components=2, random_state=42)
                )
                z2 = reducer.fit_transform(feats)
                for d in [0, 1]:
                    mask = lbls == d
                    ax.scatter(
                        z2[mask, 0],
                        z2[mask, 1],
                        c=colors[d],
                        label=names[d],
                        alpha=0.6,
                        s=20,
                    )
                try:
                    sil = float(silhouette_score(z2, lbls))
                except ValueError:
                    sil = float('nan')
                metrics[f'silhouette_{set_name}_{m}_{mode}'] = sil
                sil_str = f"{sil:.3f}" if np.isfinite(sil) else "N/A"
                ax.set_title(f"{set_titles[set_name]} ({m.upper()}, {mode}) — sil={sil_str}")
                ax.legend()
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.manifold_every < 0:
        args.manifold_every = args.sample_every

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    IS_V2 = (args.model_version == 'v2')

    print("=" * 60)
    print(f"BINARY SEPVAE  model={args.model_version}  "
          f"z=[z_common({args.z_channels_common}), z_cardio({args.z_channels_disease})]")
    print("Obj 1 — orthogonality: KL + FactorVAE MI + bbox attn")
    print("Obj 2 — crisp recon:   MSE" +
          (" + CheSS perceptual" if args.weight_perceptual > 0 else ""))
    print("=" * 60)

    # ── Output dirs ───────────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_slug    = f"{args.exp_name}-{timestamp}"
    output_dir  = Path(args.output_root) / exp_slug
    ckpt_dir    = ensure_dir(output_dir / "checkpoints")
    samples_dir = ensure_dir(output_dir / "samples")
    manifold_dir = ensure_dir(output_dir / "manifold")
    diag_dir    = ensure_dir(output_dir / "diagnostics")
    metrics_history_path = output_dir / "metrics_history.jsonl"
    print(f"Output: {output_dir}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb and _WANDB:
        wandb_kwargs = dict(project=args.wandb_project, entity=args.wandb_entity,
                            config=vars(args))
        if args.wandb_run_id:
            wandb_kwargs['id']     = args.wandb_run_id
            wandb_kwargs['resume'] = 'must'
        else:
            wandb_kwargs['name'] = exp_slug
        wandb.init(**wandb_kwargs)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VinBigDataPairDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        exclude_cross_disease_overlap=getattr(args, 'exclude_cross_disease_overlap', False),
        use_cache=args.use_cache,
        deterministic_pairs=args.deterministic_data,
        pair_seed=args.seed,
        chexmask_csv=getattr(args, 'chexmask_csv', None),
    )
    train_loader_generator = None
    worker_init_fn = None
    if args.deterministic_data:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(args.seed)
        worker_init_fn = partial(seed_data_worker, base_seed=args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=jax_pair_collate_fn,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=train_loader_generator,
    )
    print(f"Dataset: {len(dataset)} pairs, {len(loader)} steps/epoch")

    n_eval_pairs = min(max(args.eval_subset_size // 2, 1), len(dataset))
    if args.deterministic_data:
        eval_rng = np.random.default_rng(args.seed)
        eval_indices = np.sort(eval_rng.choice(len(dataset), size=n_eval_pairs, replace=False))
    else:
        eval_indices = np.arange(n_eval_pairs)

    eval_subset = Subset(dataset, eval_indices.tolist())
    eval_loader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        collate_fn=jax_pair_collate_fn,
        drop_last=False,
    )
    eval_subset_ids = []
    for idx in eval_indices.tolist():
        norm_id, cardio_id = dataset.get_pair_ids(idx)
        eval_subset_ids.append({
            'pair_index': int(idx),
            'normal_id': norm_id,
            'cardio_id': cardio_id,
        })
    with open(output_dir / "eval_subset_ids.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                'seed': args.seed,
                'deterministic_data': bool(args.deterministic_data),
                'pair_count': n_eval_pairs,
                'eval_subset_size': int(n_eval_pairs * 2),
                'pairs': eval_subset_ids,
            },
            f,
            indent=2,
        )
    print(f"Eval subset: {n_eval_pairs} fixed pairs ({n_eval_pairs * 2} images)")

    # ── Model ─────────────────────────────────────────────────────────────────
    if IS_V2:
        from models.sep_vae_v2 import SepVAEV2
        sepvae = SepVAEV2(
            z_channels_common=args.z_channels_common,
            z_channels_disease=args.z_channels_disease,
            query_dim=args.attn_query_dim,
            attn_heads=args.attn_heads,
            use_bbox_cross_attn=args.use_bbox_cross_attn,
            bbox_query_mix=args.bbox_query_mix,
            decoder_res_blocks=args.decoder_res_blocks,
            heart_out_zc=args.heart_out_zc,
            heart_in_zd=args.heart_in_zd,
        )
        vae_batch_stats = {}   # GroupNorm — no batch_stats
        dummy_x      = jnp.ones((1, args.img_size, args.img_size, 1))
        dummy_labels = jnp.array([0])
        rng, init_rng = jax.random.split(rng)
        vae_vars = sepvae.init(init_rng, dummy_x, dummy_labels, key=init_rng)
        vae_params = jax.tree_util.tree_map(jnp.array, vae_vars['params'])
        n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print(f"SepVAEV2 parameters: {n_vae_params:,}")
        print(f"  heart_out_zc: {args.heart_out_zc}  heart_in_zd: {args.heart_in_zd}  "
              f"bbox cross-attn: {args.use_bbox_cross_attn}  "
              f"img_size: {args.img_size}  decoder_res_blocks: {args.decoder_res_blocks}")

    else:
        # ── V1: CheSS backbone ────────────────────────────────────────────────
        from models.sep_vae_jax import SepVAE
        from models.resnet_jax import ResNet50CheSS
        from utils.weight_converter import convert_chess_resnet50, load_converted_weights

        sepvae = SepVAE(
            z_channels_common=args.z_channels_common,
            z_channels_disease=args.z_channels_disease,
            attn_query_dim=args.attn_query_dim,
        )

        if args.chess_converted and os.path.exists(args.chess_converted):
            chess_params, chess_batch_stats = load_converted_weights(args.chess_converted)
        else:
            chess_params, chess_batch_stats = convert_chess_resnet50(
                args.chess_checkpoint, verbose=True
            )
            from utils.weight_converter import save_converted_weights
            converted_path = output_dir / "chess_jax_params.npy"
            save_converted_weights((chess_params, chess_batch_stats), str(converted_path))
            print(f"Saved converted weights: {converted_path}")

        chess_trunk_params = {k: v for k, v in chess_params.items()
                              if not k.startswith('layer4')}
        chess_trunk_stats  = {k: v for k, v in chess_batch_stats.items()
                              if not k.startswith('layer4')}
        chess_l4_params    = {k: v for k, v in chess_params.items()
                              if k.startswith('layer4')}
        chess_l4_stats     = {k: v for k, v in chess_batch_stats.items()
                              if k.startswith('layer4')}

        dummy_x      = jnp.ones((1, args.img_size, args.img_size, 1))
        dummy_labels = jnp.array([0])
        rng, init_rng = jax.random.split(rng)
        vae_vars = sepvae.init(
            {'params': init_rng, 'dropout': init_rng},
            dummy_x, dummy_labels, key=init_rng, train=True,
        )
        vae_params      = vae_vars['params']
        vae_batch_stats = vae_vars.get('batch_stats', {})

        vae_params = dict(vae_params)
        vae_params['encoder'] = dict(vae_params['encoder'])
        vae_params['encoder']['backbone']  = chess_trunk_params
        vae_params['encoder']['bg_branch'] = chess_l4_params
        vae_params['encoder']['tg_branch'] = chess_l4_params

        if vae_batch_stats:
            vae_batch_stats = dict(vae_batch_stats)
            vae_batch_stats['encoder'] = dict(vae_batch_stats.get('encoder', {}))
            vae_batch_stats['encoder']['backbone']  = chess_trunk_stats
            vae_batch_stats['encoder']['bg_branch'] = chess_l4_stats
            vae_batch_stats['encoder']['tg_branch'] = chess_l4_stats
        else:
            vae_batch_stats = {
                'encoder': {
                    'backbone':  chess_trunk_stats,
                    'bg_branch': chess_l4_stats,
                    'tg_branch': chess_l4_stats,
                }
            }

        vae_params      = jax.tree_util.tree_map(jnp.array, vae_params)
        vae_batch_stats = jax.tree_util.tree_map(jnp.array, vae_batch_stats)
        n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print(f"SepVAE (V1) parameters: {n_vae_params:,}")

    # ── Perceptual backbone (frozen CheSS, only loaded when needed) ───────────
    backbone_for_percep  = None
    backbone_vars_percep = None
    if args.weight_perceptual > 0.0:
        if IS_V2 and not args.perceptual_only:
            raise ValueError(
                "V2 + weight_perceptual > 0 requires --perceptual_only "
                "(CheSS is not used as V2 encoder backbone)"
            )
        from models.resnet_jax import ResNet50CheSS
        from utils.weight_converter import convert_chess_resnet50, load_converted_weights
        if args.chess_converted and os.path.exists(args.chess_converted):
            _cp, _cs = load_converted_weights(args.chess_converted)
        else:
            _cp, _cs = convert_chess_resnet50(args.chess_checkpoint, verbose=False)
        backbone_for_percep  = ResNet50CheSS()
        backbone_vars_percep = jax.tree_util.tree_map(
            jnp.array, {'params': _cp, 'batch_stats': _cs}
        )
        print(f"Perceptual loss: enabled (weight={args.weight_perceptual})")
    else:
        print("Perceptual loss: disabled")

    if args.weight_bbox_attn > 0.0:
        print(f"Bbox attention supervision: enabled (weight={args.weight_bbox_attn})")

    # ── FactorVAE discriminator ────────────────────────────────────────────────
    use_factor_disc = args.weight_mi_factor > 0.0
    discriminator = None
    disc_state = None
    if use_factor_disc:
        disc_input_dim = args.z_channels_common + args.z_channels_disease
        discriminator  = FactorDiscriminator(hidden_dim=64)
        rng, disc_rng  = jax.random.split(rng)
        disc_vars      = discriminator.init(disc_rng, jnp.ones((1, disc_input_dim)))
        disc_params    = disc_vars['params']
        n_disc_params  = sum(p.size for p in jax.tree_util.tree_leaves(disc_params))
        print(f"FactorDiscriminator parameters: {n_disc_params:,}")
    else:
        print("FactorDiscriminator: disabled (weight_mi_factor=0)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if IS_V2:
        # Single AdamW — all V2 params trained at the same LR (no frozen CheSS trunk)
        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=args.lr_vae, weight_decay=args.weight_decay),
        )
        print(f"VAE optimizer:  AdamW lr={args.lr_vae}  wd={args.weight_decay}")
    else:
        # V1: backbone (layers 1-3) gets a lower LR to preserve CheSS features
        from flax import traverse_util

        def _vae_label_fn(params):
            flat = traverse_util.flatten_dict(params)
            labels = {
                k: 'backbone' if k[0] == 'encoder' and len(k) > 1 and k[1] == 'backbone'
                   else 'vae'
                for k in flat
            }
            return traverse_util.unflatten_dict(labels)

        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.multi_transform(
                transforms={
                    'backbone': optax.adamw(learning_rate=args.lr_backbone,
                                            weight_decay=args.weight_decay),
                    'vae':      optax.adamw(learning_rate=args.lr_vae,
                                            weight_decay=args.weight_decay),
                },
                param_labels=_vae_label_fn,
            ),
        )
        print(f"VAE optimizer:  AdamW backbone_lr={args.lr_backbone}  "
              f"vae_lr={args.lr_vae}  wd={args.weight_decay}")

    vae_state  = TrainState.create(apply_fn=None, params=vae_params,  tx=tx_vae)
    ema_params = jax.tree_util.tree_map(jnp.array, vae_params)
    if use_factor_disc:
        tx_disc = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adam(learning_rate=args.lr_disc),
        )
        disc_state = TrainState.create(apply_fn=None, params=disc_params, tx=tx_disc)
        print(f"Disc optimizer: Adam  (lr={args.lr_disc})")

    # ── PatchGAN discriminator (D5) ───────────────────────────────────────────
    # NLayerDiscriminator operates in image space (x_real vs x_rec). Frozen
    # during VAE updates and only initialised when GAN training is active.
    use_patch_disc = args.weight_gan > 0.0
    patch_discriminator = None
    patch_disc_state = None
    if use_patch_disc:
        patch_discriminator = NLayerDiscriminator(in_channels=1, n_layers=3)
        rng, patch_disc_rng = jax.random.split(rng)
        patch_disc_vars     = patch_discriminator.init(
            patch_disc_rng, jnp.ones((1, args.img_size, args.img_size, 1))
        )
        patch_disc_params   = patch_disc_vars['params']
        n_patch_disc_params = sum(p.size for p in jax.tree_util.tree_leaves(patch_disc_params))
        print(f"PatchGAN discriminator parameters: {n_patch_disc_params:,}")
        tx_patch_disc  = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adam(learning_rate=args.lr_patch_disc),
        )
        patch_disc_state = TrainState.create(
            apply_fn=None,
            params=patch_disc_params,
            tx=tx_patch_disc,
        )
        print(f"PatchGAN optimizer: Adam  (lr={args.lr_patch_disc}  "
              f"weight_gan={args.weight_gan}  gan_start_step={args.gan_start_step})")
    else:
        print("PatchGAN discriminator: disabled (weight_gan=0)")

    # ── Resume ────────────────────────────────────────────────────────────────
    def _find_new_keys(target, source, prefix=""):
        """Collect top-level names of keys present in *target* but not in *source*."""
        new = []
        if not isinstance(target, dict) or not isinstance(source, dict):
            return new
        for k in target:
            if k not in source:
                new.append(f"{prefix}{k}" if prefix else k)
            elif isinstance(target[k], dict):
                new.extend(_find_new_keys(target[k], source[k], prefix=f"{prefix}{k}/"))
        return new

    def _merge_params(target, source):
        """Recursively merge *source* (checkpoint) into *target* (fresh init).

        For every key in *target*:
        - If the key exists in *source*, recurse (dicts) or adopt the value (leaves).
        - If the leaf shapes differ (e.g. decoder ch_mults 64→128 changes ResBlockSE
          weight shapes), the target's fresh init is kept so JIT never sees a shape error.
        - If the key is absent in *source* (e.g. newly added layer), keep *target*'s
          freshly-initialised value so the model can still run.
        Keys present in *source* but absent in *target* are silently dropped.
        """
        if not isinstance(target, dict):
            source_arr = jnp.array(source)
            if source_arr.shape != target.shape:
                # Architecture change (e.g. decoder ch_mults 64→128): keep fresh init
                print(f"    [_merge_params] shape mismatch {source_arr.shape} → {target.shape}, keeping fresh init")
                return target
            return source_arr
        result = {}
        for k, v in target.items():
            if k in source:
                result[k] = (_merge_params(v, source[k])
                             if isinstance(v, dict) and isinstance(source[k], dict)
                             else _merge_params(v, source[k]))
            else:
                result[k] = v   # new layer — keep fresh init
        skipped = set(source.keys()) - set(target.keys()) if isinstance(source, dict) else set()
        if skipped:
            print(f"    [_merge_params] ignored stale keys: {sorted(skipped)}")
        return result

    start_epoch = 1
    global_step = 0
    phase_start_global_step = 0   # set after resume load; gan_start_step is relative to this
    if args.resume:
        with open(args.resume, 'rb') as f:
            ckpt = msgpack_restore(f.read())

        # Partial param merge: new layers (dec_attn_32, extra ResBlockSE …) keep fresh inits.
        ckpt_vae_raw   = jax.tree_util.tree_map(jnp.array, ckpt['vae_params'])
        merged_params  = _merge_params(vae_state.params, ckpt_vae_raw)
        new_keys       = _find_new_keys(vae_state.params, ckpt_vae_raw)
        if new_keys:
            print(f"  Partial warm-start: {len(new_keys)} new decoder key(s) init'd fresh → "
                  + ", ".join(sorted(new_keys)[:6]) + ("…" if len(new_keys) > 6 else ""))

        # Optimizer state: restore only when ALL param shapes are unchanged.
        # Adam mu/nu buffers must have identical shapes to params; if z_common grew
        # (e.g. 16→32) then z_proj kernel changed (3,3,32,512)→(3,3,48,512) and
        # restoring the old opt_state would cause a TypeError in the first gradient step.
        _ckpt_vae_leaves  = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: jnp.array(x).shape, ckpt_vae_raw))
        _fresh_vae_leaves = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: x.shape, merged_params))
        _vae_shapes_ok = (_ckpt_vae_leaves == _fresh_vae_leaves)
        if _vae_shapes_ok and 'vae_opt_state' in ckpt:
            try:
                restored_vae_opt = from_state_dict(vae_state.opt_state, ckpt['vae_opt_state'])
                vae_state = vae_state.replace(params=merged_params,
                                              opt_state=restored_vae_opt,
                                              step=int(ckpt['global_step']))
                print("  VAE optimizer state: restored from checkpoint")
            except Exception as _exc:
                print(f"  VAE optimizer state mismatch ({_exc.__class__.__name__}: "
                      f"{str(_exc)[:120]})")
                print("  → Starting VAE optimizer fresh (params partially restored).")
                vae_state = vae_state.replace(params=merged_params)
        else:
            if not _vae_shapes_ok:
                print("  [VAE restore] param shapes changed (e.g. z_common grew) → fresh optimizer state")
            vae_state = vae_state.replace(params=merged_params)

        if use_factor_disc and disc_state is not None and 'disc_params' in ckpt:
            # Use _merge_params so shape-changed params (e.g. disc_input_dim changed
            # when z_common grows from 16→32) get fresh init instead of a shape crash.
            restored_disc_params = _merge_params(disc_state.params, ckpt['disc_params'])
            # Only restore opt_state when ALL param shapes are unchanged — opt_state
            # (Adam mu/nu) must have the same shapes as params.  If any shape changed,
            # keep the freshly-initialised opt_state (disc re-converges in ~200 steps).
            _ckpt_disc_leaves  = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: jnp.array(x).shape, ckpt['disc_params']))
            _fresh_disc_leaves = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: x.shape, restored_disc_params))
            _disc_shapes_ok = (_ckpt_disc_leaves == _fresh_disc_leaves)
            if _disc_shapes_ok and 'disc_opt_state' in ckpt:
                try:
                    restored_disc_opt = from_state_dict(disc_state.opt_state, ckpt['disc_opt_state'])
                    disc_state = disc_state.replace(params=restored_disc_params,
                                                    opt_state=restored_disc_opt)
                except (ValueError, KeyError):
                    disc_state = disc_state.replace(params=restored_disc_params)
            else:
                if not _disc_shapes_ok:
                    print("  [disc restore] param shapes changed → fresh optimizer state")
                disc_state = disc_state.replace(params=restored_disc_params)
        if use_patch_disc and patch_disc_state is not None and 'patch_disc_params' in ckpt:
            restored_pd_params = _merge_params(patch_disc_state.params, ckpt['patch_disc_params'])
            _ckpt_pd_leaves  = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: jnp.array(x).shape, ckpt['patch_disc_params']))
            _fresh_pd_leaves = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: x.shape, restored_pd_params))
            _pd_shapes_ok = (_ckpt_pd_leaves == _fresh_pd_leaves)
            if _pd_shapes_ok and 'patch_disc_opt_state' in ckpt:
                try:
                    restored_pd_opt = from_state_dict(patch_disc_state.opt_state,
                                                       ckpt['patch_disc_opt_state'])
                    patch_disc_state = patch_disc_state.replace(params=restored_pd_params,
                                                                 opt_state=restored_pd_opt)
                except (ValueError, KeyError):
                    patch_disc_state = patch_disc_state.replace(params=restored_pd_params)
            else:
                if not _pd_shapes_ok:
                    print("  [patch_disc restore] param shapes changed → fresh optimizer state")
                patch_disc_state = patch_disc_state.replace(params=restored_pd_params)
        if vae_batch_stats and 'vae_batch_stats' in ckpt:
            vae_batch_stats = jax.tree_util.tree_map(jnp.array, ckpt['vae_batch_stats'])
        if 'ema_params' in ckpt:
            ema_params = _merge_params(ema_params, jax.tree_util.tree_map(jnp.array,
                                                                           ckpt['ema_params']))
        rng         = jnp.array(ckpt['rng'])
        start_epoch = int(ckpt['epoch']) + 1
        global_step = int(ckpt['global_step'])
        # phase_start_global_step anchors gan_start_step to this run, not the entire
        # curriculum history — prevents D5 from bypassing GAN warm-up on resume.
        phase_start_global_step = global_step
        print(f"Resumed from epoch {int(ckpt['epoch'])}, step {global_step}")

        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but --epochs={args.epochs}. --epochs is interpreted as the "
                "final epoch number after resume, so this run would execute "
                "zero training epochs."
            )

    # ── Loss config ───────────────────────────────────────────────────────────
    loss_cfg = SepVAELossConfig(
        weight_rec=args.weight_rec,
        weight_perceptual=args.weight_perceptual,
        weight_gan=args.weight_gan,
        weight_tv=args.weight_tv,
        weight_masked_rec=args.weight_masked_rec,
        weight_kl_common=args.weight_kl_common,
        weight_kl_disease=args.weight_kl_disease,
        weight_mi_factor=args.weight_mi_factor,
        weight_bbox_attn=args.weight_bbox_attn,
        weight_ctr_reg=getattr(args, 'weight_ctr_reg', 0.0),
        weight_cardio_supcon=args.weight_cardio_supcon,
        supcon_temperature=args.supcon_temperature,
        sigma_inactive=args.sigma_inactive,
        kl_free_bits=args.kl_free_bits,
    )
    print(f"\nLoss weights: rec={loss_cfg.weight_rec}  "
          f"percep={loss_cfg.weight_perceptual}  "
          f"gan={loss_cfg.weight_gan}  "
          f"tv={loss_cfg.weight_tv}  "
          f"masked_rec={loss_cfg.weight_masked_rec}  "
          f"kl_c={loss_cfg.weight_kl_common}  "
          f"kl_d={loss_cfg.weight_kl_disease}  "
          f"kl_free_bits={loss_cfg.kl_free_bits}  "
          f"mi_factor={loss_cfg.weight_mi_factor}  "
          f"supcon={loss_cfg.weight_cardio_supcon}  "
          f"bbox={loss_cfg.weight_bbox_attn}")

    # ── JIT'd steps ───────────────────────────────────────────────────────────
    _backbone_apply_fn = backbone_for_percep.apply if backbone_for_percep else None
    _backbone_vars     = backbone_vars_percep
    _batch_stats_arg   = vae_batch_stats if vae_batch_stats else None

    @jax.jit
    def update_ema(ema_p, params, decay):
        return jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, ema_p, params
        )

    @jax.jit
    def get_pooled_latents(vae_params_arg, x, bbox_full, has_bbox):
        """Encoder-only forward pass → spatially pooled z_c and z_ca."""
        variables = {'params': vae_params_arg}
        if _batch_stats_arg is not None:
            variables['batch_stats'] = _batch_stats_arg
        if IS_V2 and args.use_bbox_cross_attn:
            ld = sepvae.apply(
                variables,
                x,
                bbox=bbox_full,
                has_bbox=has_bbox,
                method=sepvae.encode,
            )
        else:
            ld = sepvae.apply(variables, x, method=sepvae.encode)
        z_c  = jnp.mean(ld['common'][0],       axis=(1, 2))
        z_ca = jnp.mean(ld['cardiomegaly'][0], axis=(1, 2))
        return z_c, z_ca

    @jax.jit
    def disc_step(disc_state_arg, z_c, z_ca, key):
        """Update discriminator to distinguish joint vs. permuted-marginal."""
        def disc_loss_fn(d_params):
            return factor_disc_loss(d_params, discriminator, z_c, z_ca, key)
        (d_loss, d_acc), grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(
            disc_state_arg.params
        )
        return disc_state_arg.apply_gradients(grads=grads), d_loss, d_acc

    _r1_weight = args.disc_r1_penalty  # captured in closure; avoids Python overhead in jit

    @jax.jit
    def patch_disc_step(patch_disc_state_arg, x_real, x_rec_stale):
        """Update PatchGAN discriminator: real vs stale reconstruction.
        x_rec is stop_gradient'd so gradients flow only through the discriminator.
        Includes optional R1 gradient penalty on real samples to prevent discriminator
        from overfitting and dominating the generator (Bug 3 fix).
        Returns updated state, disc loss, and patch disc accuracy."""
        x_real_sg = jax.lax.stop_gradient(x_real)
        x_fake_sg = jax.lax.stop_gradient(x_rec_stale)

        def patch_disc_loss_fn(pd_params):
            real_logits = patch_discriminator.apply({'params': pd_params}, x_real_sg, train=True)
            fake_logits = patch_discriminator.apply({'params': pd_params}, x_fake_sg, train=True)
            pd_loss = hinge_d_loss(real_logits, fake_logits)
            # R1 gradient penalty: penalise large discriminator gradients on real samples.
            # This prevents the discriminator from growing arbitrarily strong relative to
            # the generator.  E[||∇_x D(x_real)||^2] — standard γ=10.
            if _r1_weight > 0.0:
                def _disc_mean(x):
                    return jnp.mean(patch_discriminator.apply({'params': pd_params}, x, train=True))
                r1_grads = jax.grad(_disc_mean)(x_real_sg)                # (N,H,W,1)
                r1_penalty = 0.5 * jnp.mean(jnp.sum(jnp.square(r1_grads), axis=(1, 2, 3)))
                pd_loss = pd_loss + _r1_weight * r1_penalty
            # Accuracy: real predicted as real (>0) and fake predicted as fake (<=0)
            pd_acc = (
                jnp.mean((real_logits > 0).astype(jnp.float32)) * 0.5 +
                jnp.mean((fake_logits <= 0).astype(jnp.float32)) * 0.5
            )
            return pd_loss, pd_acc

        (pd_loss, pd_acc), grads = jax.value_and_grad(patch_disc_loss_fn, has_aux=True)(
            patch_disc_state_arg.params
        )
        return patch_disc_state_arg.apply_gradients(grads=grads), pd_loss, pd_acc

    @jax.jit
    def vae_step(vae_state_arg, batch, disc_params_frozen, patch_disc_params_frozen,
                 key, kl_anneal):
        """Update VAE with all losses including FactorVAE MI and PatchGAN (both discs frozen).
        bbox_full and has_bbox are pre-assembled in the batch dict by the train loop."""
        bbox_arg           = batch.get('bbox_full')       # (2B, 4) or None
        has_bbox_arg       = batch.get('has_bbox')        # (2B,)  or None
        has_bbox_query_arg = batch.get('has_bbox_query')  # (2B,)  or None
        heart_mask_arg     = batch.get('heart_mask')      # (2B, S, S) or None
        ctr_arg            = batch.get('ctr')             # (2B,) or None
        has_mask_arg       = batch.get('has_mask')        # (2B,) or None

        def loss_fn(params):
            total_loss, logs, z_c, z_ca, x_rec = sepvae_loss(
                sepvae, params, batch, key, loss_cfg,
                batch_stats=_batch_stats_arg,
                kl_anneal=kl_anneal,
                disc_params=disc_params_frozen,
                discriminator=discriminator,
                backbone_apply_fn=_backbone_apply_fn,
                backbone_variables=_backbone_vars,
                bbox=bbox_arg,
                has_bbox=has_bbox_arg,
                has_bbox_query=has_bbox_query_arg,
                patch_disc_params=patch_disc_params_frozen,
                patch_discriminator=patch_discriminator,
                heart_mask=heart_mask_arg,
                ctr=ctr_arg,
                has_mask=has_mask_arg,
            )
            return total_loss, (logs, z_c, z_ca, x_rec)
        (loss, (logs, z_c, z_ca, x_rec)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            vae_state_arg.params
        )
        return vae_state_arg.apply_gradients(grads=grads), logs, z_c, z_ca, x_rec

    @jax.jit
    def reconstruct_and_encode(vae_params_arg, x, labels, key, bbox_full, has_bbox,
                                heart_mask=None):
        """Full forward pass for visualisation."""
        variables = {'params': vae_params_arg}
        if _batch_stats_arg is not None:
            variables['batch_stats'] = _batch_stats_arg
        if IS_V2 and args.use_bbox_cross_attn:
            x_rec, latents_dict, _, _, _ = sepvae.apply(
                variables, x, labels, key=key, train=False,
                bbox=bbox_full, has_bbox=has_bbox,
                heart_mask=heart_mask,
            )
        else:
            x_rec, latents_dict, _, _, _ = sepvae.apply(
                variables, x, labels, key=key, train=False,
                heart_mask=heart_mask,
            )
        return x_rec, latents_dict['attn_maps']

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING" + (f" (resuming from epoch {start_epoch})" if start_epoch > 1 else ""))
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        kl_anneal = jnp.float32(
            min(1.0, epoch / args.kl_warmup_epochs) if args.kl_warmup_epochs > 0 else 1.0
        )
        anneal_str = (f"  [KL anneal={float(kl_anneal):.3f}]"
                      if args.kl_warmup_epochs > 0 else "")
        print(f"\nEpoch {epoch}/{args.epochs}{anneal_str}")

        epoch_logs = []
        last_batch = None

        for batch_torch in loader:
            batch = {
                'x_norm':         jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()),
                'x_disease1':     jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()),
                'disease_labels': jnp.array(batch_torch['disease_labels'].numpy()),
                'bbox_disease1':  jnp.array(batch_torch['bbox_disease1'].numpy()),
            }
            # CheXmask mask supervision fields (present when chexmask_csv was provided)
            if 'heart_mask' in batch_torch:
                batch['heart_mask'] = jnp.array(batch_torch['heart_mask'].numpy())   # (2B, S, S)
                batch['ctr']        = jnp.array(batch_torch['ctr'].numpy())          # (2B,)
                batch['has_mask']   = jnp.array(batch_torch['has_mask'].numpy())     # (2B,)

            # ── Assemble (2B, 4) bbox tensor for V2 cross-attention ──────────
            # Normal images get a zero bbox (will have has_bbox=0 → fallback query)
            bbox_cardio = batch['bbox_disease1']                          # (B, 4)
            bbox_full   = jnp.concatenate(
                [jnp.zeros_like(bbox_cardio), bbox_cardio], axis=0
            )                                                              # (2B, 4)
            has_bbox    = (
                (bbox_full[:, 2] - bbox_full[:, 0]) > 1e-4
            ).astype(jnp.float32)                                          # (2B,)
            has_bbox_query = has_bbox
            if args.use_bbox_cross_attn and args.bbox_dropout_prob > 0.0:
                rng, key_bbox_dropout = jax.random.split(rng)
                keep_mask = (
                    1.0 - jax.random.bernoulli(
                        key_bbox_dropout,
                        p=args.bbox_dropout_prob,
                        shape=has_bbox.shape,
                    ).astype(jnp.float32)
                )
                has_bbox_query = has_bbox * keep_mask
            batch['bbox_full'] = bbox_full
            batch['has_bbox'] = has_bbox
            batch['has_bbox_query'] = has_bbox_query

            rng, key_disc, key_vae = jax.random.split(rng, 3)
            x_full = jnp.concatenate([batch['x_norm'], batch['x_disease1']], axis=0)
            x_real_01 = (x_full + 1.0) / 2.0

            if use_factor_disc and disc_state is not None:
                z_c_curr, z_ca_curr = get_pooled_latents(
                    vae_state.params, x_full, batch['bbox_full'], batch['has_bbox_query']
                )
                disc_state, disc_loss, disc_acc = disc_step(
                    disc_state, jax.lax.stop_gradient(z_c_curr), jax.lax.stop_gradient(z_ca_curr), key_disc
                )
                disc_params_frozen = jax.lax.stop_gradient(disc_state.params)
            else:
                disc_loss = jnp.float32(0.0)
                disc_acc = jnp.float32(0.0)
                disc_params_frozen = None

            phase_local_step = global_step - phase_start_global_step
            gan_active = use_patch_disc and patch_disc_state is not None and phase_local_step >= args.gan_start_step
            patch_disc_params_frozen = (
                jax.lax.stop_gradient(patch_disc_state.params) if gan_active else None
            )

            # Step 2: VAE update on the same batch.
            vae_state, logs, _, _, x_rec = vae_step(
                vae_state, batch, disc_params_frozen, patch_disc_params_frozen,
                key_vae, kl_anneal
            )

            # Step 3: PatchGAN discriminator update on the current reconstruction.
            # D6 FIX: full batch (Normal + Cardiomegaly) — both classes need sharpening.
            if gan_active:
                patch_disc_state, patch_disc_loss, patch_disc_acc = patch_disc_step(
                    patch_disc_state, x_real_01, x_rec
                )
            else:
                patch_disc_loss = jnp.float32(0.0)
                patch_disc_acc = jnp.float32(0.0)

            if args.ema_decay > 0:
                ema_params = update_ema(ema_params, vae_state.params, args.ema_decay)

            logs = dict(logs) | {
                'loss/disc':              disc_loss,
                'metrics/disc_acc':       disc_acc,
                'loss/patch_disc':        patch_disc_loss,
                'metrics/patch_disc_acc': patch_disc_acc,
            }
            epoch_logs.append(logs)
            last_batch = batch

            if global_step % args.log_every == 0:
                print(f"  Step {global_step}: "
                      f"loss={float(logs['loss/total']):.4f}  "
                      f"rec={float(logs['loss/reconstruction']):.4f}  "
                      f"kl={float(logs['loss/kl_total']):.4f}  "
                      f"mi={float(logs['loss/mi_factor']):.4f}  "
                      f"supcon={float(logs['loss/cardio_supcon']):.4f}  "
                      f"disc={float(logs['loss/disc']):.4f}  "
                      f"D_acc={float(logs['metrics/disc_acc']):.2f}  "
                      f"bbox={float(logs['loss/bbox_attn']):.4f}  "
                      f"ratio={float(logs['metrics/z_cardio_norm_ratio']):.2f}  "
                      f"gan_g={float(logs['loss/gan_g']):.4f}  "
                      f"tv={float(logs['loss/tv']):.4f}  "
                      f"msk={float(logs['loss/masked_rec']):.4f}  "
                      f"c_ratio={float(logs['metrics/z_common_norm_ratio']):.2f}  "
                      f"PD_acc={float(logs['metrics/patch_disc_acc']):.2f}")
                if args.wandb and _WANDB:
                    wandb.log({k: float(v) for k, v in logs.items()} | {'epoch': epoch},
                              step=global_step)
            global_step += 1

        # Epoch summary
        avg = {k: float(np.mean([float(log[k]) for log in epoch_logs]))
               for k in epoch_logs[0]}
        print(f"  Summary: loss={avg['loss/total']:.4f}  "
              f"rec={avg['loss/reconstruction']:.4f}  "
              f"kl={avg['loss/kl_total']:.4f}  "
              f"mi={avg['loss/mi_factor']:.4f}  "
              f"supcon={avg['loss/cardio_supcon']:.4f}  "
              f"disc={avg['loss/disc']:.4f}  "
              f"D_acc={avg['metrics/disc_acc']:.2f}  "
              f"bbox={avg['loss/bbox_attn']:.4f}  "
              f"ratio={avg['metrics/z_cardio_norm_ratio']:.2f}  "
              f"gan_g={avg['loss/gan_g']:.4f}  "
              f"tv={avg['loss/tv']:.4f}  "
              f"msk={avg['loss/masked_rec']:.4f}  "
              f"c_ratio={avg['metrics/z_common_norm_ratio']:.2f}  "
              f"PD_acc={avg['metrics/patch_disc_acc']:.2f}")

        # Checkpoint
        ckpt_path = None
        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pkl"
            ckpt_data = {
                'epoch': epoch, 'global_step': global_step,
                'vae_params':           vae_state.params,
                'ema_params':           ema_params,
                'vae_batch_stats':      vae_batch_stats,
                'vae_opt_state':        vae_state.opt_state,
                'rng': rng, 'args': vars(args),
            }
            if use_factor_disc and disc_state is not None:
                ckpt_data['disc_params'] = disc_state.params
                ckpt_data['disc_opt_state'] = disc_state.opt_state
            if use_patch_disc and patch_disc_state is not None:
                ckpt_data['patch_disc_params'] = patch_disc_state.params
                ckpt_data['patch_disc_opt_state'] = patch_disc_state.opt_state
            with open(ckpt_path, 'wb') as f:
                f.write(to_bytes(ckpt_data))
            print(f"  Saved: {ckpt_path}")

        # Visualisations
        if last_batch is not None and args.sample_every > 0 and epoch % args.sample_every == 0:
            rng, vis_key = jax.random.split(rng)
            vis_params = ema_params if args.ema_decay > 0 else vae_state.params

            x_full   = jnp.concatenate([last_batch['x_norm'], last_batch['x_disease1']], axis=0)
            labels_v = last_batch['disease_labels']

            x_rec, attn_maps = reconstruct_and_encode(
                vis_params, x_full, labels_v, vis_key,
                last_batch['bbox_full'], last_batch['has_bbox'],
                heart_mask=last_batch.get('heart_mask'),
            )

            grid_path = samples_dir / f"recon_epoch{epoch:04d}.png"
            make_recon_grid(x_full, x_rec, labels_v,
                            n_per_class=args.n_samples_per_class).save(str(grid_path))
            print(f"  Saved recon grid: {grid_path}")

            attn_path = diag_dir / f"attn_maps_epoch{epoch:04d}.png"
            make_attention_grid(
                np.array(x_full),
                {k: np.array(v) for k, v in attn_maps.items()},
                np.array(labels_v),
                x_rec=np.array(x_rec),
                heart_masks=np.array(last_batch['heart_mask'])
                            if 'heart_mask' in last_batch else None,
                n_per_class=args.n_samples_per_class,
                bboxes_cardio=np.array(last_batch['bbox_disease1']),
            ).save(str(attn_path))
            print(f"  Saved attention maps: {attn_path}")

            if args.wandb and _WANDB:
                wandb.log({
                    "samples/recon_grid":    wandb.Image(str(grid_path)),
                    "diagnostics/attn_maps": wandb.Image(str(attn_path)),
                }, step=global_step)

            # ── Scenario overlay ──────────────────────────────────────────────
            if _DIAG and _make_scenario_overlay is not None:
                try:
                    scenario_img  = _make_scenario_overlay(metrics_history_path, epoch)
                    scenario_path = diag_dir / f"loss_scenarios_epoch{epoch:04d}.png"
                    scenario_img.save(str(scenario_path))
                    if args.wandb and _WANDB:
                        wandb.log({"diagnostics/loss_scenarios": wandb.Image(str(scenario_path))},
                                  step=global_step)
                    print(f"  Saved scenario overlay: {scenario_path}")
                except Exception as _e:
                    print(f"  [warn] scenario overlay failed: {_e}")

        # Manifold
        manifold_metrics = {}
        manifold_path = None
        if args.manifold_every > 0 and epoch % args.manifold_every == 0:
            manifold_path   = manifold_dir / f"manifold_epoch{epoch:04d}.png"
            manifold_params = ema_params if args.ema_decay > 0 else vae_state.params
            manifold_metrics = save_latent_manifold_plot(
                sepvae, manifold_params, vae_batch_stats, eval_loader,
                manifold_path, max_samples=min(args.manifold_max_samples, args.eval_subset_size),
                method=args.manifold_method,
                use_bbox_cross_attn=args.use_bbox_cross_attn,
                bbox_mode=args.manifold_bbox_mode,
            )
            if manifold_metrics:
                print("  Manifold: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in manifold_metrics.items() if np.isfinite(v)
                ))
            if args.wandb and _WANDB:
                payload = {"samples/latent_manifold": wandb.Image(str(manifold_path))}
                payload.update({f"manifold/{k}": float(v)
                                 for k, v in manifold_metrics.items() if np.isfinite(v)})
                wandb.log(payload, step=global_step)

            # ── KL heatmap (2-class) — disabled: mask curriculum ─────────────
            if False:  # disabled: mask curriculum uses ctr_reg instead
                if last_batch is not None:
                    try:
                        kl_img  = _make_kl_heatmap(
                            sepvae, manifold_params,
                            last_batch['x_norm'], last_batch['x_disease1'],
                        )
                        kl_path = diag_dir / f"kl_heatmap_epoch{epoch:04d}.png"
                        kl_img.save(str(kl_path))
                        if args.wandb and _WANDB:
                            wandb.log({"diagnostics/kl_heatmap": wandb.Image(str(kl_path))},
                                      step=global_step)
                        print(f"  Saved KL heatmap: {kl_path}")
                    except Exception as _e:
                        print(f"  [warn] KL heatmap failed: {_e}")

            # ── Counterfactual acid test — disabled: mask curriculum ──────────
            if False:  # disabled: no bbox injection in mask curriculum
                if _DIAG and _run_counterfactual_eval is not None:
                    try:
                        _run_counterfactual_eval(
                            sepvae, manifold_params, vae_batch_stats,
                            eval_loader,
                            output_dir=diag_dir,
                            epoch=epoch,
                            global_step=global_step,
                            n_samples=min(8, args.eval_subset_size),
                            use_wandb=(args.wandb and _WANDB),
                        )
                    except Exception as _e:
                        print(f"  [warn] counterfactual eval failed: {_e}")

            # ── 8-panel research claim scaffolding ───────────────────────────
            _diag_kwargs = dict(
                epoch=epoch,
                global_step=global_step,
                output_dir=diag_dir,
                use_wandb=(args.wandb and _WANDB),
                use_bbox_cross_attn=args.use_bbox_cross_attn,
                has_chexmask=getattr(args, 'chexmask_csv', None) is not None,
                heart_in_zd=getattr(args, 'heart_in_zd', False),
            )
            if _DIAG and _run_scaffolding is not None:
                try:
                    _run_scaffolding(
                        sepvae, manifold_params, vae_batch_stats,
                        eval_loader,
                        **_diag_kwargs,
                    )
                except Exception as _e:
                    print(f"  [warn] scaffolding failed: {_e}")

            # ── z_d traversal strip ──────────────────────────────────────────
            if _DIAG and _run_traversal is not None:
                try:
                    _run_traversal(
                        sepvae, manifold_params, vae_batch_stats,
                        eval_loader,
                        **_diag_kwargs,
                    )
                except Exception as _e:
                    print(f"  [warn] traversal strip failed: {_e}")

            # ── Swapped-reconstruction composition grid ──────────────────────
            if _DIAG and _run_composition is not None:
                try:
                    _run_composition(
                        sepvae, manifold_params, vae_batch_stats,
                        eval_loader,
                        **_diag_kwargs,
                    )
                except Exception as _e:
                    print(f"  [warn] composition grid failed: {_e}")

        epoch_record = {'epoch': epoch, 'global_step': global_step}
        epoch_record.update({k: float(v) for k, v in avg.items()})
        epoch_record.update({k: float(v) for k, v in manifold_metrics.items() if np.isfinite(v)})
        epoch_record['checkpoint_path'] = str(ckpt_path) if ckpt_path is not None else None
        epoch_record['manifold_path'] = str(manifold_path) if manifold_path is not None else None
        with open(metrics_history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(epoch_record) + "\n")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    final_path = ckpt_dir / "checkpoint_final.pkl"
    final_data = {
        'epoch': args.epochs, 'global_step': global_step,
        'vae_params':           vae_state.params,
        'ema_params':           ema_params,
        'vae_batch_stats':      vae_batch_stats,
        'vae_opt_state':        vae_state.opt_state,
        'rng': rng, 'args': vars(args),
    }
    if use_factor_disc and disc_state is not None:
        final_data['disc_params'] = disc_state.params
        final_data['disc_opt_state'] = disc_state.opt_state
    if use_patch_disc and patch_disc_state is not None:
        final_data['patch_disc_params'] = patch_disc_state.params
        final_data['patch_disc_opt_state'] = patch_disc_state.opt_state
    with open(final_path, 'wb') as f:
        f.write(to_bytes(final_data))
    print(f"Final checkpoint: {final_path}")

    if args.wandb and _WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
