"""
Training script for Multi-head SepVAE with frozen CheSS backbone.

This script implements the full training loop for SepVAE with:
- Dual optimizer setup (VAE + PatchGAN discriminator)
- CheSS weight loading and injection
- VinBigData triplet dataset
- W&B logging
- Checkpoint saving/loading
- Anti-aliased upsampling (bilinear/subpixel) to prevent checkerboard artifacts

GPU Memory & Batch Size Recommendations:
========================================
| GPU               | VRAM   | Recommended batch_size | With gradient_checkpointing |
|-------------------|--------|------------------------|------------------------------|
| RTX 3090          | 24 GB  | 4-6                    | 8                            |
| Quadro RTX 8000   | 48 GB  | 10-12                  | 16                           |
| A100              | 80 GB  | 14-16                  | 24+                          |

Note: These assume 512×512 images with FPN enabled and perceptual+adversarial losses.
For bf16 mixed precision (--half_precision bf16), add ~20% to batch sizes.

Usage Examples:
===============

RTX 3090 (24GB):
    python -m run.train_sep_vae --batch_size 6 --gradient_checkpointing --upsample_method bilinear

Quadro RTX 8000 (48GB):
    python -m run.train_sep_vae --batch_size 12 --half_precision bf16 --upsample_method bilinear

A100 (80GB) - Recommended for research:
    python -m run.train_sep_vae \\
        --batch_size 16 \\
        --half_precision bf16 \\
        --gradient_checkpointing \\
        --upsample_method bilinear \\
        --use_fpn \\
        --weight_perceptual 0.1 \\
        --weight_adversarial 0.1

Anti-Aliasing:
==============
The --upsample_method flag controls decoder upsampling:
- 'bilinear': Smooth interpolation (recommended, prevents checkerboard)
- 'subpixel': Learned pixel-shuffle (best quality, ~10% more memory)
- 'nearest': Legacy (causes checkerboard artifacts - avoid)
"""

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes, msgpack_restore, from_state_dict
import torch
from torch.utils.data import DataLoader

# Local imports
from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
from models.sep_vae_jax import SepVAE
from models.resnet_jax import ResNet50CheSS
from losses.sep_vae_losses import (
    DiseaseClassifier,
    SepVAELossConfig, sepvae_loss,
)
from losses.lpips_gan import NLayerDiscriminator
from utils.weight_converter import convert_chess_resnet50, load_converted_weights
from utils.sepvae_analysis import visualize_backbone_features
from utils.sepvae_diagnostics import (
    plot_latent_swap_grid,
    plot_per_channel_kl_heatmap,
    plot_head_norm_by_class,
)

# Optional W&B
try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("Multi-head SepVAE with CheSS backbone trainer")

    # Data
    p.add_argument("--dicom_dir", type=str, default="/datasets/mmolefe/vinbigdata/train",
                   help="Path to DICOM files")
    p.add_argument("--csv_path", type=str, default="/datasets/mmolefe/vinbigdata/train.csv",
                   help="Path to train.csv")
    p.add_argument("--img_size", type=int, default=512,
                   help="Image size (default: 512)")
    p.add_argument("--exclude_cross_disease_overlap", action="store_true",
                   help="Exclude samples where effusion and cardiomegaly co-occur "
                        "from disease pools to improve disentanglement purity")

    # Model architecture (spatial latents: 64×64×channels)
    p.add_argument("--z_channels_common", type=int, default=4,
                   help="Common latent channels (default: 4 for 64×64×4)")
    p.add_argument("--z_channels_disease", type=int, default=2,
                   help="Disease-specific latent channels (default: 2 for 64×64×2 each)")
    p.add_argument("--frozen_backbone", action="store_true", default=True,
                   help="Freeze CheSS backbone")
    p.add_argument("--use_fpn", action="store_true", default=False,
                   help="Enable Feature Pyramid Network for multi-scale features")
    p.add_argument("--fpn_channels", type=int, default=512,
                   help="FPN output channels (default: 512)")
    p.add_argument("--unfreeze_from", type=str, default='layer4', choices=[None, 'layer3', 'layer4'],
                   help="Partially unfreeze backbone from this layer (default: layer4)")
    p.add_argument("--use_label_attention", action="store_true", default=False,
                   help="Replace disease ConvHeads with DiseaseAttentionHead: a learned "
                        "disease prototype query attends over backbone features, yielding "
                        "explicit (64x64) spatial attention maps per disease head.")
    p.add_argument("--attn_query_dim", type=int, default=256,
                   help="Disease prototype query dimension for --use_label_attention (default: 256)")

    # R12: Spatial attention bbox supervision
    p.add_argument("--weight_spatial_attn", type=float, default=0.0,
                   help="Spatial attention bbox supervision weight (default: 0.0 = disabled). "
                        "Requires --use_label_attention. Penalises DiseaseAttentionHead for "
                        "attending outside the ground-truth VinBigData bbox.")

    # CheSS weights
    p.add_argument("--chess_checkpoint", type=str,
                   default="/datasets/mmolefe/chess/pretrained_weights.pth.tar",
                   help="Path to CheSS PyTorch checkpoint")
    p.add_argument("--chess_converted", type=str, default=None,
                   help="Path to converted JAX weights (.npy). If None, will convert on-the-fly")

    # Loss weights
    p.add_argument("--weight_rec", type=float, default=1.0,
                   help="Reconstruction loss weight")
    p.add_argument("--weight_kl_common", type=float, default=1e-4,
                   help="Common KL weight")
    p.add_argument("--weight_kl_disease", type=float, default=1e-4,
                   help="Disease KL weight")
    p.add_argument("--weight_null", type=float, default=1e-2,
                   help="Nulling loss weight")
    p.add_argument("--weight_orthogonality", type=float, default=3e-2,
                   help="Disease-head orthogonality loss weight (default: 3e-2)")
    p.add_argument("--sigma_inactive", type=float, default=0.1,
                   help="Prior std dev for inactive heads (default: 0.1 for tighter inactivity prior)")
    p.add_argument("--kl_warmup_epochs", type=int, default=0,
                   help="Linearly anneal KL weight from 0 to 1 over this many epochs (0=no warmup)")
    p.add_argument("--free_bits", type=float, default=0.0,
                   help="Per-channel free-bits for KL (0=disabled, try 1.0)")
    p.add_argument("--weight_perceptual", type=float, default=0.03,
                   help="Backbone perceptual loss weight (default: 0.03)")
    p.add_argument("--weight_adversarial", type=float, default=0.0,
                   help="PatchGAN generator loss weight (0=disabled, try 0.1)")
    p.add_argument("--disc_start_epoch", type=int, default=10,
                   help="Epoch at which PatchGAN discriminator loss activates")

    # R5a: Paired contrastive loss
    p.add_argument("--use_contrastive", action="store_true", default=False,
                   help="Enable paired contrastive loss for structural head specialisation (R5a)")
    p.add_argument("--weight_contrastive", type=float, default=0.1,
                   help="Paired contrastive loss weight (default: 0.1)")
    p.add_argument("--contrastive_margin", type=float, default=0.5,
                   help="Cosine margin for contrastive separate-push (default: 0.5)")


    # R11: Disease discriminability classifier
    p.add_argument("--use_disease_clf", action="store_true", default=False,
                   help="Enable auxiliary disease-discriminability classifiers (R11): "
                        "BCE loss on z_disease_k.mu → label_k to prevent posterior collapse")
    p.add_argument("--weight_disease_clf", type=float, default=0.1,
                   help="Disease discriminability classification loss weight (default: 0.1)")
    p.add_argument("--lr_disease_clf", type=float, default=1e-4,
                   help="Disease classifier learning rate (default: 1e-4)")

    # R3: LR cosine decay
    p.add_argument("--lr_decay_epochs", type=int, default=0,
                   help="Epoch at which cosine LR decay begins (0=disabled). "
                        "Decays to 10%% of init LR by final epoch.")

    # Performance & Anti-Aliasing
    p.add_argument("--half_precision", type=str, default="none",
                   choices=["none", "bf16", "fp16"],
                   help="Half precision dtype for compute (none=fp32, bf16=bfloat16, fp16=float16)")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="Enable gradient checkpointing (remat) to reduce memory at cost of ~30%% more compute")
    p.add_argument("--upsample_method", type=str, default="bilinear",
                   choices=["bilinear", "subpixel", "nearest"],
                   help="Decoder upsampling method (default: bilinear). "
                        "'bilinear': Smooth interpolation - prevents checkerboard artifacts (recommended). "
                        "'subpixel': Learned pixel-shuffle - best quality but more compute. "
                        "'nearest': Legacy nearest-neighbor - causes checkerboard artifacts (not recommended).")

    # Optimizer
    p.add_argument("--lr_vae", type=float, default=1e-4,
                   help="VAE learning rate")
    p.add_argument("--lr_backbone", type=float, default=1e-5,
                   help="Learning rate for unfrozen backbone layers (10x lower than VAE)")
    p.add_argument("--lr_patch_disc", type=float, default=4e-4,
                   help="PatchGAN discriminator learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping norm")

    # Training
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size (triplets, results in 3*batch_size images)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of epochs")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")

    # Logging & checkpoints
    p.add_argument("--output_root", type=str, default="runs_sepvae",
                   help="Output directory root")
    p.add_argument("--exp_name", type=str, default="sepvae_chess",
                   help="Experiment name")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log every N steps")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--sample_every", type=int, default=5,
                   help="Generate reconstruction grid every N epochs (0 to disable)")
    p.add_argument("--n_samples_per_class", type=int, default=4,
                   help="Number of samples per class in reconstruction grid")
    p.add_argument("--manifold_every", type=int, default=-1,
                   help="Generate latent manifold plot every N epochs "
                        "(-1=match sample_every, 0=disable)")
    p.add_argument("--manifold_max_samples", type=int, default=600,
                   help="Max samples used for latent manifold visualization")
    p.add_argument("--manifold_method", type=str, default="pca",
                   choices=["pca", "tsne", "both"],
                   help="Dimensionality reduction method for manifold plots")
    p.add_argument("--disent_metrics_every", type=int, default=-1,
                   help="Evaluate disentanglement metrics every N epochs "
                        "(-1=match manifold_every, 0=disable)")
    p.add_argument("--disent_metrics_max_samples", type=int, default=600,
                   help="Max triplet-derived samples for disentanglement metrics")
    p.add_argument("--overlap_eval_samples", type=int, default=128,
                   help="Number of real overlap images used for synthetic-X proximity metrics")
    p.add_argument("--overlap_eval_batch_size", type=int, default=16,
                   help="Batch size for encoding real overlap images")
    p.add_argument("--disent_eval_on_sample", action="store_true",
                   help="Also run disentanglement evaluation whenever reconstruction samples are generated")

    # Verbose diagnostics
    p.add_argument("--verbose_backbone", action="store_true",
                   help="Visualize frozen backbone features before training (PCA, class diffs, top-K channels)")
    p.add_argument("--verbose_n_samples", type=int, default=12,
                   help="Number of samples for backbone visualization (default: 12)")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")

    # W&B
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="sepvae-chess",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (username/team)")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="W&B run ID to resume logging into (e.g. 'ezobur5y')")

    return p.parse_args()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def make_sepvae_recon_grid(x_input, x_rec, labels, n_per_class=4):
    """
    Create a reconstruction grid organized by disease type.

    Layout:
        Row 0 (Original):       Normal ... | Effusion ... | Cardiomegaly ...
        Row 1 (Reconstruction): Normal ... | Effusion ... | Cardiomegaly ...

    Args:
        x_input: Original images (3B, H, W, 1) in [-1, 1]
        x_rec: Reconstructed images (3B, H, W, 1) in [0, 1]
        labels: Disease labels (3B,)
        n_per_class: Number of samples to show per class

    Returns:
        PIL.Image: The grid image
    """
    from torchvision.utils import make_grid
    from PIL import Image

    x_input_01 = (np.array(x_input) + 1.0) / 2.0  # [-1,1] -> [0,1]
    x_rec_np = np.clip(np.array(x_rec), 0.0, 1.0)
    labels_np = np.array(labels)

    originals = []
    reconstructions = []

    for cls_id in [0, 1, 2]:
        mask = labels_np == cls_id
        idxs = np.where(mask)[0][:n_per_class]
        for idx in idxs:
            originals.append(x_input_01[idx])
            reconstructions.append(x_rec_np[idx])

    all_imgs = originals + reconstructions
    nrow = len(originals)

    # NHWC -> NCHW for torchvision
    imgs_np = np.stack(all_imgs, axis=0)
    imgs_np = np.transpose(imgs_np, (0, 3, 1, 2))
    imgs_torch = torch.tensor(imgs_np).clamp(0, 1)

    grid = make_grid(imgs_torch, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_img = Image.fromarray((grid_np * 255).astype(np.uint8))

    return grid_img


def make_attention_map_grid(x_input, attn_maps, labels, n_per_class=4):
    """
    Create a figure showing per-disease attention maps overlaid on CXR images.

    Layout (3 rows × n_per_class columns):
        Each cell shows three sub-panels side-by-side:
            [Original CXR]  [Cardio attention]  [Effusion attention]
        Rows correspond to Normal / Effusion / Cardiomegaly classes.

    The attention maps are at 64×64 resolution and are bilinearly upscaled to
    match the image for display.  Cardio maps use the 'hot' colormap; effusion
    maps use 'cool'.  Both are shown as a 50/50 alpha blend over the greyscale CXR.

    Args:
        x_input:   Original images (N, H, W, 1) in [-1, 1]
        attn_maps: Dict with 'cardiomegaly' and 'effusion' (N, h, w) float arrays
        labels:    Disease labels (N,) in {0, 1, 2}
        n_per_class: Number of samples per class row

    Returns:
        PIL.Image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    x_01 = (np.array(x_input) + 1.0) / 2.0          # (N, H, W, 1) in [0,1]
    attn_cardio  = np.array(attn_maps['cardiomegaly'])  # (N, h, w)
    attn_effusion = np.array(attn_maps['effusion'])     # (N, h, w)
    labels_np = np.array(labels)

    class_names = ['Normal', 'Effusion', 'Cardiomegaly']
    n_subcols = 3  # original + cardio-attn + effusion-attn per sample
    n_cols = n_per_class * n_subcols
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_per_class * n_subcols * 1.8, n_rows * 2.2))
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for row, cls_id in enumerate([0, 1, 2]):
        mask = labels_np == cls_id
        idxs = np.where(mask)[0][:n_per_class]

        for col_pos, idx in enumerate(idxs):
            img = x_01[idx, :, :, 0]  # (H, W)
            a_c = attn_cardio[idx]     # (h, w) — 64×64
            a_e = attn_effusion[idx]   # (h, w)

            base_col = col_pos * n_subcols

            # --- Original ---
            ax = axes[row, base_col]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.axis('off')
            if col_pos == 0:
                ax.set_ylabel(class_names[cls_id], fontsize=8, rotation=90, labelpad=4)
            if row == 0 and col_pos == 0:
                ax.set_title('CXR', fontsize=7)

            # --- Cardiomegaly attention ---
            ax = axes[row, base_col + 1]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.imshow(a_c, cmap='hot', alpha=0.5,
                      extent=(0, img.shape[1], img.shape[0], 0),
                      interpolation='bilinear', vmin=0)
            ax.axis('off')
            if row == 0 and col_pos == 0:
                ax.set_title('Cardio attn', fontsize=7)

            # --- Effusion attention ---
            ax = axes[row, base_col + 2]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.imshow(a_e, cmap='cool', alpha=0.5,
                      extent=(0, img.shape[1], img.shape[0], 0),
                      interpolation='bilinear', vmin=0)
            ax.axis('off')
            if row == 0 and col_pos == 0:
                ax.set_title('Effusion attn', fontsize=7)

        # Fill any empty cells (if class has fewer than n_per_class samples)
        for col_pos in range(len(idxs), n_per_class):
            for sub in range(n_subcols):
                axes[row, col_pos * n_subcols + sub].axis('off')

    plt.tight_layout(pad=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_bbox_attn_comparison_grid(
    x_input, x_rec, attn_maps, bboxes_cardio, bboxes_effusion, B, n_per_class=4
):
    """
    Comparison grid: GT bbox | Reconstruction | Disease attention map.
    Organised as two sections (Cardiomegaly top, Pleural Effusion bottom).

    Expects:
        x_input:         (3B, H, W, 1) in [-1, 1] — concat of [normal, effusion, cardio]
        x_rec:           (3B, H, W, 1) in [0, 1]
        attn_maps:       dict 'cardiomegaly' and 'effusion' each (3B, h, w)
        bboxes_cardio:   (B, 4) [x0, y0, x1, y1] normalised [0,1] for x_disease2 (cardio)
        bboxes_effusion: (B, 4) [x0, y0, x1, y1] normalised [0,1] for x_disease1 (effusion)
        B:               batch size per class
        n_per_class:     samples to display per section

    Layout  (2 sections × 3 rows × n_per_class cols):
        ── CARDIOMEGALY ──────────────────────────────────
        row 0: CXR + GT bbox (red)
        row 1: Reconstruction
        row 2: Cardio attention map (hot)
        ── PLEURAL EFFUSION ──────────────────────────────
        row 3: CXR + GT bbox (cyan)
        row 4: Reconstruction
        row 5: Effusion attention map (cool)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO
    from PIL import Image

    x_01 = (np.array(x_input) + 1.0) / 2.0          # (3B, H, W, 1) → [0,1]
    x_rec_np = np.clip(np.array(x_rec), 0.0, 1.0)
    attn_c = np.array(attn_maps['cardiomegaly'])      # (3B, h, w)
    attn_e = np.array(attn_maps['effusion'])           # (3B, h, w)
    bboxes_c = np.array(bboxes_cardio)[:n_per_class]   # (n, 4) normalised
    bboxes_e = np.array(bboxes_effusion)[:n_per_class] # (n, 4) normalised

    # Index mapping: normal=[0..B-1], effusion=[B..2B-1], cardio=[2B..3B-1]
    cardio_idxs   = np.arange(2 * B, 2 * B + min(n_per_class, B))
    effusion_idxs = np.arange(B, B + min(n_per_class, B))

    n_cols = n_per_class
    n_rows = 6  # 3 rows per disease section
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    section_labels = {
        0: 'GT bbox',
        1: 'Recon',
        2: 'Cardio\nattn',
        3: 'GT bbox',
        4: 'Recon',
        5: 'Effusion\nattn',
    }
    disease_section_titles = {0: 'CARDIOMEGALY', 3: 'PLEURAL EFFUSION'}
    bbox_colors = {0: 'red', 3: 'cyan'}

    H, W = x_01.shape[1], x_01.shape[2]

    for col, (c_idx, e_idx) in enumerate(zip(cardio_idxs, effusion_idxs)):
        n_c = min(n_per_class, len(bboxes_c))
        n_e = min(n_per_class, len(bboxes_e))

        for row in range(n_rows):
            ax = axes[row, col]

            if row < 3:
                # ── Cardiomegaly section ──────────────────────────────────
                img = x_01[c_idx, :, :, 0]
                rec = x_rec_np[c_idx, :, :, 0]
                attn = attn_c[c_idx]
                bbox = bboxes_c[col] if col < n_c else None
                bbox_color = 'red'
            else:
                # ── Effusion section ──────────────────────────────────────
                img = x_01[e_idx, :, :, 0]
                rec = x_rec_np[e_idx, :, :, 0]
                attn = attn_e[e_idx]
                bbox = bboxes_e[col] if col < n_e else None
                bbox_color = 'cyan'

            local_row = row % 3

            if local_row == 0:
                # CXR + ground-truth bbox
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                if bbox is not None and not np.all(bbox == 0):
                    x0, y0, x1, y1 = bbox
                    rect = mpatches.Rectangle(
                        (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                        linewidth=1.5, edgecolor=bbox_color, facecolor='none'
                    )
                    ax.add_patch(rect)
            elif local_row == 1:
                # Reconstruction
                ax.imshow(rec, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            else:
                # Attention map overlay
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                cmap = 'hot' if row < 3 else 'cool'
                ax.imshow(
                    attn, cmap=cmap, alpha=0.5,
                    extent=(0, W, H, 0), interpolation='bilinear', vmin=0,
                )

            ax.axis('off')

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(section_labels[row], fontsize=7, rotation=90, labelpad=4)

            # Disease section title above first row of each section (first column only)
            if col == 0 and row in disease_section_titles:
                ax.set_title(disease_section_titles[row], fontsize=8, fontweight='bold',
                             color='white', backgroundcolor='#1a1a2e', pad=3)

    # Add a thin horizontal separator between the two sections
    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [0.5, 0.5], transform=fig.transFigure,
        color='white', linewidth=0.8, linestyle='--', alpha=0.5,
    ))

    fig.suptitle('GT bbox | Reconstruction | Attention map', fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3, h_pad=0.5)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_bbox_comparison_grid(
    x_input, x_rec, attn_maps, bbox_disease1, bbox_disease2, labels, n_per_class=4
):
    """
    GT bbox | Reconstruction | Attention map comparison grid.

    Uses disease labels to find cardiomegaly (label==2) and effusion (label==1) samples.
    Layout: 6 rows (3 per disease) × n_per_class cols.
        Cardiomegaly section: CXR+GT bbox | Recon | Cardio attn
        Effusion section:     CXR+GT bbox | Recon | Effusion attn

    Args:
        x_input:       (N, H, W, 1) in [-1, 1]
        x_rec:         (N, H, W, 1) in [0, 1]
        attn_maps:     dict with 'cardiomegaly' and 'effusion' each (N, h, w)
        bbox_disease1: (M, 4) normalised [x0,y0,x1,y1] for effusion
        bbox_disease2: (M, 4) normalised [x0,y0,x1,y1] for cardiomegaly
        labels:        (N,) disease labels {0,1,2}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO
    from PIL import Image

    x_01 = (np.array(x_input) + 1.0) / 2.0
    x_rec_np = np.clip(np.array(x_rec), 0.0, 1.0)
    attn_c = np.array(attn_maps['cardiomegaly'])
    attn_e = np.array(attn_maps['effusion'])
    labels_np = np.array(labels)
    bboxes_c = np.array(bbox_disease2)
    bboxes_e = np.array(bbox_disease1)

    cardio_idxs   = np.where(labels_np == 2)[0][:n_per_class]
    effusion_idxs = np.where(labels_np == 1)[0][:n_per_class]
    n_cols = n_per_class
    n_rows = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    section_labels = {0: 'GT bbox', 1: 'Recon', 2: 'Cardio\nattn',
                      3: 'GT bbox', 4: 'Recon', 5: 'Effusion\nattn'}
    disease_section_titles = {0: 'CARDIOMEGALY', 3: 'PLEURAL EFFUSION'}
    H, W = x_01.shape[1], x_01.shape[2]

    for col in range(n_cols):
        for row in range(n_rows):
            ax = axes[row, col]
            if row < 3:
                idxs, bboxes, bbox_color = cardio_idxs, bboxes_c, 'red'
                attn = attn_c
                cmap = 'hot'
            else:
                idxs, bboxes, bbox_color = effusion_idxs, bboxes_e, 'cyan'
                attn = attn_e
                cmap = 'cool'

            if col >= len(idxs):
                ax.axis('off')
                continue

            idx = idxs[col]
            img = x_01[idx, :, :, 0]
            rec = x_rec_np[idx, :, :, 0]
            local_row = row % 3

            if local_row == 0:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                if col < len(bboxes):
                    bbox = bboxes[col]
                    if not np.all(bbox == 0):
                        x0, y0, x1, y1 = bbox
                        rect = mpatches.Rectangle(
                            (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                            linewidth=1.5, edgecolor=bbox_color, facecolor='none')
                        ax.add_patch(rect)
            elif local_row == 1:
                ax.imshow(rec, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            else:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                ax.imshow(attn[idx], cmap=cmap, alpha=0.5,
                          extent=(0, W, H, 0), interpolation='bilinear', vmin=0)

            ax.axis('off')
            if col == 0:
                ax.set_ylabel(section_labels[row], fontsize=7, rotation=90, labelpad=4)
            if col == 0 and row in disease_section_titles:
                ax.set_title(disease_section_titles[row], fontsize=8, fontweight='bold',
                             color='white', backgroundcolor='#1a1a2e', pad=3)

    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [0.5, 0.5], transform=fig.transFigure,
        color='white', linewidth=0.8, linestyle='--', alpha=0.5,
    ))
    fig.suptitle('GT bbox | Reconstruction | Attention map', fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3, h_pad=0.5)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_heatmap_comparison_grid(
    x_input, x_rec, attn_maps, bbox_disease1, bbox_disease2, labels, n_per_class=4
):
    """
    Side-by-side heatmap vs GT bbox comparison grid.

    For each disease (cardiomegaly, effusion) shows:
        CXR | CXR + GT bbox | CXR + attention heatmap

    Args: same as make_bbox_comparison_grid
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO
    from PIL import Image

    x_01 = (np.array(x_input) + 1.0) / 2.0
    attn_c = np.array(attn_maps['cardiomegaly'])
    attn_e = np.array(attn_maps['effusion'])
    labels_np = np.array(labels)
    bboxes_c = np.array(bbox_disease2)
    bboxes_e = np.array(bbox_disease1)

    cardio_idxs   = np.where(labels_np == 2)[0][:n_per_class]
    effusion_idxs = np.where(labels_np == 1)[0][:n_per_class]

    n_subcols = 3  # CXR | GT bbox | heatmap
    n_cols = n_per_class * n_subcols
    n_rows = 2  # cardiomegaly row, effusion row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_per_class * n_subcols * 1.8, n_rows * 2.4))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    H, W = x_01.shape[1], x_01.shape[2]
    row_cfg = [
        (cardio_idxs, bboxes_c, attn_c, 'hot', 'red',  'CARDIOMEGALY'),
        (effusion_idxs, bboxes_e, attn_e, 'cool', 'cyan', 'PLEURAL EFFUSION'),
    ]

    for row, (idxs, bboxes, attn, cmap, bbox_color, title) in enumerate(row_cfg):
        for col_pos in range(n_per_class):
            base_col = col_pos * n_subcols
            if col_pos >= len(idxs):
                for sub in range(n_subcols):
                    axes[row, base_col + sub].axis('off')
                continue

            idx = idxs[col_pos]
            img = x_01[idx, :, :, 0]
            bbox = bboxes[col_pos] if col_pos < len(bboxes) else None

            # CXR
            ax = axes[row, base_col]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.axis('off')
            if col_pos == 0:
                ax.set_ylabel(title, fontsize=8, rotation=90, labelpad=4)
            if row == 0 and col_pos == 0:
                ax.set_title('CXR', fontsize=7)

            # CXR + GT bbox
            ax = axes[row, base_col + 1]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            if bbox is not None and not np.all(bbox == 0):
                x0, y0, x1, y1 = bbox
                rect = mpatches.Rectangle(
                    (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                    linewidth=1.5, edgecolor=bbox_color, facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            if row == 0 and col_pos == 0:
                ax.set_title('GT bbox', fontsize=7)

            # CXR + attention heatmap
            ax = axes[row, base_col + 2]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.imshow(attn[idx], cmap=cmap, alpha=0.5,
                      extent=(0, W, H, 0), interpolation='bilinear', vmin=0)
            ax.axis('off')
            if row == 0 and col_pos == 0:
                ax.set_title('Attn map', fontsize=7)

    plt.tight_layout(pad=0.3)
    buf = BytesIO()
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
    max_samples: int = 600,
    method: str = "pca",
):
    """
    Generate and save latent manifold visualization from current model state.

    The figure includes:
    - All-heads view: [common + cardiomegaly + effusion]
    - Disease-only view: [cardiomegaly + effusion]

    Args:
        model: SepVAE model
        vae_params: Current VAE parameters
        vae_batch_stats: Frozen backbone BN statistics
        loader: Triplet DataLoader
        save_path: Output image path
        max_samples: Max number of images across all classes
        method: 'pca', 'tsne', or 'both'

    Returns:
        Dict with silhouette metrics for each view/method combination
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    latents_common = []
    latents_cardio = []
    latents_effusion = []
    labels_all = []

    for batch_torch in loader:
        if len(labels_all) >= max_samples:
            break

        for img_key, label_val in [('x_norm', 0), ('x_disease1', 1), ('x_disease2', 2)]:
            x = jnp.array(batch_torch[img_key].permute(0, 2, 3, 1).numpy())
            latents_dict = model.apply(variables, x, method=model.encode)

            mu_c, _ = latents_dict['common']
            mu_cardio, _ = latents_dict['cardiomegaly']
            mu_effusion, _ = latents_dict['effusion']

            z_common_np = np.array(jnp.mean(mu_c, axis=(1, 2)))
            z_cardio_np = np.array(jnp.mean(mu_cardio, axis=(1, 2)))
            z_effusion_np = np.array(jnp.mean(mu_effusion, axis=(1, 2)))

            remaining = max_samples - len(labels_all)
            if remaining <= 0:
                break

            z_common_np = z_common_np[:remaining]
            z_cardio_np = z_cardio_np[:remaining]
            z_effusion_np = z_effusion_np[:remaining]

            latents_common.extend(z_common_np)
            latents_cardio.extend(z_cardio_np)
            latents_effusion.extend(z_effusion_np)
            labels_all.extend([label_val] * len(z_common_np))

            if len(labels_all) >= max_samples:
                break

    latents_common = np.array(latents_common)
    latents_cardio = np.array(latents_cardio)
    latents_effusion = np.array(latents_effusion)
    labels_all = np.array(labels_all)

    if len(labels_all) < 3:
        return {}

    feature_sets = {
        'all_heads': np.concatenate([latents_common, latents_cardio, latents_effusion], axis=1),
        'disease_only': np.concatenate([latents_cardio, latents_effusion], axis=1),
    }
    set_titles = {
        'all_heads': 'All Heads',
        'disease_only': 'Disease Only (Cardio + Effusion)',
    }

    methods = ['pca', 'tsne'] if method == 'both' else [method]
    metrics = {}

    # ── Layout: scatter cols + 1 orthogonality column ──────────────────────
    # Extra right column: prototype-vector scatter (top) + cross-corr heatmap (bottom)
    n_rows = len(feature_sets)   # 2
    n_scatter_cols = len(methods)
    n_cols_total = n_scatter_cols + 1

    fig, axes = plt.subplots(
        n_rows, n_cols_total,
        figsize=(8 * n_cols_total, 5 * n_rows),
        squeeze=False,
    )

    colors = ['blue', 'red', 'green']
    names  = ['Normal', 'Effusion', 'Cardiomegaly']

    # Store the disease-only PCA 2D projection for the orthogonality column
    _disease_pca_2d = None
    _disease_labels_2d = None

    for row_idx, (set_name, features) in enumerate(feature_sets.items()):
        for col_idx, m in enumerate(methods):
            ax = axes[row_idx][col_idx]

            if set_name == 'disease_only':
                keep_mask = labels_all != 0
                features_view = features[keep_mask]
                labels_view = labels_all[keep_mask]
                class_ids = [1, 2]
            else:
                features_view = features
                labels_view = labels_all
                class_ids = [0, 1, 2]

            if m == 'tsne':
                perp = min(30, max(5, len(features_view) // 4))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
            else:
                reducer = PCA(n_components=2, random_state=42)

            z_2d = reducer.fit_transform(features_view)
            for disease_id in class_ids:
                mask = labels_view == disease_id
                ax.scatter(
                    z_2d[mask, 0], z_2d[mask, 1],
                    c=colors[disease_id], label=names[disease_id],
                    alpha=0.6, s=20
                )

            try:
                sil = float(silhouette_score(z_2d, labels_view))
            except ValueError:
                sil = float('nan')
            metrics[f'silhouette_{set_name}_{m}'] = sil

            sil_str = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
            ax.set_title(f"{set_titles[set_name]} ({m.upper()})\nSilhouette: {sil_str}")
            ax.set_xlabel(f"{m.upper()} Component 1")
            ax.set_ylabel(f"{m.upper()} Component 2")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Cache the first PCA projection of disease-only features
            if set_name == 'disease_only' and m == 'pca' and _disease_pca_2d is None:
                _disease_pca_2d = z_2d
                _disease_labels_2d = labels_view

    # ── Orthogonality column ──────────────────────────────────────────────────
    orth_col = n_scatter_cols

    # Row 0: Prototype-vector scatter — disease-only PCA with prototype arrows
    ax_proto = axes[0][orth_col]
    if _disease_pca_2d is not None:
        z2 = _disease_pca_2d
        lv = _disease_labels_2d
        for disease_id, col, nm in zip([1, 2], ['red', 'green'], ['Effusion', 'Cardiomegaly']):
            mask = lv == disease_id
            ax_proto.scatter(z2[mask, 0], z2[mask, 1], c=col, alpha=0.35, s=15, label=nm)

        # Compute class prototype vectors in 2D PCA space
        proto_eff   = z2[lv == 1].mean(axis=0)
        proto_cardio = z2[lv == 2].mean(axis=0)

        # Draw prototype arrows from origin
        ax_proto.annotate('', xy=proto_cardio, xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
        ax_proto.annotate('', xy=proto_eff, xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

        # Compute angle between prototype vectors
        dot = float(np.dot(proto_cardio, proto_eff))
        norms = float(np.linalg.norm(proto_cardio) * np.linalg.norm(proto_eff))
        cos_theta = np.clip(dot / (norms + 1e-8), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        metrics['proto_angle_deg'] = angle_deg

        # Annotate angle (arc + text)
        ax_proto.scatter(0, 0, c='white', s=60, zorder=5)
        ax_proto.text(
            (proto_cardio[0] + proto_eff[0]) / 4,
            (proto_cardio[1] + proto_eff[1]) / 4,
            f'{angle_deg:.1f}°',
            color='yellow', fontsize=13, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6),
        )

        # Label prototype tips
        ax_proto.text(proto_cardio[0], proto_cardio[1], ' Cardio\nprototype',
                      color='green', fontsize=8, va='bottom')
        ax_proto.text(proto_eff[0], proto_eff[1], ' Effusion\nprototype',
                      color='red', fontsize=8, va='top')

        # Ideal angle reference
        title_col = 'green' if angle_deg >= 60 else ('orange' if angle_deg >= 30 else 'red')
        ax_proto.set_title(
            f'Prototype vectors (PCA)\nAngle = {angle_deg:.1f}°  '
            f'{"✓ orthogonal" if angle_deg >= 70 else "↑ approaching" if angle_deg >= 45 else "✗ correlated"}',
            color=title_col, fontsize=10,
        )
        ax_proto.axhline(0, color='grey', lw=0.4, alpha=0.4)
        ax_proto.axvline(0, color='grey', lw=0.4, alpha=0.4)
        ax_proto.legend(fontsize=8)
        ax_proto.grid(True, alpha=0.2)
        ax_proto.set_xlabel('PCA 1')
        ax_proto.set_ylabel('PCA 2')
    else:
        ax_proto.axis('off')

    # Row 1: Cross-correlation block heatmap
    ax_corr = axes[1][orth_col]
    if len(latents_cardio) > 0 and len(latents_effusion) > 0:
        # [z_c1, z_c2] vs [z_e1, z_e2] — only disease-positive samples
        keep = labels_all != 0
        zc_dis = latents_cardio[keep]       # (N_disease, C_cardio)
        ze_dis = latents_effusion[keep]     # (N_disease, C_effusion)

        C_c = zc_dis.shape[1]
        C_e = ze_dis.shape[1]
        combined = np.concatenate([zc_dis, ze_dis], axis=1)  # (N, C_c + C_e)

        # Full correlation matrix, show off-diagonal (cardio vs effusion) block
        full_corr = np.corrcoef(combined.T)          # (C_c+C_e, C_c+C_e)
        cross_block = full_corr[:C_c, C_c:]          # (C_c, C_e)

        mean_abs_corr = float(np.mean(np.abs(cross_block)))
        metrics['mean_abs_cross_corr'] = mean_abs_corr

        im = ax_corr.imshow(cross_block, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax_corr, shrink=0.8, label='Pearson r')

        # Annotate each cell
        for i in range(C_c):
            for j in range(C_e):
                ax_corr.text(j, i, f'{cross_block[i, j]:.2f}',
                             ha='center', va='center', fontsize=10, fontweight='bold',
                             color='black' if abs(cross_block[i, j]) < 0.5 else 'white')

        ax_corr.set_xticks(range(C_e))
        ax_corr.set_yticks(range(C_c))
        ax_corr.set_xticklabels([f'z_eff_{i+1}' for i in range(C_e)], fontsize=9)
        ax_corr.set_yticklabels([f'z_card_{i+1}' for i in range(C_c)], fontsize=9)
        ax_corr.set_xlabel('Effusion head channels', fontsize=9)
        ax_corr.set_ylabel('Cardio head channels', fontsize=9)

        title_col = 'green' if mean_abs_corr < 0.1 else ('orange' if mean_abs_corr < 0.25 else 'red')
        ax_corr.set_title(
            f'Cross-head correlation  (cardio ↔ effusion)\n'
            f'mean |r| = {mean_abs_corr:.3f}  '
            f'{"✓ independent" if mean_abs_corr < 0.1 else "≈ partial leakage" if mean_abs_corr < 0.25 else "✗ correlated"}',
            color=title_col, fontsize=10,
        )
    else:
        ax_corr.axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)

    return metrics


def _collect_latents_from_triplet_loader(
    model,
    variables,
    loader,
    max_samples: int = 600,
):
    """
    Collect pooled latent vectors and labels from the triplet loader.

    Returns:
        latents_common, latents_cardio, latents_effusion, labels_all
    """
    latents_common = []
    latents_cardio = []
    latents_effusion = []
    labels_all = []

    for batch_torch in loader:
        if len(labels_all) >= max_samples:
            break

        for img_key, label_val in [('x_norm', 0), ('x_disease1', 1), ('x_disease2', 2)]:
            x = jnp.array(batch_torch[img_key].permute(0, 2, 3, 1).numpy())
            latents_dict = model.apply(variables, x, method=model.encode)

            mu_c, _ = latents_dict['common']
            mu_cardio, _ = latents_dict['cardiomegaly']
            mu_effusion, _ = latents_dict['effusion']

            z_common_np = np.array(jnp.mean(mu_c, axis=(1, 2)))
            z_cardio_np = np.array(jnp.mean(mu_cardio, axis=(1, 2)))
            z_effusion_np = np.array(jnp.mean(mu_effusion, axis=(1, 2)))

            remaining = max_samples - len(labels_all)
            if remaining <= 0:
                break

            z_common_np = z_common_np[:remaining]
            z_cardio_np = z_cardio_np[:remaining]
            z_effusion_np = z_effusion_np[:remaining]

            latents_common.extend(z_common_np)
            latents_cardio.extend(z_cardio_np)
            latents_effusion.extend(z_effusion_np)
            labels_all.extend([label_val] * len(z_common_np))

            if len(labels_all) >= max_samples:
                break

    return (
        np.array(latents_common),
        np.array(latents_cardio),
        np.array(latents_effusion),
        np.array(labels_all),
    )


def _pairwise_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC for binary labels using pairwise ranking.
    """
    y = np.asarray(y_true).astype(np.int32)
    s = np.asarray(scores).astype(np.float64)

    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')

    comp = pos[:, None] - neg[None, :]
    wins = np.sum(comp > 0)
    ties = np.sum(comp == 0)
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _linear_probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    train_frac: float = 0.7,
    l2_reg: float = 1e-3,
) -> float:
    """
    Train/eval a simple linear probe (LDA-style with ridge) and return test AUC.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)

    if X.ndim != 2 or len(X) < 8:
        return float('nan')
    if np.sum(y == 0) < 4 or np.sum(y == 1) < 4:
        return float('nan')

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n_train = int(round(len(X) * train_frac))
    n_train = max(4, min(n_train, len(X) - 4))

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    if np.sum(y_train == 0) < 2 or np.sum(y_train == 1) < 2:
        return float('nan')
    if np.sum(y_test == 0) < 2 or np.sum(y_test == 1) < 2:
        return float('nan')

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    m0 = np.mean(X0, axis=0)
    m1 = np.mean(X1, axis=0)

    X0c = X0 - m0
    X1c = X1 - m1
    cov = (X0c.T @ X0c + X1c.T @ X1c) / max(len(X_train) - 2, 1)
    cov += l2_reg * np.eye(X.shape[1], dtype=np.float64)

    try:
        w = np.linalg.solve(cov, (m1 - m0))
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ (m1 - m0)
    b = -0.5 * np.dot((m1 + m0), w)

    scores = X_test @ w + b
    return _pairwise_auc_binary(y_test, scores)


def _load_overlap_image_ids(
    csv_path: str,
    class_a: int = 3,
    class_b: int = 10,
):
    """
    Find image_ids that contain both class_a and class_b in the CSV annotations.
    """
    ids_a = set()
    ids_b = set()

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'image_id' not in reader.fieldnames or 'class_id' not in reader.fieldnames:
            raise ValueError("CSV must include 'image_id' and 'class_id' columns.")

        for row in reader:
            image_id = row.get('image_id')
            class_id_raw = row.get('class_id')
            if image_id is None or class_id_raw is None:
                continue
            try:
                class_id = int(float(class_id_raw))
            except ValueError:
                continue

            if class_id == class_a:
                ids_a.add(image_id)
            elif class_id == class_b:
                ids_b.add(image_id)

    return sorted(ids_a & ids_b)


def _encode_overlap_latents(
    model,
    variables,
    dataset,
    overlap_ids,
    max_samples: int = 128,
    batch_size: int = 16,
    seed: int = 0,
):
    """
    Encode real overlap images and return concatenated disease latents [cardio, effusion].
    """
    if max_samples <= 0 or len(overlap_ids) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    rng = np.random.default_rng(seed)
    chosen = list(overlap_ids)
    if len(chosen) > max_samples:
        chosen = list(rng.choice(chosen, size=max_samples, replace=False))

    images = []
    latents = []

    def _flush_batch():
        nonlocal images
        if len(images) == 0:
            return
        x = jnp.array(np.stack(images, axis=0))
        latents_dict = model.apply(variables, x, method=model.encode)
        mu_cardio, _ = latents_dict['cardiomegaly']
        mu_effusion, _ = latents_dict['effusion']
        z_cardio = np.array(jnp.mean(mu_cardio, axis=(1, 2)))
        z_effusion = np.array(jnp.mean(mu_effusion, axis=(1, 2)))
        latents.append(np.concatenate([z_cardio, z_effusion], axis=1))
        images = []

    for image_id in chosen:
        pixel_array = dataset._load_dicom(image_id)
        if pixel_array is None:
            continue
        x_torch = dataset._preprocess_image(pixel_array)  # (1, H, W)
        x = x_torch.permute(1, 2, 0).numpy()  # (H, W, 1)
        images.append(x)

        if len(images) >= batch_size:
            _flush_batch()

    _flush_batch()

    if len(latents) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(latents, axis=0)


def compute_disentanglement_metrics(
    model,
    vae_params,
    vae_batch_stats,
    loader,
    overlap_ids,
    overlap_dataset,
    max_samples: int = 600,
    overlap_max_samples: int = 128,
    overlap_batch_size: int = 16,
    seed: int = 0,
):
    """
    Compute disentanglement metrics beyond PCA silhouette.

    Metrics:
    - Head-specific linear probe AUC
    - Cross-head leakage score (inactive/active head energy ratio)
    - Synthetic-X proximity to real overlap cluster
    """
    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    _, latents_cardio, latents_effusion, labels_all = _collect_latents_from_triplet_loader(
        model, variables, loader, max_samples=max_samples
    )

    metrics = {}
    if len(labels_all) < 10:
        return metrics

    labels_all = labels_all.astype(np.int32)

    disease_mask = labels_all != 0
    z_cardio_disease = latents_cardio[disease_mask]
    z_effusion_disease = latents_effusion[disease_mask]
    labels_disease = labels_all[disease_mask]

    if len(labels_disease) >= 12:
        y_cardio = (labels_disease == 2).astype(np.int32)
        y_effusion = (labels_disease == 1).astype(np.int32)

        auc_cardio = _linear_probe_auc(
            z_cardio_disease,
            y_cardio,
            seed=seed + 101,
        )
        auc_effusion = _linear_probe_auc(
            z_effusion_disease,
            y_effusion,
            seed=seed + 202,
        )
        metrics['probe_auc/cardiomegaly_head'] = auc_cardio
        metrics['probe_auc/effusion_head'] = auc_effusion
        if np.isfinite(auc_cardio) and np.isfinite(auc_effusion):
            metrics['probe_auc/mean'] = 0.5 * (auc_cardio + auc_effusion)

    cardio_energy = np.sum(np.square(latents_cardio), axis=1)
    effusion_energy = np.sum(np.square(latents_effusion), axis=1)

    cardio_active = cardio_energy[labels_all == 2]
    cardio_inactive_on_effusion = cardio_energy[labels_all == 1]
    effusion_active = effusion_energy[labels_all == 1]
    effusion_inactive_on_cardio = effusion_energy[labels_all == 2]

    eps = 1e-8
    if len(cardio_active) > 0 and len(cardio_inactive_on_effusion) > 0:
        leak_eff_to_cardio = float(
            np.mean(cardio_inactive_on_effusion) / (np.mean(cardio_active) + eps)
        )
        metrics['leakage/effusion_into_cardiomegaly_ratio'] = leak_eff_to_cardio
    if len(effusion_active) > 0 and len(effusion_inactive_on_cardio) > 0:
        leak_cardio_to_eff = float(
            np.mean(effusion_inactive_on_cardio) / (np.mean(effusion_active) + eps)
        )
        metrics['leakage/cardiomegaly_into_effusion_ratio'] = leak_cardio_to_eff

    leak_parts = []
    if 'leakage/effusion_into_cardiomegaly_ratio' in metrics:
        leak_parts.append(metrics['leakage/effusion_into_cardiomegaly_ratio'])
    if 'leakage/cardiomegaly_into_effusion_ratio' in metrics:
        leak_parts.append(metrics['leakage/cardiomegaly_into_effusion_ratio'])
    if len(leak_parts) > 0:
        metrics['leakage/cross_head_score'] = float(np.mean(leak_parts))

    overlap_latents = _encode_overlap_latents(
        model,
        variables,
        overlap_dataset,
        overlap_ids,
        max_samples=overlap_max_samples,
        batch_size=overlap_batch_size,
        seed=seed + 303,
    )
    if overlap_latents.shape[0] > 4:
        z_cardio_cardio = latents_cardio[labels_all == 2]
        z_effusion_eff = latents_effusion[labels_all == 1]
        n_syn = min(len(z_cardio_cardio), len(z_effusion_eff), len(overlap_latents))

        if n_syn >= 4:
            rng = np.random.default_rng(seed + 404)
            idx_c = rng.permutation(len(z_cardio_cardio))[:n_syn]
            idx_e = rng.permutation(len(z_effusion_eff))[:n_syn]
            synth_latents = np.concatenate([z_cardio_cardio[idx_c], z_effusion_eff[idx_e]], axis=1)

            mu_real = np.mean(overlap_latents, axis=0)
            mu_synth = np.mean(synth_latents, axis=0)

            centroid_l2 = float(np.linalg.norm(mu_synth - mu_real))
            real_radius = float(np.mean(np.linalg.norm(overlap_latents - mu_real, axis=1)))
            centroid_normed = centroid_l2 / (real_radius + eps)

            nn_matrix = np.linalg.norm(
                synth_latents[:, None, :] - overlap_latents[None, :, :],
                axis=2,
            )
            synth_to_real_nn = float(np.mean(np.min(nn_matrix, axis=1)))

            metrics['overlap/real_count'] = float(len(overlap_latents))
            metrics['overlap/synth_count'] = float(n_syn)
            metrics['overlap/synth_real_centroid_l2'] = centroid_l2
            metrics['overlap/synth_real_centroid_normed'] = float(centroid_normed)
            metrics['overlap/synth_real_nn_normed'] = float(
                synth_to_real_nn / (real_radius + eps)
            )

    return metrics


def main():
    args = parse_args()

    # By default, keep manifold visualization cadence aligned with reconstruction sampling.
    if args.manifold_every < 0:
        args.manifold_every = args.sample_every
    # By default, evaluate disentanglement metrics on the same cadence as manifold plots.
    if args.disent_metrics_every < 0:
        args.disent_metrics_every = args.manifold_every

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Resolve compute dtype for mixed precision
    if args.half_precision == "bf16":
        compute_dtype = jnp.bfloat16
    elif args.half_precision == "fp16":
        compute_dtype = jnp.float16
    else:
        compute_dtype = jnp.float32

    print("="*60)
    print("MULTI-HEAD SEPVAE WITH CHESS BACKBONE")
    print("="*60)
    if compute_dtype != jnp.float32:
        print(f"  Mixed precision: {args.half_precision}")
    if args.gradient_checkpointing:
        print(f"  Gradient checkpointing: enabled")

    # ===== Setup output directories =====
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_slug = f"{args.exp_name}-{timestamp}"
    output_dir = Path(args.output_root) / exp_slug
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    samples_dir = ensure_dir(output_dir / "samples")
    manifold_dir = ensure_dir(output_dir / "manifold")
    diag_dir = ensure_dir(output_dir / "diagnostics")

    print(f"\nOutput directory: {output_dir}")

    # ===== Initialize W&B =====
    if args.wandb and _WANDB:
        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        if args.wandb_run_id:
            # Resume logging into an existing W&B run
            wandb_kwargs['id'] = args.wandb_run_id
            wandb_kwargs['resume'] = 'must'
            print(f"✓ W&B resuming run: {args.wandb_run_id}")
        else:
            wandb_kwargs['name'] = exp_slug
        wandb.init(**wandb_kwargs)
        print("✓ W&B initialized")
    elif args.wandb and not _WANDB:
        print("⚠ W&B requested but not installed, skipping")

    # ===== Load dataset =====
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    dataset = VinBigDataTripletDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        exclude_cross_disease_overlap=args.exclude_cross_disease_overlap,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=jax_collate_fn,
        drop_last=True
    )

    print(f"✓ Dataset loaded: {len(dataset)} triplets")
    print(f"✓ Batch size: {args.batch_size} triplets → {args.batch_size * 3} images")
    print(f"✓ Steps per epoch: {len(loader)}")
    if args.manifold_every > 0:
        print(f"✓ Manifold plots: every {args.manifold_every} epochs "
              f"({args.manifold_method}, max_samples={args.manifold_max_samples})")
    else:
        print("  Manifold plots: disabled")

    overlap_ids = []
    disent_eval_enabled = (args.disent_metrics_every > 0) or args.disent_eval_on_sample
    if disent_eval_enabled:
        try:
            overlap_ids = _load_overlap_image_ids(args.csv_path, class_a=3, class_b=10)
            periodic_desc = (
                f"every {args.disent_metrics_every} epochs" if args.disent_metrics_every > 0 else "periodic disabled"
            )
            sample_desc = "enabled" if args.disent_eval_on_sample else "disabled"
            print(f"✓ Disentanglement metrics: periodic={periodic_desc}, on_sample={sample_desc}")
            print(f"  Eval settings: triplet_samples={args.disent_metrics_max_samples}, "
                  f"overlap_eval_samples={args.overlap_eval_samples}, "
                  f"overlap_eval_batch_size={args.overlap_eval_batch_size}")
            print(f"  Real overlap pool (cardio∩effusion): {len(overlap_ids)} images")
        except Exception as e:
            print(f"⚠ Could not initialize overlap IDs for disentanglement metrics: {e}")
            print("  Disentanglement metrics involving real overlap will be skipped.")
    else:
        print("  Disentanglement metrics: disabled")
    if args.disent_eval_on_sample and args.sample_every <= 0:
        print("  ⚠ disent_eval_on_sample requested, but sample_every <= 0 so sample-triggered eval will not run.")

    # ===== Load CheSS weights =====
    print("\n" + "="*60)
    print("LOADING CHESS WEIGHTS")
    print("="*60)

    if args.chess_converted and os.path.exists(args.chess_converted):
        print(f"Loading pre-converted weights from: {args.chess_converted}")
        chess_params, chess_batch_stats = load_converted_weights(args.chess_converted)
    else:
        print(f"Converting CheSS weights from: {args.chess_checkpoint}")
        chess_params, chess_batch_stats = convert_chess_resnet50(args.chess_checkpoint, verbose=True)

        # Save converted weights for future use
        from utils.weight_converter import save_converted_weights
        converted_path = output_dir / "chess_jax_params.npy"
        save_converted_weights((chess_params, chess_batch_stats), str(converted_path))
        print(f"✓ Saved converted weights to: {converted_path}")

    # ===== Initialize models =====
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)

    # SepVAE model
    sepvae = SepVAE(
        z_channels_common=args.z_channels_common,
        z_channels_disease=args.z_channels_disease,
        frozen_backbone=args.frozen_backbone,
        use_fpn=args.use_fpn,
        fpn_channels=args.fpn_channels,
        unfreeze_from=args.unfreeze_from,
        sigma_inactive=args.sigma_inactive,
        use_label_attention=args.use_label_attention,
        attn_query_dim=args.attn_query_dim,
        dtype=compute_dtype,
        use_remat=args.gradient_checkpointing,
        upsample_method=args.upsample_method,
    )

    # Dummy inputs for initialization
    dummy_x = jnp.ones((1, args.img_size, args.img_size, 1))
    dummy_labels = jnp.array([0])
# Initialize SepVAE
    rng, init_rng = jax.random.split(rng)
    vae_vars = sepvae.init(
        {'params': init_rng, 'dropout': init_rng},
        dummy_x, dummy_labels, key=init_rng, train=True
    )
    vae_params = vae_vars['params']
    vae_batch_stats = vae_vars.get('batch_stats', {})

    # Inject CheSS weights into backbone (both params and batch_stats)
    print("\nInjecting CheSS weights into backbone...")
    vae_params = vae_params.copy()
    vae_params['encoder']['backbone'] = chess_params

    # Inject BatchNorm running statistics from CheSS
    if vae_batch_stats:
        vae_batch_stats = vae_batch_stats.copy()
        vae_batch_stats['encoder']['backbone'] = chess_batch_stats
    else:
        vae_batch_stats = {'encoder': {'backbone': chess_batch_stats}}

    # Ensure all are JAX arrays
    vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, vae_batch_stats)

    vae_param_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    print(f"✓ SepVAE parameters: {vae_param_count:,}")
    bs_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_batch_stats))
    print(f"✓ SepVAE batch_stats: {bs_count:,} (frozen BN running stats)")

    # PatchGAN discriminator (optional)
    patch_disc = None
    patch_disc_params = None
    if args.weight_adversarial > 0.0:
        patch_disc = NLayerDiscriminator(in_channels=1, n_layers=3, dtype=compute_dtype)
        rng, init_rng = jax.random.split(rng)
        dummy_img = jnp.ones((1, args.img_size, args.img_size, 1))
        pd_vars = patch_disc.init(init_rng, dummy_img)
        patch_disc_params = pd_vars['params']
        pd_count = sum(p.size for p in jax.tree_util.tree_leaves(patch_disc_params))
        print(f"✓ PatchGAN discriminator parameters: {pd_count:,}")
    else:
        print("  PatchGAN discriminator: disabled (weight_adversarial=0)")

    # Standalone backbone for perceptual loss (shared weights, no extra params)
    backbone_for_percep = None
    backbone_variables_percep = None
    if args.weight_perceptual > 0.0:
        backbone_for_percep = ResNet50CheSS()
        backbone_variables_percep = {
            'params': chess_params,
            'batch_stats': chess_batch_stats,
        }
        backbone_variables_percep = jax.tree_util.tree_map(jnp.array, backbone_variables_percep)
        print(f"✓ Backbone perceptual loss: enabled (weight={args.weight_perceptual})")
    else:
        print("  Backbone perceptual loss: disabled (weight_perceptual=0)")

    if args.use_fpn:
        print(f"✓ FPN enabled: {args.fpn_channels} channels")
    if args.unfreeze_from:
        print(f"✓ Partial unfreezing from: {args.unfreeze_from}")
    if args.free_bits > 0:
        print(f"✓ Free-bits KL: {args.free_bits} nats per channel")

    # Upsample method (anti-aliasing)
    upsample_notes = {
        'bilinear': '(smooth, prevents checkerboard - recommended)',
        'subpixel': '(learned, best quality)',
        'nearest': '(legacy, causes checkerboard artifacts - not recommended)',
    }
    print(f"✓ Decoder upsample method: {args.upsample_method} {upsample_notes.get(args.upsample_method, '')}")

    # ===== Create optimizers =====
    print("\n" + "="*60)
    print("CREATING OPTIMIZERS")
    print("="*60)

    # ===== LR schedule (R3: cosine decay) =====
    total_steps = args.epochs * len(loader)
    decay_start_step = (
        args.lr_decay_epochs * len(loader) if args.lr_decay_epochs > 0 else total_steps
    )
    decay_steps = max(total_steps - decay_start_step, 1)

    def _make_schedule(init_lr: float) -> optax.Schedule:
        """Constant until lr_decay_epochs, then cosine decay to 10% of init_lr."""
        if args.lr_decay_epochs <= 0:
            return init_lr
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(init_lr),
                optax.cosine_decay_schedule(
                    init_value=init_lr,
                    decay_steps=decay_steps,
                    alpha=0.1,  # end lr = 0.1 * init_lr
                ),
            ],
            boundaries=[decay_start_step],
        )

    lr_vae_schedule = _make_schedule(args.lr_vae)
    lr_backbone_schedule = _make_schedule(args.lr_backbone)

    decay_desc = (f"cosine decay from epoch {args.lr_decay_epochs}"
                  if args.lr_decay_epochs > 0 else "constant")

    # VAE optimizer — multi-rate if backbone is partially unfrozen
    if args.unfreeze_from is not None:
        # Multi-rate: lower lr for backbone layers, normal lr for rest
        # Build label tree from the param structure
        from flax import traverse_util
        flat_params = traverse_util.flatten_dict(vae_params)
        flat_labels = {}
        for key_tuple in flat_params:
            path_str = '/'.join(str(k) for k in key_tuple)
            if 'encoder' in path_str and 'backbone' in path_str:
                flat_labels[key_tuple] = 'backbone'
            else:
                flat_labels[key_tuple] = 'other'
        param_labels = traverse_util.unflatten_dict(flat_labels)

        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.multi_transform(
                {
                    'backbone': optax.adamw(
                        learning_rate=lr_backbone_schedule,
                        weight_decay=args.weight_decay,
                    ),
                    'other': optax.adamw(
                        learning_rate=lr_vae_schedule,
                        weight_decay=args.weight_decay,
                    ),
                },
                param_labels,
            )
        )
        print(f"✓ VAE optimizer: Multi-rate AdamW (backbone lr={args.lr_backbone}, "
              f"rest lr={args.lr_vae}, schedule={decay_desc})")
    else:
        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=lr_vae_schedule, weight_decay=args.weight_decay)
        )
        print(f"✓ VAE optimizer: AdamW (lr={args.lr_vae}, wd={args.weight_decay}, "
              f"schedule={decay_desc})")

    vae_state = TrainState.create(apply_fn=None, params=vae_params, tx=tx_vae)

    # PatchGAN discriminator optimizer (optional)
    patch_disc_state = None
    if patch_disc is not None:
        tx_patch = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=args.lr_patch_disc, weight_decay=args.weight_decay)
        )
        patch_disc_state = TrainState.create(apply_fn=None, params=patch_disc_params, tx=tx_patch)
        print(f"✓ PatchGAN optimizer: AdamW (lr={args.lr_patch_disc})")

    # R11: Disease discriminability classifiers (optional)
    clf_cardio = None
    clf_cardio_params = None
    clf_effusion = None
    clf_effusion_params = None
    clf_cardio_state = None
    clf_effusion_state = None
    if args.use_disease_clf:
        dummy_z_disease_flat = jnp.ones((1, args.z_channels_disease))

        clf_cardio = DiseaseClassifier()
        clf_effusion = DiseaseClassifier()

        rng, init_rng = jax.random.split(rng)
        clf_cardio_params = clf_cardio.init(init_rng, dummy_z_disease_flat)['params']

        rng, init_rng = jax.random.split(rng)
        clf_effusion_params = clf_effusion.init(init_rng, dummy_z_disease_flat)['params']

        lr_clf_schedule = _make_schedule(args.lr_disease_clf)
        tx_clf = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=lr_clf_schedule, weight_decay=args.weight_decay)
        )
        clf_cardio_state = TrainState.create(
            apply_fn=None, params=clf_cardio_params, tx=tx_clf
        )
        clf_effusion_state = TrainState.create(
            apply_fn=None, params=clf_effusion_params, tx=tx_clf
        )

        clf_count = sum(p.size for p in jax.tree_util.tree_leaves(clf_cardio_params))
        print(f"✓ Disease discriminability classifiers: enabled "
              f"(params={clf_count:,} each, weight={args.weight_disease_clf}, lr={args.lr_disease_clf})")
    else:
        print("  Disease discriminability classifiers: disabled (--use_disease_clf not set)")

    if args.use_contrastive:
        print(f"✓ Paired contrastive loss: enabled "
              f"(weight={args.weight_contrastive}, margin={args.contrastive_margin})")
    else:
        print("  Paired contrastive loss: disabled (--use_contrastive not set)")


    # ===== Resume from checkpoint (optional) =====
    start_epoch = 1
    global_step = 0

    if args.resume:
        print(f"\n{'='*60}")
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        print(f"Loading: {args.resume}")

        with open(args.resume, 'rb') as f:
            ckpt = msgpack_restore(f.read())

        # Verify architecture matches
        ckpt_args = ckpt.get('args', {})
        for key in ['z_channels_common', 'z_channels_disease', 'use_fpn',
                     'fpn_channels', 'unfreeze_from', 'upsample_method',
                     'use_label_attention', 'attn_query_dim']:
            ckpt_val = ckpt_args.get(key)
            curr_val = getattr(args, key, None)
            if ckpt_val is not None and curr_val is not None and str(ckpt_val) != str(curr_val):
                print(f"  ⚠ Architecture mismatch: {key}={ckpt_val} (ckpt) vs {curr_val} (current)")

        # Restore VAE state (params as arrays, opt_state with structure template)
        restored_vae_params = jax.tree_util.tree_map(jnp.array, ckpt['vae_params'])
        restored_vae_opt = from_state_dict(vae_state.opt_state, ckpt['vae_opt_state'])
        vae_state = vae_state.replace(
            params=restored_vae_params,
            opt_state=restored_vae_opt,
            step=int(ckpt['global_step']),
        )

        # Restore batch_stats (frozen, but ensure consistency)
        vae_batch_stats = jax.tree_util.tree_map(jnp.array, ckpt['vae_batch_stats'])

        # Restore PatchGAN state (if available)
        if patch_disc_state is not None and ckpt.get('patch_disc_params') is not None:
            restored_pd_params = jax.tree_util.tree_map(jnp.array, ckpt['patch_disc_params'])
            restored_pd_opt = from_state_dict(patch_disc_state.opt_state, ckpt['patch_disc_opt_state'])
            patch_disc_state = patch_disc_state.replace(
                params=restored_pd_params,
                opt_state=restored_pd_opt,
            )
            print("  ✓ Restored PatchGAN discriminator state")

        # Restore disease classifier states (R11)
        if clf_cardio_state is not None and ckpt.get('clf_cardio_params') is not None:
            restored_clf_c_params = jax.tree_util.tree_map(jnp.array, ckpt['clf_cardio_params'])
            restored_clf_c_opt = from_state_dict(clf_cardio_state.opt_state, ckpt['clf_cardio_opt_state'])
            clf_cardio_state = clf_cardio_state.replace(
                params=restored_clf_c_params,
                opt_state=restored_clf_c_opt,
            )
            restored_clf_e_params = jax.tree_util.tree_map(jnp.array, ckpt['clf_effusion_params'])
            restored_clf_e_opt = from_state_dict(clf_effusion_state.opt_state, ckpt['clf_effusion_opt_state'])
            clf_effusion_state = clf_effusion_state.replace(
                params=restored_clf_e_params,
                opt_state=restored_clf_e_opt,
            )
            print("  ✓ Restored disease classifier states (R11)")

        # Restore RNG and training progress
        rng = jnp.array(ckpt['rng'])
        start_epoch = int(ckpt['epoch']) + 1
        global_step = int(ckpt['global_step'])

        print(f"  ✓ Restored VAE params + optimizer (Adam moments)")
        print(f"  ✓ Checkpoint was epoch {int(ckpt['epoch'])}, step {global_step}")
        print(f"  ✓ Resuming from epoch {start_epoch}")

    # ===== Loss configuration =====
    loss_cfg = SepVAELossConfig(
        weight_rec=args.weight_rec,
        weight_kl_common=args.weight_kl_common,
        weight_kl_disease=args.weight_kl_disease,
        weight_null=args.weight_null,
        weight_orthogonality=args.weight_orthogonality,
        weight_perceptual=args.weight_perceptual,
        weight_adversarial=args.weight_adversarial,
        weight_contrastive=args.weight_contrastive if args.use_contrastive else 0.0,
        weight_disease_clf=args.weight_disease_clf if args.use_disease_clf else 0.0,
        weight_spatial_attn=args.weight_spatial_attn if args.use_label_attention else 0.0,
        sigma_inactive=args.sigma_inactive,
        free_bits=args.free_bits,
        disc_start_epoch=args.disc_start_epoch,
        contrastive_margin=args.contrastive_margin,
    )

    print(f"\nLoss weights:")
    print(f"  Reconstruction: {loss_cfg.weight_rec}")
    print(f"  KL (common): {loss_cfg.weight_kl_common}")
    print(f"  KL (disease): {loss_cfg.weight_kl_disease}")
    print(f"  Nulling: {loss_cfg.weight_null}")
    print(f"  Orthogonality: {loss_cfg.weight_orthogonality}")
    print(f"  Perceptual: {loss_cfg.weight_perceptual}")
    print(f"  Adversarial: {loss_cfg.weight_adversarial}")
    print(f"  Contrastive: {loss_cfg.weight_contrastive}")
    print(f"  Disease classifier (R11): {loss_cfg.weight_disease_clf}")
    print(f"  Spatial attn (R12): {loss_cfg.weight_spatial_attn}")
    print(f"  Sigma inactive: {loss_cfg.sigma_inactive}")
    print(f"  Free-bits: {loss_cfg.free_bits}")
    if args.kl_warmup_epochs > 0:
        print(f"  KL warmup: linear anneal over {args.kl_warmup_epochs} epochs")
    if args.weight_adversarial > 0:
        print(f"  Disc start epoch: {args.disc_start_epoch}")

    # ===== Define training steps =====
    # vae_batch_stats is frozen (never updated), captured in closure by jit
    # backbone_for_percep and backbone_variables_percep are also frozen

    # Create backbone apply function for perceptual loss
    _backbone_apply_fn = backbone_for_percep.apply if backbone_for_percep is not None else None

    _use_remat = args.gradient_checkpointing  # static bool captured in jit closure

    # Capture disease-clf models in closures (None when disabled)
    _clf_cardio = clf_cardio
    _clf_effusion = clf_effusion

    @jax.jit
    def vae_step(vae_state, batch, key, kl_anneal, current_epoch,
                 pd_params=None,
                 clf_c_params=None, clf_e_params=None):
        """Update VAE parameters with all loss terms."""
        def loss_fn(params):
            total_loss, (logs, _) = sepvae_loss(
                sepvae, params,
                batch, key, loss_cfg, batch_stats=vae_batch_stats,
                kl_anneal=kl_anneal,
                patch_disc=patch_disc,
                patch_disc_params=pd_params,
                backbone_apply_fn=_backbone_apply_fn,
                backbone_variables=backbone_variables_percep,
                current_epoch=current_epoch,
                use_remat=_use_remat,
                clf_cardio=_clf_cardio,
                clf_cardio_params=clf_c_params,
                clf_effusion=_clf_effusion,
                clf_effusion_params=clf_e_params,
            )
            return total_loss, logs

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(vae_state.params)
        vae_state = vae_state.apply_gradients(grads=grads)

        return vae_state, logs

    @jax.jit
    def patch_disc_step(vae_state, pd_state, batch, key, current_epoch):
        """Update PatchGAN discriminator parameters."""
        def loss_fn(pd_params):
            _, (_, pd_loss) = sepvae_loss(
                sepvae, vae_state.params,
                batch, key, loss_cfg, batch_stats=vae_batch_stats,
                patch_disc=patch_disc,
                patch_disc_params=pd_params,
                current_epoch=current_epoch,
                use_remat=_use_remat,
            )
            return pd_loss

        pd_loss, grads = jax.value_and_grad(loss_fn)(pd_state.params)
        pd_state = pd_state.apply_gradients(grads=grads)

        return pd_state, pd_loss

    @jax.jit
    def clf_step(vae_state, clf_c_state, clf_e_state, batch, key):
        """Update disease classifier parameters (R11).

        Minimises BCE(clf_k(z_disease_k.mu), label_k) w.r.t. classifier params
        only — encoder params are treated as fixed inputs here.  The VAE sees the
        same BCE signal in vae_step (with clf params frozen there), creating
        cooperative pressure for both the encoder and classifiers to improve.
        """
        def loss_fn(clf_c_params, clf_e_params):
            _, (logs, _) = sepvae_loss(
                sepvae, vae_state.params,
                batch, key, loss_cfg, batch_stats=vae_batch_stats,
                clf_cardio=_clf_cardio,
                clf_cardio_params=clf_c_params,
                clf_effusion=_clf_effusion,
                clf_effusion_params=clf_e_params,
            )
            return logs['loss/disease_clf']

        clf_loss, (c_grads, e_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1)
        )(clf_c_state.params, clf_e_state.params)

        clf_c_state = clf_c_state.apply_gradients(grads=c_grads)
        clf_e_state = clf_e_state.apply_gradients(grads=e_grads)

        return clf_c_state, clf_e_state, clf_loss

    @jax.jit
    def encode_batch_for_attn(vae_params, x):
        """Encode a batch and return the full latents_dict (includes attn_maps when enabled)."""
        variables = {'params': vae_params}
        if vae_batch_stats:
            variables['batch_stats'] = vae_batch_stats
        return sepvae.apply(variables, x, method=sepvae.encode)

    @jax.jit
    def reconstruct_batch(vae_params, batch, key):
        """Forward pass for reconstruction visualization (no gradients)."""
        x_norm = batch['x_norm']
        x_disease1 = batch['x_disease1']
        x_disease2 = batch['x_disease2']
        labels = batch['disease_labels']

        x = jnp.concatenate([x_norm, x_disease1, x_disease2], axis=0)

        variables = {'params': vae_params}
        if vae_batch_stats:
            variables['batch_stats'] = vae_batch_stats

        x_rec, _, _ = sepvae.apply(variables, x, labels, key=key, train=False)
        return x, x_rec

    # ===== Verbose backbone diagnostics (optional) =====
    if args.verbose_backbone:
        print("\n" + "="*60)
        print("BACKBONE FEATURE DIAGNOSTICS")
        print("="*60)

        # Collect images from all three classes (normal, effusion, cardiomegaly)
        # Each triplet batch has x_norm (label=0), x_disease1 (label=1), x_disease2 (label=2)
        vis_images = []
        vis_labels = []
        n_collected = 0

        for batch_torch in loader:
            x_norm = batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()
            x_dis1 = batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()
            x_dis2 = batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy()
            B = x_norm.shape[0]

            # Stack all three: (3*B, H, W, C) with matching labels
            vis_images.append(jnp.array(np.concatenate([x_norm, x_dis1, x_dis2], axis=0)))
            vis_labels.append(np.array(batch_torch['disease_labels'].numpy()))
            n_collected += 3 * B

            if n_collected >= args.verbose_n_samples:
                break

        vis_images = jnp.concatenate(vis_images, axis=0)[:args.verbose_n_samples]
        vis_labels = jnp.array(np.concatenate(vis_labels, axis=0)[:args.verbose_n_samples])

        diag_dir = ensure_dir(output_dir / "backbone_diagnostics")
        visualize_backbone_features(
            sepvae, vae_state.params, vis_images, vis_labels,
            save_dir=str(diag_dir), batch_stats=vae_batch_stats
        )

        if args.wandb and _WANDB:
            import glob
            for img_path in sorted(glob.glob(f"{diag_dir}/*.png")):
                name = os.path.basename(img_path).replace('.png', '')
                wandb.log({f"backbone/{name}": wandb.Image(img_path)})

        print(f"Backbone diagnostics saved to: {diag_dir}")

    def run_disent_eval(epoch: int, trigger: str):
        """
        Compute and log disentanglement metrics for the current model state.

        Returns:
            status_code:
              1 = metrics computed with at least one finite value
              0 = no metrics (insufficient samples)
             -1 = metrics dictionary returned but all values non-finite
        """
        print(f"  Evaluating disentanglement metrics ({trigger})...")
        disent_metrics = compute_disentanglement_metrics(
            sepvae,
            vae_state.params,
            vae_batch_stats,
            loader,
            overlap_ids,
            dataset,
            max_samples=args.disent_metrics_max_samples,
            overlap_max_samples=args.overlap_eval_samples,
            overlap_batch_size=args.overlap_eval_batch_size,
            seed=args.seed + epoch,
        )

        finite_metrics = {}
        for key, value in disent_metrics.items():
            if np.isfinite(value):
                finite_metrics[key] = float(value)

        if not disent_metrics:
            status_code = 0
            print("    Skipped (insufficient samples).")
        elif not finite_metrics:
            status_code = -1
            print("    Disentanglement metrics computed, but values were non-finite.")
        else:
            status_code = 1
            metric_str = ", ".join(f"{k}={v:.3f}" for k, v in finite_metrics.items())
            print(f"    {metric_str}")

        if args.wandb and _WANDB:
            wandb_payload = {
                "disent/eval_success": float(status_code == 1),
                "disent/eval_status": float(status_code),
                "disent/eval_trigger/sample": float(trigger == "sample"),
                "disent/eval_trigger/periodic": float(trigger == "periodic"),
            }
            for key, value in finite_metrics.items():
                wandb_payload[f"disent/{key}"] = value
            wandb.log(wandb_payload, step=global_step)

        return status_code

    # ===== Training loop =====
    print("\n" + "="*60)
    print("STARTING TRAINING" + (f" (resuming from epoch {start_epoch})" if start_epoch > 1 else ""))
    print("="*60)

    for epoch in range(start_epoch, args.epochs + 1):
        # Compute KL annealing factor for this epoch
        if args.kl_warmup_epochs > 0:
            kl_anneal = jnp.float32(min(1.0, epoch / args.kl_warmup_epochs))
        else:
            kl_anneal = jnp.float32(1.0)

        print(f"\nEpoch {epoch}/{args.epochs}" +
              (f"  [KL anneal={float(kl_anneal):.3f}]" if args.kl_warmup_epochs > 0 else ""))

        epoch_logs = []

        for batch_idx, batch_torch in enumerate(loader):
            # Convert PyTorch tensors to JAX arrays (NCHW → NHWC)
            batch = {
                'x_norm': jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()),
                'x_disease1': jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()),
                'x_disease2': jnp.array(batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy()),
                'disease_labels': jnp.array(batch_torch['disease_labels'].numpy()),
                'bbox_disease1': jnp.array(batch_torch['bbox_disease1'].numpy()),  # (B, 4) effusion bboxes
                'bbox_disease2': jnp.array(batch_torch['bbox_disease2'].numpy()),  # (B, 4) cardio bboxes
            }

            # Get random key for this step
            rng, step_key = jax.random.split(rng)

            current_epoch_jnp = jnp.int32(epoch)

            # Update VAE (with all loss terms)
            pd_params = patch_disc_state.params if patch_disc_state is not None else None
            clf_c_params = clf_cardio_state.params if clf_cardio_state is not None else None
            clf_e_params = clf_effusion_state.params if clf_effusion_state is not None else None
            vae_state, vae_logs = vae_step(
                vae_state, batch, step_key, kl_anneal,
                current_epoch_jnp,
                pd_params=pd_params,
                clf_c_params=clf_c_params,
                clf_e_params=clf_e_params,
            )

            # Update PatchGAN discriminator (if enabled)
            pd_loss = jnp.float32(0.0)
            if patch_disc_state is not None:
                patch_disc_state, pd_loss = patch_disc_step(
                    vae_state, patch_disc_state, batch, step_key, current_epoch_jnp,
                )

            # Update disease classifiers (R11, if enabled)
            clf_loss_step = jnp.float32(0.0)
            if clf_cardio_state is not None and clf_effusion_state is not None:
                clf_cardio_state, clf_effusion_state, clf_loss_step = clf_step(
                    vae_state, clf_cardio_state, clf_effusion_state, batch, step_key
                )

            # Add discriminator/classifier losses to logs
            vae_logs['loss/patch_disc_update'] = pd_loss
            vae_logs['loss/disease_clf_update'] = clf_loss_step

            epoch_logs.append(vae_logs)

            # Logging
            if global_step % args.log_every == 0:
                log_str = (
                    f"Step {global_step}: "
                    f"Loss={vae_logs['loss/total']:.4f}, "
                    f"Rec={vae_logs['loss/reconstruction']:.4f}, "
                    f"KL={vae_logs['loss/kl_total']:.4f}, "
                    f"KLw={vae_logs['loss/kl_weighted']:.4f}, "
                    f"Null={vae_logs['loss/nulling']:.4f}, "
                    f"Ortho={vae_logs['loss/orthogonality']:.4f}"
                )
                if args.weight_perceptual > 0:
                    log_str += f", Percep={vae_logs['loss/perceptual']:.4f}"
                if args.weight_adversarial > 0:
                    log_str += f", GAN={vae_logs['loss/gen_adversarial']:.4f}"
                if args.use_contrastive:
                    log_str += f", Contra={vae_logs.get('loss/contrastive', 0.0):.4f}"
                if args.use_disease_clf:
                    log_str += f", DiseaseClf={vae_logs.get('loss/disease_clf', 0.0):.4f}"
                if args.weight_spatial_attn > 0 and args.use_label_attention:
                    log_str += f", SpatialAttn={vae_logs.get('loss/spatial_attn', 0.0):.4f}"
                print(log_str)

                if args.wandb and _WANDB:
                    # Convert JAX arrays to Python floats for W&B
                    wandb_logs = {k: float(v) for k, v in vae_logs.items()}
                    wandb_logs['epoch'] = epoch
                    wandb.log(wandb_logs, step=global_step)

            global_step += 1

        # Epoch summary
        avg_logs = {
            key: float(jnp.mean(jnp.array([log[key] for log in epoch_logs])))
            for key in epoch_logs[0].keys()
        }

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {avg_logs['loss/total']:.4f}")
        print(f"  Reconstruction: {avg_logs['loss/reconstruction']:.4f}")
        print(f"  KL (raw total): {avg_logs['loss/kl_total']:.4f}")
        print(f"  KL (weighted+annealed): {avg_logs['loss/kl_weighted']:.6f}")
        print(f"  Nulling: {avg_logs['loss/nulling']:.4f}")
        print(f"  Orthogonality: {avg_logs['loss/orthogonality']:.4f}")
        if args.weight_perceptual > 0:
            print(f"  Perceptual: {avg_logs['loss/perceptual']:.4f}")
        if args.weight_adversarial > 0:
            print(f"  GAN gen: {avg_logs['loss/gen_adversarial']:.4f}, disc: {avg_logs['loss/patch_disc']:.4f}")
            print(f"  Disc factor: {avg_logs['loss/disc_factor']:.3f}")
        if args.use_contrastive:
            print(f"  Contrastive: {avg_logs.get('loss/contrastive', 0.0):.4f}")
        if args.use_disease_clf:
            print(f"  Disease clf (R11): {avg_logs.get('loss/disease_clf', 0.0):.4f} "
                  f"[cardio={avg_logs.get('loss/disease_clf_cardio', 0.0):.4f}, "
                  f"effusion={avg_logs.get('loss/disease_clf_effusion', 0.0):.4f}]")
        if args.weight_spatial_attn > 0 and args.use_label_attention:
            print(f"  Spatial attn (R12): {avg_logs.get('loss/spatial_attn', 0.0):.4f}")
        if args.kl_warmup_epochs > 0:
            print(f"  KL anneal: {float(kl_anneal):.3f}")

        disent_eval_done = False

        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pkl"

            ckpt_data = {
                'epoch': epoch,
                'global_step': global_step,
                'vae_params': vae_state.params,
                'vae_batch_stats': vae_batch_stats,
                'vae_opt_state': vae_state.opt_state,
                'patch_disc_params': patch_disc_state.params if patch_disc_state else None,
                'patch_disc_opt_state': patch_disc_state.opt_state if patch_disc_state else None,
                # R11: disease discriminability classifier states
                'clf_cardio_params': (
                    clf_cardio_state.params if clf_cardio_state is not None else None
                ),
                'clf_cardio_opt_state': (
                    clf_cardio_state.opt_state if clf_cardio_state is not None else None
                ),
                'clf_effusion_params': (
                    clf_effusion_state.params if clf_effusion_state is not None else None
                ),
                'clf_effusion_opt_state': (
                    clf_effusion_state.opt_state if clf_effusion_state is not None else None
                ),
                'rng': rng,
                'args': vars(args),
            }

            with open(ckpt_path, 'wb') as f:
                f.write(to_bytes(ckpt_data))

            print(f"✓ Saved checkpoint: {ckpt_path}")

        # Generate reconstruction grid
        if args.sample_every > 0 and epoch % args.sample_every == 0:
            print(f"  Generating reconstruction grid...")
            rng, sample_key = jax.random.split(rng)
            x_orig, x_rec = reconstruct_batch(vae_state.params, batch, sample_key)

            grid_img = make_sepvae_recon_grid(
                x_orig, x_rec,
                batch['disease_labels'],
                n_per_class=args.n_samples_per_class
            )

            grid_path = samples_dir / f"recon_epoch{epoch:04d}.png"
            grid_img.save(str(grid_path))
            print(f"  ✓ Saved reconstruction grid: {grid_path}")

            if args.wandb and _WANDB:
                wandb.log(
                    {"samples/recon_grid": wandb.Image(str(grid_path))},
                    step=global_step
                )

            if args.disent_eval_on_sample:
                run_disent_eval(epoch, trigger="sample")
                disent_eval_done = True

            # ── Latent swap grid ──────────────────────────────────────────
            try:
                print(f"  Generating latent swap grid...")
                swap_fig = plot_latent_swap_grid(
                    sepvae, vae_state.params, vae_batch_stats,
                    batch,
                    z_channels_common=args.z_channels_common,
                    z_channels_disease=args.z_channels_disease,
                    epoch=epoch,
                )
                swap_path = diag_dir / f"swap_epoch{epoch:04d}.png"
                swap_fig.savefig(str(swap_path), dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as _plt; _plt.close(swap_fig)
                print(f"  ✓ Saved latent swap grid: {swap_path}")
                if args.wandb and _WANDB:
                    wandb.log(
                        {"diagnostics/swap_grid": wandb.Image(str(swap_path))},
                        step=global_step,
                    )
            except Exception as e:
                print(f"  ⚠ Swap grid failed: {e}")

            # ── Disease attention maps (only when --use_label_attention) ──
            if args.use_label_attention:
                try:
                    print(f"  Generating disease attention maps...")
                    # Encode current batch to get attention maps
                    _B = batch['x_norm'].shape[0]
                    x_all = jnp.concatenate([
                        batch['x_norm'], batch['x_disease1'], batch['x_disease2']
                    ], axis=0)
                    # labels_all: replicate per-class labels to match (3B,) concat order
                    labels_all = jnp.concatenate([
                        jnp.zeros(_B, dtype=jnp.int32),
                        jnp.ones(_B, dtype=jnp.int32),
                        jnp.full(_B, 2, dtype=jnp.int32),
                    ], axis=0)
                    latents_for_attn = encode_batch_for_attn(vae_state.params, x_all)
                    attn_maps_np = {
                        k: np.array(v)
                        for k, v in latents_for_attn['attn_maps'].items()
                    }
                    attn_grid = make_attention_map_grid(
                        np.array(x_all), attn_maps_np, np.array(labels_all),
                        n_per_class=args.n_samples_per_class,
                    )
                    attn_path = diag_dir / f"attn_maps_epoch{epoch:04d}.png"
                    attn_grid.save(str(attn_path))
                    print(f"  ✓ Saved attention map grid: {attn_path}")
                    if args.wandb and _WANDB:
                        wandb.log(
                            {"diagnostics/attn_maps": wandb.Image(str(attn_path))},
                            step=global_step,
                        )

                    # ── GT-bbox / Reconstruction / Attention comparison ───
                    try:
                        print(f"  Generating bbox–recon–attn comparison grid...")
                        cmp_grid = make_bbox_attn_comparison_grid(
                            np.array(x_all),
                            np.array(x_rec),
                            attn_maps_np,
                            bboxes_cardio=np.array(batch['bbox_disease2']),   # (B,4) cardio
                            bboxes_effusion=np.array(batch['bbox_disease1']), # (B,4) effusion
                            B=int(_B),
                            n_per_class=args.n_samples_per_class,
                        )
                        cmp_path = diag_dir / f"bbox_attn_cmp_epoch{epoch:04d}.png"
                        cmp_grid.save(str(cmp_path))
                        print(f"  ✓ Saved comparison grid: {cmp_path}")
                        if args.wandb and _WANDB:
                            wandb.log(
                                {"diagnostics/bbox_attn_comparison": wandb.Image(str(cmp_path))},
                                step=global_step,
                            )
                    except Exception as e:
                        print(f"  ⚠ Bbox–attn comparison grid failed: {e}")

                except Exception as e:
                    print(f"  ⚠ Attention map grid failed: {e}")

            # ── Per-channel KL heatmap ────────────────────────────────────
            try:
                print(f"  Generating per-channel KL heatmap...")
                kl_fig = plot_per_channel_kl_heatmap(
                    sepvae, vae_state.params, vae_batch_stats,
                    batch, epoch=epoch,
                )
                kl_path = diag_dir / f"kl_channels_epoch{epoch:04d}.png"
                kl_fig.savefig(str(kl_path), dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as _plt; _plt.close(kl_fig)
                print(f"  ✓ Saved KL heatmap: {kl_path}")
                if args.wandb and _WANDB:
                    wandb.log(
                        {"diagnostics/kl_heatmap": wandb.Image(str(kl_path))},
                        step=global_step,
                    )
            except Exception as e:
                print(f"  ⚠ KL heatmap failed: {e}")

        # Generate latent manifold visualization
        if args.manifold_every > 0 and epoch % args.manifold_every == 0:
            print(f"  Generating latent manifold plot ({args.manifold_method.upper()})...")
            manifold_path = manifold_dir / f"manifold_epoch{epoch:04d}.png"
            manifold_metrics = save_latent_manifold_plot(
                sepvae,
                vae_state.params,
                vae_batch_stats,
                loader,
                manifold_path,
                max_samples=args.manifold_max_samples,
                method=args.manifold_method,
            )
            print(f"  ✓ Saved latent manifold: {manifold_path}")
            if manifold_metrics:
                metric_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in manifold_metrics.items() if np.isfinite(v)
                )
                if metric_str:
                    print(f"    {metric_str}")

            if args.wandb and _WANDB:
                wandb_payload = {"samples/latent_manifold": wandb.Image(str(manifold_path))}
                for key, value in manifold_metrics.items():
                    if np.isfinite(value):
                        wandb_payload[f"manifold/{key}"] = float(value)
                wandb.log(wandb_payload, step=global_step)

            # ── Head L2-norm bar chart ────────────────────────────────────
            # Collect latents once and pass to head-norm plot (no extra forward
            # passes beyond what the manifold already implicitly triggers).
            try:
                print(f"  Generating head L2-norm bar chart...")
                variables_for_norm = {"params": vae_state.params}
                if vae_batch_stats:
                    variables_for_norm["batch_stats"] = vae_batch_stats
                lc, lcard, leff, lbls = _collect_latents_from_triplet_loader(
                    sepvae, variables_for_norm, loader,
                    max_samples=args.manifold_max_samples,
                )
                if len(lbls) >= 6:
                    norm_fig = plot_head_norm_by_class(lc, lcard, leff, lbls, epoch=epoch)
                    norm_path = diag_dir / f"head_norms_epoch{epoch:04d}.png"
                    norm_fig.savefig(str(norm_path), dpi=150, bbox_inches="tight")
                    import matplotlib.pyplot as _plt; _plt.close(norm_fig)
                    print(f"  ✓ Saved head norm chart: {norm_path}")
                    if args.wandb and _WANDB:
                        wandb.log(
                            {"diagnostics/head_norms": wandb.Image(str(norm_path))},
                            step=global_step,
                        )
            except Exception as e:
                print(f"  ⚠ Head norm chart failed: {e}")

        # Evaluate disentanglement metrics beyond manifold/PCA silhouette
        if (
            args.disent_metrics_every > 0
            and epoch % args.disent_metrics_every == 0
            and not disent_eval_done
        ):
            run_disent_eval(epoch, trigger="periodic")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / "checkpoint_final.pkl"
    ckpt_data = {
        'epoch': args.epochs,
        'global_step': global_step,
        'vae_params': vae_state.params,
        'vae_batch_stats': vae_batch_stats,
        'vae_opt_state': vae_state.opt_state,
        'patch_disc_params': patch_disc_state.params if patch_disc_state else None,
        'patch_disc_opt_state': patch_disc_state.opt_state if patch_disc_state else None,
        'rng': rng,
        'args': vars(args),
    }

    with open(final_ckpt_path, 'wb') as f:
        f.write(to_bytes(ckpt_data))

    print(f"✓ Saved final checkpoint: {final_ckpt_path}")

    if args.wandb and _WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
