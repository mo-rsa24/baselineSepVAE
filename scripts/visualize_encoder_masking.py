"""
visualize_encoder_masking.py — Shows what the MiniSepVAE encoder sees and
how head-nulling masks each latent head based on the sample's disease label.

Layout per sample (one row per image):
  [Input] | [mu_common ch0..3] | [mu_cardio ch0,1 | MASK] | [mu_effusion ch0,1 | MASK]
           +---------------------------------------------------------------------+
           | z_after_nulling: [z_common | z_cardio_masked | z_effusion_masked]  |
           +---------------------------------------------------------------------+

Color convention
  - Spatial mu maps: diverging RdBu colormap (blue=negative, red=positive)
  - Nulling mask: green border = ACTIVE head, red border = NULLED head
  - After-nulling maps: same RdBu, but near-zero for inactive heads

Usage (random/untrained weights — good for verifying shapes):
    python -m scripts.visualize_encoder_masking --n_samples 6

Usage (trained checkpoint):
    python -m scripts.visualize_encoder_masking \\
        --ckpt_path runs/my_run/ckpt_epoch30.bin \\
        --n_samples 9 --out_dir vis_masking
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---- project root on path ----
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.mini_sep_vae import MiniSepVAE, MiniSepVAEEncoder, MiniCNNBackbone
from datasets.SyntheticCXR import SyntheticCXRTripletDataset


# ============================================================================
# Checkpoint loading (optional — falls back to random init)
# ============================================================================

def _load_params(ckpt_path: str, model: MiniSepVAE, dummy_x, dummy_labels, key):
    """Load params from a flax msgpack checkpoint, or return freshly-inited params."""
    if ckpt_path is None:
        print("[INFO] No checkpoint supplied — using random initialisation.")
        variables = model.init(key, dummy_x, dummy_labels, key=key)
        return variables['params']

    from flax.serialization import from_bytes
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        blob = f.read()
    # The training script saves the full state dict; we only need VAE params.
    # Try common nesting patterns used by train_mini_sepvae / train_sep_vae.
    dummy_variables = model.init(key, dummy_x, dummy_labels, key=key)
    try:
        restored = from_bytes(dummy_variables['params'], blob)
        return restored
    except Exception:
        pass
    try:
        # Saved as {'params': ..., ...}
        import msgpack
        raw = msgpack.unpackb(blob, raw=False)
        params = raw.get('params', raw)
        restored = from_bytes(dummy_variables['params'], msgpack.packb(params))
        return restored
    except Exception as e:
        print(f"[WARN] Could not restore checkpoint ({e}). Using random params.")
        return dummy_variables['params']


# ============================================================================
# Nulling masks (pure numpy — mirrors apply_head_nulling logic)
# ============================================================================

def _nulling_masks(labels: np.ndarray):
    """
    Returns (mask_cardio, mask_effusion) as float arrays of shape (B,).
      mask_cardio[i]   = 1.0  iff  labels[i] == 2  (Cardiomegaly active)
      mask_effusion[i] = 1.0  iff  labels[i] == 1  (Effusion active)
    """
    mask_cardio   = (labels == 2).astype(np.float32)
    mask_effusion = (labels == 1).astype(np.float32)
    return mask_cardio, mask_effusion


# ============================================================================
# Encode a batch and extract per-head mu maps + z_after_nulling
# ============================================================================

def encode_batch(model: MiniSepVAE, params, x_jax, labels_jax, sigma_inactive=0.05):
    """
    Run encoder + nulling and return numpy arrays for visualisation.

    Returns
    -------
    mu_common    : (B, 16, 16, 4)
    mu_cardio    : (B, 16, 16, 2)
    mu_effusion  : (B, 16, 16, 2)
    z_common     : (B, 16, 16, 4)   — always active
    z_cardio_m   : (B, 16, 16, 2)   — after nulling
    z_effusion_m : (B, 16, 16, 2)   — after nulling
    z_concat     : (B, 16, 16, 8)
    backbone_feat: (B, 16, 16, 128) — raw backbone output (first 3 channels shown)
    """
    key = jax.random.PRNGKey(0)

    # --- Run encoder to get latents dict ---
    latents_dict = model.apply(
        {'params': params}, x_jax,
        method=model.encode,
    )

    mu_c,  logvar_c  = latents_dict['common']
    mu_d1, logvar_d1 = latents_dict['cardiomegaly']
    mu_d2, logvar_d2 = latents_dict['effusion']

    # --- Reparameterise ---
    def reparameterise(mu, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * std

    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    z_c  = reparameterise(mu_c,  logvar_c,  k0)
    z_d1 = reparameterise(mu_d1, logvar_d1, k1)
    z_d2 = reparameterise(mu_d2, logvar_d2, k2)

    # --- Soft nulling (mirrors apply_head_nulling) ---
    mask_cardio   = (labels_jax == 2).astype(jnp.float32)[:, None, None, None]
    mask_effusion = (labels_jax == 1).astype(jnp.float32)[:, None, None, None]

    noise_d1 = sigma_inactive * jax.random.normal(k3, z_d1.shape)
    noise_d2 = sigma_inactive * jax.random.normal(k4, z_d2.shape)

    z_d1_masked = z_d1 * mask_cardio   + noise_d1 * (1.0 - mask_cardio)
    z_d2_masked = z_d2 * mask_effusion + noise_d2 * (1.0 - mask_effusion)

    z_concat = jnp.concatenate([z_c, z_d1_masked, z_d2_masked], axis=-1)

    # --- Backbone features via standalone apply ---
    backbone_params = params['encoder']['backbone']
    backbone_model  = MiniCNNBackbone()
    backbone_feat   = backbone_model.apply(
        {'params': backbone_params}, x_jax, train=False
    )

    return (
        np.asarray(mu_c),
        np.asarray(mu_d1),
        np.asarray(mu_d2),
        np.asarray(z_c),
        np.asarray(z_d1_masked),
        np.asarray(z_d2_masked),
        np.asarray(z_concat),
        np.asarray(backbone_feat),
    )


# ============================================================================
# Plotting helpers
# ============================================================================

_CMAP_LATENT  = "RdBu_r"   # diverging: blue=negative, red=positive
_CMAP_BACKBONE = "viridis"
_LABEL_NAMES  = {0: "Normal", 1: "Effusion", 2: "Cardiomegaly"}
_HEAD_COLORS  = {
    'common':       '#4C72B0',  # blue
    'cardiomegaly': '#DD8452',  # orange
    'effusion':     '#55A868',  # green
}


def _imshow_latent(ax, data, title="", vmax=None, active=True):
    """Show a 2-D spatial map with symmetric colour scale."""
    if vmax is None:
        vmax = max(abs(data).max(), 1e-6)
    im = ax.imshow(data, cmap=_CMAP_LATENT, vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis('off')
    # Border: green=active, red=nulled
    colour = _HEAD_COLORS['common'] if active is None else (
        '#27AE60' if active else '#E74C3C'
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(2)
        spine.set_visible(True)
    return im


def _make_mask_annotation(ax, is_active, head_name):
    """Draw ACTIVE / NULLED badge on an axes."""
    tag  = "ACTIVE" if is_active else "NULLED"
    bg   = '#27AE60'  if is_active else '#E74C3C'
    ax.text(0.5, 0.5, f"{head_name}\n{tag}",
            ha='center', va='center', fontsize=8, fontweight='bold',
            color='white', transform=ax.transAxes,
            bbox=dict(facecolor=bg, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.axis('off')


def visualize_sample(fig, gs_row, sample_idx,
                     x_np, label,
                     mu_c, mu_d1, mu_d2,
                     z_c, z_d1_m, z_d2_m,
                     before_after=True):
    """
    Fill one row of the figure grid for a single sample.

    Column layout (16 sub-columns):
      0        : input image
      1        : mask badge – common
      2..5     : mu_common channels 0-3
      6        : mask badge – cardiomegaly
      7..8     : mu_cardio channels 0-1
      9        : separator / z_cardio after nulling (mean)
      10       : mask badge – effusion
      11..12   : mu_effusion channels 0-1
      13       : z_effusion after nulling (mean)
      14       : z_concat energy (mean across all 8 channels)
      15       : label annotation
    """
    is_cardio_active   = (label == 2)
    is_effusion_active = (label == 1)

    col = 0

    # ---- input image ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    img_show = (np.asarray(x_np).squeeze() + 1.0) / 2.0   # [-1,1] → [0,1]
    ax.imshow(np.clip(img_show, 0, 1), cmap='gray', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_title(_LABEL_NAMES[label], fontsize=7, pad=2,
                 color=_HEAD_COLORS.get('common', 'k'))
    ax.axis('off')

    # ---- COMMON head badge ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    _make_mask_annotation(ax, True, 'Common')

    # ---- mu_common channels ----
    vmax_c = max(np.abs(mu_c).max(), 1e-6)
    for ch in range(mu_c.shape[-1]):
        ax = fig.add_subplot(gs_row[col]); col += 1
        _imshow_latent(ax, mu_c[..., ch], title=f"μ_c ch{ch}",
                       vmax=vmax_c, active=None)

    # ---- CARDIOMEGALY head badge ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    _make_mask_annotation(ax, is_cardio_active, 'Cardio')

    # ---- mu_cardio channels ----
    vmax_d1 = max(np.abs(mu_d1).max(), 1e-6)
    for ch in range(mu_d1.shape[-1]):
        ax = fig.add_subplot(gs_row[col]); col += 1
        _imshow_latent(ax, mu_d1[..., ch], title=f"μ_d1 ch{ch}",
                       vmax=vmax_d1, active=is_cardio_active)

    # ---- z_cardio AFTER nulling ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    z_d1_mean = z_d1_m.mean(axis=-1)
    vmax_z1 = max(np.abs(z_d1_mean).max(), 1e-6)
    _imshow_latent(ax, z_d1_mean, title="z_d1 (post-null)",
                   vmax=vmax_z1, active=is_cardio_active)

    # ---- EFFUSION head badge ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    _make_mask_annotation(ax, is_effusion_active, 'Effusion')

    # ---- mu_effusion channels ----
    vmax_d2 = max(np.abs(mu_d2).max(), 1e-6)
    for ch in range(mu_d2.shape[-1]):
        ax = fig.add_subplot(gs_row[col]); col += 1
        _imshow_latent(ax, mu_d2[..., ch], title=f"μ_d2 ch{ch}",
                       vmax=vmax_d2, active=is_effusion_active)

    # ---- z_effusion AFTER nulling ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    z_d2_mean = z_d2_m.mean(axis=-1)
    vmax_z2 = max(np.abs(z_d2_mean).max(), 1e-6)
    _imshow_latent(ax, z_d2_mean, title="z_d2 (post-null)",
                   vmax=vmax_z2, active=is_effusion_active)

    # ---- z_concat energy ----
    ax = fig.add_subplot(gs_row[col]); col += 1
    z_energy = np.concatenate([z_c, z_d1_m, z_d2_m], axis=-1).mean(axis=-1)
    vmax_e = max(np.abs(z_energy).max(), 1e-6)
    _imshow_latent(ax, z_energy, title="z_concat energy", vmax=vmax_e, active=None)


# ============================================================================
# Main
# ============================================================================

N_COLS = 15   # 1 input + 1 badge + 4 common + 1 badge + 2 cardio + 1 z_d1 + 1 badge + 2 effusion + 1 z_d2 + 1 z_concat


def main():
    p = argparse.ArgumentParser(description="Visualise MiniSepVAE encoder masking")
    p.add_argument('--ckpt_path',  type=str,  default=None,
                   help='Path to .bin checkpoint from train_mini_sepvae.py')
    p.add_argument('--n_samples',  type=int,  default=9,
                   help='Number of samples to visualise (multiple of 3 is best)')
    p.add_argument('--sigma_inactive', type=float, default=0.05)
    p.add_argument('--z_channels_common',  type=int, default=4)
    p.add_argument('--z_channels_disease', type=int, default=2)
    p.add_argument('--use_label_attention', action='store_true')
    p.add_argument('--out_dir',   type=str, default='vis_encoder_masking')
    p.add_argument('--seed',      type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prng_key = jax.random.PRNGKey(args.seed)

    # ---- Build model ----
    model = MiniSepVAE(
        z_channels_common=args.z_channels_common,
        z_channels_disease=args.z_channels_disease,
        sigma_inactive=args.sigma_inactive,
        use_label_attention=args.use_label_attention,
    )

    # ---- Sample data from SyntheticCXR ----
    n = args.n_samples
    dataset = SyntheticCXRTripletDataset(n_samples=n, img_size=128, seed=args.seed)

    # Unpack triplets: one Normal, one Effusion, one Cardiomegaly per triplet
    images, labels_list = [], []
    for i in range(min(n // 3 + 1, len(dataset))):
        item = dataset[i]
        for img_key, lbl in [('x_norm', 0), ('x_disease1', 1), ('x_disease2', 2)]:
            if len(images) >= n:
                break
            img = np.asarray(item[img_key])   # (1, 128, 128) torch tensor
            images.append(img)
            labels_list.append(lbl)
        if len(images) >= n:
            break

    images = np.stack(images[:n], axis=0)            # (N, 1, 128, 128)
    labels = np.array(labels_list[:n], dtype=np.int32)

    # JAX convention: (B, H, W, C)
    x_jax = jnp.asarray(images.transpose(0, 2, 3, 1))  # (N, 128, 128, 1)
    labels_jax = jnp.asarray(labels)

    # ---- Init / load params ----
    key_init = jax.random.PRNGKey(42)
    dummy_x  = jnp.zeros((1, 128, 128, 1))
    dummy_l  = jnp.zeros((1,), dtype=jnp.int32)
    params   = _load_params(args.ckpt_path, model, dummy_x, dummy_l, key_init)

    # ---- Encode ----
    print(f"[INFO] Encoding {n} samples ...")
    mu_c, mu_d1, mu_d2, z_c, z_d1_m, z_d2_m, z_concat, backbone = encode_batch(
        model, params, x_jax, labels_jax, sigma_inactive=args.sigma_inactive
    )

    # ---- Build figure ----
    row_height = 1.8
    col_width  = 1.2
    fig_w = N_COLS * col_width
    fig_h = n * row_height + 1.0   # +1 for header

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
    fig.suptitle(
        "MiniSepVAE Encoder Masking\n"
        "Green border = ACTIVE head  |  Red border = NULLED head\n"
        "Columns: input | common (4ch) | cardiomegaly (2ch + z_masked) | effusion (2ch + z_masked) | z_concat energy",
        fontsize=9, y=1.0,
    )

    # One GridSpec row per sample
    gs_outer = fig.add_gridspec(n, 1, hspace=0.6)

    for i in range(n):
        gs_row = gs_outer[i].subgridspec(1, N_COLS, wspace=0.1)
        visualize_sample(
            fig, gs_row, sample_idx=i,
            x_np=images[i],
            label=int(labels[i]),
            mu_c=mu_c[i], mu_d1=mu_d1[i], mu_d2=mu_d2[i],
            z_c=z_c[i], z_d1_m=z_d1_m[i], z_d2_m=z_d2_m[i],
        )

    out_path = out_dir / "encoder_masking.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")

    # ---- Also save a per-head comparison across all samples ----
    _save_perhead_comparison(
        out_dir, images, labels, mu_c, mu_d1, mu_d2, z_d1_m, z_d2_m
    )


def _save_perhead_comparison(out_dir, images, labels,
                              mu_c, mu_d1, mu_d2, z_d1_m, z_d2_m):
    """
    Second figure: 3 columns (Normal / Effusion / Cardiomegaly), each showing
    the mean activation of each head before and after nulling.
    """
    label_groups = {0: [], 1: [], 2: []}
    for i, lbl in enumerate(labels):
        label_groups[int(lbl)].append(i)

    fig, axes = plt.subplots(
        3, 3,
        figsize=(12, 9),
        squeeze=False,
    )
    fig.suptitle(
        "Per-head mean activation  |  before vs after nulling\n"
        "Averaged over all spatial positions and channels",
        fontsize=10,
    )
    row_labels = [
        "μ_common (ch mean)",
        "μ_d1 cardio  (ch mean)  →  z after null",
        "μ_d2 effusion (ch mean)  →  z after null",
    ]
    col_titles = ["Normal (lbl=0)", "Effusion (lbl=1)", "Cardiomegaly (lbl=2)"]

    for col_idx, lbl in enumerate([0, 1, 2]):
        idxs = label_groups[lbl]
        if not idxs:
            for r in range(3):
                axes[r][col_idx].axis('off')
                axes[r][col_idx].set_title(f"{col_titles[col_idx]}\n(no samples)", fontsize=8)
            continue

        # mean across samples in this group
        mc_mean  = mu_c[idxs].mean(axis=(0, -1))    # (16, 16)
        md1_mean = mu_d1[idxs].mean(axis=(0, -1))
        md2_mean = mu_d2[idxs].mean(axis=(0, -1))
        zd1_mean = z_d1_m[idxs].mean(axis=(0, -1))
        zd2_mean = z_d2_m[idxs].mean(axis=(0, -1))

        def _show(ax, before, after, title, is_active):
            # Side-by-side before / after in a single axes using an RGB trick
            vmax = max(np.abs(before).max(), np.abs(after).max(), 1e-6)
            combined = np.hstack([before, after])
            im = ax.imshow(combined, cmap=_CMAP_LATENT, vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.axvline(x=before.shape[1] - 0.5, color='white', linewidth=1.5, linestyle='--')
            ax.text(before.shape[1] / 2, -1.2, "before null", ha='center',
                    fontsize=6, color='gray', transform=ax.get_xaxis_transform())
            ax.text(before.shape[1] * 1.5, -1.2, "after null", ha='center',
                    fontsize=6, color='gray', transform=ax.get_xaxis_transform())
            border_col = '#27AE60' if is_active else '#E74C3C'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col); spine.set_linewidth(2.5)
                spine.set_visible(True)
            ax.set_xticks([]); ax.set_yticks([])
            tag = "ACTIVE" if is_active else "NULLED"
            ax.set_title(f"{col_titles[col_idx]}\n{title} [{tag}]",
                         fontsize=7, color=border_col)

        # Row 0: common (always active)
        _show(axes[0][col_idx], mc_mean, mc_mean, row_labels[0], is_active=True)

        # Row 1: cardiomegaly (active only for lbl=2)
        _show(axes[1][col_idx], md1_mean, zd1_mean, row_labels[1],
              is_active=(lbl == 2))

        # Row 2: effusion (active only for lbl=1)
        _show(axes[2][col_idx], md2_mean, zd2_mean, row_labels[2],
              is_active=(lbl == 1))

    plt.tight_layout()
    out_path = out_dir / "perhead_comparison.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")


if __name__ == '__main__':
    main()
