"""
Latent traversal with MedSAM cardiac mask overlay — SepVAE V2.

For each Cardiomegaly image:
  1. Encode with the SepVAE checkpoint.
  2. Scale z_disease from alpha=0 (anatomy-only) to alpha_max in equal steps.
  3. Decode each step → run MedSAM with the original cardiac bbox as prompt.
  4. Produce a 3-row figure per image:
       Row 0 — grayscale reconstructions at each alpha
       Row 1 — coral mask overlay at each alpha
       Row 2 — mask-area and cardiac-width curves vs alpha
  5. Produce a summary figure with mask-area curves for all images together.

This directly validates whether z_disease encodes a controllable cardiac-size dial:
  If YES → mask area grows monotonically as alpha increases.
  If NO  → mask area is flat or erratic → z_disease is entangled / weak.

Requirements:
  conda run -n jaxstack ... (has JAX, flax, torch ≥2.3, transformers, safetensors)

Usage:
  conda run -n jaxstack python scripts/latent_traversal_medsam.py \\
      --checkpoint runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch0120.pkl \\
      --csv_path   /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \\
      --dicom_dir  /datasets/mmolefe/vinbigdata/cache_npy \\
      --output_dir results/traversal_medsam_ep120 \\
      --n_images   4
"""

import argparse
import os
import sys
from pathlib import Path

# Prevent JAX from consuming all GPU memory — leave room for MedSAM on GPU.
# MedSAM will run on CPU by default (--medsam_device cuda overrides).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flax.serialization import msgpack_restore

# ── dependency guard ──────────────────────────────────────────────────────────
try:
    from transformers import SamModel, SamProcessor
except ImportError:
    print(
        "\n[ERROR] transformers not installed in this environment.\n"
        "  conda run -n jaxstack pip install transformers accelerate\n",
        file=sys.stderr,
    )
    sys.exit(1)

import jax
import jax.numpy as jnp

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from datasets.VinBigData import VinBigDataPairDataset, jax_pair_collate_fn
from models.sep_vae_v2 import SepVAEV2

MEDSAM_ID = "wanglab/medsam-vit-base"


# ─────────────────────────────────────────────────────────────────────────────
# VAE model loading  (identical to latent_traversal.py)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_params(target, source):
    if not isinstance(target, dict):
        return jnp.array(source)
    result = {}
    for k, v in target.items():
        if k in source:
            result[k] = (
                _merge_params(v, source[k])
                if isinstance(v, dict) and isinstance(source[k], dict)
                else jnp.array(source[k])
            )
        else:
            result[k] = v
    return result


def load_vae(ckpt_path, img_size=256, z_common=16, z_disease=16,
             attn_query_dim=256, attn_heads=4, bbox_query_mix=0.7,
             decoder_res_blocks=3):
    model = SepVAEV2(
        z_channels_common=z_common,
        z_channels_disease=z_disease,
        query_dim=attn_query_dim,
        attn_heads=attn_heads,
        use_bbox_cross_attn=True,
        bbox_query_mix=bbox_query_mix,
        decoder_res_blocks=decoder_res_blocks,
    )
    dummy_x      = jnp.ones((1, img_size, img_size, 1))
    dummy_labels = jnp.array([0])
    init_rng     = jax.random.PRNGKey(0)
    fresh_vars   = model.init(init_rng, dummy_x, dummy_labels, key=init_rng)
    fresh_params = jax.tree_util.tree_map(jnp.array, fresh_vars['params'])

    with open(ckpt_path, 'rb') as f:
        ckpt = msgpack_restore(f.read())

    ckpt_params = jax.tree_util.tree_map(
        jnp.array, ckpt.get('ema_params', ckpt['vae_params']))
    params = _merge_params(fresh_params, ckpt_params)
    epoch  = int(ckpt.get('epoch', 0))
    print(f"VAE checkpoint loaded: epoch {epoch}  ({ckpt_path})")
    return model, params, epoch


def encode_one(model, params, x, bbox, has_bbox):
    ld   = model.apply({'params': params}, x, bbox=bbox, has_bbox=has_bbox,
                       method=model.encode)
    mu_c = ld['common'][0]
    mu_d = ld['cardiomegaly'][0]
    skip = ld.get('skip_feats')
    return mu_c, mu_d, skip


def decode_one(model, params, z_c, z_d, skip_feats):
    z = jnp.concatenate([z_c, z_d], axis=-1)
    return model.apply({'params': params}, z, skip_feats, method=model.decode)


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM loading  (same fallback logic as test_medsam_cardiac.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_medsam(model_id: str, device: torch.device):
    """Load MedSAM with safetensors fallback for torch < 2.6."""
    import shutil, tempfile
    from huggingface_hub import hf_hub_download

    try:
        model     = SamModel.from_pretrained(model_id).to(device).eval()
        processor = SamProcessor.from_pretrained(model_id)
        return model, processor
    except ValueError as e:
        if "CVE-2025-32434" not in str(e) and "upgrade torch" not in str(e):
            raise

    print("  [info] Converting MedSAM .bin → safetensors (torch < 2.6)…")
    bin_path  = hf_hub_download(model_id, "pytorch_model.bin")
    cfg_path  = hf_hub_download(model_id, "config.json")
    prep_path = hf_hub_download(model_id, "preprocessor_config.json")

    from safetensors.torch import save_file as st_save
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    tmpdir = Path(tempfile.mkdtemp(prefix="medsam_st_"))
    st_save(state_dict, str(tmpdir / "model.safetensors"))
    shutil.copy(cfg_path,  tmpdir / "config.json")
    shutil.copy(prep_path, tmpdir / "preprocessor_config.json")

    model     = SamModel.from_pretrained(str(tmpdir)).to(device).eval()
    processor = SamProcessor.from_pretrained(str(tmpdir))
    print(f"  [info] Safetensors written to {tmpdir}")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM inference on a single frame
# ─────────────────────────────────────────────────────────────────────────────

def segment_frame(
    sam_model, sam_proc, frame_01: np.ndarray,
    bbox_norm: tuple, device: torch.device,
) -> np.ndarray:
    """
    Run MedSAM on one grayscale frame.

    Args:
        frame_01:   float32 (H, W) in [0, 1]
        bbox_norm:  (x0, y0, x1, y1) normalised [0, 1]

    Returns:
        mask: bool (H, W)
    """
    h, w = frame_01.shape
    x0, y0, x1, y1 = bbox_norm

    img_u8  = (frame_01 * 255).clip(0, 255).astype(np.uint8)
    rgb_img = np.stack([img_u8] * 3, axis=-1)
    bbox_px = [[x0 * w, y0 * h, x1 * w, y1 * h]]

    inputs = sam_proc(images=rgb_img, input_boxes=[bbox_px], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)

    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks[0][0][0].numpy()   # bool (H, W)


def mask_stats(mask: np.ndarray) -> dict:
    if not mask.any():
        return {"area": 0, "width": 0, "height": 0}
    cols = np.where(np.any(mask, axis=0))[0]
    rows = np.where(np.any(mask, axis=1))[0]
    return {
        "area":   int(mask.sum()),
        "width":  int(cols[-1] - cols[0] + 1),
        "height": int(rows[-1] - rows[0] + 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-image traversal + segmentation
# ─────────────────────────────────────────────────────────────────────────────

def traverse_and_segment(
    vae_model, vae_params,
    sam_model, sam_proc, sam_device,
    x_cardio, bbox_ca, has_bbox,
    alphas, img_idx,
):
    """
    For one image (img_idx):
    Returns:
        frames:  list of np (H, W) float32 — one per alpha
        masks:   list of np (H, W) bool    — MedSAM output per alpha
        stats:   list of dict {area, width, height} per alpha
        orig:    np (H, W) float32 — input image in [0,1]
        bbox_norm: (x0,y0,x1,y1)
    """
    z_c, z_d, skip = encode_one(vae_model, vae_params, x_cardio, bbox_ca, has_bbox)

    zc = z_c[img_idx:img_idx+1]
    zd = z_d[img_idx:img_idx+1]
    sf = ({k: v[img_idx:img_idx+1] for k, v in skip.items()}
          if skip is not None else None)

    orig = np.clip((np.array(x_cardio)[img_idx, :, :, 0] + 1.0) / 2.0, 0, 1)
    b    = np.array(bbox_ca)[img_idx]   # [x0,y0,x1,y1] normalised
    bbox_norm = tuple(b.tolist())

    frames, masks, stats = [], [], []
    for alpha in alphas:
        out   = decode_one(vae_model, vae_params, zc, alpha * zd, sf)
        frame = np.clip(np.array(out)[0, :, :, 0], 0, 1)
        mask  = segment_frame(sam_model, sam_proc, frame, bbox_norm, sam_device)
        s     = mask_stats(mask)
        frames.append(frame)
        masks.append(mask)
        stats.append(s)
        print(f"    α={alpha:.2f}  area={s['area']:,}  w={s['width']}  h={s['height']}")

    return frames, masks, stats, orig, bbox_norm


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_CORAL_RGBA = (0.95, 0.35, 0.20, 0.42)

def _overlay(frame, mask):
    """Return RGBA overlay array (H, W, 4)."""
    ov = np.zeros((*frame.shape, 4), dtype=np.float32)
    ov[mask, :] = _CORAL_RGBA
    return ov


def save_image_figure(
    frames, masks, stats, orig, bbox_norm, alphas,
    epoch, img_idx, output_dir: Path,
):
    """3-row figure: raw frames | mask overlay | area/width curve."""
    n_alpha  = len(alphas)
    n_cols   = n_alpha + 1   # +1 for original
    cell_w   = 2.0
    cell_h   = 2.2

    fig = plt.figure(figsize=(n_cols * cell_w, 3 * cell_h + 1.0))
    gs  = fig.add_gridspec(
        3, n_cols,
        height_ratios=[1, 1, 0.9],
        hspace=0.35, wspace=0.04,
    )

    x0, y0, x1, y1 = bbox_norm
    H = orig.shape[0]

    def _draw_bbox(ax):
        rect = mpatches.Rectangle(
            (x0 * H, y0 * H), (x1 - x0) * H, (y1 - y0) * H,
            linewidth=1.5, edgecolor='lime', facecolor='none',
        )
        ax.add_patch(rect)

    # ── Row 0: raw grayscale frames ───────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(orig, cmap='gray', vmin=0, vmax=1)
    _draw_bbox(ax_orig)
    ax_orig.set_title('Original\n(input)', fontsize=7.5)
    ax_orig.axis('off')

    for ci, (alpha, frame) in enumerate(zip(alphas, frames)):
        ax = fig.add_subplot(gs[0, ci + 1])
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        label = f'α={alpha:.2f}'
        if abs(alpha) < 1e-6:
            label += '\n(anatomy)'
        elif abs(alpha - 1.0) < 1e-6:
            label += '\n(z_d=1)'
        ax.set_title(label, fontsize=7.5)
        ax.axis('off')

    # ── Row 1: mask overlay ───────────────────────────────────────────────────
    # Blank cell under original
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax_blank.set_title('(prompt bbox)', fontsize=7.5)
    _draw_bbox(ax_blank)
    ax_blank.axis('off')

    for ci, (alpha, frame, mask) in enumerate(zip(alphas, frames, masks)):
        ax = fig.add_subplot(gs[1, ci + 1])
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.imshow(_overlay(frame, mask))
        ax.contour(mask.astype(float), levels=[0.5], colors=['#FF5733'], linewidths=[1.0])
        ax.set_title(f'area={stats[ci]["area"]:,}', fontsize=7.0)
        ax.axis('off')

    # ── Row 2: area + width curves ────────────────────────────────────────────
    ax_curve = fig.add_subplot(gs[2, :])
    areas  = [s['area']  for s in stats]
    widths = [s['width'] for s in stats]

    color_area  = '#E84040'
    color_width = '#2080C8'

    ax_curve.plot(alphas, areas,  'o-', color=color_area,  lw=1.8, ms=5, label='Mask area (px)')
    ax_curve.set_ylabel('Mask area (px)', color=color_area, fontsize=8)
    ax_curve.tick_params(axis='y', labelcolor=color_area, labelsize=7)
    ax_curve.set_xlabel('α (z_disease scale factor)', fontsize=8)
    ax_curve.tick_params(axis='x', labelsize=7)
    ax_curve.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.6, label='α=1 (original)')

    ax2 = ax_curve.twinx()
    ax2.plot(alphas, widths, 's--', color=color_width, lw=1.5, ms=4, label='Cardiac width (px)')
    ax2.set_ylabel('Cardiac width (px)', color=color_width, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=color_width, labelsize=7)

    lines1, labels1 = ax_curve.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_curve.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    ax_curve.set_title(
        'Does mask area grow with α?  '
        '(monotonic ↑ = z_disease encodes cardiac size)',
        fontsize=8,
    )

    fig.suptitle(
        f'MedSAM traversal — Cardiomegaly image #{img_idx}  |  epoch {epoch}',
        fontsize=10, y=1.01,
    )

    out_path = output_dir / f'traversal_medsam_ep{epoch:04d}_img{img_idx}.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path


def save_summary_figure(all_stats, alphas, epoch, output_dir: Path, n_images: int):
    """One figure: mask-area curves for all images, mean ± std band."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, n_images))

    area_matrix  = np.array([[s['area']  for s in stats] for stats in all_stats])  # (N, A)
    width_matrix = np.array([[s['width'] for s in stats] for stats in all_stats])

    for i, (areas, widths) in enumerate(zip(area_matrix, width_matrix)):
        axes[0].plot(alphas, areas,  'o-', color=colors[i], lw=1.5, ms=4,
                     alpha=0.75, label=f'img {i}')
        axes[1].plot(alphas, widths, 's-', color=colors[i], lw=1.5, ms=4,
                     alpha=0.75, label=f'img {i}')

    # Mean ± std band
    if n_images > 1:
        mean_a = area_matrix.mean(0);  std_a = area_matrix.std(0)
        mean_w = width_matrix.mean(0); std_w = width_matrix.std(0)
        axes[0].plot(alphas, mean_a, 'k-', lw=2.5, label='mean')
        axes[0].fill_between(alphas, mean_a - std_a, mean_a + std_a,
                              color='k', alpha=0.12)
        axes[1].plot(alphas, mean_w, 'k-', lw=2.5, label='mean')
        axes[1].fill_between(alphas, mean_w - std_w, mean_w + std_w,
                              color='k', alpha=0.12)

    for ax, ylabel, title in zip(
        axes,
        ['Mask area (px)', 'Cardiac width (px)'],
        ['Mask area vs α', 'Cardiac width vs α'],
    ):
        ax.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('α (z_disease scale factor)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f'MedSAM cardiac mask vs z_disease scale  |  epoch {epoch}  |  {n_images} images\n'
        'Monotonic ↑ with α validates z_disease as a cardiac-size dial',
        fontsize=9,
    )
    plt.tight_layout()

    out = output_dir / f'traversal_medsam_summary_ep{epoch:04d}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Latent traversal + MedSAM cardiac mask overlay")
    p.add_argument('--checkpoint',         required=True)
    p.add_argument('--csv_path',           required=True)
    p.add_argument('--dicom_dir',          required=True)
    p.add_argument('--output_dir',         required=True)
    p.add_argument('--n_images',           type=int,   default=4)
    p.add_argument('--n_alphas',           type=int,   default=9,
                   help='Number of alpha steps (includes 0 and alpha_max).')
    p.add_argument('--alpha_min',          type=float, default=0.0)
    p.add_argument('--alpha_max',          type=float, default=2.0)
    p.add_argument('--img_size',           type=int,   default=256)
    p.add_argument('--z_common',           type=int,   default=16)
    p.add_argument('--z_disease',          type=int,   default=16)
    p.add_argument('--attn_query_dim',     type=int,   default=256)
    p.add_argument('--attn_heads',         type=int,   default=4)
    p.add_argument('--bbox_query_mix',     type=float, default=0.7)
    p.add_argument('--decoder_res_blocks', type=int,   default=3)
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--medsam_device',      type=str,   default='cpu',
                   help='Device for MedSAM: cpu or cuda. Default: cpu (safe with JAX GPU).')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load VAE ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading SepVAE checkpoint…")
    vae_model, vae_params, epoch = load_vae(
        args.checkpoint,
        img_size=args.img_size,
        z_common=args.z_common,
        z_disease=args.z_disease,
        attn_query_dim=args.attn_query_dim,
        attn_heads=args.attn_heads,
        bbox_query_mix=args.bbox_query_mix,
        decoder_res_blocks=args.decoder_res_blocks,
    )

    # ── Load MedSAM ───────────────────────────────────────────────────────────
    sam_device = torch.device(args.medsam_device)
    print(f"\n[2/4] Loading MedSAM on {sam_device}…")
    sam_model, sam_proc = load_medsam(MEDSAM_ID, sam_device)
    print("MedSAM loaded.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[3/4] Loading dataset…")
    dataset = VinBigDataPairDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        use_cache=True,
        deterministic_pairs=True,
        pair_seed=args.seed,
    )
    rng     = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=args.n_images, replace=False)).tolist()
    subset  = Subset(dataset, indices)
    loader  = DataLoader(
        subset,
        batch_size=args.n_images,
        shuffle=False,
        collate_fn=jax_pair_collate_fn,
        num_workers=0,
        drop_last=False,
    )
    batch    = next(iter(loader))
    x_cardio = jnp.array(batch['x_disease1'].permute(0, 2, 3, 1).numpy())
    bbox_ca  = jnp.array(batch['bbox_disease1'].numpy())
    has_bbox = ((bbox_ca[:, 2] - bbox_ca[:, 0]) > 1e-4).astype(jnp.float32)

    alphas = list(np.linspace(args.alpha_min, args.alpha_max, args.n_alphas))
    print(f"Alphas: {[f'{a:.2f}' for a in alphas]}")

    # ── Traverse + segment ────────────────────────────────────────────────────
    print("\n[4/4] Running traversal + segmentation…")
    all_stats = []

    for img_idx in range(min(args.n_images, x_cardio.shape[0])):
        print(f"\nImage {img_idx} / {args.n_images - 1}")
        frames, masks, stats, orig, bbox_norm = traverse_and_segment(
            vae_model, vae_params,
            sam_model, sam_proc, sam_device,
            x_cardio, bbox_ca, has_bbox,
            alphas, img_idx,
        )
        all_stats.append(stats)
        save_image_figure(
            frames, masks, stats, orig, bbox_norm,
            alphas, epoch, img_idx, output_dir,
        )

    save_summary_figure(all_stats, alphas, epoch, output_dir, args.n_images)

    # ── Print verdict ─────────────────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("VERDICT: Does z_disease encode a cardiac-size dial?")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for i, stats in enumerate(all_stats):
        areas = [s['area'] for s in stats]
        delta = areas[-1] - areas[0]
        trend = "↑ GROWING" if delta > 200 else ("↓ SHRINKING" if delta < -200 else "≈ FLAT")
        print(f"  img {i}: area α=0→{args.alpha_max}:  {areas[0]:,} → {areas[-1]:,}  "
              f"(Δ={delta:+,})  {trend}")
    print()
    print(f"Output: {output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
