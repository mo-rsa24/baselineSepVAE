"""
Visualize bbox annotations + Gaussian cross-attention heatmap for 4 cardiomegaly samples.

Shows three panels per sample:
  1. Raw 256×256 image with ground-truth bbox rectangle
  2. Gaussian heatmap at 16×16 (model attention resolution), upsampled to 256×256
  3. Overlay: image + heatmap blend

The Gaussian uses the exact same formula as BboxCrossAttnHead:
    cx, cy = bbox centre
    σ_x = bbox_width  / 4   (quarter-width — concentrates inside the bbox)
    σ_y = bbox_height / 4

This lets you verify:
  - Bbox annotations are correctly loaded and normalised at 256px
  - The Gaussian prior tightly covers the cardiac silhouette
  - The 16×16 attention resolution is sufficient for cardiomegaly localisation

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_bbox_heatmap.py \
        --data_dir /datasets/mmolefe/vinbigdata/cache_npy \
        --csv_path /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
        --n_samples 4 \
        --seed 42 \
        --output scripts/bbox_heatmap_check.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.VinBigData import VinBigDataPairDataset


# ── Gaussian heatmap (mirrors BboxCrossAttnHead exactly) ─────────────────────

def make_gaussian_heatmap(
    bbox: np.ndarray,          # [x0, y0, x1, y1] normalised [0, 1]
    attn_H: int = 16,          # attention map height (feature map resolution)
    attn_W: int = 16,          # attention map width
) -> np.ndarray:
    """
    Reproduce the Gaussian prior from BboxCrossAttnHead at the model's attention
    map resolution (16×16 for 256px input).

    Returns: (attn_H, attn_W) float32 heatmap in [0, 1].
    """
    x0, y0, x1, y1 = bbox

    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    sx = max((x1 - x0) * 0.25, 0.05)   # quarter-width σ — matches BboxCrossAttnHead
    sy = max((y1 - y0) * 0.25, 0.05)

    y_coords = (np.arange(attn_H, dtype=np.float32) + 0.5) / attn_H
    x_coords = (np.arange(attn_W, dtype=np.float32) + 0.5) / attn_W
    xx, yy   = np.meshgrid(x_coords, y_coords)   # (H, W)

    gauss = np.exp(
        -0.5 * (
            (xx - cx) ** 2 / sx ** 2
          + (yy - cy) ** 2 / sy ** 2
        )
    )
    # Normalise to [0, 1] for display
    gauss = gauss / (gauss.max() + 1e-8)
    return gauss.astype(np.float32)


def upsample_nearest(arr: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """Nearest-neighbour upsample (H, W) → (target_H, target_W). No scipy needed."""
    H, W = arr.shape
    row_idx = (np.arange(target_H) * H // target_H).astype(int)
    col_idx = (np.arange(target_W) * W // target_W).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


# ── Custom colourmap: transparent→opaque red ──────────────────────────────────
CARDIAC_CMAP = LinearSegmentedColormap.from_list(
    'cardiac',
    [(0.0, (1.0, 0.2, 0.1, 0.0)),   # transparent at zero
     (0.4, (1.0, 0.4, 0.0, 0.5)),   # orange-red, semi-transparent
     (1.0, (1.0, 0.0, 0.0, 0.85))], # solid red at peak
    N=256,
)


def visualize(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading VinBigDataPairDataset  (img_size=256, use_cache=True)")
    ds = VinBigDataPairDataset(
        dicom_dir=args.data_dir,
        csv_path=args.csv_path,
        img_size=256,
        use_cache=True,
    )

    # Sample n_samples cardiomegaly images that have a valid bbox
    valid_indices = [
        i for i, cid in enumerate(ds.cardio_ids)
        if cid in ds._cardio_bbox_lookup
        and (ds._cardio_bbox_lookup[cid][2] - ds._cardio_bbox_lookup[cid][0]) > 0.01
    ]
    if len(valid_indices) < args.n_samples:
        raise RuntimeError(
            f"Only {len(valid_indices)} cardiomegaly images have valid bboxes; "
            f"requested {args.n_samples}"
        )

    chosen = random.sample(valid_indices, args.n_samples)

    fig, axes = plt.subplots(
        args.n_samples, 3,
        figsize=(12, 3.5 * args.n_samples),
        constrained_layout=True,
    )
    if args.n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "256×256 image + GT bbox",
        "Gaussian prior  (16×16 → 256×256)",
        "Overlay",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    for row, idx in enumerate(chosen):
        cardio_id = ds.cardio_ids[idx]
        img_tensor, bbox_tensor = ds._load_disease_image(
            ds.cardio_ids, ds._cardio_bbox_lookup, "Cardiomegaly",
            initial_id=cardio_id, max_retries=5,
        )

        # img_tensor: (1, 256, 256) in [-1, 1]  →  (256, 256) in [0, 1]
        img_np = img_tensor.squeeze().numpy()
        img_01 = (img_np + 1.0) / 2.0
        bbox   = bbox_tensor.numpy()   # [x0, y0, x1, y1] in [0, 1]

        x0, y0, x1, y1 = bbox
        has_bbox = (x1 - x0) > 0.01

        # Gaussian heatmap at 16×16 then upsampled for display
        if has_bbox:
            heatmap_16 = make_gaussian_heatmap(bbox, attn_H=16, attn_W=16)
            heatmap_256 = upsample_nearest(heatmap_16, 256, 256)
        else:
            heatmap_256 = np.zeros((256, 256), dtype=np.float32)

        # ── Column 0: image + bbox rect ───────────────────────────────────────
        ax = axes[row, 0]
        ax.imshow(img_01, cmap='gray', vmin=0, vmax=1)
        if has_bbox:
            rect = patches.Rectangle(
                (x0 * 256, y0 * 256),
                (x1 - x0) * 256, (y1 - y0) * 256,
                linewidth=2, edgecolor='lime', facecolor='none',
                label=f'bbox  σ_x={((x1-x0)*0.25):.3f}  σ_y={((y1-y0)*0.25):.3f}',
            )
            ax.add_patch(rect)
            ax.legend(loc='lower left', fontsize=7, framealpha=0.7)
        ax.set_xlabel(f"id: {cardio_id[:12]}…  bbox=[{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}]",
                      fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Column 1: Gaussian heatmap only ───────────────────────────────────
        ax = axes[row, 1]
        im = ax.imshow(heatmap_256, cmap='hot', vmin=0, vmax=1)
        # Draw 16×16 grid to show actual attention resolution
        for k in range(1, 16):
            ax.axhline(k * 256 / 16, color='white', linewidth=0.3, alpha=0.5)
            ax.axvline(k * 256 / 16, color='white', linewidth=0.3, alpha=0.5)
        if has_bbox:
            ax.set_xlabel(
                f"attn res 16×16 | peak={heatmap_16.max():.2f}  "
                f"bbox coverage={((heatmap_16 > 0.5).sum() / 256 * 100):.1f}% cells >0.5",
                fontsize=7,
            )
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ── Column 2: overlay ─────────────────────────────────────────────────
        ax = axes[row, 2]
        ax.imshow(img_01, cmap='gray', vmin=0, vmax=1)
        ax.imshow(heatmap_256, cmap=CARDIAC_CMAP, vmin=0, vmax=1, alpha=0.6)
        if has_bbox:
            rect2 = patches.Rectangle(
                (x0 * 256, y0 * 256),
                (x1 - x0) * 256, (y1 - y0) * 256,
                linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--',
            )
            ax.add_patch(rect2)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Cardiomegaly bbox → Gaussian cross-attention prior  |  "
        f"256×256 images, 16×16 attention map  |  σ = bbox/4",
        fontsize=12, y=1.01,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"\nSaved → {output}")

    # Print bbox stats for all 4 samples
    print("\nBbox statistics (normalised [0,1]):")
    print(f"{'id':>16}  {'x0':>5} {'y0':>5} {'x1':>5} {'y1':>5}  "
          f"{'w':>5} {'h':>5}  {'σ_x':>5} {'σ_y':>5}  {'16x16 cells >0.5':>18}")
    for idx in chosen:
        cid  = ds.cardio_ids[idx]
        bbox = np.array(ds._cardio_bbox_lookup[cid], dtype=np.float32)
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        sx, sy = max(w * 0.25, 0.05), max(h * 0.25, 0.05)
        hm   = make_gaussian_heatmap(bbox)
        n_cells = int((hm > 0.5).sum())
        print(f"{cid[:16]:>16}  {x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}  "
              f"{w:.3f} {h:.3f}  {sx:.3f} {sy:.3f}  {n_cells:>18d} / 256")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str,
                   default='/datasets/mmolefe/vinbigdata/cache_npy')
    p.add_argument('--csv_path', type=str,
                   default='/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv')
    p.add_argument('--n_samples', type=int, default=4)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    type=str,
                   default='scripts/bbox_heatmap_check.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
