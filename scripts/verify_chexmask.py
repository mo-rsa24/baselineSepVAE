"""
verify_chexmask.py
------------------
Verify that CheXmask VinDr-CXR masks align with our cached .npy images.

Workflow:
  1. Load Preprocessed/VinDr-CXR.csv (1024×1024 RLE masks)
  2. Match image IDs against /datasets/mmolefe/vinbigdata/cache_npy/images/
  3. Filter by Dice RCA (Mean) >= --min_dice
  4. Pool of --pool images → keep top --n by quality score
  5. Decode heart + lung RLE masks, resize to 512×512
  6. Compute CTR from masks
  7. Save 3-column grid:  [contours | overlay | zoomed crop]

Usage:
    conda run -n cxr python scripts/verify_chexmask.py
    conda run -n cxr python scripts/verify_chexmask.py --seed 123 --n 5
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_CSV     = "/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv"
DEFAULT_NPY_DIR = "/datasets/mmolefe/vinbigdata/cache_npy/images"
DEFAULT_OUT     = "results/verify_chexmask.png"

# ---------------------------------------------------------------------------
# RLE decode  (CheXmask convention: 1-indexed, row-major)
# ---------------------------------------------------------------------------

def decode_rle(rle_string: str, height: int, width: int) -> np.ndarray:
    """Decode a CheXmask RLE string to a binary uint8 mask (0/1)."""
    if not isinstance(rle_string, str) or not rle_string.strip():
        return np.zeros((height, width), dtype=np.uint8)
    runs = np.array(rle_string.split(), dtype=np.int64)
    starts  = runs[::2] - 1   # 1-indexed → 0-indexed
    lengths = runs[1::2]
    flat = np.zeros(height * width, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        flat[s : s + l] = 1
    return flat.reshape(height, width)


def resize_mask(mask: np.ndarray, target: int = 512) -> np.ndarray:
    """Nearest-neighbour resize to (target, target)."""
    pil = Image.fromarray(mask, mode="L")
    pil = pil.resize((target, target), resample=Image.NEAREST)
    return np.array(pil, dtype=np.uint8)

# ---------------------------------------------------------------------------
# CTR
# ---------------------------------------------------------------------------

def compute_ctr(heart: np.ndarray, left_lung: np.ndarray, right_lung: np.ndarray):
    """Return CTR = cardiac_width / thoracic_width, or None if masks are empty."""
    thorax = (left_lung | right_lung).astype(bool)
    cols_h = np.where(heart.astype(bool).any(axis=0))[0]
    cols_t = np.where(thorax.any(axis=0))[0]
    if not len(cols_h) or not len(cols_t):
        return None
    cardiac_w  = int(cols_h.max() - cols_h.min())
    thoracic_w = int(cols_t.max() - cols_t.min())
    return cardiac_w / thoracic_w if thoracic_w > 0 else None

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(npy_path: Path) -> np.ndarray:
    """Load uint16 .npy → float32 [0, 1]."""
    arr = np.load(str(npy_path)).astype(np.float32) / 65535.0
    return arr  # (512, 512)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

HEART_COLOR  = (0.95, 0.35, 0.20)   # coral
LUNG_COLOR   = (0.10, 0.70, 0.70)   # teal
ALPHA_FILL   = 0.45
ALPHA_LUNG   = 0.20


def overlay_rgba(ax, mask: np.ndarray, color: tuple, alpha: float):
    """Draw a filled RGBA overlay for a binary mask."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba)


def draw_contour(ax, mask: np.ndarray, color: str, lw: float = 1.5):
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=[lw])


def zoom_crop(img: np.ndarray, mask: np.ndarray, pad: float = 0.10):
    """Return image crop centred on the heart mask bounding box."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    h, w = img.shape
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    dh = int((r1 - r0) * pad)
    dw = int((c1 - c0) * pad)
    r0 = max(0, r0 - dh); r1 = min(h, r1 + dh)
    c0 = max(0, c0 - dw); c1 = min(w, c1 + dw)
    return img[r0:r1, c0:c1]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Verify CheXmask VinDr-CXR alignment")
    p.add_argument("--csv",      default=DEFAULT_CSV)
    p.add_argument("--npy_dir",  default=DEFAULT_NPY_DIR)
    p.add_argument("--n",        type=int,   default=8,   help="images to display")
    p.add_argument("--pool",     type=int,   default=60,  help="pool size for ranked selection")
    p.add_argument("--min_dice", type=float, default=0.7, help="Dice RCA (Mean) threshold")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--output",   default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ---- Load CSV --------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}\nRun scripts/download_chexmask.sh first.")

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {len(df):,} rows | columns: {list(df.columns)}")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Identify image-ID column (first column)
    id_col = df.columns[0]
    print(f"  Image ID column: '{id_col}'")

    # ---- Match against cache --------------------------------------------
    npy_dir = Path(args.npy_dir)
    cached_ids = {p.stem for p in npy_dir.glob("*.npy")}
    print(f"  Cached images: {len(cached_ids):,}")

    df["_cached"] = df[id_col].astype(str).isin(cached_ids)
    matched = df[df["_cached"]].copy()
    print(f"  Matched to cache: {len(matched):,} / {len(df):,}")

    # ---- Quality filter -------------------------------------------------
    dice_col = next((c for c in df.columns if "Mean" in c and "Dice" in c), None)
    if dice_col:
        before = len(matched)
        matched = matched[matched[dice_col] >= args.min_dice].copy()
        print(f"  After Dice RCA (Mean) >= {args.min_dice}: {len(matched):,} (removed {before - len(matched):,})")
    else:
        print("  WARNING: Dice RCA (Mean) column not found — skipping quality filter")
        dice_col = None

    if len(matched) == 0:
        sys.exit("ERROR: No matched images after filtering.")

    # ---- Pool → top-N ---------------------------------------------------
    pool_size = min(args.pool, len(matched))
    pool = matched.sample(n=pool_size, random_state=args.seed)
    if dice_col:
        pool = pool.sort_values(dice_col, ascending=False)
    selected = pool.head(args.n)
    print(f"  Pool of {pool_size} → top {len(selected)} by quality score")

    # ---- Detect mask columns -------------------------------------------
    heart_col = next((c for c in df.columns if "Heart" in c), None)
    ll_col    = next((c for c in df.columns if "Left" in c and "Lung" in c), None)
    rl_col    = next((c for c in df.columns if "Right" in c and "Lung" in c), None)
    h_col     = next((c for c in df.columns if c.strip().lower() == "height"), None)
    w_col     = next((c for c in df.columns if c.strip().lower() == "width"), None)

    print(f"  Mask columns — Heart: '{heart_col}', LL: '{ll_col}', RL: '{rl_col}', H: '{h_col}', W: '{w_col}'")

    if not heart_col:
        sys.exit("ERROR: 'Heart' RLE column not found in CSV.")

    # ---- Build figure ---------------------------------------------------
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("CheXmask VinDr-CXR — Cardiac Segmentation Verification\n"
                 "Columns: [Contours]  [Filled overlay]  [Zoomed cardiac crop]",
                 fontsize=11, y=1.01)

    for row_idx, (_, rec) in enumerate(selected.iterrows()):
        img_id = str(rec[id_col])

        # Load image
        npy_path = npy_dir / f"{img_id}.npy"
        img = load_image(npy_path)   # (512, 512) float32 [0, 1]

        # Mask resolution
        mh = int(rec[h_col]) if h_col and pd.notna(rec.get(h_col)) else 1024
        mw = int(rec[w_col]) if w_col and pd.notna(rec.get(w_col)) else 1024

        # Decode + resize masks
        heart_mask = resize_mask(decode_rle(rec[heart_col], mh, mw))
        ll_mask    = resize_mask(decode_rle(rec[ll_col],    mh, mw)) if ll_col else np.zeros_like(heart_mask)
        rl_mask    = resize_mask(decode_rle(rec[rl_col],    mh, mw)) if rl_col else np.zeros_like(heart_mask)

        ctr   = compute_ctr(heart_mask, ll_mask, rl_mask)
        dice  = float(rec[dice_col]) if dice_col else float("nan")
        ctr_s = f"{ctr:.3f}" if ctr is not None else "n/a"
        label = "cardiomegaly" if ctr is not None and ctr > 0.5 else "normal"
        title = f"{img_id[:16]}…  │  Dice: {dice:.3f}  │  CTR: {ctr_s} ({label})"

        ax0, ax1, ax2 = axes[row_idx]

        # Col 0: contours only
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
        draw_contour(ax0, heart_mask,        color="#FF5733", lw=2.0)
        draw_contour(ax0, ll_mask | rl_mask, color="#00BFBF", lw=1.2)
        ax0.set_title(title, fontsize=7, pad=3)

        # Col 1: filled overlay
        ax1.imshow(img, cmap="gray", vmin=0, vmax=1)
        overlay_rgba(ax1, ll_mask | rl_mask, LUNG_COLOR,  ALPHA_LUNG)
        overlay_rgba(ax1, heart_mask,        HEART_COLOR, ALPHA_FILL)
        draw_contour(ax1, heart_mask, color="#FF5733", lw=1.5)

        # Legend patches
        h_patch = mpatches.Patch(color=HEART_COLOR, alpha=0.8, label="Heart")
        l_patch = mpatches.Patch(color=LUNG_COLOR,  alpha=0.6, label="Lungs")
        ax1.legend(handles=[h_patch, l_patch], fontsize=7, loc="lower right",
                   framealpha=0.6)

        # Col 2: zoomed crop
        crop = zoom_crop(img, heart_mask, pad=0.15)
        ax2.imshow(crop, cmap="gray", vmin=0, vmax=1)
        ax2.set_title("Cardiac crop", fontsize=7, pad=3)

        for ax in (ax0, ax1, ax2):
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # Summary stats
    print("\n=== Summary ===")
    print(f"  Total CSV rows : {len(df):,}")
    print(f"  Matched to cache: {len(df[df['_cached']]):,}")
    if dice_col:
        usable = df[df["_cached"] & (df[dice_col] >= args.min_dice)]
        print(f"  Usable (cached + quality): {len(usable):,}")
    print(f"  Displayed: {len(selected)}")
    if dice_col:
        print(f"  Dice RCA range (selected): {selected[dice_col].min():.3f} – {selected[dice_col].max():.3f}")


if __name__ == "__main__":
    main()
