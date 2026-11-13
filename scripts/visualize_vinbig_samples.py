"""
Visualize VinBigData samples at 128x128 with bounding box annotations.

Two grids are produced and saved:
  1. bbox_grid.png   — x-rays with coloured bounding-box rectangles + labels
  2. heatmap_grid.png — x-rays with Gaussian-blurred ROI heatmap overlay

Run from the miniSepVAE root:
    python scripts/visualize_vinbig_samples.py [--n_samples 4] [--out_dir .]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_filter

# ── paths ────────────────────────────────────────────────────────────────────
DEFAULT_DICOM_DIR = "/datasets/mmolefe/vinbigdata/train"
DEFAULT_CSV_PATH  = "/datasets/mmolefe/vinbigdata/train.csv"
IMG_SIZE          = 128
CLASS_IDS         = {"cardiomegaly": 3, "effusion": 10}
COLORS            = {"cardiomegaly": "#FF6B6B", "effusion": "#4ECDC4"}
DISPLAY_NAMES     = {"cardiomegaly": "Cardiomegaly", "effusion": "Pleural Effusion"}


# ── image loading (mirrors VinBigData.py logic) ───────────────────────────────

def _load_dicom(dicom_path: Path) -> np.ndarray | None:
    """Load and preprocess a DICOM, matching VinBigData.py's fixed loader."""
    if not dicom_path.exists():
        return None
    try:
        dcm = pydicom.dcmread(str(dicom_path), force=True)
        pixel_array = dcm.pixel_array.astype(np.float32)
    except Exception:
        return None

    # Fix 1: MONOCHROME1 inversion
    photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip()
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Fix 2: Prefer DICOM window metadata; fall back to percentile window
    wc_meta = getattr(dcm, "WindowCenter", None)
    ww_meta = getattr(dcm, "WindowWidth",  None)
    if wc_meta is not None and ww_meta is not None:
        wc = float(wc_meta[0] if isinstance(wc_meta, (list, pydicom.multival.MultiValue)) else wc_meta)
        ww = float(ww_meta[0] if isinstance(ww_meta, (list, pydicom.multival.MultiValue)) else ww_meta)
        lower = wc - ww / 2.0
        upper = wc + ww / 2.0
    else:
        lower = float(np.percentile(pixel_array, 1))
        upper = float(np.percentile(pixel_array, 99))

    windowed = np.clip(pixel_array, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8)

    # Fix 3: Reject blank/uniform images
    if windowed.std() < 0.02:
        return None

    return windowed  # [0, 1], original resolution


def _resize(pixel_array: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((pixel_array * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]


def _build_bbox_lookup(df: pd.DataFrame, class_id: int) -> dict:
    bbox_cols = ["x_min", "y_min", "x_max", "y_max"]
    if not all(c in df.columns for c in bbox_cols):
        return {}
    sub = df[df["class_id"] == class_id][["image_id"] + bbox_cols].copy()
    sub = sub[(sub["x_max"] > 0) & (sub["y_max"] > 0)]
    if sub.empty:
        return {}
    agg = sub.groupby("image_id").agg(
        x_min=("x_min", "min"), y_min=("y_min", "min"),
        x_max=("x_max", "max"), y_max=("y_max", "max"),
    ).reset_index()
    return {
        row["image_id"]: (float(row["x_min"]), float(row["y_min"]),
                          float(row["x_max"]), float(row["y_max"]))
        for _, row in agg.iterrows()
    }


def load_samples(dicom_dir: Path, df: pd.DataFrame, disease: str, n: int, seed: int = 42):
    """
    Returns a list of dicts:
        { 'image': np.ndarray (H,W) in [0,1],
          'bbox':  np.ndarray [x0,y0,x1,y1] normalised to [0,1],
          'image_id': str }
    Only image_ids that have a valid bbox AND a valid DICOM file are kept.
    """
    class_id   = CLASS_IDS[disease]
    bbox_lookup = _build_bbox_lookup(df, class_id)
    candidates  = [iid for iid in df[df["class_id"] == class_id]["image_id"].unique()
                   if iid in bbox_lookup]

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    samples = []
    for image_id in candidates:
        if len(samples) >= n:
            break
        dicom_path = dicom_dir / f"{image_id}.dicom"
        pixel_array = _load_dicom(dicom_path)  # includes MONOCHROME1 fix + percentile window
        if pixel_array is None:
            continue

        H_orig, W_orig = pixel_array.shape
        x0, y0, x1, y1 = bbox_lookup[image_id]
        bbox = np.clip(
            [x0 / W_orig, y0 / H_orig, x1 / W_orig, y1 / H_orig], 0.0, 1.0
        ).astype(np.float32)

        samples.append({
            "image":    _resize(pixel_array, IMG_SIZE),
            "bbox":     bbox,
            "image_id": image_id,
        })

    if len(samples) < n:
        print(f"  Warning: only found {len(samples)}/{n} valid samples for {disease}")
    return samples


# ── heatmap helpers ───────────────────────────────────────────────────────────

def bbox_to_heatmap(bbox: np.ndarray, H: int, W: int, sigma: float = 6.0) -> np.ndarray:
    """Gaussian-blurred binary mask from a normalised bbox."""
    mask = np.zeros((H, W), dtype=np.float32)
    x0 = int(bbox[0] * W)
    y0 = int(bbox[1] * H)
    x1 = int(bbox[2] * W)
    y1 = int(bbox[3] * H)
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    mask[y0:y1, x0:x1] = 1.0
    heatmap = gaussian_filter(mask, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ── grid plotting ─────────────────────────────────────────────────────────────

def plot_bbox_grid(cardio_samples, effusion_samples, out_path: Path):
    """4 rows × N cols: rows 0-1 = cardiomegaly, rows 2-3 = effusion."""
    all_groups = [
        ("cardiomegaly", cardio_samples),
        ("effusion",     effusion_samples),
    ]
    n = max(len(cardio_samples), len(effusion_samples))
    n_rows = 2  # one row per disease
    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * 2.4, n_rows * 2.8),
        facecolor="#1a1a2e",
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (disease, samples) in enumerate(all_groups):
        color = COLORS[disease]
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#1a1a2e")
            if col_idx >= len(samples):
                ax.axis("off")
                continue

            s = samples[col_idx]
            img   = s["image"]
            bbox  = s["bbox"]
            H, W  = img.shape

            ax.imshow(img, cmap="bone", vmin=0, vmax=1, aspect="equal")

            x0 = bbox[0] * W
            y0 = bbox[1] * H
            bw = (bbox[2] - bbox[0]) * W
            bh = (bbox[3] - bbox[1]) * H
            rect = mpatches.FancyBboxPatch(
                (x0, y0), bw, bh,
                boxstyle="square,pad=0",
                linewidth=1.8,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # Label tag above the box
            ax.text(
                x0, max(y0 - 3, 4),
                DISPLAY_NAMES[disease],
                color=color, fontsize=6.5, fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="#1a1a2e", ec="none", alpha=0.7),
            )

            short_id = s["image_id"][:8] + "…"
            ax.set_title(short_id, color="white", fontsize=6, pad=2)
            ax.axis("off")

    fig.suptitle("VinBigData 128×128 — Bounding Box Annotations",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_heatmap_grid(cardio_samples, effusion_samples, out_path: Path):
    """Same layout but with Gaussian heatmap overlay instead of rectangle."""
    all_groups = [
        ("cardiomegaly", cardio_samples),
        ("effusion",     effusion_samples),
    ]
    n = max(len(cardio_samples), len(effusion_samples))
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * 2.4, n_rows * 2.8),
        facecolor="#1a1a2e",
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n == 1:
        axes = axes[:, np.newaxis]

    cmap_map = {"cardiomegaly": "Reds", "effusion": "Blues"}

    for row_idx, (disease, samples) in enumerate(all_groups):
        cmap = cmap_map[disease]
        color = COLORS[disease]
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#1a1a2e")
            if col_idx >= len(samples):
                ax.axis("off")
                continue

            s = samples[col_idx]
            img  = s["image"]
            bbox = s["bbox"]
            H, W = img.shape

            heatmap = bbox_to_heatmap(bbox, H, W, sigma=6.0)

            ax.imshow(img, cmap="bone", vmin=0, vmax=1, aspect="equal")
            ax.imshow(heatmap, cmap=cmap, alpha=0.45, vmin=0, vmax=1, aspect="equal")

            # Small disease label in corner
            ax.text(
                2, 2,
                DISPLAY_NAMES[disease],
                color=color, fontsize=6.5, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="#1a1a2e", ec="none", alpha=0.7),
            )

            short_id = s["image_id"][:8] + "…"
            ax.set_title(short_id, color="white", fontsize=6, pad=2)
            ax.axis("off")

    fig.suptitle("VinBigData 128×128 — ROI Heatmap Silhouettes",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom_dir", default=DEFAULT_DICOM_DIR)
    parser.add_argument("--csv_path",  default=DEFAULT_CSV_PATH)
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples per disease (default: 4)")
    parser.add_argument("--out_dir",   default=".", help="Output directory for PNGs")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    print(f"Sampling {args.n_samples} cardiomegaly images…")
    cardio_samples = load_samples(dicom_dir, df, "cardiomegaly", args.n_samples, args.seed)

    print(f"Sampling {args.n_samples} effusion images…")
    effusion_samples = load_samples(dicom_dir, df, "effusion", args.n_samples, args.seed)

    print("Rendering bbox grid…")
    plot_bbox_grid(cardio_samples, effusion_samples, out_dir / "bbox_grid.png")

    print("Rendering heatmap grid…")
    plot_heatmap_grid(cardio_samples, effusion_samples, out_dir / "heatmap_grid.png")

    print("Done.")


if __name__ == "__main__":
    main()
