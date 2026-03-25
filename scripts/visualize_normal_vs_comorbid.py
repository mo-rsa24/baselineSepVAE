"""
Side-by-side: Normal (left) vs. co-morbid Cardiomegaly + Pleural Thickening (right).

  Left  — a Normal chest X-ray (no annotation)
  Right — a patient with BOTH pathologies; red fill = Cardiomegaly region,
          blue fill = Pleural Thickening region

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_normal_vs_comorbid.py \
        --data_dir /datasets/mmolefe/vinbigdata/train \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --seed 42 \
        --output scripts/normal_vs_comorbid.png
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.VinBigData import VinBigDataTripletDataset


CARDIO_FILL  = (1.0, 0.15, 0.10, 0.25)
CARDIO_EDGE  = (1.0, 0.10, 0.05, 0.95)
PLTHICK_FILL = (0.10, 0.45, 1.0,  0.25)
PLTHICK_EDGE = (0.05, 0.35, 0.95, 0.95)


def overlay_bbox(ax, bbox, img_size, facecolor, edgecolor):
    x0, y0, x1, y1 = bbox
    ax.add_patch(patches.Rectangle(
        (x0 * img_size, y0 * img_size),
        (x1 - x0) * img_size, (y1 - y0) * img_size,
        linewidth=1.8, edgecolor=edgecolor, facecolor=facecolor,
    ))


def find_comorbid_ids(ds):
    """Return image IDs that have a bbox annotation for both diseases."""
    return sorted(set(ds._cardio_bbox_lookup) & set(ds._plthick_bbox_lookup))


def normalise_bbox(raw_bb, pixel_array, use_cache: bool) -> np.ndarray:
    """
    Convert a bbox from the lookup to normalised [0,1] coords.
    use_cache=True  → already normalised in the CSV.
    use_cache=False → pixel coords; divide by the DICOM's original H/W.
    """
    x0, y0, x1, y1 = raw_bb
    if use_cache:
        return np.clip([x0, y0, x1, y1], 0.0, 1.0).astype(np.float32)
    H, W = pixel_array.shape   # original DICOM dims before resize
    return np.clip([x0/W, y0/H, x1/W, y1/H], 0.0, 1.0).astype(np.float32)


def visualize(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset …")
    ds = VinBigDataTripletDataset(
        dicom_dir=args.data_dir, csv_path=args.csv_path,
        img_size=256, use_cache=False,
    )
    img_size = 256

    # Pick samples
    norm_id      = random.choice(ds.normal_ids)
    comorbid_ids = find_comorbid_ids(ds)
    if not comorbid_ids:
        raise RuntimeError("No co-morbidity cases found.")
    comorbid_id = random.choice(comorbid_ids)

    print(f"Normal      : {norm_id}")
    print(f"Co-morbid   : {comorbid_id}")

    # Load normal
    norm_arr = ds._load_image(norm_id)
    norm_img = (ds._preprocess_image(norm_arr).squeeze().numpy() + 1.0) / 2.0

    # Load co-morbid — keep raw pixel array to get original H/W for bbox normalisation
    comorbid_arr = ds._load_image(comorbid_id)
    comorbid_img = (ds._preprocess_image(comorbid_arr).squeeze().numpy() + 1.0) / 2.0
    cardio_bbox  = normalise_bbox(ds._cardio_bbox_lookup[comorbid_id],  comorbid_arr, ds.use_cache)
    plthick_bbox = normalise_bbox(ds._plthick_bbox_lookup[comorbid_id], comorbid_arr, ds.use_cache)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), constrained_layout=True)

    # Left — Normal
    axes[0].imshow(norm_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Normal", fontsize=13, fontweight='bold', pad=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Right — Co-morbid
    axes[1].imshow(comorbid_img, cmap='gray', vmin=0, vmax=1)
    overlay_bbox(axes[1], cardio_bbox,  img_size, CARDIO_FILL,  CARDIO_EDGE)
    overlay_bbox(axes[1], plthick_bbox, img_size, PLTHICK_FILL, PLTHICK_EDGE)
    axes[1].set_title("Cardiomegaly + Pleural Thickening", fontsize=13,
                      fontweight='bold', pad=7)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=CARDIO_EDGE[:3], linewidth=2,
               marker='s', markersize=10,
               markerfacecolor=(*CARDIO_FILL[:3], 0.55), label='Cardiomegaly'),
        Line2D([0], [0], color=PLTHICK_EDGE[:3], linewidth=2,
               marker='s', markersize=10,
               markerfacecolor=(*PLTHICK_FILL[:3], 0.55), label='Pleural Thickening'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, -0.07))

    fig.suptitle("Normal vs. Co-morbid patient (Cardiomegaly ∩ Pleural Thickening)",
                 fontsize=12, y=1.03)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path', default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--output',   default='scripts/normal_vs_comorbid.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
