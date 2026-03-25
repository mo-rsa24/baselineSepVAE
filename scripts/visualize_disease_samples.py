"""
Visualize 4 Cardiomegaly (top row) and 4 Pleural Thickening (bottom row) samples
with bounding box regions faintly coloured.

Layout: 2 rows × 4 columns
  Row 0 (red tint)  — Cardiomegaly
  Row 1 (blue tint) — Pleural Thickening

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_disease_samples.py \
        --data_dir /datasets/mmolefe/vinbigdata/train \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --seed 42 \
        --output scripts/disease_samples_bbox.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.VinBigData import VinBigDataTripletDataset


# ── colour config ──────────────────────────────────────────────────────────────
CARDIO_FILL  = (1.0, 0.15, 0.10, 0.25)   # translucent red
CARDIO_EDGE  = (1.0, 0.10, 0.05, 0.90)   # solid red border
PLTHICK_FILL = (0.10, 0.45, 1.0,  0.25)  # translucent blue
PLTHICK_EDGE = (0.05, 0.35, 0.95, 0.90)  # solid blue border


def overlay_bbox(ax, bbox: np.ndarray, img_size: int,
                 facecolor, edgecolor, label: str) -> None:
    """Draw a faintly filled rectangle over the annotated region."""
    x0, y0, x1, y1 = bbox
    rect = patches.Rectangle(
        (x0 * img_size, y0 * img_size),
        (x1 - x0) * img_size,
        (y1 - y0) * img_size,
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
        label=label,
    )
    ax.add_patch(rect)


def sample_valid(ids, bbox_lookup, n, seed_offset=0):
    """Return n image IDs that have a non-trivial bbox (width > 1%)."""
    valid = [
        iid for iid in ids
        if iid in bbox_lookup
        and (bbox_lookup[iid][2] - bbox_lookup[iid][0]) > 0.01
    ]
    if len(valid) < n:
        raise RuntimeError(
            f"Only {len(valid)} images have valid bboxes; requested {n}"
        )
    rng = random.Random(seed_offset)
    return rng.sample(valid, n)


def visualize(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading VinBigDataTripletDataset (use_cache=False, raw DICOMs) …")
    ds = VinBigDataTripletDataset(
        dicom_dir=args.data_dir,
        csv_path=args.csv_path,
        img_size=256,
        use_cache=False,
    )

    n = args.n_samples
    cardio_ids  = sample_valid(ds.cardio_ids,  ds._cardio_bbox_lookup,  n, seed_offset=args.seed)
    plthick_ids = sample_valid(ds.plthick_ids, ds._plthick_bbox_lookup, n, seed_offset=args.seed + 1)

    img_size = 256
    fig, axes = plt.subplots(
        2, n,
        figsize=(3.2 * n, 7.2),
        constrained_layout=True,
    )

    row_meta = [
        ("Cardiomegaly",       cardio_ids,  ds._cardio_bbox_lookup,  CARDIO_FILL,  CARDIO_EDGE),
        ("Pleural Thickening", plthick_ids, ds._plthick_bbox_lookup, PLTHICK_FILL, PLTHICK_EDGE),
    ]

    for row, (disease_name, ids, bbox_lookup, fill, edge) in enumerate(row_meta):
        pool      = ds.cardio_ids  if row == 0 else ds.plthick_ids
        pool_name = "Cardiomegaly" if row == 0 else "PlThick"

        for col, iid in enumerate(ids):
            img_tensor, bbox_tensor = ds._load_disease_image(
                pool, bbox_lookup, pool_name,
                initial_id=iid, max_retries=5,
            )
            img_np = img_tensor.squeeze().numpy()
            img_01 = (img_np + 1.0) / 2.0
            bbox   = bbox_tensor.numpy()

            ax = axes[row, col]
            ax.imshow(img_01, cmap='gray', vmin=0, vmax=1)

            has_bbox = (bbox[2] - bbox[0]) > 0.01
            if has_bbox:
                overlay_bbox(ax, bbox, img_size, fill, edge,
                             label=f'[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]')
                ax.legend(
                    loc='lower left', fontsize=5.5,
                    framealpha=0.65, handlelength=0,
                    borderpad=0.4,
                )

            ax.set_xticks([]); ax.set_yticks([])

            # Row label on first column only
            if col == 0:
                ax.set_ylabel(disease_name, fontsize=11, fontweight='bold', labelpad=6)

    fig.suptitle(
        "VinBigData — disease samples with ground-truth bounding regions",
        fontsize=12, y=1.02,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',  default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--n_samples', type=int, default=4)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    default='scripts/disease_samples_bbox.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
