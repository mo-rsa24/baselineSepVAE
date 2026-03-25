"""
Visualize one Normal, one Cardiomegaly, and one Pleural Thickening sample
on a single row, with bounding box regions faintly coloured for the pathologies.

Layout: 1 row × 3 columns
  Col 0 — Normal            (no bbox)
  Col 1 — Cardiomegaly      (red tint over annotated region)
  Col 2 — Pleural Thickening (blue tint over annotated region)

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_triplet_row.py \
        --data_dir /datasets/mmolefe/vinbigdata/train \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --seed 42 \
        --output scripts/triplet_row_bbox.png
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
CARDIO_FILL  = (1.0, 0.15, 0.10, 0.25)
CARDIO_EDGE  = (1.0, 0.10, 0.05, 0.90)
PLTHICK_FILL = (0.10, 0.45, 1.0,  0.25)
PLTHICK_EDGE = (0.05, 0.35, 0.95, 0.90)


def overlay_bbox(ax, bbox: np.ndarray, img_size: int,
                 facecolor, edgecolor, label: str) -> None:
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


def pick_valid(ids, bbox_lookup, seed):
    """Pick one image ID that has a non-trivial bbox."""
    valid = [
        iid for iid in ids
        if iid in bbox_lookup
        and (bbox_lookup[iid][2] - bbox_lookup[iid][0]) > 0.01
    ]
    if not valid:
        raise RuntimeError("No images with valid bboxes found.")
    rng = random.Random(seed)
    return rng.choice(valid)


def load_img(ds, image_id, pool, bbox_lookup, pool_name):
    img_tensor, bbox_tensor = ds._load_disease_image(
        pool, bbox_lookup, pool_name, initial_id=image_id, max_retries=5,
    )
    img_01 = (img_tensor.squeeze().numpy() + 1.0) / 2.0
    return img_01, bbox_tensor.numpy()


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
    img_size = 256

    # Pick samples
    norm_id    = random.choice(ds.normal_ids)
    cardio_id  = pick_valid(ds.cardio_ids,  ds._cardio_bbox_lookup,  args.seed)
    plthick_id = pick_valid(ds.plthick_ids, ds._plthick_bbox_lookup, args.seed + 1)

    # Load images
    norm_arr   = ds._load_image(norm_id)
    norm_img   = (ds._preprocess_image(norm_arr).squeeze().numpy() + 1.0) / 2.0

    cardio_img,  cardio_bbox  = load_img(ds, cardio_id,  ds.cardio_ids,  ds._cardio_bbox_lookup,  "Cardiomegaly")
    plthick_img, plthick_bbox = load_img(ds, plthick_id, ds.plthick_ids, ds._plthick_bbox_lookup, "PlThick")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), constrained_layout=True)

    panels = [
        ("Normal",             norm_img,    None,         None,         None,        None),
        ("Cardiomegaly",       cardio_img,  cardio_bbox,  CARDIO_FILL,  CARDIO_EDGE, "Cardiomegaly region"),
        ("Pleural Thickening", plthick_img, plthick_bbox, PLTHICK_FILL, PLTHICK_EDGE,"Pleural Thickening region"),
    ]

    for ax, (title, img, bbox, fill, edge, lbl) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)

        if bbox is not None and (bbox[2] - bbox[0]) > 0.01:
            overlay_bbox(ax, bbox, img_size, fill, edge, lbl)
            ax.legend(
                loc='lower left', fontsize=7,
                framealpha=0.70, handlelength=1.2,
                borderpad=0.5,
            )

        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "VinBigData — Normal · Cardiomegaly · Pleural Thickening",
        fontsize=13, y=1.03,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',  default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    default='scripts/triplet_row_bbox.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
