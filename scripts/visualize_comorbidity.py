"""
Visualize co-morbidity cases: patients annotated with BOTH Cardiomegaly AND
Pleural Thickening, with each pathology's bounding region coloured differently.

  Red  fill  →  Cardiomegaly region
  Blue fill  →  Pleural Thickening region

Samples are picked from the intersection of both bbox lookup tables so every
panel is guaranteed to show two distinct annotated regions.

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_comorbidity.py \
        --data_dir /datasets/mmolefe/vinbigdata/train \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --n_samples 4 \
        --seed 42 \
        --output scripts/comorbidity_bbox.png
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


# ── colour config ──────────────────────────────────────────────────────────────
CARDIO_FILL  = (1.0, 0.15, 0.10, 0.25)
CARDIO_EDGE  = (1.0, 0.10, 0.05, 0.95)
PLTHICK_FILL = (0.10, 0.45, 1.0,  0.25)
PLTHICK_EDGE = (0.05, 0.35, 0.95, 0.95)


def overlay_bbox(ax, bbox: np.ndarray, img_size: int,
                 facecolor, edgecolor) -> None:
    x0, y0, x1, y1 = bbox
    rect = patches.Rectangle(
        (x0 * img_size, y0 * img_size),
        (x1 - x0) * img_size,
        (y1 - y0) * img_size,
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)


def find_comorbid_ids(ds):
    """Return image IDs that have a bbox annotation for both diseases."""
    return sorted(set(ds._cardio_bbox_lookup) & set(ds._plthick_bbox_lookup))


def normalise_bbox(raw_bb, pixel_array, use_cache: bool) -> np.ndarray:
    """
    Convert a bbox from the lookup table to normalised [0, 1] coords.
    use_cache=True  → CSV already has normalised values.
    use_cache=False → pixel coords; divide by original DICOM H/W.
    """
    x0, y0, x1, y1 = raw_bb
    if use_cache:
        return np.clip([x0, y0, x1, y1], 0.0, 1.0).astype(np.float32)
    H, W = pixel_array.shape
    return np.clip([x0/W, y0/H, x1/W, y1/H], 0.0, 1.0).astype(np.float32)


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

    comorbid_ids = find_comorbid_ids(ds)
    print(f"Found {len(comorbid_ids)} co-morbidity cases (Cardiomegaly ∩ Pleural Thickening).")

    if len(comorbid_ids) < args.n_samples:
        raise RuntimeError(
            f"Only {len(comorbid_ids)} co-morbidity cases available; "
            f"requested {args.n_samples}."
        )

    chosen = random.sample(comorbid_ids, args.n_samples)

    # ── Figure: 1 row per sample, each showing the single image with both overlays
    ncols = args.n_samples
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 4.2),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, iid in zip(axes, chosen):
        # Load raw pixel array — needed for dims when normalising pixel-coord bboxes
        pixel_arr = ds._load_image(iid)
        img_01 = (ds._preprocess_image(pixel_arr).squeeze().numpy() + 1.0) / 2.0

        cardio_bbox  = normalise_bbox(ds._cardio_bbox_lookup[iid],  pixel_arr, ds.use_cache)
        plthick_bbox = normalise_bbox(ds._plthick_bbox_lookup[iid], pixel_arr, ds.use_cache)

        ax.imshow(img_01, cmap='gray', vmin=0, vmax=1)

        overlay_bbox(ax, cardio_bbox,  img_size, CARDIO_FILL,  CARDIO_EDGE)
        overlay_bbox(ax, plthick_bbox, img_size, PLTHICK_FILL, PLTHICK_EDGE)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"…{iid[-10:]}", fontsize=7, pad=4)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=CARDIO_EDGE[:3],  linewidth=2,
               marker='s', markersize=9,
               markerfacecolor=(*CARDIO_FILL[:3], 0.55), label='Cardiomegaly'),
        Line2D([0], [0], color=PLTHICK_EDGE[:3], linewidth=2,
               marker='s', markersize=9,
               markerfacecolor=(*PLTHICK_FILL[:3], 0.55), label='Pleural Thickening'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=2,
        fontsize=10,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Co-morbidity: Cardiomegaly + Pleural Thickening on the same patient",
        fontsize=12, y=1.02,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")
    print(f"\nCo-morbid cases shown:")
    for iid in chosen:
        cb = ds._cardio_bbox_lookup[iid]
        pb = ds._plthick_bbox_lookup[iid]
        print(f"  {iid}  cardio={cb}  plthick={pb}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',  default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--n_samples', type=int, default=4)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    default='scripts/comorbidity_bbox.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
