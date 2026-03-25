"""
regen_filtered_csv.py — Regenerate train_filtered.csv with correct normalised bbox coords.

Reads original DICOM headers (stop_before_pixels=True — fast) to get H/W,
then normalises all bbox annotations from pixel coords to [0, 1].

Only includes image_ids whose .npy exists in the cache.

Usage:
    python scripts/regen_filtered_csv.py \
        --dicom_dir  /datasets/mmolefe/vinbigdata/train \
        --csv_path   /datasets/mmolefe/vinbigdata/train.csv \
        --cache_dir  /datasets/mmolefe/vinbigdata/cache_npy \
        --output     /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
        --num_workers 16
"""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

TARGET_CLASSES = {3, 11, 14}


# ── Per-worker: read DICOM dims from header only ───────────────────────────────

def get_dicom_dims(args):
    """Returns (image_id, H, W) or (image_id, None, None) on failure."""
    image_id, dicom_path = args
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        rows = int(dcm.Rows)
        cols = int(dcm.Columns)
        return image_id, rows, cols
    except Exception:
        # Fallback: read pixel array to get shape
        try:
            dcm = pydicom.dcmread(dicom_path)
            h, w = dcm.pixel_array.shape[:2]
            return image_id, h, w
        except Exception as e:
            return image_id, None, None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom_dir',   default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',    default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--cache_dir',   default='/datasets/mmolefe/vinbigdata/cache_npy')
    p.add_argument('--output',      default='/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv')
    p.add_argument('--num_workers', type=int, default=16)
    args = p.parse_args()

    dicom_dir = Path(args.dicom_dir)
    cache_dir = Path(args.cache_dir)
    npy_dir   = cache_dir / 'images'

    # ── 1. Load train.csv ─────────────────────────────────────────────────────
    print(f"Reading {args.csv_path} …")
    df = pd.read_csv(args.csv_path)
    df['class_id'] = df['class_id'].astype(int)
    df = df[df['class_id'].isin(TARGET_CLASSES)].copy()
    print(f"  Rows after class filter: {len(df):,}")

    # ── 2. Find image_ids present in the cache ────────────────────────────────
    cached = {p.stem for p in npy_dir.glob('*.npy')}
    print(f"  .npy files in cache: {len(cached):,}")

    df = df[df['image_id'].isin(cached)].copy()
    print(f"  Rows after cache filter: {len(df):,}")

    unique_ids = df['image_id'].unique().tolist()
    print(f"  Unique image_ids to process: {len(unique_ids):,}")

    # ── 3. Read DICOM dims in parallel ────────────────────────────────────────
    print(f"\nReading DICOM dims ({args.num_workers} workers) …")
    tasks = [
        (iid, str(dicom_dir / f"{iid}.dicom"))
        for iid in unique_ids
        if (dicom_dir / f"{iid}.dicom").exists()
    ]
    missing_dicoms = set(unique_ids) - {t[0] for t in tasks}
    if missing_dicoms:
        print(f"  WARNING: {len(missing_dicoms)} image_ids have no DICOM file — skipping")

    dims = {}   # image_id -> (H, W)
    failed = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(get_dicom_dims, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            image_id, H, W = fut.result()
            if H is not None:
                dims[image_id] = (H, W)
            else:
                failed.append(image_id)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(tasks):,}  dims_ok={len(dims):,}  failed={len(failed)}")

    print(f"  Dims read: {len(dims):,}  failed: {len(failed)}")

    # ── 4. Build output rows ──────────────────────────────────────────────────
    print("\nBuilding normalised CSV …")

    # Group annotation rows by image_id for fast lookup
    ann_by_id = defaultdict(list)
    for _, row in df.iterrows():
        ann_by_id[row['image_id']].append(row)

    out_rows = []
    class_counts = defaultdict(int)
    bbox_skipped = 0

    for image_id, annotations in ann_by_id.items():
        if image_id not in dims:
            continue   # couldn't read DICOM dims

        H, W = dims[image_id]
        class_id   = int(annotations[0]['class_id'])
        class_name = annotations[0]['class_name']

        if class_id == 14:
            # Normal — no bbox
            out_rows.append({
                'image_id':   image_id,
                'class_name': class_name,
                'class_id':   class_id,
                'x_min_norm': '',
                'y_min_norm': '',
                'x_max_norm': '',
                'y_max_norm': '',
            })
            class_counts[class_id] += 1
        else:
            # Disease — one row per annotation (radiologist)
            for ann in annotations:
                try:
                    x0 = float(ann['x_min'])
                    y0 = float(ann['y_min'])
                    x1 = float(ann['x_max'])
                    y1 = float(ann['y_max'])
                except (ValueError, TypeError):
                    bbox_skipped += 1
                    continue

                x0n = float(np.clip(x0 / W, 0.0, 1.0))
                y0n = float(np.clip(y0 / H, 0.0, 1.0))
                x1n = float(np.clip(x1 / W, 0.0, 1.0))
                y1n = float(np.clip(y1 / H, 0.0, 1.0))

                out_rows.append({
                    'image_id':   image_id,
                    'class_name': class_name,
                    'class_id':   class_id,
                    'x_min_norm': f'{x0n:.6f}',
                    'y_min_norm': f'{y0n:.6f}',
                    'x_max_norm': f'{x1n:.6f}',
                    'y_max_norm': f'{y1n:.6f}',
                })
            class_counts[class_id] += 1

    # ── 5. Write output CSV ───────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['image_id', 'class_name', 'class_id',
                  'x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWritten: {out_path}  ({len(out_rows):,} rows)")
    print(f"\nClass counts (unique images):")
    names = {3: 'Cardiomegaly', 11: 'Pleural Thickening', 14: 'No Finding'}
    for cid in sorted(class_counts):
        print(f"  class {cid:2d} ({names.get(cid, '?'):20s}): {class_counts[cid]:,} images")
    if bbox_skipped:
        print(f"  Bbox rows skipped (non-numeric coords): {bbox_skipped}")
    if failed:
        print(f"  Images dropped (DICOM unreadable): {len(failed)}")


if __name__ == '__main__':
    main()
