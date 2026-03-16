"""
cache_dicoms.py — Pre-cache VinBigData DICOMs to uint16 numpy arrays.

Filters to only the three classes used for training:
  14 = Normal (no finding)
   3 = Cardiomegaly
  11 = Pleural thickening

Outputs:
  <output_dir>/images/<image_id>.npy   — uint16 numpy array (512, 512)
  <output_dir>/train_filtered.csv      — filtered + normalised bbox CSV
  <output_dir>/cache_manifest.json     — counts and paths

Usage:
  python scripts/cache_dicoms.py \\
      --dicom_dir  /datasets/mmolefe/vinbigdata/train \\
      --csv_path   /datasets/mmolefe/vinbigdata/train.csv \\
      --output_dir /datasets/mmolefe/vinbigdata/cache_npy \\
      --img_size   512 \\
      --num_workers 16
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

TARGET_CLASSES = {'3', '11', '14'}   # Cardiomegaly, Pleural thickening, Normal
CLASS_NAMES    = {'3': 'Cardiomegaly', '11': 'Pleural thickening', '14': 'No finding'}


def load_dicom_as_uint16(dicom_path: str, img_size: int) -> np.ndarray:
    dcm = pydicom.dcmread(dicom_path)
    arr = dcm.pixel_array.astype(np.float32)

    # Normalise to [0, 65535] while staying in float32
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 65535.0

    # Resize using PIL mode 'F' (float32) — LANCZOS is supported on 'F' in all
    # Pillow versions, whereas mode 'I' (uint16→int32) is not.
    img = Image.fromarray(arr, mode='F')
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint16)


def process_one(args):
    image_id, dicom_dir, output_dir, img_size = args
    src  = os.path.join(dicom_dir, image_id + '.dicom')
    dest = os.path.join(output_dir, 'images', image_id + '.npy')

    if os.path.exists(dest):
        return image_id, True, 'cached'
    try:
        arr = load_dicom_as_uint16(src, img_size)
        np.save(dest, arr)
        return image_id, True, 'ok'
    except Exception as e:
        return image_id, False, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom_dir',   required=True)
    p.add_argument('--csv_path',    required=True)
    p.add_argument('--output_dir',  required=True)
    p.add_argument('--img_size',    type=int, default=512)
    p.add_argument('--num_workers', type=int, default=8)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)

    # ── Read CSV, collect target image IDs ────────────────────────────────────
    image_ids     = set()
    bbox_rows     = []   # rows with bbox annotations (disease images)
    image_orig_hw = {}   # image_id -> (orig_h, orig_w) from first bbox encounter

    with open(args.csv_path) as f:
        for row in csv.DictReader(f):
            if row['class_id'] not in TARGET_CLASSES:
                continue
            image_ids.add(row['image_id'])
            if row['x_min']:
                bbox_rows.append(row)

    print(f"Target image IDs: {len(image_ids):,}")
    print(f"Bbox annotation rows: {len(bbox_rows):,}")

    # ── Cache DICOMs ──────────────────────────────────────────────────────────
    tasks   = [(iid, args.dicom_dir, str(output_dir), args.img_size)
               for iid in image_ids]
    ok, err = 0, 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(process_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            iid, success, msg = fut.result()
            if success:
                ok += 1
            else:
                err += 1
                print(f"  ERROR {iid}: {msg}", file=sys.stderr)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}  ok={ok} err={err}")

    print(f"\nCaching complete: {ok} ok, {err} errors")

    # ── Get original image dimensions for bbox normalisation ─────────────────
    # Load a sample to get actual DICOM dims (needed to normalise coords)
    print("Reading original DICOM dimensions for bbox normalisation...")
    dicom_dims = {}
    sample_ids = list({r['image_id'] for r in bbox_rows})[:2000]
    for iid in sample_ids:
        try:
            dcm = pydicom.dcmread(os.path.join(args.dicom_dir, iid + '.dicom'))
            arr = dcm.pixel_array
            dicom_dims[iid] = (arr.shape[0], arr.shape[1])  # H, W
        except Exception:
            pass

    # ── Write normalised bbox CSV ─────────────────────────────────────────────
    out_csv = output_dir / 'train_filtered.csv'
    written = 0
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'class_name', 'class_id',
                         'x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm'])
        for row in bbox_rows:
            iid  = row['image_id']
            dims = dicom_dims.get(iid)
            if dims is None:
                continue
            orig_h, orig_w = dims
            x0 = float(row['x_min']) / orig_w
            y0 = float(row['y_min']) / orig_h
            x1 = float(row['x_max']) / orig_w
            y1 = float(row['y_max']) / orig_h
            writer.writerow([iid, row['class_name'], row['class_id'],
                             f'{x0:.6f}', f'{y0:.6f}', f'{x1:.6f}', f'{y1:.6f}'])
            written += 1

    print(f"Filtered CSV written: {out_csv}  ({written:,} bbox rows)")

    # ── Manifest ──────────────────────────────────────────────────────────────
    class_counts = defaultdict(int)
    for iid in image_ids:
        pass  # we don't track per-image class here easily
    # count from bbox rows + normal
    for row in bbox_rows:
        class_counts[row['class_id']] += 1

    manifest = {
        'img_size':     args.img_size,
        'total_images': ok,
        'errors':       err,
        'class_counts': {CLASS_NAMES[k]: v for k, v in class_counts.items()},
        'npy_dir':      str(output_dir / 'images'),
        'csv_path':     str(out_csv),
    }
    with open(output_dir / 'cache_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {output_dir / 'cache_manifest.json'}")


if __name__ == '__main__':
    main()
