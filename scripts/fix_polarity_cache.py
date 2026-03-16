"""
fix_polarity_cache.py — In-place polarity correction for existing .npy cache.

The original cache_dicoms.py did not check PhotometricInterpretation, so any
DICOM stored as MONOCHROME1 (bones dark, air bright) was cached with the
wrong polarity.

This script:
  1. Scans all DICOM files for MONOCHROME1 flag.
  2. For each affected image whose .npy exists, inverts it in-place:
         corrected = 65535 - cached_uint16
  3. Deletes valid_ids_*.json caches so the dataset re-validates on next run.

Run ONCE after updating cache_dicoms.py.  Safe to re-run — already-corrected
files are detected and skipped.

Usage:
    python scripts/fix_polarity_cache.py \\
        --dicom_dir  /datasets/mmolefe/vinbigdata/train \\
        --cache_dir  /datasets/mmolefe/vinbigdata/cache_npy \\
        --num_workers 16
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom


# ── Per-worker task ────────────────────────────────────────────────────────────

def check_and_fix(args):
    """
    Returns (image_id, status) where status is one of:
      'fixed'      — was MONOCHROME1, .npy inverted in-place
      'skip_mono2' — MONOCHROME2 (correct polarity, no action)
      'skip_nonpy' — MONOCHROME1 but no .npy found (not cached yet)
      'error'      — could not read DICOM
    """
    dicom_path, npy_path = args
    image_id = Path(dicom_path).stem

    try:
        dcm = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        photometric = str(
            getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
        ).strip()
    except Exception as e:
        return image_id, 'error', str(e)

    if photometric != 'MONOCHROME1':
        return image_id, 'skip_mono2', ''

    # MONOCHROME1 — check if .npy exists
    if not Path(npy_path).exists():
        return image_id, 'skip_nonpy', ''

    # Invert the cached uint16 array: corrected = 65535 - inverted
    try:
        arr = np.load(str(npy_path))           # uint16 (H, W)
        arr = (65535 - arr.astype(np.int32)).astype(np.uint16)
        np.save(str(npy_path), arr)
    except Exception as e:
        return image_id, 'error', str(e)

    return image_id, 'fixed', ''


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom_dir',   required=True,
                   help='Directory containing raw .dicom files')
    p.add_argument('--cache_dir',   required=True,
                   help='Cache root produced by cache_dicoms.py '
                        '(contains images/ sub-folder)')
    p.add_argument('--num_workers', type=int, default=8)
    args = p.parse_args()

    dicom_dir = Path(args.dicom_dir)
    cache_dir = Path(args.cache_dir)
    npy_dir   = cache_dir / 'images'

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM dir not found: {dicom_dir}")
    if not npy_dir.exists():
        raise FileNotFoundError(f"Cache images dir not found: {npy_dir}")

    dicom_files = sorted(dicom_dir.glob('*.dicom'))
    print(f"Found {len(dicom_files):,} DICOM files in {dicom_dir}")

    tasks = [
        (str(d), str(npy_dir / (d.stem + '.npy')))
        for d in dicom_files
    ]

    counts = {'fixed': 0, 'skip_mono2': 0, 'skip_nonpy': 0, 'error': 0}
    errors = []

    print(f"Scanning with {args.num_workers} workers…")
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(check_and_fix, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            image_id, status, msg = fut.result()
            counts[status] += 1
            if status == 'error':
                errors.append((image_id, msg))
            if i % 1000 == 0:
                print(f"  {i:,}/{len(tasks):,}  "
                      f"fixed={counts['fixed']}  "
                      f"mono2={counts['skip_mono2']}  "
                      f"no_npy={counts['skip_nonpy']}  "
                      f"err={counts['error']}")

    print(f"\nDone.")
    print(f"  Corrected (MONOCHROME1 → inverted .npy): {counts['fixed']:,}")
    print(f"  Already correct (MONOCHROME2):            {counts['skip_mono2']:,}")
    print(f"  MONOCHROME1 but not cached yet:           {counts['skip_nonpy']:,}")
    print(f"  Errors (DICOM unreadable):                {counts['error']:,}")

    if errors:
        print(f"\nFirst 10 errors:")
        for eid, emsg in errors[:10]:
            print(f"  {eid}: {emsg}")

    # Delete validity scan caches — dataset will re-validate on next DataLoader init
    for json_name in ('valid_ids_cache.json', 'valid_ids_pair_cache.json'):
        p = cache_dir / json_name
        if p.exists():
            p.unlink()
            print(f"Deleted scan cache: {p}")

    print("\nPolarity fix complete. Re-run training — no other changes needed.")


if __name__ == '__main__':
    main()
