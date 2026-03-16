"""
fix_polarity_cache.py — Two-step in-place polarity correction for .npy cache.

Step 1  (machine with DICOMs — cluster):
    Scans DICOM headers to find MONOCHROME1 images.  Writes a JSON list of
    affected image IDs.  Does NOT need write access to the .npy cache.

        python scripts/fix_polarity_cache.py scan \\
            --dicom_dir /datasets/mmolefe/vinbigdata/train \\
            --out        scripts/monochrome1_ids.json \\
            --num_workers 16

Step 2  (machine with the .npy cache — cluster OR RunPod):
    Reads the JSON list, inverts affected .npy files in-place
    (corrected = 65535 − cached_uint16), and removes validity scan caches
    so the dataset re-validates on the next DataLoader start.

        python scripts/fix_polarity_cache.py apply \\
            --cache_dir /workspace/vinbigdata/cache_npy \\
            --id_list   scripts/monochrome1_ids.json

Workflow when DICOMs and cache live on different machines (RunPod scenario):
    1. Run `scan` on cluster.
    2. git add scripts/monochrome1_ids.json && git commit && git push
    3. git pull on RunPod.
    4. Run `apply` on RunPod.
    5. Launch training.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom


# ── Step 1: scan ──────────────────────────────────────────────────────────────

def _check_one(dicom_path: str):
    """Returns (image_id, is_mono1, error_msg)."""
    image_id = Path(dicom_path).stem
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        photo = str(getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')).strip()
        return image_id, photo == 'MONOCHROME1', ''
    except Exception as e:
        return image_id, False, str(e)


def cmd_scan(args):
    dicom_dir = Path(args.dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM dir not found: {dicom_dir}")

    dicom_files = sorted(dicom_dir.glob('*.dicom'))
    print(f"Scanning {len(dicom_files):,} DICOMs with {args.num_workers} workers…")

    mono1_ids, errors = [], []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_check_one, str(d)): d for d in dicom_files}
        for i, fut in enumerate(as_completed(futs), 1):
            image_id, is_mono1, err = fut.result()
            if is_mono1:
                mono1_ids.append(image_id)
            if err:
                errors.append((image_id, err))
            if i % 2000 == 0:
                print(f"  {i:,}/{len(dicom_files):,}  MONOCHROME1 so far: {len(mono1_ids):,}")

    mono1_ids.sort()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(mono1_ids, f, indent=2)

    print(f"\nScan complete.")
    print(f"  MONOCHROME1 images : {len(mono1_ids):,}")
    print(f"  Unreadable DICOMs  : {len(errors):,}")
    print(f"  ID list written    : {out}")
    if errors:
        print("  First 5 errors:")
        for eid, emsg in errors[:5]:
            print(f"    {eid}: {emsg}")


# ── Step 2: apply ─────────────────────────────────────────────────────────────

def _invert_npy(npy_path: str):
    """Invert one .npy in-place. Returns (path, status, msg)."""
    p = Path(npy_path)
    if not p.exists():
        return npy_path, 'missing', ''
    try:
        arr = np.load(str(p))                                    # uint16 (H, W)
        arr = (65535 - arr.astype(np.int32)).astype(np.uint16)
        np.save(str(p), arr)
        return npy_path, 'fixed', ''
    except Exception as e:
        return npy_path, 'error', str(e)


def cmd_apply(args):
    cache_dir = Path(args.cache_dir)
    npy_dir   = cache_dir / 'images'
    if not npy_dir.exists():
        raise FileNotFoundError(f"Cache images dir not found: {npy_dir}")

    with open(args.id_list) as f:
        mono1_ids = json.load(f)

    print(f"Applying polarity fix to {len(mono1_ids):,} MONOCHROME1 images…")

    npy_paths = [str(npy_dir / f"{iid}.npy") for iid in mono1_ids]
    counts = {'fixed': 0, 'missing': 0, 'error': 0}
    errors = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_invert_npy, p): p for p in npy_paths}
        for i, fut in enumerate(as_completed(futs), 1):
            path, status, msg = fut.result()
            counts[status] += 1
            if status == 'error':
                errors.append((path, msg))
            if i % 500 == 0:
                print(f"  {i:,}/{len(npy_paths):,}  "
                      f"fixed={counts['fixed']}  "
                      f"missing={counts['missing']}  "
                      f"err={counts['error']}")

    print(f"\nApply complete.")
    print(f"  Fixed   : {counts['fixed']:,}")
    print(f"  Missing : {counts['missing']:,}  (not in this cache — OK)")
    print(f"  Errors  : {counts['error']:,}")
    if errors:
        print("  First 5 errors:")
        for ep, em in errors[:5]:
            print(f"    {ep}: {em}")

    # Remove validity scan caches — dataset re-validates on next DataLoader init
    for json_name in ('valid_ids_cache.json', 'valid_ids_pair_cache.json'):
        p = cache_dir / json_name
        if p.exists():
            p.unlink()
            print(f"Deleted scan cache: {p}")

    print("\nPolarity fix complete. Launch training.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('scan', help='Scan DICOMs → write MONOCHROME1 ID list')
    s.add_argument('--dicom_dir',   required=True)
    s.add_argument('--out',         default='scripts/monochrome1_ids.json')
    s.add_argument('--num_workers', type=int, default=8)

    a = sub.add_parser('apply', help='Invert cached .npy files from ID list')
    a.add_argument('--cache_dir',   required=True)
    a.add_argument('--id_list',     default='scripts/monochrome1_ids.json')
    a.add_argument('--num_workers', type=int, default=8)

    args = p.parse_args()
    if args.cmd == 'scan':
        cmd_scan(args)
    else:
        cmd_apply(args)


if __name__ == '__main__':
    main()
