"""
download_celeba.py — Reliable CelebA download for headless servers
==================================================================
torchvision's built-in CelebA download uses Google Drive and frequently
hits quota limits on servers. This script provides three fallback methods:

    Method 1: gdown (Google Drive) — fast if quota not hit
    Method 2: Kaggle API           — most reliable, needs kaggle.json
    Method 3: Manual instructions  — prints what to download and where to put it

Usage
-----
    # Try all methods automatically:
    conda run -n cxr python scripts/download_celeba.py

    # Specify root (default: ~/.cache/celeba):
    conda run -n cxr python scripts/download_celeba.py --root /data/celeba

    # Force a specific method:
    conda run -n cxr python scripts/download_celeba.py --method kaggle
    conda run -n cxr python scripts/download_celeba.py --method gdown
    conda run -n cxr python scripts/download_celeba.py --method manual

After download, verify:
    conda run -n cxr python scripts/download_celeba.py --verify --root /path/to/root

Directory layout torchvision expects (root/celeba/):
    root/celeba/img_align_celeba/      ← 202,599 jpg files
    root/celeba/list_attr_celeba.txt   ← attribute labels
    root/celeba/list_eval_partition.txt
    root/celeba/identity_CelebA.txt
    root/celeba/list_bbox_celeba.txt
    root/celeba/list_landmarks_align_celeba.txt
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


# ──────────────────────────── expected files ────────────────────────────────

ANNO_FILES = [
    "list_attr_celeba.txt",
    "list_eval_partition.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
]

# Google Drive file IDs (from torchvision source)
GDRIVE_IDS = {
    "img_align_celeba.zip":           "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "list_attr_celeba.txt":           "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "identity_CelebA.txt":            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
    "list_bbox_celeba.txt":           "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
    "list_landmarks_align_celeba.txt":"0B7EVK8r0v71pd0FJY3Blby1HUTQ",
    "list_eval_partition.txt":        "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
}


# ──────────────────────────── verification ──────────────────────────────────

def verify(root: Path) -> bool:
    celeba = root / "celeba"
    img_dir = celeba / "img_align_celeba"
    ok = True

    if not img_dir.is_dir():
        print(f"  [MISSING] {img_dir}")
        ok = False
    else:
        n = len(list(img_dir.glob("*.jpg")))
        if n < 200_000:
            print(f"  [PARTIAL] {img_dir}: only {n} images (expected ~202,599)")
            ok = False
        else:
            print(f"  [OK] img_align_celeba: {n} images")

    for fname in ANNO_FILES:
        fpath = celeba / fname
        if not fpath.exists():
            print(f"  [MISSING] {fpath}")
            ok = False
        else:
            print(f"  [OK] {fname}")

    return ok


def quick_torchvision_check(root: Path) -> bool:
    """Try loading the dataset with torchvision to confirm it's usable."""
    try:
        import torchvision.datasets as tvd
        ds = tvd.CelebA(root=str(root), split="valid", download=False,
                        target_type="attr")
        print(f"  torchvision CelebA loaded: {len(ds)} validation samples")
        return True
    except Exception as e:
        print(f"  torchvision load failed: {e}")
        return False


# ──────────────────────────── method 1: gdown ───────────────────────────────

def download_gdown(root: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("  gdown not installed. Run: pip install gdown")
        return False

    celeba = root / "celeba"
    celeba.mkdir(parents=True, exist_ok=True)
    print("  Downloading via gdown (Google Drive)...")

    # Download annotation files first (small, fast)
    for fname in ANNO_FILES:
        fpath = celeba / fname
        if fpath.exists():
            print(f"    {fname} already exists, skipping")
            continue
        fid = GDRIVE_IDS[fname]
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"    {fname} ...", end=" ", flush=True)
        try:
            gdown.download(url, str(fpath), quiet=True)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    # Download image zip (large ~1.4GB)
    zip_path = celeba / "img_align_celeba.zip"
    img_dir  = celeba / "img_align_celeba"
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) > 200_000:
        print("    img_align_celeba already extracted, skipping zip download")
    else:
        if not zip_path.exists():
            fid = GDRIVE_IDS["img_align_celeba.zip"]
            url = f"https://drive.google.com/uc?id={fid}"
            print(f"    img_align_celeba.zip (~1.4 GB) ...", end=" ", flush=True)
            try:
                gdown.download(url, str(zip_path), quiet=False)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                return False
        # Extract
        print(f"    Extracting img_align_celeba.zip ...", end=" ", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(celeba)
        print("done")
        zip_path.unlink()  # save disk space

    return True


# ──────────────────────────── method 2: kaggle ──────────────────────────────

def download_kaggle(root: Path) -> bool:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  kaggle not installed. Run: pip install kaggle")
        return False

    kaggle_json = Path("~/.kaggle/kaggle.json").expanduser()
    if not kaggle_json.exists():
        print("  ~/.kaggle/kaggle.json not found.")
        print("  Go to https://www.kaggle.com/settings → API → Create New Token")
        print("  Place the downloaded kaggle.json at ~/.kaggle/kaggle.json")
        print("  Then chmod 600 ~/.kaggle/kaggle.json")
        return False

    import subprocess
    celeba = root / "celeba"
    celeba.mkdir(parents=True, exist_ok=True)

    print("  Downloading via Kaggle API (jessicali9530/celeba-dataset)...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset",
         "-p", str(celeba), "--unzip"],
        capture_output=False
    )
    if result.returncode != 0:
        print("  Kaggle download failed.")
        return False

    # Kaggle dataset has a slightly different layout — fix it up
    _fix_kaggle_layout(celeba)
    return True


def _fix_kaggle_layout(celeba: Path):
    """
    Kaggle's celeba-dataset has:
        celeba/img_align_celeba/img_align_celeba/*.jpg
        celeba/list_attr_celeba.csv  (not .txt)
    torchvision expects:
        celeba/img_align_celeba/*.jpg
        celeba/list_attr_celeba.txt
    """
    # Flatten nested img dir if present
    nested = celeba / "img_align_celeba" / "img_align_celeba"
    flat   = celeba / "img_align_celeba"
    if nested.is_dir():
        print("  Fixing Kaggle nested image layout...", end=" ", flush=True)
        for f in nested.iterdir():
            f.rename(flat / f.name)
        nested.rmdir()
        print("done")

    # Kaggle anno files often come as .csv — torchvision needs space-separated .txt
    # Usually the partition file and attr file are already txt from this dataset,
    # but check and warn if not.
    for fname in ANNO_FILES:
        fpath = celeba / fname
        if not fpath.exists():
            # try .csv variant
            csv_path = celeba / fname.replace(".txt", ".csv")
            if csv_path.exists():
                print(f"  WARNING: {fname} missing but {csv_path.name} found.")
                print(f"  torchvision needs the original space-separated .txt format.")
                print(f"  You may need to download anno files separately (see --method manual).")


# ──────────────────────────── method 3: manual ──────────────────────────────

def print_manual_instructions(root: Path):
    celeba = root / "celeba"
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              MANUAL CELEBA DOWNLOAD INSTRUCTIONS                     ║
╚══════════════════════════════════════════════════════════════════════╝

Target directory: {celeba}/

── Step 1: Download image archive ────────────────────────────────────
  From: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
     → Align&Cropped Images → img_align_celeba.zip  (~1.4 GB)
  Or via Google Drive directly:
     https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM

── Step 2: Download annotation files ─────────────────────────────────
  From the same page (Anno/ folder):
     • list_attr_celeba.txt
     • list_eval_partition.txt
     • identity_CelebA.txt
     • list_bbox_celeba.txt
     • list_landmarks_align_celeba.txt

  Direct Google Drive links:
     list_attr_celeba.txt      https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U
     list_eval_partition.txt   https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk
     identity_CelebA.txt       https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS
     list_bbox_celeba.txt      https://drive.google.com/file/d/0B7EVK8r0v71pbThiMVRxWXZ4dU0
     list_landmarks_align_celeba.txt  https://drive.google.com/file/d/0B7EVK8r0v71pd0FJY3Blby1HUTQ

── Step 3: Extract and place ─────────────────────────────────────────
  mkdir -p {celeba}
  unzip img_align_celeba.zip -d {celeba}/
  mv list_attr_celeba.txt list_eval_partition.txt \\
     identity_CelebA.txt list_bbox_celeba.txt \\
     list_landmarks_align_celeba.txt {celeba}/

── Step 4: Verify ────────────────────────────────────────────────────
  conda run -n cxr python scripts/download_celeba.py --verify --root {root}

── Expected layout ───────────────────────────────────────────────────
  {celeba}/
  ├── img_align_celeba/          ← 202,599 .jpg files
  ├── list_attr_celeba.txt
  ├── list_eval_partition.txt
  ├── identity_CelebA.txt
  ├── list_bbox_celeba.txt
  └── list_landmarks_align_celeba.txt
""")


# ──────────────────────────── main ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download CelebA for miniSepVAE")
    p.add_argument("--root",   type=str, default="~/.cache/celeba",
                   help="Root directory (default: ~/.cache/celeba)")
    p.add_argument("--method", type=str, default="auto",
                   choices=["auto", "gdown", "kaggle", "manual"],
                   help="Download method (default: auto = try gdown then kaggle)")
    p.add_argument("--verify", action="store_true",
                   help="Only verify an existing download, don't download")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"CelebA root: {root}")
    print()

    if args.verify:
        print("=== Verifying existing download ===")
        ok = verify(root)
        if ok:
            print("\nChecking torchvision compatibility...")
            quick_torchvision_check(root)
            print("\n✓ CelebA is ready to use.")
        else:
            print("\n✗ Download incomplete. Re-run without --verify to download.")
        return

    # Check if already complete
    if verify(root):
        print("CelebA already downloaded and complete.")
        quick_torchvision_check(root)
        return

    print()

    if args.method == "manual":
        print_manual_instructions(root)
        return

    if args.method in ("auto", "gdown"):
        print("=== Method 1: gdown (Google Drive) ===")
        if download_gdown(root):
            print("\nVerifying...")
            if verify(root):
                quick_torchvision_check(root)
                print("\n✓ Done via gdown.")
                return
        if args.method == "gdown":
            sys.exit(1)
        print()

    if args.method in ("auto", "kaggle"):
        print("=== Method 2: Kaggle API ===")
        if download_kaggle(root):
            print("\nVerifying...")
            if verify(root):
                quick_torchvision_check(root)
                print("\n✓ Done via Kaggle.")
                return
        if args.method == "kaggle":
            sys.exit(1)
        print()

    print("=== Both automatic methods failed. Manual instructions: ===")
    print_manual_instructions(root)


if __name__ == "__main__":
    main()
