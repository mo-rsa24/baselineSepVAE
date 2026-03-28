"""
MedSAM cardiac segmentation test — refined prompting.

Two prompt strategies (--prompt_mode):

  anatomical   (default) — Fixed anatomical prior for PA CXR.
               The heart occupies a predictable region in every PA chest X-ray.
               No annotations required — works for normal and cardiomegaly images.

  expanded_bbox           — Take the VinBigData union annotation bbox and expand
               it vertically (+12% up, +10% down) to convert the flat
               CTR-width measurement box into a cardiac silhouette bbox.

Pool-based selection:
  --pool_size N candidates are sampled and all run through MedSAM.
  The top --n_images are kept by MedSAM's own predicted IoU score.
  This avoids showing bad masks caused by poor-quality source images.

Output figure:
  Col 0 — original CXR with prompt bbox (green) overlaid
  Col 1 — CXR with coral cardiac mask overlay + contour
  Col 2 — zoomed crop of the cardiac region

Requirements (beyond requirements.txt):
    pip install transformers accelerate

Usage:
    # Annotation-free anatomical prior (recommended):
    conda run -n cxr python scripts/test_medsam_cardiac.py

    # Compare with expanded annotation bbox:
    conda run -n cxr python scripts/test_medsam_cardiac.py --prompt_mode expanded_bbox

    conda run -n cxr python scripts/test_medsam_cardiac.py \\
        --prompt_mode anatomical --pool_size 60 --n_images 5 \\
        --output results/medsam_cardiac_v2.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# transformers ≥4.47 calls torch.get_default_device() which was added in
# PyTorch 2.3.  Monkey-patch for older torch builds (e.g. 2.2.x).
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: None

# ── dependency guard ──────────────────────────────────────────────────────────
try:
    from transformers import SamModel, SamProcessor
except ImportError:
    print(
        "\n[ERROR] transformers not installed.\n"
        "  pip install transformers accelerate\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
CSV_PATH  = "/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv"
NPY_DIR   = Path("/datasets/mmolefe/vinbigdata/cache_npy/images")
MEDSAM_ID = "wanglab/medsam-vit-base"
IMG_SIZE  = 512   # cached images are 512×512

# Fixed anatomical prior for the cardiac region in a PA CXR (normalised [0,1]).
# Covers: right cardiac border → left cardiac border, aortic arch → diaphragm.
# This is a deliberate over-estimate — MedSAM will refine within this region.
CARDIAC_PRIOR = (0.28, 0.35, 0.76, 0.83)   # (x0, y0, x1, y1)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_id: str) -> np.ndarray:
    """Load a cached uint16 .npy and decode to float32 [0, 1]."""
    arr = np.load(NPY_DIR / f"{image_id}.npy")
    return arr.astype(np.float32) / 65535.0   # (512, 512)


def sample_cardiomegaly(csv_path: str, n: int, seed: int) -> pd.DataFrame:
    """
    Return n rows (one per unique image_id) with union bbox across all
    radiologist annotations for that image.
    """
    df = pd.read_csv(csv_path)
    cardio = df[df["class_name"] == "Cardiomegaly"].copy()
    cardio = cardio.dropna(subset=["x_min_norm", "y_min_norm", "x_max_norm", "y_max_norm"])

    union = (
        cardio.groupby("image_id", sort=False)
        .agg(
            x_min_norm=("x_min_norm", "min"),
            y_min_norm=("y_min_norm", "min"),
            x_max_norm=("x_max_norm", "max"),
            y_max_norm=("y_max_norm", "max"),
        )
        .reset_index()
    )
    return union.sample(n=n, random_state=seed).reset_index(drop=True)


def expand_bbox(x0, y0, x1, y1, up=0.12, down=0.10, sides=0.03):
    """
    Convert a flat CTR-width measurement bbox into a cardiac silhouette bbox.

    Radiologists draw horizontal "width measurement" boxes for CTR annotation —
    these are very flat (AR ~2.5).  Expanding vertically by up/down captures
    the full cardiac height (aortic arch → diaphragm interface).
    """
    return (
        max(0.0, x0 - sides),
        max(0.0, y0 - up),
        min(1.0, x1 + sides),
        min(1.0, y1 + down),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM inference
# ─────────────────────────────────────────────────────────────────────────────

def run_medsam(
    model: SamModel,
    processor: SamProcessor,
    img_01: np.ndarray,
    bbox_norm: tuple,
    device: torch.device,
) -> tuple:
    """
    Run MedSAM on a single grayscale image.

    Uses multimask_output=True (3 candidates) and returns the mask with the
    highest predicted IoU score.

    Args:
        img_01:    float32 (H, W) in [0, 1]
        bbox_norm: (x0, y0, x1, y1) normalised [0, 1]

    Returns:
        mask:      bool (H, W)
        iou_score: float — MedSAM's predicted IoU for the chosen mask
    """
    h, w = img_01.shape
    x0, y0, x1, y1 = bbox_norm

    img_u8  = (img_01 * 255).clip(0, 255).astype(np.uint8)
    rgb_img = np.stack([img_u8] * 3, axis=-1)
    bbox_px = [[x0 * w, y0 * h, x1 * w, y1 * h]]

    inputs = processor(
        images=rgb_img,
        input_boxes=[bbox_px],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    # Pick the mask candidate with the highest predicted IoU
    iou_scores = outputs.iou_scores[0, 0]            # (3,)
    best_idx   = int(iou_scores.argmax().item())
    best_iou   = float(iou_scores[best_idx].item())

    # Post-process only the best mask
    best_pred = outputs.pred_masks[:, :, best_idx:best_idx+1, :, :]
    masks = processor.image_processor.post_process_masks(
        best_pred.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks[0][0][0].numpy(), best_iou    # bool (H, W), float


# ─────────────────────────────────────────────────────────────────────────────
# Mask stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_stats(mask: np.ndarray) -> dict:
    """Mask area, cardiac width, and approximate CTR (image-width denominator)."""
    if not mask.any():
        return {"mask_area_px": 0, "cardiac_width_px": 0, "approx_ctr": float("nan")}
    cols = np.where(np.any(mask, axis=0))[0]
    cardiac_width_px = int(cols[-1] - cols[0] + 1)
    mask_area_px     = int(mask.sum())
    approx_ctr       = cardiac_width_px / mask.shape[1]
    return {
        "mask_area_px":     mask_area_px,
        "cardiac_width_px": cardiac_width_px,
        "approx_ctr":       round(approx_ctr, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_bbox_rect(ax, bbox_norm, img_size=512, color="lime", lw=2, label=None):
    x0, y0, x1, y1 = bbox_norm
    rect = mpatches.Rectangle(
        (x0 * img_size, y0 * img_size),
        (x1 - x0) * img_size,
        (y1 - y0) * img_size,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(x0 * img_size + 3, y0 * img_size - 5, label,
                color=color, fontsize=6.5, va="bottom")


def plot_results(records: list, output_path: Path, prompt_mode: str):
    """
    records: list of dicts with keys:
        image_id, img, mask, iou_score, prompt_bbox, annot_bbox (may be None)
    """
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    mode_label = "Anatomical prior" if prompt_mode == "anatomical" else "Expanded annotation bbox"
    fig.suptitle(
        f"MedSAM Cardiac Segmentation — VinBigData Cardiomegaly\n"
        f"Prompt: {mode_label}  |  Selection: top-{n} by MedSAM IoU",
        fontsize=11, y=1.01,
    )
    axes[0, 0].set_title("CXR + prompt bbox (green)", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("Cardiac mask overlay",       fontsize=9, fontweight="bold")
    axes[0, 2].set_title("Zoomed cardiac region",      fontsize=9, fontweight="bold")

    for i, rec in enumerate(records):
        img   = rec["img"]
        mask  = rec["mask"]
        pbbox = rec["prompt_bbox"]
        abbox = rec["annot_bbox"]
        stats = compute_mask_stats(mask)
        short_id = rec["image_id"][:12] + "…"

        # ── col 0: CXR + prompt bbox ──────────────────────────────────────────
        ax0 = axes[i, 0]
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
        draw_bbox_rect(ax0, pbbox, color="lime",  lw=2, label="prompt")
        if abbox is not None:
            draw_bbox_rect(ax0, abbox, color="cyan", lw=1, label="annot")
        ax0.set_ylabel(
            f"{short_id}\nIoU={rec['iou_score']:.3f}",
            fontsize=7.5, rotation=0, labelpad=85, va="center",
        )
        ax0.axis("off")

        # ── col 1: mask overlay ───────────────────────────────────────────────
        ax1 = axes[i, 1]
        ax1.imshow(img, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*img.shape, 4), dtype=np.float32)
        overlay[mask, 0] = 0.95
        overlay[mask, 1] = 0.35
        overlay[mask, 2] = 0.20
        overlay[mask, 3] = 0.45
        ax1.imshow(overlay)
        ax1.contour(mask.astype(float), levels=[0.5], colors=["#FF5733"], linewidths=[1.2])
        ax1.set_xlabel(
            f"area={stats['mask_area_px']:,}px  "
            f"w={stats['cardiac_width_px']}px  "
            f"approx_ctr≈{stats['approx_ctr']:.3f}",
            fontsize=7,
        )
        ax1.axis("off")

        # ── col 2: zoomed crop ────────────────────────────────────────────────
        ax2 = axes[i, 2]
        x0, y0, x1, y1 = pbbox
        pad = 0.05
        r0 = int(max(0.0, y0 - pad) * IMG_SIZE)
        r1 = int(min(1.0, y1 + pad) * IMG_SIZE)
        c0 = int(max(0.0, x0 - pad) * IMG_SIZE)
        c1 = int(min(1.0, x1 + pad) * IMG_SIZE)

        crop_img  = img[r0:r1, c0:c1]
        crop_mask = mask[r0:r1, c0:c1]
        ax2.imshow(crop_img, cmap="gray", vmin=0, vmax=1)
        crop_ov = np.zeros((*crop_img.shape, 4), dtype=np.float32)
        crop_ov[crop_mask, :] = [0.95, 0.35, 0.20, 0.45]
        ax2.imshow(crop_ov)
        ax2.contour(crop_mask.astype(float), levels=[0.5], colors=["#FF5733"], linewidths=[1.5])
        ax2.axis("off")

    fig.text(
        0.5, -0.01,
        "approx_ctr uses image width as thoracic denominator — overestimates true CTR.\n"
        "True CTR requires lung segmentation (e.g., TorchXRayVision ChestX-Det).",
        ha="center", fontsize=7.5, color="gray",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM loader (safetensors fallback for torch < 2.6)
# ─────────────────────────────────────────────────────────────────────────────

def _load_medsam(model_id: str, device: torch.device):
    import shutil, tempfile
    from huggingface_hub import hf_hub_download

    try:
        model     = SamModel.from_pretrained(model_id).to(device).eval()
        processor = SamProcessor.from_pretrained(model_id)
        return model, processor
    except ValueError as e:
        if "CVE-2025-32434" not in str(e) and "upgrade torch" not in str(e):
            raise

    print("  [info] Converting MedSAM .bin → safetensors (torch < 2.6)…")
    bin_path  = hf_hub_download(model_id, "pytorch_model.bin")
    cfg_path  = hf_hub_download(model_id, "config.json")
    prep_path = hf_hub_download(model_id, "preprocessor_config.json")

    from safetensors.torch import save_file as st_save
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    tmpdir = Path(tempfile.mkdtemp(prefix="medsam_st_"))
    st_save(state_dict, str(tmpdir / "model.safetensors"))
    shutil.copy(cfg_path,  tmpdir / "config.json")
    shutil.copy(prep_path, tmpdir / "preprocessor_config.json")
    print(f"  [info] Safetensors written to {tmpdir}")

    model     = SamModel.from_pretrained(str(tmpdir)).to(device).eval()
    processor = SamProcessor.from_pretrained(str(tmpdir))
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MedSAM cardiac segmentation test — refined prompting")
    p.add_argument("--n_images",    type=int,   default=5,
                   help="Images to display (top-N by IoU from pool). Default 5.")
    p.add_argument("--pool_size",   type=int,   default=60,
                   help="Candidate pool size. All are run through MedSAM; top N kept. Default 60.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--prompt_mode", type=str,   default="anatomical",
                   choices=["anatomical", "expanded_bbox"],
                   help="anatomical: fixed PA-CXR cardiac prior (no annotations needed). "
                        "expanded_bbox: vertically expand VinBigData union bbox.")
    p.add_argument("--output", type=Path, default=Path("results/medsam_cardiac_v2.png"))
    p.add_argument("--cpu",    action="store_true", help="Force CPU.")
    p.add_argument("--min_area_px", type=int, default=10_000,
                   help="Minimum mask area in pixels to accept a segmentation. "
                        "Filters out wrong segmentations (ribs, mediastinum). Default 10000.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  prompt_mode: {args.prompt_mode}")

    # ── sample pool ───────────────────────────────────────────────────────────
    pool_n = max(args.pool_size, args.n_images)
    print(f"\nSampling pool of {pool_n} cardiomegaly images (seed={args.seed})…")
    pool = sample_cardiomegaly(CSV_PATH, n=pool_n, seed=args.seed)

    # ── load MedSAM ───────────────────────────────────────────────────────────
    print(f"\nLoading MedSAM from '{MEDSAM_ID}'…")
    model, processor = _load_medsam(MEDSAM_ID, device)
    print("Model loaded.")

    # ── run inference on entire pool ──────────────────────────────────────────
    print(f"\nRunning MedSAM on {pool_n} candidates…")
    header = (
        f"  {'image_id':<34}  {'area_px':>8}  {'card_w':>6}  "
        f"{'approx_ctr':>10}  {'iou':>6}  prompt"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    records = []
    for _, row in pool.iterrows():
        img = load_image(row["image_id"])

        # Annotation bbox (union of radiologist annotations)
        annot_bbox = (row["x_min_norm"], row["y_min_norm"],
                      row["x_max_norm"], row["y_max_norm"])

        # Choose prompt based on mode
        if args.prompt_mode == "anatomical":
            prompt_bbox = CARDIAC_PRIOR
        else:   # expanded_bbox
            prompt_bbox = expand_bbox(*annot_bbox)

        mask, iou = run_medsam(model, processor, img, prompt_bbox, device)
        stats = compute_mask_stats(mask)

        print(
            f"  {row['image_id']:<34}  "
            f"{stats['mask_area_px']:>8,}  "
            f"{stats['cardiac_width_px']:>6}  "
            f"{stats['approx_ctr']:>10.4f}  "
            f"{iou:>6.3f}  "
            f"{args.prompt_mode}"
        )

        records.append({
            "image_id":   row["image_id"],
            "img":        img,
            "mask":       mask,
            "iou_score":  iou,
            "prompt_bbox": prompt_bbox,
            "annot_bbox":  annot_bbox if args.prompt_mode == "expanded_bbox" else None,
        })

    # ── keep top N by IoU score (with minimum area filter) ───────────────────
    # Masks < min_area_px are likely wrong segmentations (ribs, mediastinum).
    # Typical cardiac mask on a 512×512 image is 15,000–45,000 px.
    min_area = args.min_area_px
    valid = [r for r in records
             if compute_mask_stats(r["mask"])["mask_area_px"] >= min_area]
    if len(valid) < args.n_images:
        print(f"  WARNING: only {len(valid)} records pass area≥{min_area}px filter "
              f"(out of {len(records)}); showing all valid records.")
    valid.sort(key=lambda r: r["iou_score"], reverse=True)
    top = valid[:args.n_images]

    print(f"\nTop {args.n_images} by IoU score (area≥{min_area}px filter applied):")
    for r in top:
        print(f"  {r['image_id'][:20]}…  IoU={r['iou_score']:.3f}  "
              f"area={compute_mask_stats(r['mask'])['mask_area_px']:,}px")

    # ── visualise ─────────────────────────────────────────────────────────────
    print("\nRendering figure…")
    plot_results(top, args.output, args.prompt_mode)
    print("Done.")


if __name__ == "__main__":
    main()
