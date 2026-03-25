"""
Bbox Quality Analysis
=====================
For Cardiomegaly (class 3) and Pleural Thickening (class 11):
  - Bbox area distribution (as % of image area) — violin + box plots
  - Inter-annotator IoU for images with multiple radiologist annotations

Outputs: scripts/bbox_quality.png

Informs:
  - Whether σ = bbox_width / 4 in BboxCrossAttnHead is well-calibrated
  - Whether weight_bbox_attn=0.1 is appropriate (high IoU → can push higher)

Usage:
    python scripts/analyze_bbox_quality.py
    python scripts/analyze_bbox_quality.py --csv /path/to/train.csv \\
        --dicom_dir /path/to/train --output scripts/bbox_quality.png
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Class registry ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Aortic enlargement', 1: 'Atelectasis',    2: 'Calcification',
    3: 'Cardiomegaly',       4: 'Consolidation',  5: 'ILD',
    6: 'Infiltration',       7: 'Lung Opacity',   8: 'Nodule/Mass',
    9: 'Other lesion',      10: 'Pleural effusion', 11: 'Pleural thickening',
   12: 'Pneumothorax',      13: 'Pulmonary fibrosis', 14: 'No finding',
}

FOCUS_CLASSES = {3: 'Cardiomegaly', 11: 'Pleural Thickening'}
COLORS        = {3: '#e84545', 11: '#3a86ff'}  # red, blue


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',       default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--dicom_dir', default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--output',    default='scripts/bbox_quality.png')
    p.add_argument('--dim_cache', default=None,
                   help='Path to dim_cache.json (default: <dicom_dir>/dim_cache.json)')
    p.add_argument('--max_dims',  type=int, default=5000,
                   help='Max number of DICOM images to read for dim cache.')
    return p.parse_args()


def load_dim_cache(cache_path: Path):
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_dim_cache(cache_path: Path, cache: dict):
    with open(cache_path, 'w') as f:
        json.dump(cache, f)


def get_dicom_dims(image_id: str, dicom_dir: Path, dim_cache: dict) -> tuple:
    """Return (H, W) for a DICOM, using cache to avoid re-reading pixel data.
    Handles two cache formats:
        {'H': h, 'W': w}  — written by this script
        [H, W]            — written by visualize_spatial_orthogonality.py
    """
    if image_id in dim_cache:
        d = dim_cache[image_id]
        if isinstance(d, (list, tuple)):
            return int(d[0]), int(d[1])
        return int(d['H']), int(d['W'])
    try:
        import pydicom
        path = dicom_dir / f'{image_id}.dicom'
        if not path.exists():
            path = dicom_dir / f'{image_id}.dcm'
        dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
        H, W = int(dcm.Rows), int(dcm.Columns)
        dim_cache[image_id] = [H, W]
        return H, W
    except Exception:
        return None, None


def iou(a, b):
    """Intersection-over-Union for two bboxes [x0, y0, x1, y1] in pixel coords."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_annotations(csv_path: Path, focus_class_ids: set):
    """
    Returns:
        per_class: {class_id: {image_id: [(rad_id, x0, y0, x1, y1), ...]}}
    """
    per_class = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row['class_id'])
            if cid not in focus_class_ids:
                continue
            iid  = row['image_id']
            rid  = row.get('rad_id', 'unknown')
            x0   = float(row['x_min'])
            y0   = float(row['y_min'])
            x1   = float(row['x_max'])
            y1   = float(row['y_max'])
            if (x1 - x0) < 1 or (y1 - y0) < 1:
                continue   # degenerate box
            per_class[cid][iid].append((rid, x0, y0, x1, y1))
    return per_class


def compute_stats(per_class, dicom_dir, dim_cache):
    """
    Returns for each class:
        areas_pct  : list of bbox areas as % of image
        widths_norm : list of normalised bbox widths
        iou_vals   : list of pairwise IoU values (multi-annotated images only)
        n_single   : count of images with 1 annotator
        n_multi    : count of images with >1 annotator
    """
    results = {}
    for cid, img_map in per_class.items():
        areas_pct   = []
        widths_norm = []
        iou_vals    = []
        n_single = n_multi = 0

        for iid, annots in img_map.items():
            H, W = get_dicom_dims(iid, dicom_dir, dim_cache)
            if H is None:
                continue

            for (rid, x0, y0, x1, y1) in annots:
                w = x1 - x0
                h = y1 - y0
                areas_pct.append(100.0 * (w * h) / (W * H))
                widths_norm.append(w / W)

            # Pairwise IoU across all annotator pairs for this image
            if len(annots) == 1:
                n_single += 1
            else:
                n_multi += 1
                boxes = [(a[1], a[2], a[3], a[4]) for a in annots]
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        iou_vals.append(iou(boxes[i], boxes[j]))

        results[cid] = {
            'areas_pct':   np.array(areas_pct),
            'widths_norm': np.array(widths_norm),
            'iou_vals':    np.array(iou_vals),
            'n_single':    n_single,
            'n_multi':     n_multi,
        }
    return results


def visualise(stats, output_path):
    classes = sorted(stats.keys())   # [3, 11]
    names   = [FOCUS_CLASSES[c] for c in classes]
    colors  = [COLORS[c]        for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Bbox Quality Analysis — Cardiomegaly vs Pleural Thickening\n'
        'Informs BboxCrossAttnHead prior calibration and weight_bbox_attn',
        fontsize=13, fontweight='bold',
    )

    # ── Panel A: Bbox area distribution (violin + box) ────────────────────────
    ax = axes[0]
    ax.set_title('(A) Bbox area (% of image)', fontsize=11)

    area_data  = [stats[c]['areas_pct'] for c in classes]
    vp = ax.violinplot(area_data, positions=[1, 2], showmedians=True,
                       showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.5)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2)

    bp = ax.boxplot(area_data, positions=[1, 2], widths=0.15,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Bbox area (% of image area)')
    ax.grid(axis='y', alpha=0.3)

    for i, (c, name) in enumerate(zip(classes, names)):
        d = stats[c]['areas_pct']
        ax.text(i + 1, ax.get_ylim()[1] * 0.97,
                f'median={np.median(d):.1f}%\np95={np.percentile(d,95):.1f}%\nn={len(d):,}',
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # ── Panel B: Bbox width distribution ──────────────────────────────────────
    ax = axes[1]
    ax.set_title('(B) Bbox width (normalised [0,1])\nσ = width/4 prior calibration', fontsize=11)

    width_data = [stats[c]['widths_norm'] for c in classes]
    vp2 = ax.violinplot(width_data, positions=[1, 2], showmedians=True, showextrema=False)
    for i, body in enumerate(vp2['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.5)
    vp2['cmedians'].set_color('black')
    vp2['cmedians'].set_linewidth(2)

    bp2 = ax.boxplot(width_data, positions=[1, 2], widths=0.15, patch_artist=True,
                     showfliers=True,
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Normalised bbox width (x_max - x_min) / W')
    ax.grid(axis='y', alpha=0.3)

    # Annotate with σ = width/4 implied prior width
    for i, (c, name) in enumerate(zip(classes, names)):
        w = stats[c]['widths_norm']
        sigma_median = np.median(w) / 4
        sigma_p95    = np.percentile(w, 95) / 4
        ax.text(i + 1, ax.get_ylim()[1] * 0.97,
                f'median={np.median(w):.3f}\n'
                f'σ_prior(median)={sigma_median:.3f}\n'
                f'σ_prior(p95)={sigma_p95:.3f}',
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # ── Panel C: Inter-annotator IoU ──────────────────────────────────────────
    ax = axes[2]
    ax.set_title('(C) Inter-annotator IoU\n(images with ≥2 radiologist annotations)',
                 fontsize=11)

    has_iou = [c for c in classes if len(stats[c]['iou_vals']) > 0]
    if has_iou:
        iou_data = [stats[c]['iou_vals'] for c in has_iou]
        iou_names = [FOCUS_CLASSES[c] for c in has_iou]
        iou_colors = [COLORS[c] for c in has_iou]

        vp3 = ax.violinplot(iou_data, positions=list(range(1, len(has_iou)+1)),
                            showmedians=True, showextrema=False)
        for i, body in enumerate(vp3['bodies']):
            body.set_facecolor(iou_colors[i])
            body.set_alpha(0.5)
        vp3['cmedians'].set_color('black')
        vp3['cmedians'].set_linewidth(2)

        bp3 = ax.boxplot(iou_data, positions=list(range(1, len(has_iou)+1)),
                         widths=0.15, patch_artist=True, showfliers=True,
                         medianprops=dict(color='black', linewidth=2),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2),
                         flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for patch, color in zip(bp3['boxes'], iou_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(list(range(1, len(has_iou)+1)))
        ax.set_xticklabels(iou_names, fontsize=10)
        ax.set_ylabel('Pairwise IoU (annotator pairs)')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.5,
                   label='IoU = 0.5')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

        for i, (c, name) in enumerate(zip(has_iou, iou_names)):
            d = stats[c]['iou_vals']
            n_multi = stats[c]['n_multi']
            ax.text(i + 1, 0.02,
                    f'median={np.median(d):.2f}\n'
                    f'n_pairs={len(d):,}\n'
                    f'n_images={n_multi:,}',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        # Annotation: implication for weight_bbox_attn
        med_iou = np.median(np.concatenate(iou_data))
        if med_iou > 0.6:
            advice = f'Median IoU={med_iou:.2f} > 0.6\n→ bbox signal reliable\n→ weight_bbox_attn can push ≥ 0.15'
            color = '#2ecc71'
        elif med_iou > 0.4:
            advice = f'Median IoU={med_iou:.2f} ≈ 0.4–0.6\n→ moderate agreement\n→ weight_bbox_attn=0.1 appropriate'
            color = '#f39c12'
        else:
            advice = f'Median IoU={med_iou:.2f} < 0.4\n→ high disagreement\n→ weight_bbox_attn ≤ 0.05 recommended'
            color = '#e74c3c'
        ax.text(0.98, 0.98, advice, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', fc=color, alpha=0.15,
                          ec=color, linewidth=1.5))
    else:
        ax.text(0.5, 0.5, 'No multi-annotator images\nfound for focus classes',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('N/A')

    # ── Legend strips ──────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=COLORS[c], alpha=0.7, label=FOCUS_CLASSES[c])
               for c in classes]
    fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    args = parse_args()
    csv_path   = Path(args.csv)
    dicom_dir  = Path(args.dicom_dir)
    cache_path = Path(args.dim_cache) if args.dim_cache else dicom_dir / 'dim_cache.json'

    print('Loading dimension cache...')
    dim_cache = load_dim_cache(cache_path)

    print('Loading annotations from CSV...')
    per_class = load_annotations(csv_path, set(FOCUS_CLASSES.keys()))

    for cid, img_map in per_class.items():
        n_imgs   = len(img_map)
        n_annots = sum(len(v) for v in img_map.values())
        print(f'  Class {cid} ({FOCUS_CLASSES[cid]}): {n_imgs:,} images, {n_annots:,} annotations')

    print('Computing stats (reading DICOM dims for uncached images)...')
    stats = compute_stats(per_class, dicom_dir, dim_cache)

    print(f'Saving updated dim_cache ({len(dim_cache):,} entries)...')
    save_dim_cache(cache_path, dim_cache)

    print('\n=== Results ===')
    for cid in sorted(stats.keys()):
        s = stats[cid]
        print(f'\n{FOCUS_CLASSES[cid]} (class {cid}):')
        print(f'  Bbox area:   median={np.median(s["areas_pct"]):.1f}%  '
              f'p5={np.percentile(s["areas_pct"],5):.1f}%  '
              f'p95={np.percentile(s["areas_pct"],95):.1f}%')
        print(f'  Bbox width:  median={np.median(s["widths_norm"]):.3f}  '
              f'σ_prior(median)={np.median(s["widths_norm"])/4:.3f}')
        print(f'  Multi-annot: {s["n_multi"]:,} images  '
              f'({s["n_single"]:,} single-annotator)')
        if len(s['iou_vals']) > 0:
            print(f'  IoU:         median={np.median(s["iou_vals"]):.3f}  '
                  f'mean={np.mean(s["iou_vals"]):.3f}  '
                  f'n_pairs={len(s["iou_vals"]):,}')
        else:
            print('  IoU:         no multi-annotator pairs found')

    visualise(stats, args.output)


if __name__ == '__main__':
    main()
