"""
Artifact / Bright-Object Prevalence Analysis
=============================================
Detects images likely containing metal implants, pacemakers, or other
high-intensity foreign objects by computing the p99/median pixel intensity
ratio per image from the pre-cached .npy files.

Informs Fix G (weight_rec calibration):
  - If >15% of Cardiomegaly images are flagged, pure MSE at weight_rec=2.0
    risks chasing bright-artifact gradients.
  - Recommendation: Charbonnier loss (robust to outliers) or clipping at p99.9.

Outputs:
  scripts/artifact_prevalence.png — three panels:
    (A) p99/median histogram per class
    (B) Flagged-image % per class (bar chart)
    (C) Gallery of 4 most extreme flagged images per class

Usage:
    python scripts/analyze_artifacts.py
    python scripts/analyze_artifacts.py --threshold 4.0 --n_gallery 4 \\
        --output scripts/artifact_prevalence.png
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CLASS_NAMES = {
    3:  'Cardiomegaly',
    11: 'Pleural Thickening',
    14: 'No Finding (Normal)',
}
CLASS_COLORS = {
    3:  '#e84545',
    11: '#3a86ff',
    14: '#555555',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npy_dir',   default='/datasets/mmolefe/vinbigdata/cache_npy/images')
    p.add_argument('--csv',       default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--threshold', type=float, default=3.5,
                   help='p99/median ratio above which an image is flagged as artifact.')
    p.add_argument('--n_samples', type=int, default=2000,
                   help='Max images to sample per class for ratio computation.')
    p.add_argument('--n_gallery', type=int, default=4,
                   help='Number of extreme examples to show in gallery panel.')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    default='scripts/artifact_prevalence.png')
    return p.parse_args()


def get_class_image_ids(csv_path, focus_classes):
    all_rows = defaultdict(set)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            all_rows[row['image_id']].add(int(row['class_id']))

    result = defaultdict(set)
    for iid, cids in all_rows.items():
        for cid in focus_classes:
            if cid == 14:
                if cids == {14}:
                    result[14].add(iid)
            else:
                if cid in cids:
                    result[cid].add(iid)
    return result


def load_npy(npy_dir: Path, image_id: str):
    path = npy_dir / f'{image_id}.npy'
    if not path.exists():
        return None
    arr = np.load(str(path)).astype(np.float32)
    return arr


def compute_artifact_stats(image_ids, npy_dir, threshold, n_samples, seed):
    """
    For each image compute p99/median ratio.
    Returns:
        ratios:    list of float — one per image
        flagged:   list of (ratio, image_id) — only flagged images, sorted desc
        n_loaded:  int
    """
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n_samples]

    ratios  = []
    flagged = []
    for iid in ids:
        arr = load_npy(npy_dir, iid)
        if arr is None:
            continue
        flat   = arr.flatten()
        median = np.median(flat)
        p99    = np.percentile(flat, 99)
        if median < 1e-6:
            continue
        ratio = p99 / median
        ratios.append(ratio)
        if ratio > threshold:
            flagged.append((ratio, iid))

    flagged.sort(reverse=True)
    return np.array(ratios), flagged, len(ratios)


def visualise(stats_dict, npy_dir, threshold, n_gallery, output_path):
    classes     = sorted(stats_dict.keys())
    n_classes   = len(classes)
    n_gallery   = min(n_gallery, 4)

    # Figure layout: top row = Panel A + B, bottom row = gallery rows
    fig = plt.figure(figsize=(20, 5 + 3.5 * n_classes))
    fig.suptitle(
        f'Artifact / Bright-Object Prevalence (threshold = p99/median > {threshold:.1f})\n'
        'Informs weight_rec calibration: high prevalence → consider Charbonnier loss',
        fontsize=13, fontweight='bold',
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                              height_ratios=[1.6, 1.0 * n_classes])

    # ── Top row: histogram + bar chart ────────────────────────────────────────
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.3)
    ax_hist = fig.add_subplot(top[0])
    ax_bar  = fig.add_subplot(top[1])

    # Panel A: Histogram of p99/median ratios
    ax_hist.set_title('(A) p99/median ratio distribution per class', fontsize=11)
    for cid in classes:
        ratios, _, n = stats_dict[cid]
        if n == 0:
            continue
        ax_hist.hist(ratios, bins=60, range=(1.0, max(10.0, ratios.max())),
                     alpha=0.5, color=CLASS_COLORS[cid],
                     label=f'{CLASS_NAMES[cid]} (n={n})', density=True)
    ax_hist.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                    label=f'Threshold = {threshold:.1f}')
    ax_hist.set_xlabel('p99 / median pixel intensity ratio')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Panel B: % flagged per class
    ax_bar.set_title('(B) % images flagged as likely artifact', fontsize=11)
    pct_flagged = []
    for cid in classes:
        ratios, flagged, n = stats_dict[cid]
        pct = 100.0 * len(flagged) / n if n > 0 else 0
        pct_flagged.append(pct)

    bars = ax_bar.bar([CLASS_NAMES[c] for c in classes], pct_flagged,
                      color=[CLASS_COLORS[c] for c in classes], alpha=0.8,
                      edgecolor='black', linewidth=0.7)
    ax_bar.axhline(15, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                   label='15% risk threshold')
    ax_bar.set_ylabel('% images with p99/median > threshold')
    ax_bar.set_ylim(0, max(max(pct_flagged) * 1.3, 20))
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis='y', alpha=0.3)

    for bar, pct in zip(bars, pct_flagged):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    # Risk annotation
    max_pct = max(pct_flagged)
    if max_pct > 15:
        risk_color = '#e74c3c'
        risk_msg   = (f'Max artifact rate = {max_pct:.1f}% > 15%\n'
                      '→ weight_rec=2.0 + MSE may over-fit to artifacts\n'
                      '→ Consider Charbonnier loss or p99.9 pixel clipping')
    elif max_pct > 8:
        risk_color = '#f39c12'
        risk_msg   = (f'Max artifact rate = {max_pct:.1f}% (moderate)\n'
                      '→ weight_rec=2.0 is acceptable\n'
                      '→ Monitor reconstruction loss on flagged images')
    else:
        risk_color = '#2ecc71'
        risk_msg   = (f'Max artifact rate = {max_pct:.1f}% (low)\n'
                      '→ weight_rec=2.0 is safe\n'
                      '→ MSE reconstruction reliable')
    ax_bar.text(0.98, 0.97, risk_msg, transform=ax_bar.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', fc=risk_color,
                          alpha=0.15, ec=risk_color, linewidth=1.5))

    # ── Bottom row: gallery of extreme examples ────────────────────────────────
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_classes, n_gallery, subplot_spec=outer[1], hspace=0.3, wspace=0.05
    )

    for row, cid in enumerate(classes):
        _, flagged_list, _ = stats_dict[cid]
        extreme = flagged_list[:n_gallery]

        for col in range(n_gallery):
            ax = fig.add_subplot(bottom[row, col])
            if col < len(extreme):
                ratio, iid = extreme[col]
                arr = load_npy(npy_dir, iid)
                if arr is not None:
                    if arr.ndim == 3:
                        arr = arr.squeeze()
                    # Clip display at p99.5 for visibility
                    vmax = np.percentile(arr, 99.5)
                    ax.imshow(arr, cmap='gray', vmin=0, vmax=vmax, aspect='auto')
                    ax.set_title(f'ratio={ratio:.1f}', fontsize=7, color='red')
                else:
                    ax.text(0.5, 0.5, 'file\nmissing', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7)
            else:
                ax.set_visible(False)
                continue

            ax.axis('off')
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cid], fontsize=8, labelpad=2)

    # Row header labels (left side)
    for row, cid in enumerate(classes):
        ax_lbl = fig.add_subplot(bottom[row, 0])
        ax_lbl.axis('off')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    args    = parse_args()
    npy_dir = Path(args.npy_dir)
    csv_path = Path(args.csv)

    focus = {3, 11, 14}
    print('Loading class → image_id mapping...')
    class_ids = get_class_image_ids(csv_path, focus)
    for cid in sorted(focus):
        print(f'  Class {cid} ({CLASS_NAMES[cid]}): {len(class_ids[cid]):,} images')

    stats_dict = {}
    for cid in sorted(focus):
        print(f'\nProcessing class {cid} ({CLASS_NAMES[cid]})...')
        ratios, flagged, n = compute_artifact_stats(
            class_ids[cid], npy_dir, args.threshold, args.n_samples, args.seed
        )
        stats_dict[cid] = (ratios, flagged, n)
        pct = 100.0 * len(flagged) / n if n > 0 else 0
        print(f'  Loaded {n} images')
        print(f'  p99/median > {args.threshold}: {len(flagged)} images ({pct:.1f}%)')
        if flagged:
            print(f'  Most extreme ratios: ' +
                  ', '.join(f'{r:.1f}' for r, _ in flagged[:5]))

    print('\nRendering figure...')
    visualise(stats_dict, npy_dir, args.threshold, args.n_gallery, args.output)

    # Summary recommendation
    print('\n=== Recommendation ===')
    for cid in sorted(focus):
        ratios, flagged, n = stats_dict[cid]
        if n == 0:
            continue
        pct = 100.0 * len(flagged) / n
        suffix = ''
        if pct > 15:
            suffix = '  ← consider Charbonnier loss or pixel clipping'
        elif pct > 8:
            suffix = '  ← monitor closely'
        print(f'  {CLASS_NAMES[cid]}: {pct:.1f}% flagged{suffix}')


if __name__ == '__main__':
    main()
