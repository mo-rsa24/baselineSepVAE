"""
Class frequency bar chart for VinBigData (all 15 classes).

Illustrates the long-tail distribution of medical imaging data:
  - "No Finding" dominates the dataset
  - Disease classes are far less frequent and unevenly distributed
  - Motivates the balanced triplet sampling strategy used in SepVAE training

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_class_distribution.py \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --output   scripts/class_distribution.png
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# SepVAE training classes — highlighted in the chart
SEPVAE_CLASSES = {"No finding", "Cardiomegaly", "Pleural thickening"}

# Full ordered class list (class_id 0–14)
CLASS_ORDER = [
    "No finding",           # 14 — dominant healthy class
    "Aortic enlargement",   #  0
    "Cardiomegaly",         #  3  ← SepVAE
    "Pleural thickening",   # 11  ← SepVAE
    "Pleural effusion",     # 10
    "Nodule/Mass",          #  8
    "Pulmonary fibrosis",   # 13
    "ILD",                  #  5
    "Other lesion",         #  9
    "Lung Opacity",         #  7
    "Atelectasis",          #  1
    "Calcification",        #  2
    "Consolidation",        #  4
    "Infiltration",         #  6
    "Pneumothorax",         # 12
]


def count_unique_images_per_class(csv_path: str) -> dict[str, int]:
    """Count unique image_ids per class (one image may have multiple radiologist annotations)."""
    class_images: dict[str, set] = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            cls = row['class_name'].strip()
            iid = row['image_id'].strip()
            class_images.setdefault(cls, set()).add(iid)
    return {cls: len(ids) for cls, ids in class_images.items()}


def visualize(args):
    print(f"Reading {args.csv_path} …")
    counts = count_unique_images_per_class(args.csv_path)

    # Align to our preferred order; fall back gracefully for unknown classes
    known   = [c for c in CLASS_ORDER if c in counts]
    unknown = [c for c in counts if c not in CLASS_ORDER]
    ordered_classes = known + unknown
    values = [counts[c] for c in ordered_classes]
    total_images = len(set())   # placeholder — compute below

    # Total unique images (any class)
    with open(args.csv_path, newline='') as f:
        all_ids = {row['image_id'] for row in csv.DictReader(f)}
    total_images = len(all_ids)

    # ── colours ───────────────────────────────────────────────────────────────
    no_finding_colour = '#4a90d9'   # blue  — dominant healthy class
    sepvae_colour     = '#e05c2a'   # orange-red — SepVAE training classes
    other_colour      = '#9e9e9e'   # grey  — remaining disease classes

    bar_colours = []
    for cls in ordered_classes:
        if cls == "No finding":
            bar_colours.append(no_finding_colour)
        elif cls in SEPVAE_CLASSES:
            bar_colours.append(sepvae_colour)
        else:
            bar_colours.append(other_colour)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(ordered_classes))
    bars = ax.bar(x, values, color=bar_colours, edgecolor='white',
                  linewidth=0.6, width=0.72, zorder=3)

    # Value labels on each bar
    for bar, val in zip(bars, values):
        pct = 100 * val / total_images
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 60,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=6.8, color='#333333',
        )

    # ── Annotations ──────────────────────────────────────────────────────────
    # Long-tail bracket
    tail_start = ordered_classes.index("Pleural effusion") if "Pleural effusion" in ordered_classes else 4
    # ax.annotate(
    #     '"Long tail" — rare disease classes',
    #     xy=(tail_start, values[tail_start] * 1.05),
    #     xytext=(tail_start + 3, max(values) * 0.55),
    #     fontsize=8.5, color='#555555',
    #     arrowprops=dict(arrowstyle='->', color='#777777', lw=1.2),
    # )

    # SepVAE bracket — span the two highlighted bars
    cardio_x  = ordered_classes.index("Cardiomegaly")  if "Cardiomegaly"       in ordered_classes else None
    plthick_x = ordered_classes.index("Pleural thickening") if "Pleural thickening" in ordered_classes else None
    # if cardio_x is not None and plthick_x is not None:
    #     lo, hi = min(cardio_x, plthick_x), max(cardio_x, plthick_x)
    #     y_bracket = max(values[lo], values[hi]) * 1.38
    #     ax.annotate(
    #         '', xy=(lo - 0.35, y_bracket), xytext=(hi + 0.35, y_bracket),
    #         arrowprops=dict(arrowstyle='<->', color=sepvae_colour, lw=1.5),
    #     )
        # ax.text((lo + hi) / 2, y_bracket + max(values) * 0.01,
        #         'SepVAE training classes\n(balanced sampling)', ha='center',
        #         fontsize=8, color=sepvae_colour, fontweight='bold')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_classes, rotation=35, ha='right', fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_ylabel("Number of unique images", fontsize=10)
    ax.set_ylim(0, max(values) * 1.55)
    ax.set_xlim(-0.6, len(ordered_classes) - 0.4)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Colour legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=no_finding_colour, label=f'"No finding"  (n={counts.get("No finding",0):,})'),
        Patch(facecolor=sepvae_colour,     label='SepVAE training classes'),
        Patch(facecolor=other_colour,      label='Other disease classes'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right',
              framealpha=0.85, edgecolor='#cccccc')

    ax.set_title(
        f"VinBigData — Class distribution across {total_images:,} unique chest X-rays\n"
        f"\"No finding\" dominates, motivating balanced sampling across pathology classes",
        fontsize=11, pad=10,
    )

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")

    # Console summary
    print(f"\nClass distribution (unique images per class):")
    print(f"{'Class':<25} {'Images':>7}  {'% of total':>10}")
    print("-" * 46)
    for cls, val in zip(ordered_classes, values):
        marker = " ◄ SepVAE" if cls in SEPVAE_CLASSES else ""
        print(f"{cls:<25} {val:>7,}  {100*val/total_images:>9.1f}%{marker}")
    print(f"\nTotal unique images: {total_images:,}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--output',   default='scripts/class_distribution.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
