"""
Co-occurrence & Scarcity Heatmap
==================================
Two-panel figure that provides empirical motivation for SepVAE's compositional
approach to rare co-morbidities.

Panel A — Co-occurrence counts (log scale)
  Diagonal  = how many images carry each disease individually (bright)
  Off-diag  = how many images carry BOTH diseases together (much darker)
  → Visually shows the training data is dominated by single-disease images.

Panel B — Scarcity ratio  (co-occurrence / min prevalence × 100 %)
  Each off-diagonal cell shows: of all images with disease A,
  what fraction ALSO has disease B?
  → A very low ratio for Cardiomegaly × Pleural Thickening means the model
    has almost no "A+B" examples to learn from directly — justifying
    compositional generation from independently learned heads.

"No finding" is excluded — it carries no spatial annotation and conflating
healthy images with disease co-occurrences muddies the clinical argument.

Flags:
    --single    Produce a single-panel figure (counts only, no scarcity ratio)
    --no-log    Use raw counts instead of log scale on the counts heatmap

Usage:
    # Default: two panels, log scale
    python scripts/visualize_cooccurrence_heatmap.py \
        --csv_path /datasets/mmolefe/vinbigdata/train.csv \
        --output   scripts/cooccurrence_heatmap.png

    # Single panel, linear scale
    python scripts/visualize_cooccurrence_heatmap.py \
        --single --no-log \
        --output scripts/cooccurrence_heatmap_single.png
"""

import argparse
import csv as csv_mod
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ── Class registry — No finding (14) excluded intentionally ───────────────────
CLASS_NAMES = {
    0:  "Aortic enlargement",
    1:  "Atelectasis",
    2:  "Calcification",
    3:  "Cardiomegaly",
    4:  "Consolidation",
    5:  "ILD",
    6:  "Infiltration",
    7:  "Lung Opacity",
    8:  "Nodule/Mass",
    9:  "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
}
IDS    = sorted(CLASS_NAMES)        # 0-13
N      = len(IDS)
LABELS = [CLASS_NAMES[i] for i in IDS]
ID_TO_IDX = {cid: idx for idx, cid in enumerate(IDS)}

# SepVAE target pair — given special callout
PAIR = (3, 11)   # Cardiomegaly, Pleural thickening


def build_matrix(csv_path: str):
    """
    Returns (N, N) matrix where:
      matrix[i, i] = # unique images with class i
      matrix[i, j] = # unique images with both class i and class j  (i ≠ j)
    Only counts classes 0-13 (No finding excluded).
    """
    image_classes: dict[str, set] = {}
    with open(csv_path, newline='') as f:
        for row in csv_mod.DictReader(f):
            cid = int(row['class_id'])
            if cid not in ID_TO_IDX:
                continue
            image_classes.setdefault(row['image_id'], set()).add(cid)

    mat = np.zeros((N, N), dtype=np.int32)
    for classes in image_classes.values():
        lst = [ID_TO_IDX[c] for c in sorted(classes) if c in ID_TO_IDX]
        for idx in lst:
            mat[idx, idx] += 1
        for ii, jj in combinations(lst, 2):
            mat[ii, jj] += 1
            mat[jj, ii] += 1
    return mat


def scarcity_ratio(mat: np.ndarray) -> np.ndarray:
    """
    ratio[i, j] = mat[i, j] / min(mat[i,i], mat[j,j])  (off-diagonal only)
    Diagonal is set to 1.0 (100%) so it reads as "fully present".
    """
    diag = np.diag(mat).astype(float)
    ratio = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                ratio[i, j] = 1.0
            else:
                denom = min(diag[i], diag[j])
                ratio[i, j] = mat[i, j] / denom if denom > 0 else 0.0
    return ratio


def add_cell_border(ax, row, col, color, lw=2.5, zorder=10):
    ax.add_patch(Rectangle(
        (col - 0.5, row - 0.5), 1, 1,
        fill=False, edgecolor=color, linewidth=lw, zorder=zorder,
    ))


def annotate_pair(ax, i, j, text, color='black', dx=0.6, dy=-0.6, fontsize=10):
    """Add a callout annotation pointing to cell (i,j)."""
    ax.annotate(
        text,
        xy=(j, i), xytext=(j + dx, i + dy),
        fontsize=fontsize, color=color, ha='left', va='bottom',
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color,
                  alpha=0.88, lw=1.0),
        zorder=20,
    )


def draw_heatmap(ax, data, cmap, norm, labels, title, fmt_fn,
                 cbar_label, text_thresh=0.5):
    """Generic heatmap renderer shared by both panels."""
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='equal')
    for i in range(N):
        for j in range(N):
            v = data[i, j]
            if v == 0:
                continue
            normed = norm(v)
            tc = 'white' if normed > text_thresh else '#222'
            fw = 'bold' if i == j else 'normal'
            ax.text(j, i, fmt_fn(v), ha='center', va='center',
                    fontsize=9, color=tc, fontweight=fw)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Bold + colour tick labels for the SepVAE pair
    for cid, ticks in [(PAIR[0], ax.get_xticklabels()),
                       (PAIR[1], ax.get_xticklabels())]:
        pass   # applied below
    for tick_list, axis_ids in [
        (ax.get_xticklabels(), IDS), (ax.get_yticklabels(), IDS)
    ]:
        for tick, cid in zip(tick_list, axis_ids):
            if cid in PAIR:
                tick.set_fontweight('bold')
                tick.set_color('#b5000a')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    return im


def visualize(args):
    print(f"Building co-occurrence matrix from {args.csv_path} …")
    mat   = build_matrix(args.csv_path)
    ratio = scarcity_ratio(mat)
    diag  = np.diag(mat)
    print("Done.")

    # Key numbers for the SepVAE pair
    pi, pj = ID_TO_IDX[PAIR[0]], ID_TO_IDX[PAIR[1]]
    n_a    = mat[pi, pi]
    n_b    = mat[pj, pj]
    n_ab   = mat[pi, pj]
    pct_a  = 100 * n_ab / n_a  if n_a > 0 else 0
    pct_b  = 100 * n_ab / n_b  if n_b > 0 else 0

    print(f"\n{'='*55}")
    print(f"  Cardiomegaly        individual : {n_a:>6,} images")
    print(f"  Pleural Thickening  individual : {n_b:>6,} images")
    print(f"  Co-occurring both              : {n_ab:>6,} images")
    print(f"  Scarcity ratio (% of Cardio)   : {pct_a:.1f}%")
    print(f"  Scarcity ratio (% of PlThick)  : {pct_b:.1f}%")
    print(f"{'='*55}")

    # ── Build counts display data (log or linear) ─────────────────────────────
    raw_mat = mat.astype(float)
    if args.no_log:
        disp_mat   = raw_mat
        disp_norm  = mcolors.Normalize(vmin=0, vmax=raw_mat.max())
        cbar_label = "# images"
        fmt_fn     = lambda v: f'{int(v):,}' if v > 0 else ''
        scale_tag  = "linear scale"
    else:
        disp_mat   = np.log1p(raw_mat)
        disp_norm  = mcolors.Normalize(vmin=0, vmax=disp_mat.max())
        cbar_label = "log(1 + # images)"
        fmt_fn     = lambda v: f'{int(np.expm1(v)):,}' if v > 0 else ''
        scale_tag  = "log₁₊ₓ scale"

    # ── Figure layout ─────────────────────────────────────────────────────────
    ncols     = 1 if args.single else 2
    fig_w     = 14 if args.single else 26
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 10),
                             constrained_layout=True)
    if args.single:
        axes = [axes]   # make iterable

    # ── Panel A: co-occurrence counts ─────────────────────────────────────────
    panel_label = "" if args.single else "(A)  "
    draw_heatmap(
        axes[0], disp_mat,
        cmap='YlOrRd', norm=disp_norm, labels=LABELS,
        title=f"{panel_label}VinBigData co-occurrence matrix  [{scale_tag}]",
        fmt_fn=fmt_fn,
        cbar_label=cbar_label,
        text_thresh=0.55,
    )
    add_cell_border(axes[0], pi, pj, color='#1a6fba', lw=3.0)
    add_cell_border(axes[0], pj, pi, color='#1a6fba', lw=3.0)
    annotate_pair(
        axes[0], pi, pj,
        f"Cardiomegaly ∩ Pleural Thickening\n"
        f"n={n_ab:,}  ({pct_a:.1f}% of Cardiomegaly,\n"
        f"{pct_b:.1f}% of Pleural Thickening)",
        color='#1a6fba', dx=1.2, dy=-2.5, fontsize=10,
    )

    # ── Panel B: scarcity ratio (skipped in --single mode) ────────────────────
    if not args.single:
        off_ratio = ratio.copy()
        np.fill_diagonal(off_ratio, 0.0)
        ratio_norm = mcolors.Normalize(vmin=0, vmax=off_ratio.max())

        draw_heatmap(
            axes[1], off_ratio,
            cmap='YlOrRd', norm=ratio_norm, labels=LABELS,
            title="(B)  Co-morbidity scarcity ratio  [% of rarer class]\n"
                  "cell(i,j) = co-occurrence / min(prevalence_i, prevalence_j) × 100",
            fmt_fn=lambda v: f'{v*100:.0f}%' if v > 0 else '',
            cbar_label="Co-morbidity rate  (% of rarer class)",
            text_thresh=0.55,
        )
        for i in range(N):
            axes[1].add_patch(Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                facecolor='#bbbbbb', edgecolor='white', linewidth=0.5, zorder=2,
            ))
            axes[1].text(i, i, f'{diag[i]:,}', ha='center', va='center',
                         fontsize=9, color='#333', fontweight='bold', zorder=3)
        add_cell_border(axes[1], pi, pj, color='#1a6fba', lw=3.0)
        add_cell_border(axes[1], pj, pi, color='#1a6fba', lw=3.0)
        annotate_pair(
            axes[1], pi, pj,
            f"Only {pct_a:.1f}% of Cardiomegaly patients\n"
            f"also have Pleural Thickening.\n"
            f"Model must compose from {n_a:,} + {n_b:,}\n"
            f"individual examples, not {n_ab:,} joint ones.",
            color='#1a6fba', dx=1.2, dy=-2.5, fontsize=8,
        )

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0],[0], color='#b5000a', lw=2,
               label='SepVAE training classes (red tick labels)'),
        Rectangle((0,0), 1, 1, fc='none', ec='#1a6fba', lw=2.5,
                  label='Cardiomegaly × Pleural Thickening (target pair)'),
    ]
    if not args.single:
        legend_handles.append(
            Rectangle((0,0), 1, 1, fc='#bbbbbb', ec='white',
                       label='Diagonal — individual prevalence (Panel B)')
        )
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    if not args.single:
        fig.suptitle(
            "VinBigData — Co-occurrence & Scarcity",
            fontsize=14, y=1.02,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"\nSaved → {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--output',   default='scripts/cooccurrence_heatmap.png')
    p.add_argument('--single',   action='store_true',
                   help='Produce a single-panel figure (counts only, no scarcity ratio)')
    p.add_argument('--no-log',   dest='no_log', action='store_true',
                   help='Use raw counts instead of log scale')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
