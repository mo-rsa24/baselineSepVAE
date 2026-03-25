"""
Spatial Orthogonality Ranking
==============================
Computes the Bhattacharyya coefficient (BC) between the aggregate bbox density
maps of every pair of annotated disease classes in VinBigData, then:

  1. Prints a ranked table — lowest BC first (most spatially distinct pairs)
  2. Saves a BC heatmap over all class pairs
  3. Highlights the Cardiomegaly × Pleural Thickening cell used in SepVAE

Lower BC → hotspots barely overlap → stronger spatial independence argument.

Usage:
    cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE
    python scripts/visualize_spatial_orthogonality_ranking.py \
        --dicom_dir /datasets/mmolefe/vinbigdata/train \
        --csv_path  /datasets/mmolefe/vinbigdata/train.csv \
        --output    scripts/spatial_orthogonality_ranking.png
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CLASS_NAMES = {
    0: "Aortic enlarg.",   1: "Atelectasis",      2: "Calcification",
    3: "Cardiomegaly",     4: "Consolidation",    5: "ILD",
    6: "Infiltration",     7: "Lung Opacity",     8: "Nodule/Mass",
    9: "Other lesion",    10: "Pleural effusion", 11: "Pleural thick.",
   12: "Pneumothorax",    13: "Pulm. fibrosis",
   # 14 (No finding) excluded — no bbox annotations
}

GRID  = 64
SIGMA = 1.5

# Pairs of interest to call out explicitly in the thesis
HIGHLIGHT_PAIRS = {
    (3, 11): "SepVAE pair",
    (3, 10): "Cardio + Effusion\n(edema proxy)",
    (4, 10): "Consolidation +\nEffusion (pneumonia\nproxy)",
    (12, 10): "Pneumothorax +\nEffusion",
}


# ── Core helpers (same as visualize_spatial_orthogonality.py) ─────────────────

def build_density(bboxes: list, grid: int = GRID) -> np.ndarray:
    acc   = np.zeros((grid, grid), dtype=np.float64)
    edges = np.linspace(0, 1, grid + 1)
    for x0, y0, x1, y1 in bboxes:
        cx0 = max(0,    int(np.searchsorted(edges, x0, 'right') - 1))
        cy0 = max(0,    int(np.searchsorted(edges, y0, 'right') - 1))
        cx1 = min(grid, int(np.searchsorted(edges, x1, 'left')))
        cy1 = min(grid, int(np.searchsorted(edges, y1, 'left')))
        acc[cy0:cy1, cx0:cx1] += 1.0
    acc = gaussian_filter(acc, sigma=SIGMA)
    if acc.max() > 0:
        acc /= acc.max()
    return acc.astype(np.float32)


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (a.sum() + 1e-12)
    b_n = b / (b.sum() + 1e-12)
    return float(np.sum(np.sqrt(a_n * b_n)))


def get_dicom_dims(dicom_dir: Path, image_ids: set) -> dict:
    import pydicom
    cache_path = dicom_dir / "dim_cache.json"
    dims = {}
    if cache_path.exists():
        with open(cache_path) as f:
            dims = {k: tuple(v) for k, v in json.load(f).items()}
    missing = image_ids - set(dims)
    if missing:
        print(f"  Reading DICOM metadata for {len(missing):,} images …")
        new_dims = {}
        for i, iid in enumerate(sorted(missing), 1):
            p = dicom_dir / f"{iid}.dicom"
            if not p.exists():
                continue
            try:
                dcm = pydicom.dcmread(str(p), stop_before_pixels=True)
                dims[iid] = (int(dcm.Rows), int(dcm.Columns))
                new_dims[iid] = list(dims[iid])
            except Exception:
                pass
            if i % 500 == 0:
                print(f"    {i}/{len(missing)}")
        merged = {k: list(v) for k, v in dims.items()}
        merged.update(new_dims)
        with open(cache_path, "w") as f:
            json.dump(merged, f)
        print(f"  Dimension cache updated → {cache_path}")
    return dims


def load_all_bboxes(csv_path: str, dicom_dir: Path) -> dict:
    raw = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            xmn = row.get('x_min', '').strip()
            if not xmn:
                continue
            try:
                cid = int(row['class_id'])
                bb  = (cid, float(row['x_min']), float(row['y_min']),
                       float(row['x_max']), float(row['y_max']))
            except ValueError:
                continue
            raw.setdefault(row['image_id'], []).append(bb)

    dims = get_dicom_dims(dicom_dir, set(raw))

    by_class: dict[int, list] = {}
    for iid, entries in raw.items():
        if iid not in dims:
            continue
        H, W = dims[iid]
        for cid, x0, y0, x1, y1 in entries:
            norm = (max(0., min(1., x0/W)), max(0., min(1., y0/H)),
                    max(0., min(1., x1/W)), max(0., min(1., y1/H)))
            by_class.setdefault(cid, []).append(norm)
    return by_class


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(args):
    print(f"Loading bboxes from {args.csv_path} …")
    by_class = load_all_bboxes(args.csv_path, Path(args.dicom_dir))

    # Keep only classes that exist in our registry AND have annotations
    available = sorted(cid for cid in CLASS_NAMES if cid in by_class and len(by_class[cid]) >= 10)
    print(f"\nClasses with ≥10 annotations: "
          f"{[CLASS_NAMES[c] for c in available]}")

    # Build density maps once per class
    print("\nBuilding density maps …")
    densities = {cid: build_density(by_class[cid]) for cid in available}

    # Compute BC for every pair
    n = len(available)
    bc_matrix = np.full((n, n), np.nan)
    results = []
    for i, j in combinations(range(n), 2):
        ci, cj = available[i], available[j]
        bc = bhattacharyya(densities[ci], densities[cj])
        bc_matrix[i, j] = bc
        bc_matrix[j, i] = bc
        results.append((bc, ci, cj))

    results.sort()   # lowest BC first → most spatially distinct

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'BC':>6}  {'Class A':<24} {'Class B':<24}  Note")
    print("-" * 80)
    for rank, (bc, ci, cj) in enumerate(results, 1):
        note = HIGHLIGHT_PAIRS.get((ci, cj), HIGHLIGHT_PAIRS.get((cj, ci), ''))
        note = note.replace('\n', ' ')
        print(f"{rank:<5} {bc:>6.4f}  {CLASS_NAMES[ci]:<24} {CLASS_NAMES[cj]:<24}  {note}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    labels = [CLASS_NAMES[c] for c in available]
    n_ann  = [len(by_class[c]) for c in available]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(bc_matrix, cmap='RdYlGn_r', vmin=0.3, vmax=1.0,
                   aspect='equal')

    # Cell annotations
    for i in range(n):
        for j in range(n):
            if np.isnan(bc_matrix[i, j]):
                continue
            val = bc_matrix[i, j]
            text_col = 'white' if val > 0.82 or val < 0.42 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=text_col)

    # Highlight cells for pairs of interest
    for (ci, cj), label in HIGHLIGHT_PAIRS.items():
        if ci in available and cj in available:
            ii, jj = available.index(ci), available.index(cj)
            for (ri, rj) in [(ii, jj), (jj, ii)]:
                ax.add_patch(plt.Rectangle(
                    (rj - 0.5, ri - 0.5), 1, 1,
                    fill=False, edgecolor='black', linewidth=2.5, zorder=5,
                ))

    # Diagonal label: class name + annotation count
    for i in range(n):
        ax.text(i, i, f'n={n_ann[i]:,}', ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='#444', alpha=0.7))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
    cbar.set_label("Bhattacharyya coefficient\n(lower = more spatially distinct)",
                   fontsize=9)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Legend for highlighted cells
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], color='black', lw=2.5, label='Pairs of interest (outlined)'),
        Patch(facecolor=plt.cm.RdYlGn_r(0.0),  label='Low BC → spatially distinct ✓'),
        Patch(facecolor=plt.cm.RdYlGn_r(1.0),  label='High BC → spatially similar ✗'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.18, 1.0), framealpha=0.85)

    # Annotate outlined pairs with labels
    for (ci, cj), label in HIGHLIGHT_PAIRS.items():
        if ci in available and cj in available:
            ii, jj = available.index(ci), available.index(cj)
            bc_val = bc_matrix[ii, jj]
            short  = label.split('\n')[0]
            ax.text(jj + 0.45, ii - 0.45, short,
                    fontsize=5.5, color='black', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75))

    ax.set_title(
        "Spatial Orthogonality Ranking — Bhattacharyya coefficient for all disease pairs\n"
        "Green = low overlap (spatially independent) · Red = high overlap · "
        "Black border = pairs of interest",
        fontsize=10, pad=12,
    )

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"\nSaved → {output}")

    # Print top-5 most and least orthogonal pairs
    print("\nTop-5 most spatially DISTINCT pairs (lowest BC):")
    for bc, ci, cj in results[:5]:
        print(f"  BC={bc:.4f}  {CLASS_NAMES[ci]}  ×  {CLASS_NAMES[cj]}")
    print("\nTop-5 most spatially SIMILAR pairs (highest BC):")
    for bc, ci, cj in results[-5:][::-1]:
        print(f"  BC={bc:.4f}  {CLASS_NAMES[ci]}  ×  {CLASS_NAMES[cj]}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom_dir', default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',  default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--output',    default='scripts/spatial_orthogonality_ranking.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
