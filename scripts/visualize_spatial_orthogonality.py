"""
Spatial Orthogonality Density Plot
===================================
Aggregates ALL bounding box annotations for any two VinBigData disease classes
into side-by-side 2-D density maps and measures their spatial overlap.

Non-overlapping hotspots → spatial orthogonality → justifies independent
latent experts in a disentangled model.

VinBigData class IDs
---------------------
  0  Aortic enlargement      6  Infiltration
  1  Atelectasis             7  Lung Opacity
  2  Calcification           8  Nodule/Mass
  3  Cardiomegaly            9  Other lesion
  4  Consolidation          10  Pleural effusion
  5  ILD                    11  Pleural thickening
                            12  Pneumothorax
                            13  Pulmonary fibrosis
                            14  No finding

Suggested equivalent pairs for diseases not in VinBigData:
  "Pulmonary Edema"  → Pleural Effusion      (class_id 10)
  "Pneumonia"        → Consolidation          (class_id  4)

Example usage:
    # Default: Cardiomegaly vs Pleural Thickening
    python scripts/visualize_spatial_orthogonality.py

    # Cardiomegaly vs Pleural Effusion  (heart-failure equivalent of edema)
    python scripts/visualize_spatial_orthogonality.py --class_a 3 --class_b 10

    # Consolidation (pneumonia) vs Pleural Effusion
    python scripts/visualize_spatial_orthogonality.py --class_a 4 --class_b 10

    # Pneumothorax vs Pleural Effusion  (upper vs lower periphery)
    python scripts/visualize_spatial_orthogonality.py --class_a 12 --class_b 10

Notes
-----
Uses raw train.csv (pixel coordinates); normalises by actual DICOM dimensions
read from file metadata (stop_before_pixels — fast). Dimension cache written to
<dicom_dir>/dim_cache.json so subsequent runs are instant.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Class registry ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Aortic enlargement", 1: "Atelectasis",      2: "Calcification",
    3: "Cardiomegaly",       4: "Consolidation",    5: "ILD",
    6: "Infiltration",       7: "Lung Opacity",     8: "Nodule/Mass",
    9: "Other lesion",      10: "Pleural effusion", 11: "Pleural thickening",
   12: "Pneumothorax",      13: "Pulmonary fibrosis", 14: "No finding",
}

GRID  = 64    # density map resolution
SIGMA = 1.5   # Gaussian smoothing (grid cells)

# Two-colour palette (A = warm red, B = cool blue)
CMAP_A = mcolors.LinearSegmentedColormap.from_list(
    'cls_a', [(1,1,1,0), (1.0,0.55,0.1,0.5), (0.85,0.1,0.05,1.0)])
CMAP_B = mcolors.LinearSegmentedColormap.from_list(
    'cls_b', [(1,1,1,0), (0.15,0.6,1.0,0.5), (0.0,0.2,0.9,1.0)])


# ── Core maths ────────────────────────────────────────────────────────────────

def build_density(bboxes: list, grid: int = GRID) -> np.ndarray:
    """Rasterise normalised bboxes into a smoothed (grid, grid) density map."""
    acc    = np.zeros((grid, grid), dtype=np.float64)
    edges  = np.linspace(0, 1, grid + 1)
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
    """Bhattacharyya coefficient ∈ [0,1]. Lower → more spatially distinct."""
    a_n = a / (a.sum() + 1e-12)
    b_n = b / (b.sum() + 1e-12)
    return float(np.sum(np.sqrt(a_n * b_n)))


def centroid(density: np.ndarray, grid: int = GRID):
    coords = (np.arange(grid) + 0.5) / grid
    total  = density.sum() + 1e-12
    cx = (density.sum(axis=0) @ coords) / total
    cy = (density.sum(axis=1) @ coords) / total
    return float(cx), float(cy)


# ── DICOM dimension cache ─────────────────────────────────────────────────────

def get_dicom_dims(dicom_dir: Path, image_ids: set) -> dict:
    """Read (H, W) from DICOM metadata; caches to dim_cache.json."""
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_bboxes(csv_path: str, dicom_dir: Path) -> dict:
    """
    Return {class_id: [(x0,y0,x1,y1), ...]} with normalised [0,1] coords
    for every class that has pixel-coord bbox rows in train.csv.
    """
    raw = {}   # image_id → [(cid, x0, y0, x1, y1)]
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            xmn = row.get('x_min', '').strip()
            if not xmn:
                continue
            try:
                cid = int(row['class_id'])
                bb  = (cid,
                       float(row['x_min']), float(row['y_min']),
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
            norm = (
                max(0.0, min(1.0, x0 / W)),
                max(0.0, min(1.0, y0 / H)),
                max(0.0, min(1.0, x1 / W)),
                max(0.0, min(1.0, y1 / H)),
            )
            by_class.setdefault(cid, []).append(norm)
    return by_class


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_pair(bboxes_a: list, bboxes_b: list,
              name_a: str, name_b: str, output: Path):

    density_a = build_density(bboxes_a)
    density_b = build_density(bboxes_b)
    bc        = bhattacharyya(density_a, density_b)
    cx_a, cy_a = centroid(density_a)
    cx_b, cy_b = centroid(density_b)
    dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

    print(f"\nSpatial Orthogonality: {name_a}  vs  {name_b}")
    print(f"  Bhattacharyya coefficient : {bc:.4f}  "
          f"({'low overlap ✓' if bc < 0.55 else 'moderate–high overlap'})")
    print(f"  {name_a:<28} centroid: ({cx_a:.3f}, {cy_a:.3f})")
    print(f"  {name_b:<28} centroid: ({cx_b:.3f}, {cy_b:.3f})")
    print(f"  Centroid distance         : {dist:.3f}  (fraction of image)")

    extent = [0, 1, 1, 0]
    xs = np.linspace(0, 1, GRID)
    levels = np.linspace(0.05, 1.0, 12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)

    for ax, density, cmap, cx, cy, name, n_ann, col_str in [
        (axes[0], density_a, CMAP_A, cx_a, cy_a, name_a, len(bboxes_a), 'Reds',   ),
        (axes[1], density_b, CMAP_B, cx_b, cy_b, name_b, len(bboxes_b), 'Blues',  ),
    ]:
        im = ax.imshow(density, extent=extent, origin='upper',
                       cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.contour(xs, xs, density, levels=levels, cmap=col_str,
                   linewidths=0.6, alpha=0.75)
        ax.scatter([cx], [cy], s=90, color='black', zorder=5,
                   marker='+', linewidths=2,
                   label=f'centroid ({cx:.2f}, {cy:.2f})')
        for v in [1/3, 2/3]:
            ax.axhline(v, color='grey', lw=0.5, ls='--', alpha=0.4)
            ax.axvline(v, color='grey', lw=0.5, ls='--', alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.set_xlabel("Normalised x  (left → right)", fontsize=9)
        ax.set_title(f"{name}\nn={n_ann:,} annotations",
                     fontsize=11, fontweight='bold', pad=8)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Relative density')

    # Overlay panel
    ax = axes[2]
    ax.imshow(density_a, extent=extent, origin='upper',
              cmap=CMAP_A, vmin=0, vmax=1, aspect='equal', alpha=0.85)
    ax.imshow(density_b, extent=extent, origin='upper',
              cmap=CMAP_B, vmin=0, vmax=1, aspect='equal', alpha=0.75)
    ax.contour(xs, xs, density_a, levels=[0.3,0.6,0.9],
               colors=['darkred'],  linewidths=1.0, alpha=0.8)
    ax.contour(xs, xs, density_b, levels=[0.3,0.6,0.9],
               colors=['darkblue'], linewidths=1.0, alpha=0.8)
    ax.scatter([cx_a], [cy_a], s=100, color='darkred',  zorder=6,
               marker='+', linewidths=2.5)
    ax.scatter([cx_b], [cy_b], s=100, color='darkblue', zorder=6,
               marker='+', linewidths=2.5)
    ax.annotate('', xy=(cx_b, cy_b), xytext=(cx_a, cy_a),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((cx_a+cx_b)/2 + 0.03, (cy_a+cy_b)/2,
            f'd={dist:.2f}', fontsize=8, color='black', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    for v in [1/3, 2/3]:
        ax.axhline(v, color='grey', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(v, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.set_xlabel("Normalised x  (left → right)", fontsize=9)
    ax.set_title("Overlay",
                 fontsize=11, fontweight='bold', pad=8)
    ax.legend(handles=[
        Line2D([0],[0], color='darkred',  lw=2, marker='+', ms=8, label=name_a),
        Line2D([0],[0], color='darkblue', lw=2, marker='+', ms=8, label=name_b),
    ], fontsize=8, loc='lower right', framealpha=0.8)

    fig.suptitle(
        f"Aggregate annotation density: {name_a}  vs  {name_b}",
        fontsize=12, y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")
    plt.close(fig)


def visualize(args):
    print(f"Loading bboxes from {args.csv_path} …")
    by_class = load_all_bboxes(args.csv_path, Path(args.dicom_dir))

    for cid in (args.class_a, args.class_b):
        n = len(by_class.get(cid, []))
        print(f"  {CLASS_NAMES.get(cid, f'class {cid}'):<28}: {n:,} annotations")
        if n == 0:
            raise RuntimeError(
                f"No bbox annotations found for class_id={cid} "
                f"({CLASS_NAMES.get(cid, '?')}). "
                f"Available classes: {sorted(by_class)}"
            )

    plot_pair(
        by_class[args.class_a], by_class[args.class_b],
        CLASS_NAMES.get(args.class_a, f"class {args.class_a}"),
        CLASS_NAMES.get(args.class_b, f"class {args.class_b}"),
        Path(args.output),
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--dicom_dir', default='/datasets/mmolefe/vinbigdata/train')
    p.add_argument('--csv_path',  default='/datasets/mmolefe/vinbigdata/train.csv')
    p.add_argument('--class_a',   type=int, default=3,
                   help='First class ID  (default: 3 = Cardiomegaly)')
    p.add_argument('--class_b',   type=int, default=11,
                   help='Second class ID (default: 11 = Pleural thickening)')
    p.add_argument('--output',    default='scripts/spatial_orthogonality.png')
    return p.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
