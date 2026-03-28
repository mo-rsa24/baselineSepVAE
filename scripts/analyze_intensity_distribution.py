"""
Intensity distribution EDA — VinBigData CXR dataset.

Demonstrates the heterogeneity problem and compares normalisation strategies.

Layout of main output figure:
  Each COLUMN = one sampled X-ray (N_SHOW images selected to span the diversity)
  Rows:
    0 — Raw image (uint16 / 65535, no post-processing)
    1 — Histogram of that image's pixel values
    2 — Per-image min-max normalisation
    3 — Per-image CLAHE (local contrast equalisation)
    4 — Histogram matching to a reference image (dataset median)
    5 — z-score normalisation (clip to [μ−3σ, μ+3σ])

Additional outputs:
  mean_std_scatter.png    — (global_mean, global_std) scatter coloured by label
  histogram_overlay.png   — 200 image histograms overlaid, coloured by label
  normalization_demo.png  — the main before/after grid

Usage:
  python scripts/analyze_intensity_distribution.py \\
      --csv_path  /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \\
      --dicom_dir /datasets/mmolefe/vinbigdata/cache_npy \\
      --output_dir results/intensity_eda \\
      --n_show 10 \\
      --n_stats 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path',    required=True)
    p.add_argument('--dicom_dir',   required=True)
    p.add_argument('--output_dir',  default='results/intensity_eda')
    p.add_argument('--img_size',    type=int, default=256)
    p.add_argument('--n_show',      type=int, default=10,
                   help='Number of images in the demo grid (columns)')
    p.add_argument('--n_stats',     type=int, default=1000,
                   help='Number of images for statistics plots')
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


# ── Image loading ─────────────────────────────────────────────────────────────

def load_npy(dicom_dir: Path, image_id: str, img_size: int) -> np.ndarray:
    """Return (H, W) float32 in [0, 1].  Raises FileNotFoundError on missing."""
    # Cache layout: <dicom_dir>/images/<image_id>.npy
    for sub in ['images', '']:
        p = dicom_dir / sub / f'{image_id}.npy' if sub else dicom_dir / f'{image_id}.npy'
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 65535.0
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape[0] != img_size or arr.shape[1] != img_size:
                from PIL import Image as PILImage
                pil = PILImage.fromarray((arr * 65535).astype(np.uint16))
                pil = pil.resize((img_size, img_size), PILImage.LANCZOS)
                arr = np.array(pil).astype(np.float32) / 65535.0
            return arr
    raise FileNotFoundError(f'{image_id}.npy not found under {dicom_dir}')


# ── Normalisation methods ─────────────────────────────────────────────────────

def norm_minmax(img: np.ndarray) -> np.ndarray:
    """Stretch [min, max] → [0, 1] per image."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def norm_percentile(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    """Clip to [p1, p99] then min-max.  More robust than pure min-max."""
    lo = np.percentile(img, lo_p)
    hi = np.percentile(img, hi_p)
    clipped = np.clip(img, lo, hi)
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return (clipped - lo) / (hi - lo)


def norm_zscore(img: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """z-score, clip to ±n_sigma, rescale to [0,1]."""
    mu, sigma = img.mean(), img.std()
    if sigma < 1e-6:
        return np.zeros_like(img)
    z = (img - mu) / sigma
    z = np.clip(z, -n_sigma, n_sigma)
    return (z + n_sigma) / (2 * n_sigma)


def norm_clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE — local contrast equalisation.  Input and output in [0,1]."""
    try:
        import cv2
        img_u8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        out = clahe.apply(img_u8).astype(np.float32) / 255.0
        return out
    except ImportError:
        # Fallback: global histogram equalization via numpy
        hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 1))
        cdf = hist.cumsum().astype(np.float32)
        cdf = (cdf - cdf.min()) / (cdf[-1] - cdf.min() + 1e-9)
        img_idx = (img * 255).astype(np.int32).clip(0, 255)
        return cdf[img_idx]


def norm_histmatch(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of img to reference.  Both [0,1]."""
    n_bins = 256
    src_hist, _ = np.histogram(img.ravel(),       bins=n_bins, range=(0, 1))
    ref_hist, _ = np.histogram(reference.ravel(), bins=n_bins, range=(0, 1))
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # For each src bin, find the ref bin with the closest CDF value
    lut = np.zeros(n_bins, dtype=np.float32)
    j = 0
    for i in range(n_bins):
        while j < n_bins - 1 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j / (n_bins - 1)

    img_idx = (img * (n_bins - 1)).astype(np.int32).clip(0, n_bins - 1)
    return lut[img_idx]


# ── Build reference image (median of a sample) ────────────────────────────────

def compute_reference_image(images: list) -> np.ndarray:
    """Pixel-wise median across a list of (H,W) arrays."""
    stack = np.stack(images, axis=0)
    return np.median(stack, axis=0).astype(np.float32)


# ── Outlier comparison grid ───────────────────────────────────────────────────

# Problem categories with their detection criterion (applied to per-image stats dict)
_PROBLEM_CATEGORIES = [
    {
        'name': 'Washed-out\n(over-bright)',
        'desc': 'global_mean > 0.55 — heavy scatter, obesity, or over-exposure',
        'key':  'global_mean',
        'rank': 'descending',
    },
    {
        'name': 'Very dark\n(under-exposed)',
        'desc': 'global_mean < 0.25 — under-exposed or dark equipment',
        'key':  'global_mean',
        'rank': 'ascending',
    },
    {
        'name': 'Hazy background\n(scanner scatter)',
        'desc': 'bg_mean > 0.15 — bright corners = no clean black border',
        'key':  'bg_mean',
        'rank': 'descending',
    },
    {
        'name': 'Flat / low contrast\n(small dynamic range)',
        'desc': 'dynamic_range < 0.35 — p99-p1 gap too small; poor tissue separation',
        'key':  'dynamic_range',
        'rank': 'ascending',
    },
    {
        'name': 'Clipped highlights\n(saturated)',
        'desc': 'p99 > 0.97 — bright regions hit scanner ceiling',
        'key':  'p99',
        'rank': 'descending',
    },
    {
        'name': 'Low sharpness\n(blurry)',
        'desc': 'sharpness < 0.02 — motion blur or soft-tissue-only view',
        'key':  'sharpness',
        'rank': 'ascending',
    },
]


def _compute_full_stats(img: np.ndarray) -> dict:
    """All stats needed for problem categorisation."""
    H, W = img.shape
    p = max(1, int(0.1 * min(H, W)))
    corners = np.concatenate([img[:p,:p].ravel(), img[:p,-p:].ravel(),
                               img[-p:,:p].ravel(), img[-p:,-p:].ravel()])
    ch, cw = H // 4, W // 4
    crop = img[ch:H-ch, cw:W-cw]
    gx, gy = np.diff(crop, axis=1), np.diff(crop, axis=0)
    grad = np.sqrt(gx[:gy.shape[0], :gx.shape[1]]**2 + gy[:gy.shape[0], :gx.shape[1]]**2)
    return {
        'global_mean':   float(img.mean()),
        'global_std':    float(img.std()),
        'bg_mean':       float(corners.mean()),
        'dynamic_range': float(np.percentile(img, 99) - np.percentile(img, 1)),
        'p1':            float(np.percentile(img, 1)),
        'p99':           float(np.percentile(img, 99)),
        'sharpness':     float(grad.mean()),
    }


def save_outlier_comparison_grid(
    pool_imgs: list,
    pool_ids: list,
    pool_labels: list,
    reference_img: np.ndarray,
    save_path: Path,
    n_per_category: int = 3,
):
    """
    One row per problematic image.  Layout:

      [Problem label] | Raw + stats | Histogram | p1-p99 norm | CLAHE | Hist-match

    Images are selected as the most extreme examples in each problem category.
    A thin coloured band on the left encodes the problem type.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Compute stats for every image in the pool
    print("  Computing stats for outlier detection …")
    all_stats = [_compute_full_stats(img) for img in pool_imgs]

    # For each category, rank images and pick the top n_per_category
    selected = []   # list of (img, image_id, label, stats, category_dict)
    seen_ids = set()

    for cat in _PROBLEM_CATEGORIES:
        key   = cat['key']
        vals  = np.array([s[key] for s in all_stats])
        order = np.argsort(vals)
        if cat['rank'] == 'descending':
            order = order[::-1]

        count = 0
        for idx in order:
            iid = pool_ids[idx]
            if iid not in seen_ids and count < n_per_category:
                selected.append((pool_imgs[idx], iid, pool_labels[idx],
                                 all_stats[idx], cat))
                seen_ids.add(iid)
                count += 1

    N_ROWS = len(selected)
    # Columns: Raw image | Histogram | p1-p99 | CLAHE | Hist-match
    N_IMG_COLS = 4   # three fix columns
    N_HIST_COL = 1
    N_COLS = 1 + N_HIST_COL + N_IMG_COLS   # raw + hist + 4 fixed

    col_titles = [
        'Raw image',
        'Histogram\n(raw)',
        'p1–p99\nnorm',
        'CLAHE\n(local contrast)',
        'Histogram\nmatching',
        'z-score\n±3σ',
    ]

    label_w  = 2.2
    cell_w   = 1.9
    hist_w   = 2.0
    cell_h   = 2.0
    fig_w    = label_w + cell_w + hist_w + N_IMG_COLS * cell_w
    fig_h    = N_ROWS * cell_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    # GridSpec: rows=N_ROWS, cols = label + raw + hist + 4 fix cols
    total_cols = 1 + 1 + 1 + N_IMG_COLS   # label, raw, hist, fixes
    width_ratios = ([label_w / cell_w] +   # label
                    [1.0] +                # raw image
                    [hist_w / cell_w] +    # histogram
                    [1.0] * N_IMG_COLS)    # fix methods

    gs = GridSpec(
        N_ROWS, total_cols,
        figure=fig,
        hspace=0.06,
        wspace=0.06,
        top=0.95, bottom=0.02,
        left=0.01, right=0.99,
        width_ratios=width_ratios,
    )

    # Category colours for left band
    cat_names = [c['name'] for c in _PROBLEM_CATEGORIES]
    palette   = plt.cm.Set2(np.linspace(0, 1, len(_PROBLEM_CATEGORIES)))
    cat_color = {c['name']: palette[i] for i, c in enumerate(_PROBLEM_CATEGORIES)}

    # Column headers on first row
    col_header_texts = ['Raw image', 'Histogram (raw)',
                        'p1–p99 clip', 'CLAHE', 'Hist match', 'z-score ±3σ']
    for ci, txt in enumerate(col_header_texts):
        ax = fig.add_subplot(gs[0, ci + 1])   # +1 to skip label col
        ax.set_title(txt, fontsize=8, pad=3, fontweight='bold')
        ax.axis('off')

    fix_fns = [
        ('p1–p99',    norm_percentile),
        ('CLAHE',     norm_clahe),
        ('Hist-match', lambda x: norm_histmatch(x, reference_img)),
        ('z-score',   norm_zscore),
    ]

    prev_cat_name = None
    for row, (img, iid, label, stats, cat) in enumerate(selected):
        cat_name = cat['name']
        color    = cat_color[cat_name]

        # ── Left label cell ──────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_facecolor((*color[:3], 0.25))
        # Print category name only when it changes
        display_name = cat_name if cat_name != prev_cat_name else ''
        ax_lbl.text(0.97, 0.75, display_name,
                    transform=ax_lbl.transAxes,
                    fontsize=7.5, fontweight='bold', va='top', ha='right',
                    color=(*color[:3], 1.0), linespacing=1.3)
        # Always print per-image stats
        stat_txt = (f'{label}\n'
                    f'μ={stats["global_mean"]:.3f}  σ={stats["global_std"]:.3f}\n'
                    f'bg={stats["bg_mean"]:.3f}  dr={stats["dynamic_range"]:.3f}\n'
                    f'p99={stats["p99"]:.3f}  sharp={stats["sharpness"]:.3f}')
        ax_lbl.text(0.97, 0.38, stat_txt,
                    transform=ax_lbl.transAxes,
                    fontsize=6, va='center', ha='right', color='#222222',
                    linespacing=1.4, family='monospace')
        ax_lbl.axis('off')
        # Left-edge colour bar
        ax_lbl.axvline(x=0.0, color=color, linewidth=6, solid_capstyle='butt')
        prev_cat_name = cat_name

        # ── Raw image ────────────────────────────────────────────────────────
        ax_raw = fig.add_subplot(gs[row, 1])
        ax_raw.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax_raw.axis('off')
        # Highlight the problematic metric value in red
        key_val = stats[cat['key']]
        ax_raw.set_xlabel(f'{cat["key"]}={key_val:.3f}', fontsize=6,
                           color='crimson', labelpad=1)
        ax_raw.xaxis.set_label_position('bottom')
        ax_raw.set_visible(True)

        # ── Histogram ────────────────────────────────────────────────────────
        ax_hist = fig.add_subplot(gs[row, 2])
        ax_hist.hist(img.ravel(), bins=100, range=(0, 1),
                     color=color, alpha=0.85, linewidth=0)
        # Vertical lines at p1, p99, mean
        ax_hist.axvline(stats['p1'],   color='blue',   linewidth=1.0, linestyle='--', label='p1')
        ax_hist.axvline(stats['p99'],  color='red',    linewidth=1.0, linestyle='--', label='p99')
        ax_hist.axvline(stats['global_mean'], color='black', linewidth=1.2, label='mean')
        ax_hist.set_xlim(0, 1)
        ax_hist.tick_params(labelsize=5.5)
        ax_hist.set_ylabel('count', fontsize=5.5)
        if row == 0:
            ax_hist.legend(fontsize=5, loc='upper right', framealpha=0.6)
        for spine in ax_hist.spines.values():
            spine.set_linewidth(0.4)

        # ── Fix columns ──────────────────────────────────────────────────────
        for fi, (fix_name, fix_fn) in enumerate(fix_fns):
            normed = np.clip(fix_fn(img.copy()), 0, 1)
            ax_fix = fig.add_subplot(gs[row, 3 + fi])
            ax_fix.imshow(normed, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax_fix.axis('off')
            # Show post-normalisation stats
            ax_fix.set_xlabel(f'μ={normed.mean():.3f} σ={normed.std():.3f}',
                               fontsize=5.5, color='#444444', labelpad=1)
            ax_fix.xaxis.set_label_position('bottom')

    # Category legend at bottom
    legend_handles = [
        mpatches.Patch(facecolor=(*cat_color[c['name']][:3], 0.7),
                       label=c['name'].replace('\n', ' ') + f'  ({c["desc"].split("—")[0].strip()})')
        for c in _PROBLEM_CATEGORIES
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=7, framealpha=0.8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Problematic images — most extreme examples per category\n'
        'Columns: Raw → its histogram (p1/mean/p99 marked) → 4 normalisation fixes',
        fontsize=10, y=0.975,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Outlier grid → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(args.dicom_dir)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ── Load CSV ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv_path)
    print(f"CSV: {len(df)} rows, columns: {list(df.columns)}")

    # Identify label column
    label_col = next((c for c in ['class_name', 'label', 'class_id'] if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    normal_mask  = df[label_col].astype(str).str.lower().str.contains('normal')
    cardio_mask  = df[label_col].astype(str).str.lower().str.contains('cardiomegaly')

    normal_ids  = df[normal_mask]['image_id'].unique()
    cardio_ids  = df[cardio_mask]['image_id'].unique()
    print(f"Normal: {len(normal_ids)},  Cardiomegaly: {len(cardio_ids)}")

    # ── Sample images spanning diversity for the demo grid ──────────────────
    # Strategy: load a random pool, then pick images that span the full range
    # of global_mean (covers dark → gray → bright → washed-out)
    pool_size = min(300, len(normal_ids) + len(cardio_ids))
    pool_ids  = rng.choice(
        np.concatenate([normal_ids, cardio_ids]),
        size=min(pool_size, len(normal_ids) + len(cardio_ids)),
        replace=False,
    )

    all_ids_with_labels = (
        [(i, 'Normal')       for i in normal_ids] +
        [(i, 'Cardiomegaly') for i in cardio_ids]
    )
    id_to_label = dict(all_ids_with_labels)

    print(f"Loading pool of {len(pool_ids)} images to find diverse examples …")
    pool_imgs, pool_means, pool_ids_ok, pool_labels = [], [], [], []
    for iid in pool_ids:
        try:
            img = load_npy(dicom_dir, iid, args.img_size)
            pool_imgs.append(img)
            pool_means.append(float(img.mean()))
            pool_ids_ok.append(iid)
            pool_labels.append(id_to_label.get(iid, 'Unknown'))
        except Exception:
            continue

    pool_means = np.array(pool_means)
    # Pick n_show images uniformly spaced across the mean-intensity range
    sorted_idx = np.argsort(pool_means)
    step = max(1, len(sorted_idx) // args.n_show)
    demo_indices = sorted_idx[::step][:args.n_show]
    demo_imgs = [pool_imgs[i] for i in demo_indices]
    demo_ids  = [pool_ids_ok[i] for i in demo_indices]
    demo_means = pool_means[demo_indices]

    print(f"Demo images: global_mean range [{demo_means.min():.3f}, {demo_means.max():.3f}]")

    # Reference image for histogram matching = median of full pool
    reference_img = compute_reference_image(pool_imgs)

    # ── Outlier comparison grid ──────────────────────────────────────────────
    print("\nBuilding outlier comparison grid …")
    save_outlier_comparison_grid(
        pool_imgs=pool_imgs,
        pool_ids=pool_ids_ok,
        pool_labels=pool_labels,
        reference_img=reference_img,
        save_path=out / 'outlier_comparison.png',
        n_per_category=3,
    )

    # ── Normalisation methods ────────────────────────────────────────────────
    methods = [
        ('Raw\n(÷ 65535, no post-proc)',   lambda x: x),
        ('Per-image\nmin–max',             norm_minmax),
        ('Per-image\np1–p99 clip',         norm_percentile),
        ('Per-image\nz-score ±3σ',         norm_zscore),
        ('CLAHE\n(local contrast)',         norm_clahe),
        ('Histogram matching\n(to dataset median)', lambda x: norm_histmatch(x, reference_img)),
    ]
    N_METHODS = len(methods)
    N_COLS    = args.n_show

    # ── Figure 1: normalisation demo grid ───────────────────────────────────
    # Layout: N_METHODS image rows + 1 histogram row per image
    # Group: for each method: one row of images + one row of histograms
    N_ROWS = N_METHODS * 2  # image row + histogram row per method

    cell_h = 1.6   # image cell height
    hist_h = 0.9   # histogram row height
    row_heights = []
    for _ in range(N_METHODS):
        row_heights.append(cell_h)
        row_heights.append(hist_h)

    label_col_w = 1.5
    img_col_w   = 1.8
    fig_w = label_col_w + N_COLS * img_col_w
    fig_h = sum(row_heights) + 0.6  # title

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=label_col_w / fig_w, right=0.99,
                        top=0.97, bottom=0.02,
                        hspace=0.05, wspace=0.03)

    gs = GridSpec(
        N_ROWS, N_COLS,
        figure=fig,
        hspace=0.05,
        wspace=0.03,
        top=0.96, bottom=0.02,
        left=label_col_w / fig_w, right=0.99,
    )

    hist_colors = plt.cm.tab10(np.linspace(0, 1, N_COLS))

    for m_idx, (method_name, norm_fn) in enumerate(methods):
        img_row  = m_idx * 2
        hist_row = m_idx * 2 + 1

        for c_idx, raw_img in enumerate(demo_imgs):
            normed = np.clip(norm_fn(raw_img.copy()), 0, 1)

            # Image cell
            ax_img = fig.add_subplot(gs[img_row, c_idx])
            ax_img.imshow(normed, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax_img.axis('off')

            # Column header (only on first method row)
            if m_idx == 0:
                mu = demo_means[c_idx]
                ax_img.set_title(
                    f'μ={mu:.3f}\nstd={raw_img.std():.3f}',
                    fontsize=6.5, pad=2, linespacing=1.2,
                )

            # Histogram cell
            ax_h = fig.add_subplot(gs[hist_row, c_idx])
            vals = normed.ravel()
            ax_h.hist(vals, bins=64, range=(0, 1),
                      color=hist_colors[c_idx], alpha=0.75, linewidth=0)
            ax_h.set_xlim(0, 1)
            ax_h.set_ylim(bottom=0)
            ax_h.tick_params(labelleft=False, labelbottom=(c_idx == 0),
                              left=False, bottom=True, labelsize=5)
            if c_idx == 0:
                ax_h.set_xlabel('pixel value', fontsize=5)
            # Small stats
            ax_h.text(0.97, 0.90, f'μ={vals.mean():.3f}\nσ={vals.std():.3f}',
                      transform=ax_h.transAxes,
                      fontsize=5, va='top', ha='right', color='#333333')
            for spine in ax_h.spines.values():
                spine.set_linewidth(0.4)

        # Row label (outside GridSpec — use fig.text)
        row_center_y = 1.0 - (img_row * (cell_h + hist_h) + cell_h / 2) / fig_h
        fig.text(
            label_col_w / fig_w - 0.01,
            row_center_y,
            method_name,
            fontsize=8, va='center', ha='right',
            fontweight='bold' if m_idx == 0 else 'normal',
        )

    fig.suptitle(
        'Intensity distribution EDA — each column is one X-ray sorted by global mean (dark→bright)\n'
        'Rows: raw image + its histogram, then the same image after each normalisation method',
        fontsize=9, y=0.995,
    )

    demo_path = out / 'normalization_demo.png'
    plt.savefig(str(demo_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Demo grid → {demo_path}")

    # ── Figure 2: global_mean vs global_std scatter, coloured by class ──────
    print(f"\nComputing stats on {args.n_stats} images for scatter plot …")
    n_per_class = args.n_stats // 2
    sample_normal = rng.choice(normal_ids, size=min(n_per_class, len(normal_ids)), replace=False)
    sample_cardio = rng.choice(cardio_ids, size=min(n_per_class, len(cardio_ids)), replace=False)

    records = []
    for iid, label in ([(i, 'Normal') for i in sample_normal] +
                       [(i, 'Cardiomegaly') for i in sample_cardio]):
        try:
            img = load_npy(dicom_dir, iid, args.img_size)
            H, W = img.shape
            p = max(1, int(0.1 * min(H, W)))
            corners = np.concatenate([img[:p,:p].ravel(), img[:p,-p:].ravel(),
                                       img[-p:,:p].ravel(), img[-p:,-p:].ravel()])
            records.append({
                'image_id':   iid,
                'label':      label,
                'global_mean': float(img.mean()),
                'global_std':  float(img.std()),
                'bg_mean':     float(corners.mean()),
                'p1':          float(np.percentile(img, 1)),
                'p99':         float(np.percentile(img, 99)),
                'dynamic_range': float(np.percentile(img, 99) - np.percentile(img, 1)),
            })
        except Exception:
            continue

    stats_df = pd.DataFrame(records)
    stats_df.to_csv(out / 'intensity_stats.csv', index=False)

    colors = {'Normal': '#4878d0', 'Cardiomegaly': '#ee854a'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Scatter: global_mean vs global_std
    ax = axes[0]
    for label, color in colors.items():
        sub = stats_df[stats_df['label'] == label]
        ax.scatter(sub['global_mean'], sub['global_std'],
                   c=color, alpha=0.25, s=8, label=label, rasterized=True)
    ax.set_xlabel('Global mean intensity', fontsize=9)
    ax.set_ylabel('Global std (contrast)', fontsize=9)
    ax.set_title('Mean vs Std — where images cluster\nWide spread = high heterogeneity', fontsize=9)
    ax.legend(fontsize=8, markerscale=3)

    # Scatter: bg_mean vs dynamic_range
    ax = axes[1]
    for label, color in colors.items():
        sub = stats_df[stats_df['label'] == label]
        ax.scatter(sub['bg_mean'], sub['dynamic_range'],
                   c=color, alpha=0.25, s=8, label=label, rasterized=True)
    ax.set_xlabel('Background brightness (corner mean)', fontsize=9)
    ax.set_ylabel('Dynamic range (p99 − p1)', fontsize=9)
    ax.set_title('Background vs Dynamic range\nHigh bg_mean = scanner scatter / obesity', fontsize=9)
    ax.legend(fontsize=8, markerscale=3)

    # Marginal distributions of global_mean
    ax = axes[2]
    from scipy import stats as scipy_stats
    for label, color in colors.items():
        vals = stats_df[stats_df['label'] == label]['global_mean'].values
        kde = scipy_stats.gaussian_kde(vals)
        xs = np.linspace(stats_df['global_mean'].min(), stats_df['global_mean'].max(), 300)
        ax.fill_between(xs, kde(xs), alpha=0.35, color=color)
        ax.plot(xs, kde(xs), color=color, linewidth=2, label=f'{label} (n={len(vals)})')
    ax.set_xlabel('Global mean intensity (raw, no post-proc)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('KDE of global mean by class\nOverlap → label-correlated brightness', fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    scatter_path = out / 'mean_std_scatter.png'
    plt.savefig(str(scatter_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Scatter plot → {scatter_path}")

    # ── Figure 3: histogram overlay — 200 images ────────────────────────────
    overlay_n = min(200, len(records))
    overlay_sample = stats_df.sample(n=overlay_n, random_state=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color in colors.items():
        sub_ids = overlay_sample[overlay_sample['label'] == label]['image_id'].values[:100]
        for iid in sub_ids:
            try:
                img = load_npy(dicom_dir, iid, args.img_size)
                axes[0].hist(img.ravel(), bins=128, range=(0, 1), histtype='step',
                             color=color, alpha=0.07, linewidth=0.5, density=True)
            except Exception:
                continue

    axes[0].set_xlabel('Raw pixel value (÷ 65535)', fontsize=9)
    axes[0].set_ylabel('Density', fontsize=9)
    axes[0].set_title('Raw histograms — 200 images overlaid\n'
                      'Blue=Normal, Orange=Cardiomegaly\n'
                      'Wide spread confirms heterogeneity', fontsize=8)
    # Add legend patches manually
    import matplotlib.patches as mpatches
    axes[0].legend(handles=[
        mpatches.Patch(color='#4878d0', alpha=0.5, label='Normal'),
        mpatches.Patch(color='#ee854a', alpha=0.5, label='Cardiomegaly'),
    ], fontsize=8)

    # Same after p1-p99 normalisation
    for label, color in colors.items():
        sub_ids = overlay_sample[overlay_sample['label'] == label]['image_id'].values[:100]
        for iid in sub_ids:
            try:
                img = load_npy(dicom_dir, iid, args.img_size)
                normed = norm_percentile(img)
                axes[1].hist(normed.ravel(), bins=128, range=(0, 1), histtype='step',
                             color=color, alpha=0.07, linewidth=0.5, density=True)
            except Exception:
                continue

    axes[1].set_xlabel('Pixel value after p1–p99 normalisation', fontsize=9)
    axes[1].set_ylabel('Density', fontsize=9)
    axes[1].set_title('After per-image p1–p99 clip + min-max\n'
                      'Histograms should align — residual spread = real signal',
                      fontsize=8)
    axes[1].legend(handles=[
        mpatches.Patch(color='#4878d0', alpha=0.5, label='Normal'),
        mpatches.Patch(color='#ee854a', alpha=0.5, label='Cardiomegaly'),
    ], fontsize=8)

    plt.tight_layout()
    overlay_path = out / 'histogram_overlay.png'
    plt.savefig(str(overlay_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Histogram overlay → {overlay_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    from scipy import stats as scipy_stats
    lines = [
        'Intensity Distribution Summary',
        '=' * 45,
        '',
        f'Images analysed: {len(stats_df)}',
        f'Normal:          {len(stats_df[stats_df.label=="Normal"])}',
        f'Cardiomegaly:    {len(stats_df[stats_df.label=="Cardiomegaly"])}',
        '',
        'Raw (no normalisation):',
        f'  global_mean: min={stats_df.global_mean.min():.3f}  '
        f'max={stats_df.global_mean.max():.3f}  '
        f'std={stats_df.global_mean.std():.3f}',
        f'  global_std:  min={stats_df.global_std.min():.3f}  '
        f'max={stats_df.global_std.max():.3f}',
        '',
        'Difference between classes (global_mean):',
    ]
    n_mu = stats_df[stats_df.label=='Normal']['global_mean'].mean()
    c_mu = stats_df[stats_df.label=='Cardiomegaly']['global_mean'].mean()
    stat, pval = scipy_stats.mannwhitneyu(
        stats_df[stats_df.label=='Normal']['global_mean'],
        stats_df[stats_df.label=='Cardiomegaly']['global_mean'],
        alternative='two-sided'
    )
    n1 = stats_df[stats_df.label=='Normal']['global_mean'].shape[0]
    n2 = stats_df[stats_df.label=='Cardiomegaly']['global_mean'].shape[0]
    r = 1 - (2 * stat) / (n1 * n2)
    lines += [
        f'  Normal mean μ:      {n_mu:.4f}',
        f'  Cardiomegaly mean μ:{c_mu:.4f}',
        f'  Mann-Whitney p:     {pval:.3e}',
        f'  Effect size r:      {r:.3f}',
        '',
        'Recommended normalisation for D4+ training:',
        '  Per-image p1–p99 clip + min-max (robust, fast, no library dependency).',
        '  Apply in VinBigData._load_npy() after the / 65535.0 step.',
        '  CLAHE is stronger but adds cv2 dependency and ~3ms per image.',
        '',
        'If |r| > 0.1 and p < 0.001 → add global intensity augmentation in training.',
    ]
    summary_path = out / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print('\n' + '\n'.join(lines))
    print(f"\nAll outputs in: {out}")


if __name__ == '__main__':
    main()
