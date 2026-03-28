"""
Background confound analysis — VinBigData CXR dataset.

Measures whether nuisance image statistics (background brightness, global
contrast, sharpness) correlate with the Cardiomegaly label.  If they do,
the model may learn to use these shortcuts instead of cardiac geometry.

Metrics computed per image:
  bg_mean      — mean intensity of four corner patches (scanner background)
  bg_std       — std of background patches
  global_mean  — mean intensity over the full image
  global_std   — std over the full image (contrast proxy)
  sharpness    — mean gradient magnitude (Sobel) in central crop
  bbox_area    — normalised bbox area for Cardiomegaly (0 for Normal)

Outputs (all in --output_dir):
  background_stats.csv          — per-image stats
  background_distributions.png  — violin/box plots per class
  correlation_heatmap.png       — Pearson r between all metrics
  bg_mean_vs_label.png          — scatter + KDE: bg_mean by class
  sharpness_vs_label.png        — scatter + KDE: sharpness by class
  summary.txt                   — Mann-Whitney U p-values + effect sizes

Usage:
  python scripts/analyze_background_confound.py \\
      --csv_path  /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \\
      --dicom_dir /datasets/mmolefe/vinbigdata/cache_npy \\
      --output_dir results/background_confound \\
      --img_size 256 \\
      --max_per_class 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path',      required=True)
    p.add_argument('--dicom_dir',     required=True)
    p.add_argument('--output_dir',    default='results/background_confound')
    p.add_argument('--img_size',      type=int, default=256)
    p.add_argument('--max_per_class', type=int, default=2000,
                   help='Max images per class to analyse (for speed)')
    p.add_argument('--corner_frac',   type=float, default=0.1,
                   help='Fraction of image width/height to use as corner patch')
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


# ── Image statistics ──────────────────────────────────────────────────────────

def _load_npy(dicom_dir: str, image_id: str, img_size: int) -> np.ndarray:
    """Load cached NPY for one image.  Returns (H,W) float32 in [0,1]."""
    path = Path(dicom_dir) / f'{image_id}.npy'
    arr = np.load(str(path)).astype(np.float32)
    # Normalise from uint16 storage range
    if arr.max() > 1.5:
        arr = arr / 65535.0
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    # Resize with simple area interpolation if needed
    if arr.shape[0] != img_size or arr.shape[1] != img_size:
        from PIL import Image as PILImage
        pil = PILImage.fromarray((arr * 65535).astype(np.uint16))
        pil = pil.resize((img_size, img_size), PILImage.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 65535.0
    return arr


def compute_image_stats(img: np.ndarray, corner_frac: float = 0.1) -> dict:
    """
    img: (H, W) float32 in [0, 1].
    Returns dict of scalar statistics.
    """
    H, W = img.shape
    p = max(1, int(corner_frac * min(H, W)))

    corners = np.concatenate([
        img[:p, :p].ravel(),
        img[:p, -p:].ravel(),
        img[-p:, :p].ravel(),
        img[-p:, -p:].ravel(),
    ])

    # Sharpness: Sobel gradient magnitude in central 50% crop
    ch, cw = H // 4, W // 4
    crop = img[ch:H - ch, cw:W - cw]
    gx = np.diff(crop, axis=1)
    gy = np.diff(crop, axis=0)
    # make same shape for mean
    grad_mag = np.sqrt(gx[:gy.shape[0], :gx.shape[1]]**2 +
                       gy[:gy.shape[0], :gx.shape[1]]**2)

    return {
        'bg_mean':     float(corners.mean()),
        'bg_std':      float(corners.std()),
        'global_mean': float(img.mean()),
        'global_std':  float(img.std()),
        'sharpness':   float(grad_mag.mean()),
        'p10':         float(np.percentile(img, 10)),
        'p90':         float(np.percentile(img, 90)),
        'dynamic_range': float(np.percentile(img, 90) - np.percentile(img, 10)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV: {len(df)} rows.  Columns: {list(df.columns)}")

    # Identify label column
    if 'class_name' in df.columns:
        label_col = 'class_name'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError(f"Cannot find label column in {list(df.columns)}")

    normal_ids  = df[df[label_col].str.lower().str.contains('normal')]['image_id'].unique()
    cardio_ids  = df[df[label_col].str.lower().str.contains('cardiomegaly')]['image_id'].unique()

    print(f"Normal images: {len(normal_ids)},  Cardiomegaly: {len(cardio_ids)}")

    # Subsample if large
    if len(normal_ids) > args.max_per_class:
        normal_ids = rng.choice(normal_ids, size=args.max_per_class, replace=False)
    if len(cardio_ids) > args.max_per_class:
        cardio_ids = rng.choice(cardio_ids, size=args.max_per_class, replace=False)

    # Compute bbox area for Cardiomegaly (from CSV if available)
    bbox_area = {}
    if all(c in df.columns for c in ['x_min', 'y_min', 'x_max', 'y_max']):
        cardio_rows = df[df[label_col].str.lower().str.contains('cardiomegaly')]
        for _, row in cardio_rows.iterrows():
            w = row['x_max'] - row['x_min']
            h = row['y_max'] - row['y_min']
            if w > 0 and h > 0:
                bbox_area[row['image_id']] = float(w * h)

    records = []
    total = len(normal_ids) + len(cardio_ids)
    for idx, (iid, label) in enumerate(
        [(i, 'Normal') for i in normal_ids] + [(i, 'Cardiomegaly') for i in cardio_ids]
    ):
        if idx % 200 == 0:
            print(f"  {idx}/{total} …", flush=True)
        try:
            img = _load_npy(args.dicom_dir, iid, args.img_size)
        except Exception as e:
            print(f"  skip {iid}: {e}")
            continue

        stats = compute_image_stats(img, args.corner_frac)
        stats['image_id'] = iid
        stats['label']    = label
        stats['bbox_area'] = bbox_area.get(iid, 0.0)
        records.append(stats)

    stats_df = pd.DataFrame(records)
    stats_csv = out / 'background_stats.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nStats saved → {stats_csv}  ({len(stats_df)} images)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    metric_cols = ['bg_mean', 'bg_std', 'global_mean', 'global_std',
                   'sharpness', 'dynamic_range']
    metric_labels = {
        'bg_mean':       'Background brightness\n(corner mean)',
        'bg_std':        'Background variability\n(corner std)',
        'global_mean':   'Global mean intensity',
        'global_std':    'Global std (contrast)',
        'sharpness':     'Sharpness\n(Sobel gradient mean)',
        'dynamic_range': 'Dynamic range\n(p90 − p10)',
    }

    normal_df  = stats_df[stats_df['label'] == 'Normal']
    cardio_df  = stats_df[stats_df['label'] == 'Cardiomegaly']

    # ── 1. Violin plots ──────────────────────────────────────────────────────
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.8, 4.5))
    colors = {'Normal': '#4878d0', 'Cardiomegaly': '#ee854a'}

    for ax, col in zip(axes, metric_cols):
        data  = [normal_df[col].dropna().values, cardio_df[col].dropna().values]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True)
        for i, (pc, label) in enumerate(zip(parts['bodies'], ['Normal', 'Cardiomegaly'])):
            pc.set_facecolor(colors[label])
            pc.set_alpha(0.7)

        # Mann-Whitney U test
        stat, pval = scipy_stats.mannwhitneyu(
            normal_df[col].dropna(), cardio_df[col].dropna(), alternative='two-sided'
        )
        # Effect size: rank-biserial correlation
        n1, n2 = normal_df[col].dropna().shape[0], cardio_df[col].dropna().shape[0]
        r = 1 - (2 * stat) / (n1 * n2)

        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        ax.set_title(f'{metric_labels[col]}\np={pval:.2e} {sig}\nr={r:.3f}', fontsize=8, linespacing=1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Cardio'], fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    fig.suptitle('Image statistics: Normal vs Cardiomegaly\n'
                 'Significant differences → potential confounds for SepVAE',
                 fontsize=10)
    plt.tight_layout()
    violin_path = out / 'background_distributions.png'
    plt.savefig(str(violin_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Distribution plot → {violin_path}")

    # ── 2. Correlation heatmap ───────────────────────────────────────────────
    all_metrics = metric_cols + ['bbox_area']
    corr_df = stats_df[all_metrics].dropna()
    corr = corr_df.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(all_metrics)))
    ax.set_yticks(range(len(all_metrics)))
    ax.set_xticklabels(all_metrics, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(all_metrics, fontsize=8)
    for i in range(len(all_metrics)):
        for j in range(len(all_metrics)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(corr.values[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('Correlation between image statistics\n(incl. bbox_area for Cardiomegaly)', fontsize=9)
    plt.tight_layout()
    corr_path = out / 'correlation_heatmap.png'
    plt.savefig(str(corr_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Correlation heatmap → {corr_path}")

    # ── 3. Scatter: bg_mean vs label, coloured ───────────────────────────────
    for metric in ['bg_mean', 'sharpness', 'global_std']:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        # KDE plot
        ax = axes[0]
        for label, color in colors.items():
            vals = stats_df[stats_df['label'] == label][metric].dropna().values
            kernel = scipy_stats.gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.fill_between(xs, kernel(xs), alpha=0.4, color=color, label=label)
            ax.plot(xs, kernel(xs), color=color, linewidth=1.5)
        ax.set_xlabel(metric_labels.get(metric, metric), fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(f'{metric} — density by class', fontsize=9)

        # Box plot
        ax = axes[1]
        data = [normal_df[metric].dropna().values, cardio_df[metric].dropna().values]
        bp = ax.boxplot(data, labels=['Normal', 'Cardiomegaly'], patch_artist=True,
                        notch=True, medianprops=dict(color='black', linewidth=2))
        for patch, label in zip(bp['boxes'], ['Normal', 'Cardiomegaly']):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.7)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
        ax.set_title(f'{metric} — boxplot', fontsize=9)
        ax.tick_params(labelsize=8)

        plt.tight_layout()
        m_path = out / f'{metric}_by_label.png'
        plt.savefig(str(m_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"{metric} plot → {m_path}")

    # ── 4. Summary text ──────────────────────────────────────────────────────
    lines = [
        'Background Confound Analysis — Summary',
        '=' * 50,
        f'Normal images analysed:       {len(normal_df)}',
        f'Cardiomegaly images analysed: {len(cardio_df)}',
        '',
        'Mann-Whitney U test results (two-sided):',
        f'  {"Metric":<20}  {"p-value":>12}  {"effect r":>10}  {"significance":>14}  {"normal μ":>10}  {"cardio μ":>10}',
        '-' * 90,
    ]
    for col in metric_cols:
        n_vals = normal_df[col].dropna().values
        c_vals = cardio_df[col].dropna().values
        stat, pval = scipy_stats.mannwhitneyu(n_vals, c_vals, alternative='two-sided')
        r = 1 - (2 * stat) / (len(n_vals) * len(c_vals))
        sig = '*** SIGNIFICANT' if pval < 0.001 else ('** SIGNIFICANT' if pval < 0.01
              else ('* MARGINAL' if pval < 0.05 else 'ns (no effect)'))
        lines.append(
            f'  {col:<20}  {pval:>12.3e}  {r:>10.3f}  {sig:>14}  '
            f'{n_vals.mean():>10.4f}  {c_vals.mean():>10.4f}'
        )

    lines += [
        '',
        'Interpretation:',
        '  r > 0.1 with p < 0.01 → meaningful confound; consider augmentation or stratification.',
        '  bg_mean differs → cardiomegaly patients have different scanner background.',
        '  sharpness differs → equipment/protocol confound; model may use texture as shortcut.',
        '',
        'Recommended mitigations:',
        '  1. Add global intensity augmentation in the dataloader (shift/scale ±15%).',
        '  2. Check if bg_mean correlates with bbox_area in the Cardiomegaly subset.',
        '  3. If sharpness differs strongly: add blur augmentation.',
        '  4. Consider CLAHE preprocessing to normalise local contrast.',
    ]

    summary_path = out / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSummary → {summary_path}")
    print('\n'.join(lines[6:]))  # print to stdout too

    print(f"\nAll outputs in: {out}")


if __name__ == '__main__':
    main()
