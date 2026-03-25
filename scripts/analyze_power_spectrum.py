"""
Radially-Averaged Power Spectrum Analysis
==========================================
Loads pre-cached 256×256 .npy images (already preprocessed), computes the
2D FFT and averages power radially to get a 1-D frequency profile per class.

Informs Fix D (decoder channel schedule):
  - If meaningful energy exists up to ~64–128 cycles/image → 128 channels
    at 256×256 is justified (or still undershooting).
  - If power drops to noise floor below ~32 cycles/image → 64 channels was
    already sufficient.

Also compares Normal vs Cardiomegaly vs Pleural Thickening: if their spectra
differ substantially in the mid/high-freq bands, the reconstruction loss
needs to weight those bands differently.

Outputs: scripts/power_spectrum.png

Usage:
    python scripts/analyze_power_spectrum.py
    python scripts/analyze_power_spectrum.py --npy_dir /path/to/cache_npy/images \\
        --csv /path/to/train.csv --n_samples 300 --output scripts/power_spectrum.png
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

CLASS_NAMES = {
    3:  'Cardiomegaly',
    11: 'Pleural Thickening',
    14: 'No Finding (Normal)',
}
CLASS_COLORS = {
    3:  '#e84545',   # red
    11: '#3a86ff',   # blue
    14: '#555555',   # dark grey
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npy_dir',   default='/datasets/mmolefe/vinbigdata/cache_npy/images',
                   help='Directory containing {image_id}.npy files (uint16, 256×256)')
    p.add_argument('--csv',       default='/datasets/mmolefe/vinbigdata/train.csv',
                   help='train.csv for class labels')
    p.add_argument('--n_samples', type=int, default=300,
                   help='Max images to sample per class')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--output',    default='scripts/power_spectrum.png')
    return p.parse_args()


def radial_average(power2d):
    """Compute radially-averaged 1-D power spectrum from 2-D FFT magnitude²."""
    H, W = power2d.shape
    cy, cx = H // 2, W // 2

    y_idx, x_idx = np.indices(power2d.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    profile = np.zeros(max_r + 1)
    counts  = np.zeros(max_r + 1)

    for ri in range(max_r + 1):
        mask = r == ri
        if mask.any():
            profile[ri] = power2d[mask].mean()
            counts[ri]  = mask.sum()

    return profile[:max_r + 1]


def get_class_image_ids(csv_path, focus_classes):
    """
    Returns {class_id: set(image_ids)}.
    For No Finding (14), image_id has class_id=14 AND no other annotation row.
    """
    all_rows = defaultdict(set)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            all_rows[row['image_id']].add(int(row['class_id']))

    result = defaultdict(set)
    for iid, cids in all_rows.items():
        for cid in focus_classes:
            if cid == 14:
                # Truly normal: only class 14, no disease annotations
                if cids == {14}:
                    result[14].add(iid)
            else:
                if cid in cids:
                    result[cid].add(iid)
    return result


def load_npy_image(npy_dir: Path, image_id: str):
    """Load a uint16 .npy, normalise to [0, 1] float32."""
    path = npy_dir / f'{image_id}.npy'
    if not path.exists():
        return None
    arr = np.load(str(path))
    arr = arr.astype(np.float32)
    # Normalise to [0, 1] regardless of original range
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def compute_mean_spectrum(image_ids, npy_dir, n_samples, rng):
    """Sample up to n_samples images and return the mean radial power spectrum."""
    ids = list(image_ids)
    rng.shuffle(ids)
    ids = ids[:n_samples]

    spectra = []
    for iid in ids:
        arr = load_npy_image(npy_dir, iid)
        if arr is None:
            continue
        # Ensure 2-D
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim != 2:
            continue

        # 2-D FFT → shift DC to centre → power spectrum
        f2d    = np.fft.fft2(arr)
        f2d_sh = np.fft.fftshift(f2d)
        power  = np.abs(f2d_sh) ** 2
        profile = radial_average(power)
        if profile.sum() > 0:
            spectra.append(profile)

    if not spectra:
        return None, 0

    # Pad to equal length (all same size, so should be identical length)
    min_len = min(len(s) for s in spectra)
    stack   = np.stack([s[:min_len] for s in spectra], axis=0)
    return stack.mean(axis=0), len(spectra)


def visualise(spectra_dict, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Radially-Averaged Power Spectrum — Normal vs Cardiomegaly vs Pleural Thickening\n'
        'Informs decoder channel schedule (Fix D): where does meaningful frequency energy end?',
        fontsize=12, fontweight='bold',
    )

    # ── Panel A: Log-scale power vs spatial frequency ────────────────────────
    ax = axes[0]
    ax.set_title('(A) Mean power spectrum (log scale)', fontsize=11)

    max_len = 0
    for cid, (profile, n) in spectra_dict.items():
        if profile is not None:
            freqs = np.arange(len(profile))
            # Avoid log(0)
            log_power = np.log10(profile + 1e-10)
            ax.plot(freqs, log_power, color=CLASS_COLORS[cid],
                    label=f'{CLASS_NAMES[cid]} (n={n})', linewidth=2, alpha=0.85)
            max_len = max(max_len, len(profile))

    # Mark key spatial frequencies (cycles per image at 256px)
    for freq, label in [(16, '16 cyc\n(16px)'), (32, '32 cyc\n(8px)'),
                        (64, '64 cyc\n(4px)'), (96, '96 cyc\n(~3px)')]:
        if freq < max_len:
            ax.axvline(freq, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.text(freq + 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > -99 else 0,
                    label, fontsize=7, color='grey', va='bottom')

    ax.set_xlabel('Spatial frequency (cycles / image width)')
    ax.set_ylabel('log₁₀ mean power')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_len)

    # ── Panel B: Normalised (each curve normalised to its DC component) ───────
    ax = axes[1]
    ax.set_title('(B) Normalised power (DC=1) — relative high-freq content per class',
                 fontsize=11)

    for cid, (profile, n) in spectra_dict.items():
        if profile is None or profile[0] == 0:
            continue
        freqs = np.arange(len(profile))
        norm_profile = profile / profile[0]   # normalise to DC
        ax.plot(freqs, norm_profile, color=CLASS_COLORS[cid],
                label=f'{CLASS_NAMES[cid]} (n={n})', linewidth=2, alpha=0.85)

    # Shade the region that 128-ch decoder can represent (rough guide)
    # 256×256 → 128×128 after one SmoothUp level; useful freq range ≈ 0–64 cycles
    ax.axvspan(0, 64,  alpha=0.04, color='green', label='≤64 cyc (128ch decoder range)')
    ax.axvspan(64, 128, alpha=0.04, color='orange', label='64–128 cyc (beyond 128ch)')

    ax.set_xlabel('Spatial frequency (cycles / image width)')
    ax.set_ylabel('Normalised mean power (DC component = 1.0)')
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_len)

    # Mark key frequencies
    for freq, label in [(16, '16'), (32, '32'), (64, '64'), (96, '96')]:
        if freq < max_len:
            ax.axvline(freq, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    args   = parse_args()
    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    npy_dir  = Path(args.npy_dir)
    csv_path = Path(args.csv)

    focus = {3, 11, 14}
    print('Loading class → image_id mapping...')
    class_ids = get_class_image_ids(csv_path, focus)
    for cid in sorted(focus):
        print(f'  Class {cid} ({CLASS_NAMES[cid]}): {len(class_ids[cid]):,} images')

    spectra = {}
    for cid in sorted(focus):
        print(f'\nComputing spectrum for class {cid} ({CLASS_NAMES[cid]})...')
        ids   = list(class_ids[cid])
        rng.shuffle(ids)
        profile, n = compute_mean_spectrum(set(ids), npy_dir, args.n_samples, rng)
        spectra[cid] = (profile, n)
        if profile is not None:
            # Report where power drops to 1% of DC
            dc = profile[0]
            if dc > 0:
                drop_idx = np.where(profile / dc < 0.01)[0]
                drop_at  = drop_idx[0] if len(drop_idx) > 0 else len(profile)
                print(f'  Power < 1% of DC at: {drop_at} cycles/image  '
                      f'(using n={n} images)')
        else:
            print(f'  No npy files found for class {cid}')

    print('\nRendering figure...')
    visualise(spectra, args.output)


if __name__ == '__main__':
    main()
