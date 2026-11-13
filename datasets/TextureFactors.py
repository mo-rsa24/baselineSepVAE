"""
TextureFactors — Domain C
==========================
Tests whether the d0–d5 framework succeeds when the spatial locality
assumption (A2) is ABSENT. Both factors are global Fourier textures
that pervade the entire image — no bounding box is meaningful.

Assumption violations:
    A2 — Spatial locality is VIOLATED: factors are globally distributed
         sinusoidal patterns with no spatial concentration.
         → Bbox annotations are all-zero sentinels; D1 spatial gate disabled.

All other assumptions remain satisfied:
    A1 — Statistical independence: textures are generated from independent
         rng streams; no spurious co-occurrence in training.
    A3 — Null class: Label 0 = background only.
    A4 — Additive composition: pixel-level texture superposition is linear.

Factor design:
    Factor 1: Horizontal sinusoidal stripes  k_y ∈ {3, 4, 5} cycles/image
    Factor 2: Vertical   sinusoidal stripes  k_x ∈ {3, 4, 5} cycles/image
    Background: Gaussian noise field (σ = 0.15 in normalised space)

Image formula (in [0, 1] before normalisation to [-1, 1]):
    img = bg + α·[sin(2π·k_y·y/H)·0.5 + 0.5] + β·[sin(2π·k_x·x/W)·0.5 + 0.5]
    where α, β ∈ {0, 1} according to the label.

Key experimental question:
    Does the MI discriminator alone (without spatial prior) achieve
    G1 ✓ G2 ✓ G3 ✓ for independent global factors?
    Expected answer: YES — demonstrating that A2 is sufficient but not
    necessary; statistical independence (A1) is the load-bearing assumption.

Label scheme (identical to SynDis / Domain A):
    0 — Normal             (background only)
    1 — Factor-1           (horizontal stripes)
    2 — Factor-2           (vertical stripes)
    3 — Both               (H + V stripes; withheld from training)

Auxiliary:
    fourier_composition_score(img, ...) — checks Fourier power at target
    frequencies; used to evaluate composition quality for this domain.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FREQ_F1_RANGE = (3, 5)   # k_y choices for horizontal stripes
FREQ_F2_RANGE = (3, 5)   # k_x choices for vertical stripes

# Sentinel bboxes (all-zero) — signals "no valid spatial annotation" to the
# training harness, which zeros out the spatial gate loss for this domain.
BBOX_F1 = (0.0, 0.0, 0.0, 0.0)
BBOX_F2 = (0.0, 0.0, 0.0, 0.0)

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1   # horizontal stripes
LABEL_FACTOR2 = 2   # vertical stripes
LABEL_BOTH    = 3


# ─────────────────────────────────────────────────────────────────────────────
# Low-level image generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_background(rng: np.random.RandomState, H: int = 64, W: int = 64) -> np.ndarray:
    """Gaussian noise background, values in [0, 1]."""
    bg = rng.normal(0.5, 0.15, (H, W)).astype(np.float32)
    return np.clip(bg, 0, 1)


def _make_stripe_h(rng: np.random.RandomState, H: int = 64, W: int = 64) -> np.ndarray:
    """
    Factor-1 signal: horizontal Fourier stripes.
    Frequency jittered within FREQ_F1_RANGE; amplitude 0.40–0.55.
    Returns additive signal in [0, ~0.55].
    """
    k   = rng.randint(FREQ_F1_RANGE[0], FREQ_F1_RANGE[1] + 1)
    amp = rng.uniform(0.40, 0.55)
    y   = np.arange(H, dtype=np.float32)
    # Sine wave mapped to [0, amp] so it always adds positive signal
    wave = amp * (np.sin(2 * np.pi * k * y / H) * 0.5 + 0.5)  # (H,)
    return np.tile(wave[:, None], (1, W)).astype(np.float32)   # (H, W)


def _make_stripe_v(rng: np.random.RandomState, H: int = 64, W: int = 64) -> np.ndarray:
    """
    Factor-2 signal: vertical Fourier stripes.
    Frequency jittered within FREQ_F2_RANGE; amplitude 0.40–0.55.
    """
    k   = rng.randint(FREQ_F2_RANGE[0], FREQ_F2_RANGE[1] + 1)
    amp = rng.uniform(0.40, 0.55)
    x   = np.arange(W, dtype=np.float32)
    wave = amp * (np.sin(2 * np.pi * k * x / W) * 0.5 + 0.5)  # (W,)
    return np.tile(wave[None, :], (H, 1)).astype(np.float32)   # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Composition quality metric (Fourier power spectrum)
# ─────────────────────────────────────────────────────────────────────────────

def fourier_composition_score(
    img_np: np.ndarray,
    k_y_range: tuple = FREQ_F1_RANGE,
    k_x_range: tuple = FREQ_F2_RANGE,
) -> tuple:
    """
    Evaluate whether a composed image contains both texture factors by
    checking elevated Fourier power at the target frequency bands.

    Parameters
    ----------
    img_np : np.ndarray
        (H, W) image in [0, 1].
    k_y_range : (lo, hi)
        Expected horizontal-stripe frequency range (cycles/image).
    k_x_range : (lo, hi)
        Expected vertical-stripe frequency range.

    Returns
    -------
    (score_h, score_v) : (float, float)
        Mean Fourier power in each factor's frequency band.
        Higher → more signal present. Compare against a null-decoded
        (z_d=0) baseline to get a delta score.
    """
    H, W = img_np.shape
    fft       = np.abs(np.fft.fft2(img_np - img_np.mean()))
    fft_shift = np.fft.fftshift(fft)
    cy, cx    = H // 2, W // 2

    k_y_lo, k_y_hi = k_y_range
    k_x_lo, k_x_hi = k_x_range
    col_band = slice(cx - 2, cx + 3)  # near-zero x-frequency (for H stripes)
    row_band = slice(cy - 2, cy + 3)  # near-zero y-frequency (for V stripes)

    # Horizontal stripe band: rows at ±k_y from centre, columns near 0
    h_pos = fft_shift[cy - k_y_hi: cy - k_y_lo + 1, col_band]
    h_neg = fft_shift[cy + k_y_lo: cy + k_y_hi + 1, col_band]
    score_h = float(np.mean(h_pos) + np.mean(h_neg))

    # Vertical stripe band: columns at ±k_x from centre, rows near 0
    v_pos = fft_shift[row_band, cx - k_x_hi: cx - k_x_lo + 1]
    v_neg = fft_shift[row_band, cx + k_x_lo: cx + k_x_hi + 1]
    score_v = float(np.mean(v_pos) + np.mean(v_neg))

    return score_h, score_v


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextureFactorsDataset(Dataset):
    """
    Domain C: Statistically-independent but NOT spatially localized factors.

    Tests whether MI disentanglement (D2) succeeds when the spatial prior
    (D1 bbox cross-attention) provides no useful signal.

    Returns (from __getitem__ in 'flat' mode):
        x : float32 Tensor (1, H, W) in [-1, 1]
        y : long Tensor scalar — class label in {0, 1, 2, 3}
    """

    def __init__(
        self,
        n_normal:    int  = 1500,
        n_factor1:   int  = 750,
        n_factor2:   int  = 750,
        n_both:      int  = 0,
        img_size:    int  = 64,
        global_seed: int  = 42,
        mode:        str  = 'flat',
    ):
        assert mode in ('flat', 'triplet')
        self.H = self.W  = img_size
        self.mode        = mode
        self._seeds:  list = []
        self._labels: list = []

        base = global_seed * 1_000_000
        for label, count in [
            (LABEL_NORMAL,  n_normal),
            (LABEL_FACTOR1, n_factor1),
            (LABEL_FACTOR2, n_factor2),
            (LABEL_BOTH,    n_both),
        ]:
            for i in range(count):
                self._seeds.append(base + i)
                self._labels.append(label)
            base += 100_000

        self.labels = np.array(self._labels, dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, idx: int):
        img = self._generate(self._seeds[idx], self._labels[idx])
        x   = torch.from_numpy(img[None]).float() * 2.0 - 1.0
        y   = torch.tensor(self._labels[idx], dtype=torch.long)
        if self.mode == 'flat':
            return x, y
        return {
            'x':            x,
            'y':            y,
            # All-zero bboxes signal "no spatial annotation" to the harness
            'bbox_factor1': torch.zeros(4, dtype=torch.float32),
            'bbox_factor2': torch.zeros(4, dtype=torch.float32),
        }

    def _generate(self, seed: int, label: int,
                  inject_f1: bool = False, inject_f2: bool = False) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)
        add_f1 = (label in (LABEL_FACTOR1, LABEL_BOTH)) or inject_f1
        add_f2 = (label in (LABEL_FACTOR2, LABEL_BOTH)) or inject_f2
        if add_f1:
            img = np.clip(img + _make_stripe_h(rng, self.H, self.W), 0, 1)
        if add_f2:
            img = np.clip(img + _make_stripe_v(rng, self.H, self.W), 0, 1)
        return img

    # ──────────────────────────────────────────────────────────────────────
    def has_factor1(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR1) |
                (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_factor2(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR2) |
                (self.labels == LABEL_BOTH)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quick preview
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = TextureFactorsDataset(n_normal=8, n_factor1=4, n_factor2=4, n_both=4,
                               global_seed=42)
    class_names = ["Normal", "Factor-1\n(H-stripes)", "Factor-2\n(V-stripes)", "Both"]

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for row, label in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds.labels) if l == label][:5]
        for col, idx in enumerate(idxs):
            x, y = ds[idx]
            img  = (x.squeeze().numpy() + 1) / 2
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(class_names[label], fontsize=9)

    plt.suptitle("TextureFactors (Domain C) — one row per class", fontsize=12)
    plt.tight_layout()
    plt.savefig("texture_factors_preview.png", dpi=120, bbox_inches="tight")
    print("Saved → texture_factors_preview.png")
