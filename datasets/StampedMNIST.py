"""
StampedComposites (aka StampedMNIST) — Domain A
================================================
Positive control for the d0–d5 generalizability suite.

Two geometric stamps placed at fixed, non-overlapping spatial quadrants on a
smooth Gaussian background. ALL four SepVAE framework assumptions are satisfied:

    A1 — Statistical independence : stamps are generated from independent rng
                                     streams; no spurious co-occurrence in train
    A2 — Spatial locality         : each factor confined to a known quadrant
                                     → exact bbox annotations available
    A3 — Null class               : Label 0 = no stamps (Normal analog)
    A4 — Additive composition     : pixel-level stamp addition is linear

Factor design:
    Factor 1 (stamp A): filled circle in top-left quadrant   (rows 2–30, cols 2–30)
    Factor 2 (stamp B): cross/plus in bottom-right quadrant  (rows 34–62, cols 34–62)

Label scheme (matches SynDis and SyntheticCXR):
    0 — Normal   (background only)
    1 — Factor-1 (stamp A only)
    2 — Factor-2 (stamp B only)
    3 — Both     (stamp A + stamp B; withheld from training)

Recommended splits:
    train : n_normal=1500, n_factor1=750, n_factor2=750, n_both=0
    val   : n_normal=300,  n_factor1=150, n_factor2=150, n_both=150
    test  : n_normal=300,  n_factor1=150, n_factor2=150, n_both=150

All images are 64×64 grayscale, normalised to [-1, 1].
Images are generated on-the-fly from integer seeds (zero memory cost).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Canonical spatial regions (pixel indices, 64×64 image)
# ─────────────────────────────────────────────────────────────────────────────

STAMP_A_ROWS = (2,  30)   # top-left quadrant
STAMP_A_COLS = (2,  30)
STAMP_B_ROWS = (34, 62)   # bottom-right quadrant
STAMP_B_COLS = (34, 62)

# Normalised bounding boxes [x_min, y_min, x_max, y_max] ∈ [0, 1]
# Note: x = columns, y = rows (image convention)
BBOX_F1 = (2/64, 2/64, 30/64, 30/64)
BBOX_F2 = (34/64, 34/64, 62/64, 62/64)

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1   # stamp A (circle)
LABEL_FACTOR2 = 2   # stamp B (cross)
LABEL_BOTH    = 3   # comorbid — withheld from training


# ─────────────────────────────────────────────────────────────────────────────
# Low-level image generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_background(rng: np.random.RandomState, H: int = 64, W: int = 64) -> np.ndarray:
    """
    Smooth Gaussian random field (matches SynDis background for consistency).
    Sum of 2–5 isotropic Gaussians + very low noise. Values in [0, 1].
    """
    bg = np.zeros((H, W), dtype=np.float32)
    n = rng.randint(2, 6)
    y, x = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cx    = rng.uniform(5,  W - 5)
        cy    = rng.uniform(5,  H - 5)
        sigma = rng.uniform(10, 25)
        amp   = rng.uniform(0.10, 0.30)
        bg   += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    bg += rng.normal(0, 0.015, (H, W)).astype(np.float32)
    return np.clip(bg, 0, 1)


def _stamp_circle(img: np.ndarray, rng: np.random.RandomState,
                  rows: tuple, cols: tuple,
                  H: int = 64, W: int = 64) -> np.ndarray:
    """
    Factor-1 stamp: filled circle confined to [rows, cols] region.
    Centre jittered ±4 px inside region; radius 6–10 px.
    """
    r0, r1 = rows
    c0, c1 = cols
    cr     = rng.uniform(r0 + 4, r1 - 4)
    cc     = rng.uniform(c0 + 4, c1 - 4)
    radius = rng.uniform(6, 10)
    amp    = rng.uniform(0.55, 0.80)
    y, x   = np.mgrid[0:H, 0:W]
    mask   = ((x - cc)**2 + (y - cr)**2) <= radius**2
    img    = img.copy()
    img[mask] = np.clip(img[mask] + amp, 0, 1)
    return img


def _stamp_cross(img: np.ndarray, rng: np.random.RandomState,
                 rows: tuple, cols: tuple,
                 H: int = 64, W: int = 64) -> np.ndarray:
    """
    Factor-2 stamp: cross/plus shape confined to [rows, cols] region.
    Centre jittered; arm length 7–11 px; thickness 3–5 px.
    """
    r0, r1 = rows
    c0, c1 = cols
    cr        = int(rng.uniform(r0 + 6, r1 - 6))
    cc        = int(rng.uniform(c0 + 6, c1 - 6))
    thickness = int(rng.uniform(3, 6))
    arm_len   = int(rng.uniform(7, 11))
    amp       = rng.uniform(0.55, 0.80)
    img       = img.copy()

    # Horizontal arm
    r_lo = max(0, cr - thickness);  r_hi = min(H, cr + thickness + 1)
    c_lo = max(0, cc - arm_len);    c_hi = min(W, cc + arm_len + 1)
    img[r_lo:r_hi, c_lo:c_hi] = np.clip(img[r_lo:r_hi, c_lo:c_hi] + amp, 0, 1)

    # Vertical arm
    r_lo2 = max(0, cr - arm_len);  r_hi2 = min(H, cr + arm_len + 1)
    c_lo2 = max(0, cc - thickness); c_hi2 = min(W, cc + thickness + 1)
    img[r_lo2:r_hi2, c_lo2:c_hi2] = np.clip(img[r_lo2:r_hi2, c_lo2:c_hi2] + amp, 0, 1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StampedMNISTDataset(Dataset):
    """
    Domain A: Positive control for the d0–d5 generalisation suite.
    All SepVAE framework assumptions satisfied by construction.

    Parameters
    ----------
    n_normal, n_factor1, n_factor2, n_both : int
        Images per class. Set n_both=0 for the training split.
    img_size : int
        Square image side length (default 64).
    global_seed : int
        Seed for deterministic generation; use different values for
        train / val / test to avoid overlap.
    mode : str
        'flat'    → returns (x: Tensor (1,H,W), y: LongTensor scalar)
        'triplet' → returns dict matching SyntheticCXRTripletDataset
                    interface for the optional JAX/MiniSepVAE path.
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
        assert mode in ('flat', 'triplet'), f"mode must be 'flat' or 'triplet', got {mode!r}"
        self.H = self.W  = img_size
        self.mode        = mode
        self._seeds:  list = []
        self._labels: list = []

        # Widely separated seed ranges so classes never share a seed
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
        x   = torch.from_numpy(img[None]).float() * 2.0 - 1.0   # (1, H, W) in [-1, 1]
        y   = torch.tensor(self._labels[idx], dtype=torch.long)
        if self.mode == 'flat':
            return x, y
        return self._to_triplet(x, y)

    def _generate(self, seed: int, label: int,
                  inject_f1: bool = False, inject_f2: bool = False) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)
        add_f1 = (label in (LABEL_FACTOR1, LABEL_BOTH)) or inject_f1
        add_f2 = (label in (LABEL_FACTOR2, LABEL_BOTH)) or inject_f2
        if add_f1:
            img = _stamp_circle(img, rng, STAMP_A_ROWS, STAMP_A_COLS, self.H, self.W)
        if add_f2:
            img = _stamp_cross(img,  rng, STAMP_B_ROWS, STAMP_B_COLS, self.H, self.W)
        return img

    def _to_triplet(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Return dict matching SyntheticCXRTripletDataset interface.
        Used by the optional JAX/MiniSepVAE (128×128) validation path.
        """
        return {
            'x':            x,
            'y':            y,
            'bbox_factor1': torch.tensor(BBOX_F1, dtype=torch.float32),
            'bbox_factor2': torch.tensor(BBOX_F2, dtype=torch.float32),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Convenience helpers (mirrors SynDisDataset API)

    def has_factor1(self) -> np.ndarray:
        """Binary array: 1 where Factor-1 (circle) is present."""
        return ((self.labels == LABEL_FACTOR1) |
                (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_factor2(self) -> np.ndarray:
        """Binary array: 1 where Factor-2 (cross) is present."""
        return ((self.labels == LABEL_FACTOR2) |
                (self.labels == LABEL_BOTH)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check visualisation (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = StampedMNISTDataset(n_normal=8, n_factor1=4, n_factor2=4, n_both=4,
                             global_seed=42)
    class_names = ["Normal", "Factor-1\n(circle)", "Factor-2\n(cross)", "Both"]

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

    plt.suptitle("StampedMNIST (Domain A) — one row per class", fontsize=12)
    plt.tight_layout()
    out = "stamped_mnist_preview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved preview → {out}")
