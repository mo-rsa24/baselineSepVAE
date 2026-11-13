"""
SynDis — Synthetic Disease Disentanglement Dataset
====================================================
Purpose-built testbed for the SepVAE disentanglement hypothesis.

Structure mirrors VinBigData (cardiomegaly / effusion):

    Label 0 — Normal:    smooth Gaussian background only
    Label 1 — Disease 1: background + Gaussian blob at centre-left
                         (cardiomegaly analog: enlarged cardiac silhouette)
    Label 2 — Disease 2: background + Gaussian band at bottom
                         (effusion analog: blunting of costophrenic angles)
    Label 3 — Both:      background + blob + band
                         (comorbid — withheld from training, used only for
                          composition evaluation)

All images are 64×64 grayscale, normalised to [-1, 1].
Images are generated on-the-fly from integer seeds (zero memory cost).

Recommended split
-----------------
    train : n_normal=1500, n_disease1=750, n_disease2=750, n_both=0
    val   : n_normal=300,  n_disease1=150, n_disease2=150, n_both=150
    test  : n_normal=300,  n_disease1=150, n_disease2=150, n_both=150

Spatial regions (pixel indices, can be used for composition scoring)
--------------------------------------------------------------------
    BLOB_ROWS   = (20, 50),  BLOB_COLS   = (10, 38)  # centre-left
    STRIPE_ROWS = (48, 64),  STRIPE_COLS = (0,  64)  # bottom band
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical spatial regions for each disease signal
BLOB_ROWS   = (20, 50)
BLOB_COLS   = (10, 38)
STRIPE_ROWS = (48, 64)
STRIPE_COLS = (0,  64)

LABEL_NORMAL   = 0
LABEL_DISEASE1 = 1   # blob
LABEL_DISEASE2 = 2   # stripe
LABEL_BOTH     = 3   # comorbid


# ---------------------------------------------------------------------------
# Low-level image generators
# ---------------------------------------------------------------------------

def _make_background(rng: np.random.RandomState, H: int = 64, W: int = 64) -> np.ndarray:
    """
    Smooth Gaussian random field representing anatomy / acquisition variation.
    Sum of 3-6 isotropic Gaussians at random positions + low-level noise.
    Values in [0, 1].
    """
    bg = np.zeros((H, W), dtype=np.float32)
    n = rng.randint(3, 7)
    y, x = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cx    = rng.uniform(5,  W - 5)
        cy    = rng.uniform(5,  H - 5)
        sigma = rng.uniform(8,  20)
        amp   = rng.uniform(0.15, 0.40)
        bg   += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    bg += rng.normal(0, 0.02, (H, W)).astype(np.float32)
    return np.clip(bg, 0, 1)


def _add_disease1(img: np.ndarray, rng: np.random.RandomState,
                  H: int = 64, W: int = 64) -> np.ndarray:
    """
    Cardiomegaly analog — Gaussian blob at centre-left (cardiac silhouette region).
    Centre jittered within BLOB_ROWS / BLOB_COLS neighbourhood.
    """
    cx    = rng.uniform(18, 28)   # cols 18-28 (left of centre)
    cy    = rng.uniform(28, 40)   # rows 28-40 (mid-height)
    sigma = rng.uniform(5,   9)
    amp   = rng.uniform(0.40, 0.65)
    y, x  = np.mgrid[0:H, 0:W]
    blob  = amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return np.clip(img + blob.astype(np.float32), 0, 1)


def _add_disease2(img: np.ndarray, rng: np.random.RandomState,
                  H: int = 64, W: int = 64) -> np.ndarray:
    """
    Effusion analog — Gaussian band at bottom (costophrenic angle region).
    Centre row jittered in 50-58; full-width horizontal.
    """
    center_row = rng.uniform(50, 58)
    width      = rng.uniform(4,   8)
    amp        = rng.uniform(0.35, 0.60)
    y          = np.arange(H, dtype=np.float32)
    profile    = amp * np.exp(-((y - center_row)**2) / (2 * width**2))
    stripe     = np.tile(profile[:, None], (1, W)).astype(np.float32)
    return np.clip(img + stripe, 0, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynDisDataset(Dataset):
    """
    PyTorch Dataset for the SynDis synthetic disease testbed.

    Parameters
    ----------
    n_normal, n_disease1, n_disease2, n_both : int
        Number of images per class.  Set n_both=0 for the training split.
    img_size : int
        Square image side length (default 64).
    global_seed : int
        Seed for deterministic generation.  Use different values for
        train / val / test to avoid overlap.

    Returns (from __getitem__)
    --------------------------
    x : float32 tensor (1, H, W) in [-1, 1]
    y : long tensor scalar — class label in {0, 1, 2, 3}
    """

    def __init__(
        self,
        n_normal:   int = 1500,
        n_disease1: int = 750,
        n_disease2: int = 750,
        n_both:     int = 0,
        img_size:   int = 64,
        global_seed: int = 42,
    ):
        self.H = self.W = img_size
        self._seeds:  list[int] = []
        self._labels: list[int] = []

        # Use widely separated seed ranges so classes never share a seed
        base = global_seed * 1_000_000
        for label, count in [
            (LABEL_NORMAL,   n_normal),
            (LABEL_DISEASE1, n_disease1),
            (LABEL_DISEASE2, n_disease2),
            (LABEL_BOTH,     n_both),
        ]:
            for i in range(count):
                self._seeds.append(base + i)
                self._labels.append(label)
            base += 100_000

        self.labels = np.array(self._labels, dtype=np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, idx: int):
        img = self._generate(self._seeds[idx], self._labels[idx])
        # Normalise to [-1, 1] (matches SepVAE convention)
        x = torch.from_numpy(img[None]).float() * 2.0 - 1.0   # (1, H, W)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)
        if label in (LABEL_DISEASE1, LABEL_BOTH):
            img = _add_disease1(img, rng, self.H, self.W)
        if label in (LABEL_DISEASE2, LABEL_BOTH):
            img = _add_disease2(img, rng, self.H, self.W)
        return img

    # ------------------------------------------------------------------
    # Convenience helpers

    def has_disease1(self) -> np.ndarray:
        """Binary array: 1 where disease-1 (blob) is present."""
        return ((self.labels == LABEL_DISEASE1) | (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_disease2(self) -> np.ndarray:
        """Binary array: 1 where disease-2 (stripe) is present."""
        return ((self.labels == LABEL_DISEASE2) | (self.labels == LABEL_BOTH)).astype(np.float32)


# ---------------------------------------------------------------------------
# Quick sanity-check visualisation (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_train = SynDisDataset(n_normal=8, n_disease1=4, n_disease2=4, n_both=4,
                             global_seed=42)
    class_names = ["Normal", "Disease-1\n(blob)", "Disease-2\n(stripe)", "Both"]

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for row, label in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds_train.labels) if l == label][:5]
        for col, idx in enumerate(idxs):
            x, y = ds_train[idx]
            img = (x.squeeze().numpy() + 1) / 2   # to [0, 1]
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(class_names[label], fontsize=9)

    plt.suptitle("SynDis — one row per class", fontsize=12)
    plt.tight_layout()
    out = "syndis_preview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved preview → {out}")
