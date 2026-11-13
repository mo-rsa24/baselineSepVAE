"""
CorrelatedTextures — Domain D
==============================
The hardest case: BOTH key assumptions are violated simultaneously.

Assumption violations:
    A1 — Statistical independence is VIOLATED:
         High correlation_rate (default 0.50) means half of all single-factor
         training images secretly contain both texture factors.
    A2 — Spatial locality is VIOLATED:
         Both factors are global Fourier textures; no meaningful bbox exists.

All other assumptions hold (A3 null class, A4 additive composition).

Key experimental purpose:
    Document the BOUNDARY CONDITION of the d0–d5 framework.
    Expected outcome: G2 and G3 NOT met at any stage. This is an
    HONEST NEGATIVE CONTROL that shows where the framework breaks down,
    supporting the claim that both A1 (independence) AND A2 (spatial prior)
    can independently contribute to framework success, but neither alone
    can overcome the double violation.

    Specifically: since Domain C succeeds without A2, and Domain B
    partially succeeds without A1, Domain D (lacking both) should fail —
    confirming the 2×2 factorial structure of the experiment.

Label scheme (identical to all other domains):
    0 — Normal      (background only)
    1 — Factor-1    (H-stripes; may also secretly contain V-stripes in train)
    2 — Factor-2    (V-stripes; may also secretly contain H-stripes in train)
    3 — Both        (clean composition set, withheld from training)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.TextureFactors import (
    _make_background, _make_stripe_h, _make_stripe_v,
    FREQ_F1_RANGE, FREQ_F2_RANGE,
    BBOX_F1, BBOX_F2,
    LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH,
    fourier_composition_score,   # re-export for train_generalize.py
)


class CorrelatedTexturesDataset(Dataset):
    """
    Domain D: Global + correlated factors (double assumption violation).

    Identical to TextureFactorsDataset except that `correlation_rate`
    fraction of single-factor training images secretly contain the
    other texture. The label remains binary (corruption is latent).

    Parameters
    ----------
    correlation_rate : float
        Fraction of factor1-labelled images that also contain factor2
        (and vice versa). Recommended 0.50 for the standard experiment.
        Set 0.0 for val/test splits.
    """

    def __init__(
        self,
        n_normal:         int   = 1500,
        n_factor1:        int   = 750,
        n_factor2:        int   = 750,
        n_both:           int   = 0,
        img_size:         int   = 64,
        global_seed:      int   = 42,
        correlation_rate: float = 0.50,
        mode:             str   = 'flat',
    ):
        assert mode in ('flat', 'triplet')
        assert 0.0 <= correlation_rate <= 1.0
        self.H = self.W          = img_size
        self.mode                = mode
        self.correlation_rate    = correlation_rate
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
            'bbox_factor1': torch.zeros(4, dtype=torch.float32),
            'bbox_factor2': torch.zeros(4, dtype=torch.float32),
        }

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)

        add_f1 = label in (LABEL_FACTOR1, LABEL_BOTH)
        add_f2 = label in (LABEL_FACTOR2, LABEL_BOTH)

        # Inject spurious co-occurrence (same rng stream → reproducible)
        if self.correlation_rate > 0.0:
            if label == LABEL_FACTOR1 and rng.random() < self.correlation_rate:
                add_f2 = True
            elif label == LABEL_FACTOR2 and rng.random() < self.correlation_rate:
                add_f1 = True

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

    ds = CorrelatedTexturesDataset(n_normal=8, n_factor1=8, n_factor2=8, n_both=4,
                                    global_seed=42, correlation_rate=0.50)
    class_names = ["Normal", "Label=F1\n(50% also have F2)", "Label=F2\n(50% also have F1)", "Both"]

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for row, label in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds.labels) if l == label][:5]
        for col, idx in enumerate(idxs):
            x, y = ds[idx]
            img  = (x.squeeze().numpy() + 1) / 2
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(class_names[label], fontsize=8)

    plt.suptitle("CorrelatedTextures (Domain D, corr=0.50) — one row per label", fontsize=11)
    plt.tight_layout()
    plt.savefig("correlated_textures_preview.png", dpi=120, bbox_inches="tight")
    print("Saved → correlated_textures_preview.png")
