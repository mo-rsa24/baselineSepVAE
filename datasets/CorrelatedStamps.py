"""
CorrelatedStamps — Domain B
============================
Same spatial layout as Domain A (StampedMNIST) but with a controlled
spurious co-occurrence rate in the training set, simulating the
cardiomegaly/pleural-effusion comorbidity structure in VinBigData.

Assumption violation:
    A1 — Statistical independence is VIOLATED:
         `correlation_rate` fraction of single-factor training images
         secretly also contain the other factor (unannounced in the label).
         This mimics real-world comorbidity where pathologies co-occur
         at ~30% rate but labels may only record the primary diagnosis.

All other assumptions remain satisfied (A2 spatial locality, A3 null class,
A4 additive composition).

Key experimental purpose:
    Test whether the MI discriminator (D2) can detect and partially recover
    from spurious correlation. Compare cross_head_score to Domain A at
    identical stages to quantify the degradation and recovery.

Label scheme (identical to Domain A / SynDis):
    0 — Normal   (background only)
    1 — Factor-1 (circle, possibly +cross in training)
    2 — Factor-2 (cross, possibly +circle in training)
    3 — Both     (clean — withheld from training and used for composition eval)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.StampedMNIST import (
    _make_background, _stamp_circle, _stamp_cross,
    STAMP_A_ROWS, STAMP_A_COLS, STAMP_B_ROWS, STAMP_B_COLS,
    BBOX_F1, BBOX_F2,
    LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH,
)


class CorrelatedStampsDataset(Dataset):
    """
    Domain B: Spatial + correlated factors.

    Identical to StampedMNISTDataset except that `correlation_rate` fraction
    of single-factor images in the *training* split secretly contain the
    other factor. The label remains binary, so the corruption is latent.

    For val/test splits, always use correlation_rate=0.0 (clean labels).

    Parameters
    ----------
    correlation_rate : float
        Fraction of factor1-labelled images that also contain factor2 (and
        vice versa). Should be 0.0 for val/test. Default 0.30.
    """

    def __init__(
        self,
        n_normal:         int   = 1500,
        n_factor1:        int   = 750,
        n_factor2:        int   = 750,
        n_both:           int   = 0,
        img_size:         int   = 64,
        global_seed:      int   = 42,
        correlation_rate: float = 0.30,
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
            'bbox_factor1': torch.tensor(BBOX_F1, dtype=torch.float32),
            'bbox_factor2': torch.tensor(BBOX_F2, dtype=torch.float32),
        }

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)

        add_f1 = label in (LABEL_FACTOR1, LABEL_BOTH)
        add_f2 = label in (LABEL_FACTOR2, LABEL_BOTH)

        # Inject spurious co-occurrence for single-factor images only.
        # The rng.random() call is reproducible: same seed → same corruption.
        if self.correlation_rate > 0.0:
            if label == LABEL_FACTOR1 and rng.random() < self.correlation_rate:
                add_f2 = True   # secretly add factor2 even though label=1
            elif label == LABEL_FACTOR2 and rng.random() < self.correlation_rate:
                add_f1 = True   # secretly add factor1 even though label=2

        if add_f1:
            img = _stamp_circle(img, rng, STAMP_A_ROWS, STAMP_A_COLS, self.H, self.W)
        if add_f2:
            img = _stamp_cross(img,  rng, STAMP_B_ROWS, STAMP_B_COLS, self.H, self.W)
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

    ds = CorrelatedStampsDataset(n_normal=8, n_factor1=8, n_factor2=8, n_both=4,
                                  global_seed=42, correlation_rate=0.30)

    # Count how many factor1 images got spuriously injected with factor2
    n_f1  = (ds.labels == LABEL_FACTOR1).sum()
    corrupted = sum(
        1 for i, l in enumerate(ds.labels)
        if l == LABEL_FACTOR1 and
        np.random.RandomState(ds._seeds[i]).random() <  # re-simulate
        ds.correlation_rate
    )
    print(f"Factor1 images: {n_f1}, spuriously showing factor2: {corrupted} "
          f"({100*corrupted/n_f1:.1f}%)")

    class_names = ["Normal", "Factor-1\n(label=1,\nmay have F2)", "Factor-2\n(label=2)", "Both"]
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

    plt.suptitle("CorrelatedStamps (Domain B, corr=0.30) — one row per label", fontsize=11)
    plt.tight_layout()
    plt.savefig("correlated_stamps_preview.png", dpi=120, bbox_inches="tight")
    print("Saved → correlated_stamps_preview.png")
