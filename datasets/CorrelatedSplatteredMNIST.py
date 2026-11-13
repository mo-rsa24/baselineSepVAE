"""
CorrelatedSplatteredMNIST — Domain B: Spatial, Correlated
==========================================================
Extends SplatteredMNIST with spurious co-occurrence during training.

    correlation_rate fraction of single-label training images secretly also
    contain the other factor (latent corruption, not reflected in labels).
    Val/test always use correlation_rate=0.0.

Assumption violated: A1 (statistical independence).
Tests whether the MI discriminator (D2) recovers independence under
realistic comorbidity confounding.
"""

import numpy as np
import torch

from datasets.SplatteredMNIST import (
    SplatteredMNISTDataset,
    _render_digit_into_slot,
    SLOT_F1_ROWS, SLOT_F1_COLS,
    SLOT_F2_ROWS, SLOT_F2_COLS,
    LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH,
    BBOX_F1, BBOX_F2,
)


class CorrelatedSplatteredMNISTDataset(SplatteredMNISTDataset):
    """
    Domain B — Spatial, correlated.

    Parameters
    ----------
    correlation_rate : float
        Fraction of single-label training images that secretly also contain
        the other digit.  Use 0.0 for val/test.
    **kwargs :
        All SplatteredMNISTDataset parameters.
    """

    def __init__(self, correlation_rate: float = 0.30, **kwargs):
        super().__init__(**kwargs)
        self.correlation_rate = correlation_rate

    def _generate(self, seed: int, label: int) -> np.ndarray:
        # Base image (parent logic)
        img = super()._generate(seed, label)

        if label in (LABEL_FACTOR1, LABEL_FACTOR2) and self.correlation_rate > 0:
            # Independent RNG stream for corruption decision
            rng_c = np.random.RandomState(seed ^ 0xDEADBEEF)
            if rng_c.random() < self.correlation_rate:
                rng_add = np.random.RandomState(seed + 500_000_000)
                if label == LABEL_FACTOR1:
                    d = self._digits_f2[rng_add.randint(0, len(self._digits_f2))]
                    patch = _render_digit_into_slot(
                        d, rng_add, SLOT_F2_ROWS, SLOT_F2_COLS, self.H, self.W)
                else:
                    d = self._digits_f1[rng_add.randint(0, len(self._digits_f1))]
                    patch = _render_digit_into_slot(
                        d, rng_add, SLOT_F1_ROWS, SLOT_F1_COLS, self.H, self.W)
                import numpy as _np
                img = _np.clip(img + patch, 0, 1)

        return img
