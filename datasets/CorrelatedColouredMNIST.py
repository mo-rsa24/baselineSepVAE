"""
CorrelatedColouredMNIST — Domain D: Non-spatial, Correlated
============================================================
Extends ColouredMNIST with spurious co-occurrence: a fraction of warm-tint
images also secretly get the cool-tint boost (and vice versa).

Both A1 (independence) and A2 (spatial locality) are violated.
Expected outcome: G2 ✗ G3 ✗ even at d5 — documents framework boundary condition.
"""

import numpy as np
from datasets.ColouredMNIST import (
    ColouredMNISTDataset,
    _make_coloured_image,
    LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH,
)


class CorrelatedColouredMNISTDataset(ColouredMNISTDataset):
    """
    Domain D — Non-spatial, correlated.

    Parameters
    ----------
    correlation_rate : float
        Fraction of single-label images that secretly also get the other boost.
    **kwargs : ColouredMNISTDataset parameters.
    """

    def __init__(self, correlation_rate: float = 0.50, **kwargs):
        super().__init__(**kwargs)
        self.correlation_rate = correlation_rate

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        digit_idx = rng.randint(0, len(self._all_digits))
        f1 = label in (LABEL_FACTOR1, LABEL_BOTH)
        f2 = label in (LABEL_FACTOR2, LABEL_BOTH)

        # Spurious co-occurrence on single-label training samples
        if label in (LABEL_FACTOR1, LABEL_FACTOR2) and self.correlation_rate > 0:
            rng_c = np.random.RandomState(seed ^ 0xABCD1234)
            if rng_c.random() < self.correlation_rate:
                if label == LABEL_FACTOR1:
                    f2 = True
                else:
                    f1 = True

        return _make_coloured_image(
            self._all_digits[digit_idx], rng, f1, f2, self.H, self.W)
