"""
CorrelatedDSprites — Domain D: Non-spatial, Correlated
=======================================================
Extends DSprites with spurious co-occurrence during training.

    correlation_rate fraction of single-label training images secretly show
    the "Both" version (large scale AND high orientation).
    Labels remain wrong (e.g. label=1 for what is actually a Both image).
    Val/test always use correlation_rate=0.0.

Assumptions violated: A1 (statistical independence) AND A2 (spatial locality).
Expected outcome: G2 ✗ G3 ✗ — documents the double-violation boundary.
"""

import numpy as np
import torch

from datasets.DSprites import (
    DSpritesDataset,
    LABEL_FACTOR1, LABEL_FACTOR2,
    BBOX_F1, BBOX_F2,
)


class CorrelatedDSpritesDataset(DSpritesDataset):
    """
    Domain D — Non-spatial, correlated.

    Parameters
    ----------
    correlation_rate : float
        Fraction of F1/F2 training images that secretly show a Both sample.
        Use 0.0 for val/test.
    **kwargs : DSpritesDataset parameters.
    """

    def __init__(self, correlation_rate: float = 0.50, **kwargs):
        super().__init__(**kwargs)
        self.correlation_rate = correlation_rate

        # Decide which F1/F2 items to corrupt (deterministic per seed)
        corr_seed = kwargs.get("global_seed", 42) ^ 0xABCD1234
        rng_c     = np.random.RandomState(corr_seed)

        # Build a cycling index into the both_pool
        self._corrupt_flags: list = []   # True → swap image with both_pool sample
        self._corrupt_pool_idx: list = []

        pool_ptr = 0
        for img_np, label in self._items:
            if label in (LABEL_FACTOR1, LABEL_FACTOR2) and correlation_rate > 0:
                if rng_c.random() < correlation_rate:
                    self._corrupt_flags.append(True)
                    self._corrupt_pool_idx.append(
                        pool_ptr % len(self._both_pool))
                    pool_ptr += 1
                    continue
            self._corrupt_flags.append(False)
            self._corrupt_pool_idx.append(-1)

    def __getitem__(self, idx: int):
        img_np, label = self._items[idx]

        if self._corrupt_flags[idx] and self._both_pool:
            # Secretly replace image with a "both" sample; label stays as-is
            img_np = self._both_pool[self._corrupt_pool_idx[idx]]

        x = self._to_tensor(img_np)
        y = torch.tensor(label, dtype=torch.long)
        if self.mode == 'triplet':
            return {"x": x, "y": y,
                    "bbox_factor1": self._bbox_zero.clone(),
                    "bbox_factor2": self._bbox_zero.clone()}
        return x, y
