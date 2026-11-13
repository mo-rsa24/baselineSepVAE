"""
CelebAFactor — Domains E (balanced) and F (natural)
====================================================
Real face images with two approximately independent binary attributes:

    Factor 1: Eyeglasses  — eye/glasses region
    Factor 2: Bald        — crown/hair region

The independence argument mirrors the CXR thesis claim:
    "Wearing glasses does not cause baldness; dataset correlations
     are demographic confounding, not causal."
        ↕  (same structure as)
    "Cardiomegaly does not cause pleural effusion; dataset correlations
     are comorbidity confounding, not causal mechanisms."

Domain E (balanced=True):
    Subsample to ~n_per_class per single-factor class, removing the natural
    demographic co-occurrence.  ~750/class → ~3000 total (same as toy domains).
    All assumptions A1–A4 approximately satisfied.  Expected: G1 ✓ G2 ✓ G3 ✓.

Domain F (balanced=False):
    Use raw CelebA split with natural co-occurrence rates preserved.
    A1 is violated by real-world demographic correlation (not injected).
    Tests whether MI discriminator recovers independence from dataset-level
    correlations that were not artificially injected.

Approximate bboxes for 64×64 face images [x_min, y_min, x_max, y_max]:
    BBOX_F1 (glasses): (0.15, 0.44, 0.85, 0.65)  — eye/glasses horizontal band
    BBOX_F2 (bald):    (0.15, 0.00, 0.85, 0.32)  — crown of head

Output: 3-channel RGB, (3, 64, 64) in [-1, 1].

Download
--------
    torchvision.datasets.CelebA(root, download=True)
    Note: Google Drive download can fail on headless servers.
    Manual fallback: download img_align_celeba.zip + Anno files to celeba_root/celeba/.
    See: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

BBOX_F1 = (0.15, 0.44, 0.85, 0.65)   # eye/glasses band
BBOX_F2 = (0.15, 0.00, 0.85, 0.32)   # crown of head

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1   # Eyeglasses
LABEL_FACTOR2 = 2   # Bald
LABEL_BOTH    = 3

_ATTR_F1 = "Eyeglasses"
_ATTR_F2 = "Bald"


class CelebAFactorDataset(Dataset):
    """
    Domains E and F — Real images, spatial, approximate independence.

    Parameters
    ----------
    celeba_root : str
        Root directory for CelebA (will download if missing).
    split : str
        'train', 'valid', or 'test'.
    balanced : bool
        True  = Domain E: subsample to equal class sizes.
        False = Domain F: natural distribution.
    n_per_class : int
        Target samples per single-factor class in balanced mode.
    global_seed : int
    mode : str — 'flat' or 'triplet'
    img_size : int — 64
    """

    def __init__(
        self,
        celeba_root:  str = "~/.cache/celeba",
        split:        str = "train",
        balanced:     bool = True,
        n_per_class:  int = 750,
        global_seed:  int = 42,
        mode:         str = 'flat',
        img_size:     int = 64,
    ):
        self.mode = mode
        self._bbox_f1 = torch.tensor(BBOX_F1, dtype=torch.float32)
        self._bbox_f2 = torch.tensor(BBOX_F2, dtype=torch.float32)

        try:
            import torchvision.datasets as tvd
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "torchvision is required for CelebAFactor. "
                "Install with:  pip install torchvision")

        celeba_root = str(Path(celeba_root).expanduser())
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(img_size),
            T.ToTensor(),          # (3, H, W) in [0, 1]
        ])

        raw = tvd.CelebA(
            root=celeba_root, split=split, download=True,
            target_type="attr", transform=transform)

        # Locate attribute columns
        attr_names = raw.attr_names
        f1_idx = attr_names.index(_ATTR_F1)
        f2_idx = attr_names.index(_ATTR_F2)

        # Build per-sample labels
        all_attrs = raw.attr.numpy()   # (N, 40) int
        f1 = all_attrs[:, f1_idx].astype(bool)
        f2 = all_attrs[:, f2_idx].astype(bool)

        labels = np.where(
            f1 & f2,  LABEL_BOTH,
            np.where(f1, LABEL_FACTOR1,
            np.where(f2, LABEL_FACTOR2, LABEL_NORMAL))).astype(np.int64)

        indices = np.arange(len(raw))

        if balanced:
            rng = np.random.RandomState(global_seed)
            sel = []
            for cls in [LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2]:
                cls_idx = indices[labels == cls]
                n = min(n_per_class, len(cls_idx))
                sel.append(rng.choice(cls_idx, n, replace=False))
            sel.append(indices[labels == LABEL_BOTH])   # keep all "both"
            indices = np.concatenate(sel)
            indices.sort()
            labels  = labels[indices]

        self._raw     = raw
        self._indices = indices
        self.labels   = labels

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        x_raw, _ = self._raw[int(self._indices[idx])]   # (3, H, W) in [0, 1]
        x = x_raw * 2.0 - 1.0                           # → [-1, 1]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.mode == 'triplet':
            return {"x": x, "y": y,
                    "bbox_factor1": self._bbox_f1.clone(),
                    "bbox_factor2": self._bbox_f2.clone()}
        return x, y

    def has_factor1(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR1) |
                (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_factor2(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR2) |
                (self.labels == LABEL_BOTH)).astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CelebAFactorDataset(balanced=True, n_per_class=4)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    names = ["Normal", "F1 (glasses)", "F2 (bald)", "Both"]
    for row, lbl in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds.labels) if l == lbl][:4]
        for col, idx in enumerate(idxs[:4]):
            x, _ = ds[idx]
            img = np.clip((x.numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(names[row])
    plt.suptitle("CelebAFactor — 4 rows per class")
    plt.tight_layout()
    plt.savefig("celeba_factor_preview.png", dpi=100)
    print("Saved celeba_factor_preview.png")
