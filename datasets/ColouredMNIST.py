"""
ColouredMNIST — Domain C: Non-spatial, Independent
===================================================
Global colour channel boosts applied uniformly to MNIST digit images.

    Factor 1: R-channel boost (+0.40 to all pixels)  → "warm tint"
    Factor 2: B-channel boost (+0.40 to all pixels)  → "cool tint"

Both factors are spatially uniform — every pixel is equally affected,
so the model cannot use spatial features to separate them.  Only MI-driven
independence enforcement can force the encoder to allocate separate latent
dimensions for R-variation vs B-variation.

Why this tests A2 (spatial locality not required):
    The d0–d5 framework's D1 spatial gate is designed for localised factors.
    Colour boosts are global: the spatial gate would activate everywhere equally
    and provide no useful routing signal.  If disentanglement succeeds here,
    it is because of MI loss (D2) alone, proving the spatial prior is sufficient
    but not necessary.

Output: 3-channel RGB, shape (3, 64, 64) in [-1, 1].
BBOX_F1 = BBOX_F2 = (0.0, 0.0, 0.0, 0.0)  — sentinel for "no spatial prior".

Labels
------
    0 — Normal  : neutral tint (greyscale MNIST digit + noise)
    1 — Factor1 : + warm R boost
    2 — Factor2 : + cool B boost
    3 — Both    : + R boost AND + B boost (purple tint)

Requirements: torchvision (MNIST), Pillow
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

BBOX_F1 = (0.0, 0.0, 0.0, 0.0)   # sentinel: no spatial prior
BBOX_F2 = (0.0, 0.0, 0.0, 0.0)

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1
LABEL_FACTOR2 = 2
LABEL_BOTH    = 3

BOOST_R = 0.40   # R-channel additive boost for factor 1
BOOST_B = 0.40   # B-channel additive boost for factor 2


def _load_all_mnist(mnist_root: str, train: bool = True) -> list:
    """Load all MNIST images as a flat list of (28×28) uint8 arrays."""
    try:
        import torchvision.datasets as tvd
    except ImportError:
        raise ImportError(
            "torchvision is required for ColouredMNIST. "
            "Install with:  pip install torchvision")
    ds = tvd.MNIST(root=mnist_root, train=train, download=True)
    return [np.array(img) for img, _ in ds]


def _make_coloured_image(
    digit_np: np.ndarray,           # (28, 28) uint8
    rng: np.random.RandomState,
    factor1: bool,
    factor2: bool,
    H: int = 64,
    W: int = 64,
) -> np.ndarray:
    """
    Returns (3, H, W) float32 in [0, 1].

    digit_np provides the greyscale texture (background digit).
    Colour boosts are applied globally and uniformly.
    """
    from PIL import Image as PILImage

    # Resize digit to H×W
    grey = np.array(
        PILImage.fromarray(digit_np).resize((W, H), PILImage.BILINEAR),
        dtype=np.float32) / 255.0

    # Add light noise (makes it look more natural, same as SynDis approach)
    grey = np.clip(grey + rng.normal(0, 0.04, (H, W)).astype(np.float32), 0, 1)

    # Start with neutral 3-channel image
    img = np.stack([grey, grey, grey], axis=0)   # (3, H, W)

    # Apply global channel boosts
    if factor1:
        img[0] = np.clip(img[0] + BOOST_R, 0, 1)   # R boost (warm)
    if factor2:
        img[2] = np.clip(img[2] + BOOST_B, 0, 1)   # B boost (cool)

    return img   # (3, H, W) in [0, 1]


class ColouredMNISTDataset(Dataset):
    """
    Domain C — Non-spatial, independent.

    Parameters
    ----------
    mnist_root : str
        Directory for MNIST download/cache.
    n_normal, n_factor1, n_factor2, n_both : int
        Samples per class. Set n_both=0 for training split.
    img_size : int
    global_seed : int
    mode : str — 'flat' or 'triplet'
    """

    def __init__(
        self,
        mnist_root:  str = "~/.cache/mnist",
        n_normal:    int = 1500,
        n_factor1:   int = 750,
        n_factor2:   int = 750,
        n_both:      int = 0,
        img_size:    int = 64,
        global_seed: int = 42,
        mode:        str = 'flat',
    ):
        self.H = self.W = img_size
        self.mode = mode
        self._all_digits = _load_all_mnist(
            str(Path(mnist_root).expanduser()), train=True)

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

        self.labels    = np.array(self._labels, dtype=np.int64)
        self._bbox_zero = torch.zeros(4, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, idx: int):
        x_np = self._generate(self._seeds[idx], self._labels[idx])
        x = torch.from_numpy(x_np).float() * 2.0 - 1.0   # (3, H, W) in [-1, 1]
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        if self.mode == 'triplet':
            return {"x": x, "y": y,
                    "bbox_factor1": self._bbox_zero.clone(),
                    "bbox_factor2": self._bbox_zero.clone()}
        return x, y

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        digit_idx = rng.randint(0, len(self._all_digits))
        f1 = label in (LABEL_FACTOR1, LABEL_BOTH)
        f2 = label in (LABEL_FACTOR2, LABEL_BOTH)
        return _make_coloured_image(
            self._all_digits[digit_idx], rng, f1, f2, self.H, self.W)

    def has_factor1(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR1) |
                (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_factor2(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR2) |
                (self.labels == LABEL_BOTH)).astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ColouredMNISTDataset(n_normal=4, n_factor1=4, n_factor2=4, n_both=4)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    names = ["Normal", "F1 (warm R)", "F2 (cool B)", "Both (purple)"]
    for row, lbl in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds.labels) if l == lbl][:4]
        for col, idx in enumerate(idxs):
            x, _ = ds[idx]
            img = np.clip((x.numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(names[row])
    plt.suptitle("ColouredMNIST — 4 rows per class")
    plt.tight_layout()
    plt.savefig("coloured_mnist_preview.png", dpi=100)
    print("Saved coloured_mnist_preview.png")
