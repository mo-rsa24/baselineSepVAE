"""
SplatteredMNIST — Domain A: Spatial, Independent
=================================================
Two MNIST digits rendered with soft Gaussian edges and internal splatters,
placed in fixed non-overlapping quadrants on a smooth Gaussian background.

    Factor 1: digit '1' in top-left slot (rows 4–30, cols 4–30)  ← cardiomegaly analog
    Factor 2: digit '7' in bottom-right slot (rows 34–60, cols 34–60)  ← effusion analog

"Splattering" = Gaussian envelope (soft edges) + small random Gaussian blobs
within the digit region, mimicking real radiological findings:
    - Soft boundary ↔ gradual opacity transition of a real pathology
    - Internal splatters ↔ parenchymal texture changes within a pathological region

Why better than StampedMNIST:
    - Hard-edged geometric stamps can be separated by simple spatial thresholding.
    - Splattered digits require the model to learn shape + texture, not just location.
    - Soft envelope means each digit grades into the background like a true CXR finding.

Labels
------
    0 — Normal  : background only
    1 — Factor1 : digit '1' in top-left
    2 — Factor2 : digit '7' in bottom-right
    3 — Both    : both digits (withheld from training, used in val/test composition eval)

Bboxes (normalised [x_min, y_min, x_max, y_max])
-------------------------------------------------
    BBOX_F1 = (4/64, 4/64, 30/64, 30/64)
    BBOX_F2 = (34/64, 34/64, 60/64, 60/64)

Requirements: torchvision (MNIST download), Pillow (resize)
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ─── Slot pixel indices (pixel coords in 64×64 image) ─────────────────────────
SLOT_F1_ROWS = (4,  30)
SLOT_F1_COLS = (4,  30)
SLOT_F2_ROWS = (34, 60)
SLOT_F2_COLS = (34, 60)

# ─── Normalised bboxes [x_min, y_min, x_max, y_max] ──────────────────────────
BBOX_F1 = (SLOT_F1_COLS[0] / 64, SLOT_F1_ROWS[0] / 64,
           SLOT_F1_COLS[1] / 64, SLOT_F1_ROWS[1] / 64)
BBOX_F2 = (SLOT_F2_COLS[0] / 64, SLOT_F2_ROWS[0] / 64,
           SLOT_F2_COLS[1] / 64, SLOT_F2_ROWS[1] / 64)

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1
LABEL_FACTOR2 = 2
LABEL_BOTH    = 3

DIGIT_F1 = 1   # factor-1 digit class: '1' (thin vertical stroke)
DIGIT_F2 = 7   # factor-2 digit class: '7' (angled with serif)


# ─── MNIST loader ─────────────────────────────────────────────────────────────

def _load_mnist_by_class(mnist_root: str, train: bool = True) -> dict:
    """Load MNIST and return {class_int: list[np.ndarray (28×28 uint8)]}."""
    try:
        import torchvision.datasets as tvd
    except ImportError:
        raise ImportError(
            "torchvision is required for SplatteredMNIST. "
            "Install with:  pip install torchvision")
    ds = tvd.MNIST(root=mnist_root, train=train, download=True)
    by_class: dict[int, list] = {}
    for img, label in ds:
        k = int(label)
        by_class.setdefault(k, []).append(np.array(img))
    return by_class


# ─── Image generators ─────────────────────────────────────────────────────────

def _make_background(rng: np.random.RandomState,
                     H: int = 64, W: int = 64) -> np.ndarray:
    """SynDis-style smooth Gaussian random field. Values in [0, 1]."""
    bg = np.zeros((H, W), dtype=np.float32)
    y, x = np.mgrid[0:H, 0:W]
    for _ in range(rng.randint(3, 7)):
        cx    = rng.uniform(5, W - 5)
        cy    = rng.uniform(5, H - 5)
        sigma = rng.uniform(8, 20)
        amp   = rng.uniform(0.10, 0.30)
        bg   += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    bg += rng.normal(0, 0.02, (H, W)).astype(np.float32)
    return np.clip(bg, 0, 1)


def _render_digit_into_slot(
    digit_np: np.ndarray,          # (28, 28) uint8
    rng: np.random.RandomState,
    slot_rows: tuple,              # (r_min, r_max)
    slot_cols: tuple,              # (c_min, c_max)
    H: int = 64,
    W: int = 64,
) -> np.ndarray:
    """
    Render a 28×28 MNIST digit into a slot with:
      1. Bilinear resize to slot dimensions
      2. Gaussian envelope — soft boundary (mimics gradual opacity transition)
      3. Gaussian splatters within slot (mimics parenchymal texture changes)
    Returns (H, W) float32 patch, zeros outside slot.
    """
    from PIL import Image as PILImage

    slot_h = slot_rows[1] - slot_rows[0]
    slot_w = slot_cols[1] - slot_cols[0]

    # 1. Resize
    digit = np.array(
        PILImage.fromarray(digit_np).resize((slot_w, slot_h), PILImage.BILINEAR),
        dtype=np.float32) / 255.0

    # 2. Gaussian envelope
    y_g, x_g = np.mgrid[0:slot_h, 0:slot_w].astype(np.float32)
    cy, cx   = (slot_h - 1) / 2.0, (slot_w - 1) / 2.0
    sigma_env = rng.uniform(slot_h * 0.38, slot_h * 0.52)
    envelope  = np.exp(-((x_g - cx) ** 2 + (y_g - cy) ** 2) / (2 * sigma_env ** 2))
    digit_soft = digit * envelope

    # 3. Splatters
    splatters = np.zeros((slot_h, slot_w), dtype=np.float32)
    for _ in range(rng.randint(3, 7)):
        spx = rng.uniform(2, slot_w - 2)
        spy = rng.uniform(2, slot_h - 2)
        sig = rng.uniform(1.0, 3.0)
        amp = rng.uniform(0.10, 0.25)
        splatters += amp * np.exp(
            -((x_g - spx) ** 2 + (y_g - spy) ** 2) / (2 * sig ** 2))

    patch = rng.uniform(0.50, 0.80) * (digit_soft + 0.3 * splatters)

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[slot_rows[0]:slot_rows[1], slot_cols[0]:slot_cols[1]] = patch
    return canvas


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SplatteredMNISTDataset(Dataset):
    """
    Domain A — Spatial, independent.

    Parameters
    ----------
    mnist_root : str
        Directory for MNIST download/cache.
    n_normal, n_factor1, n_factor2, n_both : int
        Samples per class. Set n_both=0 for training split.
    img_size : int
    global_seed : int
    mode : str
        'flat'    → (x, y)
        'triplet' → dict with bbox_factor1, bbox_factor2 tensors
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
        self.mnist_root = str(Path(mnist_root).expanduser())

        by_class = _load_mnist_by_class(self.mnist_root, train=True)
        self._digits_f1 = by_class[DIGIT_F1]
        self._digits_f2 = by_class[DIGIT_F2]

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

        self.labels   = np.array(self._labels, dtype=np.int64)
        self._bbox_f1 = torch.tensor(BBOX_F1, dtype=torch.float32)
        self._bbox_f2 = torch.tensor(BBOX_F2, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, idx: int):
        x_np = self._generate(self._seeds[idx], self._labels[idx])
        x = torch.from_numpy(x_np[None]).float() * 2.0 - 1.0   # (1, H, W) in [-1,1]
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        if self.mode == 'triplet':
            return {"x": x, "y": y,
                    "bbox_factor1": self._bbox_f1.clone(),
                    "bbox_factor2": self._bbox_f2.clone()}
        return x, y

    def _generate(self, seed: int, label: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = _make_background(rng, self.H, self.W)
        if label in (LABEL_FACTOR1, LABEL_BOTH):
            d = self._digits_f1[rng.randint(0, len(self._digits_f1))]
            img = np.clip(img + _render_digit_into_slot(
                d, rng, SLOT_F1_ROWS, SLOT_F1_COLS, self.H, self.W), 0, 1)
        if label in (LABEL_FACTOR2, LABEL_BOTH):
            d = self._digits_f2[rng.randint(0, len(self._digits_f2))]
            img = np.clip(img + _render_digit_into_slot(
                d, rng, SLOT_F2_ROWS, SLOT_F2_COLS, self.H, self.W), 0, 1)
        return img

    def has_factor1(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR1) |
                (self.labels == LABEL_BOTH)).astype(np.float32)

    def has_factor2(self) -> np.ndarray:
        return ((self.labels == LABEL_FACTOR2) |
                (self.labels == LABEL_BOTH)).astype(np.float32)


# ─── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SplatteredMNISTDataset(n_normal=4, n_factor1=4, n_factor2=4, n_both=4)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for row, lbl in enumerate([0, 1, 2, 3]):
        idxs = [i for i, l in enumerate(ds.labels) if l == lbl][:4]
        for col, idx in enumerate(idxs):
            x, _ = ds[idx]
            axes[row, col].imshow((x[0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(["Normal", "F1 (digit-1)", "F2 (digit-7)", "Both"][row])
    plt.suptitle("SplatteredMNIST — 4 rows per class")
    plt.tight_layout()
    plt.savefig("splattered_mnist_preview.png", dpi=100)
    print("Saved splattered_mnist_preview.png")
