"""
SyntheticCXR — fast 128×128 synthetic chest X-ray triplet dataset.

Purpose: provide a zero-I/O, ~5-second-per-epoch training signal for rapid
ablation of SepVAE loss terms. The synthetic images mimic the key spatial
structure of chest X-rays:

    Normal:        Gaussian texture + blank lung fields
    Cardiomegaly:  Bright ellipse in the central cardiac region (enlarged heart)
    Effusion:      Bright wedge in the lower-right pleural angle

Bounding boxes are deterministic per disease, with minor per-sample jitter,
and are returned in the same format as VinBigDataTripletDataset so that
train_mini_sepvae.py can switch datasets with a single flag.

Interface contract (identical to VinBigDataTripletDataset):
    __getitem__ returns:
        {
            'x_norm':       FloatTensor (1, 128, 128)   range [-1, 1]
            'x_disease1':   FloatTensor (1, 128, 128)   (effusion)
            'x_disease2':   FloatTensor (1, 128, 128)   (cardiomegaly)
            'disease_labels': LongTensor (3,)            [0, 1, 2]
            'bbox_disease1': FloatTensor (4,)            effusion bbox in [0,1]
            'bbox_disease2': FloatTensor (4,)            cardio  bbox in [0,1]
        }

    jax_collate_fn is imported from VinBigData and works without modification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================
# Image generators
# ============================================================================

def _gaussian_texture(H: int, W: int, rng: np.random.Generator,
                       amplitude: float = 0.15) -> np.ndarray:
    """Low-amplitude Gaussian noise field in [-amplitude, amplitude]."""
    return rng.standard_normal((H, W)).astype(np.float32) * amplitude


def _make_normal(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """
    Normal CXR: lung-field structure as two dark ellipses on a mid-grey
    background, plus subtle noise. Returns image in [-1, 1].
    """
    img = np.full((H, W), -0.3, dtype=np.float32)   # mid-dark background
    img += _gaussian_texture(H, W, rng, amplitude=0.08)
    # Two lung fields: left and right
    for cx_frac, cy_frac, rx_frac, ry_frac in [
        (0.32, 0.45, 0.18, 0.28),   # right lung
        (0.68, 0.45, 0.18, 0.28),   # left lung
    ]:
        cx, cy = int(cx_frac * W), int(cy_frac * H)
        rx, ry = int(rx_frac * W), int(ry_frac * H)
        ys, xs = np.ogrid[:H, :W]
        mask = ((xs - cx)**2 / rx**2 + (ys - cy)**2 / ry**2) < 1.0
        img[mask] = -0.7 + rng.uniform(-0.05, 0.05)
    img = np.clip(img, -1.0, 1.0)
    return img


def _make_cardiomegaly(H: int, W: int, rng: np.random.Generator) -> tuple:
    """
    Cardiomegaly: enlarged bright ellipse in the central cardiac region.
    The ellipse radius is jittered ±15% per sample.

    Returns (image, bbox_xyxy_norm) where bbox is [x_min, y_min, x_max, y_max]
    normalised to [0, 1] in (x→W, y→H) convention.
    """
    img = _make_normal(H, W, rng)

    # Cardiac region: ~centre of image, slightly left of midline
    cx_frac = rng.uniform(0.44, 0.54)
    cy_frac = rng.uniform(0.38, 0.50)
    rx_frac = rng.uniform(0.16, 0.22)   # enlarged: wider than normal ~0.12
    ry_frac = rng.uniform(0.20, 0.28)

    cx, cy = int(cx_frac * W), int(cy_frac * H)
    rx, ry = int(rx_frac * W), int(ry_frac * H)
    ys, xs = np.ogrid[:H, :W]
    mask = ((xs - cx)**2 / rx**2 + (ys - cy)**2 / ry**2) < 1.0
    img[mask] = np.clip(0.6 + rng.uniform(-0.1, 0.1), 0.3, 1.0)

    # Bounding box (tight around the ellipse)
    x_min = max(cx - rx, 0) / W
    y_min = max(cy - ry, 0) / H
    x_max = min(cx + rx, W) / W
    y_max = min(cy + ry, H) / H
    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    img = np.clip(img, -1.0, 1.0)
    return img, bbox


def _make_effusion(H: int, W: int, rng: np.random.Generator) -> tuple:
    """
    Effusion: bright wedge / opacity in the lower-right costophrenic angle.
    Height jittered ±20% per sample.

    Returns (image, bbox_xyxy_norm).
    """
    img = _make_normal(H, W, rng)

    # Lower right quadrant
    x0_frac = rng.uniform(0.55, 0.65)
    y0_frac = rng.uniform(0.62, 0.70)
    x1_frac = rng.uniform(0.88, 0.98)
    y1_frac = rng.uniform(0.88, 0.98)

    x0, y0 = int(x0_frac * W), int(y0_frac * H)
    x1, y1 = int(x1_frac * W), int(y1_frac * H)
    img[y0:y1, x0:x1] = np.clip(0.5 + rng.uniform(-0.1, 0.1), 0.2, 1.0)

    bbox = np.array([x0_frac, y0_frac, x1_frac, y1_frac], dtype=np.float32)
    img = np.clip(img, -1.0, 1.0)
    return img, bbox


# ============================================================================
# Dataset class
# ============================================================================

class SyntheticCXRTripletDataset(Dataset):
    """
    Fast synthetic triplet dataset for MiniSepVAE ablation experiments.

    Each call to __getitem__ generates a fresh triplet (normal, effusion,
    cardiomegaly) on-the-fly using NumPy — no disk I/O, no DICOM loading.

    Args:
        n_samples: Number of (virtual) samples. Each __getitem__ call uses
                   a deterministic seed derived from the index, so the dataset
                   is reproducible across epochs.
        img_size:  Image resolution (default 128). Used for both H and W.
        seed:      Base random seed.
    """
    def __init__(self, n_samples: int = 2000, img_size: int = 128, seed: int = 42):
        self.n_samples = n_samples
        self.img_size  = img_size
        self.seed      = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        H = W = self.img_size
        rng = np.random.default_rng(self.seed + idx)

        x_norm,            _          = _make_normal(H, W, rng), None
        x_norm             = _make_normal(H, W, rng)
        x_cardio, bbox_cardio          = _make_cardiomegaly(H, W, np.random.default_rng(self.seed + idx + 100_000))
        x_effusion, bbox_effusion      = _make_effusion(H, W, np.random.default_rng(self.seed + idx + 200_000))

        def to_tensor(arr):
            return torch.from_numpy(arr[None, :, :].copy())  # (1, H, W)

        return {
            'x_norm':         to_tensor(x_norm),
            'x_disease1':     to_tensor(x_effusion),    # effusion
            'x_disease2':     to_tensor(x_cardio),      # cardiomegaly
            'disease_labels': torch.tensor([0, 1, 2], dtype=torch.long),
            'bbox_disease1':  torch.from_numpy(bbox_effusion),   # (4,)
            'bbox_disease2':  torch.from_numpy(bbox_cardio),     # (4,)
        }


# ============================================================================
# Re-export jax_collate_fn from VinBigData — interface is identical
# ============================================================================
from datasets.VinBigData import jax_collate_fn  # noqa: F401


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    ds = SyntheticCXRTripletDataset(n_samples=8, img_size=128)
    loader = DataLoader(ds, batch_size=4, collate_fn=jax_collate_fn)
    batch = next(iter(loader))

    print("Keys:", list(batch.keys()))
    print("x_norm shape:",    batch['x_norm'].shape)
    print("x_disease1 shape:", batch['x_disease1'].shape)
    print("disease_labels:",   batch['disease_labels'])
    print("bbox_disease1:",    batch['bbox_disease1'])
    print("bbox_disease2:",    batch['bbox_disease2'])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, key, title in zip(axes,
                               ['x_norm', 'x_disease1', 'x_disease2'],
                               ['Normal', 'Effusion', 'Cardiomegaly']):
        img = np.array(batch[key][0, :, :, 0])   # first sample, JAX → (H,W)
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/synthetic_cxr_check.png')
    print("Saved /tmp/synthetic_cxr_check.png")
