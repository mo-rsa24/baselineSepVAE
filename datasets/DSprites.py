"""
dSprites — Domain C: Non-spatial, Independent
==============================================
Scale (Factor 1) and Orientation (Factor 2) from the canonical dSprites benchmark.
Used in β-VAE, FactorVAE, Locatello et al. (ICML 2019/2020).

    Factor 1: large scale  (scale_class in {3,4,5}: sizes 0.75, 0.875, 1.0)
    Factor 2: high orientation (orient_class in {20..39}: angles π → 2π)

Both factors affect the full spatial support of the shape — the spatial gate
is uninformative.  Only the MI discriminator can separate scale from orientation.

This tests A2 violation: the d0–d5 framework's spatial gate is inert
(both factors share the same pixels), so disentanglement at d5 must come
entirely from the MI independence loss (D2), proving spatial prior is
sufficient but not necessary.

Why better than ColouredMNIST:
    - Scale and orientation are canonical disentanglement benchmark factors
      (cited in every major disentanglement paper since 2017)
    - Factors are not trivially channel-separable (any linear probe cannot
      split scale from orientation without learning shape geometry)
    - Binary images remove colour confounds — pure geometry
    - Factor interaction: large + high-orientation shapes have a qualitatively
      different appearance than just adding R vs B channel boosts

Labels
------
    0 — Normal  : small scale + low orientation  (scale 0–2, orient 0–19)
    1 — Factor1 : large scale + low orientation  (scale 3–5, orient 0–19)
    2 — Factor2 : small scale + high orientation (scale 0–2, orient 20–39)
    3 — Both    : large scale + high orientation (scale 3–5, orient 20–39)

BBOX_F1 = BBOX_F2 = (0.0, 0.0, 0.0, 0.0)  — sentinel: no spatial prior.

Output: (1, 64, 64) greyscale in [-1, 1].

Requirements: numpy, torch, urllib (auto-download, ~2.5 MB)
"""

import numpy as np
import torch
import urllib.request
from pathlib import Path
from torch.utils.data import Dataset

# ─── Download constants ────────────────────────────────────────────────────────

DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

# ─── Latent factor column indices in latents_classes ──────────────────────────
# latents order: [color(1), shape(3), scale(6), orientation(40), posX(32), posY(32)]
COL_SCALE  = 2
COL_ORIENT = 3
COL_POSX   = 4
COL_POSY   = 5

# Factor 1: large scale (classes 3,4,5 → actual sizes 0.75, 0.875, 1.0)
# Factor 2: high orientation (classes 20–39 → angles π to 2π)
SCALE_LARGE_MIN  = 3    # scale_class >= 3
ORIENT_HIGH_MIN  = 20   # orient_class >= 20

# Position window: restrict to a 7×7 centred block (classes 13–19 out of 0–31).
# Why: dSprites constrains large shapes to near-centre positions (they'd fall
# outside the image otherwise), creating a spurious scale → position correlation
# if we allow all 32×32 positions.  Fixing the same window for all scale levels
# breaks that confound while keeping enough positional jitter to avoid trivial
# solutions.  Pool size per label: 3 shapes × 3 scales × 20 orients × 49 pos = 8820.
POS_MIN = 13   # inclusive
POS_MAX = 19   # inclusive

# No spatial prior — same sentinel as ColouredMNIST
BBOX_F1 = (0.0, 0.0, 0.0, 0.0)
BBOX_F2 = (0.0, 0.0, 0.0, 0.0)

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1
LABEL_FACTOR2 = 2
LABEL_BOTH    = 3


# ─── Download / load ──────────────────────────────────────────────────────────

def _download_dsprites(cache_dir: str) -> Path:
    """Download dSprites NPZ (~2.5 MB) if not cached. Returns local path."""
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / DSPRITES_FILENAME
    if not npz_path.exists():
        print(f"  Downloading dSprites (~2.5 MB) → {npz_path}")
        urllib.request.urlretrieve(DSPRITES_URL, npz_path)
        print("  Download complete.")
    return npz_path


def _load_dsprites(cache_dir: str):
    """Load dSprites NPZ.  Returns (imgs, latents_classes)."""
    npz_path = _download_dsprites(cache_dir)
    data = np.load(npz_path, allow_pickle=True, encoding="bytes")
    imgs            = data["imgs"]             # (737280, 64, 64) uint8 {0,1}
    latents_classes = data["latents_classes"]  # (737280, 6) int64
    return imgs, latents_classes


def _build_label_pools(imgs: np.ndarray,
                       latents_classes: np.ndarray) -> dict:
    """
    Return dict label → np.ndarray of images for all four label classes.

    Position is restricted to a centred 7×7 window (POS_MIN..POS_MAX) to break
    the inherent scale-position confound in dSprites (large shapes are constrained
    to near-centre positions by the dataset's boundary-clipping rule).
    """
    sc    = latents_classes[:, COL_SCALE]
    oc    = latents_classes[:, COL_ORIENT]
    px    = latents_classes[:, COL_POSX]
    py    = latents_classes[:, COL_POSY]

    centred = (px >= POS_MIN) & (px <= POS_MAX) & (py >= POS_MIN) & (py <= POS_MAX)
    large   = sc >= SCALE_LARGE_MIN
    high    = oc >= ORIENT_HIGH_MIN

    pools = {
        LABEL_NORMAL:  imgs[ centred & ~large & ~high],   # small + low-θ
        LABEL_FACTOR1: imgs[ centred &  large & ~high],   # large + low-θ
        LABEL_FACTOR2: imgs[ centred & ~large &  high],   # small + high-θ
        LABEL_BOTH:    imgs[ centred &  large &  high],   # large + high-θ
    }
    return pools


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DSpritesDataset(Dataset):
    """
    Domain C — Non-spatial, independent.

    Scale and orientation are disentanglement targets.  No spatial bbox.
    The spatial gate in SepVAE is inert; only MI loss drives disentanglement.

    Parameters
    ----------
    dsprites_root : str
        Directory for dSprites download/cache.
    n_normal, n_factor1, n_factor2, n_both : int
        Samples per class.  Set n_both=0 for training split.
    img_size : int
        Output resolution (dSprites is natively 64×64).
    global_seed : int
    mode : str — 'flat' or 'triplet'
    """

    def __init__(
        self,
        dsprites_root: str  = "~/.cache/dsprites",
        n_normal:      int  = 1500,
        n_factor1:     int  = 750,
        n_factor2:     int  = 750,
        n_both:        int  = 0,
        img_size:      int  = 64,
        global_seed:   int  = 42,
        mode:          str  = 'flat',
    ):
        self.H = self.W   = img_size
        self.mode         = mode
        self._img_size    = img_size

        imgs, latents_classes = _load_dsprites(str(Path(dsprites_root).expanduser()))
        pools = _build_label_pools(imgs, latents_classes)

        rng = np.random.RandomState(global_seed)
        self._items: list  = []   # list of (img_np uint8, label)
        self._both_pool: list = []  # used by correlated subclass

        for label, count in [
            (LABEL_NORMAL,  n_normal),
            (LABEL_FACTOR1, n_factor1),
            (LABEL_FACTOR2, n_factor2),
            (LABEL_BOTH,    n_both),
        ]:
            pool = pools[label]
            chosen_idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
            for i in chosen_idx:
                self._items.append((pool[i], label))

        # Build a separate "both" corruption pool for subclasses
        both_pool = pools[LABEL_BOTH]
        corr_idx  = rng.choice(len(both_pool),
                               size=min(max(n_factor1, n_factor2) * 2, len(both_pool)),
                               replace=False)
        self._both_pool = [both_pool[i] for i in corr_idx]

        self.labels      = np.array([item[1] for item in self._items], dtype=np.int64)
        self._bbox_zero  = torch.zeros(4, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        img_np, label = self._items[idx]
        x = self._to_tensor(img_np)
        y = torch.tensor(label, dtype=torch.long)
        if self.mode == 'triplet':
            return {"x": x, "y": y,
                    "bbox_factor1": self._bbox_zero.clone(),
                    "bbox_factor2": self._bbox_zero.clone()}
        return x, y

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """(64,64) uint8 {0,1} → (1, H, W) float32 in [-1, 1]."""
        arr = img_np.astype(np.float32)
        if self._img_size != 64:
            from PIL import Image as PILImage
            arr = np.array(
                PILImage.fromarray((arr * 255).astype(np.uint8)).resize(
                    (self._img_size, self._img_size), PILImage.NEAREST),
                dtype=np.float32) / 255.0
        return torch.from_numpy(arr[None]).float() * 2.0 - 1.0   # (1, H, W)

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

    ds = DSpritesDataset(n_normal=4, n_factor1=4, n_factor2=4, n_both=4)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    names = ["Normal (small, low-θ)", "F1 (large, low-θ)",
             "F2 (small, high-θ)", "Both (large, high-θ)"]
    for row, lbl in enumerate([0, 1, 2, 3]):
        idxs = [i for i, item in enumerate(ds._items) if item[1] == lbl][:4]
        for col, idx in enumerate(idxs):
            x, _ = ds[idx]
            axes[row, col].imshow((x[0].numpy() + 1) / 2,
                                  cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(names[row], fontsize=8)
    plt.suptitle("dSprites — Domain C: Scale × Orientation")
    plt.tight_layout()
    plt.savefig("dsprites_preview.png", dpi=100)
    print("Saved dsprites_preview.png")
    print(f"Pool sizes  —  N:{sum(ds.labels==0)}  F1:{sum(ds.labels==1)}"
          f"  F2:{sum(ds.labels==2)}  Both:{sum(ds.labels==3)}")
