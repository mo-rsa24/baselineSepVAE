# VinBigData.py - Multi-head Salient sepVAE Dataset
# Returns triplets of images (Normal, Cardiomegaly, Pleural Effusion) for head-nulling loss training.
# Each image is a 512x512 grayscale DICOM, normalized to [-1, 1] for CheSS backbone compatibility.

import os
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


class VinBigDataTripletDataset(Dataset):
    """
    VinBigDataTripletDataset for Multi-head Salient sepVAE training.

    Returns triplets of images from three distinct pathology pools:
    - Normal (class_id: 14)
    - Cardiomegaly (class_id: 3)
    - Pleural Effusion (class_id: 10)

    Directory structure:
        /datasets/mmolefe/vinbigdata/train/  -> *.dicom files
        /datasets/mmolefe/vinbigdata/train.csv -> image_id to class_id mapping

    Each __getitem__ returns:
        {
            'x_norm': torch.Tensor (1, 512, 512),
            'x_disease1': torch.Tensor (1, 512, 512),  # Pleural Effusion
            'x_disease2': torch.Tensor (1, 512, 512),  # Cardiomegaly
            'disease_labels': torch.Tensor (3,)  # [0, 1, 2] for head-nulling gating
        }
    """

    def __init__(
        self,
        dicom_dir: str = "/datasets/mmolefe/vinbigdata/train",
        csv_path: str = "/datasets/mmolefe/vinbigdata/train.csv",
        img_size: int = 512,
        window_center: float = 40.0,
        window_width: float = 400.0,
        exclude_cross_disease_overlap: bool = False,
    ):
        """
        Args:
            dicom_dir: Path to folder containing .dicom files
            csv_path: Path to train.csv with image_id and class_id columns
            img_size: Target image size (default: 512 for CheSS backbone)
            window_center: Default window center for VOI LUT
            window_width: Default window width for VOI LUT
            exclude_cross_disease_overlap:
                If True, remove samples where cardiomegaly and pleural
                effusion co-occur from disease-specific pools. This improves
                supervision purity for disentanglement/orthogonality.
        """
        self.dicom_dir = Path(dicom_dir)
        self.img_size = int(img_size)
        self.window_center = window_center
        self.window_width = window_width
        self.exclude_cross_disease_overlap = bool(exclude_cross_disease_overlap)
        self._df = None  # kept for bbox lookup; set below

        if not self.dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.dicom_dir}")
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV and pre-calculate class pools
        df = pd.read_csv(csv_path)
        self._df = df

        # Build class pools
        self.normal_ids = self._filter_class(df, class_id=14)
        raw_cardio_ids = self._filter_class(df, class_id=3)
        raw_effusion_ids = self._filter_class(df, class_id=10)

        if self.exclude_cross_disease_overlap:
            self.cardio_ids = self._filter_class(df, class_id=3, exclude_class_id=10)
            self.effusion_ids = self._filter_class(df, class_id=10, exclude_class_id=3)
        else:
            self.cardio_ids = raw_cardio_ids
            self.effusion_ids = raw_effusion_ids

        # Validation
        if len(self.normal_ids) == 0:
            raise RuntimeError("No Normal (class_id=14) images found in CSV")
        if len(self.cardio_ids) == 0:
            raise RuntimeError("No Cardiomegaly (class_id=3) images found in CSV")
        if len(self.effusion_ids) == 0:
            raise RuntimeError("No Pleural Effusion (class_id=10) images found in CSV")

        # Build per-image bbox lookup tables (union of all radiologist annotations)
        self._cardio_bbox_lookup = self._build_bbox_lookup(df, class_id=3)
        self._effusion_bbox_lookup = self._build_bbox_lookup(df, class_id=10)

        # Pre-scan all pools and drop corrupt/blank DICOMs so __getitem__ never
        # needs to retry.  The result is cached next to the DICOM directory so
        # subsequent runs load instantly instead of re-scanning.
        cache_path = self.dicom_dir.parent / "valid_ids_cache.json"
        all_unique = list(set(self.normal_ids) | set(self.cardio_ids) | set(self.effusion_ids))
        if cache_path.exists():
            import json
            with open(cache_path) as f:
                valid = set(json.load(f))
            # Only keep IDs that are both in the cache and in our current pools.
            valid = valid & set(all_unique)
            print(f"[VinBigData] Loaded {len(valid)} valid IDs from cache ({cache_path.name}).")
        else:
            import json
            print(f"[VinBigData] Pre-scanning {len(all_unique)} unique DICOMs (first run; result will be cached)…")
            valid = {iid for iid in all_unique if self._load_dicom(iid) is not None}
            with open(cache_path, "w") as f:
                json.dump(sorted(valid), f)
            n_invalid = len(all_unique) - len(valid)
            print(f"  Scan complete. Dropped {n_invalid} corrupt/blank DICOMs. Cache saved → {cache_path}")
        self.normal_ids   = [iid for iid in self.normal_ids   if iid in valid]
        self.cardio_ids   = [iid for iid in self.cardio_ids   if iid in valid]
        self.effusion_ids = [iid for iid in self.effusion_ids if iid in valid]

        print(f"[VinBigData Dataset Initialized]")
        print(f"  Normal: {len(self.normal_ids)} images")
        print(f"  Cardiomegaly: {len(self.cardio_ids)} images ({len(self._cardio_bbox_lookup)} with bbox)")
        print(f"  Pleural Effusion: {len(self.effusion_ids)} images ({len(self._effusion_bbox_lookup)} with bbox)")
        if self.exclude_cross_disease_overlap:
            print("  Cross-disease overlap filtering: enabled")
            print(f"    Removed from Cardiomegaly pool: {len(raw_cardio_ids) - len(self.cardio_ids)}")
            print(f"    Removed from Effusion pool: {len(raw_effusion_ids) - len(self.effusion_ids)}")
        print(f"  Total triplets: {len(self)}")

    def _filter_class(
        self,
        df: pd.DataFrame,
        class_id: int,
        exclude_class_id: Optional[int] = None,
    ) -> List[str]:
        """
        Extract unique image_ids for a class, optionally excluding overlap with another class.

        Args:
            df: VinBigData dataframe with columns ['image_id', 'class_id']
            class_id: Target class ID
            exclude_class_id: Optional class ID to exclude if co-occurring
        """
        filtered = df[df['class_id'] == class_id]['image_id'].unique().tolist()
        if exclude_class_id is not None:
            excluded_ids = set(df[df['class_id'] == exclude_class_id]['image_id'].unique().tolist())
            filtered = [image_id for image_id in filtered if image_id not in excluded_ids]
        return filtered

    def _build_bbox_lookup(self, df: pd.DataFrame, class_id: int) -> dict:
        """
        Build {image_id: (x_min, y_min, x_max, y_max)} in original pixel coords.

        Takes the union of all radiologist annotations per image (min of mins /
        max of maxes), giving the most inclusive bbox for spatial supervision.
        Returns an empty dict if the CSV does not contain bbox columns.
        """
        bbox_cols = ['x_min', 'y_min', 'x_max', 'y_max']
        if not all(c in df.columns for c in bbox_cols):
            return {}

        sub = df[df['class_id'] == class_id][['image_id'] + bbox_cols].copy()
        # Drop rows where bbox is effectively absent (all-zero annotations for normal class)
        sub = sub[(sub['x_max'] > 0) & (sub['y_max'] > 0)]
        if sub.empty:
            return {}

        agg = sub.groupby('image_id').agg(
            x_min=('x_min', 'min'),
            y_min=('y_min', 'min'),
            x_max=('x_max', 'max'),
            y_max=('y_max', 'max'),
        ).reset_index()

        return {
            row['image_id']: (float(row['x_min']), float(row['y_min']),
                               float(row['x_max']), float(row['y_max']))
            for _, row in agg.iterrows()
        }

    def _load_disease_image(
        self,
        pool: List[str],
        bbox_lookup: dict,
        pool_name: str,
        initial_id: Optional[str] = None,
        max_retries: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a disease image together with its ground-truth bbox.

        Retries with random replacements on corrupt files.  The bbox is
        normalised to [0, 1] using the original DICOM pixel dimensions so it
        remains valid after the image is resized to (img_size, img_size).

        Returns:
            image:  (1, img_size, img_size) float32 tensor in [-1, 1]
            bbox:   (4,) float32 tensor [x_min, y_min, x_max, y_max] in [0, 1].
                    All-zeros when no annotation is available for this image.
        """
        candidates = [initial_id] if initial_id is not None else []

        for attempt in range(max_retries):
            if attempt < len(candidates):
                image_id = candidates[attempt]
            else:
                image_id = pool[random.randint(0, len(pool) - 1)]

            pixel_array = self._load_dicom(image_id)
            if pixel_array is None:
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {pool_name} pool (corrupt: {image_id})")
                continue

            H_orig, W_orig = pixel_array.shape
            image = self._preprocess_image(pixel_array)

            if image_id in bbox_lookup:
                x0, y0, x1, y1 = bbox_lookup[image_id]
                bbox = np.array(
                    [x0 / W_orig, y0 / H_orig, x1 / W_orig, y1 / H_orig],
                    dtype=np.float32,
                )
                bbox = np.clip(bbox, 0.0, 1.0)
            else:
                bbox = np.zeros(4, dtype=np.float32)

            return image, torch.from_numpy(bbox)

        raise RuntimeError(
            f"Too many corrupt files in {pool_name} pool ({max_retries} consecutive failures)"
        )

    def __len__(self) -> int:
        """
        Dataset length is determined by the smallest class pool to ensure
        balanced sampling across all three pathologies.
        """
        return min(len(self.normal_ids), len(self.cardio_ids), len(self.effusion_ids))

    def _load_dicom(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load a DICOM file and apply VOI LUT windowing.

        Handles two common data-quality issues in VinBigData:
          1. MONOCHROME1 photometric interpretation — intensity is inverted
             relative to the standard clinical view (bone dark, soft tissue bright).
             We correct this before windowing.
          2. Missing WindowCenter / WindowWidth metadata — falling back to CT
             defaults (wc=40, ww=400) clips all X-ray pixels (0–4095) to white.
             When DICOM window metadata is absent we use a percentile-based window
             (1st–99th percentile) so the full tonal range is preserved.
        Blank images (std < 0.02 after normalisation) are rejected so the retry
        logic in callers can substitute a valid sample.

        Args:
            image_id: Image identifier (without extension)

        Returns:
            np.ndarray or None: Windowed pixel array (H, W) in [0, 1] range,
                                or None if the file is corrupt/blank/unreadable.
        """
        dicom_path = self.dicom_dir / f"{image_id}.dicom"

        if not dicom_path.exists():
            logger.error(f"DICOM file not found: {dicom_path}")
            return None

        try:
            dcm = pydicom.dcmread(str(dicom_path), force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"CORRUPT DICOM DETECTED: {dicom_path} | Error: {e}")
            return None

        # Fix 1: Invert MONOCHROME1 images before any windowing.
        # In MONOCHROME1 the minimum stored value is displayed as white, so
        # bones appear dark — the opposite of the MONOCHROME2 convention used
        # by the rest of the dataset.  Inverting aligns all images to the same
        # visual convention and prevents the model from learning polarity as a
        # spurious feature.
        photometric = str(getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')).strip()
        if photometric == 'MONOCHROME1':
            pixel_array = pixel_array.max() - pixel_array

        # Fix 2: Windowing.
        # Use DICOM window metadata when present.  When absent, fall back to a
        # robust percentile window (1st–99th) instead of the CT-derived defaults
        # (wc=40, ww=400) which clip all X-ray pixels in the 0–4095 range to
        # a single value, producing blank white images.
        wc_meta = getattr(dcm, 'WindowCenter', None)
        ww_meta = getattr(dcm, 'WindowWidth',  None)

        if wc_meta is not None and ww_meta is not None:
            window_center = float(wc_meta[0] if isinstance(wc_meta, (list, pydicom.multival.MultiValue)) else wc_meta)
            window_width  = float(ww_meta[0] if isinstance(ww_meta, (list, pydicom.multival.MultiValue)) else ww_meta)
            lower = window_center - window_width / 2.0
            upper = window_center + window_width / 2.0
        else:
            lower = float(np.percentile(pixel_array, 1))
            upper = float(np.percentile(pixel_array, 99))

        windowed = np.clip(pixel_array, lower, upper)
        windowed = (windowed - lower) / (upper - lower + 1e-8)

        # Fix 3: Reject blank / near-uniform images.
        # A handful of DICOMs load without error but contain a uniform or near-
        # uniform pixel array (detector malfunction, empty scan, etc.).  These
        # have std ≈ 0 after normalisation and carry no diagnostic signal; worse,
        # they all look identical regardless of disease label, polluting every
        # loss that compares image representations.
        if windowed.std() < 0.02:
            logger.warning(f"Blank/uniform image rejected (std={windowed.std():.4f}): {dicom_path}")
            return None

        return windowed

    def _preprocess_image(self, pixel_array: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize image to [-1, 1] range for sepVAE.

        Args:
            pixel_array: (H, W) array in [0, 1] range

        Returns:
            torch.Tensor: (1, img_size, img_size) in [-1, 1] range
        """
        # Convert to PIL for resizing
        img = Image.fromarray((pixel_array * 255).astype(np.uint8), mode='L')
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        # Convert back to array and normalize to [-1, 1]
        x = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]
        x = x * 2.0 - 1.0  # [-1, 1]
        x = x[None, ...]  # (1, H, W)

        return torch.from_numpy(x)

    def _load_with_retry(self, pool: List[str], pool_name: str, max_retries: int = 2) -> torch.Tensor:
        """
        Load and preprocess an image from a pool, retrying with random
        replacements if corrupt files are encountered.

        Args:
            pool: List of image_ids to sample from
            pool_name: Name of the pool (for logging)
            max_retries: Maximum number of retry attempts

        Returns:
            torch.Tensor: Preprocessed image (1, img_size, img_size)
        """
        for attempt in range(max_retries):
            image_id = pool[random.randint(0, len(pool) - 1)]
            pixel_array = self._load_dicom(image_id)
            if pixel_array is not None:
                return self._preprocess_image(pixel_array)
            logger.warning(f"Retry {attempt + 1}/{max_retries} for {pool_name} pool (corrupt: {image_id})")
        raise RuntimeError(f"Too many corrupt files in {pool_name} pool ({max_retries} consecutive failures)")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a triplet of images for head-nulling loss training.

        Args:
            idx: Index (determines which Normal image to use as anchor)

        Returns:
            dict: {
                'x_norm': Normal image,
                'x_disease1': Pleural Effusion image,
                'x_disease2': Cardiomegaly image,
                'disease_labels': [0, 1, 2] tensor for head gating
            }
        """
        max_retries = 10

        # --- Normal (anchor) ---
        norm_id = self.normal_ids[idx % len(self.normal_ids)]
        pixel_array = self._load_dicom(norm_id)
        if pixel_array is not None:
            x_norm = self._preprocess_image(pixel_array)
        else:
            x_norm = self._load_with_retry(self.normal_ids, "Normal", max_retries)

        # --- Pleural Effusion ---
        effusion_id = self.effusion_ids[random.randint(0, len(self.effusion_ids) - 1)]
        x_effusion, bbox_effusion = self._load_disease_image(
            self.effusion_ids, self._effusion_bbox_lookup,
            "Effusion", initial_id=effusion_id, max_retries=max_retries,
        )

        # --- Cardiomegaly ---
        cardio_id = self.cardio_ids[random.randint(0, len(self.cardio_ids) - 1)]
        x_cardio, bbox_cardio = self._load_disease_image(
            self.cardio_ids, self._cardio_bbox_lookup,
            "Cardiomegaly", initial_id=cardio_id, max_retries=max_retries,
        )

        # Disease labels for head-nulling loss gating
        # 0: Normal, 1: Disease1 (Effusion), 2: Disease2 (Cardiomegaly)
        disease_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        return {
            'x_norm': x_norm,
            'x_disease1': x_effusion,
            'x_disease2': x_cardio,
            'disease_labels': disease_labels,
            'bbox_disease1': bbox_effusion,   # (4,) [x_min, y_min, x_max, y_max] in [0,1]
            'bbox_disease2': bbox_cardio,      # (4,) [x_min, y_min, x_max, y_max] in [0,1]
        }


def jax_collate_fn(batch_list):
    """
    JAX-compatible collate function for VinBigData triplets.

    This function stacks triplets and creates proper disease labels for training.
    Input batch contains B triplets, output contains 3*B images with labels.

    Args:
        batch_list: List of dicts from __getitem__, each with:
                   - 'x_norm': Normal image (1, 512, 512)
                   - 'x_disease1': Pleural Effusion image (1, 512, 512)
                   - 'x_disease2': Cardiomegaly image (1, 512, 512)
                   - 'disease_labels': [0, 1, 2] (unused)

    Returns:
        Dict with:
            - 'x_norm': (B, 1, 512, 512) Normal images
            - 'x_disease1': (B, 1, 512, 512) Pleural Effusion images
            - 'x_disease2': (B, 1, 512, 512) Cardiomegaly images
            - 'disease_labels': (3*B,) labels [0,0,...,1,1,...,2,2,...]
                               Where: 0=Normal, 1=Pleural Effusion, 2=Cardiomegaly

    Example:
        With batch_size=4:
        - Input: 4 triplets
        - Output: 12 images with labels [0,0,0,0, 1,1,1,1, 2,2,2,2]
    """
    B = len(batch_list)

    # Stack, convert to numpy, and transpose to NHWC so JAX JIT can trace them.
    # _preprocess_image returns (1, H, W) PyTorch-convention; JAX models expect (H, W, 1).
    def _np_nhwc(tensors):
        arr = np.stack([t.numpy() for t in tensors])  # (B, 1, H, W)
        return arr.transpose(0, 2, 3, 1)              # (B, H, W, 1)

    def _np(tensors):
        return np.stack([t.numpy() for t in tensors])

    x_norm     = _np_nhwc([b['x_norm']     for b in batch_list])  # (B, H, W, 1)
    x_disease1 = _np_nhwc([b['x_disease1'] for b in batch_list])  # Effusion
    x_disease2 = _np_nhwc([b['x_disease2'] for b in batch_list])  # Cardiomegaly

    # Labels as numpy int32
    labels = np.array(
        [0] * B + [1] * B + [2] * B,
        dtype=np.int32,
    )  # (3*B,)  0=Normal, 1=Effusion, 2=Cardiomegaly

    bbox_disease1 = _np([b['bbox_disease1'] for b in batch_list])  # (B, 4)
    bbox_disease2 = _np([b['bbox_disease2'] for b in batch_list])  # (B, 4)

    return {
        'x_norm':        x_norm,
        'x_disease1':    x_disease1,
        'x_disease2':    x_disease2,
        'disease_labels': labels,
        'bbox_disease1': bbox_disease1,  # (B, 4) effusion bboxes, [0,1] normalised
        'bbox_disease2': bbox_disease2,  # (B, 4) cardio bboxes,   [0,1] normalised
    }


class VinBigDataSingleClassDataset(Dataset):
    """
    Flat single-class dataset over VinBigData for pre-encoding.

    Returns individual images (not triplets) from one pathology class.
    Suitable for use with preencode_sepvae_latents.py.

    Args:
        dicom_dir: Path to folder containing .dicom files
        csv_path: Path to train.csv with image_id and class_id columns
        disease: Which class to load - 'cardiomegaly', 'effusion', or 'normal'
        img_size: Target image size (default: 512)
        window_center: Default window center for VOI LUT
        window_width: Default window width for VOI LUT
        exclude_cross_disease_overlap: If True, drop cardiomegaly/effusion images
            that co-occur with the other disease (improves label purity)
    """

    CLASS_IDS = {
        'normal': 14,
        'cardiomegaly': 3,
        'effusion': 10,
    }

    def __init__(
        self,
        dicom_dir: str = "/datasets/mmolefe/vinbigdata/train",
        csv_path: str = "/datasets/mmolefe/vinbigdata/train.csv",
        disease: str = 'cardiomegaly',
        img_size: int = 512,
        window_center: float = 40.0,
        window_width: float = 400.0,
        exclude_cross_disease_overlap: bool = False,
    ):
        if disease not in self.CLASS_IDS:
            raise ValueError(f"disease must be one of {list(self.CLASS_IDS)}, got '{disease}'")

        self.dicom_dir = Path(dicom_dir)
        self.img_size = int(img_size)
        self.window_center = window_center
        self.window_width = window_width
        self.disease = disease

        if not self.dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.dicom_dir}")
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        class_id = self.CLASS_IDS[disease]
        image_ids = df[df['class_id'] == class_id]['image_id'].unique().tolist()

        if exclude_cross_disease_overlap and disease in ('cardiomegaly', 'effusion'):
            other = 'effusion' if disease == 'cardiomegaly' else 'cardiomegaly'
            other_id = self.CLASS_IDS[other]
            excluded = set(df[df['class_id'] == other_id]['image_id'].unique().tolist())
            image_ids = [i for i in image_ids if i not in excluded]

        self.image_ids = image_ids
        print(f"[VinBigDataSingleClassDataset] disease={disease}, n={len(self.image_ids)}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_dicom(self, image_id: str) -> Optional[np.ndarray]:
        """Same loading/cleaning logic as VinBigDataTripletDataset._load_dicom."""
        dicom_path = self.dicom_dir / f"{image_id}.dicom"
        if not dicom_path.exists():
            logger.error(f"DICOM file not found: {dicom_path}")
            return None
        try:
            dcm = pydicom.dcmread(str(dicom_path), force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"CORRUPT DICOM DETECTED: {dicom_path} | Error: {e}")
            return None

        # Fix 1: Invert MONOCHROME1 before windowing.
        photometric = str(getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')).strip()
        if photometric == 'MONOCHROME1':
            pixel_array = pixel_array.max() - pixel_array

        # Fix 2: Prefer DICOM window metadata; fall back to percentile window.
        wc_meta = getattr(dcm, 'WindowCenter', None)
        ww_meta = getattr(dcm, 'WindowWidth',  None)
        if wc_meta is not None and ww_meta is not None:
            window_center = float(wc_meta[0] if isinstance(wc_meta, (list, pydicom.multival.MultiValue)) else wc_meta)
            window_width  = float(ww_meta[0] if isinstance(ww_meta, (list, pydicom.multival.MultiValue)) else ww_meta)
            lower = window_center - window_width / 2.0
            upper = window_center + window_width / 2.0
        else:
            lower = float(np.percentile(pixel_array, 1))
            upper = float(np.percentile(pixel_array, 99))

        windowed = np.clip(pixel_array, lower, upper)
        windowed = (windowed - lower) / (upper - lower + 1e-8)

        # Fix 3: Reject blank/uniform images.
        if windowed.std() < 0.02:
            logger.warning(f"Blank/uniform image rejected (std={windowed.std():.4f}): {dicom_path}")
            return None

        return windowed

    def _preprocess_image(self, pixel_array: np.ndarray) -> torch.Tensor:
        img = Image.fromarray((pixel_array * 255).astype(np.uint8), mode='L')
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = x * 2.0 - 1.0
        x = x[None, ...]  # (1, H, W)
        return torch.from_numpy(x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: (1, img_size, img_size) float tensor in [-1, 1]
            label: int class label (0=normal, 1=effusion, 2=cardiomegaly)
            image_id: str identifier
        """
        label_map = {'normal': 0, 'effusion': 1, 'cardiomegaly': 2}
        label = label_map[self.disease]
        image_id = self.image_ids[idx]

        pixel_array = self._load_dicom(image_id)
        if pixel_array is None:
            # On corrupt file: try a random replacement
            for _ in range(10):
                fallback_id = self.image_ids[random.randint(0, len(self.image_ids) - 1)]
                pixel_array = self._load_dicom(fallback_id)
                if pixel_array is not None:
                    image_id = fallback_id
                    break
            if pixel_array is None:
                raise RuntimeError(f"Too many consecutive corrupt DICOM files in {self.disease} pool")

        return self._preprocess_image(pixel_array), label, image_id


# Example usage and validation
if __name__ == "__main__":
    # Test dataset initialization
    dataset = VinBigDataTripletDataset()
    print(f"\nDataset length: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  x_norm shape: {sample['x_norm'].shape}, range: [{sample['x_norm'].min():.3f}, {sample['x_norm'].max():.3f}]")
    print(f"  x_disease1 shape: {sample['x_disease1'].shape}, range: [{sample['x_disease1'].min():.3f}, {sample['x_disease1'].max():.3f}]")
    print(f"  x_disease2 shape: {sample['x_disease2'].shape}, range: [{sample['x_disease2'].min():.3f}, {sample['x_disease2'].max():.3f}]")
    print(f"  disease_labels: {sample['disease_labels']}")

    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch shapes (default collation):")
    print(f"  x_norm: {batch['x_norm'].shape}")
    print(f"  x_disease1: {batch['x_disease1'].shape}")
    print(f"  x_disease2: {batch['x_disease2'].shape}")
    print(f"  disease_labels: {batch['disease_labels'].shape}")

    # Test with JAX collation
    loader_jax = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=jax_collate_fn)
    batch_jax = next(iter(loader_jax))
    print(f"\nBatch shapes (JAX collation):")
    print(f"  x_norm: {batch_jax['x_norm'].shape}")
    print(f"  x_disease1: {batch_jax['x_disease1'].shape}")
    print(f"  x_disease2: {batch_jax['x_disease2'].shape}")
    print(f"  disease_labels: {batch_jax['disease_labels'].shape}")
    print(f"  disease_labels values: {batch_jax['disease_labels']}")
