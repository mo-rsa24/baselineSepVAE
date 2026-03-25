import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedLatentDataset(Dataset):
    """Dataset for paired z_common / z_cardio .npy files."""

    def __init__(self, root_dir: str, manifest_name: str = "manifest.jsonl"):
        self.root = Path(root_dir)
        manifest_path = self.root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records: List[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, idx: int) -> dict:
        return self.records[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        z_common_path = self.root / rec["z_common_path"]
        z_cardio_path = self.root / rec["z_cardio_path"]
        if not z_common_path.exists():
            raise FileNotFoundError(f"Common latent missing: {z_common_path}")
        if not z_cardio_path.exists():
            raise FileNotFoundError(f"Cardio latent missing: {z_cardio_path}")

        z_common = np.load(z_common_path).astype(np.float32)
        z_cardio = np.load(z_cardio_path).astype(np.float32)

        expected_common = tuple(rec.get("z_common_shape", z_common.shape))
        expected_cardio = tuple(rec.get("z_cardio_shape", z_cardio.shape))
        if z_common.shape != expected_common:
            raise ValueError(
                f"z_common shape mismatch for idx={idx}: got {z_common.shape}, expected {expected_common}"
            )
        if z_cardio.shape != expected_cardio:
            raise ValueError(
                f"z_cardio shape mismatch for idx={idx}: got {z_cardio.shape}, expected {expected_cardio}"
            )

        if z_common.ndim != 3 or z_cardio.ndim != 3:
            raise ValueError(
                f"Expected NHWC-free spatial tensors (H, W, C), got {z_common.shape} and {z_cardio.shape}"
            )

        label = int(rec.get("label", 0))
        return torch.from_numpy(z_cardio), torch.from_numpy(z_common), label
