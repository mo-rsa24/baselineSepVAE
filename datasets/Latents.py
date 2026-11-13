import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PreencodedLatentDataset(Dataset):
    """Dataset for pre-encoded latent .npy files produced by scripts/preencode_latents.py."""

    def __init__(self, root_dir: str, manifest_name: str = "manifest.jsonl"):
        root = Path(root_dir)
        manifest_path = root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records: List[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        if not records:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

        self.root = root
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        latent_path = self.root / rec["latent_path"]
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file missing: {latent_path}")
        latents = np.load(latent_path)
        label = int(rec.get("label", 0))
        return torch.from_numpy(latents), label