#!/usr/bin/env python3
# Example:
"""
python3 scripts/preencode_latents.py \
--data_root "/home-mscluster/mmolefe/Playground/PhD/datasets/cleaned" \
--task TB --split train --img_size 256 \
--ae_ckpt_path /home-mscluster/mmolefe/cluster_staging/unified-ae-proto-eb7c6d6_20260112-063726/runs/unified-ae-proto-increase-ae-autoencoder-eb7c6d6-20260112-063726/20260112-063740/ckpts/last.flax \
--ae_config_path /home-mscluster/mmolefe/cluster_staging/unified-ae-proto-eb7c6d6_20260112-063726/runs/unified-ae-proto-increase-ae-autoencoder-eb7c6d6-20260112-063726/20260112-063740/run_meta.json \
--latent_scale_factor 0.999373 \
--output_dir "./preencoded_latents/normal"

"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder_


class IndexedDataset(Dataset):
    def __init__(self, base: ChestXrayDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        image, label = self.base[idx]
        path = str(self.base.paths[idx])
        return image, int(label), int(idx), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-encode dataset images into VAE latents.")
    parser.add_argument("--data_root", required=True, help="Root path to dataset directory.")
    parser.add_argument("--task", default="TB", help="Dataset task (e.g., TB, PNEUMONIA).")
    parser.add_argument("--split", default="train", help="Dataset split to encode.")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size.")
    parser.add_argument("--class_filter", type=int, default=None, help="Optional class filter index.")
    parser.add_argument("--ae_ckpt_path", required=True, help="Path to AE checkpoint (last.flax).")
    parser.add_argument("--ae_config_path", required=True, help="Path to AE run_meta.json.")
    parser.add_argument("--latent_scale_factor", type=float, required=True, help="Latent scale factor.")
    parser.add_argument("--output_dir", required=True, help="Directory to write latents and manifest.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for posterior sampling.")
    parser.add_argument(
        "--manifest_name",
        default="manifest.jsonl",
        help="Manifest filename (jsonl) to write in output_dir.",
    )
    return parser.parse_args()


def _write_jsonl_line(handle, payload: dict) -> None:
    handle.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    base_ds = ChestXrayDataset(
        root_dir=args.data_root,
        task=args.task,
        split=args.split,
        img_size=args.img_size,
        class_filter=args.class_filter,
    )
    ds = IndexedDataset(base_ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ae_model, ae_params = load_autoencoder_(args.ae_config_path, args.ae_ckpt_path)
    rng = jax.random.PRNGKey(args.seed)

    meta = {
        "data_root": args.data_root,
        "task": args.task,
        "split": args.split,
        "img_size": args.img_size,
        "class_filter": args.class_filter,
        "ae_ckpt_path": args.ae_ckpt_path,
        "ae_config_path": args.ae_config_path,
        "latent_scale_factor": args.latent_scale_factor,
        "num_samples": len(ds),
        "manifest_name": args.manifest_name,
        "latents_dir": "latents",
        "seed": args.seed,
    }
    (output_dir / "latent_meta.json").write_text(json.dumps(meta, indent=2))

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for batch in tqdm(loader, desc="Encoding latents"):
            images, labels, indices, paths = batch
            x = jnp.asarray(images.numpy()).transpose(0, 2, 3, 1)
            x = (x + 1.0) / 2.0
            rng, rng_batch = jax.random.split(rng)
            posterior = ae_model.apply({"params": ae_params}, x, method=ae_model.encode, train=False)
            latents = posterior.sample(rng_batch) * args.latent_scale_factor
            latents_np = np.asarray(latents)

            for latent, label, idx, path in zip(latents_np, labels, indices, paths):
                latent_name = f"{int(idx):08d}.npy"
                latent_path = latents_dir / latent_name
                np.save(latent_path, latent.astype(np.float32))
                _write_jsonl_line(
                    manifest_handle,
                    {
                        "index": int(idx),
                        "label": int(label),
                        "latent_path": str(Path("latents") / latent_name),
                        "source_path": str(path),
                    },
                )

    print(f"✅ Saved {len(ds)} latents to {output_dir}")


if __name__ == "__main__":
    main()