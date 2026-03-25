#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.VinBigData import VinBigDataBinaryFlatDataset
from utils.baseline_ldm import load_sepvae_v2_checkpoint


def parse_args():
    parser = argparse.ArgumentParser("Pre-encode VinBigData images into paired SepVAE V2 latents.")
    parser.add_argument("--sepvae_ckpt", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--latent_mode", default="mean", choices=["mean", "sample"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--exclude_cross_disease_overlap", action="store_true")
    return parser.parse_args()


def _sample_latent(mu, logvar, rng):
    rng, sample_rng = jax.random.split(rng)
    eps = jax.random.normal(sample_rng, mu.shape, dtype=mu.dtype)
    return mu + jnp.exp(0.5 * logvar) * eps, rng


def _update_stats(stats, array):
    flat = array.astype(np.float64).reshape(-1)
    stats["count"] += int(flat.size)
    stats["sum"] += float(flat.sum())
    stats["sum_sq"] += float(np.square(flat).sum())
    stats["min"] = min(stats["min"], float(flat.min()))
    stats["max"] = max(stats["max"], float(flat.max()))


def _finalize_stats(stats):
    mean = stats["sum"] / max(stats["count"], 1)
    variance = stats["sum_sq"] / max(stats["count"], 1) - mean ** 2
    return {
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "min": stats["min"],
        "max": stats["max"],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    z_common_dir = output_dir / "z_common"
    z_cardio_dir = output_dir / "z_cardio"
    z_common_dir.mkdir(parents=True, exist_ok=True)
    z_cardio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    sepvae_model, sepvae_variables, _, ckpt_args = load_sepvae_v2_checkpoint(args.sepvae_ckpt, use_ema=True)
    img_size = int(ckpt_args.get("img_size", 256))

    dataset = VinBigDataBinaryFlatDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=img_size,
        split=args.split,
        exclude_cross_disease_overlap=args.exclude_cross_disease_overlap,
        use_cache=args.use_cache,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    rng = jax.random.PRNGKey(args.seed)
    common_stats = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}
    cardio_stats = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        global_idx = 0
        for batch in tqdm(loader, desc=f"Encoding {args.split}"):
            images = batch["image"].permute(0, 2, 3, 1).numpy().astype(np.float32)
            bbox = batch["bbox"].numpy().astype(np.float32)
            has_bbox = batch["has_bbox"].numpy().astype(np.float32)

            latents = sepvae_model.apply(
                sepvae_variables,
                jnp.asarray(images),
                bbox=jnp.asarray(bbox),
                has_bbox=jnp.asarray(has_bbox),
                method=sepvae_model.encode,
            )
            mu_common, logvar_common = latents["common"]
            mu_cardio, logvar_cardio = latents["cardiomegaly"]

            if args.latent_mode == "mean":
                z_common_batch = np.asarray(mu_common)
                z_cardio_batch = np.asarray(mu_cardio)
            else:
                z_common, rng = _sample_latent(mu_common, logvar_common, rng)
                z_cardio, rng = _sample_latent(mu_cardio, logvar_cardio, rng)
                z_common_batch = np.asarray(z_common)
                z_cardio_batch = np.asarray(z_cardio)

            for i in range(z_common_batch.shape[0]):
                common_name = f"{global_idx:08d}.npy"
                cardio_name = f"{global_idx:08d}.npy"
                np.save(z_common_dir / common_name, z_common_batch[i].astype(np.float32))
                np.save(z_cardio_dir / cardio_name, z_cardio_batch[i].astype(np.float32))

                _update_stats(common_stats, z_common_batch[i])
                _update_stats(cardio_stats, z_cardio_batch[i])

                record = {
                    "image_id": str(batch["image_id"][i]),
                    "split": str(batch["split"][i]),
                    "label": int(batch["label"][i]),
                    "z_common_path": str(Path("z_common") / common_name),
                    "z_cardio_path": str(Path("z_cardio") / cardio_name),
                    "latent_format": "NHWC",
                    "z_common_shape": list(z_common_batch[i].shape),
                    "z_cardio_shape": list(z_cardio_batch[i].shape),
                    "sepvae_ckpt": str(Path(args.sepvae_ckpt).resolve()),
                }
                manifest_file.write(json.dumps(record) + "\n")
                global_idx += 1

    meta = {
        "sepvae_ckpt": str(Path(args.sepvae_ckpt).resolve()),
        "csv_path": args.csv_path,
        "dicom_dir": args.dicom_dir,
        "split": args.split,
        "use_cache": args.use_cache,
        "latent_mode": args.latent_mode,
        "num_samples": len(dataset),
        "img_size": img_size,
        "z_common_stats": _finalize_stats(common_stats),
        "z_cardio_stats": _finalize_stats(cardio_stats),
        "sepvae_args": {
            "z_channels_common": int(ckpt_args.get("z_channels_common", 16)),
            "z_channels_disease": int(ckpt_args.get("z_channels_disease", 16)),
            "attn_query_dim": int(ckpt_args.get("attn_query_dim", 256)),
            "attn_heads": int(ckpt_args.get("attn_heads", 4)),
            "use_bbox_cross_attn": bool(ckpt_args.get("use_bbox_cross_attn", False)),
        },
    }
    (output_dir / "latent_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Paired latents saved to {output_dir}")


if __name__ == "__main__":
    main()
