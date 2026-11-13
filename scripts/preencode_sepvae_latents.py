#!/usr/bin/env python3
"""
Pre-encode VinBigData images into SepVAE disease-head latents.

Loads a trained SepVAE checkpoint, encodes either cardiomegaly or pleural
effusion images through the appropriate disease head, and saves the resulting
2-channel spatial latents as .npy files with a manifest.jsonl index.

The output directory can then be fed directly into the LDM trainer via
--preencoded_latents_dir.

Resume support
--------------
If the job is interrupted, resubmit with --resume. The script reads the
existing manifest.jsonl, skips already-encoded image_ids, and appends new
records without re-scaling already-saved latents.

Usage
-----
# Cardiomegaly (disentangle ep180 checkpoint)
python scripts/preencode_sepvae_latents.py \
  --sepvae_ckpt runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl \
  --disease cardiomegaly \
  --output_dir preencoded_latents/disentangle_cardio

# Pleural Effusion (disentangle ep180 checkpoint)
python scripts/preencode_sepvae_latents.py \
  --sepvae_ckpt runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl \
  --disease effusion \
  --output_dir preencoded_latents/disentangle_effusion

# Resume a partial run
python scripts/preencode_sepvae_latents.py \
  --sepvae_ckpt ... --disease cardiomegaly --output_dir preencoded_latents/disentangle_cardio \
  --resume
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.serialization import msgpack_restore
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.VinBigData import VinBigDataSingleClassDataset
from models.sep_vae_jax import SepVAE


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-encode VinBigData images into SepVAE disease-head latents."
    )
    p.add_argument(
        "--sepvae_ckpt", required=True,
        help="Path to SepVAE checkpoint .pkl (e.g. checkpoint_final.pkl).",
    )
    p.add_argument(
        "--disease", required=True, choices=["cardiomegaly", "effusion"],
        help="Which disease head to extract latents from.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to write latents and manifest.jsonl.",
    )
    p.add_argument(
        "--dicom_dir", default="/datasets/mmolefe/vinbigdata/train",
        help="Path to VinBigData DICOM files.",
    )
    p.add_argument(
        "--csv_path", default="/datasets/mmolefe/vinbigdata/train.csv",
        help="Path to VinBigData train.csv.",
    )
    p.add_argument(
        "--img_size", type=int, default=512,
        help="Image size expected by the SepVAE (default: 512).",
    )
    p.add_argument(
        "--exclude_cross_disease_overlap", action="store_true",
        help="Exclude images where cardiomegaly and effusion co-occur.",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for encoding (default: 8).",
    )
    p.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes (default: 4).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for posterior sampling (default: 42).",
    )
    p.add_argument(
        "--manifest_name", default="manifest.jsonl",
        help="Manifest filename (default: manifest.jsonl).",
    )
    p.add_argument(
        "--use_mean", action="store_true",
        help="Store the posterior mean instead of a reparameterised sample. "
             "Produces deterministic latents but may underestimate variance.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume a partial run. Reads existing manifest.jsonl, skips "
             "already-encoded image_ids, and appends new records. "
             "The scale-factor re-scaling pass is skipped for already-saved latents.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# SepVAE loading
# ---------------------------------------------------------------------------

def load_sepvae(ckpt_path: str):
    """
    Load SepVAE model and parameters from a .pkl checkpoint.

    Returns
    -------
    model : SepVAE
        Flax module (architecture reconstructed from saved args).
    variables : dict
        {'params': ..., 'batch_stats': ...} ready for model.apply().
    ckpt_args : dict
        Original training args saved inside the checkpoint.
    """
    print(f"Loading SepVAE checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = msgpack_restore(f.read())

    ckpt_args = ckpt.get("args", {})
    print(f"  epoch={ckpt.get('epoch', '?')}  global_step={ckpt.get('global_step', '?')}")
    print(f"  z_channels_common={ckpt_args.get('z_channels_common', 4)}, "
          f"z_channels_disease={ckpt_args.get('z_channels_disease', 2)}, "
          f"use_fpn={ckpt_args.get('use_fpn', False)}")

    model = SepVAE(
        z_channels_common=ckpt_args.get("z_channels_common", 4),
        z_channels_disease=ckpt_args.get("z_channels_disease", 2),
        frozen_backbone=True,           # always True for inference
        use_fpn=ckpt_args.get("use_fpn", False),
        fpn_channels=ckpt_args.get("fpn_channels", 512),
        unfreeze_from=ckpt_args.get("unfreeze_from", "layer4"),
        sigma_inactive=ckpt_args.get("sigma_inactive", 0.1),
        dtype=jnp.float32,              # float32 for inference
        use_remat=False,                # no remat needed at inference
        upsample_method=ckpt_args.get("upsample_method", "bilinear"),
    )

    vae_params = jax.tree_util.tree_map(jnp.array, ckpt["vae_params"])
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, ckpt["vae_batch_stats"])

    variables = {"params": vae_params}
    if vae_batch_stats:
        variables["batch_stats"] = vae_batch_stats

    print("SepVAE loaded.")
    return model, variables, ckpt_args


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_batch(model, variables, images_np: np.ndarray, rng, use_mean: bool):
    """
    Encode a numpy batch and return a dict of latents per head.

    Parameters
    ----------
    images_np : (B, H, W, 1)  float32, range [-1, 1]

    Returns
    -------
    latents_dict : dict  {head_name: z_array}  each (B, 64, 64, ch)
    """
    x = jnp.asarray(images_np)
    latents_dist = model.apply(variables, x, method=model.encode)

    result = {}
    for head_name, (mu, logvar) in latents_dist.items():
        if use_mean:
            z = mu
        else:
            rng, rng_sample = jax.random.split(rng)
            eps = jax.random.normal(rng_sample, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
        result[head_name] = np.asarray(z)

    return result, rng


# ---------------------------------------------------------------------------
# Scale-factor computation
# ---------------------------------------------------------------------------

def compute_scale_factor(all_latents: np.ndarray) -> float:
    """
    Compute latent_scale_factor = 1 / std(latents).

    This normalises stored latents so that the LDM sees unit-variance inputs.
    """
    std = float(np.std(all_latents))
    if std < 1e-8:
        raise RuntimeError(
            f"Latent std is near zero ({std:.2e}). "
            "Check that the SepVAE checkpoint is correct and images are loading properly."
        )
    return 1.0 / std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    # ── Dataset ────────────────────────────────────────────────────────────
    ds = VinBigDataSingleClassDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        disease=args.disease,
        img_size=args.img_size,
        exclude_cross_disease_overlap=args.exclude_cross_disease_overlap,
    )

    # ── Resume: read already-done image_ids from existing manifest ─────────
    done_image_ids: set[str] = set()
    next_global_idx: int = 0
    manifest_mode = "w"

    if args.resume and manifest_path.exists():
        manifest_mode = "a"
        existing_records: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_image_ids.add(rec["image_id"])
                    existing_records.append(rec)
        next_global_idx = len(existing_records)
        print(f"[resume] {len(done_image_ids)} images already encoded. "
              f"Continuing from index {next_global_idx}.")
        # Filter dataset to only remaining image_ids
        ds.image_ids = [iid for iid in ds.image_ids if iid not in done_image_ids]
        if not ds.image_ids:
            print("[resume] Nothing left to encode — all images already done.")
            return
        print(f"[resume] {len(ds.image_ids)} images remaining.")
    elif args.resume:
        print("[resume] No existing manifest found — starting fresh.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"Dataset size: {len(ds)} images  ({len(loader)} batches @ bs={args.batch_size})")

    # ── Model ──────────────────────────────────────────────────────────────
    model, variables, ckpt_args = load_sepvae(args.sepvae_ckpt)
    rng = jax.random.PRNGKey(args.seed)

    # ── Encode ─────────────────────────────────────────────────────────────
    # First pass: encode all images and accumulate disease-head latents for
    # scale-factor computation.
    all_disease_latents: list[np.ndarray] = []
    records: list[dict] = []   # manifest entries (without scale factor yet)

    global_idx = next_global_idx
    with manifest_path.open(manifest_mode, encoding="utf-8") as mf:
        for batch in tqdm(loader, desc=f"Encoding {args.disease}"):
            images, labels, image_ids = batch

            # DataLoader returns (1, H, W) per image; convert to (B, H, W, 1)
            images_np = images.numpy().transpose(0, 2, 3, 1).astype(np.float32)

            latents_dict, rng = encode_batch(model, variables, images_np, rng, args.use_mean)
            disease_z = latents_dict[args.disease]   # (B, 64, 64, 2)

            for i in range(disease_z.shape[0]):
                latent_name = f"{global_idx:08d}.npy"
                latent_path = latents_dir / latent_name
                np.save(latent_path, disease_z[i].astype(np.float32))

                record = {
                    "index": global_idx,
                    "label": int(labels[i]),
                    "image_id": str(image_ids[i]),
                    "latent_path": str(Path("latents") / latent_name),
                    "disease": args.disease,
                }
                mf.write(json.dumps(record) + "\n")
                records.append(record)
                all_disease_latents.append(disease_z[i])
                global_idx += 1

    print(f"\nSaved {global_idx} raw latents to {latents_dir}/")

    # ── Scale factor ───────────────────────────────────────────────────────
    all_latents_np = np.stack(all_disease_latents, axis=0)   # (N, 64, 64, 2)
    scale_factor = compute_scale_factor(all_latents_np)

    latent_std  = float(np.std(all_latents_np))
    latent_mean = float(np.mean(all_latents_np))
    latent_min  = float(np.min(all_latents_np))
    latent_max  = float(np.max(all_latents_np))

    print(f"\nLatent statistics (before scaling):")
    print(f"  mean={latent_mean:.4f}  std={latent_std:.4f}  "
          f"min={latent_min:.4f}  max={latent_max:.4f}")
    print(f"\nlatent_scale_factor = {scale_factor:.8f}  "
          f"(pass this to --latent_scale_factor when launching LDM training)")

    # ── Re-scale stored latents in-place ──────────────────────────────────
    print("Re-scaling saved latents by scale_factor...")
    for record in tqdm(records, desc="Scaling"):
        latent_path = output_dir / record["latent_path"]
        z = np.load(latent_path)
        np.save(latent_path, (z * scale_factor).astype(np.float32))

    # ── Meta ───────────────────────────────────────────────────────────────
    meta = {
        "sepvae_ckpt": str(args.sepvae_ckpt),
        "disease": args.disease,
        "dicom_dir": args.dicom_dir,
        "csv_path": args.csv_path,
        "img_size": args.img_size,
        "exclude_cross_disease_overlap": args.exclude_cross_disease_overlap,
        "use_mean": args.use_mean,
        "seed": args.seed,
        "num_samples": global_idx,
        "latent_shape": list(all_disease_latents[0].shape),   # [64, 64, 2]
        "latent_scale_factor": scale_factor,
        "latent_stats_before_scaling": {
            "mean": latent_mean,
            "std": latent_std,
            "min": latent_min,
            "max": latent_max,
        },
        "sepvae_args": {
            "z_channels_common": ckpt_args.get("z_channels_common", 4),
            "z_channels_disease": ckpt_args.get("z_channels_disease", 2),
            "use_fpn": ckpt_args.get("use_fpn", False),
        },
        "manifest_name": args.manifest_name,
        "latents_dir": "latents",
    }
    meta_path = output_dir / "latent_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*60}")
    print(f"Done!  {global_idx} scaled latents saved to: {output_dir}")
    print(f"  manifest : {manifest_path}")
    print(f"  meta     : {meta_path}")
    print(f"\nTo launch LDM training, use:")
    print(f"  --preencoded_latents_dir {output_dir}")
    print(f"  --latent_scale_factor    {scale_factor:.8f}")
    print(f"  --ldm_z_channels         2")
    print(f"  --latent_size            64")
    print(f"  --vae_z_channels         2")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
