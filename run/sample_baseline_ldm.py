import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from datasets.VinBigData import VinBigDataBinaryFlatDataset
from datasets.paired_latents import PairedLatentDataset
from diffusion.conditional_sampling import refine_z_cardio, sample_z_cardio
from utils.baseline_ldm import (
    decode_sepvae_latents,
    load_baseline_ldm_checkpoint,
    load_sepvae_v2_checkpoint,
    save_image_grid,
)


def parse_args():
    parser = argparse.ArgumentParser("Sample or refine with baselineLDM")
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument("--sepvae_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="sample", choices=["sample", "refine"])
    parser.add_argument("--preencoded_latents_dir", default=None)
    parser.add_argument("--preencoded_manifest", default="manifest.jsonl")
    parser.add_argument("--record_indices", default=None, help="Comma-separated indices into the latent manifest.")
    parser.add_argument("--num_records", type=int, default=8)
    parser.add_argument("--dicom_dir", default=None)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--image_ids", nargs="*", default=None)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--n_steps", type=int, default=250)
    parser.add_argument("--refine_t_start", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_indices(record_indices, num_records):
    if record_indices:
        return [int(idx) for idx in record_indices.split(",") if idx.strip()]
    return list(range(num_records))


def _load_preencoded_conditions(args):
    dataset = PairedLatentDataset(args.preencoded_latents_dir, args.preencoded_manifest)
    indices = _parse_indices(args.record_indices, args.num_records)
    selected = [dataset[idx] for idx in indices]
    records = [dataset.get_record(idx) for idx in indices]
    z_cardio = jnp.asarray(np.stack([np.asarray(item[0]) for item in selected], axis=0))
    z_common = jnp.asarray(np.stack([np.asarray(item[1]) for item in selected], axis=0))
    labels = np.asarray([int(item[2]) for item in selected], dtype=np.int32)
    image_ids = [record["image_id"] for record in records]
    return z_cardio, z_common, labels, image_ids


def _encode_image_ids(args, sepvae_model, sepvae_variables, sepvae_args):
    dataset = VinBigDataBinaryFlatDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=int(sepvae_args.get("img_size", 256)),
        split="all",
        use_cache=args.use_cache,
    )
    lookup = {record["image_id"]: idx for idx, record in enumerate(dataset.records)}
    missing = [image_id for image_id in args.image_ids if image_id not in lookup]
    if missing:
        raise ValueError(f"Image IDs not found in dataset: {missing}")

    samples = [dataset[lookup[image_id]] for image_id in args.image_ids]
    images = np.stack([sample["image"].numpy().transpose(1, 2, 0) for sample in samples], axis=0).astype(np.float32)
    bbox = np.stack([sample["bbox"].numpy() for sample in samples], axis=0).astype(np.float32)
    has_bbox = np.stack([sample["has_bbox"].numpy() for sample in samples], axis=0).astype(np.float32)
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int32)

    latents = sepvae_model.apply(
        sepvae_variables,
        jnp.asarray(images),
        bbox=jnp.asarray(bbox),
        has_bbox=jnp.asarray(has_bbox),
        method=sepvae_model.encode,
    )
    z_common = jnp.asarray(latents["common"][0])
    z_cardio = jnp.asarray(latents["cardiomegaly"][0])
    return images, z_cardio, z_common, labels, list(args.image_ids)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sepvae_model, sepvae_variables, _, sepvae_args = load_sepvae_v2_checkpoint(args.sepvae_ckpt, use_ema=True)
    ldm_model, ldm_params, ldm_ckpt = load_baseline_ldm_checkpoint(args.ldm_ckpt, use_ema=True)
    ldm_args = ldm_ckpt["args"]

    reference_images = None
    if args.preencoded_latents_dir:
        z_cardio, z_common, labels, image_ids = _load_preencoded_conditions(args)
    else:
        if not args.image_ids or not args.dicom_dir or not args.csv_path:
            raise ValueError("Provide either --preencoded_latents_dir or direct image inputs (--dicom_dir, --csv_path, --image_ids).")
        reference_images, z_cardio, z_common, labels, image_ids = _encode_image_ids(
            args, sepvae_model, sepvae_variables, sepvae_args
        )
        save_image_grid((reference_images + 1.0) / 2.0, output_dir / "input_images.png")

    rng = jax.random.PRNGKey(args.seed)
    z_common = z_common.astype(jnp.float32)
    z_cardio = z_cardio.astype(jnp.float32)

    if args.mode == "sample":
        generated_cardio, _ = sample_z_cardio(
            rng,
            ldm_model,
            ldm_params,
            z_common.astype(jnp.float32),
            n_steps=args.n_steps,
        )
    else:
        generated_cardio, _ = refine_z_cardio(
            rng,
            ldm_model,
            ldm_params,
            z_common.astype(jnp.float32),
            z_cardio.astype(jnp.float32),
            t_start=args.refine_t_start,
            n_steps=args.n_steps,
        )

    decoded_generated = decode_sepvae_latents(
        sepvae_model,
        sepvae_variables,
        z_common.astype(jnp.float32),
        generated_cardio.astype(jnp.float32),
    )
    decoded_reference = decode_sepvae_latents(
        sepvae_model,
        sepvae_variables,
        z_common.astype(jnp.float32),
        z_cardio.astype(jnp.float32),
    )

    save_image_grid(np.asarray(decoded_reference), output_dir / "reference_decode.png")
    save_image_grid(np.asarray(decoded_generated), output_dir / f"{args.mode}_decode.png")

    meta = {
        "mode": args.mode,
        "image_ids": image_ids,
        "labels": labels.tolist(),
        "ldm_ckpt": str(Path(args.ldm_ckpt).resolve()),
        "sepvae_ckpt": str(Path(args.sepvae_ckpt).resolve()),
        "ldm_args": dict(ldm_args),
    }
    (output_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
