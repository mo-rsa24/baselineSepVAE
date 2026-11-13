import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import torch
import tensorflow as tf

from run.ldm import load_autoencoder

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ChestXRay import ChestXrayDataset


def main():
    parser = argparse.ArgumentParser(description="Find the correct latent scale factor for a single image.")
    parser.add_argument("--ae_config_path", type=str, required=True, help="Path to the AE run_meta.json config.")
    parser.add_argument("--ae_ckpt_path", type=str, required=True, help="Path to the AE checkpoint (.flax).")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--task", type=str, required=True, help="Dataset task (e.g., TB).")
    parser.add_argument("--img_size", type=int, default=256, help="Image size.")
    parser.add_argument("--class_filter", type=int, default=None, help="Class to filter for.")

    args = parser.parse_args()

    # Prevent TF from grabbing GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    print("--- Loading Model and Dataset ---")
    # ae_model, ae_params, _ = load_autoencoder(Path(args.ae_config_path), Path(args.ae_ckpt_path))
    ae_model, ae_params = load_autoencoder(args.ae_config_path, args.ae_ckpt_path)

    ds = ChestXrayDataset(
        root_dir=args.data_root,
        task=args.task,
        split="train",
        img_size=args.img_size,
        class_filter=args.class_filter
    )

    if len(ds) == 0:
        print("Error: Dataset is empty. Check data_root, task, and class_filter.")
        return

    print(f"Dataset loaded with {len(ds)} images. Using the first one for scaling.")

    # Get the very first image
    x_torch, _ = ds[0]
    x_torch = x_torch.unsqueeze(0)  # Add batch dimension

    # Convert to JAX array in [0, 1] range
    x_np = x_torch.numpy().transpose(0, 2, 3, 1)
    x_jax = jnp.asarray((x_np + 1.0) / 2.0)

    print("\n--- Encoding Image ---")
    posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
    z = posterior.mode()  # Use mode for a deterministic result

    unscaled_std = jnp.std(z)
    scale_factor = 1.0 / unscaled_std

    print("\n--- Results ---")
    print(f"Unscaled Latent Standard Deviation: {unscaled_std:.6f}")
    print(f"Calculated Scale Factor (1.0 / std): {scale_factor:.6f}")

    scaled_z = z * scale_factor
    print(f"Verification: Scaled Latent Std Dev: {jnp.std(scaled_z):.6f}")

    print("\n✅ Success! Use the 'Calculated Scale Factor' in your training script.")


if __name__ == "__main__":
    main()