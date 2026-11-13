import sys
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder_


def main():
    parser = argparse.ArgumentParser(description="Diagnose per-channel latent statistics.")
    parser.add_argument("--ae_config_path", type=str, required=True)
    parser.add_argument("--ae_ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task", type=str, default="TB")
    parser.add_argument("--num_samples", type=int, default=2259)
    args = parser.parse_args()

    print("--- Loading AE and Data ---")
    ae_model, ae_params = load_autoencoder_(args.ae_config_path, args.ae_ckpt_path)
    dataset = ChestXrayDataset(root_dir=args.data_root, task=args.task, split='train', img_size=128)
    loader = DataLoader(Subset(dataset, range(args.num_samples)), batch_size=16, shuffle=False)

    all_latents = []
    print("--- Encoding ---")
    for x_torch, _ in tqdm(loader):
        x_jax = (jnp.asarray(x_torch.numpy().transpose(0, 2, 3, 1)) + 1.0) / 2.0
        posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
        z = posterior.mode()  # Use mode to check pure signal structure, or sample() for full distribution
        all_latents.append(np.asarray(z))

    full_latents = np.concatenate(all_latents, axis=0)  # [N, H, W, C]

    print("\n--- 📊 Per-Channel Diagnosis ---")
    num_channels = full_latents.shape[-1]

    means = np.mean(full_latents, axis=(0, 1, 2))
    stds = np.std(full_latents, axis=(0, 1, 2))

    active_candidate = -1
    max_structure = -1

    for i in range(num_channels):
        # In a VAE, "dead" channels often collapse to the prior N(0,1) or have very low variance if pruned.
        # Structure often manifests as variance distinct from the noise floor.
        print(f"Channel {i}: Mean = {means[i]:.4f} | Std = {stds[i]:.4f}")

        # Heuristic: The active channel usually has the lowest standard deviation in X-ray VAEs
        # (compressed structure) OR highest deviation from Gaussian noise.
        # Based on your log, the active channel is likely the one that is NOT ~1.0 or ~0.0

    print("\nℹ️  Interpretation:")
    print("   - Channels with Std ≈ 1.0 are likely pure noise (posterior collapse).")
    print("   - Channels with Std < 0.5 usually contain the compressed structural signal.")

    # Prompt user or auto-select
    # Assuming the channel with the *lowest* std is the structural one for this calculation
    # (Common in KL-penalized medical models where structure is highly compressed)
    suggested_idx = np.argmin(stds)
    print(f"\n✅ Suggested Active Channel: {suggested_idx} (Std: {stds[suggested_idx]:.4f})")

    new_scale = 1.0 / stds[suggested_idx]
    print(f"🚀 Recommended Scale Factor for Channel {suggested_idx}: {new_scale:.6f}")
    print(f"   (Use this with --latent_scale_factor {new_scale:.6f} and --select_channel {suggested_idx})")


if __name__ == "__main__":
    main()