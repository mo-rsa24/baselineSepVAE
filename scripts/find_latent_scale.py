import sys
import os
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder, load_autoencoder_


# --- Visualization and helper functions ---

def _save_reconstruction_grid(ae_model, ae_params, loader, output_dir, n_samples=16):
    """Encodes and decodes a batch of images and saves the comparison."""
    x_real, _ = next(iter(loader))
    x_real = x_real[:n_samples]
    x_np = x_real.numpy().transpose(0, 2, 3, 1)
    x_jax = (jnp.asarray(x_np) + 1.0) / 2.0  #
    posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
    rng = jax.random.PRNGKey(0)
    z = posterior.sample(rng)
    x_rec_jax = ae_model.apply({'params': ae_params}, z, method=ae_model.decode, train=False)
    x_rec_torch = torch.from_numpy(np.asarray(x_rec_jax)).permute(0, 3, 1, 2).clamp(0, 1)
    x_real_norm = (x_real + 1.0) / 2.0
    comparison_grid = torch.cat([x_real_norm, x_rec_torch])
    grid_img = make_grid(comparison_grid, nrow=x_real.shape[0])
    save_image(grid_img, output_dir / "reconstruction_comparison.png")
    print(f"✅ Saved reconstruction grid to {output_dir}")


def _save_latent_histograms_matplotlib(latents, output_dir):
    """Saves histograms of the latent space distribution using Matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.hist(latents.flatten(), bins=100, color='steelblue', alpha=0.7)
    plt.title("Latent Space Distribution (Overall)", fontsize=16)
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "latent_distribution_overall.png", dpi=150)
    plt.close()
    print(f"✅ Saved overall latent distribution plot to {output_dir}")


def _visualize_latent_channels(latents, output_dir):
    """Visualizes the activation map of each latent channel for the first image."""
    first_image_latents = latents[0]
    num_channels = first_image_latents.shape[-1]
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle("Latent Channel Activations (First Image)", fontsize=20)
    for i in range(num_channels):
        ax = axes.flat[i]
        im = ax.imshow(first_image_latents[:, :, i], cmap='viridis')
        ax.set_title(f"Channel {i + 1}", fontsize=10)
        ax.axis('off')
    for i in range(num_channels, len(axes.flat)):
        axes.flat[i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = output_dir / "latent_channel_activations.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved channel activation map to {save_path}")


# --- NEW: Function to analyze channel contributions ---
def _visualize_channel_contributions(latents, output_dir):
    """
    Calculates the standard deviation of each channel and plots it as a sorted bar chart
    to identify the most significant latent dimensions.
    """
    print("📊 Analyzing latent channel contributions...")
    if latents.ndim != 4:
        print("⚠️ Cannot analyze channels, latent tensor is not 4D.")
        return

    # Calculate std dev for each channel across all other dimensions (N, H, W)
    # Shape is (N, H, W, C) -> std dev will have shape (C,)
    channel_stds = np.std(latents, axis=(0, 1, 2))

    # Sort the standard deviations in descending order for visualization
    sorted_indices = np.argsort(channel_stds)[::-1]
    sorted_stds = channel_stds[sorted_indices]

    plt.figure(figsize=(16, 8))
    plt.bar(range(len(sorted_stds)), sorted_stds, color='cyan')
    plt.title("Per-Channel Latent Standard Deviation (Sorted)", fontsize=18)
    plt.xlabel("Latent Channel Index (Sorted by Importance)", fontsize=12)
    plt.ylabel("Standard Deviation", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = output_dir / "latent_channel_contributions.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved channel contribution analysis to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Find latent scale factor and generate diagnostic visualizations.")
    # (Arguments remain the same)
    parser.add_argument("--ae_config_path", type=str, required=True, help="Path to the AE run_meta.json config.")
    parser.add_argument("--ae_ckpt_path", type=str, required=True, help="Path to the AE checkpoint (.flax).")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--task", type=str, required=True, help="Dataset task (e.g., TB).")
    parser.add_argument("--img_size", type=int, default=128, help="Image size.")
    parser.add_argument("--class_filter", type=int, default=1, help="Class to filter for.")
    parser.add_argument("--num_samples_for_scale", type=int, default=2259,
                        help="Number of images to use for scale calculation.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    args = parser.parse_args()

    run_dir = Path(args.ae_ckpt_path).parent.parent
    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    print("--- Loading Autoencoder & Dataset ---")
    ae_model, ae_params = load_autoencoder_(args.ae_config_path, args.ae_ckpt_path)
    dataset = ChestXrayDataset(root_dir=args.data_root, task=args.task, split='train', img_size=args.img_size,
                               class_filter=args.class_filter)
    num_samples = min(len(dataset), args.num_samples_for_scale)
    ds_subset = Subset(dataset, range(num_samples))
    loader = DataLoader(ds_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Using {num_samples} images for diagnostics.")

    print("\n--- Generating Reconstructions ---")
    _save_reconstruction_grid(ae_model, ae_params, loader, output_dir, n_samples=8)

    print("\n--- Encoding Images ---")
    all_latents = []
    for batch in tqdm(loader, desc="Encoding batches"):
        x_torch, _ = batch
        x_np = x_torch.numpy().transpose(0, 2, 3, 1)
        x_jax = (jnp.asarray(x_np) + 1.0) / 2.0  # <-- scale to [0,1]
        posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
        rng = jax.random.PRNGKey(0)
        z = posterior.sample(rng)
        all_latents.append(np.asarray(z))

    full_latents = np.concatenate(all_latents, axis=0)
    unscaled_std = full_latents.std()
    scale_factor = 1.0 / unscaled_std

    print("\n--- Results ---")
    print(f"Unscaled Latent Standard Deviation: {unscaled_std:.6f}")
    print(f"Calculated Scale Factor (1.0 / std): {scale_factor:.6f}")

    output_path = run_dir / "latent_scale_factor.txt"
    with open(output_path, 'w') as f:
        f.write(str(scale_factor))
    print(f"✅ Saved scale factor to {output_path}")

    print("\n--- Generating Latent Space Visualizations ---")
    _save_latent_histograms_matplotlib(full_latents, output_dir)
    _visualize_latent_channels(full_latents, output_dir)

    # --- NEW: Call the contribution analysis function ---
    _visualize_channel_contributions(full_latents, output_dir)

    print("\nDiagnostic script finished.")


if __name__ == "__main__":
    main()