"""
superdiff_and.py

Implements the SuperDiff algorithm for logical AND composition of two pretrained
unconditional Latent Diffusion Models (LDMs), based on Algorithm 1 from the paper:
"The Superposition of Pretrained Diffusion Models Using the Itô Density Estimator"
(arXiv:2412.17762v2).

This script is designed to work within the existing JAX-based LDM framework.
"""

import os
import sys
import json
import functools
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

# --- ensure repo root on sys.path ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# --- Project module imports ---
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder
from diffusion.vp_equation import (
    marginal_prob_std,
    diffusion_coeff,
)

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')  # Prevent TF from claiming GPU


def load_ldm(config_path: str, ckpt_path: str):
    """Loads a pretrained LDM ScoreNet model and its parameters."""
    print(f"▶️ Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model
    model = ScoreNet(
        marginal_prob_std_fn=functools.partial(marginal_prob_std, sigma=config['model']['sigma']),
        **config['model']
    )

    print(f"▶️ Loading LDM checkpoint from: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        ckpt_bytes = f.read()

    # Use flax serialization to restore params and state
    # We only need the parameters for inference.
    state_dict = from_bytes(target=None, encoded_bytes=ckpt_bytes)
    params = state_dict['params']

    # For compatibility with older checkpoints that might have ema_params
    if 'ema_params' in state_dict and state_dict['ema_params'] is not None:
        print("💡 Using EMA parameters for the model.")
        params = state_dict['ema_params']

    print("✅ LDM loaded successfully.")
    return model, params, config


def solve_kappa_for_and(s1: jnp.ndarray, s2: jnp.ndarray) -> jnp.ndarray:
    """
    Solves the linear system for kappa for logical AND with two score functions.
    This corresponds to Algorithm 1, line 6.

    The system is:
    [ s1·s1   s1·s2 ] [ κ1 ] = [ 1 ]
    [ s2·s1   s2·s2 ] [ κ2 ]   [ 1 ]

    Args:
        s1: Score estimate from model 1. Shape: (batch, H, W, C)
        s2: Score estimate from model 2. Shape: (batch, H, W, C)

    Returns:
        kappa: The solved weights. Shape: (batch, 2)
    """
    # Sum over spatial and channel dimensions for the dot product
    dot_product_axes = tuple(range(1, s1.ndim))
    s1_dot_s1 = jnp.sum(s1 * s1, axis=dot_product_axes)
    s2_dot_s2 = jnp.sum(s2 * s2, axis=dot_product_axes)
    s1_dot_s2 = jnp.sum(s1 * s2, axis=dot_product_axes)

    # Construct the batched 2x2 matrix A
    # Shape: (batch_size, 2, 2)
    A = jnp.array([
        [s1_dot_s1, s1_dot_s2],
        [s1_dot_s2, s2_dot_s2]
    ]).transpose((2, 0, 1))  # Transpose to get batch dim first

    # Construct the batched vector b
    # Shape: (batch_size, 2)
    b = jnp.ones((s1.shape[0], 2))

    # Add a small epsilon to the diagonal for numerical stability
    eps = 1e-6
    A += jnp.eye(2) * eps

    # Solve the system Ax = b for each item in the batch
    kappa = jnp.linalg.solve(A, b)
    return kappa


def get_composed_score(s1: jnp.ndarray, s2: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the composed score as a weighted sum using kappa.
    This corresponds to Algorithm 1, line 7.

    Args:
        s1: Score from model 1.
        s2: Score from model 2.
        kappa: Solved weights. Shape: (batch, 2)

    Returns:
        The composed score. Shape: (batch, H, W, C)
    """
    # Reshape kappa for broadcasting: (batch, 2) -> (batch, 1, 1, 1, 2)
    kappa_reshaped = kappa.reshape(kappa.shape[0], 1, 1, 1, 2)

    # Stack scores along a new last dimension: (batch, H, W, C, 2)
    scores_stacked = jnp.stack([s1, s2], axis=-1)

    # Perform weighted sum via dot product
    composed_score = jnp.sum(scores_stacked * kappa_reshaped, axis=-1)

    return composed_score


def sample_and_track(
        score_fn_tb,
        score_fn_normal,
        rng,
        shape,
        num_steps,
        sigma,
        eps=1e-3
):
    """
    Main sampling loop implementing Algorithm 1 with Euler-Maruyama discretization.
    Tracks diagnostics like score norms and kappa values over time.

    Args:
        score_fn_tb: JAX function for the TB LDM score.
        score_fn_normal: JAX function for the Normal LDM score.
        rng: JAX random key.
        shape: The shape of the latent variables, e.g., (batch, H, W, C).
        num_steps: Number of discretization steps (N in Algorithm 1).
        sigma: The sigma parameter of the SDE.
        eps: The final time, close to zero.

    Returns:
        x_final: The final sampled latent.
        diagnostics_history: A list of dictionaries containing metrics from each step.
    """
    batch_size = shape[0]
    diagnostics_history = []

    # --- Algorithm 1, line 3: Initialize from prior ---
    rng, step_rng = jax.random.split(rng)
    x = jax.random.normal(step_rng, shape)

    time_steps = jnp.linspace(1., eps, num_steps)
    dt = time_steps[0] - time_steps[1]

    print("🚀 Starting SuperDiff AND sampling...")
    for t in tqdm(time_steps, desc="Sampling"):
        # Create a batched time vector
        t_vec = jnp.ones(batch_size) * t

        # --- Algorithm 1, line 5: Get scores from pretrained models ---
        s_tb = score_fn_tb(x, t_vec)
        s_normal = score_fn_normal(x, t_vec)

        # --- Algorithm 1, line 6: Solve for kappa ---
        kappa = solve_kappa_for_and(s_tb, s_normal)

        # --- Algorithm 1, line 7: Compute composed score ---
        s_composed = get_composed_score(s_tb, s_normal, kappa)

        # --- Algorithm 1, line 9: Euler-Maruyama update step ---
        rng, step_rng = jax.random.split(rng)
        g = diffusion_coeff(t, sigma)
        drift = g ** 2 * s_composed * dt
        diffusion = g * jnp.sqrt(dt) * jax.random.normal(step_rng, x.shape)
        x = x + drift + diffusion

        # --- Diagnostics Tracking ---
        diagnostics = {
            'time': t,
            'kappa': np.array(kappa),
            'score_norm_tb': np.linalg.norm(s_tb.reshape(batch_size, -1), axis=-1),
            'score_norm_normal': np.linalg.norm(s_normal.reshape(batch_size, -1), axis=-1),
            'score_norm_composed': np.linalg.norm(s_composed.reshape(batch_size, -1), axis=-1),
        }
        diagnostics_history.append(diagnostics)

    print("✅ Sampling complete.")
    return x, diagnostics_history


def plot_diagnostics(diagnostics_history, save_path):
    """Plots and saves the diagnostics from the sampling process."""
    if not diagnostics_history:
        print("⚠️ No diagnostics to plot.")
        return

    print("📊 Processing and saving diagnostics plots...")
    diag_agg = {k: np.array([d[k] for d in diagnostics_history]) for k in diagnostics_history[0]}
    steps_axis = np.arange(len(diagnostics_history))

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Score Norms
    axs[0].plot(steps_axis, diag_agg['score_norm_tb'].mean(axis=-1), label="TB Score Norm")
    axs[0].plot(steps_axis, diag_agg['score_norm_normal'].mean(axis=-1), label="Normal Score Norm")
    axs[0].plot(steps_axis, diag_agg['score_norm_composed'].mean(axis=-1), label="Composed Score Norm", linestyle='--',
                alpha=0.9)
    axs[0].set_title("Mean Score Norms Over Sampling Steps")
    axs[0].set_xlabel("Sampling Step (T -> 0)")
    axs[0].set_ylabel("L2 Norm")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Kappa Values
    axs[1].plot(steps_axis, diag_agg['kappa'][:, :, 0].mean(axis=-1), label="κ_TB (TB weight)")
    axs[1].plot(steps_axis, diag_agg['kappa'][:, :, 1].mean(axis=-1), label="κ_Normal (Normal weight)")
    axs[1].set_title("Mean Kappa Values Over Sampling Steps")
    axs[1].set_xlabel("Sampling Step (T -> 0)")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Diagnostics plot saved to {save_path}")
    plt.close(fig)


def main(args):
    # --- Setup ---
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    # --- Load Autoencoder ---
    vae, vae_params = load_autoencoder(args.ae_path)
    # The latent shape is determined by a forward pass of the VAE encoder
    dummy_image = jnp.ones((1, args.image_size, args.image_size, 1))
    latent_shape = vae.encode(vae_params, dummy_image).shape
    full_latent_shape = (args.batch_size,) + latent_shape[1:]
    print(f"✅ VAE loaded. Latent shape: {full_latent_shape}")

    # --- Load LDMs ---
    ldm_tb, ldm_params_tb, config_tb = load_ldm(args.config_tb, args.ckpt_tb)
    ldm_normal, ldm_params_normal, config_normal = load_ldm(args.config_normal, args.ckpt_normal)

    # Verify that sigmas match for consistent diffusion processes
    assert config_tb['model']['sigma'] == config_normal['model']['sigma'], \
        "Sigma values for both LDMs must match."
    sigma = config_tb['model']['sigma']

    # --- Create JIT-compiled score functions for performance ---
    score_fn_tb = jax.jit(lambda x, t: ldm_tb.apply({'params': ldm_params_tb}, x, t))
    score_fn_normal = jax.jit(lambda x, t: ldm_normal.apply({'params': ldm_params_normal}, x, t))

    # --- Run Sampling ---
    rng, sample_rng = jax.random.split(rng)
    final_latents, diagnostics = sample_and_track(
        score_fn_tb=score_fn_tb,
        score_fn_normal=score_fn_normal,
        rng=sample_rng,
        shape=full_latent_shape,
        num_steps=args.num_steps,
        sigma=sigma,
    )

    # --- Decode and Save Images ---
    print("🎨 Decoding latents into images...")
    decoded_images = vae.decode(vae_params, final_latents)
    # Clamp and normalize images to [0, 1] for saving
    decoded_images = jnp.clip(decoded_images, -1., 1.)
    decoded_images = (decoded_images + 1.) / 2.

    # Use torch to save a grid of images
    torch_images = torch.from_numpy(np.array(decoded_images.transpose(0, 3, 1, 2)))
    save_path = os.path.join(output_dir, "composed_samples.png")
    save_image(torch_images, save_path, nrow=int(np.sqrt(args.batch_size)))
    print(f"✅ Composed images saved to {save_path}")

    # --- Save Diagnostics ---
    plot_diagnostics(diagnostics, os.path.join(output_dir, "diagnostics.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SuperDiff AND Composition Sampler")
    # Model Paths
    parser.add_argument("--config_tb", type=str, required=True, help="Path to the config file for the TB LDM.")
    parser.add_argument("--ckpt_tb", type=str, required=True, help="Path to the checkpoint file for the TB LDM.")
    parser.add_argument("--config_normal", type=str, required=True, help="Path to the config file for the Normal LDM.")
    parser.add_argument("--ckpt_normal", type=str, required=True,
                        help="Path to the checkpoint file for the Normal LDM.")
    parser.add_argument("--ae_path", type=str, required=True, help="Path to the pretrained autoencoder checkpoint.")

    # Sampling Parameters
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of sampling steps (N).")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of the input images to determine latent shape.")

    # Misc
    parser.add_argument("--output_dir", type=str, default="superdiff_and_output",
                        help="Directory to save generated images and logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)