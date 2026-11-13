"""
SepVAE Latent Manifold Analysis Script.

This script provides advanced diagnostic tools to investigate the "health" of
the SepVAE latent space manifold, beyond simple t-SNE/PCA visualization.

Analyses included:
1. LATENT TRAVERSALS & GEODESICS
   - Linear Interpolation (LERP): Straight-line paths between latent points
   - Spherical Interpolation (SLERP): Paths along the hypersphere surface
   - Structured traversals: Vary z_common vs z_disease independently

2. LATENT DENSITY ESTIMATION
   - Gaussian Mixture Model (GMM) fitting on latent representations
   - Kernel Density Estimation (KDE) for continuous density maps
   - "Vacuum" sampling: Generate images from low-density regions to find manifold holes

3. TOPOLOGICAL DATA ANALYSIS (TDA) - Optional
   - Persistent homology to identify topological "holes" in the manifold
   - Mapper algorithm for graph-based manifold visualization

Usage:
======
Basic analysis:
    python -m run.analyze_sepvae_manifold --checkpoint runs_sepvae/exp/checkpoints/checkpoint_epoch0050.pkl

Full analysis with TDA:
    python -m run.analyze_sepvae_manifold \\
        --checkpoint runs_sepvae/exp/checkpoints/checkpoint_final.pkl \\
        --enable_tda \\
        --n_samples 500 \\
        --n_interpolation_steps 10

Structured latent traversals only:
    python -m run.analyze_sepvae_manifold \\
        --checkpoint checkpoint.pkl \\
        --analysis traversals
"""

import argparse
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from flax.serialization import msgpack_restore
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

# Local imports
from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
from models.sep_vae_jax import SepVAE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def load_checkpoint(ckpt_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load a SepVAE checkpoint.

    Returns:
        vae_params: Model parameters
        vae_batch_stats: BatchNorm statistics
        args: Training arguments dict
    """
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        ckpt = msgpack_restore(f.read())

    vae_params = jax.tree_util.tree_map(jnp.array, ckpt['vae_params'])
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, ckpt['vae_batch_stats'])
    args = ckpt.get('args', {})

    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")

    return vae_params, vae_batch_stats, args


def create_model_from_args(ckpt_args: Dict, override_upsample: str = None) -> SepVAE:
    """
    Create SepVAE model from checkpoint args.

    Args:
        ckpt_args: Arguments dict from checkpoint
        override_upsample: If provided, override the checkpoint's upsample_method.
                          Useful for loading old checkpoints trained with 'nearest'
                          but wanting to use 'bilinear' for analysis.
    """
    # Determine upsample method: CLI override > checkpoint > default
    upsample_method = override_upsample or ckpt_args.get('upsample_method', 'nearest')

    return SepVAE(
        z_channels_common=ckpt_args.get('z_channels_common', 4),
        z_channels_disease=ckpt_args.get('z_channels_disease', 2),
        frozen_backbone=ckpt_args.get('frozen_backbone', True),
        use_fpn=ckpt_args.get('use_fpn', False),
        fpn_channels=ckpt_args.get('fpn_channels', 512),
        unfreeze_from=ckpt_args.get('unfreeze_from', None),
        sigma_inactive=ckpt_args.get('sigma_inactive', 1.0),
        upsample_method=upsample_method,
    )


# =============================================================================
# 1. LATENT TRAVERSALS & GEODESICS
# =============================================================================

def lerp(z1: jnp.ndarray, z2: jnp.ndarray, t: float) -> jnp.ndarray:
    """Linear interpolation between two latent points."""
    return (1.0 - t) * z1 + t * z2


def slerp(z1: jnp.ndarray, z2: jnp.ndarray, t: float, eps: float = 1e-8) -> jnp.ndarray:
    """
    Spherical linear interpolation (SLERP) between two latent points.

    Better suited for VAE latent spaces which are often Gaussian-distributed,
    as it interpolates along the surface of a hypersphere rather than
    cutting through the "empty" center of the distribution.
    """
    # Flatten for norm computation
    z1_flat = z1.reshape(-1)
    z2_flat = z2.reshape(-1)

    # Normalize to unit sphere
    z1_norm = z1_flat / (jnp.linalg.norm(z1_flat) + eps)
    z2_norm = z2_flat / (jnp.linalg.norm(z2_flat) + eps)

    # Compute angle between vectors
    dot = jnp.clip(jnp.dot(z1_norm, z2_norm), -1.0, 1.0)
    omega = jnp.arccos(dot)

    # Handle near-parallel vectors (fall back to LERP)
    sin_omega = jnp.sin(omega)
    if sin_omega < eps:
        return lerp(z1, z2, t)

    # SLERP formula
    s1 = jnp.sin((1.0 - t) * omega) / sin_omega
    s2 = jnp.sin(t * omega) / sin_omega

    # Scale back to original magnitudes
    mag1 = jnp.linalg.norm(z1_flat)
    mag2 = jnp.linalg.norm(z2_flat)
    mag_interp = (1.0 - t) * mag1 + t * mag2

    result = (s1 * z1_norm + s2 * z2_norm) * mag_interp
    return result.reshape(z1.shape)


def interpolate_latents(
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    n_steps: int = 10,
    method: str = 'lerp'
) -> List[jnp.ndarray]:
    """
    Generate interpolation path between two latent points.

    Args:
        z1, z2: Latent tensors (spatial: 64×64×C or flattened)
        n_steps: Number of interpolation steps (including endpoints)
        method: 'lerp' for linear, 'slerp' for spherical

    Returns:
        List of interpolated latent tensors
    """
    interp_fn = slerp if method == 'slerp' else lerp
    t_values = np.linspace(0, 1, n_steps)
    return [interp_fn(z1, z2, t) for t in t_values]


def extract_latents_for_traversal(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    images: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Extract latent representations for a batch of images.

    Returns dict with keys:
        - 'z_common': (B, 64, 64, 4) common latent means
        - 'z_cardiomegaly': (B, 64, 64, 2) disease latent means
        - 'z_effusion': (B, 64, 64, 2) disease latent means
        - 'z_concat': (B, 64, 64, 8) concatenated latents
    """
    variables = {'params': params, 'batch_stats': batch_stats}
    latents_dict = model.apply(variables, images, method=model.encode)

    mu_c, _ = latents_dict['common']
    mu_cardio, _ = latents_dict['cardiomegaly']
    mu_effusion, _ = latents_dict['effusion']

    z_concat = jnp.concatenate([mu_c, mu_cardio, mu_effusion], axis=-1)

    return {
        'z_common': mu_c,
        'z_cardiomegaly': mu_cardio,
        'z_effusion': mu_effusion,
        'z_concat': z_concat,
        'labels': labels,
    }


def decode_latents(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    z_concat: jnp.ndarray
) -> jnp.ndarray:
    """Decode concatenated latents to images."""
    variables = {'params': params, 'batch_stats': batch_stats}
    return model.apply(variables, z_concat, method=model.decode)


def visualize_latent_traversal(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    z_start: jnp.ndarray,
    z_end: jnp.ndarray,
    n_steps: int = 10,
    method: str = 'lerp',
    title: str = "Latent Traversal",
    save_path: str = None
):
    """
    Visualize decoded images along an interpolation path.

    Shows if the manifold is continuous (smooth transitions) or has
    "holes" (images become mushy/anatomically incorrect mid-traversal).
    """
    # Interpolate
    z_path = interpolate_latents(z_start, z_end, n_steps, method)
    z_stack = jnp.stack(z_path, axis=0)  # (n_steps, 64, 64, 8)

    # Decode all at once
    x_decoded = decode_latents(model, params, batch_stats, z_stack)

    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2.5))
    fig.suptitle(f"{title} ({method.upper()})", fontsize=12)

    for i, ax in enumerate(axes):
        img = np.array(x_decoded[i, :, :, 0])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"t={i/(n_steps-1):.2f}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    return x_decoded


def structured_traversal(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    z_source: Dict[str, jnp.ndarray],
    z_target: Dict[str, jnp.ndarray],
    vary_component: str = 'common',
    n_steps: int = 10,
    save_path: str = None
):
    """
    Structured traversal: Vary one latent component while keeping others fixed.

    This is crucial for SepVAE to verify disentanglement:
    - Varying z_common should change patient identity/anatomy but not pathology
    - Varying z_disease should change pathology but preserve anatomy

    Args:
        z_source, z_target: Dicts with 'z_common', 'z_cardiomegaly', 'z_effusion'
        vary_component: Which component to interpolate ('common', 'cardiomegaly', 'effusion', 'all')
    """
    z_paths = []

    for t in np.linspace(0, 1, n_steps):
        z_interp = {}

        for key in ['z_common', 'z_cardiomegaly', 'z_effusion']:
            if vary_component == 'all' or key == f'z_{vary_component}':
                # Interpolate this component
                z_interp[key] = lerp(z_source[key][0], z_target[key][0], t)
            else:
                # Keep source value fixed
                z_interp[key] = z_source[key][0]

        # Concatenate
        z_concat = jnp.concatenate([
            z_interp['z_common'],
            z_interp['z_cardiomegaly'],
            z_interp['z_effusion']
        ], axis=-1)
        z_paths.append(z_concat)

    z_stack = jnp.stack(z_paths, axis=0)
    x_decoded = decode_latents(model, params, batch_stats, z_stack)

    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2.5))
    title = f"Structured Traversal: Varying {vary_component}"
    fig.suptitle(title, fontsize=12)

    for i, ax in enumerate(axes):
        img = np.array(x_decoded[i, :, :, 0])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"t={i/(n_steps-1):.2f}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    return x_decoded


def run_traversal_analysis(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    loader: DataLoader,
    output_dir: str,
    n_steps: int = 10
):
    """
    Run comprehensive latent traversal analysis.

    Generates:
    1. Cross-class traversals (Normal ↔ Effusion ↔ Cardiomegaly)
    2. Within-class traversals (variation within same disease)
    3. Structured traversals (vary z_common vs z_disease independently)
    """
    print("\n" + "="*60)
    print("LATENT TRAVERSAL ANALYSIS")
    print("="*60)

    # Collect samples from each class
    samples_by_class = {0: [], 1: [], 2: []}  # Normal, Effusion, Cardiomegaly

    for batch in loader:
        x_norm = jnp.array(batch['x_norm'].permute(0, 2, 3, 1).numpy())
        x_eff = jnp.array(batch['x_disease1'].permute(0, 2, 3, 1).numpy())
        x_card = jnp.array(batch['x_disease2'].permute(0, 2, 3, 1).numpy())

        # Extract latents
        for x, cls_id in [(x_norm, 0), (x_eff, 1), (x_card, 2)]:
            labels = jnp.full((x.shape[0],), cls_id)
            latents = extract_latents_for_traversal(model, params, batch_stats, x, labels)
            samples_by_class[cls_id].append(latents)

        # Collect enough samples
        if len(samples_by_class[0]) >= 3:
            break

    # Concatenate samples per class
    for cls_id in samples_by_class:
        if samples_by_class[cls_id]:
            samples_by_class[cls_id] = {
                key: jnp.concatenate([s[key] for s in samples_by_class[cls_id]], axis=0)
                for key in samples_by_class[cls_id][0].keys()
            }

    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']

    # --- 1. Cross-class traversals ---
    print("\n[1/3] Cross-class traversals...")
    cross_pairs = [(0, 1), (0, 2), (1, 2)]  # Normal↔Eff, Normal↔Cardio, Eff↔Cardio

    for cls_a, cls_b in cross_pairs:
        z_a = samples_by_class[cls_a]['z_concat'][0]
        z_b = samples_by_class[cls_b]['z_concat'][0]

        for method in ['lerp', 'slerp']:
            title = f"{disease_names[cls_a]} → {disease_names[cls_b]}"
            save_path = f"{output_dir}/traversal_{disease_names[cls_a].lower()}_{disease_names[cls_b].lower()}_{method}.png"

            visualize_latent_traversal(
                model, params, batch_stats,
                z_a, z_b, n_steps=n_steps,
                method=method, title=title, save_path=save_path
            )

    # --- 2. Within-class traversals ---
    print("\n[2/3] Within-class traversals...")
    for cls_id in range(3):
        if samples_by_class[cls_id]['z_concat'].shape[0] >= 2:
            z_a = samples_by_class[cls_id]['z_concat'][0]
            z_b = samples_by_class[cls_id]['z_concat'][1]

            title = f"Within {disease_names[cls_id]}"
            save_path = f"{output_dir}/traversal_within_{disease_names[cls_id].lower()}.png"

            visualize_latent_traversal(
                model, params, batch_stats,
                z_a, z_b, n_steps=n_steps,
                method='lerp', title=title, save_path=save_path
            )

    # --- 3. Structured traversals (disentanglement check) ---
    print("\n[3/3] Structured traversals (disentanglement verification)...")

    # Take Normal and Cardiomegaly samples
    z_normal = {k: v for k, v in samples_by_class[0].items() if k.startswith('z_')}
    z_cardio = {k: v for k, v in samples_by_class[2].items() if k.startswith('z_')}

    for component in ['common', 'cardiomegaly', 'effusion', 'all']:
        save_path = f"{output_dir}/structured_traversal_vary_{component}.png"
        structured_traversal(
            model, params, batch_stats,
            z_normal, z_cardio,
            vary_component=component,
            n_steps=n_steps,
            save_path=save_path
        )

    print(f"\nTraversal visualizations saved to: {output_dir}")


# =============================================================================
# 2. LATENT DENSITY ESTIMATION
# =============================================================================

def extract_all_latents(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    loader: DataLoader,
    max_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract flattened latent representations from the dataset.

    Returns:
        latents: (N, D) array of flattened latent means
        labels: (N,) array of class labels
    """
    all_latents = []
    all_labels = []
    n_collected = 0

    for batch in loader:
        if n_collected >= max_samples:
            break

        # Process all three image types
        for x_key, cls_id in [('x_norm', 0), ('x_disease1', 1), ('x_disease2', 2)]:
            x = jnp.array(batch[x_key].permute(0, 2, 3, 1).numpy())
            B = x.shape[0]

            variables = {'params': params, 'batch_stats': batch_stats}
            latents_dict = model.apply(variables, x, method=model.encode)

            # Global average pool spatial dims, then concatenate heads
            mu_c, _ = latents_dict['common']
            mu_cardio, _ = latents_dict['cardiomegaly']
            mu_eff, _ = latents_dict['effusion']

            # (B, 64, 64, C) -> (B, C)
            z_flat = jnp.concatenate([
                jnp.mean(mu_c, axis=(1, 2)),
                jnp.mean(mu_cardio, axis=(1, 2)),
                jnp.mean(mu_eff, axis=(1, 2))
            ], axis=-1)

            all_latents.append(np.array(z_flat))
            all_labels.extend([cls_id] * B)
            n_collected += B

    latents = np.concatenate(all_latents, axis=0)[:max_samples]
    labels = np.array(all_labels)[:max_samples]

    return latents, labels


def fit_density_model(
    latents: np.ndarray,
    method: str = 'gmm',
    n_components: int = 10,
    bandwidth: float = 0.5
):
    """
    Fit a density model to the latent representations.

    Args:
        latents: (N, D) array of latent vectors
        method: 'gmm' for Gaussian Mixture, 'kde' for Kernel Density
        n_components: Number of GMM components (ignored for KDE)
        bandwidth: KDE bandwidth (ignored for GMM)

    Returns:
        Fitted density model
    """
    if method == 'gmm':
        model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            n_init=3
        )
        model.fit(latents)
    elif method == 'kde':
        model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        model.fit(latents)
    else:
        raise ValueError(f"Unknown density method: {method}")

    return model


def find_low_density_regions(
    density_model,
    latents: np.ndarray,
    n_samples: int = 50,
    percentile: float = 5.0
) -> np.ndarray:
    """
    Find samples in low-density regions of the latent space.

    Args:
        density_model: Fitted GMM or KDE model
        latents: (N, D) latent samples
        n_samples: Number of low-density samples to return
        percentile: Bottom percentile to consider "low density"

    Returns:
        (n_samples, D) array of low-density latent points
    """
    # Score all points
    if hasattr(density_model, 'score_samples'):
        # Both GMM and KDE have score_samples
        log_probs = density_model.score_samples(latents)
    else:
        raise ValueError("Model must have score_samples method")

    # Find threshold for low density
    threshold = np.percentile(log_probs, percentile)
    low_density_mask = log_probs <= threshold
    low_density_latents = latents[low_density_mask]

    # Return subset
    if len(low_density_latents) > n_samples:
        indices = np.random.choice(len(low_density_latents), n_samples, replace=False)
        return low_density_latents[indices]

    return low_density_latents


def sample_from_vacuum(
    density_model,
    latents: np.ndarray,
    n_samples: int = 20,
    temperature: float = 1.5
) -> np.ndarray:
    """
    Sample from "vacuum" regions (areas between clusters).

    Strategy: Interpolate between random pairs of points, then filter
    for those with low density scores.

    Args:
        density_model: Fitted density model
        latents: (N, D) existing latent samples
        n_samples: Number of vacuum samples to generate
        temperature: Controls how far from the manifold to sample
    """
    vacuum_samples = []
    attempts = 0
    max_attempts = n_samples * 20

    # Get density distribution statistics
    log_probs = density_model.score_samples(latents)
    mean_log_prob = np.mean(log_probs)
    std_log_prob = np.std(log_probs)
    vacuum_threshold = mean_log_prob - temperature * std_log_prob

    while len(vacuum_samples) < n_samples and attempts < max_attempts:
        # Pick two random points
        idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
        z1, z2 = latents[idx1], latents[idx2]

        # Interpolate at random t
        t = np.random.uniform(0.3, 0.7)  # Avoid endpoints
        z_interp = (1 - t) * z1 + t * z2

        # Check if it's in a vacuum
        log_prob = density_model.score_samples(z_interp.reshape(1, -1))[0]
        if log_prob < vacuum_threshold:
            vacuum_samples.append(z_interp)

        attempts += 1

    if len(vacuum_samples) == 0:
        print("  Warning: No vacuum samples found, using random interpolations")
        # Fallback: just return random interpolations
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
            t = np.random.uniform(0.3, 0.7)
            z_interp = (1 - t) * latents[idx1] + t * latents[idx2]
            vacuum_samples.append(z_interp)

    return np.array(vacuum_samples[:n_samples])


def decode_flattened_latents(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    z_flat: np.ndarray,
    z_channels_common: int = 4,
    z_channels_disease: int = 2,
    spatial_size: int = 64
) -> np.ndarray:
    """
    Decode flattened latent vectors back to images.

    Since we global-average-pooled the spatial dimensions, we need to
    "broadcast" the flat vector back to spatial format.
    """
    B, D = z_flat.shape

    # Split into components
    z_c = z_flat[:, :z_channels_common]
    z_cardio = z_flat[:, z_channels_common:z_channels_common + z_channels_disease]
    z_eff = z_flat[:, z_channels_common + z_channels_disease:]

    # Broadcast to spatial (B, 64, 64, C) - constant across space
    z_c_spatial = np.tile(z_c[:, None, None, :], (1, spatial_size, spatial_size, 1))
    z_cardio_spatial = np.tile(z_cardio[:, None, None, :], (1, spatial_size, spatial_size, 1))
    z_eff_spatial = np.tile(z_eff[:, None, None, :], (1, spatial_size, spatial_size, 1))

    # Concatenate
    z_concat = np.concatenate([z_c_spatial, z_cardio_spatial, z_eff_spatial], axis=-1)
    z_concat_jax = jnp.array(z_concat)

    # Decode
    variables = {'params': params, 'batch_stats': batch_stats}
    x_decoded = model.apply(variables, z_concat_jax, method=model.decode)

    return np.array(x_decoded)


def visualize_density_analysis(
    latents: np.ndarray,
    labels: np.ndarray,
    density_model,
    low_density_latents: np.ndarray,
    vacuum_decoded: np.ndarray,
    output_dir: str
):
    """
    Create comprehensive density analysis visualizations.
    """
    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']
    colors = ['blue', 'red', 'green']

    # --- Figure 1: Density heatmap in 2D PCA space ---
    print("  Creating density heatmap...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter with class colors
    for cls_id in range(3):
        mask = labels == cls_id
        axes[0].scatter(
            latents_2d[mask, 0], latents_2d[mask, 1],
            c=colors[cls_id], label=disease_names[cls_id],
            alpha=0.5, s=20
        )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Latent Space (PCA)')
    axes[0].legend()

    # Right: Density heatmap
    log_probs = density_model.score_samples(latents)
    scatter = axes[1].scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=log_probs, cmap='viridis', s=20, alpha=0.7
    )
    plt.colorbar(scatter, ax=axes[1], label='Log Density')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Density Estimation (GMM)')

    # Mark low-density points
    if len(low_density_latents) > 0:
        low_2d = pca.transform(low_density_latents)
        axes[1].scatter(
            low_2d[:, 0], low_2d[:, 1],
            c='red', marker='x', s=50, linewidths=2,
            label='Low density'
        )
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/density_heatmap.png")
    plt.show()

    # --- Figure 2: Vacuum samples decoded ---
    if vacuum_decoded is not None and len(vacuum_decoded) > 0:
        print("  Creating vacuum sample visualization...")
        n_show = min(len(vacuum_decoded), 16)
        n_cols = min(n_show, 8)
        n_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        axes = np.atleast_2d(axes)
        fig.suptitle('Decoded Samples from Low-Density Regions ("Vacuum Test")', fontsize=12)

        for i in range(n_show):
            row, col = i // n_cols, i % n_cols
            img = vacuum_decoded[i, :, :, 0]
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')

        # Hide unused axes
        for i in range(n_show, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/vacuum_samples.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/vacuum_samples.png")
        plt.show()

    # --- Figure 3: Density histogram ---
    print("  Creating density histogram...")
    fig, ax = plt.subplots(figsize=(10, 5))

    for cls_id in range(3):
        mask = labels == cls_id
        cls_log_probs = log_probs[mask]
        ax.hist(cls_log_probs, bins=50, alpha=0.5, label=disease_names[cls_id],
                color=colors[cls_id], density=True)

    ax.axvline(np.percentile(log_probs, 5), color='red', linestyle='--',
               label='5th percentile (vacuum threshold)')
    ax.set_xlabel('Log Density')
    ax.set_ylabel('Frequency')
    ax.set_title('Density Distribution by Disease Class')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_histogram.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/density_histogram.png")
    plt.show()


def run_density_analysis(
    model: SepVAE,
    params: Dict,
    batch_stats: Dict,
    loader: DataLoader,
    output_dir: str,
    n_samples: int = 500,
    n_vacuum_samples: int = 20,
    z_channels_common: int = 4,
    z_channels_disease: int = 2
):
    """
    Run comprehensive latent density estimation analysis.
    """
    print("\n" + "="*60)
    print("LATENT DENSITY ANALYSIS")
    print("="*60)

    # Extract latents
    print(f"\nExtracting latents from {n_samples} samples...")
    latents, labels = extract_all_latents(
        model, params, batch_stats, loader, max_samples=n_samples
    )
    print(f"  Extracted: {latents.shape[0]} samples, {latents.shape[1]} dimensions")

    # Fit density model
    print("\nFitting GMM density model...")
    density_model = fit_density_model(latents, method='gmm', n_components=10)

    # Score samples
    log_probs = density_model.score_samples(latents)
    print(f"  Log-prob range: [{log_probs.min():.2f}, {log_probs.max():.2f}]")
    print(f"  Mean log-prob: {log_probs.mean():.2f} ± {log_probs.std():.2f}")

    # Find low-density samples
    print("\nFinding low-density regions...")
    low_density_latents = find_low_density_regions(
        density_model, latents, n_samples=n_vacuum_samples, percentile=5.0
    )
    print(f"  Found {len(low_density_latents)} low-density samples")

    # Sample from vacuum and decode
    print("\nSampling from vacuum regions...")
    vacuum_latents = sample_from_vacuum(
        density_model, latents, n_samples=n_vacuum_samples, temperature=1.5
    )

    print("Decoding vacuum samples...")
    vacuum_decoded = decode_flattened_latents(
        model, params, batch_stats, vacuum_latents,
        z_channels_common=z_channels_common,
        z_channels_disease=z_channels_disease
    )

    # Visualize
    print("\nGenerating visualizations...")
    visualize_density_analysis(
        latents, labels, density_model,
        low_density_latents, vacuum_decoded, output_dir
    )

    # Summary statistics
    print("\n" + "-"*40)
    print("DENSITY ANALYSIS SUMMARY")
    print("-"*40)

    for cls_id, name in enumerate(['Normal', 'Effusion', 'Cardiomegaly']):
        mask = labels == cls_id
        cls_log_probs = log_probs[mask]
        print(f"  {name}: mean={cls_log_probs.mean():.2f}, std={cls_log_probs.std():.2f}")

    # Check for class separation in density
    from scipy.stats import ks_2samp
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mask_i, mask_j = labels == i, labels == j
        stat, pval = ks_2samp(log_probs[mask_i], log_probs[mask_j])
        names = ['Normal', 'Effusion', 'Cardiomegaly']
        print(f"  KS test {names[i]} vs {names[j]}: stat={stat:.3f}, p={pval:.4f}")


# =============================================================================
# 3. TOPOLOGICAL DATA ANALYSIS (TDA)
# =============================================================================

def run_tda_analysis(
    latents: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    max_dimension: int = 1,
    n_samples_tda: int = 300
):
    """
    Run Topological Data Analysis using persistent homology.

    Requires: pip install ripser persim

    This analysis identifies topological "holes" in the latent manifold,
    which indicate regions where the model cannot generate valid data.

    Args:
        latents: (N, D) latent vectors
        labels: (N,) class labels
        output_dir: Where to save figures
        max_dimension: Maximum homology dimension to compute (0=components, 1=holes)
        n_samples_tda: Subsample for computational efficiency
    """
    print("\n" + "="*60)
    print("TOPOLOGICAL DATA ANALYSIS (TDA)")
    print("="*60)

    try:
        from ripser import ripser
        from persim import plot_diagrams
    except ImportError:
        print("TDA requires ripser and persim packages.")
        print("Install with: pip install ripser persim")
        return

    # Subsample for efficiency
    if len(latents) > n_samples_tda:
        indices = np.random.choice(len(latents), n_samples_tda, replace=False)
        latents_sub = latents[indices]
        labels_sub = labels[indices]
    else:
        latents_sub = latents
        labels_sub = labels

    print(f"Running persistent homology on {len(latents_sub)} samples...")

    # Compute persistence diagrams
    result = ripser(latents_sub, maxdim=max_dimension)
    diagrams = result['dgms']

    # Plot persistence diagrams
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Persistence diagram
    plot_diagrams(diagrams, ax=axes[0], show=False)
    axes[0].set_title('Persistence Diagram')

    # Right: Barcode plot
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue

        # Sort by persistence (death - birth)
        persistence = dgm[:, 1] - dgm[:, 0]
        sorted_idx = np.argsort(-persistence)

        for i, idx in enumerate(sorted_idx[:20]):  # Top 20 features
            birth, death = dgm[idx]
            if np.isinf(death):
                death = dgm[:, 1][~np.isinf(dgm[:, 1])].max() * 1.1
            axes[1].barh(i + dim * 25, death - birth, left=birth,
                        color=['blue', 'red'][dim], alpha=0.7,
                        label=f'H{dim}' if i == 0 else '')

    axes[1].set_xlabel('Filtration Value')
    axes[1].set_ylabel('Feature Index')
    axes[1].set_title('Persistence Barcode (Top 20 per dimension)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tda_persistence.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/tda_persistence.png")
    plt.show()

    # Summary statistics
    print("\nPersistence Summary:")
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        persistence = dgm[:, 1] - dgm[:, 0]
        persistence = persistence[~np.isinf(persistence)]
        if len(persistence) > 0:
            print(f"  H{dim}: {len(dgm)} features, max persistence={persistence.max():.3f}")

    # Per-class TDA (optional)
    print("\nPer-class persistent features (H0 = connected components):")
    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']
    for cls_id in range(3):
        mask = labels_sub == cls_id
        if mask.sum() > 10:
            cls_latents = latents_sub[mask]
            cls_result = ripser(cls_latents, maxdim=0)
            n_components = len(cls_result['dgms'][0])
            print(f"  {disease_names[cls_id]}: {n_components} H0 features (connected components)")


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser("SepVAE Manifold Analysis")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SepVAE checkpoint (.pkl)")

    # Data
    parser.add_argument("--dicom_dir", type=str, default="/datasets/mmolefe/vinbigdata/train",
                       help="Path to DICOM files")
    parser.add_argument("--csv_path", type=str, default="/datasets/mmolefe/vinbigdata/train.csv",
                       help="Path to train.csv")
    parser.add_argument("--img_size", type=int, default=512,
                       help="Image size (must match checkpoint)")

    # Analysis options
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "traversals", "density", "tda"],
                       help="Which analysis to run")
    parser.add_argument("--n_samples", type=int, default=500,
                       help="Number of samples for density estimation")
    parser.add_argument("--n_interpolation_steps", type=int, default=10,
                       help="Number of steps in latent traversals")
    parser.add_argument("--enable_tda", action="store_true",
                       help="Enable TDA (requires ripser/persim)")

    # Model architecture overrides (for backwards compatibility with old checkpoints)
    parser.add_argument("--upsample_method", type=str, default=None,
                       choices=["bilinear", "subpixel", "nearest"],
                       help="Override decoder upsample method. Use this when loading old "
                            "checkpoints trained with 'nearest' that don't have this arg saved. "
                            "If not specified, uses checkpoint's value (default: 'nearest' for old ckpts).")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: checkpoint_dir/manifold_analysis)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("SEPVAE LATENT MANIFOLD ANALYSIS")
    print("="*60)

    # Load checkpoint
    vae_params, vae_batch_stats, ckpt_args = load_checkpoint(args.checkpoint)

    # Create model (with optional upsample method override for old checkpoints)
    model = create_model_from_args(ckpt_args, override_upsample=args.upsample_method)
    z_channels_common = ckpt_args.get('z_channels_common', 4)
    z_channels_disease = ckpt_args.get('z_channels_disease', 2)

    # Determine actual upsample method being used
    actual_upsample = args.upsample_method or ckpt_args.get('upsample_method', 'nearest')
    ckpt_upsample = ckpt_args.get('upsample_method', 'not saved (old checkpoint)')

    print(f"\nModel config:")
    print(f"  z_channels_common: {z_channels_common}")
    print(f"  z_channels_disease: {z_channels_disease}")
    print(f"  use_fpn: {ckpt_args.get('use_fpn', False)}")
    print(f"  upsample_method: {actual_upsample} (checkpoint: {ckpt_upsample})")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir = Path(args.checkpoint).parent.parent
        output_dir = str(ckpt_dir / "manifold_analysis")

    ensure_dir(output_dir)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    dataset = VinBigDataTripletDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=jax_collate_fn,
        drop_last=False
    )

    # Run analyses
    if args.analysis in ["all", "traversals"]:
        traversal_dir = ensure_dir(f"{output_dir}/traversals")
        run_traversal_analysis(
            model, vae_params, vae_batch_stats, loader,
            output_dir=traversal_dir,
            n_steps=args.n_interpolation_steps
        )

    if args.analysis in ["all", "density"]:
        density_dir = ensure_dir(f"{output_dir}/density")
        run_density_analysis(
            model, vae_params, vae_batch_stats, loader,
            output_dir=density_dir,
            n_samples=args.n_samples,
            z_channels_common=z_channels_common,
            z_channels_disease=z_channels_disease
        )

    if args.analysis == "tda" or (args.analysis == "all" and args.enable_tda):
        tda_dir = ensure_dir(f"{output_dir}/tda")
        # Extract latents for TDA
        latents, labels = extract_all_latents(
            model, vae_params, vae_batch_stats, loader,
            max_samples=args.n_samples
        )
        run_tda_analysis(latents, labels, output_dir=tda_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
