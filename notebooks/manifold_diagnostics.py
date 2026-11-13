#!/usr/bin/env python3
"""
Manifold Diagnostics and Visualization for Text-to-Image Models

Compares 6 conditions:
  Monolithic:
    1. "A Dog"
    2. "A Cat"
    3. "A Dog And Cat"
    4. "A Dog On The Left And Cat On The Right"
  SuperDiff AND:
    5. "A Dog" AND "A Cat"
    6. "A Dog On The Left" AND "A Cat On The Right"

Produces:
  - 6-row image grid (4 columns per condition)
  - Interactive 2D/3D Plotly plots (PCA, UMAP, t-SNE) with cluster centroids
  - Geometric comparison: dotted lines connecting base concepts
  - Intersection analysis: where do Mono AND vs SuperDiff AND land?

Usage:
    python notebooks/manifold_diagnostics.py
    python notebooks/manifold_diagnostics.py --num-runs 10 --steps 100
"""

import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diffusers import EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

# ---------------------------------------------------------------------------
# Suppress non-actionable diffusers warning
# ---------------------------------------------------------------------------
# warnings.filterwarnings("ignore", message=".*scale_model_input.*")
# warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
# warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ManifoldConfig:
    """Configuration for manifold diagnostics."""
    # Monolithic prompts
    mono_prompts: List[str] = field(default_factory=lambda: [
        "A Dog",
        "A Cat",
        "A Dog And Cat",
        "A Dog On The Left And Cat On The Right",
    ])

    # SuperDiff AND compositions (pairs of prompts)
    superdiff_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("A Dog", "A Cat"),
        ("A Dog On The Left", "A Cat On The Right"),
    ])

    # Generation parameters
    num_runs: int = 10
    num_inference_steps: int = 100
    guidance_scale: float = 7.5
    lift: float = 0.0

    # Display parameters
    display_cols: int = 4  # Number of columns to show in image grid

    # Model
    model_id: str = "Manojb/stable-diffusion-2-1-base"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Output
    output_dir: str = "outputs/sd_2.1_manifold_diagnostics"


# ---------------------------------------------------------------------------
# Condition labels and colors - reorganized for clarity
# ---------------------------------------------------------------------------
CONDITION_STYLES = {
    # Base concepts (will be connected with dotted lines)
    "mono_dog": {
        "label": "Base: A Dog",
        "short": "Dog",
        "color": "#3498db",
        "marker": "circle",
        "is_base": True,
    },
    "mono_cat": {
        "label": "Base: A Cat",
        "short": "Cat",
        "color": "#e67e22",
        "marker": "circle",
        "is_base": True,
    },
    # Monolithic AND (CLIP-based composition)
    "mono_dog_and_cat": {
        "label": "Mono AND: Dog+Cat",
        "short": "Mono AND",
        "color": "#2ecc71",
        "marker": "diamond",
        "is_base": False,
    },
    "mono_spatial": {
        "label": "Mono AND: Spatial",
        "short": "Mono Spatial",
        "color": "#9b59b6",
        "marker": "diamond",
        "is_base": False,
    },
    # SuperDiff AND (stochastic score composition)
    "superdiff_semantic": {
        "label": "SuperDiff AND: Dog+Cat",
        "short": "SD AND",
        "color": "#e74c3c",
        "marker": "star",
        "is_base": False,
    },
    "superdiff_spatial": {
        "label": "SuperDiff AND: Spatial",
        "short": "SD Spatial",
        "color": "#1abc9c",
        "marker": "star",
        "is_base": False,
    },
}

# Semantic groupings for comparison
SEMANTIC_PAIR = {
    "base_a": "mono_dog",
    "base_b": "mono_cat",
    "mono_and": "mono_dog_and_cat",
    "superdiff_and": "superdiff_semantic",
}

SPATIAL_PAIR = {
    "base_a": "mono_dog",  # Shares base concepts
    "base_b": "mono_cat",
    "mono_and": "mono_spatial",
    "superdiff_and": "superdiff_spatial",
}


# ---------------------------------------------------------------------------
# Model loading and utilities
# ---------------------------------------------------------------------------
def load_models(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load Stable Diffusion components."""
    print("Loading Stable Diffusion models...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    print("Models loaded.\n")
    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "scheduler": scheduler,
    }


@torch.no_grad()
def get_text_embedding(prompt, tokenizer, text_encoder, device):
    """Encode text prompt to CLIP embeddings."""
    if isinstance(prompt, str):
        prompt = [prompt]
    text_input = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(device))[0]


@torch.no_grad()
def decode_latents(vae, latents) -> List[Image.Image]:
    """Decode latents to PIL images."""
    latents = latents.to(dtype=vae.dtype)
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    images_tensor = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
    images_tensor = (images_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)

    return [Image.fromarray(img.cpu().numpy()) for img in images_tensor]


# ---------------------------------------------------------------------------
# Monolithic generation (standard CFG)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_monolithic(
    prompt: str,
    models: dict,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    batch_size: int = 4,
    seed: int = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate images using standard classifier-free guidance."""
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "Manojb/stable-diffusion-2-1-base", subfolder="scheduler"
    )

    # Get embeddings
    text_emb = get_text_embedding(prompt, tokenizer, text_encoder, device)
    text_emb = text_emb.repeat(batch_size, 1, 1)
    uncond_emb = get_text_embedding("", tokenizer, text_encoder, device)
    uncond_emb = uncond_emb.repeat(batch_size, 1, 1)

    # Initialize latents
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    latents = torch.randn(
        (batch_size, 4, 64, 64),
        generator=generator, device=device, dtype=dtype
    )

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        latent_input = latents / ((sigma ** 2 + 1) ** 0.5)

        # Unconditional prediction
        noise_uncond = unet(latent_input, t, encoder_hidden_states=uncond_emb).sample

        # Conditional prediction
        noise_cond = unet(latent_input, t, encoder_hidden_states=text_emb).sample

        # CFG
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Euler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# ---------------------------------------------------------------------------
# SuperDiff AND generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_superdiff_and(
    prompt_a: str,
    prompt_b: str,
    models: dict,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    lift: float = 0.0,
    batch_size: int = 4,
    seed: int = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate images using SuperDiff AND composition.

    Returns:
        Tuple of (latents, kappa_trajectory)
    """
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "Manojb/stable-diffusion-2-1-base", subfolder="scheduler"
    )

    # Get embeddings
    emb_a = get_text_embedding([prompt_a] * batch_size, tokenizer, text_encoder, device)
    emb_b = get_text_embedding([prompt_b] * batch_size, tokenizer, text_encoder, device)
    uncond_emb = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    # Initialize latents
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    latents = torch.randn(
        (batch_size, 4, 64, 64),
        generator=generator, device=device, dtype=dtype
    )

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    kappa_history = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    kappa_history[0] = 0.5

    # Denoising loop with SuperDiff
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        denom = torch.sqrt(sigma * sigma + 1.0)
        latent_input = latents / denom

        # Get velocity predictions
        vel_a = unet(latent_input, t, encoder_hidden_states=emb_a).sample
        vel_b = unet(latent_input, t, encoder_hidden_states=emb_b).sample
        vel_uncond = unet(latent_input, t, encoder_hidden_states=uncond_emb).sample

        # Stochastic noise
        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

        # Individual step (using prompt B as base)
        dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_b - vel_uncond)) + noise

        # Compute kappa
        term1 = (torch.abs(dsigma) * (vel_b - vel_a) * (vel_b + vel_a)).sum((1, 2, 3))
        term2 = (dx_ind * (vel_a - vel_b)).sum((1, 2, 3))
        term3 = sigma * lift / num_inference_steps

        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * guidance_scale * ((vel_a - vel_b) ** 2).sum((1, 2, 3))

        kappa = numerator / (denominator + 1e-8)
        kappa_history[i + 1] = kappa

        # Composite vector field
        vf = vel_uncond + guidance_scale * (
            (vel_b - vel_uncond) + kappa[:, None, None, None] * (vel_a - vel_b)
        )

        dx = 2 * dsigma * vf + noise
        latents = latents + dx

    return latents, kappa_history


# ---------------------------------------------------------------------------
# Geometric Analysis Utilities
# ---------------------------------------------------------------------------
def compute_geometric_metrics(centroids: Dict[int, np.ndarray],
                             condition_keys: List[str]) -> Dict[str, float]:
    """Compute geometric metrics for comparing composition methods.

    Returns metrics like:
    - Distance from Mono AND to linear midpoint
    - Distance from SuperDiff AND to linear midpoint
    - Agreement (distance between Mono AND and SuperDiff AND centroids)
    """
    # Map keys to indices
    key_to_idx = {k: i for i, k in enumerate(condition_keys)}

    # Get centroids for base concepts (semantic pair)
    dog_cent = centroids.get(key_to_idx["mono_dog"])
    cat_cent = centroids.get(key_to_idx["mono_cat"])

    if dog_cent is None or cat_cent is None:
        return {}

    # Linear midpoint between Dog and Cat
    midpoint = (dog_cent + cat_cent) / 2

    # Distance between base concepts
    base_dist = np.linalg.norm(dog_cent - cat_cent)

    metrics = {"base_distance": base_dist, "midpoint": midpoint}

    # Mono AND centroid
    mono_and_cent = centroids.get(key_to_idx["mono_dog_and_cat"])
    if mono_and_cent is not None:
        metrics["mono_and_to_midpoint"] = np.linalg.norm(mono_and_cent - midpoint)
        metrics["mono_and_to_dog"] = np.linalg.norm(mono_and_cent - dog_cent)
        metrics["mono_and_to_cat"] = np.linalg.norm(mono_and_cent - cat_cent)

    # SuperDiff AND centroid
    sd_and_cent = centroids.get(key_to_idx["superdiff_semantic"])
    if sd_and_cent is not None:
        metrics["superdiff_to_midpoint"] = np.linalg.norm(sd_and_cent - midpoint)
        metrics["superdiff_to_dog"] = np.linalg.norm(sd_and_cent - dog_cent)
        metrics["superdiff_to_cat"] = np.linalg.norm(sd_and_cent - cat_cent)

    # Agreement: distance between Mono AND and SuperDiff AND
    if mono_and_cent is not None and sd_and_cent is not None:
        metrics["agreement_distance"] = np.linalg.norm(mono_and_cent - sd_and_cent)
        # Normalized agreement (0 = perfect agreement, 1 = as far as base concepts)
        metrics["agreement_normalized"] = metrics["agreement_distance"] / (base_dist + 1e-8)

    return metrics


# ---------------------------------------------------------------------------
# Main Diagnostics Class
# ---------------------------------------------------------------------------
class ManifoldDiagnostics:
    """Comprehensive manifold analysis comparing monolithic vs SuperDiff."""

    def __init__(self, config: ManifoldConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        self.models = load_models(config.model_id, self.device, self.dtype)

        # Storage for results
        self.results: Dict[str, Dict] = {}

        # Condition keys in order
        self.condition_keys = [
            "mono_dog", "mono_cat", "mono_dog_and_cat", "mono_spatial",
            "superdiff_semantic", "superdiff_spatial"
        ]

        # Storage for projections (for reuse across visualizations)
        self._projections = None
        self._all_latents = None
        self._condition_indices = None

    def run_all_generations(self):
        """Run all 6 conditions for num_runs iterations."""
        print("\n" + "=" * 80)
        print("RUNNING ALL GENERATIONS")
        print("=" * 80)

        for key in self.condition_keys:
            self.results[key] = {"latents": [], "images": [], "kappa": None}

        # Run multiple iterations
        for run_idx in range(self.config.num_runs):
            seed = 42 + run_idx
            print(f"\n--- Run {run_idx + 1}/{self.config.num_runs} (seed={seed}) ---")

            # Monolithic generations
            for i, prompt in enumerate(self.config.mono_prompts):
                key = self.condition_keys[i]
                print(f"  {key}: '{prompt[:40]}...'")

                latents = generate_monolithic(
                    prompt, self.models,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    batch_size=1,
                    seed=seed,
                    device=self.device,
                    dtype=self.dtype,
                )
                images = decode_latents(self.models["vae"], latents)

                self.results[key]["latents"].append(latents.cpu())
                self.results[key]["images"].extend(images)

            # SuperDiff AND generations
            for j, (prompt_a, prompt_b) in enumerate(self.config.superdiff_pairs):
                key = self.condition_keys[4 + j]
                print(f"  {key}: '{prompt_a}' AND '{prompt_b}'")

                latents, kappa = generate_superdiff_and(
                    prompt_a, prompt_b, self.models,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    lift=self.config.lift,
                    batch_size=1,
                    seed=seed,
                    device=self.device,
                    dtype=self.dtype,
                )
                images = decode_latents(self.models["vae"], latents)

                self.results[key]["latents"].append(latents.cpu())
                self.results[key]["images"].extend(images)
                self.results[key]["kappa"] = kappa.cpu()

        print(f"\n  Completed {self.config.num_runs} runs for all 6 conditions")

    def create_image_grid(self, output_dir: str):
        """Create 6-row image grid with display_cols columns."""
        print("\n" + "=" * 80)
        print("CREATING IMAGE GRID")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_cols = min(self.config.display_cols, self.config.num_runs)
        n_rows = 6

        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        gs = GridSpec(n_rows, n_cols + 1, width_ratios=[0.18] + [1] * n_cols,
                      wspace=0.05, hspace=0.1)

        row_labels = [
            "Base:\nA Dog",
            "Base:\nA Cat",
            "Mono AND:\nDog + Cat",
            "Mono AND:\nSpatial",
            "SuperDiff:\nDog AND Cat",
            "SuperDiff:\nSpatial",
        ]

        for row_idx, key in enumerate(self.condition_keys):
            style = CONDITION_STYLES[key]

            # Row label with color indicator
            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.text(0.5, 0.5, row_labels[row_idx],
                         ha="center", va="center", fontsize=10, fontweight="bold",
                         color=style["color"],
                         transform=ax_label.transAxes)
            ax_label.axis("off")

            # Images
            images = self.results[key]["images"][:n_cols]
            for col_idx, img in enumerate(images):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                ax.imshow(img)
                ax.axis("off")

                if row_idx == 0:
                    ax.set_title(f"Run {col_idx + 1}", fontsize=10)

        plt.suptitle("Monolithic vs SuperDiff AND Comparison",
                    fontsize=16, fontweight="bold", y=0.98)

        save_path = output_path / "image_grid_6x4.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
        plt.close()

    def _collect_latents_for_projection(self) -> Tuple[np.ndarray, List[str], List[int]]:
        """Flatten all latents and return with condition labels."""
        all_latents = []
        condition_labels = []
        condition_indices = []

        for cond_idx, key in enumerate(self.condition_keys):
            latent_list = self.results[key]["latents"]
            for latent in latent_list:
                flat = latent.flatten().numpy()
                all_latents.append(flat)
                condition_labels.append(key)
                condition_indices.append(cond_idx)

        return np.array(all_latents), condition_labels, condition_indices

    def _compute_centroids(self, projections: np.ndarray,
                           condition_indices: List[int]) -> Dict[int, np.ndarray]:
        """Compute centroids for each condition cluster."""
        centroids = {}
        for cond_idx in range(6):
            mask = np.array(condition_indices) == cond_idx
            if mask.sum() > 0:
                centroids[cond_idx] = projections[mask].mean(axis=0)
        return centroids

    def _compute_all_projections(self):
        """Compute and cache all projections."""
        if self._projections is not None:
            return

        all_latents, _, condition_indices = self._collect_latents_for_projection()
        self._all_latents = all_latents
        self._condition_indices = condition_indices

        print("\nComputing PCA...")
        pca_3d = PCA(n_components=3).fit_transform(all_latents)

        print("Computing t-SNE...")
        tsne_2d = TSNE(n_components=2, perplexity=min(30, len(all_latents) - 1),
                       random_state=42).fit_transform(all_latents)
        tsne_3d = TSNE(n_components=3, perplexity=min(30, len(all_latents) - 1),
                       random_state=42).fit_transform(all_latents)

        print("Computing UMAP...")
        umap_2d = umap.UMAP(n_neighbors=min(15, len(all_latents) - 1),
                           n_components=2, random_state=42).fit_transform(all_latents)
        umap_3d = umap.UMAP(n_neighbors=min(15, len(all_latents) - 1),
                           n_components=3, random_state=42).fit_transform(all_latents)

        self._projections = {
            "PCA": {"2d": pca_3d[:, :2], "3d": pca_3d},
            "t-SNE": {"2d": tsne_2d, "3d": tsne_3d},
            "UMAP": {"2d": umap_2d, "3d": umap_3d},
        }

    def create_geometric_comparison(self, output_dir: str):
        """Create the key geometric comparison visualization.

        This is the main interpretive plot showing:
        - Base concepts connected by dotted lines
        - Linear midpoint marked
        - Mono AND vs SuperDiff AND positions
        - Distance annotations
        """
        print("\n" + "=" * 80)
        print("CREATING GEOMETRIC COMPARISON")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._compute_all_projections()
        condition_indices = self._condition_indices

        # Create figure with 3 columns (PCA, t-SNE, UMAP)
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        methods = ["PCA", "t-SNE", "UMAP"]
        all_metrics = {}

        for ax, method in zip(axes, methods):
            proj = self._projections[method]["2d"]
            centroids = self._compute_centroids(proj, condition_indices)
            metrics = compute_geometric_metrics(centroids, self.condition_keys)
            all_metrics[method] = metrics

            key_to_idx = {k: i for i, k in enumerate(self.condition_keys)}

            # 1. Draw dotted line connecting base concepts (Dog <-> Cat)
            dog_cent = centroids.get(key_to_idx["mono_dog"])
            cat_cent = centroids.get(key_to_idx["mono_cat"])

            if dog_cent is not None and cat_cent is not None:
                ax.plot([dog_cent[0], cat_cent[0]], [dog_cent[1], cat_cent[1]],
                       'k:', linewidth=2, alpha=0.6, label="Base axis", zorder=1)

                # Mark linear midpoint
                midpoint = metrics.get("midpoint")
                if midpoint is not None:
                    ax.scatter(midpoint[0], midpoint[1],
                              c="gray", s=150, marker="X", edgecolors="black",
                              linewidths=2, zorder=10, label="Linear midpoint")

            # 2. Plot base concepts (larger, emphasized)
            for key in ["mono_dog", "mono_cat"]:
                style = CONDITION_STYLES[key]
                idx = key_to_idx[key]
                mask = np.array(condition_indices) == idx
                points = proj[mask]

                ax.scatter(points[:, 0], points[:, 1],
                          c=style["color"], s=80, alpha=0.5,
                          edgecolors="white", linewidths=0.5, zorder=2)

                # Centroid
                if idx in centroids:
                    cent = centroids[idx]
                    ax.scatter(cent[0], cent[1],
                              c=style["color"], s=300, marker="o",
                              edgecolors="black", linewidths=2, zorder=11,
                              label=style["short"])

            # 3. Plot Mono AND (diamond marker)
            mono_and_key = "mono_dog_and_cat"
            style = CONDITION_STYLES[mono_and_key]
            idx = key_to_idx[mono_and_key]
            mask = np.array(condition_indices) == idx
            points = proj[mask]

            ax.scatter(points[:, 0], points[:, 1],
                      c=style["color"], s=60, alpha=0.5,
                      edgecolors="white", linewidths=0.5, zorder=3)

            if idx in centroids:
                cent = centroids[idx]
                ax.scatter(cent[0], cent[1],
                          c=style["color"], s=350, marker="D",
                          edgecolors="black", linewidths=2, zorder=12,
                          label=style["short"])

                # Draw line from midpoint to Mono AND centroid
                if midpoint is not None:
                    ax.plot([midpoint[0], cent[0]], [midpoint[1], cent[1]],
                           color=style["color"], linestyle="--", linewidth=1.5,
                           alpha=0.7, zorder=5)

            # 4. Plot SuperDiff AND (star marker)
            sd_and_key = "superdiff_semantic"
            style = CONDITION_STYLES[sd_and_key]
            idx = key_to_idx[sd_and_key]
            mask = np.array(condition_indices) == idx
            points = proj[mask]

            ax.scatter(points[:, 0], points[:, 1],
                      c=style["color"], s=60, alpha=0.5,
                      edgecolors="white", linewidths=0.5, zorder=4)

            if idx in centroids:
                cent = centroids[idx]
                ax.scatter(cent[0], cent[1],
                          c=style["color"], s=400, marker="*",
                          edgecolors="black", linewidths=1.5, zorder=13,
                          label=style["short"])

                # Draw line from midpoint to SuperDiff AND centroid
                if midpoint is not None:
                    ax.plot([midpoint[0], cent[0]], [midpoint[1], cent[1]],
                           color=style["color"], linestyle="--", linewidth=1.5,
                           alpha=0.7, zorder=5)

            # Styling
            ax.set_xlabel(f"{method} 1", fontsize=11)
            ax.set_ylabel(f"{method} 2", fontsize=11)
            ax.set_title(method, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(
            "Geometric Comparison: Where do Mono AND and SuperDiff AND intersect?",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        save_path = output_path / "geometric_comparison.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
        plt.close()

        # Save metrics
        self._save_geometric_report(all_metrics, output_path)

        return all_metrics

    def _save_geometric_report(self, all_metrics: Dict, output_path: Path):
        """Save a text report with geometric analysis."""
        report_lines = [
            "=" * 70,
            "GEOMETRIC ANALYSIS REPORT",
            "=" * 70,
            "",
            "Question: Do Mono AND and SuperDiff AND agree on where the",
            "          semantic intersection lies?",
            "",
        ]

        for method, metrics in all_metrics.items():
            report_lines.append(f"--- {method} ---")

            if not metrics:
                report_lines.append("  (insufficient data)")
                continue

            base_dist = metrics.get("base_distance", 0)
            report_lines.append(f"  Base concept distance (Dog <-> Cat): {base_dist:.3f}")

            mono_mid = metrics.get("mono_and_to_midpoint")
            sd_mid = metrics.get("superdiff_to_midpoint")

            if mono_mid is not None:
                report_lines.append(f"  Mono AND distance to midpoint:      {mono_mid:.3f}")
            if sd_mid is not None:
                report_lines.append(f"  SuperDiff AND distance to midpoint: {sd_mid:.3f}")

            agreement = metrics.get("agreement_distance")
            agreement_norm = metrics.get("agreement_normalized")

            if agreement is not None:
                report_lines.append(f"  Agreement (Mono AND <-> SD AND):    {agreement:.3f}")
                report_lines.append(f"  Agreement (normalized):             {agreement_norm:.3f}")

                # Interpretation
                if agreement_norm < 0.2:
                    interp = "STRONG AGREEMENT - methods land in similar region"
                elif agreement_norm < 0.5:
                    interp = "MODERATE AGREEMENT - some divergence"
                else:
                    interp = "WEAK AGREEMENT - methods diverge significantly"
                report_lines.append(f"  Interpretation: {interp}")

            report_lines.append("")

        # Summary
        report_lines.extend([
            "=" * 70,
            "INTERPRETATION GUIDE",
            "=" * 70,
            "",
            "1. If BOTH methods land NEAR the linear midpoint:",
            "   -> Composition acts as simple averaging/interpolation",
            "   -> Both find the same 'intersection' in latent space",
            "",
            "2. If Mono AND is near midpoint but SuperDiff is NOT:",
            "   -> CLIP-based composition interpolates embeddings",
            "   -> SuperDiff explores different manifold regions",
            "   -> SuperDiff may find off-manifold or novel intersections",
            "",
            "3. If SuperDiff is near midpoint but Mono AND is NOT:",
            "   -> CLIP prompt 'Dog and Cat' has distinct semantics",
            "   -> SuperDiff score composition approximates linear blend",
            "",
            "4. If NEITHER is near midpoint:",
            "   -> Both methods find semantic intersections",
            "   -> 'AND' means something beyond linear interpolation",
            "",
            "5. If agreement is LOW (normalized > 0.5):",
            "   -> Methods fundamentally disagree on composition",
            "   -> Worth investigating which produces better results",
        ])

        report_path = output_path / "geometric_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Saved: {report_path}")

    def create_interactive_plots(self, output_dir: str):
        """Create interactive 2D and 3D Plotly plots with geometric annotations."""
        print("\n" + "=" * 80)
        print("CREATING INTERACTIVE PROJECTIONS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._compute_all_projections()
        condition_indices = self._condition_indices

        print(f"Total samples: {len(self._all_latents)} ({self.config.num_runs} runs x 6 conditions)")

        # Create 2D interactive plot with geometric annotations
        self._create_2d_interactive_geometric(output_path)

        # Create 3D interactive plots
        for method_name in ["PCA", "t-SNE", "UMAP"]:
            self._create_3d_interactive(
                self._projections[method_name]["3d"],
                condition_indices,
                method_name,
                output_path
            )

        print("\n  Interactive plots saved")

    def _create_2d_interactive_geometric(self, output_path: Path):
        """Create combined 2D interactive plot with geometric annotations."""
        condition_indices = self._condition_indices

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "<b>PCA</b><br><sup>Principal Component Analysis</sup>",
                "<b>t-SNE</b><br><sup>t-Distributed Stochastic Neighbor Embedding</sup>",
                "<b>UMAP</b><br><sup>Uniform Manifold Approximation</sup>",
            ],
            horizontal_spacing=0.06,
        )

        methods = ["PCA", "t-SNE", "UMAP"]
        key_to_idx = {k: i for i, k in enumerate(self.condition_keys)}

        for col_idx, method in enumerate(methods, start=1):
            proj_2d = self._projections[method]["2d"]
            centroids = self._compute_centroids(proj_2d, condition_indices)

            # Get base concept centroids
            dog_cent = centroids.get(key_to_idx["mono_dog"])
            cat_cent = centroids.get(key_to_idx["mono_cat"])

            if dog_cent is not None and cat_cent is not None:
                midpoint = (dog_cent + cat_cent) / 2
                base_dist = np.linalg.norm(dog_cent - cat_cent)

                # 1. DOTTED LINE: Base concept axis (Dog <-> Cat)
                fig.add_trace(
                    go.Scatter(
                        x=[dog_cent[0], cat_cent[0]],
                        y=[dog_cent[1], cat_cent[1]],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0.5)", width=3, dash="dot"),
                        name="Base Axis (Dog ↔ Cat)" if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        legendgroup="base_axis",
                        legendgrouptitle_text="GEOMETRY" if col_idx == 1 else None,
                        hovertemplate=f"Base Concept Axis<br>Distance: {base_dist:.3f}<extra></extra>",
                    ),
                    row=1, col=col_idx
                )

                # 2. Linear midpoint marker with label
                fig.add_trace(
                    go.Scatter(
                        x=[midpoint[0]], y=[midpoint[1]],
                        mode="markers+text",
                        marker=dict(size=16, color="gray", symbol="x",
                                   line=dict(width=3, color="black")),
                        text=["Midpoint"],
                        textposition="top center",
                        textfont=dict(size=9, color="gray"),
                        name="Linear Midpoint" if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        legendgroup="midpoint",
                        hovertemplate="<b>Linear Midpoint</b><br>Halfway between Dog & Cat<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
                    ),
                    row=1, col=col_idx
                )

            # 3. Plot BASE CONCEPTS first (Dog, Cat)
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES[key]
                if not style.get("is_base"):
                    continue

                mask = np.array(condition_indices) == cond_idx
                points = proj_2d[mask]

                # Sample points (no legend entry)
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0], y=points[:, 1],
                        mode="markers",
                        name=style["label"],
                        showlegend=False,
                        legendgroup=key,
                        marker=dict(
                            size=10,
                            color=style["color"],
                            opacity=0.5,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra>Sample</extra>",
                    ),
                    row=1, col=col_idx
                )

                # Centroid with text label (show in legend)
                if cond_idx in centroids:
                    cent = centroids[cond_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[cent[0]], y=[cent[1]],
                            mode="markers+text",
                            name=f"● Centroid: {style['short']}" if col_idx == 1 else None,
                            showlegend=(col_idx == 1),
                            legendgroup=key,
                            marker=dict(
                                size=22,
                                color=style["color"],
                                symbol="circle",
                                line=dict(width=3, color="black"),
                            ),
                            text=[style["short"]],
                            textposition="bottom center",
                            textfont=dict(size=10, color=style["color"], family="Arial Black"),
                            hovertemplate=f"<b>{style['label']} CENTROID</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra>Cluster Center</extra>",
                        ),
                        row=1, col=col_idx
                    )

            # 4. Plot CLIP AND compositions (diamonds)
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES[key]
                if style.get("is_base") or "superdiff" in key:
                    continue

                mask = np.array(condition_indices) == cond_idx
                points = proj_2d[mask]

                # Sample points (no legend entry)
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0], y=points[:, 1],
                        mode="markers",
                        name=style["label"],
                        showlegend=False,
                        legendgroup=key,
                        marker=dict(
                            size=9,
                            color=style["color"],
                            opacity=0.6,
                            symbol="diamond",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra>Sample</extra>",
                    ),
                    row=1, col=col_idx
                )

                if cond_idx in centroids and dog_cent is not None and cat_cent is not None:
                    cent = centroids[cond_idx]
                    midpoint = (dog_cent + cat_cent) / 2
                    dist_to_mid = np.linalg.norm(cent - midpoint)

                    # Dashed line from midpoint to centroid
                    fig.add_trace(
                        go.Scatter(
                            x=[midpoint[0], cent[0]],
                            y=[midpoint[1], cent[1]],
                            mode="lines",
                            line=dict(color=style["color"], width=2, dash="dash"),
                            showlegend=False,
                            hovertemplate=f"Distance to midpoint: {dist_to_mid:.3f}<extra></extra>",
                        ),
                        row=1, col=col_idx
                    )

                    # Centroid marker (show in legend)
                    fig.add_trace(
                        go.Scatter(
                            x=[cent[0]], y=[cent[1]],
                            mode="markers+text",
                            name=f"◆ Centroid: {style['short']}" if col_idx == 1 else None,
                            showlegend=(col_idx == 1),
                            legendgroup=key,
                            marker=dict(
                                size=20,
                                color=style["color"],
                                symbol="diamond",
                                line=dict(width=3, color="black"),
                            ),
                            text=[style["short"]],
                            textposition="top center",
                            textfont=dict(size=10, color=style["color"], family="Arial Black"),
                            hovertemplate=f"<b>{style['label']} CENTROID</b><br>Distance to midpoint: {dist_to_mid:.3f}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
                        ),
                        row=1, col=col_idx
                    )

            # 5. Plot SUPERDIFF AND compositions (stars)
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES[key]
                if "superdiff" not in key:
                    continue

                mask = np.array(condition_indices) == cond_idx
                points = proj_2d[mask]

                # Sample points (no legend entry)
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0], y=points[:, 1],
                        mode="markers",
                        name=style["label"],
                        showlegend=False,
                        legendgroup=key,
                        marker=dict(
                            size=10,
                            color=style["color"],
                            opacity=0.6,
                            symbol="star",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra>Sample</extra>",
                    ),
                    row=1, col=col_idx
                )

                if cond_idx in centroids and dog_cent is not None and cat_cent is not None:
                    cent = centroids[cond_idx]
                    midpoint = (dog_cent + cat_cent) / 2
                    dist_to_mid = np.linalg.norm(cent - midpoint)

                    # Dashed line from midpoint to centroid
                    fig.add_trace(
                        go.Scatter(
                            x=[midpoint[0], cent[0]],
                            y=[midpoint[1], cent[1]],
                            mode="lines",
                            line=dict(color=style["color"], width=2, dash="dash"),
                            showlegend=False,
                            hovertemplate=f"Distance to midpoint: {dist_to_mid:.3f}<extra></extra>",
                        ),
                        row=1, col=col_idx
                    )

                    # Centroid marker (show in legend)
                    fig.add_trace(
                        go.Scatter(
                            x=[cent[0]], y=[cent[1]],
                            mode="markers+text",
                            name=f"★ Centroid: {style['short']}" if col_idx == 1 else None,
                            showlegend=(col_idx == 1),
                            legendgroup=key,
                            marker=dict(
                                size=24,
                                color=style["color"],
                                symbol="star",
                                line=dict(width=3, color="black"),
                            ),
                            text=[style["short"]],
                            textposition="top center",
                            textfont=dict(size=10, color=style["color"], family="Arial Black"),
                            hovertemplate=f"<b>{style['label']} CENTROID</b><br>Distance to midpoint: {dist_to_mid:.3f}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
                        ),
                        row=1, col=col_idx
                    )

            # Update axes labels
            fig.update_xaxes(title_text=f"{method} Component 1", row=1, col=col_idx)
            fig.update_yaxes(title_text=f"{method} Component 2", row=1, col=col_idx)

        # Layout with improved legend
        fig.update_layout(
            title=dict(
                text="<b>Latent Space Geometry: CLIP AND vs SuperDiff AND</b><br><sup>Dotted line = base concept axis | Dashed lines = distance to linear midpoint</sup>",
                font=dict(size=18),
                x=0.5,
            ),
            width=1900,
            height=700,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=10),
                tracegroupgap=10,
            ),
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=11, font_family="Arial"),
        )

        # Add explanatory annotation
        fig.add_annotation(
            text="<b>Legend:</b> ● Base concepts | ◆ CLIP AND | ★ SuperDiff AND | ✕ Linear midpoint",
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            showarrow=False,
            font=dict(size=11, color="gray"),
        )

        save_path = output_path / "projections_2d_interactive.html"
        fig.write_html(str(save_path), include_plotlyjs=True, full_html=True)
        print(f"Saved: {save_path}")

        try:
            fig.write_image(str(output_path / "projections_2d.png"), scale=2)
            print(f"Saved: {output_path / 'projections_2d.png'}")
        except Exception as e:
            print(f"Warning: Could not save static image: {e}")

    def _create_3d_interactive(self, proj_3d: np.ndarray, condition_indices: List[int],
                               method_name: str, output_path: Path):
        """Create 3D interactive plot for a single projection method."""
        fig = go.Figure()

        centroids = self._compute_centroids(proj_3d, condition_indices)
        key_to_idx = {k: i for i, k in enumerate(self.condition_keys)}

        # Get base concept centroids
        dog_cent = centroids.get(key_to_idx["mono_dog"])
        cat_cent = centroids.get(key_to_idx["mono_cat"])
        midpoint = None
        base_dist = 0

        if dog_cent is not None and cat_cent is not None:
            midpoint = (dog_cent + cat_cent) / 2
            base_dist = np.linalg.norm(dog_cent - cat_cent)

            # 1. DOTTED LINE: Base concept axis (Dog <-> Cat)
            # Create dotted effect with multiple short segments
            n_segments = 20
            for i in range(n_segments):
                if i % 2 == 0:  # Only draw even segments for dotted effect
                    t1 = i / n_segments
                    t2 = (i + 1) / n_segments
                    p1 = dog_cent + t1 * (cat_cent - dog_cent)
                    p2 = dog_cent + t2 * (cat_cent - dog_cent)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[p1[0], p2[0]],
                            y=[p1[1], p2[1]],
                            z=[p1[2], p2[2]],
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0.6)", width=6),
                            showlegend=False,  # Legend added separately
                            legendgroup="base_axis",
                            hoverinfo="skip",
                        )
                    )

            # 2. Linear midpoint marker
            fig.add_trace(
                go.Scatter3d(
                    x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
                    mode="markers+text",
                    marker=dict(size=10, color="gray", symbol="x",
                               line=dict(width=2, color="black")),
                    text=["Midpoint"],
                    textposition="top center",
                    textfont=dict(size=10, color="gray"),
                    showlegend=False,  # Legend added separately
                    legendgroup="midpoint",
                    hovertemplate="<b>Linear Midpoint</b><br>Halfway between Dog & Cat<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
                )
            )

        # 3. Plot BASE CONCEPTS (Dog, Cat)
        for cond_idx, key in enumerate(self.condition_keys):
            style = CONDITION_STYLES[key]
            if not style.get("is_base"):
                continue

            mask = np.array(condition_indices) == cond_idx
            points = proj_3d[mask]

            # Sample points (no legend entry)
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers",
                    name=style["label"],
                    showlegend=False,
                    legendgroup=key,
                    marker=dict(
                        size=5,
                        color=style["color"],
                        opacity=0.5,
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra>Sample</extra>",
                )
            )

            if cond_idx in centroids:
                cent = centroids[cond_idx]
                fig.add_trace(
                    go.Scatter3d(
                        x=[cent[0]], y=[cent[1]], z=[cent[2]],
                        mode="markers+text",
                        name=f"⬤  Centroid: {style['short']}",
                        showlegend=True,
                        legendgroup=f"{key}_centroid",
                        marker=dict(
                            size=12,
                            color=style["color"],
                            symbol="circle",
                            line=dict(width=2, color="black"),
                        ),
                        text=[style["short"]],
                        textposition="top center",
                        textfont=dict(size=11, color=style["color"]),
                        hovertemplate=f"<b>{style['label']} CENTROID</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
                    )
                )

        # 4. Plot CLIP AND compositions
        for cond_idx, key in enumerate(self.condition_keys):
            style = CONDITION_STYLES[key]
            if style.get("is_base") or "superdiff" in key:
                continue

            mask = np.array(condition_indices) == cond_idx
            points = proj_3d[mask]

            # Sample points (no legend entry)
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers",
                    name=style["label"],
                    showlegend=False,
                    legendgroup=key,
                    marker=dict(
                        size=5,
                        color=style["color"],
                        opacity=0.6,
                        symbol="diamond",
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra>Sample</extra>",
                )
            )

            if cond_idx in centroids and midpoint is not None:
                cent = centroids[cond_idx]
                dist_to_mid = np.linalg.norm(cent - midpoint)

                # Dashed line from midpoint to centroid (using segments, no legend)
                n_dash = 8
                for i in range(n_dash):
                    if i % 2 == 0:
                        t1 = i / n_dash
                        t2 = (i + 1) / n_dash
                        p1 = midpoint + t1 * (cent - midpoint)
                        p2 = midpoint + t2 * (cent - midpoint)
                        fig.add_trace(
                            go.Scatter3d(
                                x=[p1[0], p2[0]],
                                y=[p1[1], p2[1]],
                                z=[p1[2], p2[2]],
                                mode="lines",
                                line=dict(color=style["color"], width=4),
                                showlegend=False,
                                legendgroup=f"{key}_line",
                                hoverinfo="skip",
                            )
                        )

                # Centroid marker (show in legend)
                fig.add_trace(
                    go.Scatter3d(
                        x=[cent[0]], y=[cent[1]], z=[cent[2]],
                        mode="markers+text",
                        name=f"◆ Centroid: {style['short']}",
                        showlegend=True,
                        legendgroup=f"{key}_centroid",
                        marker=dict(
                            size=12,
                            color=style["color"],
                            symbol="diamond",
                            line=dict(width=2, color="black"),
                        ),
                        text=[style["short"]],
                        textposition="top center",
                        textfont=dict(size=11, color=style["color"]),
                        hovertemplate=f"<b>{style['label']} CENTROID</b><br>Distance to midpoint: {dist_to_mid:.3f}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
                    )
                )

        # 5. Plot SUPERDIFF AND compositions
        for cond_idx, key in enumerate(self.condition_keys):
            style = CONDITION_STYLES[key]
            if "superdiff" not in key:
                continue

            mask = np.array(condition_indices) == cond_idx
            points = proj_3d[mask]

            # Sample points (no legend entry)
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers",
                    name=style["label"],
                    showlegend=False,
                    legendgroup=key,
                    marker=dict(
                        size=6,
                        color=style["color"],
                        opacity=0.6,
                        symbol="diamond",  # Using diamond as star not well supported in 3D
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate=f"<b>{style['label']}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra>Sample</extra>",
                )
            )

            if cond_idx in centroids and midpoint is not None:
                cent = centroids[cond_idx]
                dist_to_mid = np.linalg.norm(cent - midpoint)

                # Dashed line from midpoint to centroid (no legend)
                n_dash = 8
                for i in range(n_dash):
                    if i % 2 == 0:
                        t1 = i / n_dash
                        t2 = (i + 1) / n_dash
                        p1 = midpoint + t1 * (cent - midpoint)
                        p2 = midpoint + t2 * (cent - midpoint)
                        fig.add_trace(
                            go.Scatter3d(
                                x=[p1[0], p2[0]],
                                y=[p1[1], p2[1]],
                                z=[p1[2], p2[2]],
                                mode="lines",
                                line=dict(color=style["color"], width=4),
                                showlegend=False,
                                legendgroup=f"{key}_line",
                                hoverinfo="skip",
                            )
                        )

                # Centroid marker (show in legend)
                fig.add_trace(
                    go.Scatter3d(
                        x=[cent[0]], y=[cent[1]], z=[cent[2]],
                        mode="markers+text",
                        name=f"★ Centroid: {style['short']}",
                        showlegend=True,
                        legendgroup=f"{key}_centroid",
                        marker=dict(
                            size=14,
                            color=style["color"],
                            symbol="diamond",
                            line=dict(width=2, color="black"),
                        ),
                        text=[style["short"]],
                        textposition="top center",
                        textfont=dict(size=11, color=style["color"]),
                        hovertemplate=f"<b>{style['label']} CENTROID</b><br>Distance to midpoint: {dist_to_mid:.3f}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
                    )
                )

        # Add legend entry for linear midpoint (centroids-only legend)
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color="gray", symbol="x"),
                name="✕ Linear Midpoint",
                showlegend=True,
                legendgroup="legend_midpoint",
            )
        )

        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>{method_name} 3D: CLIP AND vs SuperDiff AND</b><br><sup>Dotted line = base concept axis | Dashed lines = distance to midpoint</sup>",
                font=dict(size=16),
                x=0.5,
            ),
            width=1200,
            height=850,
            scene=dict(
                xaxis_title=f"{method_name} Component 1",
                yaxis_title=f"{method_name} Component 2",
                zaxis_title=f"{method_name} Component 3",
                xaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="lightgray"),
                yaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="lightgray"),
                zaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="lightgray"),
                domain=dict(x=[0, 0.75]),  # Make room for legend
            ),
            legend=dict(
                title=dict(text="<b>LEGEND</b>", font=dict(size=12)),
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.78,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=11),
                itemsizing="constant",
                tracegroupgap=5,
            ),
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=11),
            margin=dict(r=250),  # Right margin for legend
        )

        # Add annotation
        fig.add_annotation(
            text="<b>Tip:</b> Click and drag to rotate | Scroll to zoom | Double-click to reset",
            xref="paper", yref="paper",
            x=0.5, y=-0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        save_path = output_path / f"projection_3d_{method_name.lower().replace('-', '')}_interactive.html"
        fig.write_html(str(save_path), include_plotlyjs=True, full_html=True)
        print(f"Saved: {save_path}")

    def create_static_projections(self, output_dir: str):
        """Create static matplotlib projections (backup)."""
        print("\n" + "=" * 80)
        print("CREATING STATIC PROJECTIONS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._compute_all_projections()
        condition_indices = self._condition_indices

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        methods = ["PCA", "t-SNE", "UMAP"]

        for ax, method in zip(axes, methods):
            proj = self._projections[method]["2d"]
            centroids = self._compute_centroids(proj, condition_indices)
            key_to_idx = {k: i for i, k in enumerate(self.condition_keys)}

            # Dotted line between base concepts
            dog_cent = centroids.get(key_to_idx["mono_dog"])
            cat_cent = centroids.get(key_to_idx["mono_cat"])

            if dog_cent is not None and cat_cent is not None:
                ax.plot([dog_cent[0], cat_cent[0]], [dog_cent[1], cat_cent[1]],
                       'k:', linewidth=2, alpha=0.6, zorder=1)

                midpoint = (dog_cent + cat_cent) / 2
                ax.scatter(midpoint[0], midpoint[1],
                          c="gray", s=120, marker="X", edgecolors="black",
                          linewidths=2, zorder=10, label="Midpoint")

            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES[key]
                mask = np.array(condition_indices) == cond_idx
                points = proj[mask]

                ax.scatter(points[:, 0], points[:, 1],
                          c=style["color"], s=50, alpha=0.5,
                          edgecolors="white", linewidths=0.5,
                          label=style["short"])

                if cond_idx in centroids:
                    cent = centroids[cond_idx]
                    marker = "D" if not style.get("is_base") else "o"
                    ax.scatter(cent[0], cent[1],
                              c=style["color"], s=200, marker=marker,
                              edgecolors="black", linewidths=2, zorder=11)

            ax.set_xlabel(f"{method} 1", fontsize=11)
            ax.set_ylabel(f"{method} 2", fontsize=11)
            ax.set_title(method, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4,
                  bbox_to_anchor=(0.5, -0.02), fontsize=9)

        plt.suptitle("Latent Space Projections",
                    fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = output_path / "projections_2d_static.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
        plt.close()

    def run_full_analysis(self, include_enhanced: bool = True):
        """Run the complete analysis pipeline.

        Args:
            include_enhanced: If True, also generate enhanced visualizations
        """
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("MANIFOLD DIAGNOSTICS: MONOLITHIC vs SUPERDIFF AND")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Runs: {self.config.num_runs}")
        print(f"  Steps: {self.config.num_inference_steps}")
        print(f"  Output: {self.config.output_dir}")
        print("=" * 80)

        # 1. Run all generations
        self.run_all_generations()

        # 2. Create image grid
        self.create_image_grid(self.config.output_dir)

        # 3. Create geometric comparison (key visualization)
        metrics = self.create_geometric_comparison(self.config.output_dir)

        # 4. Create interactive plots
        self.create_interactive_plots(self.config.output_dir)

        # 5. Create static plots as backup
        self.create_static_projections(self.config.output_dir)

        # 6. Create enhanced visualizations (optional)
        if include_enhanced:
            self._create_enhanced_visualizations(self.config.output_dir)

        # Summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {output_path.absolute()}")
        print("\nKey files:")
        print(f"  1. {output_path / 'geometric_comparison.png'}")
        print(f"     -> THE KEY PLOT: Shows where compositions land relative to midpoint")
        print(f"  2. {output_path / 'geometric_analysis_report.txt'}")
        print(f"     -> Quantitative analysis of agreement/divergence")
        print(f"  3. {output_path / 'image_grid_6x4.png'}")
        print(f"     -> Visual comparison of generated images")
        print(f"  4. {output_path / 'projections_2d_interactive.html'}")
        print(f"     -> Interactive exploration (open in browser)")
        if include_enhanced:
            print(f"  5. {output_path / 'projections_annotated_interactive.html'}")
            print(f"     -> Fully annotated interactive plot with legends")
            print(f"  6. {output_path / 'hypothesis_visualization.png'}")
            print(f"     -> Hypothesis-specific analysis")

        # Print key insight
        print("\n" + "=" * 80)
        print("KEY QUESTION ANSWERED:")
        print("=" * 80)

        pca_metrics = metrics.get("PCA", {})
        agreement = pca_metrics.get("agreement_normalized")

        if agreement is not None:
            if agreement < 0.2:
                conclusion = "AGREE - Both methods find similar intersection region"
            elif agreement < 0.5:
                conclusion = "PARTIAL AGREEMENT - Some divergence in composition"
            else:
                conclusion = "DISAGREE - Methods find different intersections"

            mono_mid = pca_metrics.get("mono_and_to_midpoint", 0)
            sd_mid = pca_metrics.get("superdiff_to_midpoint", 0)
            base_dist = pca_metrics.get("base_distance", 1)

            print(f"\n  Mono AND distance to midpoint:      {mono_mid:.3f} ({100*mono_mid/base_dist:.1f}% of base)")
            print(f"  SuperDiff AND distance to midpoint: {sd_mid:.3f} ({100*sd_mid/base_dist:.1f}% of base)")
            print(f"  Agreement between methods:          {100*(1-agreement):.1f}%")
            print(f"\n  CONCLUSION: {conclusion}")
        else:
            print("\n  (Insufficient data for conclusion)")

        print("\n" + "=" * 80 + "\n")

        return metrics

    def _create_enhanced_visualizations(self, output_dir: str):
        """Create enhanced visualizations using the EnhancedManifoldVisualizer."""
        try:
            from enhanced_manifold_viz import EnhancedManifoldVisualizer
        except ImportError:
            print("\n  Note: enhanced_manifold_viz not found, skipping enhanced visualizations")
            return

        print("\n" + "=" * 80)
        print("CREATING ENHANCED VISUALIZATIONS")
        print("=" * 80)

        # Prepare data for enhanced visualizer
        viz = EnhancedManifoldVisualizer(
            projections=self._projections,
            condition_indices=self._condition_indices,
            condition_keys=self.condition_keys,
            all_latents=self._all_latents,
        )

        viz.create_all_visualizations(output_dir)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Manifold diagnostics comparing Monolithic vs SuperDiff AND"
    )

    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of runs per condition")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lift", type=float, default=0.0,
                       help="Lift parameter for SuperDiff")
    parser.add_argument("--output-dir", type=str, default="outputs/manifold_diagnostics")
    parser.add_argument("--display-cols", type=int, default=4,
                       help="Number of columns to display in image grid")

    args = parser.parse_args()

    config = ManifoldConfig(
        num_runs=args.num_runs,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lift=args.lift,
        output_dir=args.output_dir,
        display_cols=args.display_cols,
    )

    diagnostics = ManifoldDiagnostics(config)
    diagnostics.run_full_analysis()


if __name__ == "__main__":
    main()
