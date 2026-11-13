#!/usr/bin/env python3
"""
Extended Manifold Analysis: CLIP AND vs SuperDiff AND

Systematic comparison of composition strategies with:
1. Denoising trajectory analysis
2. Velocity/score field comparison
3. Local manifold geometry
4. Cross-attention analysis
5. Off-manifold drift detection
6. Semantic disentanglement metrics

Research Questions:
- Do CLIP AND and SuperDiff AND agree on semantic intersection location?
- Do spatial constraints reshape manifold geometry or just guide decoding?
- Does SuperDiff drift off the learned data manifold?
- When do the two methods diverge in their trajectories?

Usage:
    python notebooks/extended_manifold_analysis.py --num-runs 5 --steps 100
"""

import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import umap
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diffusers import EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

# Suppress warnings
# warnings.filterwarnings("ignore", message=".*scale_model_input.*")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExtendedAnalysisConfig:
    """Configuration for extended manifold analysis."""

    # The 4 key conditions to compare
    conditions: Dict[str, Dict] = field(default_factory=lambda: {
        "clip_semantic": {
            "type": "monolithic",
            "prompt": "a dog and a cat",
            "label": "CLIP AND: semantic",
            "color": "#2ecc71",
        },
        "clip_spatial": {
            "type": "monolithic",
            "prompt": "a dog on the left and a cat on the right",
            "label": "CLIP AND: spatial",
            "color": "#9b59b6",
        },
        "superdiff_semantic": {
            "type": "superdiff",
            "prompt_a": "a dog",
            "prompt_b": "a cat",
            "label": "SuperDiff AND: semantic",
            "color": "#e74c3c",
        },
        "superdiff_spatial": {
            "type": "superdiff",
            "prompt_a": "a dog on the left",
            "prompt_b": "a cat on the right",
            "label": "SuperDiff AND: spatial",
            "color": "#1abc9c",
        },
    })

    # Base concepts for reference
    base_concepts: Dict[str, str] = field(default_factory=lambda: {
        "dog": "a dog",
        "cat": "a cat",
    })

    # Generation parameters
    num_runs: int = 5
    num_inference_steps: int = 100
    guidance_scale: float = 7.5
    lift: float = 0.0

    # Trajectory sampling
    trajectory_sample_interval: int = 10  # Save latent every N steps

    # Model
    model_id: str = "Manojb/stable-diffusion-2-1-base"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Output
    output_dir: str = "outputs/extended_analysis"


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_models(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load Stable Diffusion components."""
    print("Loading models...")
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
    return {"vae": vae, "tokenizer": tokenizer, "text_encoder": text_encoder,
            "unet": unet, "scheduler": scheduler}


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


# ---------------------------------------------------------------------------
# Generation with Trajectory Tracking
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_monolithic_with_trajectory(
    prompt: str,
    models: dict,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    seed: int = None,
    trajectory_interval: int = 10,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """Generate with CLIP AND, tracking full trajectory and velocities."""
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "Manojb/stable-diffusion-2-1-base", subfolder="scheduler"
    )

    # Get embeddings
    text_emb = get_text_embedding(prompt, tokenizer, text_encoder, device)
    uncond_emb = get_text_embedding("", tokenizer, text_encoder, device)

    # Initialize latents
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # Storage for trajectory analysis
    trajectory = [latents.cpu().clone()]
    velocities = []
    timesteps_saved = [0]

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        latent_input = latents / ((sigma ** 2 + 1) ** 0.5)

        # Get predictions
        noise_uncond = unet(latent_input, t, encoder_hidden_states=uncond_emb).sample
        noise_cond = unet(latent_input, t, encoder_hidden_states=text_emb).sample

        # CFG velocity
        velocity = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Euler step
        latents = scheduler.step(velocity, t, latents).prev_sample

        # Save trajectory points
        if (i + 1) % trajectory_interval == 0 or i == len(scheduler.timesteps) - 1:
            trajectory.append(latents.cpu().clone())
            velocities.append(velocity.cpu().clone())
            timesteps_saved.append(i + 1)

    return {
        "final_latents": latents,
        "trajectory": trajectory,
        "velocities": velocities,
        "timesteps": timesteps_saved,
    }


@torch.no_grad()
def generate_superdiff_with_trajectory(
    prompt_a: str,
    prompt_b: str,
    models: dict,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    lift: float = 0.0,
    seed: int = None,
    trajectory_interval: int = 10,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """Generate with SuperDiff AND, tracking trajectory, velocities, and kappa."""
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "Manojb/stable-diffusion-2-1-base", subfolder="scheduler"
    )

    # Get embeddings
    emb_a = get_text_embedding(prompt_a, tokenizer, text_encoder, device)
    emb_b = get_text_embedding(prompt_b, tokenizer, text_encoder, device)
    uncond_emb = get_text_embedding("", tokenizer, text_encoder, device)

    # Initialize latents
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # Storage
    trajectory = [latents.cpu().clone()]
    velocities = []
    velocities_a = []
    velocities_b = []
    kappa_history = []
    timesteps_saved = [0]

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
        term1 = (torch.abs(dsigma) * (vel_b - vel_a) * (vel_b + vel_a)).sum()
        term2 = (dx_ind * (vel_a - vel_b)).sum()
        term3 = sigma * lift / num_inference_steps

        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * guidance_scale * ((vel_a - vel_b) ** 2).sum()

        kappa = numerator / (denominator + 1e-8)
        kappa_history.append(kappa.cpu().item())

        # Composite vector field
        velocity = vel_uncond + guidance_scale * (
            (vel_b - vel_uncond) + kappa * (vel_a - vel_b)
        )

        dx = 2 * dsigma * velocity + noise
        latents = latents + dx

        # Save trajectory points
        if (i + 1) % trajectory_interval == 0 or i == len(scheduler.timesteps) - 1:
            trajectory.append(latents.cpu().clone())
            velocities.append(velocity.cpu().clone())
            velocities_a.append(vel_a.cpu().clone())
            velocities_b.append(vel_b.cpu().clone())
            timesteps_saved.append(i + 1)

    return {
        "final_latents": latents,
        "trajectory": trajectory,
        "velocities": velocities,
        "velocities_a": velocities_a,
        "velocities_b": velocities_b,
        "kappa_history": kappa_history,
        "timesteps": timesteps_saved,
    }


# ---------------------------------------------------------------------------
# Trajectory Analysis Functions
# ---------------------------------------------------------------------------
def compute_trajectory_length(trajectory: List[torch.Tensor]) -> float:
    """Compute total path length through latent space."""
    total_length = 0.0
    for i in range(1, len(trajectory)):
        diff = trajectory[i].flatten() - trajectory[i-1].flatten()
        total_length += torch.norm(diff).item()
    return total_length


def compute_trajectory_curvature(trajectory: List[torch.Tensor]) -> List[float]:
    """Compute curvature at each point (angle change between segments)."""
    curvatures = []
    for i in range(1, len(trajectory) - 1):
        v1 = trajectory[i].flatten() - trajectory[i-1].flatten()
        v2 = trajectory[i+1].flatten() - trajectory[i].flatten()

        # Cosine of angle between vectors
        cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        # Clamp for numerical stability
        cos_angle = max(-1, min(1, cos_angle))
        angle = np.arccos(cos_angle)
        curvatures.append(angle)

    return curvatures


def compute_velocity_alignment(vel1: torch.Tensor, vel2: torch.Tensor) -> float:
    """Compute cosine similarity between two velocity fields."""
    return F.cosine_similarity(vel1.flatten().unsqueeze(0),
                               vel2.flatten().unsqueeze(0)).item()


def compute_trajectory_divergence(traj1: List[torch.Tensor],
                                  traj2: List[torch.Tensor]) -> List[float]:
    """Compute distance between two trajectories at each saved timestep."""
    min_len = min(len(traj1), len(traj2))
    distances = []
    for i in range(min_len):
        dist = torch.norm(traj1[i].flatten() - traj2[i].flatten()).item()
        distances.append(dist)
    return distances


# ---------------------------------------------------------------------------
# Manifold Geometry Analysis
# ---------------------------------------------------------------------------
def estimate_local_intrinsic_dimensionality(
    latents: np.ndarray,
    k: int = 10
) -> float:
    """Estimate local intrinsic dimensionality using MLE estimator."""
    if len(latents) < k + 1:
        return float('nan')

    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(latents)
    distances, _ = nn.kneighbors(latents)

    # Skip self-distance (first column)
    distances = distances[:, 1:]

    # MLE estimator for intrinsic dimensionality
    # d = 1 / (1/k * sum(log(r_k / r_j) for j in 1..k-1))
    log_ratios = np.log(distances[:, -1:] / (distances[:, :-1] + 1e-10))
    lid_estimates = (k - 1) / np.sum(log_ratios, axis=1)

    return np.median(lid_estimates)


def compute_reconstruction_error(
    vae,
    latents: torch.Tensor
) -> float:
    """Compute VAE encode-decode reconstruction error."""
    with torch.no_grad():
        # Decode
        latents = latents.to(dtype=vae.dtype, device=vae.device)
        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

        # Re-encode
        recon_latents = vae.encode(images).latent_dist.sample()
        recon_latents = recon_latents * vae.config.scaling_factor

        # L2 error
        error = torch.norm(latents - recon_latents).item()

    return error


# ---------------------------------------------------------------------------
# Semantic Analysis
# ---------------------------------------------------------------------------
def compute_concept_balance(
    latent: torch.Tensor,
    base_centroids: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Compute relative distance to each base concept."""
    latent_flat = latent.flatten().numpy()

    distances = {}
    for concept, centroid in base_centroids.items():
        distances[concept] = np.linalg.norm(latent_flat - centroid)

    # Normalize to get balance ratio
    total = sum(distances.values())
    balance = {k: 1 - (v / total) for k, v in distances.items()}

    return balance


def compute_semantic_overlap_region(
    clip_latents: np.ndarray,
    superdiff_latents: np.ndarray,
    n_components: int = 3
) -> Dict[str, float]:
    """Analyze overlap between CLIP AND and SuperDiff AND clusters."""
    # Combine for shared PCA
    combined = np.vstack([clip_latents, superdiff_latents])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(combined)

    clip_proj = projected[:len(clip_latents)]
    sd_proj = projected[len(clip_latents):]

    # Compute centroids
    clip_centroid = clip_proj.mean(axis=0)
    sd_centroid = sd_proj.mean(axis=0)

    # Compute cluster radii (std)
    clip_radius = np.mean([np.linalg.norm(p - clip_centroid) for p in clip_proj])
    sd_radius = np.mean([np.linalg.norm(p - sd_centroid) for p in sd_proj])

    # Centroid distance
    centroid_dist = np.linalg.norm(clip_centroid - sd_centroid)

    # Overlap coefficient (simplified)
    # If centroid_dist < clip_radius + sd_radius, clusters overlap
    overlap_threshold = clip_radius + sd_radius
    overlap_ratio = max(0, 1 - centroid_dist / overlap_threshold)

    return {
        "centroid_distance": centroid_dist,
        "clip_radius": clip_radius,
        "superdiff_radius": sd_radius,
        "overlap_ratio": overlap_ratio,
        "clip_centroid": clip_centroid,
        "superdiff_centroid": sd_centroid,
    }


# ---------------------------------------------------------------------------
# Main Analysis Class
# ---------------------------------------------------------------------------
class ExtendedManifoldAnalysis:
    """Comprehensive comparison of CLIP AND vs SuperDiff AND."""

    def __init__(self, config: ExtendedAnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        self.models = load_models(config.model_id, self.device, self.dtype)

        # Storage
        self.results: Dict[str, List[Dict]] = {}
        self.base_results: Dict[str, List[Dict]] = {}

    def run_all_generations(self):
        """Generate all conditions with trajectory tracking."""
        print("\n" + "=" * 80)
        print("GENERATING ALL CONDITIONS WITH TRAJECTORY TRACKING")
        print("=" * 80)

        # Initialize storage
        for key in self.config.conditions:
            self.results[key] = []
        for key in self.config.base_concepts:
            self.base_results[key] = []

        for run_idx in range(self.config.num_runs):
            seed = 42 + run_idx
            print(f"\n--- Run {run_idx + 1}/{self.config.num_runs} (seed={seed}) ---")

            # Generate base concepts
            for concept_key, prompt in self.config.base_concepts.items():
                print(f"  Base: {concept_key}")
                result = generate_monolithic_with_trajectory(
                    prompt, self.models,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    seed=seed,
                    trajectory_interval=self.config.trajectory_sample_interval,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.base_results[concept_key].append(result)

            # Generate comparison conditions
            for cond_key, cond_config in self.config.conditions.items():
                print(f"  {cond_key}: {cond_config['label']}")

                if cond_config["type"] == "monolithic":
                    result = generate_monolithic_with_trajectory(
                        cond_config["prompt"], self.models,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        seed=seed,
                        trajectory_interval=self.config.trajectory_sample_interval,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:  # superdiff
                    result = generate_superdiff_with_trajectory(
                        cond_config["prompt_a"],
                        cond_config["prompt_b"],
                        self.models,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        lift=self.config.lift,
                        seed=seed,
                        trajectory_interval=self.config.trajectory_sample_interval,
                        device=self.device,
                        dtype=self.dtype,
                    )

                self.results[cond_key].append(result)

        print(f"\n  Completed {self.config.num_runs} runs for all conditions")

    def analyze_trajectories(self, output_dir: str):
        """Analyze and visualize trajectory differences."""
        print("\n" + "=" * 80)
        print("TRAJECTORY ANALYSIS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Compute metrics for each condition
        trajectory_metrics = {}

        for cond_key in self.config.conditions:
            metrics = {
                "lengths": [],
                "curvatures": [],
            }

            for result in self.results[cond_key]:
                length = compute_trajectory_length(result["trajectory"])
                curvature = compute_trajectory_curvature(result["trajectory"])

                metrics["lengths"].append(length)
                metrics["curvatures"].append(np.mean(curvature) if curvature else 0)

            trajectory_metrics[cond_key] = metrics

        # Plot trajectory metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Trajectory lengths
        ax = axes[0]
        conditions = list(self.config.conditions.keys())
        colors = [self.config.conditions[k]["color"] for k in conditions]
        lengths = [trajectory_metrics[k]["lengths"] for k in conditions]

        bp = ax.boxplot(lengths, labels=[self.config.conditions[k]["label"]
                                         for k in conditions], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Trajectory Length")
        ax.set_title("Path Length Through Latent Space")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(alpha=0.3)

        # 2. Trajectory curvatures
        ax = axes[1]
        curvatures = [trajectory_metrics[k]["curvatures"] for k in conditions]

        bp = ax.boxplot(curvatures, labels=[self.config.conditions[k]["label"]
                                            for k in conditions], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Mean Curvature (radians)")
        ax.set_title("Trajectory Curvature (Higher = More Navigation)")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "trajectory_metrics.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path / 'trajectory_metrics.png'}")
        plt.close()

        return trajectory_metrics

    def analyze_trajectory_divergence(self, output_dir: str):
        """Compare when CLIP and SuperDiff trajectories diverge."""
        print("\n" + "=" * 80)
        print("TRAJECTORY DIVERGENCE ANALYSIS")
        print("=" * 80)

        output_path = Path(output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Compare semantic conditions
        ax = axes[0]
        divergences_semantic = []

        for i in range(self.config.num_runs):
            clip_traj = self.results["clip_semantic"][i]["trajectory"]
            sd_traj = self.results["superdiff_semantic"][i]["trajectory"]
            div = compute_trajectory_divergence(clip_traj, sd_traj)
            divergences_semantic.append(div)

        timesteps = self.results["clip_semantic"][0]["timesteps"][:len(divergences_semantic[0])]

        # Plot mean and std
        div_array = np.array(divergences_semantic)
        mean_div = div_array.mean(axis=0)
        std_div = div_array.std(axis=0)

        ax.plot(timesteps, mean_div, 'b-', linewidth=2, label="Mean divergence")
        ax.fill_between(timesteps, mean_div - std_div, mean_div + std_div, alpha=0.3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("L2 Distance Between Trajectories")
        ax.set_title("CLIP AND vs SuperDiff AND (Semantic)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Compare spatial conditions
        ax = axes[1]
        divergences_spatial = []

        for i in range(self.config.num_runs):
            clip_traj = self.results["clip_spatial"][i]["trajectory"]
            sd_traj = self.results["superdiff_spatial"][i]["trajectory"]
            div = compute_trajectory_divergence(clip_traj, sd_traj)
            divergences_spatial.append(div)

        div_array = np.array(divergences_spatial)
        mean_div = div_array.mean(axis=0)
        std_div = div_array.std(axis=0)

        ax.plot(timesteps, mean_div, 'r-', linewidth=2, label="Mean divergence")
        ax.fill_between(timesteps, mean_div - std_div, mean_div + std_div, alpha=0.3, color='red')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("L2 Distance Between Trajectories")
        ax.set_title("CLIP AND vs SuperDiff AND (Spatial)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "trajectory_divergence.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path / 'trajectory_divergence.png'}")
        plt.close()

        # Compute divergence onset (when distance exceeds threshold)
        semantic_onset = self._find_divergence_onset(divergences_semantic, timesteps)
        spatial_onset = self._find_divergence_onset(divergences_spatial, timesteps)

        print(f"\n  Semantic divergence onset: timestep ~{semantic_onset}")
        print(f"  Spatial divergence onset: timestep ~{spatial_onset}")

        return {
            "semantic_divergence": divergences_semantic,
            "spatial_divergence": divergences_spatial,
            "semantic_onset": semantic_onset,
            "spatial_onset": spatial_onset,
        }

    def _find_divergence_onset(self, divergences: List[List[float]],
                               timesteps: List[int], threshold_percentile: float = 50) -> int:
        """Find timestep when divergence exceeds threshold."""
        div_array = np.array(divergences)
        mean_div = div_array.mean(axis=0)

        # Threshold: when divergence exceeds 50% of final divergence
        threshold = mean_div[-1] * (threshold_percentile / 100)

        for i, (t, d) in enumerate(zip(timesteps, mean_div)):
            if d > threshold:
                return t

        return timesteps[-1]

    def analyze_kappa_dynamics(self, output_dir: str):
        """Analyze SuperDiff kappa trajectories."""
        print("\n" + "=" * 80)
        print("KAPPA DYNAMICS ANALYSIS")
        print("=" * 80)

        output_path = Path(output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Semantic kappa
        ax = axes[0]
        for i, result in enumerate(self.results["superdiff_semantic"]):
            kappa = result["kappa_history"]
            ax.plot(kappa, alpha=0.5, color="#e74c3c")

        ax.axhline(y=0.5, color='gray', linestyle='--', label="Balanced (κ=0.5)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Kappa (κ)")
        ax.set_title("SuperDiff AND Semantic: Kappa Dynamics")
        ax.legend()
        ax.grid(alpha=0.3)

        # Spatial kappa
        ax = axes[1]
        for i, result in enumerate(self.results["superdiff_spatial"]):
            kappa = result["kappa_history"]
            ax.plot(kappa, alpha=0.5, color="#1abc9c")

        ax.axhline(y=0.5, color='gray', linestyle='--', label="Balanced (κ=0.5)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Kappa (κ)")
        ax.set_title("SuperDiff AND Spatial: Kappa Dynamics")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "kappa_dynamics.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path / 'kappa_dynamics.png'}")
        plt.close()

        # Compute kappa statistics
        semantic_kappas = [r["kappa_history"] for r in self.results["superdiff_semantic"]]
        spatial_kappas = [r["kappa_history"] for r in self.results["superdiff_spatial"]]

        semantic_mean = np.mean([np.mean(k) for k in semantic_kappas])
        semantic_var = np.mean([np.var(k) for k in semantic_kappas])
        spatial_mean = np.mean([np.mean(k) for k in spatial_kappas])
        spatial_var = np.mean([np.var(k) for k in spatial_kappas])

        print(f"\n  Semantic κ: mean={semantic_mean:.3f}, variance={semantic_var:.3f}")
        print(f"  Spatial κ:  mean={spatial_mean:.3f}, variance={spatial_var:.3f}")

        return {
            "semantic_mean": semantic_mean,
            "semantic_var": semantic_var,
            "spatial_mean": spatial_mean,
            "spatial_var": spatial_var,
        }

    def analyze_manifold_geometry(self, output_dir: str):
        """Analyze local manifold properties."""
        print("\n" + "=" * 80)
        print("MANIFOLD GEOMETRY ANALYSIS")
        print("=" * 80)

        output_path = Path(output_dir)

        # Collect final latents
        all_latents = {}
        for cond_key in self.config.conditions:
            latents = [r["final_latents"].cpu().flatten().numpy()
                      for r in self.results[cond_key]]
            all_latents[cond_key] = np.array(latents)

        # Compute Local Intrinsic Dimensionality
        lid_results = {}
        print("\n  Local Intrinsic Dimensionality:")
        for cond_key, latents in all_latents.items():
            lid = estimate_local_intrinsic_dimensionality(latents, k=min(5, len(latents)-1))
            lid_results[cond_key] = lid
            label = self.config.conditions[cond_key]["label"]
            print(f"    {label}: {lid:.2f}")

        # Compute reconstruction error (off-manifold measure)
        recon_errors = {}
        print("\n  Reconstruction Error (VAE cycle):")
        for cond_key in self.config.conditions:
            errors = []
            for result in self.results[cond_key]:
                error = compute_reconstruction_error(
                    self.models["vae"],
                    result["final_latents"]
                )
                errors.append(error)
            recon_errors[cond_key] = np.mean(errors)
            label = self.config.conditions[cond_key]["label"]
            print(f"    {label}: {recon_errors[cond_key]:.3f}")

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        conditions = list(self.config.conditions.keys())
        colors = [self.config.conditions[k]["color"] for k in conditions]
        labels = [self.config.conditions[k]["label"] for k in conditions]

        # LID
        ax = axes[0]
        lids = [lid_results[k] for k in conditions]
        bars = ax.bar(range(len(conditions)), lids, color=colors, alpha=0.7)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Local Intrinsic Dimensionality")
        ax.set_title("Local Manifold Complexity")
        ax.grid(alpha=0.3, axis='y')

        # Reconstruction error
        ax = axes[1]
        errors = [recon_errors[k] for k in conditions]
        bars = ax.bar(range(len(conditions)), errors, color=colors, alpha=0.7)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Off-Manifold Drift (Higher = Further from Data Manifold)")
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path / "manifold_geometry.png", dpi=200, bbox_inches="tight")
        print(f"\nSaved: {output_path / 'manifold_geometry.png'}")
        plt.close()

        return {"lid": lid_results, "reconstruction_error": recon_errors}

    def analyze_semantic_overlap(self, output_dir: str):
        """Analyze semantic overlap between methods."""
        print("\n" + "=" * 80)
        print("SEMANTIC OVERLAP ANALYSIS")
        print("=" * 80)

        output_path = Path(output_dir)

        # Collect latents
        clip_semantic = np.array([r["final_latents"].cpu().flatten().numpy()
                                  for r in self.results["clip_semantic"]])
        sd_semantic = np.array([r["final_latents"].cpu().flatten().numpy()
                               for r in self.results["superdiff_semantic"]])
        clip_spatial = np.array([r["final_latents"].cpu().flatten().numpy()
                                for r in self.results["clip_spatial"]])
        sd_spatial = np.array([r["final_latents"].cpu().flatten().numpy()
                              for r in self.results["superdiff_spatial"]])

        # Analyze overlap
        semantic_overlap = compute_semantic_overlap_region(clip_semantic, sd_semantic)
        spatial_overlap = compute_semantic_overlap_region(clip_spatial, sd_spatial)

        print(f"\n  Semantic conditions:")
        print(f"    Centroid distance: {semantic_overlap['centroid_distance']:.3f}")
        print(f"    CLIP cluster radius: {semantic_overlap['clip_radius']:.3f}")
        print(f"    SuperDiff cluster radius: {semantic_overlap['superdiff_radius']:.3f}")
        print(f"    Overlap ratio: {semantic_overlap['overlap_ratio']:.3f}")

        print(f"\n  Spatial conditions:")
        print(f"    Centroid distance: {spatial_overlap['centroid_distance']:.3f}")
        print(f"    CLIP cluster radius: {spatial_overlap['clip_radius']:.3f}")
        print(f"    SuperDiff cluster radius: {spatial_overlap['superdiff_radius']:.3f}")
        print(f"    Overlap ratio: {spatial_overlap['overlap_ratio']:.3f}")

        # Visualization: PCA projection
        combined_semantic = np.vstack([clip_semantic, sd_semantic])
        combined_spatial = np.vstack([clip_spatial, sd_spatial])

        pca = PCA(n_components=2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Semantic
        ax = axes[0]
        proj = pca.fit_transform(combined_semantic)
        clip_proj = proj[:len(clip_semantic)]
        sd_proj = proj[len(clip_semantic):]

        ax.scatter(clip_proj[:, 0], clip_proj[:, 1],
                  c=self.config.conditions["clip_semantic"]["color"],
                  s=100, alpha=0.7, label="CLIP AND", edgecolors='white')
        ax.scatter(sd_proj[:, 0], sd_proj[:, 1],
                  c=self.config.conditions["superdiff_semantic"]["color"],
                  s=100, alpha=0.7, label="SuperDiff AND", edgecolors='white')

        # Mark centroids
        clip_cent = clip_proj.mean(axis=0)
        sd_cent = sd_proj.mean(axis=0)
        ax.scatter(clip_cent[0], clip_cent[1], c="black", s=200, marker="D",
                  edgecolors="white", linewidths=2, zorder=10)
        ax.scatter(sd_cent[0], sd_cent[1], c="black", s=200, marker="*",
                  edgecolors="white", linewidths=2, zorder=10)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Semantic: 'dog AND cat'\nOverlap: {semantic_overlap['overlap_ratio']:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Spatial
        ax = axes[1]
        proj = pca.fit_transform(combined_spatial)
        clip_proj = proj[:len(clip_spatial)]
        sd_proj = proj[len(clip_spatial):]

        ax.scatter(clip_proj[:, 0], clip_proj[:, 1],
                  c=self.config.conditions["clip_spatial"]["color"],
                  s=100, alpha=0.7, label="CLIP AND", edgecolors='white')
        ax.scatter(sd_proj[:, 0], sd_proj[:, 1],
                  c=self.config.conditions["superdiff_spatial"]["color"],
                  s=100, alpha=0.7, label="SuperDiff AND", edgecolors='white')

        clip_cent = clip_proj.mean(axis=0)
        sd_cent = sd_proj.mean(axis=0)
        ax.scatter(clip_cent[0], clip_cent[1], c="black", s=200, marker="D",
                  edgecolors="white", linewidths=2, zorder=10)
        ax.scatter(sd_cent[0], sd_cent[1], c="black", s=200, marker="*",
                  edgecolors="white", linewidths=2, zorder=10)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Spatial: 'dog left AND cat right'\nOverlap: {spatial_overlap['overlap_ratio']:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "semantic_overlap.png", dpi=200, bbox_inches="tight")
        print(f"\nSaved: {output_path / 'semantic_overlap.png'}")
        plt.close()

        return {"semantic": semantic_overlap, "spatial": spatial_overlap}

    def create_hypothesis_report(self, output_dir: str, all_results: Dict):
        """Generate research hypothesis report."""
        print("\n" + "=" * 80)
        print("GENERATING HYPOTHESIS REPORT")
        print("=" * 80)

        output_path = Path(output_dir)

        lines = [
            "=" * 70,
            "EXTENDED MANIFOLD ANALYSIS: HYPOTHESIS TESTING REPORT",
            "=" * 70,
            "",
            f"Configuration: {self.config.num_runs} runs, {self.config.num_inference_steps} steps",
            "",
        ]

        # H1: Semantic Intersection Location
        lines.extend([
            "-" * 70,
            "H1: SEMANTIC INTERSECTION LOCATION",
            "-" * 70,
            "Hypothesis: CLIP AND and SuperDiff AND locate semantic intersections",
            "            at different points in latent space.",
            "",
        ])

        overlap = all_results.get("overlap", {})
        if overlap:
            sem = overlap.get("semantic", {})
            spa = overlap.get("spatial", {})

            if sem:
                lines.append(f"  Semantic condition centroid distance: {sem.get('centroid_distance', 'N/A'):.3f}")
                lines.append(f"  Semantic overlap ratio: {sem.get('overlap_ratio', 'N/A'):.3f}")
            if spa:
                lines.append(f"  Spatial condition centroid distance: {spa.get('centroid_distance', 'N/A'):.3f}")
                lines.append(f"  Spatial overlap ratio: {spa.get('overlap_ratio', 'N/A'):.3f}")

            # Verdict
            if sem and sem.get('overlap_ratio', 0) < 0.3:
                lines.append("\n  VERDICT: SUPPORTED - Methods find different intersection regions")
            elif sem and sem.get('overlap_ratio', 0) > 0.7:
                lines.append("\n  VERDICT: REJECTED - Methods largely agree on intersection")
            else:
                lines.append("\n  VERDICT: PARTIAL - Some divergence in intersection location")

        lines.append("")

        # H2: Spatial Constraints
        lines.extend([
            "-" * 70,
            "H2: SPATIAL CONSTRAINTS AS MANIFOLD RESHAPING",
            "-" * 70,
            "Hypothesis: Spatial constraints reshape local manifold geometry",
            "            rather than merely guiding decoding.",
            "",
        ])

        geometry = all_results.get("geometry", {})
        if geometry:
            lid = geometry.get("lid", {})
            if lid:
                sem_lid = lid.get("clip_semantic", float('nan'))
                spa_lid = lid.get("clip_spatial", float('nan'))
                lines.append(f"  CLIP semantic LID: {sem_lid:.2f}")
                lines.append(f"  CLIP spatial LID: {spa_lid:.2f}")

                if not np.isnan(sem_lid) and not np.isnan(spa_lid):
                    if spa_lid < sem_lid * 0.9:
                        lines.append("\n  VERDICT: SUPPORTED - Spatial constraints reduce dimensionality")
                    else:
                        lines.append("\n  VERDICT: NOT SUPPORTED - Similar dimensionality")

        lines.append("")

        # H3: Trajectory Divergence
        lines.extend([
            "-" * 70,
            "H3: TRAJECTORY DIVERGENCE TIMING",
            "-" * 70,
            "Hypothesis: Early divergence = different semantics",
            "            Late divergence = different decoding",
            "",
        ])

        divergence = all_results.get("divergence", {})
        if divergence:
            sem_onset = divergence.get("semantic_onset", "N/A")
            spa_onset = divergence.get("spatial_onset", "N/A")
            lines.append(f"  Semantic divergence onset: timestep {sem_onset}")
            lines.append(f"  Spatial divergence onset: timestep {spa_onset}")

            if isinstance(sem_onset, (int, float)) and isinstance(spa_onset, (int, float)):
                if sem_onset < spa_onset:
                    lines.append("\n  VERDICT: SUPPORTED - Semantic diverges earlier than spatial")
                else:
                    lines.append("\n  VERDICT: PARTIAL - Similar divergence timing")

        lines.append("")

        # H4: Off-Manifold Drift
        lines.extend([
            "-" * 70,
            "H4: OFF-MANIFOLD DRIFT",
            "-" * 70,
            "Hypothesis: SuperDiff AND drifts further from data manifold than CLIP AND",
            "",
        ])

        if geometry:
            recon = geometry.get("reconstruction_error", {})
            if recon:
                clip_err = recon.get("clip_semantic", 0)
                sd_err = recon.get("superdiff_semantic", 0)
                lines.append(f"  CLIP AND reconstruction error: {clip_err:.3f}")
                lines.append(f"  SuperDiff AND reconstruction error: {sd_err:.3f}")

                if sd_err > clip_err * 1.2:
                    lines.append("\n  VERDICT: SUPPORTED - SuperDiff shows more off-manifold drift")
                elif clip_err > sd_err * 1.2:
                    lines.append("\n  VERDICT: OPPOSITE - CLIP shows more drift")
                else:
                    lines.append("\n  VERDICT: NOT SUPPORTED - Similar manifold adherence")

        lines.append("")

        # H5: Kappa Dynamics
        lines.extend([
            "-" * 70,
            "H5/H6: KAPPA DYNAMICS REFLECT CONCEPT BALANCE",
            "-" * 70,
            "Hypothesis: Kappa variance correlates with concept balance quality",
            "",
        ])

        kappa = all_results.get("kappa", {})
        if kappa:
            lines.append(f"  Semantic κ mean: {kappa.get('semantic_mean', 'N/A'):.3f}")
            lines.append(f"  Semantic κ variance: {kappa.get('semantic_var', 'N/A'):.3f}")
            lines.append(f"  Spatial κ mean: {kappa.get('spatial_mean', 'N/A'):.3f}")
            lines.append(f"  Spatial κ variance: {kappa.get('spatial_var', 'N/A'):.3f}")

            sem_var = kappa.get('semantic_var', 0)
            spa_var = kappa.get('spatial_var', 0)

            if sem_var > spa_var * 1.5:
                lines.append("\n  OBSERVATION: Semantic composition shows more κ instability")
            elif spa_var > sem_var * 1.5:
                lines.append("\n  OBSERVATION: Spatial composition shows more κ instability")
            else:
                lines.append("\n  OBSERVATION: Similar κ dynamics across conditions")

        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        report_path = output_path / "hypothesis_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved: {report_path}")

        # Print summary
        print("\n" + "\n".join(lines))

    def run_full_analysis(self):
        """Run complete extended analysis pipeline."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("EXTENDED MANIFOLD ANALYSIS: CLIP AND vs SuperDiff AND")
        print("=" * 80)

        # 1. Generate all conditions
        self.run_all_generations()

        # 2. Trajectory analysis
        traj_metrics = self.analyze_trajectories(self.config.output_dir)

        # 3. Trajectory divergence
        divergence_results = self.analyze_trajectory_divergence(self.config.output_dir)

        # 4. Kappa dynamics
        kappa_results = self.analyze_kappa_dynamics(self.config.output_dir)

        # 5. Manifold geometry
        geometry_results = self.analyze_manifold_geometry(self.config.output_dir)

        # 6. Semantic overlap
        overlap_results = self.analyze_semantic_overlap(self.config.output_dir)

        # 7. Generate hypothesis report
        all_results = {
            "trajectory": traj_metrics,
            "divergence": divergence_results,
            "kappa": kappa_results,
            "geometry": geometry_results,
            "overlap": overlap_results,
        }
        self.create_hypothesis_report(self.config.output_dir, all_results)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {output_path.absolute()}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extended manifold analysis comparing CLIP AND vs SuperDiff AND"
    )

    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lift", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs/extended_analysis")
    parser.add_argument("--trajectory-interval", type=int, default=10)

    args = parser.parse_args()

    config = ExtendedAnalysisConfig(
        num_runs=args.num_runs,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lift=args.lift,
        output_dir=args.output_dir,
        trajectory_sample_interval=args.trajectory_interval,
    )

    analysis = ExtendedManifoldAnalysis(config)
    analysis.run_full_analysis()


if __name__ == "__main__":
    main()
