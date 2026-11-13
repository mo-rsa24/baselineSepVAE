"""
Controlled Experiments for Understanding SUPERDIFF Composition Semantics

This module implements diagnostic experiments to investigate why SUPERDIFF AND
produces hybridization rather than co-presence, analyzing:
1. Latent space geometry
2. Trajectory dynamics
3. Manifold properties
4. Semantic interpretation

Based on "The Superposition of Diffusion Models Using the Itô Density Estimator"
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from diffusers import EulerDiscreteScheduler
from notebooks.dynamics import stochastic_super_diff_and, get_latents, get_vel
from notebooks.utils import get_sd_models, get_text_embedding, get_image


@dataclass
class ExperimentConfig:
    """Configuration for composition experiments"""
    # Prompts
    prompt_a: str = "A photograph of a cat"
    prompt_b: str = "A photograph of a dog"
    prompt_composed: str = "A photograph of a cat and a dog"

    # Sampling parameters
    num_runs: int = 20  # Multiple stochastic runs
    batch_size: int = 4
    num_inference_steps: int = 500
    guidance_scale: float = 7.5

    # Model parameters
    height: int = 512
    width: int = 512
    latent_height: int = 64
    latent_width: int = 64
    z_channels: int = 4

    # SUPERDIFF parameters
    lift: float = 0.0

    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Output
    output_dir: str = "experiments/composition_analysis"


class LatentTrajectoryCollector:
    """Collects latent trajectories during diffusion sampling"""

    def __init__(self, num_steps: int, batch_size: int, z_channels: int,
                 latent_height: int, latent_width: int):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shape = (z_channels, latent_height, latent_width)

        # Storage for trajectories
        self.trajectories = torch.zeros(
            (num_steps + 1, batch_size, z_channels, latent_height, latent_width)
        )
        self.velocities = torch.zeros(
            (num_steps, batch_size, z_channels, latent_height, latent_width)
        )
        self.sigmas = torch.zeros(num_steps + 1)
        self.timesteps = torch.zeros(num_steps)

    def store_step(self, step: int, latents: torch.Tensor, velocity: torch.Tensor,
                   sigma: float, timestep: float):
        """Store trajectory information at each step"""
        self.trajectories[step] = latents.detach().cpu()
        if velocity is not None:
            self.velocities[step] = velocity.detach().cpu()
        self.sigmas[step] = sigma
        if timestep is not None:
            self.timesteps[step] = timestep

    def store_final(self, latents: torch.Tensor):
        """Store final latent state"""
        self.trajectories[-1] = latents.detach().cpu()


def sample_with_trajectory_tracking(
    latents: torch.Tensor,
    prompt: str,
    scheduler: EulerDiscreteScheduler,
    unet,
    tokenizer,
    text_encoder,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 100,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, LatentTrajectoryCollector]:
    """
    Standard classifier-free guidance sampling with trajectory tracking
    """
    embeddings = get_text_embedding([prompt] * batch_size, tokenizer, text_encoder, device)
    uncond_embeddings = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3]
    )

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        # Get velocities
        vel_cond, _ = get_vel(unet, t, sigma, latents, [embeddings], device=device, dtype=dtype)
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeddings], device=device, dtype=dtype)

        # Classifier-free guidance
        vf = vel_uncond + guidance_scale * (vel_cond - vel_uncond)

        # Euler step with noise
        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        dx = 2 * dsigma * vf + noise

        # Store trajectory
        tracker.store_step(i, latents, vf, sigma.item(), t.item())

        # Update
        latents = latents + dx

    tracker.store_final(latents)
    return latents, tracker


def superdiff_with_trajectory_tracking(
    latents: torch.Tensor,
    obj_prompt: str,
    bg_prompt: str,
    scheduler: EulerDiscreteScheduler,
    unet,
    tokenizer,
    text_encoder,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 100,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    lift: float = 0.0
) -> Tuple[torch.Tensor, LatentTrajectoryCollector, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SUPERDIFF AND with trajectory tracking
    Returns: (final_latents, tracker, kappa, ll_obj, ll_bg)
    """
    obj_embeddings = get_text_embedding([obj_prompt] * batch_size, tokenizer, text_encoder, device)
    bg_embeddings = get_text_embedding([bg_prompt] * batch_size, tokenizer, text_encoder, device)
    uncond_embeddings = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3]
    )

    ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        vel_obj, _ = get_vel(unet, t, sigma, latents, [obj_embeddings], device=device, dtype=dtype)
        vel_bg, _ = get_vel(unet, t, sigma, latents, [bg_embeddings], device=device, dtype=dtype)
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeddings], device=device, dtype=dtype)

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise

        # Compute kappa
        term1 = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
        term2 = (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        term3 = sigma * lift / num_inference_steps
        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))
        kappa[i + 1] = numerator / (denominator + 1e-8)

        # Composite vector field
        vf = vel_uncond + guidance_scale * (
            (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg)
        )

        dx = 2 * dsigma * vf + noise

        # Store trajectory
        tracker.store_step(i, latents, vf, sigma.item(), t.item())

        # Update
        latents = latents + dx

        # Update log-likelihoods
        ll_obj[i + 1] = ll_obj[i] + (-torch.abs(dsigma) / sigma * (vel_obj) ** 2 - (dx * (vel_obj / sigma))).sum((1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (-torch.abs(dsigma) / sigma * (vel_bg) ** 2 - (dx * (vel_bg / sigma))).sum((1, 2, 3))

    tracker.store_final(latents)
    return latents, tracker, kappa, ll_obj, ll_bg


class CompositionExperimentSuite:
    """Complete suite of composition experiments"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        print("Loading models...")
        models = get_sd_models(dtype=config.dtype, device=torch.device(config.device))
        self.vae = models["vae"]
        self.tokenizer = models["tokenizer"]
        self.text_encoder = models["text_encoder"]
        self.unet = models["unet"]
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            "Manojb/stable-diffusion-2-1-base", subfolder="scheduler"
        )

        # Storage for results
        self.results = {
            'monolithic': {'latents': [], 'trajectories': []},
            'prompt_a': {'latents': [], 'trajectories': []},
            'prompt_b': {'latents': [], 'trajectories': []},
            'superdiff': {'latents': [], 'trajectories': [], 'kappas': [], 'll_obj': [], 'll_bg': []}
        }

    def run_all_experiments(self):
        """Run complete experimental suite"""
        print("\n" + "="*80)
        print("SUPERDIFF COMPOSITION ANALYSIS - EXPERIMENTAL SUITE")
        print("="*80)

        for run_idx in range(self.config.num_runs):
            print(f"\n--- Run {run_idx + 1}/{self.config.num_runs} ---")

            # Use different seed per run to explore stochastic variation
            # All conditions within a run share the SAME initial noise for fair comparison
            initial_latents = get_latents(
                self.scheduler,
                z_channels=self.config.z_channels,
                device=torch.device(self.config.device),
                dtype=self.config.dtype,
                num_inference_steps=self.config.num_inference_steps,
                batch_size=self.config.batch_size,
                latent_width=self.config.latent_width,
                latent_height=self.config.latent_height,
                seed=run_idx  # Different seed per run
            )

            # Experiment 1: Monolithic prompt
            print(f"  Sampling: '{self.config.prompt_composed}'")
            latents_mono, traj_mono = sample_with_trajectory_tracking(
                initial_latents.clone(),
                self.config.prompt_composed,
                self.scheduler,
                self.unet,
                self.tokenizer,
                self.text_encoder,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                batch_size=self.config.batch_size,
                device=torch.device(self.config.device),
                dtype=self.config.dtype
            )
            self.results['monolithic']['latents'].append(latents_mono)
            self.results['monolithic']['trajectories'].append(traj_mono)

            # Experiment 2: Individual prompt A
            print(f"  Sampling: '{self.config.prompt_a}'")
            latents_a, traj_a = sample_with_trajectory_tracking(
                initial_latents.clone(),
                self.config.prompt_a,
                self.scheduler,
                self.unet,
                self.tokenizer,
                self.text_encoder,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                batch_size=self.config.batch_size,
                device=torch.device(self.config.device),
                dtype=self.config.dtype
            )
            self.results['prompt_a']['latents'].append(latents_a)
            self.results['prompt_a']['trajectories'].append(traj_a)

            # Experiment 3: Individual prompt B
            print(f"  Sampling: '{self.config.prompt_b}'")
            latents_b, traj_b = sample_with_trajectory_tracking(
                initial_latents.clone(),
                self.config.prompt_b,
                self.scheduler,
                self.unet,
                self.tokenizer,
                self.text_encoder,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                batch_size=self.config.batch_size,
                device=torch.device(self.config.device),
                dtype=self.config.dtype
            )
            self.results['prompt_b']['latents'].append(latents_b)
            self.results['prompt_b']['trajectories'].append(traj_b)

            # Experiment 4: SUPERDIFF composition
            print(f"  SUPERDIFF: '{self.config.prompt_a}' ∧ '{self.config.prompt_b}'")
            latents_sd, traj_sd, kappa, ll_obj, ll_bg = superdiff_with_trajectory_tracking(
                initial_latents.clone(),
                self.config.prompt_a,
                self.config.prompt_b,
                self.scheduler,
                self.unet,
                self.tokenizer,
                self.text_encoder,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                batch_size=self.config.batch_size,
                device=torch.device(self.config.device),
                dtype=self.config.dtype,
                lift=self.config.lift
            )
            self.results['superdiff']['latents'].append(latents_sd)
            self.results['superdiff']['trajectories'].append(traj_sd)
            self.results['superdiff']['kappas'].append(kappa)
            self.results['superdiff']['ll_obj'].append(ll_obj)
            self.results['superdiff']['ll_bg'].append(ll_bg)

        print("\n" + "="*80)
        print("Data collection complete. Running diagnostics...")
        print("="*80 + "\n")

        # Run all diagnostic analyses
        self.analyze_results()

    def analyze_results(self):
        """Run all diagnostic analyses"""
        self.generate_sample_images()
        self.analyze_trajectory_geometry()
        self.analyze_centroid_statistics()
        self.analyze_kappa_dynamics()
        self.analyze_pca_projections()
        self.analyze_manifold_distances()
        self.analyze_velocity_field_alignment()
        self.generate_summary_report()

    def generate_sample_images(self):
        """Generate and save sample images from each condition"""
        print("Generating sample images...")

        fig, axes = plt.subplots(4, self.config.num_runs,
                                figsize=(3*self.config.num_runs, 13))

        conditions = ['monolithic', 'prompt_a', 'prompt_b', 'superdiff']
        titles = [
            f'monolithic \n"{self.config.prompt_composed}"',
            f'individual A\n"{self.config.prompt_a}"',
            f'individual B\n"{self.config.prompt_b}"',
            f'Superdiff (A ∧ B)\n"{self.config.prompt_a}" ∧\n"{self.config.prompt_b}"'
        ]

        # Color coding for different condition types
        label_colors = ['wheat', 'lightblue', 'lightgreen', 'lightcoral']

        for cond_idx, (condition, title, color) in enumerate(zip(conditions, titles, label_colors)):
            axes[cond_idx, 0].set_ylabel(title, fontsize=9, rotation=0,
                                         ha='right', va='center',
                                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.4))

            for run_idx in range(min(self.config.num_runs, axes.shape[1])):
                latents = self.results[condition]['latents'][run_idx][0:1]  # First sample
                img = get_image(self.vae, latents, nrow=1, ncol=1)

                axes[cond_idx, run_idx].imshow(img)
                axes[cond_idx, run_idx].axis('off')
                if cond_idx == 0:
                    axes[cond_idx, run_idx].set_title(f'Run {run_idx+1}', fontsize=10, fontweight='bold')

        # Add overall title
        fig.suptitle('SUPERDIFF Composition Analysis: Sample Images Comparison',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: sample_images_comparison.png")

    def analyze_trajectory_geometry(self):
        """Analyze trajectory paths in latent space"""
        print("\nAnalyzing trajectory geometry...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Flatten latents to vectors for analysis
        def flatten_trajectory(traj):
            # traj.trajectories: (num_steps+1, batch_size, C, H, W)
            return traj.trajectories.reshape(traj.trajectories.shape[0],
                                            traj.trajectories.shape[1], -1)

        # Plot 1: Trajectory norms over time
        ax = axes[0, 0]
        for run_idx in range(self.config.num_runs):
            for condition, color, label in [
                ('monolithic', 'green', f'Monolithic: "{self.config.prompt_composed}"'),
                ('prompt_a', 'blue', f'Prompt A: "{self.config.prompt_a}"'),
                ('prompt_b', 'orange', f'Prompt B: "{self.config.prompt_b}"'),
                ('superdiff', 'red', f'SUPERDIFF: A ∧ B')
            ]:
                traj = self.results[condition]['trajectories'][run_idx]
                flat = flatten_trajectory(traj)
                norms = torch.norm(flat, dim=2).mean(dim=1)  # Average over batch

                alpha = 0.3 if run_idx > 0 else 1.0
                ax.plot(norms.numpy(), color=color, alpha=alpha,
                       label=label if run_idx == 0 else None)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('L2 Norm of Latent', fontsize=11)
        ax.set_title('Trajectory Norms: Denoising Progression', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Pairwise distances between conditions over time
        ax = axes[0, 1]
        run_idx = 0  # Use first run for clarity

        traj_mono = flatten_trajectory(self.results['monolithic']['trajectories'][run_idx])
        traj_a = flatten_trajectory(self.results['prompt_a']['trajectories'][run_idx])
        traj_b = flatten_trajectory(self.results['prompt_b']['trajectories'][run_idx])
        traj_sd = flatten_trajectory(self.results['superdiff']['trajectories'][run_idx])

        # Compute distances (average over batch dimension)
        dist_sd_mono = torch.norm(traj_sd - traj_mono, dim=2).mean(dim=1)
        dist_sd_a = torch.norm(traj_sd - traj_a, dim=2).mean(dim=1)
        dist_sd_b = torch.norm(traj_sd - traj_b, dim=2).mean(dim=1)
        dist_a_b = torch.norm(traj_a - traj_b, dim=2).mean(dim=1)

        ax.plot(dist_sd_mono.numpy(), label='SUPERDIFF vs Monolithic', color='purple', linewidth=2)
        ax.plot(dist_sd_a.numpy(), label='SUPERDIFF vs A', color='blue', linewidth=2)
        ax.plot(dist_sd_b.numpy(), label='SUPERDIFF vs B', color='orange', linewidth=2)
        ax.plot(dist_a_b.numpy(), label='A vs B', color='gray', linestyle='--', linewidth=2)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('L2 Distance', fontsize=11)
        ax.set_title('Trajectory Distances Over Time\n(Run 1 shown)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Velocity magnitudes
        ax = axes[1, 0]
        for run_idx in range(min(3, self.config.num_runs)):  # Show first 3 runs
            for condition, color, label in [
                ('monolithic', 'green', f'Monolithic: "{self.config.prompt_composed}"'),
                ('superdiff', 'red', 'SUPERDIFF: A ∧ B')
            ]:
                traj = self.results[condition]['trajectories'][run_idx]
                # velocities: (num_steps, batch_size, C, H, W)
                vel = traj.velocities.reshape(traj.velocities.shape[0],
                                             traj.velocities.shape[1], -1)
                vel_mag = torch.norm(vel, dim=2).mean(dim=1)

                alpha = 0.4 if run_idx > 0 else 1.0
                ax.plot(vel_mag.numpy(), color=color, alpha=alpha,
                       label=f'{label} (run {run_idx+1})' if run_idx < 2 else None)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Velocity Magnitude', fontsize=11)
        ax.set_title('Vector Field Magnitudes\n(First 3 runs shown)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Trajectory curvature (angle between consecutive velocity vectors)
        ax = axes[1, 1]

        def compute_curvature(velocities):
            # Compute cosine similarity between consecutive velocity vectors
            v = velocities.reshape(velocities.shape[0], velocities.shape[1], -1)
            v_norm = F.normalize(v, p=2, dim=2)
            # Dot product between consecutive steps
            cos_sim = (v_norm[:-1] * v_norm[1:]).sum(dim=2)
            return cos_sim.mean(dim=1)  # Average over batch

        for condition, color, label in [
            ('monolithic', 'green', f'Monolithic: "{self.config.prompt_composed}"'),
            ('superdiff', 'red', 'SUPERDIFF: A ∧ B')
        ]:
            curvatures = []
            for run_idx in range(self.config.num_runs):
                traj = self.results[condition]['trajectories'][run_idx]
                curv = compute_curvature(traj.velocities)
                curvatures.append(curv.numpy())

            curvatures = np.array(curvatures)
            mean_curv = curvatures.mean(axis=0)
            std_curv = curvatures.std(axis=0)

            steps = np.arange(len(mean_curv))
            ax.plot(steps, mean_curv, color=color, label=label, linewidth=2)
            ax.fill_between(steps, mean_curv - std_curv, mean_curv + std_curv,
                           color=color, alpha=0.2)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Cosine Similarity (consecutive velocities)', fontsize=11)
        ax.set_title('Trajectory Smoothness (higher = smoother)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Add overall figure title
        fig.suptitle(f'Trajectory Geometry Analysis\nPrompts: "{self.config.prompt_a}" and "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectory_geometry.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: trajectory_geometry.png")

    def analyze_centroid_statistics(self):
        """Analyze final latent centroids and distributions"""
        print("\nAnalyzing centroid statistics...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Collect final latents
        def get_final_latents(condition):
            latents = [l.cpu().flatten(1) for l in self.results[condition]['latents']]  # Flatten spatial dims
            return torch.cat(latents, dim=0)  # (num_runs * batch_size, latent_dim)

        latents_mono = get_final_latents('monolithic')
        latents_a = get_final_latents('prompt_a')
        latents_b = get_final_latents('prompt_b')
        latents_sd = get_final_latents('superdiff')

        # Compute centroids
        centroid_mono = latents_mono.mean(dim=0)
        centroid_a = latents_a.mean(dim=0)
        centroid_b = latents_b.mean(dim=0)
        centroid_sd = latents_sd.mean(dim=0)

        # Plot 1: Centroid distances
        ax = axes[0, 0]

        distances = {
            'SD to Mono': torch.norm(centroid_sd - centroid_mono).item(),
            'SD to A': torch.norm(centroid_sd - centroid_a).item(),
            'SD to B': torch.norm(centroid_sd - centroid_b).item(),
            'SD to (A+B)/2': torch.norm(centroid_sd - (centroid_a + centroid_b) / 2).item(),
            'A to B': torch.norm(centroid_a - centroid_b).item(),
            'Mono to (A+B)/2': torch.norm(centroid_mono - (centroid_a + centroid_b) / 2).item(),
        }

        bars = ax.bar(range(len(distances)), list(distances.values()),
                     color=['red', 'blue', 'orange', 'purple', 'gray', 'green'])
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(list(distances.keys()), rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('L2 Distance', fontsize=11)
        ax.set_title('Centroid Distances in Latent Space\n(SD=SUPERDIFF, Mono=Monolithic, A/B=Individual Prompts)',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Variance comparison
        ax = axes[0, 1]

        variances = {
            f'Monolithic\n"{self.config.prompt_composed}"': latents_mono.var(dim=0).mean().item(),
            f'Prompt A\n"{self.config.prompt_a}"': latents_a.var(dim=0).mean().item(),
            f'Prompt B\n"{self.config.prompt_b}"': latents_b.var(dim=0).mean().item(),
            'SUPERDIFF\nA ∧ B': latents_sd.var(dim=0).mean().item(),
        }

        bars = ax.bar(range(len(variances)), list(variances.values()),
                     color=['green', 'blue', 'orange', 'red'])
        ax.set_xticks(range(len(variances)))
        ax.set_xticklabels(list(variances.keys()), rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('Mean Variance', fontsize=11)
        ax.set_title('Latent Space Variance Across Conditions', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 3: PCA variance explained (2D projection)
        ax = axes[1, 0]

        # Combine all latents for PCA
        all_latents = torch.cat([latents_mono, latents_a, latents_b, latents_sd], dim=0)
        pca = PCA(n_components=10)
        pca.fit(all_latents.numpy())

        ax.plot(range(1, 11), pca.explained_variance_ratio_, marker='o', linewidth=2, color='royalblue')
        ax.set_xlabel('Principal Component', fontsize=11)
        ax.set_ylabel('Explained Variance Ratio', fontsize=11)
        ax.set_title('PCA: Variance Explained by Top Components', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add cumulative variance text
        cumvar_2d = pca.explained_variance_ratio_[:2].sum()
        cumvar_3d = pca.explained_variance_ratio_[:3].sum()
        ax.text(0.95, 0.95, f'PC1+PC2: {cumvar_2d:.1%}\nPC1+PC2+PC3: {cumvar_3d:.1%}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        # Plot 4: Distribution overlap (1D projection onto PC1)
        ax = axes[1, 1]

        proj_mono = pca.transform(latents_mono.numpy())[:, 0]
        proj_a = pca.transform(latents_a.numpy())[:, 0]
        proj_b = pca.transform(latents_b.numpy())[:, 0]
        proj_sd = pca.transform(latents_sd.numpy())[:, 0]

        ax.hist(proj_mono, bins=30, alpha=0.5, label=f'Monolithic: "{self.config.prompt_composed}"', color='green', density=True)
        ax.hist(proj_a, bins=30, alpha=0.5, label=f'Prompt A: "{self.config.prompt_a}"', color='blue', density=True)
        ax.hist(proj_b, bins=30, alpha=0.5, label=f'Prompt B: "{self.config.prompt_b}"', color='orange', density=True)
        ax.hist(proj_sd, bins=30, alpha=0.5, label='SUPERDIFF: A ∧ B', color='red', density=True)

        ax.set_xlabel('PC1 Projection', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distribution Overlap on First Principal Component', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add overall figure title
        fig.suptitle(f'Centroid Statistics and Distribution Analysis\nPrompts: "{self.config.prompt_a}" and "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'centroid_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: centroid_statistics.png")

        # Save numerical results
        with open(self.output_dir / 'centroid_distances.txt', 'w') as f:
            f.write("CENTROID DISTANCE ANALYSIS\n")
            f.write("="*60 + "\n\n")
            for key, val in distances.items():
                f.write(f"{key:25s}: {val:10.4f}\n")
            f.write("\n\nVARIANCE ANALYSIS\n")
            f.write("="*60 + "\n\n")
            for key, val in variances.items():
                f.write(f"{key:25s}: {val:10.6f}\n")

        print(f"  Saved: centroid_distances.txt")

    def analyze_kappa_dynamics(self):
        """Analyze kappa evolution in SUPERDIFF"""
        print("\nAnalyzing kappa dynamics...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Collect all kappa trajectories
        kappas = torch.stack([k.cpu() for k in self.results['superdiff']['kappas']], dim=0)
        # Shape: (num_runs, num_steps+1, batch_size)

        # Plot 1: Kappa evolution over time
        ax = axes[0, 0]
        for run_idx in range(min(5, self.config.num_runs)):
            kappa_mean = kappas[run_idx].mean(dim=1)  # Average over batch
            alpha = 0.3 if run_idx > 0 else 1.0
            ax.plot(kappa_mean.numpy(), color='royalblue', alpha=alpha,
                   label=f'Run {run_idx+1}' if run_idx < 3 else None)

        # Add mean and std across all runs
        kappa_mean_all = kappas.mean(dim=2).mean(dim=0)  # Average over runs and batch
        kappa_std_all = kappas.mean(dim=2).std(dim=0)

        steps = np.arange(len(kappa_mean_all))
        ax.plot(steps, kappa_mean_all.numpy(), color='darkblue', linewidth=3,
               label='Mean across runs')
        ax.fill_between(steps,
                       (kappa_mean_all - kappa_std_all).numpy(),
                       (kappa_mean_all + kappa_std_all).numpy(),
                       color='blue', alpha=0.2)

        ax.axhline(y=0.5, color='red', linestyle='--', label='κ = 0.5 (equal weight)')
        ax.axhline(y=0.0, color='orange', linestyle='--', label=f'κ = 0 (only B: "{self.config.prompt_b}")')
        ax.axhline(y=1.0, color='green', linestyle='--', label=f'κ = 1 (only A: "{self.config.prompt_a}")')

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('κ (Kappa)', fontsize=11)
        ax.set_title(f'Kappa Evolution: Balance Between A and B\nSUPERDIFF: "{self.config.prompt_a}" ∧ "{self.config.prompt_b}"',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Kappa distribution across time
        ax = axes[0, 1]

        # Flatten kappas for histogram
        kappa_flat = kappas.flatten().numpy()
        ax.hist(kappa_flat, bins=50, color='royalblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='κ = 0.5 (balanced)')
        ax.axvline(x=kappa_flat.mean(), color='darkblue', linestyle='-', linewidth=2,
                  label=f'Mean κ = {kappa_flat.mean():.3f}')

        ax.set_xlabel('κ Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Kappa Values\nAcross All Runs and Timesteps', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Kappa variance over time
        ax = axes[1, 0]

        kappa_var = kappas.var(dim=2).mean(dim=0)  # Variance across batch, mean across runs
        ax.plot(kappa_var.numpy(), color='purple', linewidth=2)
        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Variance of κ', fontsize=11)
        ax.set_title('Kappa Variance: Consistency Across Batch', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 4: Relationship between kappa and log-likelihood difference
        ax = axes[1, 1]

        ll_objs = torch.stack([ll.cpu() for ll in self.results['superdiff']['ll_obj']], dim=0)
        ll_bgs = torch.stack([ll.cpu() for ll in self.results['superdiff']['ll_bg']], dim=0)

        # Use first run for clarity
        ll_diff = (ll_objs[0] - ll_bgs[0]).mean(dim=1).numpy()  # Average over batch
        kappa_run0 = kappas[0].mean(dim=1).numpy()

        ax.scatter(ll_diff, kappa_run0, alpha=0.5, s=20, color='royalblue')
        ax.set_xlabel(f'log p(A) - log p(B)\n(A="{self.config.prompt_a}", B="{self.config.prompt_b}")', fontsize=10)
        ax.set_ylabel('κ', fontsize=11)
        ax.set_title('Kappa vs Log-Likelihood Difference\n(Run 1 shown)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(ll_diff, kappa_run0)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        # Add overall figure title
        fig.suptitle(f'Kappa Dynamics Analysis: SUPERDIFF "{self.config.prompt_a}" ∧ "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'kappa_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: kappa_dynamics.png")

    def analyze_pca_projections(self):
        """Project final latents onto 2D using PCA and t-SNE"""
        print("\nAnalyzing PCA/t-SNE projections...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Collect and label data
        def get_labeled_latents():
            data = []
            labels = []
            colors = []

            for condition, label, color in [
                ('monolithic', f'Monolithic: "{self.config.prompt_composed}"', 'green'),
                ('prompt_a', f'Prompt A: "{self.config.prompt_a}"', 'blue'),
                ('prompt_b', f'Prompt B: "{self.config.prompt_b}"', 'orange'),
                ('superdiff', 'SUPERDIFF: A ∧ B', 'red')
            ]:
                latents = [l.cpu().flatten(1) for l in self.results[condition]['latents']]
                latents = torch.cat(latents, dim=0).numpy()
                data.append(latents)
                labels.extend([label] * len(latents))
                colors.extend([color] * len(latents))

            data = np.vstack(data)
            return data, labels, colors

        data, labels, colors = get_labeled_latents()

        # PCA projection
        ax = axes[0]
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        for label, color in [(f'Monolithic: "{self.config.prompt_composed}"', 'green'),
                            (f'Prompt A: "{self.config.prompt_a}"', 'blue'),
                            (f'Prompt B: "{self.config.prompt_b}"', 'orange'),
                            ('SUPERDIFF: A ∧ B', 'red')]:
            mask = np.array(labels) == label
            ax.scatter(data_pca[mask, 0], data_pca[mask, 1],
                      label=label, color=color, alpha=0.6, s=30)

        # Add centroids
        for label, color, marker in [(f'Monolithic: "{self.config.prompt_composed}"', 'green', 's'),
                                     (f'Prompt A: "{self.config.prompt_a}"', 'blue', '^'),
                                     (f'Prompt B: "{self.config.prompt_b}"', 'orange', 'v'),
                                     ('SUPERDIFF: A ∧ B', 'red', '*')]:
            mask = np.array(labels) == label
            centroid = data_pca[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], color=color, marker=marker,
                      s=300, edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
        ax.set_title('PCA: Latent Space Structure\n(Centroids marked with large symbols)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # t-SNE projection
        ax = axes[1]
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        data_tsne = tsne.fit_transform(data)

        for label, color in [(f'Monolithic: "{self.config.prompt_composed}"', 'green'),
                            (f'Prompt A: "{self.config.prompt_a}"', 'blue'),
                            (f'Prompt B: "{self.config.prompt_b}"', 'orange'),
                            ('SUPERDIFF: A ∧ B', 'red')]:
            mask = np.array(labels) == label
            ax.scatter(data_tsne[mask, 0], data_tsne[mask, 1],
                      label=label, color=color, alpha=0.6, s=30)

        # Add centroids
        for label, color, marker in [(f'Monolithic: "{self.config.prompt_composed}"', 'green', 's'),
                                     (f'Prompt A: "{self.config.prompt_a}"', 'blue', '^'),
                                     (f'Prompt B: "{self.config.prompt_b}"', 'orange', 'v'),
                                     ('SUPERDIFF: A ∧ B', 'red', '*')]:
            mask = np.array(labels) == label
            centroid = data_tsne[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], color=color, marker=marker,
                      s=300, edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.set_title('t-SNE: Latent Space Clustering\n(Centroids marked with large symbols)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add overall figure title
        fig.suptitle(f'PCA and t-SNE Projections: "{self.config.prompt_a}" and "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_tsne_projections.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: pca_tsne_projections.png")

    def analyze_manifold_distances(self):
        """Analyze distances to estimate manifold properties"""
        print("\nAnalyzing manifold distances...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get latents
        latents_mono = torch.cat([l.cpu().flatten(1) for l in self.results['monolithic']['latents']], dim=0).numpy()
        latents_a = torch.cat([l.cpu().flatten(1) for l in self.results['prompt_a']['latents']], dim=0).numpy()
        latents_b = torch.cat([l.cpu().flatten(1) for l in self.results['prompt_b']['latents']], dim=0).numpy()
        latents_sd = torch.cat([l.cpu().flatten(1) for l in self.results['superdiff']['latents']], dim=0).numpy()

        # Plot 1: Nearest neighbor distances (manifold density estimate)
        ax = axes[0, 0]

        def compute_nn_distances(data, k=5):
            """Compute k-nearest neighbor distances"""
            dists = cdist(data, data, metric='euclidean')
            np.fill_diagonal(dists, np.inf)
            nn_dists = np.sort(dists, axis=1)[:, :k]
            return nn_dists.mean(axis=1)

        nn_mono = compute_nn_distances(latents_mono)
        nn_a = compute_nn_distances(latents_a)
        nn_b = compute_nn_distances(latents_b)
        nn_sd = compute_nn_distances(latents_sd)

        ax.hist(nn_mono, bins=30, alpha=0.5, label=f'Monolithic: "{self.config.prompt_composed}"', color='green', density=True)
        ax.hist(nn_a, bins=30, alpha=0.5, label=f'Prompt A: "{self.config.prompt_a}"', color='blue', density=True)
        ax.hist(nn_b, bins=30, alpha=0.5, label=f'Prompt B: "{self.config.prompt_b}"', color='orange', density=True)
        ax.hist(nn_sd, bins=30, alpha=0.5, label='SUPERDIFF: A ∧ B', color='red', density=True)

        ax.set_xlabel('Mean k-NN Distance', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Manifold Density: Nearest Neighbor Distances\n(k=5)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Distance to centroid of other conditions
        ax = axes[0, 1]

        centroid_mono = latents_mono.mean(axis=0)
        centroid_a = latents_a.mean(axis=0)
        centroid_b = latents_b.mean(axis=0)

        dist_sd_to_mono = np.linalg.norm(latents_sd - centroid_mono, axis=1)
        dist_sd_to_a = np.linalg.norm(latents_sd - centroid_a, axis=1)
        dist_sd_to_b = np.linalg.norm(latents_sd - centroid_b, axis=1)

        ax.violinplot([dist_sd_to_mono, dist_sd_to_a, dist_sd_to_b],
                     positions=[1, 2, 3],
                     showmeans=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['To Monolithic\ncentroid', 'To A\ncentroid', 'To B\ncentroid'], fontsize=10)
        ax.set_ylabel('Distance', fontsize=11)
        ax.set_title('SUPERDIFF Distance to Other Centroids\n(Distribution of samples)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Interpolation analysis
        ax = axes[1, 0]

        # Linear interpolation between A and B centroids
        alphas = np.linspace(0, 1, 100)
        interpolated = np.array([alpha * centroid_a + (1 - alpha) * centroid_b
                                for alpha in alphas])

        # Distance from SUPERDIFF samples to interpolation line
        dist_to_interp = []
        for sd_sample in latents_sd:
            # Find closest point on interpolation line
            dists = np.linalg.norm(interpolated - sd_sample, axis=1)
            dist_to_interp.append(dists.min())

        # Distance from monolithic samples to interpolation line
        dist_mono_to_interp = []
        for mono_sample in latents_mono:
            dists = np.linalg.norm(interpolated - mono_sample, axis=1)
            dist_mono_to_interp.append(dists.min())

        ax.hist(dist_to_interp, bins=30, alpha=0.6, label='SUPERDIFF: A ∧ B', color='red', density=True)
        ax.hist(dist_mono_to_interp, bins=30, alpha=0.6, label=f'Monolithic: "{self.config.prompt_composed}"', color='green', density=True)
        ax.set_xlabel('Distance to A-B Interpolation Line', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distance to Linear Interpolation Between A and B\n(Tests linear interpolation hypothesis)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Convex hull membership
        ax = axes[1, 1]

        # Project to 2D for visualization
        pca = PCA(n_components=2)
        all_data = np.vstack([latents_mono, latents_a, latents_b, latents_sd])
        all_projected = pca.fit_transform(all_data)

        n_mono = len(latents_mono)
        n_a = len(latents_a)
        n_b = len(latents_b)

        proj_mono = all_projected[:n_mono]
        proj_a = all_projected[n_mono:n_mono+n_a]
        proj_b = all_projected[n_mono+n_a:n_mono+n_a+n_b]
        proj_sd = all_projected[n_mono+n_a+n_b:]

        ax.scatter(proj_a[:, 0], proj_a[:, 1], color='blue', alpha=0.5, s=30, label=f'A: "{self.config.prompt_a}"')
        ax.scatter(proj_b[:, 0], proj_b[:, 1], color='orange', alpha=0.5, s=30, label=f'B: "{self.config.prompt_b}"')
        ax.scatter(proj_mono[:, 0], proj_mono[:, 1], color='green', alpha=0.5, s=30, label='Monolithic')
        ax.scatter(proj_sd[:, 0], proj_sd[:, 1], color='red', alpha=0.8, s=50,
                  marker='*', label='SUPERDIFF', edgecolors='black', linewidths=0.5)

        # Draw line between A and B centroids
        centroid_a_2d = proj_a.mean(axis=0)
        centroid_b_2d = proj_b.mean(axis=0)
        ax.plot([centroid_a_2d[0], centroid_b_2d[0]],
               [centroid_a_2d[1], centroid_b_2d[1]],
               'k--', linewidth=2, label='A-B centroid line')

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('2D Projection: Relative Positions\n(Assesses geometric relationship)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add overall figure title
        fig.suptitle(f'Manifold Distances Analysis: "{self.config.prompt_a}" and "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'manifold_distances.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: manifold_distances.png")

    def analyze_velocity_field_alignment(self):
        """Analyze alignment between velocity fields"""
        print("\nAnalyzing velocity field alignment...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Use first run for detailed analysis
        run_idx = 0
        traj_mono = self.results['monolithic']['trajectories'][run_idx]
        traj_a = self.results['prompt_a']['trajectories'][run_idx]
        traj_b = self.results['prompt_b']['trajectories'][run_idx]
        traj_sd = self.results['superdiff']['trajectories'][run_idx]

        def flatten_velocities(traj):
            return traj.velocities.reshape(traj.velocities.shape[0],
                                          traj.velocities.shape[1], -1)

        vel_mono = flatten_velocities(traj_mono)
        vel_a = flatten_velocities(traj_a)
        vel_b = flatten_velocities(traj_b)
        vel_sd = flatten_velocities(traj_sd)

        # Plot 1: Cosine similarity over time
        ax = axes[0, 0]

        def cosine_similarity_time(v1, v2):
            v1_norm = F.normalize(v1, p=2, dim=2)
            v2_norm = F.normalize(v2, p=2, dim=2)
            cos_sim = (v1_norm * v2_norm).sum(dim=2)
            return cos_sim.mean(dim=1)  # Average over batch

        cos_sd_mono = cosine_similarity_time(vel_sd, vel_mono)
        cos_sd_a = cosine_similarity_time(vel_sd, vel_a)
        cos_sd_b = cosine_similarity_time(vel_sd, vel_b)
        cos_a_b = cosine_similarity_time(vel_a, vel_b)

        steps = np.arange(len(cos_sd_mono))
        ax.plot(steps, cos_sd_mono.numpy(), label='SUPERDIFF vs Monolithic', color='purple', linewidth=2)
        ax.plot(steps, cos_sd_a.numpy(), label='SUPERDIFF vs A', color='blue', linewidth=2)
        ax.plot(steps, cos_sd_b.numpy(), label='SUPERDIFF vs B', color='orange', linewidth=2)
        ax.plot(steps, cos_a_b.numpy(), label='A vs B', color='gray', linestyle='--', linewidth=2)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title('Velocity Field Alignment Over Time\n(Run 1 shown)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Plot 2: Velocity magnitude ratios
        ax = axes[0, 1]

        mag_mono = torch.norm(vel_mono, dim=2).mean(dim=1)
        mag_a = torch.norm(vel_a, dim=2).mean(dim=1)
        mag_b = torch.norm(vel_b, dim=2).mean(dim=1)
        mag_sd = torch.norm(vel_sd, dim=2).mean(dim=1)

        ax.plot(steps, mag_sd.numpy() / mag_mono.numpy(), label='SUPERDIFF / Monolithic',
               color='purple', linewidth=2)
        ax.plot(steps, mag_sd.numpy() / mag_a.numpy(), label='SUPERDIFF / A',
               color='blue', linewidth=2)
        ax.plot(steps, mag_sd.numpy() / mag_b.numpy(), label='SUPERDIFF / B',
               color='orange', linewidth=2)

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Ratio = 1')
        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Magnitude Ratio', fontsize=11)
        ax.set_title('Velocity Magnitude Ratios\n(Relative strength of vector fields)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Angular divergence (angle between velocity fields)
        ax = axes[1, 0]

        def angular_divergence(v1, v2):
            cos_sim = cosine_similarity_time(v1, v2)
            angles = torch.acos(cos_sim.clamp(-1, 1)) * 180 / np.pi
            return angles

        angle_sd_mono = angular_divergence(vel_sd, vel_mono)
        angle_sd_a = angular_divergence(vel_sd, vel_a)
        angle_sd_b = angular_divergence(vel_sd, vel_b)

        ax.plot(steps, angle_sd_mono.numpy(), label='SUPERDIFF vs Monolithic', color='purple', linewidth=2)
        ax.plot(steps, angle_sd_a.numpy(), label='SUPERDIFF vs A', color='blue', linewidth=2)
        ax.plot(steps, angle_sd_b.numpy(), label='SUPERDIFF vs B', color='orange', linewidth=2)

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('Angle (degrees)', fontsize=11)
        ax.set_title('Angular Divergence Between Velocity Fields\n(90° = orthogonal)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Orthogonal')

        # Plot 4: Velocity composition test
        ax = axes[1, 1]

        # Test if SD velocity is a weighted combination of A and B
        # v_sd ≈ α·v_a + (1-α)·v_b
        # Solve for α at each timestep

        alphas = []
        for t in range(len(vel_sd)):
            v_sd_t = vel_sd[t].mean(dim=0)  # Average over batch
            v_a_t = vel_a[t].mean(dim=0)
            v_b_t = vel_b[t].mean(dim=0)

            # Least squares: find α that minimizes ||v_sd - α·v_a - (1-α)·v_b||²
            # This reduces to: α = (v_sd·(v_a - v_b)) / ||v_a - v_b||²
            diff = v_a_t - v_b_t
            if torch.norm(diff) > 1e-6:
                alpha = torch.dot(v_sd_t, diff) / torch.dot(diff, diff)
                alphas.append(alpha.item())
            else:
                alphas.append(0.5)

        ax.plot(steps, alphas, color='royalblue', linewidth=2, label='Estimated α')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label=f'α = 0.5 (equal weight)')
        ax.axhline(y=0.0, color='orange', linestyle='--', alpha=0.5, label=f'α = 0 (only B: "{self.config.prompt_b}")')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label=f'α = 1 (only A: "{self.config.prompt_a}")')

        ax.set_xlabel('Diffusion Step', fontsize=11)
        ax.set_ylabel('α (weight on A)', fontsize=11)
        ax.set_title('Linear Decomposition: v_SD ≈ α·v_A + (1-α)·v_B\n(Tests velocity field composition)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 1.5])

        # Add overall figure title
        fig.suptitle(f'Velocity Field Alignment Analysis\nSUPERDIFF: "{self.config.prompt_a}" ∧ "{self.config.prompt_b}"',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'velocity_field_alignment.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: velocity_field_alignment.png")

    def generate_summary_report(self):
        """Generate comprehensive text summary"""
        print("\nGenerating summary report...")

        # Collect key statistics
        latents_mono = torch.cat([l.cpu().flatten(1) for l in self.results['monolithic']['latents']], dim=0)
        latents_a = torch.cat([l.cpu().flatten(1) for l in self.results['prompt_a']['latents']], dim=0)
        latents_b = torch.cat([l.cpu().flatten(1) for l in self.results['prompt_b']['latents']], dim=0)
        latents_sd = torch.cat([l.cpu().flatten(1) for l in self.results['superdiff']['latents']], dim=0)

        centroid_mono = latents_mono.mean(dim=0)
        centroid_a = latents_a.mean(dim=0)
        centroid_b = latents_b.mean(dim=0)
        centroid_sd = latents_sd.mean(dim=0)
        centroid_midpoint = (centroid_a + centroid_b) / 2

        kappas = torch.stack([k.cpu() for k in self.results['superdiff']['kappas']], dim=0)
        kappa_mean = kappas.mean().item()
        kappa_std = kappas.std().item()

        report = f"""
{'='*80}
SUPERDIFF COMPOSITION ANALYSIS - SUMMARY REPORT
{'='*80}

EXPERIMENTAL CONFIGURATION
{'='*80}
Prompts:
  A: "{self.config.prompt_a}"
  B: "{self.config.prompt_b}"
  Monolithic: "{self.config.prompt_composed}"

Sampling:
  Number of runs: {self.config.num_runs}
  Batch size: {self.config.batch_size}
  Total samples per condition: {self.config.num_runs * self.config.batch_size}
  Inference steps: {self.config.num_inference_steps}
  Guidance scale: {self.config.guidance_scale}
  SUPERDIFF lift parameter: {self.config.lift}

{'='*80}
KEY FINDINGS
{'='*80}

1. CENTROID ANALYSIS
{'='*80}

Distance metrics (L2 norm in latent space):

  SUPERDIFF positioning:
    - Distance to Monolithic:     {torch.norm(centroid_sd - centroid_mono).item():10.4f}
    - Distance to A:              {torch.norm(centroid_sd - centroid_a).item():10.4f}
    - Distance to B:              {torch.norm(centroid_sd - centroid_b).item():10.4f}
    - Distance to (A+B)/2:        {torch.norm(centroid_sd - centroid_midpoint).item():10.4f}

  Reference distances:
    - A to B:                     {torch.norm(centroid_a - centroid_b).item():10.4f}
    - Monolithic to (A+B)/2:      {torch.norm(centroid_mono - centroid_midpoint).item():10.4f}

  Interpretation:
    - SUPERDIFF centroid is {"CLOSER" if torch.norm(centroid_sd - centroid_midpoint) < torch.norm(centroid_sd - centroid_mono) else "FARTHER"} to the linear midpoint (A+B)/2
      than to the monolithic prompt.
    - This {"suggests" if torch.norm(centroid_sd - centroid_midpoint) < torch.norm(centroid_sd - centroid_mono) else "does not suggest"} that SUPERDIFF performs approximate linear interpolation in latent space.

2. KAPPA DYNAMICS
{'='*80}

  Mean κ across all steps:      {kappa_mean:10.4f}
  Std deviation of κ:            {kappa_std:10.4f}

  Interpretation:
    - κ = 0.5 represents equal weighting between A and B
    - κ > 0.5 biases toward A ("{self.config.prompt_a}")
    - κ < 0.5 biases toward B ("{self.config.prompt_b}")

    Mean κ = {kappa_mean:.4f} indicates {"balanced" if abs(kappa_mean - 0.5) < 0.1 else f"{'A-biased' if kappa_mean > 0.5 else 'B-biased'}"} composition.

3. VARIANCE ANALYSIS
{'='*80}

  Mean variance across dimensions:
    - Monolithic:  {latents_mono.var(dim=0).mean().item():.6f}
    - Prompt A:    {latents_a.var(dim=0).mean().item():.6f}
    - Prompt B:    {latents_b.var(dim=0).mean().item():.6f}
    - SUPERDIFF:   {latents_sd.var(dim=0).mean().item():.6f}

  Interpretation:
    {"Higher variance in SUPERDIFF suggests greater diversity/uncertainty in composition." if latents_sd.var(dim=0).mean() > latents_mono.var(dim=0).mean() else "Lower variance in SUPERDIFF suggests mode collapse or reduced diversity."}

{'='*80}
THEORETICAL IMPLICATIONS
{'='*80}

Based on the analysis, we can draw the following conclusions:

1. GEOMETRIC INTERPRETATION:

   The SUPERDIFF AND operation appears to {"perform approximate linear interpolation" if torch.norm(centroid_sd - centroid_midpoint) < 0.3 * torch.norm(centroid_a - centroid_b) else "deviate significantly from linear interpolation"}
   in the latent manifold. This {"supports" if torch.norm(centroid_sd - centroid_midpoint) < 0.3 * torch.norm(centroid_a - centroid_b) else "challenges"} the hypothesis that composition
   occurs via Euclidean averaging of velocity fields.

2. MANIFOLD HYPOTHESIS:

   {"The proximity of SUPERDIFF samples to the A-B interpolation line suggests composition   stays approximately on-manifold, reducing the likelihood of off-manifold artifacts." if torch.norm(centroid_sd - centroid_midpoint) < 0.3 * torch.norm(centroid_a - centroid_b) else "The deviation of SUPERDIFF samples from the A-B interpolation line suggests potential   off-manifold trajectories, which could explain hybridization artifacts."}

3. SEMANTIC INTERPRETATION:

   The observed {"similarity" if torch.norm(centroid_sd - centroid_mono) < 0.5 * torch.norm(centroid_a - centroid_b) else "dissimilarity"} between SUPERDIFF and monolithic prompts
   indicates that the mathematical AND operation {"aligns with" if torch.norm(centroid_sd - centroid_mono) < 0.5 * torch.norm(centroid_a - centroid_b) else "differs from"} natural language conjunction semantics.

   This {"validates" if torch.norm(centroid_sd - centroid_mono) < 0.5 * torch.norm(centroid_a - centroid_b) else "challenges"} the hypothesis that hybridization is a consequence of energy-based
   intersection rather than semantic co-presence.

{'='*80}
RECOMMENDATIONS FOR FURTHER INVESTIGATION
{'='*80}

1. Vary the lift parameter systematically to observe its effect on composition geometry
2. Test with prompts that have clear spatial semantics (e.g., "left" vs "right")
3. Analyze intermediate timesteps to identify when hybridization emerges
4. Compare with alternative composition operators (arithmetic mean, geometric mean)
5. Investigate whether different prompt embeddings lead to more manifold-aligned compositions
6. Test with CLIP-based semantic similarity metrics to quantify hybridization

{'='*80}
EXPERIMENTAL DATA SAVED TO:
{'='*80}

{self.output_dir.absolute()}

Files generated:
  - sample_images_comparison.png
  - trajectory_geometry.png
  - centroid_statistics.png
  - kappa_dynamics.png
  - pca_tsne_projections.png
  - manifold_distances.png
  - velocity_field_alignment.png
  - centroid_distances.txt
  - summary_report.txt (this file)

{'='*80}
END OF REPORT
{'='*80}
"""

        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report)

        print(report)
        print(f"\n  Saved: summary_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_composition_experiments(config: Optional[ExperimentConfig] = None):
    """
    Main entry point for running composition experiments

    Args:
        config: Optional custom configuration. If None, uses default config.

    Example:
        >>> config = ExperimentConfig(
        ...     prompt_a="A photograph of a cat",
        ...     prompt_b="A photograph of a dog",
        ...     prompt_composed="A photograph of a cat and a dog",
        ...     num_runs=10,
        ...     num_inference_steps=500
        ... )
        >>> run_composition_experiments(config)
    """
    if config is None:
        config = ExperimentConfig()

    suite = CompositionExperimentSuite(config)
    suite.run_all_experiments()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {suite.output_dir.absolute()}")
    print("\nReview the generated visualizations and summary report for detailed analysis.")


if __name__ == "__main__":
    # Example usage with custom configuration
    config = ExperimentConfig(
        prompt_a="A photograph of a cat",
        prompt_b="A photograph of a dog",
        prompt_composed="A photograph of a cat and a dog",
        num_runs=20,
        batch_size=4,
        num_inference_steps=500,
        guidance_scale=7.5,
        lift=0.0,
        output_dir="experiments/cat_dog_composition"
    )

    run_composition_experiments(config)
