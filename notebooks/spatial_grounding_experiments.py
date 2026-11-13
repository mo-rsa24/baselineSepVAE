"""
Spatial Grounding Experiments for SUPERDIFF Composition

This module tests the critical hypothesis:
Does SUPERDIFF hybridization result from LACK OF SPATIAL GROUNDING
or from fundamental GEOMETRIC/MATHEMATICAL limitations?

Experimental Design:
====================

1. Semantic SUPERDIFF: "cat" ∧ "dog"
   - No spatial constraints
   - Tests pure probability product behavior

2. Spatial SUPERDIFF: "cat on left" ∧ "dog on right"
   - Explicit spatial grounding
   - Tests if spatial constraints enable co-presence

3. Monolithic Semantic: "a cat and a dog"
   - Baseline for vague semantic conjunction

4. Monolithic Spatial: "a cat on the left and a dog on the right"
   - Baseline for explicit spatial composition

Key Questions:
==============
- Does spatial grounding eliminate hybridization?
- If yes: Issue is lack of spatial inductive bias
- If no: Issue is geometric/mathematical (off-manifold, probability product)

This distinguishes between:
- Architectural limitations (UNet lacks spatial reasoning)
- Mathematical limitations (probability product induces interpolation)
- Geometric artifacts (off-manifold trajectories)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List

from notebooks.composition_experiments import (
    ExperimentConfig,
    CompositionExperimentSuite
)


@dataclass
class SpatialGroundingConfig:
    """Configuration for spatial grounding experiments"""

    # Semantic prompts (no spatial constraints)
    semantic_a: str = "a cat"
    semantic_b: str = "a dog"
    semantic_composed: str = "a cat and a dog"

    # Spatial prompts (explicit positioning)
    spatial_a: str = "a cat on the left side"
    spatial_b: str = "a dog on the right side"
    spatial_composed: str = "a cat on the left side and a dog on the right side"

    # Sampling parameters
    num_runs: int = 15
    batch_size: int = 4
    num_inference_steps: int = 500
    guidance_scale: float = 7.5
    lift: float = 0.0

    # Output
    output_dir: str = "experiments/spatial_grounding_analysis"

    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class SpatialGroundingExperimentSuite:
    """
    Comprehensive comparison of semantic vs. spatial SUPERDIFF composition
    """

    def __init__(self, config: SpatialGroundingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for both experiment suites
        self.semantic_suite = None
        self.spatial_suite = None

    def run_all_experiments(self):
        """Run both semantic and spatial experiments"""

        print("\n" + "="*80)
        print("SPATIAL GROUNDING ANALYSIS - EXPERIMENTAL SUITE")
        print("="*80)
        print("\nThis experiment tests whether spatial grounding eliminates hybridization")
        print("by comparing semantic vs. spatial SUPERDIFF composition.\n")

        # Experiment 1: Semantic SUPERDIFF (no spatial constraints)
        print("="*80)
        print("EXPERIMENT 1: SEMANTIC SUPERDIFF (No Spatial Grounding)")
        print("="*80)

        semantic_config = ExperimentConfig(
            prompt_a=self.config.semantic_a,
            prompt_b=self.config.semantic_b,
            prompt_composed=self.config.semantic_composed,
            num_runs=self.config.num_runs,
            batch_size=self.config.batch_size,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            lift=self.config.lift,
            output_dir=str(self.output_dir / "semantic")
        )

        self.semantic_suite = CompositionExperimentSuite(semantic_config)
        self.semantic_suite.run_all_experiments()

        # Experiment 2: Spatial SUPERDIFF (explicit spatial grounding)
        print("\n" + "="*80)
        print("EXPERIMENT 2: SPATIAL SUPERDIFF (Explicit Spatial Grounding)")
        print("="*80)

        spatial_config = ExperimentConfig(
            prompt_a=self.config.spatial_a,
            prompt_b=self.config.spatial_b,
            prompt_composed=self.config.spatial_composed,
            num_runs=self.config.num_runs,
            batch_size=self.config.batch_size,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            lift=self.config.lift,
            output_dir=str(self.output_dir / "spatial")
        )

        self.spatial_suite = CompositionExperimentSuite(spatial_config)
        self.spatial_suite.run_all_experiments()

        # Comparative analysis
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS: Semantic vs. Spatial")
        print("="*80)

        self.generate_comparative_analysis()

    def generate_comparative_analysis(self):
        """Generate comparative visualizations and metrics"""

        print("\nGenerating comparative analysis...")

        # 1. Side-by-side image comparison
        self._compare_sample_images()

        # 2. Geometric comparison
        self._compare_geometry()

        # 3. Kappa dynamics comparison
        self._compare_kappa_dynamics()

        # 4. Success metrics
        self._compute_success_metrics()

        # 5. Final report
        self._generate_comparative_report()

    def _compare_sample_images(self):
        """Generate side-by-side comparison of semantic vs. spatial outputs"""
        print("  - Generating image comparison...")

        from notebooks.utils import get_image

        fig, axes = plt.subplots(4, min(8, self.config.num_runs),
                                figsize=(2.5*min(8, self.config.num_runs), 12))

        n_display = min(8, self.config.num_runs)

        # Create detailed labels with actual prompts
        row_labels = [
            f'Semantic\nMonolithic\n"{self.config.semantic_composed}"',
            f'Semantic\nSUPERDIFF\n"{self.config.semantic_a}" ∧\n"{self.config.semantic_b}"',
            f'Spatial\nMonolithic\n"{self.config.spatial_composed}"',
            f'Spatial\nSUPERDIFF\n"{self.config.spatial_a}" ∧\n"{self.config.spatial_b}"'
        ]

        for run_idx in range(n_display):
            # Row 0: Semantic monolithic
            latents = self.semantic_suite.results['monolithic']['latents'][run_idx][0:1]
            img = get_image(self.semantic_suite.vae, latents, nrow=1, ncol=1)
            axes[0, run_idx].imshow(img)
            axes[0, run_idx].axis('off')
            if run_idx == 0:
                axes[0, run_idx].set_ylabel(row_labels[0], fontsize=8,
                                           rotation=0, ha='right', va='center',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # Row 1: Semantic SUPERDIFF
            latents = self.semantic_suite.results['superdiff']['latents'][run_idx][0:1]
            img = get_image(self.semantic_suite.vae, latents, nrow=1, ncol=1)
            axes[1, run_idx].imshow(img)
            axes[1, run_idx].axis('off')
            if run_idx == 0:
                axes[1, run_idx].set_ylabel(row_labels[1], fontsize=8,
                                           rotation=0, ha='right', va='center',
                                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

            # Row 2: Spatial monolithic
            latents = self.spatial_suite.results['monolithic']['latents'][run_idx][0:1]
            img = get_image(self.spatial_suite.vae, latents, nrow=1, ncol=1)
            axes[2, run_idx].imshow(img)
            axes[2, run_idx].axis('off')
            if run_idx == 0:
                axes[2, run_idx].set_ylabel(row_labels[2], fontsize=8,
                                           rotation=0, ha='right', va='center',
                                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

            # Row 3: Spatial SUPERDIFF
            latents = self.spatial_suite.results['superdiff']['latents'][run_idx][0:1]
            img = get_image(self.spatial_suite.vae, latents, nrow=1, ncol=1)
            axes[3, run_idx].imshow(img)
            axes[3, run_idx].axis('off')
            if run_idx == 0:
                axes[3, run_idx].set_ylabel(row_labels[3], fontsize=8,
                                           rotation=0, ha='right', va='center',
                                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

            # Column titles
            if run_idx < n_display:
                axes[0, run_idx].set_title(f'Run {run_idx+1}', fontsize=10, fontweight='bold')

        # Add overall title
        fig.suptitle('Semantic vs Spatial SUPERDIFF Comparison\n' +
                    'Critical Test: Does spatial grounding eliminate hybridization?',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'semantic_vs_spatial_comparison.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved: semantic_vs_spatial_comparison.png")

    def _compare_geometry(self):
        """Compare geometric properties of semantic vs. spatial SUPERDIFF"""
        print("  - Comparing geometric properties...")

        # Extract latents
        semantic_mono = torch.cat([l.cpu().flatten(1) for l in
                                  self.semantic_suite.results['monolithic']['latents']], dim=0)
        semantic_sd = torch.cat([l.cpu().flatten(1) for l in
                                self.semantic_suite.results['superdiff']['latents']], dim=0)
        semantic_a = torch.cat([l.cpu().flatten(1) for l in
                               self.semantic_suite.results['prompt_a']['latents']], dim=0)
        semantic_b = torch.cat([l.cpu().flatten(1) for l in
                               self.semantic_suite.results['prompt_b']['latents']], dim=0)

        spatial_mono = torch.cat([l.cpu().flatten(1) for l in
                                 self.spatial_suite.results['monolithic']['latents']], dim=0)
        spatial_sd = torch.cat([l.cpu().flatten(1) for l in
                               self.spatial_suite.results['superdiff']['latents']], dim=0)
        spatial_a = torch.cat([l.cpu().flatten(1) for l in
                              self.spatial_suite.results['prompt_a']['latents']], dim=0)
        spatial_b = torch.cat([l.cpu().flatten(1) for l in
                              self.spatial_suite.results['prompt_b']['latents']], dim=0)

        # Compute centroids
        semantic_centroid_sd = semantic_sd.mean(dim=0)
        semantic_centroid_mono = semantic_mono.mean(dim=0)
        semantic_centroid_mid = (semantic_a.mean(dim=0) + semantic_b.mean(dim=0)) / 2
        semantic_dist_a_b = torch.norm(semantic_a.mean(dim=0) - semantic_b.mean(dim=0)).item()

        spatial_centroid_sd = spatial_sd.mean(dim=0)
        spatial_centroid_mono = spatial_mono.mean(dim=0)
        spatial_centroid_mid = (spatial_a.mean(dim=0) + spatial_b.mean(dim=0)) / 2
        spatial_dist_a_b = torch.norm(spatial_a.mean(dim=0) - spatial_b.mean(dim=0)).item()

        # Key distances
        semantic_dist_sd_mid = torch.norm(semantic_centroid_sd - semantic_centroid_mid).item()
        semantic_dist_sd_mono = torch.norm(semantic_centroid_sd - semantic_centroid_mono).item()

        spatial_dist_sd_mid = torch.norm(spatial_centroid_sd - spatial_centroid_mid).item()
        spatial_dist_sd_mono = torch.norm(spatial_centroid_sd - spatial_centroid_mono).item()

        # Normalized metrics
        semantic_normalized = semantic_dist_sd_mid / semantic_dist_a_b
        spatial_normalized = spatial_dist_sd_mid / spatial_dist_a_b

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Distance to midpoint comparison
        ax = axes[0]
        conditions = ['Semantic\nSUPERDIFF', 'Spatial\nSUPERDIFF']
        distances = [semantic_dist_sd_mid, spatial_dist_sd_mid]
        colors = ['coral', 'steelblue']

        bars = ax.bar(conditions, distances, color=colors, alpha=0.7, edgecolor='black')
        for i, (bar, dist) in enumerate(zip(bars, distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dist:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Distance to (A+B)/2', fontsize=12)
        ax.set_title('Linear Interpolation Test\n(Lower = More Linear)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Distance to monolithic comparison
        ax = axes[1]
        distances_mono = [semantic_dist_sd_mono, spatial_dist_sd_mono]

        bars = ax.bar(conditions, distances_mono, color=colors, alpha=0.7, edgecolor='black')
        for i, (bar, dist) in enumerate(zip(bars, distances_mono)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dist:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Distance to Monolithic Prompt', fontsize=12)
        ax.set_title('Semantic Alignment Test\n(Lower = Better Match)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Normalized distance comparison
        ax = axes[2]
        normalized_distances = [semantic_normalized, spatial_normalized]

        bars = ax.bar(conditions, normalized_distances, color=colors, alpha=0.7, edgecolor='black')
        for i, (bar, dist) in enumerate(zip(bars, normalized_distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dist:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, alpha=0.5,
                  label='Linear threshold (0.3)')
        ax.set_ylabel('Normalized Distance to Midpoint', fontsize=12)
        ax.set_title('Relative Interpolation\n(< 0.3 = Linear)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'geometric_comparison.png', dpi=150)
        plt.close()

        print(f"    Saved: geometric_comparison.png")

        # Store for report
        self.geometry_metrics = {
            'semantic': {
                'dist_sd_mid': semantic_dist_sd_mid,
                'dist_sd_mono': semantic_dist_sd_mono,
                'normalized': semantic_normalized
            },
            'spatial': {
                'dist_sd_mid': spatial_dist_sd_mid,
                'dist_sd_mono': spatial_dist_sd_mono,
                'normalized': spatial_normalized
            }
        }

    def _compare_kappa_dynamics(self):
        """Compare kappa evolution in semantic vs. spatial SUPERDIFF"""
        print("  - Comparing kappa dynamics...")

        semantic_kappas = torch.stack([k.cpu() for k in
                                      self.semantic_suite.results['superdiff']['kappas']], dim=0)
        spatial_kappas = torch.stack([k.cpu() for k in
                                     self.spatial_suite.results['superdiff']['kappas']], dim=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Kappa evolution over time
        ax = axes[0]

        semantic_kappa_mean = semantic_kappas.mean(dim=2).mean(dim=0)
        semantic_kappa_std = semantic_kappas.mean(dim=2).std(dim=0)

        spatial_kappa_mean = spatial_kappas.mean(dim=2).mean(dim=0)
        spatial_kappa_std = spatial_kappas.mean(dim=2).std(dim=0)

        steps = np.arange(len(semantic_kappa_mean))

        ax.plot(steps, semantic_kappa_mean.numpy(), color='coral', linewidth=2.5,
               label='Semantic')
        ax.fill_between(steps,
                       (semantic_kappa_mean - semantic_kappa_std).numpy(),
                       (semantic_kappa_mean + semantic_kappa_std).numpy(),
                       color='coral', alpha=0.2)

        ax.plot(steps, spatial_kappa_mean.numpy(), color='steelblue', linewidth=2.5,
               label='Spatial')
        ax.fill_between(steps,
                       (spatial_kappa_mean - spatial_kappa_std).numpy(),
                       (spatial_kappa_mean + spatial_kappa_std).numpy(),
                       color='steelblue', alpha=0.2)

        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='κ = 0.5 (balanced)')

        ax.set_xlabel('Diffusion Step', fontsize=12)
        ax.set_ylabel('κ (Balance Parameter)', fontsize=12)
        ax.set_title('Kappa Evolution: Semantic vs. Spatial', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution comparison
        ax = axes[1]

        semantic_kappa_flat = semantic_kappas.flatten().numpy()
        spatial_kappa_flat = spatial_kappas.flatten().numpy()

        ax.hist(semantic_kappa_flat, bins=50, alpha=0.6, color='coral',
               label=f'Semantic (μ={semantic_kappa_flat.mean():.3f})', density=True)
        ax.hist(spatial_kappa_flat, bins=50, alpha=0.6, color='steelblue',
               label=f'Spatial (μ={spatial_kappa_flat.mean():.3f})', density=True)

        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='κ = 0.5')

        ax.set_xlabel('κ Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Kappa Distribution Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'kappa_comparison.png', dpi=150)
        plt.close()

        print(f"    Saved: kappa_comparison.png")

        # Store for report
        self.kappa_metrics = {
            'semantic_mean': semantic_kappa_flat.mean(),
            'semantic_std': semantic_kappa_flat.std(),
            'spatial_mean': spatial_kappa_flat.mean(),
            'spatial_std': spatial_kappa_flat.std()
        }

    def _compute_success_metrics(self):
        """Compute success metrics for co-presence vs. hybridization"""
        print("  - Computing success metrics...")

        # Note: This is a placeholder for future object detection-based metrics
        # In practice, you would:
        # 1. Run object detection on generated images
        # 2. Count number of detected objects (cat, dog)
        # 3. Measure spatial separation
        # 4. Score as "success" if both objects detected and separated

        print("    [Note: Quantitative object detection metrics require external models]")
        print("    [Manual inspection of sample_images_comparison.png recommended]")

        self.success_metrics = {
            'note': 'Visual inspection required - see semantic_vs_spatial_comparison.png'
        }

    def _generate_comparative_report(self):
        """Generate comprehensive comparison report"""
        print("  - Generating comparative report...")

        report = f"""
{'='*80}
SPATIAL GROUNDING ANALYSIS - COMPARATIVE REPORT
{'='*80}

EXPERIMENTAL QUESTION:
Does spatial grounding eliminate SUPERDIFF hybridization?

PROMPTS TESTED:
{'='*80}

Semantic Prompts (No Spatial Constraints):
  A:          "{self.config.semantic_a}"
  B:          "{self.config.semantic_b}"
  Monolithic: "{self.config.semantic_composed}"
  SUPERDIFF:  A ∧ B

Spatial Prompts (Explicit Positioning):
  A:          "{self.config.spatial_a}"
  B:          "{self.config.spatial_b}"
  Monolithic: "{self.config.spatial_composed}"
  SUPERDIFF:  A ∧ B

SAMPLING PARAMETERS:
{'='*80}
  Runs:            {self.config.num_runs}
  Batch size:      {self.config.batch_size}
  Steps:           {self.config.num_inference_steps}
  Guidance scale:  {self.config.guidance_scale}
  Lift:            {self.config.lift}

GEOMETRIC ANALYSIS:
{'='*80}

1. Linear Interpolation Test (Distance to (A+B)/2):

  Semantic SUPERDIFF:  {self.geometry_metrics['semantic']['dist_sd_mid']:10.4f}
  Spatial SUPERDIFF:   {self.geometry_metrics['spatial']['dist_sd_mid']:10.4f}

  Normalized (relative to A-B distance):
    Semantic:  {self.geometry_metrics['semantic']['normalized']:.4f}
    Spatial:   {self.geometry_metrics['spatial']['normalized']:.4f}

  Interpretation:
    - Values < 0.3: Linear interpolation in latent space
    - Values > 0.5: Non-linear composition

    {"✓ Semantic SUPERDIFF shows linear interpolation" if self.geometry_metrics['semantic']['normalized'] < 0.3 else "✗ Semantic SUPERDIFF deviates from linear interpolation"}
    {"✓ Spatial SUPERDIFF shows linear interpolation" if self.geometry_metrics['spatial']['normalized'] < 0.3 else "✗ Spatial SUPERDIFF deviates from linear interpolation"}

2. Semantic Alignment Test (Distance to Monolithic):

  Semantic SUPERDIFF to Monolithic:  {self.geometry_metrics['semantic']['dist_sd_mono']:10.4f}
  Spatial SUPERDIFF to Monolithic:   {self.geometry_metrics['spatial']['dist_sd_mono']:10.4f}

  Interpretation:
    - Lower values indicate better alignment with natural language semantics

    Semantic: {"Closer to monolithic" if self.geometry_metrics['semantic']['dist_sd_mono'] < self.geometry_metrics['semantic']['dist_sd_mid'] else "Farther from monolithic"}
    Spatial:  {"Closer to monolithic" if self.geometry_metrics['spatial']['dist_sd_mono'] < self.geometry_metrics['spatial']['dist_sd_mid'] else "Farther from monolithic"}

KAPPA DYNAMICS:
{'='*80}

  Mean κ (balance between A and B):
    Semantic:  {self.kappa_metrics['semantic_mean']:.4f} ± {self.kappa_metrics['semantic_std']:.4f}
    Spatial:   {self.kappa_metrics['spatial_mean']:.4f} ± {self.kappa_metrics['spatial_std']:.4f}

  Interpretation:
    - κ ≈ 0.5: Balanced composition
    - κ > 0.5: Biased toward A
    - κ < 0.5: Biased toward B

    Semantic: {"Balanced" if abs(self.kappa_metrics['semantic_mean'] - 0.5) < 0.1 else f"{'A-biased' if self.kappa_metrics['semantic_mean'] > 0.5 else 'B-biased'}"}
    Spatial:  {"Balanced" if abs(self.kappa_metrics['spatial_mean'] - 0.5) < 0.1 else f"{'A-biased' if self.kappa_metrics['spatial_mean'] > 0.5 else 'B-biased'}"}

KEY FINDINGS:
{'='*80}

1. DOES SPATIAL GROUNDING REDUCE HYBRIDIZATION?

   Visual Inspection Required:
   → See: semantic_vs_spatial_comparison.png

   Compare rows:
   - Row 2 (Semantic SUPERDIFF) vs Row 4 (Spatial SUPERDIFF)

   Questions to answer:
   a) Does spatial SUPERDIFF show more co-presence (two distinct objects)?
   b) Does spatial SUPERDIFF produce less morphological fusion?
   c) Do spatial prompts improve monolithic baseline (Row 3 vs Row 1)?

2. GEOMETRIC INTERPRETATION:

   Distance Comparison:
     Semantic SUPERDIFF to midpoint: {self.geometry_metrics['semantic']['dist_sd_mid']:.2f}
     Spatial SUPERDIFF to midpoint:  {self.geometry_metrics['spatial']['dist_sd_mid']:.2f}
     {"→ Spatial grounding INCREASES distance (more non-linear)" if self.geometry_metrics['spatial']['dist_sd_mid'] > self.geometry_metrics['semantic']['dist_sd_mid'] else "→ Spatial grounding DECREASES distance (more linear)"}

   If spatial shows LOWER distance to midpoint:
     → Spatial constraints don't prevent linear interpolation
     → Issue may be fundamental to probability product

   If spatial shows co-presence despite linear interpolation:
     → Linear interpolation is OK when spatially constrained
     → Issue is lack of spatial grounding, not geometry

3. BALANCE DYNAMICS:

   Kappa Difference:
     |Δκ| = |{self.kappa_metrics['semantic_mean']:.4f} - {self.kappa_metrics['spatial_mean']:.4f}| = {abs(self.kappa_metrics['semantic_mean'] - self.kappa_metrics['spatial_mean']):.4f}

   {"→ Spatial grounding significantly changes balance" if abs(self.kappa_metrics['semantic_mean'] - self.kappa_metrics['spatial_mean']) > 0.1 else "→ Spatial grounding does NOT significantly change balance"}

THEORETICAL IMPLICATIONS:
{'='*80}

Scenario A: Spatial grounding ELIMINATES hybridization
  Evidence: Row 4 shows co-presence, Row 2 shows hybrids
  Conclusion:
    - Hybridization is due to LACK OF SPATIAL INDUCTIVE BIAS
    - The UNet/VAE architecture CAN support composition
    - Probability product is fine; just needs spatial constraints
  Recommendation:
    → Always use spatially grounded prompts with SUPERDIFF
    → Or develop spatial conditioning mechanisms

Scenario B: Spatial grounding REDUCES but doesn't eliminate hybridization
  Evidence: Row 4 shows improvement but still some fusion
  Conclusion:
    - Spatial grounding helps but is insufficient
    - Partial geometric/mathematical limitation
    - May need both spatial grounding AND geometric fixes
  Recommendation:
    → Combine spatial prompts with manifold-aware composition
    → Investigate geodesic interpolation

Scenario C: Spatial grounding has NO EFFECT
  Evidence: Row 4 looks similar to Row 2 (both show hybrids)
  Conclusion:
    - Issue is FUNDAMENTAL to probability product or geometry
    - Spatial information is ignored or lost in composition
    - Architectural limitation of SUPERDIFF operator
  Recommendation:
    → Rethink composition operator entirely
    → Consider alternative formulations (mixture, attention-based)

NEXT STEPS:
{'='*80}

1. VISUAL INSPECTION:
   - Carefully examine semantic_vs_spatial_comparison.png
   - Count instances of co-presence vs. hybridization
   - Rate quality of spatial adherence (left/right positioning)

2. QUANTITATIVE ANALYSIS (if available):
   - Run object detection (YOLO, Faster R-CNN)
   - Count detected objects per image
   - Measure spatial separation between objects
   - Compute co-presence success rate

3. PARAMETER VARIATIONS:
   - Test different spatial descriptions:
     * "foreground/background"
     * "top/bottom"
     * "near/far"
   - Vary lift parameter
   - Test with more explicit spatial language

4. ALTERNATIVE COMPOSITIONS:
   - Test other prompt types (attributes, styles)
   - Multi-object composition (A ∧ B ∧ C)
   - Hierarchical composition (objects + attributes)

FILES GENERATED:
{'='*80}

Main comparison:
  - semantic_vs_spatial_comparison.png    : Side-by-side visual comparison
  - geometric_comparison.png              : Distance metrics
  - kappa_comparison.png                  : Balance dynamics

Individual experiments:
  - semantic/                             : Full semantic SUPERDIFF analysis
  - spatial/                              : Full spatial SUPERDIFF analysis

{'='*80}
END OF COMPARATIVE REPORT
{'='*80}

CRITICAL QUESTION FOR YOUR RESEARCH:

Based on visual inspection of semantic_vs_spatial_comparison.png,
does spatial grounding enable SUPERDIFF to achieve co-presence?

If YES → The issue is lack of spatial inductive bias (solvable)
If NO  → The issue is geometric/mathematical (fundamental)

This single finding will guide all subsequent theoretical interpretation.
"""

        with open(self.output_dir / 'comparative_report.txt', 'w') as f:
            f.write(report)

        print(report)
        print(f"\n    Saved: comparative_report.txt")


def run_spatial_grounding_experiments(config: SpatialGroundingConfig = None):
    """
    Main entry point for spatial grounding experiments

    Example:
        >>> config = SpatialGroundingConfig(
        ...     semantic_a="a cat",
        ...     semantic_b="a dog",
        ...     spatial_a="a cat on the left",
        ...     spatial_b="a dog on the right",
        ...     num_runs=15
        ... )
        >>> run_spatial_grounding_experiments(config)
    """
    if config is None:
        config = SpatialGroundingConfig()

    suite = SpatialGroundingExperimentSuite(config)
    suite.run_all_experiments()

    print("\n" + "="*80)
    print("SPATIAL GROUNDING EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {suite.output_dir.absolute()}")
    print("\nKey findings in: comparative_report.txt")
    print("Visual comparison: semantic_vs_spatial_comparison.png")


if __name__ == "__main__":
    # Example: Cat and dog with left/right positioning
    config = SpatialGroundingConfig(
        semantic_a="a cat",
        semantic_b="a dog",
        semantic_composed="a cat and a dog",

        spatial_a="a cat on the left side",
        spatial_b="a dog on the right side",
        spatial_composed="a cat on the left side and a dog on the right side",

        num_runs=15,
        batch_size=4,
        num_inference_steps=500,
        output_dir="experiments/spatial_grounding_cat_dog"
    )

    run_spatial_grounding_experiments(config)
