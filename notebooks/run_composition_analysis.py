#!/usr/bin/env python3
"""
Simple runner script for SUPERDIFF composition analysis experiments.

Usage:
    python run_composition_analysis.py

or with custom prompts:
    python run_composition_analysis.py --prompt-a "a red car" --prompt-b "a blue truck"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.composition_experiments import ExperimentConfig, run_composition_experiments
from notebooks.manifold_geometry_analysis import (
    analyze_composition_geometry,
    analyze_trajectory_curvature
)
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Run SUPERDIFF composition analysis experiments'
    )

    # Prompt configuration
    parser.add_argument('--prompt-a', type=str, default="A photograph of a cat",
                      help='First prompt for composition')
    parser.add_argument('--prompt-b', type=str, default="A photograph of a dog",
                      help='Second prompt for composition')
    parser.add_argument('--prompt-composed', type=str, default=None,
                      help='Monolithic prompt (default: "<prompt-a> and <prompt-b>")')

    # Experiment parameters
    parser.add_argument('--num-runs', type=int, default=10,
                      help='Number of stochastic runs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size per run (default: 4)')
    parser.add_argument('--steps', type=int, default=500,
                      help='Number of inference steps (default: 500)')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                      help='Guidance scale (default: 7.5)')
    parser.add_argument('--lift', type=float, default=0.0,
                      help='SUPERDIFF lift parameter (default: 0.0)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: experiments/<timestamp>)')

    # Analysis options
    parser.add_argument('--skip-manifold-analysis', action='store_true',
                      help='Skip detailed manifold geometry analysis (faster)')
    parser.add_argument('--quick', action='store_true',
                      help='Quick run: 5 runs, 100 steps, skip manifold analysis')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_runs = 5
        args.steps = 100
        args.skip_manifold_analysis = True
        print("\n[QUICK MODE] Running with reduced parameters for faster results")

    # Set default composed prompt
    if args.prompt_composed is None:
        args.prompt_composed = f"{args.prompt_a} and {args.prompt_b}"

    # Set default output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/composition_analysis_{timestamp}"

    # Create configuration
    config = ExperimentConfig(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        prompt_composed=args.prompt_composed,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lift=args.lift,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("SUPERDIFF COMPOSITION ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Prompt A:        {config.prompt_a}")
    print(f"  Prompt B:        {config.prompt_b}")
    print(f"  Monolithic:      {config.prompt_composed}")
    print(f"  Runs:            {config.num_runs}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Steps:           {config.num_inference_steps}")
    print(f"  Guidance scale:  {config.guidance_scale}")
    print(f"  Lift:            {config.lift}")
    print(f"  Output:          {config.output_dir}")
    print(f"\n  Total samples per condition: {config.num_runs * config.batch_size}")
    print("="*80 + "\n")

    # Run main experiments
    print("Starting main experiments...")
    from notebooks.composition_experiments import CompositionExperimentSuite
    suite = CompositionExperimentSuite(config)
    suite.run_all_experiments()

    # Additional manifold analysis if requested
    if not args.skip_manifold_analysis:
        print("\n" + "="*80)
        print("Running additional manifold geometry analysis...")
        print("="*80 + "\n")

        # Collect latents
        latents_mono = torch.cat([l.cpu().flatten(1) for l in suite.results['monolithic']['latents']], dim=0)
        latents_a = torch.cat([l.cpu().flatten(1) for l in suite.results['prompt_a']['latents']], dim=0)
        latents_b = torch.cat([l.cpu().flatten(1) for l in suite.results['prompt_b']['latents']], dim=0)
        latents_sd = torch.cat([l.cpu().flatten(1) for l in suite.results['superdiff']['latents']], dim=0)

        analyze_composition_geometry(
            latents_mono, latents_a, latents_b, latents_sd,
            output_dir=config.output_dir
        )

        # Trajectory curvature analysis
        trajectories_dict = {
            'monolithic': suite.results['monolithic']['trajectories'],
            'prompt_a': suite.results['prompt_a']['trajectories'],
            'prompt_b': suite.results['prompt_b']['trajectories'],
            'superdiff': suite.results['superdiff']['trajectories']
        }

        analyze_trajectory_curvature(trajectories_dict, output_dir=config.output_dir)

    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {Path(config.output_dir).absolute()}")
    print("\nGenerated files:")
    print("  - sample_images_comparison.png      : Visual comparison of outputs")
    print("  - trajectory_geometry.png           : Trajectory analysis")
    print("  - centroid_statistics.png           : Latent space statistics")
    print("  - kappa_dynamics.png                : SUPERDIFF weight evolution")
    print("  - pca_tsne_projections.png          : Dimensionality reduction views")
    print("  - manifold_distances.png            : Manifold distance analysis")
    print("  - velocity_field_alignment.png      : Vector field comparisons")
    print("  - summary_report.txt                : Comprehensive text summary")

    if not args.skip_manifold_analysis:
        print("  - manifold_geometry_analysis.png    : Advanced manifold geometry")
        print("  - trajectory_curvature_analysis.png : Curvature analysis")
        print("  - manifold_geometry_results.txt     : Numerical results")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
