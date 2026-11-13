#!/usr/bin/env python3
"""
Spatial Grounding Experiments Runner

Tests the critical hypothesis:
Does spatial grounding eliminate SUPERDIFF hybridization?

Usage:
    python run_spatial_grounding.py

or with custom prompts:
    python run_spatial_grounding.py \
        --object-a "cat" \
        --object-b "dog" \
        --spatial-desc-a "on the left side" \
        --spatial-desc-b "on the right side"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.spatial_grounding_experiments import (
    SpatialGroundingConfig,
    run_spatial_grounding_experiments
)


def main():
    parser = argparse.ArgumentParser(
        description='Test whether spatial grounding prevents SUPERDIFF hybridization'
    )

    # Object descriptions
    parser.add_argument('--object-a', type=str, default="a cat",
                       help='First object (semantic)')
    parser.add_argument('--object-b', type=str, default="a dog",
                       help='Second object (semantic)')

    # Spatial descriptors
    parser.add_argument('--spatial-desc-a', type=str, default="on the left side",
                       help='Spatial location for object A')
    parser.add_argument('--spatial-desc-b', type=str, default="on the right side",
                       help='Spatial location for object B')

    # Sampling parameters
    parser.add_argument('--num-runs', type=int, default=15,
                       help='Number of stochastic runs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size per run')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Guidance scale')
    parser.add_argument('--lift', type=float, default=0.0,
                       help='SUPERDIFF lift parameter')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick run: fewer iterations')

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.num_runs = 5
        args.steps = 100
        print("\n[QUICK MODE] Using reduced parameters for faster testing")

    # Construct prompts
    semantic_a = args.object_a
    semantic_b = args.object_b
    semantic_composed = f"{args.object_a} and {args.object_b}"

    spatial_a = f"{args.object_a} {args.spatial_desc_a}"
    spatial_b = f"{args.object_b} {args.spatial_desc_b}"
    spatial_composed = f"{args.object_a} {args.spatial_desc_a} and {args.object_b} {args.spatial_desc_b}"

    # Default output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/spatial_grounding_{timestamp}"

    # Create config
    config = SpatialGroundingConfig(
        semantic_a=semantic_a,
        semantic_b=semantic_b,
        semantic_composed=semantic_composed,
        spatial_a=spatial_a,
        spatial_b=spatial_b,
        spatial_composed=spatial_composed,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lift=args.lift,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("SPATIAL GROUNDING EXPERIMENT")
    print("="*80)
    print("\nTesting whether spatial grounding eliminates hybridization")
    print("\nPrompts:")
    print(f"\n  SEMANTIC (no spatial constraints):")
    print(f"    A:          {config.semantic_a}")
    print(f"    B:          {config.semantic_b}")
    print(f"    Composed:   {config.semantic_composed}")
    print(f"    SUPERDIFF:  A ∧ B")
    print(f"\n  SPATIAL (explicit positioning):")
    print(f"    A:          {config.spatial_a}")
    print(f"    B:          {config.spatial_b}")
    print(f"    Composed:   {config.spatial_composed}")
    print(f"    SUPERDIFF:  A ∧ B")
    print(f"\nParameters:")
    print(f"  Runs:           {config.num_runs}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Steps:          {config.num_inference_steps}")
    print(f"  Guidance scale: {config.guidance_scale}")
    print(f"  Lift:           {config.lift}")
    print(f"  Output:         {config.output_dir}")
    print("="*80 + "\n")

    # Run experiments
    run_spatial_grounding_experiments(config)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - NEXT STEPS")
    print("="*80)
    print("\n1. VISUAL INSPECTION (CRITICAL):")
    print(f"   Open: {Path(config.output_dir).absolute()}/semantic_vs_spatial_comparison.png")
    print("\n   Compare rows:")
    print("   - Row 2 (Semantic SUPERDIFF) vs Row 4 (Spatial SUPERDIFF)")
    print("\n   Questions:")
    print("   • Does spatial grounding produce co-presence (two distinct objects)?")
    print("   • Or does it still show hybridization (morphological fusion)?")
    print("\n2. READ INTERPRETATION:")
    print(f"   Open: {Path(config.output_dir).absolute()}/comparative_report.txt")
    print("\n3. THEORETICAL CONCLUSION:")
    print("   • If spatial works → Issue is lack of spatial inductive bias")
    print("   • If spatial fails → Issue is geometric/mathematical (fundamental)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
