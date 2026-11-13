#!/usr/bin/env python3
"""
Complete SUPERDIFF Analysis with Enhanced Visualizations

Runs:
1. Spatial grounding experiments (semantic vs. spatial)
2. Enhanced visualizations (unified 2D/3D, trajectories, temporal)
3. Generates comprehensive report

Usage:
    python notebooks/run_full_analysis.py
    python notebooks/run_full_analysis.py --quick
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.spatial_grounding_experiments import (
    SpatialGroundingConfig,
    SpatialGroundingExperimentSuite
)
from notebooks.enhanced_visualizations import generate_enhanced_visualizations
from notebooks.manifold_geometry_analysis import (
    analyze_composition_geometry,
    analyze_trajectory_curvature
)
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Complete SUPERDIFF analysis with enhanced visualizations'
    )

    # Prompts
    parser.add_argument('--object-a', type=str, default="a cat")
    parser.add_argument('--object-b', type=str, default="a dog")
    parser.add_argument('--spatial-desc-a', type=str, default="on the left side")
    parser.add_argument('--spatial-desc-b', type=str, default="on the right side")

    # Parameters
    parser.add_argument('--num-runs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--lift', type=float, default=0.0)

    # Options
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-enhanced-viz', action='store_true',
                       help='Skip enhanced visualizations (faster)')
    parser.add_argument('--unified', action='store_true',
                       help='Generate unified 6-condition comparison '
                            '(all semantic + spatial conditions in one view)')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.quick:
        args.num_runs = 5
        args.steps = 100
        print("\n[QUICK MODE] Reduced parameters for faster testing\n")

    # Construct prompts
    semantic_a = args.object_a
    semantic_b = args.object_b
    semantic_composed = f"{args.object_a} and {args.object_b}"

    spatial_a = f"{args.object_a} {args.spatial_desc_a}"
    spatial_b = f"{args.object_b} {args.spatial_desc_b}"
    spatial_composed = f"{args.object_a} {args.spatial_desc_a} and {args.object_b} {args.spatial_desc_b}"

    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/full_analysis_{timestamp}"

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
    print("COMPLETE SUPERDIFF ANALYSIS")
    print("="*80)
    print("\nThis runs:")
    print("  1. Spatial grounding experiments (semantic vs. spatial)")
    print("  2. Enhanced visualizations (unified 2D/3D, trajectories, temporal)")
    print("  3. Comprehensive analysis report")
    print("\n" + "="*80)
    print("\nConfiguration:")
    print(f"\n  SEMANTIC:")
    print(f"    {config.semantic_a} and {config.semantic_b}")
    print(f"  SPATIAL:")
    print(f"    {config.spatial_a} and {config.spatial_b}")
    print(f"\n  Runs: {config.num_runs}")
    print(f"  Steps: {config.num_inference_steps}")
    print(f"  Output: {config.output_dir}")
    print("="*80 + "\n")

    # Step 1: Run spatial grounding experiments
    print("\n" + "="*80)
    print("STEP 1: SPATIAL GROUNDING EXPERIMENTS")
    print("="*80 + "\n")

    suite = SpatialGroundingExperimentSuite(config)
    suite.run_all_experiments()

    # Step 2: Generate enhanced visualizations
    if not args.skip_enhanced_viz:
        print("\n" + "="*80)
        print("STEP 2: ENHANCED VISUALIZATIONS")
        print("="*80)

        # Enhanced viz for semantic prompts
        print("\n--- Semantic Prompts ---")
        generate_enhanced_visualizations(
            suite.semantic_suite.results,
            str(Path(config.output_dir) / "semantic" / "enhanced"),
            suite.semantic_suite.config
        )

        # Enhanced viz for spatial prompts
        print("\n--- Spatial Prompts ---")
        generate_enhanced_visualizations(
            suite.spatial_suite.results,
            str(Path(config.output_dir) / "spatial" / "enhanced"),
            suite.spatial_suite.config
        )

    # Step 3: Advanced manifold geometry analysis
    if not args.skip_enhanced_viz:  # Use same flag as enhanced viz
        print("\n" + "="*80)
        print("STEP 3: ADVANCED MANIFOLD GEOMETRY ANALYSIS")
        print("="*80)

        # Collect latents for semantic prompts
        print("\n--- Semantic Prompts ---")
        semantic_mono = torch.cat([l.cpu().flatten(1) for l in
                                   suite.semantic_suite.results['monolithic']['latents']], dim=0)
        semantic_a = torch.cat([l.cpu().flatten(1) for l in
                               suite.semantic_suite.results['prompt_a']['latents']], dim=0)
        semantic_b = torch.cat([l.cpu().flatten(1) for l in
                               suite.semantic_suite.results['prompt_b']['latents']], dim=0)
        semantic_sd = torch.cat([l.cpu().flatten(1) for l in
                                suite.semantic_suite.results['superdiff']['latents']], dim=0)

        analyze_composition_geometry(
            semantic_mono, semantic_a, semantic_b, semantic_sd,
            output_dir=str(Path(config.output_dir) / "semantic" / "manifold")
        )

        # Trajectory curvature for semantic
        semantic_trajectories = {
            'monolithic': suite.semantic_suite.results['monolithic']['trajectories'],
            'prompt_a': suite.semantic_suite.results['prompt_a']['trajectories'],
            'prompt_b': suite.semantic_suite.results['prompt_b']['trajectories'],
            'superdiff': suite.semantic_suite.results['superdiff']['trajectories']
        }
        analyze_trajectory_curvature(
            semantic_trajectories,
            output_dir=str(Path(config.output_dir) / "semantic" / "manifold")
        )

        # Collect latents for spatial prompts
        print("\n--- Spatial Prompts ---")
        spatial_mono = torch.cat([l.cpu().flatten(1) for l in
                                 suite.spatial_suite.results['monolithic']['latents']], dim=0)
        spatial_a = torch.cat([l.cpu().flatten(1) for l in
                              suite.spatial_suite.results['prompt_a']['latents']], dim=0)
        spatial_b = torch.cat([l.cpu().flatten(1) for l in
                              suite.spatial_suite.results['prompt_b']['latents']], dim=0)
        spatial_sd = torch.cat([l.cpu().flatten(1) for l in
                               suite.spatial_suite.results['superdiff']['latents']], dim=0)

        analyze_composition_geometry(
            spatial_mono, spatial_a, spatial_b, spatial_sd,
            output_dir=str(Path(config.output_dir) / "spatial" / "manifold")
        )

        # Trajectory curvature for spatial
        spatial_trajectories = {
            'monolithic': suite.spatial_suite.results['monolithic']['trajectories'],
            'prompt_a': suite.spatial_suite.results['prompt_a']['trajectories'],
            'prompt_b': suite.spatial_suite.results['prompt_b']['trajectories'],
            'superdiff': suite.spatial_suite.results['superdiff']['trajectories']
        }
        analyze_trajectory_curvature(
            spatial_trajectories,
            output_dir=str(Path(config.output_dir) / "spatial" / "manifold")
        )

    # Step 4: Unified 6-condition comparison
    if args.unified:
        print("\n" + "="*80)
        print("STEP 4: UNIFIED 6-CONDITION COMPARISON")
        print("="*80)

        from notebooks.unified_comparison import (
            build_unified_conditions,
            generate_unified_comparison
        )

        conditions = build_unified_conditions(
            suite.semantic_suite,
            suite.spatial_suite,
            config
        )

        unified_output_dir = str(Path(config.output_dir) / "unified")

        generate_unified_comparison(
            conditions=conditions,
            vae=suite.semantic_suite.vae,
            output_dir=unified_output_dir,
            num_runs=config.num_runs
        )

    # Step 5: Final summary
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED!")
    print("="*80)

    output_path = Path(config.output_dir)

    print(f"\nResults saved to: {output_path.absolute()}")
    print("\n" + "="*80)
    print("KEY FILES TO REVIEW (IN ORDER)")
    print("="*80)

    print("\n1. CRITICAL VISUAL INSPECTION (Most Important!):")
    print(f"   {output_path / 'semantic_vs_spatial_comparison.png'}")
    print("   → Compare Row 2 (semantic SUPERDIFF) vs Row 4 (spatial SUPERDIFF)")
    print("   → Does spatial grounding produce co-presence or still hybrid?")
    print("   → This SINGLE IMAGE answers Hypothesis 0 (spatial inductive bias)")

    print("\n2. COMPREHENSIVE INTERPRETATION:")
    print(f"   {output_path / 'comparative_report.txt'}")
    print("   → Full analysis with automatic interpretation")
    print("   → Includes all distance metrics and kappa dynamics")

    print("\n3. UNIFIED LATENT SPACE (Linear Interpolation Test):")
    print(f"   {output_path / 'semantic/enhanced/unified_latent_space_2d.png'}")
    print(f"   {output_path / 'semantic/enhanced/unified_latent_space_3d_interactive.html'}")
    print("   → All conditions in single plot")
    print("   → Check: Is SUPERDIFF centroid on A-B line near midpoint?")
    print("   → Tests Hypothesis 1 (intrinsic hybridization)")

    print("\n4. TRAJECTORY DYNAMICS (Manifold Adherence Test):")
    print(f"   {output_path / 'semantic/enhanced/trajectory_evolution_3d_run0_sample0.png'}")
    print(f"   {output_path / 'semantic/enhanced/trajectory_evolution_3d_interactive_run0_sample0.html'}")
    print("   → Path evolution through latent space")
    print("   → Check: Smooth paths (on-manifold) or sharp turns (off-manifold)?")
    print("   → Tests Hypothesis 2 (geometric artifacts)")

    print("\n5. TEMPORAL ANALYSIS (Phase Transitions):")
    print(f"   {output_path / 'semantic/enhanced/temporal_phase_diagram.png'}")
    print("   → When does divergence occur? (early/mid/late)")
    print("   → Critical timesteps for composition mechanism")

    print("\n6. ADVANCED MANIFOLD GEOMETRY (Quantitative Validation):")
    print(f"   {output_path / 'semantic/manifold/manifold_geometry_analysis.png'}")
    print(f"   {output_path / 'semantic/manifold/manifold_geometry_results.txt'}")
    print("   → Intrinsic dimensionality (MLE, correlation dimension)")
    print("   → Geodesic vs. Euclidean distances")
    print("   → Local tangent space alignment")
    print("   → Provides quantitative evidence for on/off-manifold behavior")

    print("\n7. TRAJECTORY CURVATURE (Off-Manifold Diagnostics):")
    print(f"   {output_path / 'semantic/manifold/trajectory_curvature_analysis.png'}")
    print("   → Menger curvature along trajectories")
    print("   → High curvature = potential off-manifold shortcuts")
    print("   → Cumulative bending analysis")

    print("\n8. SPATIAL PROMPTS (All above analyses for spatial condition):")
    print(f"   {output_path / 'spatial/enhanced/'}")
    print(f"   {output_path / 'spatial/manifold/'}")
    print("   → Compare spatial results to semantic results")
    print("   → Does spatial grounding change geometric properties?")

    if args.unified:
        print("\n9. UNIFIED 6-CONDITION COMPARISON:")
        print(f"   {output_path / 'unified/sample_images_comparison.png'}")
        print("   → 6 rows (one per condition) x N runs of decoded images")
        print(f"   {output_path / 'unified/pca_tsne_projections.png'}")
        print("   → PCA and t-SNE with all 6 conditions + centroids")
        print(f"   {output_path / 'unified/unified_latent_space_3d_interactive.html'}")
        print("   → Interactive 3D PCA with all 6 conditions")
        print(f"   {output_path / 'unified/trajectory_evolution_3d_interactive_averaged.html'}")
        print("   → Averaged trajectories for all 6 conditions in 3D")

    print("\n" + "="*80)
    print("INTERPRETATION DECISION TREE")
    print("="*80)

    print("""
1. Visual Inspection (semantic_vs_spatial_comparison.png):

   IF Row 4 (spatial) shows co-presence:
     → Issue is LACK OF SPATIAL INDUCTIVE BIAS
     → Solution: Use spatial prompts
     → Hybridization is NOT fundamental

   ELSE IF Row 4 still shows hybrids:
     → Continue to geometric analysis...

2. Geometric Analysis (unified_latent_space_2d.png):

   IF SUPERDIFF centroid near (A+B)/2 midpoint:
     → Linear interpolation
     → Check manifold adherence...

     IF Geodesic ≈ Euclidean (trajectory smooth):
       → INTRINSIC to probability product
       → Need different composition operator

     ELSE IF Geodesic >> Euclidean (sharp turns):
       → GEOMETRIC ARTIFACT (off-manifold)
       → Fixable with geodesic interpolation

   ELSE IF SUPERDIFF centroid near monolithic:
     → Composition discovers natural semantics
     → Mathematical AND ≈ linguistic "and"

3. Temporal Analysis (temporal_phase_diagram.png):

   → Identify WHEN composition mechanism engages
   → Analyze velocity fields at critical timesteps
   → Understanding temporal dynamics informs mechanism
""")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    print("""
1. Open semantic_vs_spatial_comparison.png
   → This SINGLE IMAGE answers the spatial grounding hypothesis

2. Read comparative_report.txt
   → Automated interpretation of all metrics

3. Explore interactive 3D plots (*.html files)
   → Rotate to see geometric relationships
   → Verify 2D projection interpretations

4. Based on findings, run parameter sweeps:
   - Vary lift parameter
   - Test different spatial descriptions
   - Try alternative prompt types
""")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
