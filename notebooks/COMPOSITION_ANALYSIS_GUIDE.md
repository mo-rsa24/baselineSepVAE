# SUPERDIFF Composition Analysis: Understanding Hybridization vs. Co-Presence

## Overview

This experimental framework investigates why SUPERDIFF's logical AND operation produces **hybridization** (e.g., a cat-dog chimera) rather than **co-presence** (e.g., an image containing both a cat and a dog). The analysis combines controlled experiments, latent space diagnostics, and manifold geometry investigations to understand the mathematical and semantic properties of SUPERDIFF composition.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Experimental Design](#experimental-design)
3. [Running the Experiments](#running-the-experiments)
4. [Interpreting Results](#interpreting-results)
5. [Key Diagnostics](#key-diagnostics)
6. [Theoretical Implications](#theoretical-implications)

---

## Theoretical Background

### SUPERDIFF AND Operation

The SUPERDIFF method, as described in *"The Superposition of Diffusion Models Using the Itô Density Estimator,"* defines a logical AND between two prompts A and B by constructing a composite vector field:

```
v_composite = v_uncond + γ[(v_B - v_uncond) + κ(v_A - v_B)]
```

Where:
- `v_A`, `v_B`: Conditional velocity fields for prompts A and B
- `v_uncond`: Unconditional velocity field
- `γ`: Guidance scale
- `κ`: Dynamic weight optimizing log-density ratios

### The Hybridization Hypothesis

We hypothesize that hybridization occurs due to:

1. **Energy-Based Intersection**: SUPERDIFF maximizes `p(x|A) · p(x|B)`, seeking states that are probable under *both* models. This favors feature blending over spatial composition.

2. **Off-Manifold Trajectories**: Linear interpolation in latent space (`κ·v_A + (1-κ)·v_B`) may create trajectories that cut through off-manifold regions, leading to artifacts.

3. **Non-Euclidean Geometry**: The latent manifold has curved geometry, but SUPERDIFF performs Euclidean operations (weighted sums), potentially violating manifold constraints.

4. **Semantic Mismatch**: Mathematical AND (probability product) ≠ linguistic AND (set union/co-presence).

### Research Questions

1. **Geometric**: Does SUPERDIFF composition stay on-manifold or take off-manifold shortcuts?
2. **Trajectory**: How do velocity fields evolve during composition? Do they interpolate linearly?
3. **Statistical**: Where do SUPERDIFF samples lie relative to individual prompt distributions?
4. **Semantic**: Is there a systematic relationship between mathematical composition and human expectations?

---

## Experimental Design

### Comparison Groups

We compare four conditions, all starting from the **same initial noise** for fair comparison:

1. **Monolithic Prompt**: `"A photograph of a cat and a dog"`
   - Baseline: How the model naturally interprets "A and B"

2. **Individual Prompt A**: `"A photograph of a cat"`
   - Reference distribution for concept A alone

3. **Individual Prompt B**: `"A photograph of a dog"`
   - Reference distribution for concept B alone

4. **SUPERDIFF Composition**: `A ∧ B`
   - Mathematical composition via SUPERDIFF AND

### Experimental Parameters

- **Multiple Stochastic Runs**: 10-20 runs with different random seeds to capture distribution statistics
- **Batch Sampling**: 4 samples per run for statistical robustness
- **Trajectory Tracking**: Record full latent trajectories and velocity fields at each diffusion step
- **Log-Likelihood Tracking**: Monitor p(x|A) and p(x|B) throughout sampling

### Controlled Variables

- Same initial noise across conditions (per run)
- Same scheduler, guidance scale, number of steps
- Same model (SD 1.5)

---

## Running the Experiments

### Quick Start

```bash
# Default experiment (cat and dog)
python notebooks/run_composition_analysis.py

# Quick test (fewer runs and steps)
python notebooks/run_composition_analysis.py --quick

# Custom prompts
python notebooks/run_composition_analysis.py \
    --prompt-a "a red sports car" \
    --prompt-b "a blue truck" \
    --num-runs 20 \
    --steps 500
```

### Advanced Options

```bash
python notebooks/run_composition_analysis.py \
    --prompt-a "a cat" \
    --prompt-b "a dog" \
    --prompt-composed "a cat and a dog together" \
    --num-runs 20 \
    --batch-size 4 \
    --steps 500 \
    --guidance-scale 7.5 \
    --lift 0.0 \
    --output-dir experiments/my_experiment
```

### Using as a Library

```python
from notebooks.composition_experiments import ExperimentConfig, run_composition_experiments

config = ExperimentConfig(
    prompt_a="a photograph of a cat",
    prompt_b="a photograph of a dog",
    prompt_composed="a photograph of a cat and a dog",
    num_runs=20,
    num_inference_steps=500,
    output_dir="experiments/cat_dog_analysis"
)

run_composition_experiments(config)
```

---

## Interpreting Results

### Generated Visualizations

The analysis produces several diagnostic plots:

#### 1. **sample_images_comparison.png**
   - **What it shows**: Visual grid comparing outputs from all conditions
   - **What to look for**:
     - Does SUPERDIFF produce hybrids or co-presence?
     - How similar is SUPERDIFF to the monolithic prompt?
     - Are individual prompts clearly distinct?

#### 2. **trajectory_geometry.png**
   - **What it shows**:
     - Latent norm evolution (denoising progression)
     - Distances between trajectories over time
     - Velocity magnitudes
     - Trajectory smoothness (curvature)
   - **What to look for**:
     - When do trajectories diverge?
     - Is SUPERDIFF closer to monolithic or to (A+B)/2?
     - Are velocity magnitudes comparable?
     - High curvature suggests sharp turns (possibly off-manifold)

#### 3. **centroid_statistics.png**
   - **What it shows**:
     - L2 distances between centroids
     - Variance within each condition
     - PCA variance explained
     - Distribution overlap on PC1
   - **What to look for**:
     - **Key metric**: Distance from SUPERDIFF centroid to (A+B)/2
       - Small distance → linear interpolation
       - Large distance → non-linear composition
     - Variance comparison:
       - Higher variance in SUPERDIFF → greater uncertainty
       - Lower variance → mode collapse

#### 4. **kappa_dynamics.png**
   - **What it shows**: Evolution of κ (the A-B balance parameter)
   - **What to look for**:
     - κ ≈ 0.5: Equal weighting
     - κ > 0.5: Biased toward A
     - κ < 0.5: Biased toward B
     - Temporal trends: Does κ stabilize or oscillate?

#### 5. **pca_tsne_projections.png**
   - **What it shows**: 2D projections of latent space structure
   - **What to look for**:
     - Do SUPERDIFF samples form a cluster between A and B?
     - Or do they overlap with one condition?
     - Are distributions clearly separated or overlapping?

#### 6. **manifold_distances.png**
   - **What it shows**:
     - k-NN distances (manifold density)
     - Distance to centroids
     - Distance to A-B interpolation line
     - 2D projection with interpolation line
   - **What to look for**:
     - **Critical**: Distance to interpolation line
       - Small → linear interpolation
       - Large → off-manifold or non-linear

#### 7. **velocity_field_alignment.png**
   - **What it shows**:
     - Cosine similarity between velocity fields
     - Magnitude ratios
     - Angular divergence
     - Linear decomposition test: v_SD ≈ α·v_A + (1-α)·v_B
   - **What to look for**:
     - Does SUPERDIFF velocity decompose as α·v_A + (1-α)·v_B?
     - Is α ≈ 0.5 (equal weighting)?
     - Alignment with monolithic prompt?

#### 8. **manifold_geometry_analysis.png** (if enabled)
   - **What it shows**:
     - Intrinsic dimensionality estimates
     - Local tangent space alignment
     - Geodesic vs Euclidean distances
   - **What to look for**:
     - **Key metric**: Geodesic/Euclidean ratio
       - Ratio ≈ 1: On-manifold (Euclidean ≈ geodesic)
       - Ratio > 1: Off-manifold shortcuts
     - Dimensionality changes suggest manifold transitions

#### 9. **trajectory_curvature_analysis.png** (if enabled)
   - **What it shows**: Menger curvature along trajectories
   - **What to look for**:
     - High curvature = sharp bending (potential off-manifold)
     - Compare SUPERDIFF vs monolithic curvature
     - When does curvature peak? (early/mid/late diffusion)

---

## Key Diagnostics

### Primary Evidence for Hybridization Mechanism

| Evidence Type | Metric | Interpretation |
|--------------|--------|----------------|
| **Linear Interpolation** | Distance(SUPERDIFF, (A+B)/2) | Small → linear; Large → non-linear |
| **On-Manifold** | Geodesic/Euclidean ratio | ~1 → on-manifold; >1 → off-manifold |
| **Semantic Alignment** | Distance(SUPERDIFF, Monolithic) | Small → semantic match; Large → mismatch |
| **Balance** | Mean κ | ~0.5 → balanced; else biased |
| **Trajectory Shortcuts** | Curvature | High → sharp turns (off-manifold) |
| **Tangent Alignment** | PCA alignment | High → same local structure |

### Decision Tree for Interpretation

```
Is Distance(SUPERDIFF, (A+B)/2) small?
├─ YES → Linear interpolation in latent space
│   └─ Is Geodesic/Euclidean ≈ 1?
│       ├─ YES → On-manifold interpolation
│       │   └─ Hybridization is INTRINSIC to composition formula
│       └─ NO → Off-manifold shortcuts
│           └─ Hybridization due to GEOMETRIC artifacts
└─ NO → Non-linear composition
    └─ Is Distance(SUPERDIFF, Monolithic) small?
        ├─ YES → SUPERDIFF ≈ monolithic prompt
        │   └─ Composition aligns with natural language
        └─ NO → SUPERDIFF is distinct
            └─ Composition creates novel semantic space
```

---

## Theoretical Implications

### Scenario 1: Linear Interpolation + On-Manifold

**Evidence**:
- SUPERDIFF centroid near (A+B)/2
- Geodesic ≈ Euclidean
- κ ≈ 0.5

**Conclusion**: Hybridization is **intrinsic to the SUPERDIFF formula**. The probability product `p(x|A)·p(x|B)` naturally favors feature blending because:
- States with *both* cat and dog features have high joint probability
- States with *only* cat or *only* dog have lower joint probability
- This is mathematically correct but semantically misaligned with linguistic "and"

**Implications**:
- Not a bug, but a **semantic mismatch** between math and language
- To achieve co-presence, need different composition operator (e.g., spatial conditioning)

---

### Scenario 2: Off-Manifold Shortcuts

**Evidence**:
- SUPERDIFF centroid near (A+B)/2
- Geodesic >> Euclidean (ratio > 1.2)
- High trajectory curvature

**Conclusion**: Hybridization due to **geometric artifacts**. Linear combination of velocity fields creates trajectories that cut through off-manifold regions, leading to:
- Unrealistic feature combinations
- Poor image quality in hybrid regions
- Artifacts due to leaving training distribution

**Implications**:
- Could be improved with manifold-aware composition
- Geodesic interpolation or tangent space projections may help
- Suggests architectural improvements possible

---

### Scenario 3: Non-Linear, Semantic Match

**Evidence**:
- SUPERDIFF centroid far from (A+B)/2
- SUPERDIFF centroid near Monolithic
- Low variance

**Conclusion**: SUPERDIFF composition **converges to natural language interpretation**. The mathematical AND, through its dynamics, discovers a semantic representation aligned with "cat and dog."

**Implications**:
- Composition operator is semantically appropriate
- Hybridization may not always occur (depends on concepts)
- Learned representations may encode compositionality

---

### Scenario 4: Non-Linear, Distinct Space

**Evidence**:
- SUPERDIFF centroid far from all references
- High variance
- Distinct PCA cluster

**Conclusion**: SUPERDIFF creates a **novel semantic space** not captured by monolithic or individual prompts. This could represent:
- True compositional reasoning beyond training data
- Or a failure mode (compositional OOD)

**Implications**:
- Requires further investigation with semantic metrics (CLIP similarity, object detection)
- May reveal limitations or capabilities of composition

---

## Advanced Analyses

### Custom Prompt Experiments

Test different semantic relationships:

```bash
# Spatial composition
python run_composition_analysis.py \
    --prompt-a "a cat on the left" \
    --prompt-b "a dog on the right"

# Feature composition
python run_composition_analysis.py \
    --prompt-a "a red object" \
    --prompt-b "a round object"

# Conflicting attributes
python run_composition_analysis.py \
    --prompt-a "a small elephant" \
    --prompt-b "a large mouse"
```

### Parameter Sweeps

Investigate effect of lift parameter:

```python
from notebooks.composition_experiments import ExperimentConfig, CompositionExperimentSuite

for lift in [-20, -10, 0, 10, 20]:
    config = ExperimentConfig(
        prompt_a="a cat",
        prompt_b="a dog",
        lift=lift,
        num_runs=10,
        output_dir=f"experiments/lift_sweep/lift_{lift}"
    )
    suite = CompositionExperimentSuite(config)
    suite.run_all_experiments()
```

### Temporal Analysis

Identify when hybridization emerges:

```python
# Analyze trajectory at specific timesteps
early_latents = trajectories.trajectories[50]   # Early diffusion
mid_latents = trajectories.trajectories[250]    # Mid diffusion
late_latents = trajectories.trajectories[450]   # Late diffusion

# Compute distances at each stage
# (code similar to centroid_statistics analysis)
```

---

## Recommendations for Publication

### Key Figures to Include

1. **Figure 1**: Sample images comparison (4×N grid)
2. **Figure 2**: Centroid distance analysis (bar plot + 2D projection)
3. **Figure 3**: Kappa dynamics (mean ± std over time)
4. **Figure 4**: Geodesic vs Euclidean distances (scatter + ratio histogram)
5. **Figure 5**: Velocity field decomposition (α over time)

### Key Metrics to Report

1. Distance(SUPERDIFF, (A+B)/2) / Distance(A, B)
   - Normalized measure of linear interpolation

2. Geodesic/Euclidean ratio (mean ± std)
   - Evidence for on-manifold vs off-manifold

3. Mean κ (± std) and temporal trend
   - Balance between concepts

4. CLIP similarity scores (if available)
   - Semantic alignment with prompts

### Statistical Tests

```python
from scipy.stats import ttest_ind, ks_2samp

# Test if SUPERDIFF distances differ from monolithic
dist_sd_midpoint = [...]  # Your distances
dist_mono_midpoint = [...]

t_stat, p_value = ttest_ind(dist_sd_midpoint, dist_mono_midpoint)
print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
```

---

## References

### Theoretical Foundation

- *"The Superposition of Diffusion Models Using the Itô Density Estimator"*
- Diffusion probabilistic models: Ho et al., 2020
- Manifold hypothesis in deep learning: Bengio et al., 2013

### Manifold Analysis Methods

- Intrinsic dimensionality estimation: Levina & Bickel, 2005
- Geodesic distance approximation: Tenenbaum et al., 2000 (Isomap)
- Menger curvature: Geometric measure theory

### Related Work

- Compositional generation: Liu et al., 2022 (Composable Diffusion)
- Concept arithmetic: Rombach et al., 2022 (Stable Diffusion)
- Latent space geometry: Karras et al., 2020 (StyleGAN)

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size or number of runs
python run_composition_analysis.py --batch-size 2 --num-runs 5

# Use quick mode
python run_composition_analysis.py --quick
```

### Slow Execution

```bash
# Skip manifold analysis (saves ~50% time)
python run_composition_analysis.py --skip-manifold-analysis

# Reduce inference steps (use 100 instead of 500)
python run_composition_analysis.py --steps 100
```

### CUDA Errors

Check GPU memory:
```python
import torch
print(torch.cuda.memory_summary())
```

Clear cache between runs:
```python
torch.cuda.empty_cache()
```

---

## Contact & Contributions

This experimental framework is designed to be extensible. Potential additions:

- CLIP-based semantic similarity metrics
- Object detection to quantify co-presence
- Attention map visualization
- Alternative composition operators (OR, NOT, XOR)
- Multi-prompt composition (A ∧ B ∧ C)
- Temporal composition analysis

For questions or contributions, please open an issue or pull request.

---

## Citation

If you use this experimental framework in your research, please cite:

```bibtex
@software{superdiff_composition_analysis,
  title={SUPERDIFF Composition Analysis Framework},
  author={[Your Name]},
  year={2026},
  url={https://github.com/yourusername/superdiff-ldm}
}
```

And the original SUPERDIFF paper:
```bibtex
@article{superdiff_2024,
  title={The Superposition of Diffusion Models Using the Itô Density Estimator},
  author={[Original Authors]},
  journal={[Journal]},
  year={2024}
}
```
