# Enhanced Visualization Guide

## Summary: Do Current Visualizations Clearly Differentiate Conditions?

**Short Answer**: The **basic visualizations** provide good overview but have limitations. The **enhanced visualizations** address all your concerns with:

1. ✅ **Unified latent space plots** showing ALL conditions in a single view (2D + 3D)
2. ✅ **3D projections** alongside 2D (both static and interactive)
3. ✅ **Trajectory evolution** showing path dynamics over timesteps
4. ✅ **Temporal phase analysis** identifying when divergence occurs

---

## Current Visualizations: Strengths & Weaknesses

### ✅ What Works Well

| Visualization | Strength | File |
|--------------|----------|------|
| **Sample images** | Direct visual comparison | `sample_images_comparison.png` |
| **Kappa dynamics** | Clear temporal evolution | `kappa_dynamics.png` |
| **Centroid distances** | Quantitative metrics | `centroid_statistics.png` |
| **PCA/t-SNE** | Dimensionality reduction | `pca_tsne_projections.png` |

### ⚠️ Limitations Addressed

| Issue | Problem | Solution |
|-------|---------|----------|
| **No unified view** | Conditions shown as separate clusters | → **Unified latent space plot** |
| **2D projection loss** | PCA loses ~30-40% variance | → **3D projections** |
| **Static plots** | Can't rotate/explore | → **Interactive Plotly 3D** |
| **Trajectory compression** | Only summary statistics shown | → **Full trajectory paths** |
| **Temporal dynamics** | Hard to see when divergence occurs | → **Phase diagrams** |

---

## Enhanced Visualizations

### 1. Unified Latent Space (2D + 3D)

**Problem Solved**: Shows ALL conditions in a single plot for direct comparison

**Files Generated**:
- `unified_latent_space_2d.png` - 2D PCA with all conditions
- `unified_latent_space_3d.png` - 3D PCA with all conditions
- `unified_latent_space_3d_interactive.html` - **Interactive 3D** (rotatable!)

**What You Can See**:
- Relative positioning of Monolithic, A, B, and SUPERDIFF centroids
- Distribution overlap between conditions
- Lines connecting key centroids (A-B line, SD to midpoint, SD to monolithic)
- Clear visual test of linear interpolation hypothesis

**Interpretation**:
```
If SUPERDIFF centroid is ON the A-B line near midpoint:
  → Linear interpolation in latent space
  → Hybridization likely intrinsic

If SUPERDIFF centroid is OFF the A-B line:
  → Non-linear composition
  → Different mechanism at play

If SUPERDIFF centroid near Monolithic:
  → Mathematical AND ≈ linguistic "and"
  → Composition semantically aligned
```

### 2. Trajectory Evolution (2D + 3D)

**Problem Solved**: Shows actual paths through latent space over time

**Files Generated** (per run):
- `trajectory_evolution_2d_run{i}_sample{j}.png` - 2D path with time gradient
- `trajectory_evolution_3d_run{i}_sample{j}.png` - 3D path visualization
- `trajectory_evolution_3d_interactive_run{i}_sample{j}.html` - **Interactive 3D trajectories**

**What You Can See**:
- Start points (white circles) and end points (colored stars)
- Paths colored with gradient (light → dark = early → late)
- Trajectory curvature (sharp turns indicate potential off-manifold behavior)
- When trajectories diverge from each other

**Interpretation**:
```
If SUPERDIFF trajectory follows smooth path:
  → On-manifold composition
  → Geodesic ≈ Euclidean

If SUPERDIFF trajectory has sharp bends:
  → Possible off-manifold shortcuts
  → Geodesic > Euclidean

If SUPERDIFF trajectory diverges EARLY from monolithic:
  → Composition mechanism kicks in immediately
  → Mathematical formulation dominates

If SUPERDIFF trajectory diverges LATE from monolithic:
  → Initial agreement, then specialization
  → Phase transition in composition
```

### 3. Temporal Phase Diagram

**Problem Solved**: Identifies WHEN during diffusion critical events occur

**File Generated**:
- `temporal_phase_diagram.png`

**What You Can See**:
- **Panel 1**: Trajectory divergence over time
- **Panel 2**: Rate of divergence (derivative) - shows acceleration
- **Panel 3**: Velocity magnitude evolution
- **Panel 4**: Critical timesteps (peaks = phase transitions)

**Phases Marked**:
- **Early diffusion** (blue): High noise, coarse structure
- **Mid diffusion** (green): Medium-scale features emerge
- **Late diffusion** (red): Fine details, convergence

**Interpretation**:
```
If divergence happens in EARLY phase:
  → Composition affects coarse structure
  → Fundamental difference in generation strategy

If divergence happens in MID phase:
  → Composition affects medium-scale features
  → Objects/layout level

If divergence happens in LATE phase:
  → Composition affects fine details only
  → Surface appearance, not semantic content

Peak locations in rate plot:
  → Critical timesteps where composition "kicks in"
  → Analyze velocity fields at these steps for mechanism
```

---

## Usage

### Basic Composition Experiments + Enhanced Visualizations

```bash
# Run experiments with enhanced visualizations
python notebooks/run_composition_analysis.py --num-runs 10
```

Then in Python:

```python
from notebooks.enhanced_visualizations import generate_enhanced_visualizations
from notebooks.composition_experiments import CompositionExperimentSuite, ExperimentConfig

# Load your completed experiment
config = ExperimentConfig(
    prompt_a="a cat",
    prompt_b="a dog",
    num_runs=10,
    output_dir="experiments/my_experiment"
)

suite = CompositionExperimentSuite(config)
suite.run_all_experiments()  # or load existing results

# Generate enhanced visualizations
generate_enhanced_visualizations(
    suite.results,
    suite.output_dir,
    suite.config
)
```

### Spatial Grounding + Enhanced Visualizations

```python
from notebooks.spatial_grounding_experiments import (
    SpatialGroundingExperimentSuite,
    SpatialGroundingConfig
)
from notebooks.enhanced_visualizations import generate_enhanced_visualizations

config = SpatialGroundingConfig(num_runs=15)
suite = SpatialGroundingExperimentSuite(config)
suite.run_all_experiments()

# Generate for semantic prompts
generate_enhanced_visualizations(
    suite.semantic_suite.results,
    suite.output_dir / "semantic_enhanced",
    suite.semantic_suite.config
)

# Generate for spatial prompts
generate_enhanced_visualizations(
    suite.spatial_suite.results,
    suite.output_dir / "spatial_enhanced",
    suite.spatial_suite.config
)
```

---

## Visualization Decision Tree

### Which visualization answers which question?

```
Question: Does SUPERDIFF produce hybrids or co-presence?
└─> sample_images_comparison.png (VISUAL INSPECTION)

Question: Is composition linear in latent space?
└─> unified_latent_space_2d.png
    └─> Check: Is SUPERDIFF centroid on A-B line near midpoint?

Question: Does composition stay on-manifold?
└─> trajectory_evolution_3d.png
    └─> Check: Smooth paths (on-manifold) or sharp turns (off-manifold)?
    └─> Also: manifold_geometry_analysis.png (geodesic/Euclidean ratio)

Question: When does composition mechanism engage?
└─> temporal_phase_diagram.png
    └─> Check: Early, mid, or late divergence?
    └─> Check: Peak locations in divergence rate

Question: Is composition balanced between A and B?
└─> kappa_dynamics.png
    └─> Check: Mean κ ≈ 0.5?

Question: Does spatial grounding help?
└─> semantic_vs_spatial_comparison.png (SPATIAL EXPERIMENTS)
    └─> Compare Row 2 vs Row 4
```

---

## Interpretation Framework

### Scenario A: Linear Interpolation + On-Manifold

**Evidence**:
- unified_latent_space_2d.png: SUPERDIFF centroid near midpoint
- trajectory_evolution_3d.png: Smooth paths
- manifold_geometry_analysis.png: Geodesic/Euclidean ≈ 1

**Conclusion**: Hybridization is **intrinsic** to probability product formula

**Implication**: Not fixable with geometry; need different composition operator

---

### Scenario B: Linear Interpolation + Off-Manifold

**Evidence**:
- unified_latent_space_2d.png: SUPERDIFF centroid near midpoint
- trajectory_evolution_3d.png: Sharp bends in paths
- manifold_geometry_analysis.png: Geodesic/Euclidean > 1.2

**Conclusion**: Hybridization is a **geometric artifact**

**Implication**: Fixable with geodesic interpolation or manifold-aware composition

---

### Scenario C: Non-Linear, Semantic Match

**Evidence**:
- unified_latent_space_2d.png: SUPERDIFF centroid near monolithic
- temporal_phase_diagram.png: Early divergence from A and B, convergence to monolithic

**Conclusion**: Composition **discovers** natural language semantics

**Implication**: Mathematical AND successfully aligns with linguistic "and"

---

### Scenario D: Spatial Grounding Resolves Hybridization

**Evidence** (from spatial experiments):
- semantic_vs_spatial_comparison.png: Row 4 shows co-presence, Row 2 shows hybrids
- unified_latent_space_2d.png (spatial): SUPERDIFF centroid still near midpoint
- BUT visual outcome changes

**Conclusion**: Issue is **lack of spatial inductive bias**, not geometry

**Implication**: Use spatial prompts; the composition operator itself is fine

---

## 3D vs. 2D: When to Use Which?

### Use 2D PCA when:
- You want maximum variance captured in minimal dimensions
- You need publication-quality static figures
- You're comparing centroids and distances quantitatively

### Use 3D PCA when:
- 2D explains <70% variance (check `pca.explained_variance_ratio_`)
- You need to see trajectory paths clearly
- You want to detect manifold curvature visually
- You're exploring complex geometric relationships

### Use Interactive 3D when:
- You need to rotate and explore from different angles
- Trajectories cross in 2D projection but are separated in 3D
- You're diagnosing off-manifold behavior
- You want to share with collaborators who need exploratory view

---

## Recommendations

### For Your Research

1. **Always generate both 2D and 3D** unified latent space plots
   - 2D for quantitative analysis (distances, angles)
   - 3D for qualitative exploration (curvature, separation)

2. **Use interactive 3D for exploration**, static 2D/3D for publication
   - Rotate interactive plots to find best viewing angle
   - Export that angle as static image for paper

3. **Generate trajectory evolutions for at least 3 runs**
   - Check consistency across stochastic runs
   - Identify outliers or mode collapse

4. **Always check temporal phase diagram**
   - Identifies critical timesteps for further analysis
   - Can then analyze velocity fields specifically at those steps

### For Publication

**Figure 1**: Sample images comparison (4×N grid)
**Figure 2**: Unified latent space 2D (with centroids and connecting lines)
**Figure 3**: Trajectory evolution 3D (showing divergence)
**Figure 4**: Temporal phase diagram (identifying critical timesteps)
**Supplementary**: Interactive 3D HTML files (online repository)

---

## Technical Notes

### PCA Variance Explained

Typical results for latent diffusion:
- PC1: ~40-50% variance
- PC2: ~15-25% variance
- PC3: ~8-15% variance

**Total in 2D**: ~60-75%
**Total in 3D**: ~70-85%

If 2D explains <60%, **use 3D visualizations** for better representation.

### Interactive Plotly Requirements

```python
# Install if needed
pip install plotly

# To view .html files:
# 1. Open in web browser directly
# 2. Or use in Jupyter:
from IPython.display import IFrame
IFrame('unified_latent_space_3d_interactive.html', width=1000, height=800)
```

### Trajectory Sampling

By default, we visualize:
- First 3 runs (runs 0, 1, 2)
- First sample in batch (sample_idx=0)

To visualize more:

```python
for run_idx in range(suite.config.num_runs):
    for sample_idx in range(suite.config.batch_size):
        plot_trajectory_evolution_2d3d(
            suite.results,
            suite.output_dir,
            suite.config,
            run_idx=run_idx,
            sample_idx=sample_idx
        )
```

---

## Summary: Are Visualizations Sufficient?

### Before Enhanced Visualizations: ⚠️ Partial

**Could answer**:
- Does SUPERDIFF produce hybrids? (YES, via sample images)
- Is κ balanced? (YES, via kappa plot)
- Are centroids different? (YES, via distance bars)

**Could NOT clearly answer**:
- Is composition linear? (UNCLEAR, needed unified view)
- Is it on-manifold? (UNCLEAR, needed trajectory paths)
- When does divergence occur? (UNCLEAR, needed temporal analysis)

### After Enhanced Visualizations: ✅ Comprehensive

**Can now clearly answer**:
- ✅ Is composition linear? (unified latent space: position of SD relative to A-B line)
- ✅ Is it on-manifold? (trajectory paths: smooth vs. sharp turns)
- ✅ When does divergence occur? (temporal phase: early/mid/late + peak detection)
- ✅ Does spatial grounding help? (semantic vs. spatial unified plots)

**Additional capabilities**:
- ✅ Interactive exploration (3D rotation)
- ✅ Quantitative + qualitative diagnostics
- ✅ Publication-ready figures
- ✅ Disambiguates all three hypotheses (intrinsic, geometric, spatial bias)

---

## Conclusion

The enhanced visualizations **fully address** your concerns:

1. **Single unified plots** ✅ - All conditions in one view
2. **3D alongside 2D** ✅ - Static + interactive
3. **Trajectory dynamics** ✅ - Full path evolution over time
4. **Temporal analysis** ✅ - Phase identification and critical timesteps

These visualizations provide **sufficient diagnostic power** to:
- Disambiguate semantic conjunction vs. latent intersection vs. off-manifold drift
- Identify whether hybridization is intrinsic, geometric, or spatial
- Determine critical timesteps for mechanistic analysis
- Support rigorous publication claims

**Recommendation**: Run spatial grounding experiments first, then generate enhanced visualizations for both semantic and spatial conditions. This will definitively answer whether the issue is spatial grounding or fundamental to the composition operator.
