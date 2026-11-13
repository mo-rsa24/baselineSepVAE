# Script Comparison Guide

## Quick Answer

**Use `run_full_analysis.py` for complete analysis** — it now includes everything.

## Detailed Comparison

### 1. run_composition_analysis.py (Basic Single-Prompt Experiment)

**Use When:** Testing a single prompt pair without spatial comparison

**Command:**
```bash
python notebooks/run_composition_analysis.py --prompt-a "cat" --prompt-b "dog"
```

**What It Runs:**
- Single experiment: Monolithic, A, B, SUPERDIFF (4 conditions)
- One prompt type (semantic OR spatial, not both)

**Analyses Included:**
- ✅ Sample images comparison
- ✅ Trajectory geometry (norms, distances over time)
- ✅ Centroid statistics (distances, variance, PCA)
- ✅ Kappa dynamics
- ✅ PCA/t-SNE projections
- ✅ Manifold distances (k-NN, interpolation line)
- ✅ Velocity field alignment

**Analyses NOT Included:**
- ❌ Semantic vs. spatial comparison
- ❌ Enhanced visualizations (unified 2D/3D, trajectory paths, temporal phases)
- ❌ Advanced manifold geometry (geodesic distances, intrinsic dimensionality)

**Output:** `experiments/composition_analysis_<timestamp>/`

---

### 2. run_spatial_grounding.py (Semantic vs. Spatial Diagnostic)

**Use When:** Testing whether spatial grounding eliminates hybridization

**Command:**
```bash
python notebooks/run_spatial_grounding.py --object-a "cat" --object-b "dog"
```

**What It Runs:**
- TWO experiments: Semantic prompts + Spatial prompts
- 2×2 design: (Semantic, Spatial) × (Monolithic, SUPERDIFF)
- Comparative analysis between semantic and spatial

**Analyses Included:**
- ✅ Everything from `run_composition_analysis.py` × 2 (semantic + spatial)
- ✅ Side-by-side visual comparison (4-row image grid)
- ✅ Geometric comparison (distances between semantic vs. spatial)
- ✅ Kappa comparison
- ✅ Comparative report with interpretation

**Analyses NOT Included:**
- ❌ Enhanced visualizations (unified 2D/3D, trajectory paths, temporal phases)
- ❌ Advanced manifold geometry (geodesic distances, intrinsic dimensionality)

**Output:** `experiments/spatial_grounding_<timestamp>/`

**Key File:** `semantic_vs_spatial_comparison.png` — **MOST CRITICAL DIAGNOSTIC**

---

### 3. run_full_analysis.py (COMPLETE - Recommended!)

**Use When:** Running the complete analysis with all diagnostics

**Command:**
```bash
python notebooks/run_full_analysis.py --num-runs 15 --steps 500
```

**What It Runs:**
- Everything from `run_spatial_grounding.py`
- Enhanced visualizations for both semantic AND spatial
- Advanced manifold geometry for both semantic AND spatial

**Analyses Included:**
✅ **Everything from run_spatial_grounding.py:**
  - Semantic vs. spatial comparison
  - Standard diagnostics × 2

✅ **Enhanced Visualizations (NEW):**
  - Unified 2D latent space (all conditions in one plot)
  - Unified 3D latent space (static + interactive HTML)
  - Trajectory evolution 2D/3D (path visualization with time gradient)
  - Interactive 3D trajectories (rotatable Plotly HTML)
  - Temporal phase diagrams (critical timesteps)

✅ **Advanced Manifold Geometry (NOW INCLUDED!):**
  - Intrinsic dimensionality estimation (MLE method)
  - Intrinsic dimensionality (correlation dimension)
  - Geodesic vs. Euclidean distance comparison
  - Local tangent space alignment (PCA-based)
  - Trajectory curvature analysis (Menger curvature)
  - Manifold density estimation (k-NN distances)

**All analyses run for BOTH semantic and spatial prompts**

**Output:** `experiments/full_analysis_<timestamp>/`

**Directory Structure:**
```
experiments/full_analysis_<timestamp>/
├── semantic_vs_spatial_comparison.png     # Critical diagnostic
├── comparative_report.txt                 # Interpretation guide
├── geometric_comparison.png
├── kappa_comparison.png
│
├── semantic/                              # Semantic prompt results
│   ├── sample_images_comparison.png
│   ├── trajectory_geometry.png
│   ├── centroid_statistics.png
│   ├── kappa_dynamics.png
│   ├── pca_tsne_projections.png
│   ├── manifold_distances.png
│   ├── velocity_field_alignment.png
│   ├── summary_report.txt
│   │
│   ├── enhanced/                          # Enhanced visualizations
│   │   ├── unified_latent_space_2d.png
│   │   ├── unified_latent_space_3d.png
│   │   ├── unified_latent_space_3d_interactive.html
│   │   ├── trajectory_evolution_2d_run0_sample0.png
│   │   ├── trajectory_evolution_3d_run0_sample0.png
│   │   ├── trajectory_evolution_3d_interactive_run0_sample0.html
│   │   ├── trajectory_evolution_*_run1_*.png (for other runs)
│   │   └── temporal_phase_diagram.png
│   │
│   └── manifold/                          # Advanced geometry
│       ├── manifold_geometry_analysis.png
│       ├── manifold_geometry_results.txt
│       └── trajectory_curvature_analysis.png
│
└── spatial/                               # Spatial prompt results
    ├── (same structure as semantic/)
    ├── enhanced/
    └── manifold/
```

---

## Which Script Answers Which Hypothesis?

### Hypothesis 0: Lack of Spatial Inductive Bias ⭐

**Test:** Does spatial grounding eliminate hybridization?

**Script:** `run_spatial_grounding.py` or `run_full_analysis.py`

**Key File:** `semantic_vs_spatial_comparison.png`

**What to Look For:**
- Row 2 (semantic SUPERDIFF) vs Row 4 (spatial SUPERDIFF)
- If Row 4 shows co-presence → Hypothesis 0 confirmed
- If Row 4 still shows hybrids → Continue to geometric analysis

---

### Hypothesis 1: Intrinsic Hybridization (Probability Product)

**Test:** Is composition linear in latent space?

**Script:** `run_full_analysis.py` (needs unified latent space plot)

**Key Files:**
- `semantic/enhanced/unified_latent_space_2d.png`
- `semantic/enhanced/unified_latent_space_3d_interactive.html`

**What to Look For:**
- Is SUPERDIFF centroid on A-B line near midpoint?
- Distance(SUPERDIFF, midpoint) / Distance(A, B) < 0.3 → Linear interpolation
- If linear AND on-manifold → Intrinsic hybridization

---

### Hypothesis 2: Geometric Artifacts (Off-Manifold)

**Test:** Do trajectories take off-manifold shortcuts?

**Script:** `run_full_analysis.py` (needs trajectory paths + manifold analysis)

**Key Files:**
- `semantic/enhanced/trajectory_evolution_3d_interactive.html`
- `semantic/manifold/manifold_geometry_results.txt`
- `semantic/manifold/trajectory_curvature_analysis.png`

**What to Look For:**
- Trajectory curvature: High values = sharp turns = potential off-manifold
- Geodesic/Euclidean ratio > 1.2 → Off-manifold shortcuts
- Sharp bends in trajectory paths → Geometric artifacts

---

### Hypothesis 3: Semantic Mismatch

**Test:** Does SUPERDIFF match natural language interpretation?

**Script:** Any script (uses centroid distances)

**Key Files:**
- `comparative_report.txt`
- `centroid_statistics.png`

**What to Look For:**
- Distance(SUPERDIFF, Monolithic) < Distance(SUPERDIFF, midpoint)
- If close to monolithic → Semantic alignment
- If far from monolithic → Semantic mismatch

---

## Recommendation: Use run_full_analysis.py

### Why?

1. **Complete Coverage:** Tests all 4 hypotheses with all necessary diagnostics
2. **Semantic vs. Spatial:** Critical comparison built-in
3. **Enhanced Visualizations:** Unified plots, 3D, trajectories
4. **Advanced Geometry:** Geodesic distances, intrinsic dimensionality, curvature
5. **Both Conditions:** Analyzes semantic AND spatial prompts fully
6. **Publication Ready:** Generates all figures needed for paper

### Usage

```bash
# Full analysis (recommended for publication)
python notebooks/run_full_analysis.py \
    --num-runs 15 \
    --steps 500 \
    --object-a "a cat" \
    --object-b "a dog"

# Quick test (verify pipeline works)
python notebooks/run_full_analysis.py --quick

# Skip advanced visualizations (faster, for preliminary testing)
python notebooks/run_full_analysis.py --skip-enhanced-viz
```

### What You Get

After running, you'll have:

1. **Hypothesis 0 Test** → `semantic_vs_spatial_comparison.png`
2. **Hypothesis 1 Test** → `semantic/enhanced/unified_latent_space_*.png`
3. **Hypothesis 2 Test** → `semantic/manifold/manifold_geometry_results.txt`
4. **Hypothesis 3 Test** → `comparative_report.txt`
5. **All diagnostics for spatial prompts too**

---

## Workflow Recommendation

### Phase 1: Quick Test
```bash
python notebooks/run_full_analysis.py --quick
```
- Verify pipeline works (5 runs, 100 steps)
- Check output files are generated
- Review structure

### Phase 2: Full Experiment
```bash
python notebooks/run_full_analysis.py \
    --num-runs 15 \
    --steps 500 \
    --output-dir experiments/cat_dog_final
```
- Run with publication parameters
- Takes ~45-60 minutes

### Phase 3: Analysis
1. **Visual Inspection** → Open `semantic_vs_spatial_comparison.png`
2. **Read Report** → Open `comparative_report.txt`
3. **Explore 3D** → Open `semantic/enhanced/unified_latent_space_3d_interactive.html`
4. **Check Geometry** → Read `semantic/manifold/manifold_geometry_results.txt`

### Phase 4: Parameter Sweeps (if needed)
```bash
# Vary lift parameter
for lift in -20 -10 0 10 20; do
    python notebooks/run_full_analysis.py \
        --lift $lift \
        --output-dir experiments/lift_sweep/lift_$lift
done
```

---

## Summary Table

| Feature | run_composition_analysis | run_spatial_grounding | run_full_analysis ⭐ |
|---------|-------------------------|----------------------|---------------------|
| **Single prompt experiment** | ✅ | ❌ | ❌ |
| **Semantic vs. spatial** | ❌ | ✅ | ✅ |
| **Standard diagnostics** | ✅ | ✅ × 2 | ✅ × 2 |
| **Comparative analysis** | ❌ | ✅ | ✅ |
| **Enhanced visualizations** | ❌ | ❌ | ✅ |
| **Unified 2D/3D plots** | ❌ | ❌ | ✅ |
| **Trajectory paths** | ❌ | ❌ | ✅ |
| **Temporal phases** | ❌ | ❌ | ✅ |
| **Intrinsic dimensionality** | ❌ | ❌ | ✅ |
| **Geodesic distances** | ❌ | ❌ | ✅ |
| **Trajectory curvature** | ❌ | ❌ | ✅ |
| **Interactive 3D HTML** | ❌ | ❌ | ✅ |
| **Tests Hypothesis 0** | ❌ | ✅ | ✅ |
| **Tests Hypothesis 1** | Partial | Partial | ✅ |
| **Tests Hypothesis 2** | ❌ | ❌ | ✅ |
| **Tests Hypothesis 3** | ✅ | ✅ | ✅ |
| **Publication complete** | ❌ | ❌ | ✅ |

---

## Final Answer

**Yes, `run_full_analysis.py` now includes EVERYTHING:**

✅ Spatial grounding experiments (Hypothesis 0)
✅ Standard composition diagnostics
✅ Enhanced visualizations (unified plots, trajectories, temporal)
✅ Advanced manifold geometry (geodesic, intrinsic dimensionality, curvature)
✅ All analyses for BOTH semantic and spatial prompts
✅ All 4 hypotheses tested
✅ Publication-ready output

**Use this for your complete experimental pipeline.**
