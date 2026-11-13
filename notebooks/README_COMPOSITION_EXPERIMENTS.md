# SUPERDIFF Composition Analysis Framework

A comprehensive experimental framework for investigating why SUPERDIFF's logical AND operation produces **hybridization** rather than **co-presence**.

## ⚠️ CRITICAL EXPERIMENTAL DISTINCTION

This framework emphasizes a **crucial experimental control**:

**Semantic vs. Spatial Grounding**

- **Semantic prompts**: `"a cat and a dog"` (no spatial constraints)
- **Spatial prompts**: `"a cat on the left and a dog on the right"` (explicit positioning)

This distinction is critical for diagnosing whether hybridization results from:
1. **Lack of spatial inductive bias** (model doesn't know *where* to place things), OR
2. **Fundamental geometric/mathematical limitations** (off-manifold trajectories, probability product semantics)

**See: [Spatial Grounding Experiments](#spatial-grounding-experiments) below**

## 🎯 Quick Start

### Basic Composition Analysis
```bash
# Run default experiment (cat and dog)
python notebooks/run_composition_analysis.py

# Quick test (faster, for testing)
python notebooks/run_composition_analysis.py --quick

# Custom prompts
python notebooks/run_composition_analysis.py \
    --prompt-a "a red sports car" \
    --prompt-b "a blue truck" \
    --num-runs 20
```

### **Spatial Grounding Experiments (RECOMMENDED FIRST)**
```bash
# Test whether spatial grounding prevents hybridization
python notebooks/run_spatial_grounding.py

# Quick test
python notebooks/run_spatial_grounding.py --quick

# Custom objects and spatial descriptors
python notebooks/run_spatial_grounding.py \
    --object-a "a cat" \
    --object-b "a dog" \
    --spatial-desc-a "on the left side" \
    --spatial-desc-b "on the right side"
```

Results will be saved to `experiments/*/`

## 📁 Framework Structure

```
notebooks/
├── composition_experiments.py          # Main experiment suite
├── spatial_grounding_experiments.py    # Semantic vs. spatial comparison ⭐
├── manifold_geometry_analysis.py       # Advanced geometric diagnostics
├── run_composition_analysis.py         # Command-line interface
├── run_spatial_grounding.py            # Spatial grounding CLI ⭐
├── interactive_analysis.ipynb          # Jupyter notebook interface
├── COMPOSITION_ANALYSIS_GUIDE.md       # Comprehensive user guide
├── THEORETICAL_INSIGHTS.md             # Theoretical background
└── README_COMPOSITION_EXPERIMENTS.md   # This file
```

⭐ = **Recommended starting point for diagnosing hybridization**

## 🧪 What This Framework Does

### Experiments

#### Core Composition Experiments

Compares four sampling conditions:

1. **Monolithic**: `"A photograph of a cat and a dog"` (baseline)
2. **Individual A**: `"A photograph of a cat"` (concept A alone)
3. **Individual B**: `"A photograph of a dog"` (concept B alone)
4. **SUPERDIFF**: `A ∧ B` (mathematical composition)

All conditions start from the **same initial noise** for fair comparison.

#### Spatial Grounding Experiments ⭐

**Critical experimental control** comparing semantic vs. spatial prompts:

**Semantic (no spatial constraints):**
- Monolithic: `"a cat and a dog"`
- SUPERDIFF: `"a cat" ∧ "a dog"`

**Spatial (explicit positioning):**
- Monolithic: `"a cat on the left and a dog on the right"`
- SUPERDIFF: `"a cat on the left" ∧ "a dog on the right"`

**Key Question**: Does spatial grounding eliminate hybridization?

**Diagnostic Interpretation**:
- **If spatial grounding works** → Issue is **lack of spatial inductive bias** (solvable with better prompting)
- **If spatial grounding fails** → Issue is **geometric/mathematical** (fundamental limitation of probability product)

### Diagnostics

Analyzes **10+ metrics** across multiple dimensions:

#### Visual Analysis
- Sample image grids
- Trajectory evolution plots
- PCA/t-SNE projections

#### Statistical Analysis
- Centroid distances
- Distribution variances
- Kappa (balance) dynamics
- Velocity field alignment

#### Geometric Analysis
- Intrinsic dimensionality
- Geodesic vs. Euclidean distances
- Manifold curvature
- Tangent space alignment

## 📊 Key Outputs

### Generated Files

```
experiments/composition_analysis_<timestamp>/
├── sample_images_comparison.png        # Visual comparison
├── trajectory_geometry.png             # Trajectory analysis
├── centroid_statistics.png             # Distribution analysis
├── kappa_dynamics.png                  # Balance evolution
├── pca_tsne_projections.png            # 2D projections
├── manifold_distances.png              # Geometric distances
├── velocity_field_alignment.png        # Vector field analysis
├── manifold_geometry_analysis.png      # Advanced geometry
├── trajectory_curvature_analysis.png   # Curvature analysis
├── summary_report.txt                  # Comprehensive summary
└── manifold_geometry_results.txt       # Numerical results
```

### Key Metrics

| Metric | What It Tells Us |
|--------|------------------|
| `Distance(SD, (A+B)/2)` | Linear interpolation test |
| `Geodesic/Euclidean ratio` | On-manifold vs off-manifold |
| `Distance(SD, Monolithic)` | Semantic alignment |
| `Mean κ` | Balance between concepts |
| `Trajectory curvature` | Sharp turns (off-manifold shortcuts) |
| `Tangent alignment` | Local manifold structure |

## 🎓 Theoretical Framework

### Four Hypotheses (Hierarchical)

#### **Hypothesis 0: Lack of Spatial Inductive Bias** ⭐ (Test this FIRST)
- **Theory**: The model doesn't know *where* to place objects without explicit spatial prompts
- **Test**: Spatial grounding experiments
- **Evidence**: Spatial SUPERDIFF produces co-presence, semantic SUPERDIFF produces hybrids
- **Implication**: Hybridization is a **prompting issue**, not a mathematical issue
- **Solution**: Use spatially grounded prompts

#### 1. **Intrinsic Hybridization** (Probability Product)
- **Theory**: `p(x|A∧B) ∝ p(x|A)·p(x|B)` rewards feature blending
- **Test**: Linear interpolation test, even with spatial grounding
- **Evidence**: Linear interpolation (distance to midpoint < 0.3), on-manifold composition
- **Implication**: Not a bug, but semantic mismatch between math and language
- **Solution**: Alternative composition operators (mixture, attention-based)

#### 2. **Geometric Artifacts** (Off-Manifold)
- **Theory**: Euclidean operations violate manifold structure
- **Test**: Geodesic vs. Euclidean distance ratio
- **Evidence**: Geodesic >> Euclidean (ratio > 1.2), high trajectory curvature
- **Implication**: Can be improved with manifold-aware methods
- **Solution**: Geodesic interpolation, tangent space projection

#### 3. **Semantic Mismatch** (Math ≠ Language)
- **Theory**: Mathematical AND ≠ linguistic "and" at a fundamental level
- **Test**: Distance to monolithic prompt
- **Evidence**: SUPERDIFF ≠ monolithic prompt, even with spatial grounding
- **Implication**: Need structured compositional representations
- **Solution**: Object-centric models, scene graphs, hierarchical generation

### Decision Tree

```
Is Distance(SUPERDIFF, (A+B)/2) small?
├─ YES → Linear interpolation
│   └─ Is Geodesic/Euclidean ≈ 1?
│       ├─ YES → Intrinsic hybridization (Hypothesis 1)
│       └─ NO → Off-manifold artifacts (Hypothesis 2)
└─ NO → Non-linear composition
    └─ Is Distance(SUPERDIFF, Monolithic) small?
        ├─ YES → Semantic alignment
        └─ NO → Novel semantic space
```

## 🚀 Usage Examples

### Command Line

```bash
# Basic usage
python notebooks/run_composition_analysis.py

# Custom experiment
python notebooks/run_composition_analysis.py \
    --prompt-a "a photograph of a cat" \
    --prompt-b "a photograph of a dog" \
    --prompt-composed "a photograph of a cat and a dog" \
    --num-runs 20 \
    --steps 500 \
    --output-dir experiments/my_experiment

# Quick test
python notebooks/run_composition_analysis.py --quick

# Skip manifold analysis (faster)
python notebooks/run_composition_analysis.py --skip-manifold-analysis
```

### Python API

```python
from notebooks.composition_experiments import ExperimentConfig, run_composition_experiments

config = ExperimentConfig(
    prompt_a="A photograph of a cat",
    prompt_b="A photograph of a dog",
    prompt_composed="A photograph of a cat and a dog",
    num_runs=20,
    batch_size=4,
    num_inference_steps=500,
    guidance_scale=7.5,
    lift=0.0,
    output_dir="experiments/cat_dog_analysis"
)

run_composition_experiments(config)
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/interactive_analysis.ipynb
```

Then run all cells for an interactive exploration interface.

## 📖 Documentation

### Comprehensive Guides

1. **[COMPOSITION_ANALYSIS_GUIDE.md](COMPOSITION_ANALYSIS_GUIDE.md)**
   - Complete user guide
   - How to interpret results
   - Experimental design
   - Decision trees for interpretation

2. **[THEORETICAL_INSIGHTS.md](THEORETICAL_INSIGHTS.md)**
   - Theoretical background
   - Three hypotheses explained
   - Connection to literature
   - Open research questions

3. **[interactive_analysis.ipynb](interactive_analysis.ipynb)**
   - Step-by-step Jupyter notebook
   - Interactive visualizations
   - Export utilities

## 🔬 Advanced Usage

### Parameter Sweeps

```python
from notebooks.composition_experiments import ExperimentConfig, CompositionExperimentSuite

# Sweep over lift parameter
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

### Custom Analysis

```python
from notebooks.manifold_geometry_analysis import ManifoldGeometryAnalyzer

# Analyze custom latent samples
analyzer = ManifoldGeometryAnalyzer(latent_samples)
intrinsic_dim = analyzer.estimate_intrinsic_dimension(k=20)
alignment = analyzer.compute_local_pca_alignment(reference_samples)
```

### Trajectory Analysis

```python
from notebooks.manifold_geometry_analysis import analyze_trajectory_curvature

trajectories_dict = {
    'monolithic': suite.results['monolithic']['trajectories'],
    'superdiff': suite.results['superdiff']['trajectories']
}

analyze_trajectory_curvature(trajectories_dict, output_dir='experiments/curvature')
```

## 🔬 Spatial Grounding Experiments

### Why This Matters

The **most important diagnostic** for understanding SUPERDIFF hybridization is testing whether **spatial grounding** changes the outcome.

### Experimental Design

Compares 2×2 conditions:

|  | **Semantic** | **Spatial** |
|---|---|---|
| **Monolithic** | "cat and dog" | "cat on left, dog on right" |
| **SUPERDIFF** | "cat" ∧ "dog" | "cat on left" ∧ "dog on right" |

### Running the Experiment

```bash
# Default (cat and dog with left/right positioning)
python notebooks/run_spatial_grounding.py

# Quick test
python notebooks/run_spatial_grounding.py --quick

# Custom objects
python notebooks/run_spatial_grounding.py \
    --object-a "a red car" \
    --object-b "a blue truck" \
    --spatial-desc-a "in the foreground" \
    --spatial-desc-b "in the background"

# Different spatial relations
python notebooks/run_spatial_grounding.py \
    --spatial-desc-a "at the top" \
    --spatial-desc-b "at the bottom"
```

### Output Files

```
experiments/spatial_grounding_*/
├── semantic_vs_spatial_comparison.png  # Side-by-side visual comparison ⭐
├── geometric_comparison.png            # Distance metrics
├── kappa_comparison.png                # Balance dynamics
├── comparative_report.txt              # Interpretation guide ⭐
├── semantic/                           # Full semantic analysis
└── spatial/                            # Full spatial analysis
```

### Interpreting Results

**Visual Inspection** (most important):

Open `semantic_vs_spatial_comparison.png` and compare:
- **Row 2** (Semantic SUPERDIFF): "cat" ∧ "dog"
- **Row 4** (Spatial SUPERDIFF): "cat on left" ∧ "dog on right"

**Three Possible Outcomes**:

#### Outcome A: Spatial grounding WORKS
- **Evidence**: Row 4 shows co-presence (two distinct objects), Row 2 shows hybrids
- **Conclusion**: Hybridization is due to **lack of spatial inductive bias**
- **Implication**: The UNet/VAE architecture CAN support composition when spatially grounded
- **Solution**: Always use spatial prompts with SUPERDIFF, or develop spatial conditioning

#### Outcome B: Spatial grounding HELPS but doesn't fully solve it
- **Evidence**: Row 4 shows improvement but still some fusion
- **Conclusion**: Partial geometric/mathematical limitation + spatial bias
- **Implication**: Need both spatial grounding AND geometric improvements
- **Solution**: Combine spatial prompts with manifold-aware composition

#### Outcome C: Spatial grounding has NO EFFECT
- **Evidence**: Row 4 looks similar to Row 2 (both show hybrids)
- **Conclusion**: Issue is **fundamental** to probability product or geometry
- **Implication**: Spatial information is ignored or lost during composition
- **Solution**: Rethink composition operator entirely (mixture, attention-based, etc.)

### Geometric Analysis

The experiment also measures:

1. **Linear interpolation test**: Does spatial change distance to (A+B)/2?
2. **Semantic alignment**: Does spatial improve match to monolithic prompt?
3. **Balance (κ)**: Does spatial change A-B weighting dynamics?

See `comparative_report.txt` for detailed interpretation.

---

## 🎨 Additional Experimental Scenarios

### Test Different Semantic Relationships

```bash
# Foreground/background
python run_spatial_grounding.py \
    --spatial-desc-a "in the foreground" \
    --spatial-desc-b "in the background"

# Top/bottom
python run_spatial_grounding.py \
    --spatial-desc-a "at the top" \
    --spatial-desc-b "at the bottom"

# Near/far
python run_spatial_grounding.py \
    --spatial-desc-a "near the camera" \
    --spatial-desc-b "far from the camera"
```

### Attribute Composition

```bash
python run_composition_analysis.py \
    --prompt-a "a red object" \
    --prompt-b "a round object"

# With spatial grounding
python run_spatial_grounding.py \
    --object-a "a red object" \
    --object-b "a round object" \
    --spatial-desc-a "on the left" \
    --spatial-desc-b "on the right"
```

### Style Composition

```bash
python run_composition_analysis.py \
    --prompt-a "a photograph in Van Gogh style" \
    --prompt-b "a photograph in Picasso style"
```

## 📈 Interpreting Results

### Key Questions to Answer

1. **Does SUPERDIFF produce hybrids or co-presence?**
   - Look at: `sample_images_comparison.png`

2. **Is composition linear in latent space?**
   - Look at: `centroid_statistics.png`
   - Metric: `Distance(SD, (A+B)/2) / Distance(A, B)`
   - < 0.3 = linear, > 0.5 = non-linear

3. **Does composition stay on-manifold?**
   - Look at: `manifold_geometry_analysis.png`
   - Metric: `Geodesic / Euclidean ratio`
   - ≈ 1.0 = on-manifold, > 1.2 = off-manifold

4. **Is composition balanced?**
   - Look at: `kappa_dynamics.png`
   - Metric: `Mean κ`
   - ≈ 0.5 = balanced, else biased

5. **Does SUPERDIFF match natural language semantics?**
   - Look at: `centroid_statistics.png`
   - Metric: `Distance(SD, Monolithic) / Distance(A, B)`
   - < 0.5 = semantic match, > 0.5 = mismatch

### Reading the Summary Report

The `summary_report.txt` provides:
- Configuration parameters
- Key numerical results
- Automatic interpretation
- Recommendations for further investigation

## 🔧 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python run_composition_analysis.py --batch-size 2

# Reduce number of runs
python run_composition_analysis.py --num-runs 5

# Use quick mode
python run_composition_analysis.py --quick
```

### Slow Execution

```bash
# Skip manifold analysis (saves ~50% time)
python run_composition_analysis.py --skip-manifold-analysis

# Reduce inference steps
python run_composition_analysis.py --steps 100

# Both
python run_composition_analysis.py --quick
```

### CUDA Errors

```python
import torch
torch.cuda.empty_cache()
```

## 🔗 Dependencies

Required packages:
```
torch
diffusers
transformers
matplotlib
seaborn
scikit-learn
scipy
numpy
Pillow
```

Install with:
```bash
pip install torch diffusers transformers matplotlib seaborn scikit-learn scipy numpy Pillow
```

## 📚 References

### SUPERDIFF Paper
- *"The Superposition of Diffusion Models Using the Itô Density Estimator"*

### Related Methods
- Composable Diffusion (Liu et al., 2022)
- Stable Diffusion (Rombach et al., 2022)
- GLIDE (Nichol et al., 2022)

### Manifold Learning
- Isomap (Tenenbaum et al., 2000)
- LLE (Roweis & Saul, 2000)
- Intrinsic Dimensionality (Levina & Bickel, 2005)

## 🤝 Contributing

Potential extensions:
- CLIP-based semantic metrics
- Object detection for quantifying co-presence
- Attention map visualization
- Alternative composition operators (OR, NOT, XOR)
- Multi-prompt composition (A ∧ B ∧ C)
- Temporal analysis of when hybridization emerges

## 📝 Citation

If you use this framework in your research:

```bibtex
@software{superdiff_composition_framework,
  title={SUPERDIFF Composition Analysis Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/superdiff-ldm}
}
```

## 📧 Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: your.email@domain.com

## 📜 License

[Your chosen license]

---

## 🎯 Quick Reference

### Command Cheat Sheet

```bash
# Basic
python notebooks/run_composition_analysis.py

# Quick test
python notebooks/run_composition_analysis.py --quick

# Custom prompts
python notebooks/run_composition_analysis.py \
    --prompt-a "PROMPT_A" \
    --prompt-b "PROMPT_B"

# Full control
python notebooks/run_composition_analysis.py \
    --prompt-a "PROMPT_A" \
    --prompt-b "PROMPT_B" \
    --num-runs 20 \
    --batch-size 4 \
    --steps 500 \
    --guidance-scale 7.5 \
    --lift 0.0 \
    --output-dir experiments/my_exp

# Fast mode
python notebooks/run_composition_analysis.py \
    --quick \
    --skip-manifold-analysis
```

### Key Files

| File | Purpose |
|------|---------|
| `composition_experiments.py` | Main experiment code |
| `manifold_geometry_analysis.py` | Geometric diagnostics |
| `run_composition_analysis.py` | CLI interface |
| `interactive_analysis.ipynb` | Jupyter interface |
| `COMPOSITION_ANALYSIS_GUIDE.md` | User manual |
| `THEORETICAL_INSIGHTS.md` | Theory & background |

### Key Metrics

| Metric | File | Interpretation |
|--------|------|----------------|
| Linear interpolation | `centroid_statistics.png` | < 0.3 = linear |
| On-manifold | `manifold_geometry_analysis.png` | ratio ≈ 1 |
| Balanced | `kappa_dynamics.png` | κ ≈ 0.5 |
| Semantic match | `centroid_statistics.png` | < 0.5 = match |

---

**Happy experimenting! 🚀**
