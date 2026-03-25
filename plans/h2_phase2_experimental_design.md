# H2 Phase 2 — Experimental Design Plan

## Context

H2 (The Representation Hypothesis) claims the composability gap is primarily a
**representation problem, not an inference problem**. Specifically: entangled latent
representations cause composition to fail, and this cannot be corrected by modifying
the inference-time composition rule alone.

The current implementation tests this only partially — it compares d0 (entangled) vs
d3 (factorized) under a single composition rule (κ_subst). The **load-bearing
falsification test (Case B)** — entangled model + improved inference rule — is missing.
Without Case B, a reviewer can argue the improvement came from switching composition
rules, not from fixing the representation.

---

## What the Inference Rules Are

There are two distinct classes of "inference rule" in this thesis — do not conflate them:

### Type A — Score-based composition (Phase 1, diffusion models)
Used in SD 3.5 experiments. At each denoising step t:
```
∇_x log p(x | c₁, c₂)  ≈  ∇_x log p(x | c₁) + ∇_x log p(x | c₂) − ∇_x log p(x)
```
Variants: PoE (equal weights), SuperDiff AND (solve Gκ = b per step).
These modify the score field at test time — no retraining required.

### Type B — Latent substitution (Phase 2, SepVAE / miniSepVAE)
Given three encoder calls, combine the latent heads:
```
z_composed = κ( z^(normal), z^(F1-only), z^(F2-only) )
```

**Currently implemented (only):**
```
κ_subst:  z_composed = [ z_c^(normal),  z_d1^(F1),  z_d2^(F2) ]
```

**Not yet implemented (needed for H2 ablation):**
```
κ_diff:   Δ₁ = z_d1^(F1) − z_d1^(normal)        # concept arithmetic
          Δ₂ = z_d2^(F2) − z_d2^(normal)
          z_composed = z^(normal) + [0, Δ₁, Δ₂]

κ_slerp:  per-head spherical linear interpolation between z^(normal) and z^(factor)
          (SLERP already implemented in run/analyze_sepvae_manifold.py — needs porting)

κ_PoE:    combine posterior Gaussians multiplicatively per head:
          μ* = (μ₁/σ₁² + μ₂/σ₂²) / (1/σ₁² + 1/σ₂²)
```

---

## The 2×2 Ablation (Core H2 Test)

|  | κ_subst (simple) | κ_diff / κ_slerp (improved) |
|---|---|---|
| **d0 — entangled** | Case A ✓ (already run) | **Case B — NOT YET (critical)** |
| **d3 — factorized** | Case C ✓ (already run) | Case D — nice to have |

**H2 prediction:**
- Cases A and B both fail (composition score low regardless of rule)
- Cases C and D both pass (composition score high)

**If Case B passes** (SLERP fixes entangled composition): H2 is falsified. The gap
was geometric, not representational. Do not publish H2 as stated.

**If Case B fails as predicted**: H2 is confirmed. The representation is the bottleneck.

---

## Why the Current Implementation Is Vulnerable

The failure mechanism for entangled d0:

> z_d1 encoded from x^(F1) leaks information about the common anatomy of x^(F1)
> specifically. The decoder was trained expecting correlated (z_c, z_d1). Substituting
> z_d1^(F1) into a composition with z_c^(normal) from a *different* image creates a
> contradiction the decoder cannot resolve. No inference rule can correct this because
> the problem is in what the encoder *encoded*, not how the codes are *combined*.

This argument needs to be verified empirically, not just stated. Case B is the test.

---

## Additional Experiments Needed

### 1. Causal curve: MI(z_d1, z_d2) vs composition score
Train five models with MI loss weight λ_mi ∈ {0, 0.01, 0.05, 0.10, 0.20}.
At each checkpoint, measure:
- Estimated I(z_d1; z_d2) via the trained MI discriminator
- Composition score under κ_subst

Plot composition_score vs MI_estimate. A monotonic negative relationship is causal
evidence, not just an ablation.

**File to create:** `scripts/plot_mi_vs_composition.py`

### 2. Stage-by-stage composition logging
Currently composition_score is only logged at the final training stage.
Log it at the end of each d0→d1→d2→d3→d4 stage to show which specific loss term
closes the gap most (nulling? MI? orthogonality?).

**Modification:** `run/train_generalize.py` — add composition eval to stage checkpoint.

### 3. Address I(z_d1; z_c) — known gap
The MI loss minimises I(z_d1; z_d2) only (independence between disease heads).
Composition also requires I(z_d1; z_c) ≈ 0 (disease independent of common).

Options:
- (a) Add a second MI loss term: D_mi_dc that distinguishes joint (z_d1, z_c) from
  shuffled marginals. Add at d3 or as a new d4 stage.
- (b) Measure I(z_d1; z_c) empirically and show it is already low as a side effect
  of nulling + spatial gate. If so, document it; no code change needed.

**Measure first (b), implement (a) only if needed.**

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `run/eval_h2_ablation.py` | CREATE | Load d0 + d3 checkpoints, run all κ variants, output 2×2 table |
| `run/train_generalize.py` | MODIFY | Add κ_diff, κ_slerp to evaluate_composition_multi_rule(); log per-stage |
| `scripts/plot_h2_ablation.py` | CREATE | Read CSVs from results/, plot 2×2 heatmap grid per domain |
| `scripts/plot_mi_vs_composition.py` | CREATE | Plot MI(z_d1,z_d2) vs composition_score across λ_mi sweep |

SLERP source to port: `run/analyze_sepvae_manifold.py` → `run/eval_h2_ablation.py`

---

## Relationship to PhD Thesis Structure

| Phase | Hypothesis | Status |
|-------|-----------|--------|
| Phase 1 | H1 — gap exists, is characterisable | Done. SD 3.5, 576 records, trajectory + reachability. PhD-level. |
| Phase 2 | H2 — gap is representation not inference | Partially done. Cases A+C exist. **Case B missing.** Borderline. |
| Phase 3 | H3 — clinical independence is sufficient | Not yet executed. Requires H2 established first. |

**The chain is PhD-quality in architecture. The execution gap is Case B.**

---

## Honest Critical Assessment

**H1**: Strong. The cycle-consistency reachability argument is the best part — it's a
lower-bound structural claim, not just a visual observation.

**H2**: Vulnerable without Case B. Also vulnerable to the charge that Phase 1 (diffusion,
score-based rules) and Phase 2 (VAE, latent substitution) are different model classes
tested with different rules — you changed both representation AND rule simultaneously.
The 2×2 ablation is the only clean answer to this.

**H3**: Most original claim in the thesis ("approximate clinical independence is
sufficient" relaxes mathematical to clinical). Silicosis-TB is compelling. But it is
entirely contingent on H2 being established, and needs a formal definition of
"ecologically valid" output quality.

---

## Verification Sequence

```bash
# Step 1: Train d0 and d3 on Domain A
conda run -n cxr python -m run.train_generalize --domain A --stage d0 --epochs 30
conda run -n cxr python -m run.train_generalize --domain A --stage d3 --epochs 30

# Step 2: Run full 2x2 ablation (Cases A, B, C, D)
conda run -n cxr python run/eval_h2_ablation.py \
  --domain A \
  --d0_ckpt runs_generalize/A_d0/best.pt \
  --d3_ckpt runs_generalize/A_d3/best.pt \
  --out results/h2_ablation_A.csv

# Step 3: Plot
conda run -n cxr python scripts/plot_h2_ablation.py --results_dir results/

# Step 4: MI sweep (5 runs with λ_mi ∈ {0, 0.01, 0.05, 0.10, 0.20})
for lmi in 0.0 0.01 0.05 0.10 0.20; do
  conda run -n cxr python -m run.train_generalize \
    --domain A --stage d2 --mi_weight $lmi --epochs 30 \
    --run_name A_d2_lmi${lmi}
done
conda run -n cxr python scripts/plot_mi_vs_composition.py --results_dir results/
```
