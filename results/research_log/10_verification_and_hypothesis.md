# Chapter 10 — Verification Gates, Proposed Additions, and Revised Hypothesis

**Previous chapter:** [09 Composition Theory](09_composition_theory.md)
**Next chapter:** [11 Roadmap and Commands](11_roadmap_and_commands.md)

---

## 11. Verification Gates Before Composition

**Nothing in Phase 8 (full Strategy A LDM training) should begin until all four gates pass.**

### Gate definitions

| Gate | What we measure | How we measure it | Threshold |
|------|----------------|-------------------|-----------|
| G1 | Heads carry disease information | Probe AUC (rolling 3-epoch average) | Both heads > 0.75 |
| G2 | No cross-head leakage | Cross-head score at ep50 and ep100 | < 0.65 |
| G3 | Clean single-disease edits | Edit purity ratio V1, V2 | Both > 2.0 |
| G4 | $z_{\text{common}}$ preserves anatomy | SSIM on non-disease regions during anatomy transplant | > 0.85 |

### Gate status after Phase 2

| Gate | Best result to date | Threshold | Status |
|------|---------------------|-----------|--------|
| G1 | 0.774 mean (disentangle-E ep185) | > 0.75 | **Borderline — not stable** |
| G2 | 0.744 (inactivity-G ep50) | < 0.65 | **Not met** |
| G3 | Not yet measured | > 2.0 | **Not implemented** |
| G4 | Not yet measured | > 0.85 | **Not implemented** |

### How to implement G3 (edit purity)

```python
# Approximate spatial masks
cardiac_mask  = make_region_mask(H, W, row=(0.35, 0.55), col=(0.30, 0.70))
effusion_mask = make_region_mask(H, W, row=(0.70, 1.00), col=(0.10, 0.90))

# Swap z_cardio from a cardiomegaly image into a normal image
x_base    = decode(z_common, z_cardio_normal, z_effusion)
x_swapped = decode(z_common, z_cardio_disease, z_effusion)
delta = jnp.abs(x_swapped - x_base)

cardio_change   = delta[cardiac_mask].mean()
effusion_change = delta[effusion_mask].mean()
specificity_ratio = cardio_change / (effusion_change + 1e-6)
# Target: > 2.0
```

```bash
python utils/sepvae_diagnostics.py \
  --mode edit_purity \
  --checkpoint runs_sepvae/sepvae_full/best_checkpoint \
  --n_samples 200 \
  --cardiac_row_range 0.35 0.55 --cardiac_col_range 0.30 0.70 \
  --effusion_row_range 0.70 1.00 --effusion_col_range 0.10 0.90 \
  --out results/research_log/edit_purity_report.json
```

---

## Supplementary: Four Verifiability Criteria (V1–V4), Four Proposed Additions (P1–P4), and Revised Hypothesis

*The following provides the full treatment of each verifiability criterion with code, each proposed addition with implementation details, and the formal research hypothesis.*

---

### Criterion V1 — Changing $z_{\text{cardio}}$ alters heart enlargement only

**What to measure:** Decode $\hat{x}_{\text{edited}} = \text{Dec}(z_{\text{common}}, z_{\text{cardio}}^*, z_{\text{effusion}})$ where $z_{\text{cardio}}^*$ comes from a cardiomegaly image. Measure pixel-change map $|\hat{x}_{\text{edited}} - \hat{x}_{\text{original}}|$.

**Ideal result:** Change map concentrates in cardiac region (centre, mediastinum); pleural regions (lower lateral) unchanged.

**Current support:** `plot_latent_swap_grid` in `utils/sepvae_diagnostics.py` does this visually.

**Additional support when `--use_label_attention` is enabled:** The cardiomegaly attention map $A_\text{cardio} \in \mathbb{R}^{64 \times 64}$ is a direct spatial certificate of where the cardiomegaly head is routing its information. If $A_\text{cardio}$ concentrates over the cardiac silhouette region (central 35–55% height, 30–70% width) rather than the pleural region, V1 spatial selectivity is structurally enforced at the encoder level. The attention map can be thresholded and compared against an anatomical cardiac mask as a cheap approximation of the specificity ratio before running full latent swap experiments.

**Gap:** No quantitative metric. Implementation:
```python
cardiac_mask  = make_region_mask(H, W, row=(0.35, 0.55), col=(0.30, 0.70))
effusion_mask = make_region_mask(H, W, row=(0.70, 1.00), col=(0.10, 0.90))

x_base    = decode(z_common, z_cardio_normal,   z_effusion)
x_swapped = decode(z_common, z_cardio_disease, z_effusion)
delta     = jnp.abs(x_swapped - x_base)

cardio_change   = delta[cardiac_mask].mean()    # should be HIGH
effusion_change = delta[effusion_mask].mean()   # should be LOW
edit_purity_cardio = cardio_change / (effusion_change + 1e-6)
# Target: > 2.0
```

### Criterion V2 — Changing $z_{\text{effusion}}$ alters pleural fluid patterns only

Same methodology as V1 but swapping $z_{\text{effusion}}^*$ from an effusion image while keeping $z_{\text{cardio}}$ fixed.

**Ideal result:** Change map concentrates in costophrenic angles and lower lateral lung; cardiac silhouette unchanged.

**Additional support when `--use_label_attention` is enabled:** $A_\text{effusion}$ provides the symmetric spatial certificate for the effusion head. Effusion concentration should be easier to achieve than cardiomegaly given the localised, high-contrast nature of the finding. If $A_\text{effusion}$ fails to localise to the lower lateral regions after training, it suggests the backbone features at the relevant spatial positions are insufficient for effusion separation, motivating partial unfreezing (`--unfreeze_from layer3`).

**Target:** `specificity_ratio > 2.0` for effusion swaps.

### Criterion V3 — $z_{\text{common}}$ preserves anatomy, pose, and acquisition

**What to measure:** Decode with $z_{\text{common}}$ transferred between two normal images while keeping disease heads fixed. Measure SSIM between transferred and target on non-disease regions.

**Ideal result:** SSIM > 0.85 on lung fields; probe classifiers for cardiomegaly/effusion should give near-chance predictions.

**Gap:** Not implemented. Proposed implementation:
```python
z_common_new = encode_common(x_different_patient)
x_edited = decode(z_common_new, z_cardio, z_effusion)  # transplant anatomy
ssim_non_disease = compute_ssim(x_original, x_edited, mask=~(cardiac_mask | effusion_mask))
# Should be LOW (anatomy changed) — confirms z_common carries anatomy
```

The reciprocal: fix z_common, swap z_disease → SSIM over non-disease region should be HIGH (anatomy unchanged).

### Criterion V4 — Jointly composed sample resembles real co-morbid images

This is the ultimate test. A composed image from Strategy A should be closer to held-out real co-morbid images than either single-disease synthesis.

**What to measure:**
- FID between composed images and held-out co-morbid split
- Nearest-neighbour distance in VGG feature space
- (Ideal) CheXNet classifier should predict *both* cardiomegaly and effusion with high confidence

**Gap:** No co-morbid evaluation split prepared. No composition pipeline implemented yet (SepVAE training is the prerequisite).

```bash
# Generate N composed samples
python run/compose_diseases.py \
  --strategy a \
  --vae_checkpoint runs_sepvae/best \
  --ldm_common_checkpoint runs_ldm/common \
  --ldm_cardio_checkpoint runs_ldm/cardio \
  --ldm_effusion_checkpoint runs_ldm/effusion \
  --n_samples 1000 \
  --out_dir results/composed_samples/

# Compute FID against comorbid hold-out
python run/eval_fid.py \
  --generated results/composed_samples/ \
  --reference data/vinbig/comorbid_holdout/
```

### Summary of verifiability gaps

| Criterion | Qualitative | Quantitative | Gap |
|-----------|-------------|--------------|-----|
| V1: Cardiac edit purity | Swap grid ✓ | Edit purity metric | Not implemented |
| V2: Effusion edit purity | Swap grid ✓ | Edit purity metric | Not implemented |
| V3: z_common anatomy fidelity | PCA ✓ | SSIM on swapped anatomy | Not implemented |
| V4: Composition ≈ real comorbid | — | FID, NN-distance | Neither implemented |

**Recommendation:** Implement V1 and V2 metrics in `utils/sepvae_diagnostics.py` as a gate before committing to LDM training. These can be computed from existing SepVAE checkpoints with no additional training.

---

### Addition P1 — Explicit shared anatomical/background expert (LDM over $z_{\text{common}}$)

$z_{\text{common}}$ already plays the role of the anatomical/background expert in the SepVAE encoder. The critical missing piece is the **LDM over $z_{\text{common}}$** — without it, $z_{\text{common}}$ must be synthesised by the disease LDMs, which forces them to encode anatomy and reintroduces double-counting.

**Design:** `LDM_common` is an unconditional diffusion model trained on $z_{\text{common}}$ spatial maps. In Strategy A, it is sampled first; the disease LDMs are then conditioned on $z_{\text{common}}$.

**Architecture:**
```
LDM_common: p(z_common) — trained on z_common from normal images only
LDM_cardio: p(z_cardio | z_common) — z_common is the conditioning signal
LDM_effusion: p(z_effusion | z_common) — z_common is the conditioning signal
```

**Why this matters:** Once $z_{\text{common}}$ is sampled and fixed, the disease LDMs only need to model residual variation in their sub-blocks. Anatomy is already determined. This is the principal motivation for the factorised design.

### Addition P2 — Quantitative single-disease editing tests before conjunction

Conjunction will almost certainly fail if single-factor interventions are not clean. This should be a hard gate: the project should not proceed to composition experiments until V1 and V2 are passing quantitatively.

Proposed entry criterion: `specificity_ratio > 2.0` on both V1 and V2 on a 200-sample validation set, measured at the best SepVAE checkpoint.

```python
# For each checkpoint to be used for LDM training:
purity_cardio  = eval_edit_purity(model, split="cardio_only")
purity_effusion = eval_edit_purity(model, split="effusion_only")

if purity_cardio < 2.0 or purity_effusion < 2.0:
    print("GATE FAILED: head not clean enough for composition")
    sys.exit(1)
```

Additionally: pass decoded swapped images through a CheXNet-style classifier. An image with $z_{\text{cardio}}^*$ swapped in should increase classifier confidence for cardiomegaly; effusion confidence should be unchanged.

### Addition P3 — Jointly trained conditional model as ablation baseline

Without this, it is impossible to distinguish:
- **(a)** Composition fails because the semantic assumptions are wrong (diseases are intrinsically entangled)
- **(b)** Composition fails because the SepVAE disentanglement is insufficient

**Proposed baseline:** train a single conditional VAE or LDM with a 3-class label (normal, cardio, effusion, both) and condition on (cardio=1, effusion=1) at inference.

```
Single conditional VAE: encode(x, y=[0,1,2]) → z
Single conditional LDM: p(z | y_cardio=1, y_effusion=1) → composed z
```

**Evaluation table:**

| Model | FID on comorbid | SSIM cardiac region | SSIM effusion region |
|---|---|---|---|
| SepVAE + Strategy A | ? | ? | ? |
| Jointly conditional LDM | ? | ? | ? |
| Single-disease LDMs (no composition) | ? | ? | ? |

**If jointly-trained wins:** the assumption about score factorisation is wrong.
**If Strategy A wins:** disentangled composition is adding value beyond what a monolithic model can do.

### Addition P4 — CFG with unconditional = common expert

*Full mathematical treatment is in [Chapter 09](09_composition_theory.md) §CFG Addition P4.*

If the downstream LDMs are trained in a classifier-free guidance style, the unconditional score can be set to the common-expert output — i.e., the score of the LDM when all disease conditions are turned off:

$$\nabla_z \log p(z \mid c, e) = (1 - 2\lambda)\nabla_z \log p_\text{common}(z) + \lambda\nabla_z \log p(z \mid c) + \lambda\nabla_z \log p(z \mid e)$$

**Implementation note:** requires training a single conditional LDM with CFG dropout (where the condition is zeroed with probability $p_\text{uncond}$ during training) and treating the unconditioned output as the common-expert baseline. Architecturally simpler than Strategy A because there is only one LDM to train.

---

### Revised, Falsifiable Research Hypothesis

#### Original hypothesis (from initial framing):
> "If disease-related variation can be approximately factorized from shared anatomical and acquisition variation, then score composition in latent space should better approximate multi-pathology generation than composition in a fully entangled latent space."

#### Revised, falsifiable hypothesis:

> **If disease-related variation can be approximately factorized from shared anatomical and acquisition variation in the SepVAE latent space** — as measured by probe AUC > 0.75 per head, cross-head score < 0.65, and edit purity ratio > 2.0 — **then conditional sequential sampling from disease-specific sub-block LDMs conditioned on $z_{\text{common}}$ (Strategy A) will better approximate multi-pathology co-morbid synthesis**, as measured by:
> 1. FID against held-out comorbid images, compared to a jointly-trained conditional baseline
> 2. Dual-disease classifier confidence (both cardiomegaly and effusion high) on synthesised images
> 3. Anatomical consistency score (SSIM of non-disease regions vs. single-disease reference)

**Necessary conditions (gates before composition):**

| Gate | Metric | Threshold |
|------|--------|-----------|
| G1: Heads carry disease information | Probe AUC (both heads) | > 0.75 |
| G2: No cross-head leakage | Cross-head score | < 0.65 |
| G3: Clean single-disease edits | Edit purity ratio | > 2.0 |
| G4: z_common preserves anatomy | SSIM on normal-image swap | > 0.85 |

**If G1–G4 pass:** Run Strategy A composition experiments.
**If G1–G4 fail:** Continue SepVAE training improvements (current phase).
**Comparison baseline:** Jointly-trained conditional model (P3).
**Variant:** CFG-guided composition (P4) as a comparison within composition strategies.

#### What the results so far tell us about the gates:

| Gate | Best result to date | Threshold | Status |
|------|---------------------|-----------|--------|
| G1 (Probe AUC) | 0.774 mean (C/D/E ep185) | > 0.75 | **Borderline — not stable** |
| G2 (Cross-head) | 0.744 (inactivity-G ep50) | < 0.65 | **Not met** |
| G3 (Edit purity) | Not measured | > 2.0 | **Not measured** |
| G4 (z_common SSIM) | Not measured | > 0.85 | **Not measured** |

Current training phase is focused on closing G2 (R5a, R5b, R7) and stabilising G1 (R1–R3).

#### Key unknowns that determine whether the hypothesis can be tested:

1. Can the SepVAE achieve `specificity_ratio > 2.0`? (Current best is unmeasured; qualitative swap grid looks marginal.)
2. Does the VinBigData comorbid hold-out split have sufficient samples for FID to be meaningful? (FID requires ~1000 real samples for stability.)
3. Is cardiomegaly fundamentally too distributed a feature to be localised in $z_\text{cardio}$ given the stride-32 backbone? (The consistent cardiomegaly AUC underperformance suggests yes.)

If (3) is a hard barrier, the fallback position: "score composition achieves plausible multi-pathology generation even when the factorisation is imperfect, whereas direct joint conditioning fails to generalise to the comorbid case due to data sparsity."

---

*End of Chapter 10. Continue to [Chapter 11: Roadmap and Commands](11_roadmap_and_commands.md).*
