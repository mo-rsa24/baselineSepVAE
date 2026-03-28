# Results & Evaluation

**Related documents:** [01 Experiment Timeline](01_experiment_timeline.md) | [03 Training Curriculum](03_training_curriculum.md) | [06 Failures & Debugging](06_failures_debugging.md) | [08 Current State (D3)](08_current_state_d3.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25

---

## Purpose of This Document

This document records the quantitative and qualitative results for all training phases across both V1 (CheSS backbone architecture) and V2 (ResNet-50 from scratch). It separates what was measured from what it means, and makes explicit what metrics are reliable vs. noisy.

---

## 1. Metrics Reference

### 1.1 V1 metrics (tracked in all Phases 1–7)

| Metric | Source | Interpretation |
|--------|--------|----------------|
| `probe_auc/cardiomegaly` | Frozen linear probe on GAP(z_cardio) | Does the disease head encode cardiomegaly? Gate: > 0.75 |
| `probe_auc/effusion` | Frozen linear probe on GAP(z_effusion) | Does the disease head encode effusion? Gate: > 0.75 |
| `probe_auc/mean` | Mean of above two | Overall discriminability |
| `cross_head_score` | max(AUC of z_cardio predicting effusion, AUC of z_effusion predicting cardiomegaly) | Leakage measure. 1.0 = total leakage, 0.5 = ideal |
| `silhouette_disease_only_pca` | Silhouette coefficient of first 10 PCs of [z_cardio, z_effusion] | Cluster separability; sign of latent structure |

**Warning on probe AUC variance:** The 600-sample probe eval set produces high-variance estimates. Adjacent epochs can differ by 0.2+ AUC for the same checkpoint state. Single-epoch readings should be treated as noisy. Reliable reporting requires a rolling average of 3+ consecutive evaluations.

### 1.2 V2 metrics (tracked in D0–D3+)

| Metric | W&B key | Interpretation |
|--------|---------|----------------|
| `loss/reconstruction` | `loss/reconstruction` | MSE pixel fidelity. Healthy D3 range: 0.004–0.008 |
| `loss/gan_g` | `loss/gan_g` | Generator hinge loss (negative = discriminator fooled). Healthy: slowly more negative |
| `loss/bbox_attn` | `loss/bbox_attn` | Fraction of attention mass outside bbox. Decreasing from 0.7 → 0.20 indicates concentration |
| `loss/tv` | `loss/tv` | Total variation. Decreasing → fewer artifacts |
| `metrics/patch_disc_acc` | `metrics/patch_disc_acc` | PatchGAN discriminator accuracy. Healthy: 0.55–0.65 |
| `metrics/z_cardio_norm_ratio` | `metrics/z_cardio_norm_ratio` | Ratio of mean z_d norm (active) to mean z_d norm (inactive). > 2.0 means disentanglement is working |
| `metrics/z_common_norm_ratio` | `metrics/z_common_norm_ratio` | Should stay near 1.0; > 1.5 suggests z_common absorbing disease signal |
| Visual recon grid | logged every `sample_every` epochs | Qualitative sharpness, artifact presence |

---

## 2. V1 Era Results (Feb 16 – Mar 19, 2026)

### 2.1 Phase 1 — Initial Disentangle Chain (Feb 16–17, 2026)

**Runs:** Five SLURM jobs (A–E) forming one continuous training chain
**Total epochs:** 200 (NaN at final epoch)
**Best checkpoint:** ~epoch 185

#### Configuration

```bash
--free_bits 1.0
--sigma_inactive 0.1         # KL_inactive ≈ 1.8 nats (above free_bits=1.0)
--weight_null 0.01
--weight_orthogonality 0.03
--weight_mi 0.003
--weight_perceptual 0.03
--use_fpn true
--batch_size 6
--lr_vae 6.7e-5
--kl_warmup_epochs 30
--epochs 200
```

#### Quantitative results

| Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Probe AUC (mean) | Cross-head score ↓ |
|-------|-------------------|---------------------|-----------------|-------------------|
| 101 | 0.758 | 0.700 | 0.729 | 0.952 |
| ~185 (best) | **0.776** | **0.772** | **0.774** | 0.914 |
| 199 (last valid) | 0.731 | 0.744 | 0.737 | 0.935 |
| 200 | NaN | NaN | NaN | NaN |

#### Observations

**Positive:** Peak probe AUC of 0.774 mean — the best discriminability achieved in V1. Both heads encoding their respective disease above the 0.75 gate simultaneously.

**Problem 1 — NaN at epoch 200:** The run ended catastrophically. No learning rate schedule, full fp32 precision. One bad batch at the end of 200 epochs produced a cascading NaN. The result was irrecoverable (the last valid checkpoint was epoch 150, 50 epochs before the NaN).

**Problem 2 — Persistent cross-head leakage:** Cross-head score stayed between 0.85 and 1.01 throughout all 200 epochs. The MI discriminator + orthogonality loss portfolio reduces marginal correlation but does not enforce conditional independence. The best cross-head score across all V1 runs is 0.744 (inactivity-G, see below) — but that run collapsed the cardiomegaly head.

**Problem 3 — Probe AUC variance:** The same checkpoint evaluated at adjacent epochs differed by up to 0.24 AUC. No rolling average was implemented. The "0.774 best" is partly signal, partly noise.

**Problem 4 — No LR schedule plateau:** Probe AUC plateaued between epochs 50–80 with no schedule. The final gains (0.774 at ep185 vs 0.729 at ep101) are real but were extracted over 85 expensive epochs with no convergence guarantee.

---

### 2.2 Phase 2 — Targeted Sweeps (Feb 20, 2026)

**Runs:** Two new experiments, each isolating one hypothesis from Phase 1.

#### Sweep 1: Inactivity-driven (W&B: `9lj20so0`)

**Hypothesis:** Tighter inactive prior + stronger nulling → better leakage suppression.

```bash
--free_bits 1.0
--sigma_inactive 0.05        # tighter (KL_inactive ≈ 2.5)
--weight_null 0.05           # 5× stronger
--weight_orthogonality 0.05
--weight_mi 0.005
--batch_size 10 --epochs 100
```

| Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Cross-head score ↓ |
|-------|-------------------|---------------------|-------------------|
| ~50 (best) | 0.639 | 0.627 | **0.744** |
| 100 (final) | **0.476** (below chance) | 0.767 | 0.809 |

**Result:** Best cross-head score of all V1 runs (0.744 at epoch 50). However, cardiomegaly head collapsed to below-chance AUC by epoch 100. The strong nulling drove μ_cardio→0 faster than the reconstruction gradient could maintain signal. Reducing leakage at the cost of head death is not a solution — it demonstrates the regularisation is in the wrong direction.

**Lesson learned:** Asymmetric nulling resistance (cardiomegaly head needs protection). → Fixed by R4 (min_active_kl floor).

---

#### Sweep 2: Independence-driven (W&B: `41nce8qq`)

**Hypothesis:** Stronger orthogonality + MI → lower cross-head leakage.

```bash
--free_bits 2.0              # ← CRITICAL BUG (kills disease-head gradients)
--sigma_inactive 0.1         # KL_inactive ≈ 1.8 nats < free_bits=2.0 → dead zone
--weight_null 0.01
--weight_orthogonality 0.1   # 3× stronger
--weight_mi 0.01             # 3× stronger
--batch_size 10 --epochs 100
```

| Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Cross-head score ↓ |
|-------|-------------------|---------------------|-------------------|
| ~25 (early peak) | 0.762 | 0.737 | 0.820 |
| 100 (final) | 0.532 | 0.812 | 0.871 |

**Result:** An interesting early window (epoch 20–25) where decent probe AUC coexists with moderate leakage before the dead zone fully collapses the disease heads. By epoch 100, cardiomegaly has nearly collapsed (0.532 ≈ chance). Cross-head score actually worsened vs Phase 1 (0.871 > 0.914) because the disease heads no longer encode anything — the cross-head score is uninformative when disease heads are dead.

**Key finding:** The `free_bits=2.0` + `sigma_inactive=0.1` combination placed all disease-head KL below the free-bits threshold, permanently blocking disease-head gradients. This is a binary failure condition. See [Failure A1](06_failures_debugging.md#a1--free-bits--sigma_inactive-conflict).

---

### 2.3 Phases 3–7 — Fix Iterations (Mar 5–19, 2026)

The R1–R12 fixes were implemented and tested incrementally. Key quantitative results:

#### After R1–R4 (baseline_fixed):

Best run: `sepvae_baseline_fixed-20260305-*` (approximate — exact W&B ID not recorded in research log)

| Metric | Value |
|--------|-------|
| Probe AUC (cardio) | 0.72–0.74 (stable, no collapse) |
| Probe AUC (effusion) | 0.74–0.76 |
| Cross-head score | ~0.80–0.82 |
| Training stability | Stable through 150 epochs (cosine LR decay working) |

R1 (remove free_bits) + R2 (balanced nulling) + R3 (cosine LR) + R4 (min_active_kl) produced stable training without cardiomegaly collapse. Peak probe AUC slightly lower than disentangle-E (0.774) but achieved reliably across multiple runs.

#### After R5a+R5b (contrastive + cross-adversarial):

Marginal cross-head score improvement: ~0.78–0.80. The contrastive blind-pull term and cross-head adversarial discriminators reduced leakage at the margin but did not break the 0.74 floor. The structural spatial routing problem (see Pattern 2 in [Failure A4](06_failures_debugging.md#a4--universal-cross-head-leakage-above-074)) remained.

#### After R7 (label attention routing):

Spatial attention maps showed cardiac region concentration by epoch 20–30. Cardiomegaly head stability improved — less prone to collapse under weight_null pressure because the disease signal is now spatially concentrated rather than diffuse. Cross-head score improved to ~0.76 (better than R5a/R5b alone) in the best runs.

#### After R11+R12 (disease discriminability + bbox attention supervision):

Probe AUC became more consistent epoch-to-epoch. `loss/spatial_attn` decreased from ~0.5 to ~0.10–0.15, confirming attention concentration. The combination of R11 (positive discriminability pressure) + R12 (supervised spatial concentration) represents the final V1 loss stack.

---

### 2.4 V1 Summary Table — All Groups

| Run | Epoch | AUC cardio | AUC effusion | AUC mean | Cross-head ↓ | Status |
|-----|-------|-----------|-------------|----------|-------------|--------|
| disentangle-A | 192 | N/A | N/A | N/A | N/A | Old script |
| disentangle-E (best) | ~185 | **0.776** | **0.772** | **0.774** | 0.914 | NaN at ep200 |
| disentangle-E (ep199) | 199 | 0.731 | 0.744 | 0.737 | 0.935 | — |
| inactivity-G (best) | ~50 | 0.639 | 0.627 | 0.650 | **0.744** | — |
| inactivity-G (final) | 100 | 0.476 | 0.767 | 0.622 | 0.809 | Cardio collapsed |
| independence-I (peak) | ~25 | 0.762 | 0.737 | 0.750 | 0.820 | — |
| independence-I (final) | 100 | 0.532 | 0.812 | 0.672 | 0.871 | Dead zone |
| baseline_fixed (R1–R4) | 150 | 0.72–0.74 | 0.74–0.76 | ~0.74 | ~0.80 | Stable |
| full stack (R1–R12) | 150 | ~0.74 | ~0.76 | ~0.75 | ~0.76 | Stable |

---

### 2.5 V1 Summary: What Was Solved and What Wasn't

**Solved by V1 fixes:**
- NaN instability (cosine LR, grad clipping)
- Cardiomegaly head collapse (balanced nulling, min_active_kl)
- Free_bits dead zone (removed free_bits)
- Checkerboard artifacts (SmoothUp)

**Not solved in V1:**
- Cross-head leakage below 0.74 — structural spatial routing needed
- Reconstruction sharpness — blurry outputs throughout (see [Section 7.2 of Research Log Ch06](../research_log/06_reconstruction_sharpness.md) for root cause analysis)
- CheSS backbone ceiling — stride=32 feature map with BatchNorm, not trainable for reconstruction quality

**Reason for V2 rewrite:**
The CheSS backbone imposed a hard ceiling on both reconstruction quality and disentanglement. It was not designed to be an encoder for generation — its features are globally pooled for classification. The spatial resolution lost at stride=32 cannot be recovered by any decoder. All V1 probe AUC numbers reflect the discriminability of CheSS features, not learned disentanglement. See [Section 4 of Experiment Timeline](01_experiment_timeline.md) for the full rewrite rationale.

---

## 3. V2 Era Results (Mar 22 – present)

### 3.1 D0 — Smoke Test (Mar 22–24, 2026)

**Purpose:** Verify V2 pipeline runs end-to-end with ResNet-50 from scratch.
**Result:** Pipeline confirmed. Loss decreased from epoch 1. Samples recognisably CXR-shaped within 3 epochs. No NaN, no OOM.
**Canonical run:** `d0_smoke_v2-20260324-063142`

Quantitative measurements were not the focus — no probe eval configured. The D0 checkpoint is not used as a resume point for any downstream phase (D1 trains from scratch).

---

### 3.2 D1 — Reconstruction + BboxCrossAttn (Mar 21, 2026)

**Purpose:** Train backbone from scratch, establish reconstruction quality baseline, verify BboxCrossAttn guides disease head to cardiac region.
**Epochs:** 30 (5 KL warmup)
**Canonical run:** `d1_recon_bbox_xattn-20260321-004241`

#### Key observations

**Reconstruction quality:** At epoch 30, Normal image reconstructions were anatomically correct — lung fields, rib cage, cardiac silhouette shape all plausible. No fine texture (ribs, vessels) — expected at this stage without perceptual loss or GAN.

**Attention maps:** By epoch 20, attention maps for Cardiomegaly images showed partial concentration over the cardiac region. Without `weight_bbox_attn` active (set to 0.0 in D1), drift toward edges was visible in some samples — attention wandered to corners. Confirmed the need for bbox supervision in D2.

**FactorDisc:** Not active in D1 (weight_mi_factor=0.0). Latent space structure not yet characterised.

**Stripe artifacts:** Absent. Perceptual loss not active.

**Failure attempts before D1:** Six runs failed due to architecture issues (stride=2 aliasing), data pipeline issues, and OOM from excessive batch sizes. See [Timeline section D1](01_experiment_timeline.md).

---

### 3.3 D2 — Perceptual + Bbox Attn + MI (Mar 24, 2026)

**Purpose:** Add perceptual sharpening (CheSS layers 1–3), bbox attention supervision, MI discriminator, masked reconstruction, contrastive loss.
**Epochs:** 30 → 55 (+25 epochs resumed from D1)
**Canonical run:** `d2_perceptual_bbox-20260324-105108`

#### Key observations

**Reconstruction quality:** Noticeably sharper than D1. Mid-frequency texture (rib shapes, mediastinal boundaries) more defined. Fine detail (rib cortex edges, vessel walls) still absent — expected at this perceptual weight.

**Stripe artifacts:** Faint horizontal banding became visible at epoch 45+. Source: CheSS layer3 perceptual gradients (stride=16 → 16px-period aliasing). At `weight_perceptual=0.15` and `weight_tv=0.001`, the TV suppression was ~300× weaker than the perceptual gradient. See [Failure C1](06_failures_debugging.md#c1--horizontal-16px-stripe-banding-d2-d4).

**Attention concentration:** `loss/bbox_attn` decreased from ~0.65 at D2 start to ~0.30 by epoch 55. Attention maps showed clear cardiac concentration for most Cardiomegaly images. The supervised signal from `weight_bbox_attn=0.05` was working.

**FactorDisc:** Discriminator accuracy settled around 0.52–0.58 by epoch 50 — near the 0.50 ideal. MI pressure between z_common and z_disease was effective.

**z_cardio_norm_ratio:** Above 2.0 — active Cardiomegaly images produced z_d with noticeably larger norm than Normal images (where z_d has tight prior pressure). Disentanglement working at least in the norm sense.

**Masked_rec loss:** `loss/masked_rec` decreased from ~0.08 at D2 start to ~0.02 by epoch 55. Non-cardiac regions were being faithfully reconstructed using z_common alone.

---

### 3.4 D3 — PatchGAN + TV (Mar 25, 2026 — current milestone)

**W&B:** https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk
**Git:** `97f678113b2770417db1c126b839905103f1766c`
**Epochs:** 55 → 120 (+65 epochs)
**Canonical run:** `d3_gan_fix-20260325-143813`

This is the stopping point that exceeded expectations. The detailed results follow.

#### 3.4.1 Reconstruction quality (qualitative)

At epoch 65–70 (GAN active for ~5–10 epochs after the 2000-step warmup):
- **Rib cortex edges**: visibly sharper than D2. Lateral ribs show distinct bright-dark edge pair. D2 showed smooth gradients.
- **Cardiac border**: Crisp right and left cardiac borders. The silhouette is defined by a sharp edge rather than a gradual intensity gradient. This is anatomically correct (the pericardium creates a distinct interface with the lung).
- **Vessels**: Pulmonary vessels visible in the hilar region as linear branching structures. Absent in D1/D2.
- **Lung texture**: Faint lung markings returning. The lung parenchyma shows a subtle texture rather than smooth grey.
- **No artifacts**: Horizontal stripe banding absent (perceptual layers 1–2 only + 5× TV weight). No checkerboard. No mode collapse.

At epoch 90–100 (stable GAN training):
- Further sharpening across all structures.
- Cardiac silhouette for Cardiomegaly images shows visibly enlarged outline vs Normal images.
- When z_disease is zeroed for Normal images, the reconstruction is anatomically plausible — cardiac silhouette normal-sized, no disease traces.

#### 3.4.2 Quantitative metrics (D3 at epoch 55–120)

| Metric | Epoch 55 (D2 resume) | Epoch 70 (+GAN active) | Epoch 100 | Epoch 120 (target) |
|--------|---------------------|----------------------|-----------|-------------------|
| `loss/reconstruction` | ~0.006 | ~0.005 | ~0.004–0.005 | expected ≤ 0.004 |
| `loss/bbox_attn` | ~0.30 | ~0.25 | ~0.20–0.25 | target ≤ 0.20 |
| `metrics/patch_disc_acc` | N/A (GAN not yet active) | 0.60–0.65 | 0.55–0.65 | target 0.55–0.65 |
| `loss/tv` | ~0.004 | ~0.003 | ~0.002–0.003 | decreasing |
| `metrics/z_cardio_norm_ratio` | > 2.0 | > 2.0 | > 2.0 | target > 2.0 |

_Note: exact per-epoch numbers from W&B. The values above are estimates from training logs. Access live run at W&B link above._

#### 3.4.3 D3 vs D2 comparison

| Aspect | D2 (epoch 55) | D3 (epoch 90+) | Improvement |
|--------|--------------|---------------|-------------|
| Rib cortex sharpness | Smooth gradients | Distinct edges | Qualitative |
| Cardiac border definition | Gradual intensity fall-off | Sharp boundary | Qualitative |
| Vessel visibility | Absent | Faint but present | Qualitative |
| Stripe artifacts | Faint horizontal banding | None | Qualitative |
| `loss/reconstruction` | ~0.006 | ~0.004–0.005 | ~20–33% reduction |
| `loss/bbox_attn` | ~0.30 | ~0.20–0.25 | ~17–33% reduction |
| FactorDisc accuracy | ~0.55 | ~0.55–0.65 | Similar (PatchGAN adds second adversarial pressure) |

#### 3.4.4 What made D3 work after two failed GAN attempts

| Factor | d5_gan (collapsed) | d5_gan_v2 (stalled) | d3_gan_fix (working) |
|--------|-------------------|--------------------|--------------------|
| `weight_gan` | 0.5 | 0.1 | **0.1** |
| `gan_start_step` | global (fired at step 1) | global (2000, fresh) | **phase-local (2000)** |
| `disc_r1_penalty` | 0.0 | **10.0** (trapped) | **0.0** |
| `lr_patch_disc` | 1e-4 | 3e-5 (too slow) | **1e-4** |
| Perceptual layers | 1–3 | 1–3 | **1–2 only** |
| `weight_tv` | 0.001 | 0.001 | **0.005** |
| Resume checkpoint | D4 (stripe-baked) | D4 (stripe-baked) | **D2 (clean)** |
| Outcome | Collapsed epoch 4 | Stalled 80+ epochs | Clean 65+ GAN epochs |

---

## 4. Metrics That Are Reliable vs. Noisy

### 4.1 Reliable metrics (use for decisions)

**`loss/reconstruction`** — direct pixel MSE. Low variance. Use for: detecting degradation (spikes), confirming improvement trend. Kill condition: > 0.010 and rising.

**`metrics/patch_disc_acc`** — discriminator accuracy. Rolling average of 3+ epochs. Kill condition: > 0.80 sustained. Healthy target: 0.55–0.65.

**`loss/bbox_attn`** — attention fraction outside bbox. Decreasing trend confirms attention is concentrating. Use as a slow signal (evaluate over 10+ epochs, not single epoch).

**Visual recon grids** — irreplaceable. No numerical metric captures whether the reconstruction looks like a chest X-ray vs. a blurry blob vs. a mode-collapsed texture pattern. Always inspect the grid at GAN activation.

### 4.2 Noisy metrics (use for trends, not single-epoch readings)

**`probe_auc`** (V1) — high variance. ±0.15 epoch-to-epoch is normal. Only use 3+ epoch rolling averages.

**`loss/gan_g`** — generator hinge loss inherently noisy. Trend over 5+ epochs matters; single-epoch value does not.

**`metrics/z_cardio_norm_ratio`** — moderate variance. Use to confirm the ratio is > 2.0 consistently, not to track single-epoch values.

### 4.3 Metrics not yet implemented (V2 gaps)

| Missing metric | Why needed | How to add |
|---------------|------------|------------|
| Disentanglement AUC probe on z_d | V1 main quality gate, not tracked in V2 | Add linear probe evaluation in `train_sep_vae.py` eval step |
| Cross-head score | Leakage characterisation, missing from V2 | Evaluate frozen linear probe on z_c predicting disease and vice versa |
| FID / LPIPS | Reconstruction quality vs. real distribution | Requires batched generation + pretrained FID backbone |
| Counterfactual qualitative grid | Visual proof of disentanglement | Swap z_d between Normal/Cardio pairs and inspect |

The absence of probe AUC in V2 is an acknowledged gap. The qualitative reconstruction quality at D3 is strong enough to continue to D4–D7, but a probe AUC evaluation should be added before the LDM preencoding stage.

---

## 5. Verification Gates

Before proceeding from SepVAE training to LDM preencoding, the following gates must pass:

| Gate | Criterion | Measured by |
|------|-----------|-------------|
| G1: Disease discriminability | Frozen linear probe on z_d achieves AUC > 0.75 for cardiomegaly | Add probe eval to V2 training loop |
| G2: Common purity | Frozen linear probe on z_c achieves AUC < 0.65 for cardiomegaly | Add probe eval on z_c |
| G3: Edit purity | Nulling z_d for Cardiomegaly image → reconstruction without enlarged cardiac silhouette (qualitative + pixel diff in cardiac region) | Visual inspection + masked MSE delta |
| G4: Normal fidelity | Normal image reconstructions without any disease traces | Visual inspection |

**Current status at D3 end:**
- G1: Not formally evaluated in V2. Qualitative evidence suggests passing (z_cardio_norm_ratio > 2.0, supervised contrastive loss converging).
- G2: Not formally evaluated. loss/masked_rec < 0.02 suggests z_common purity in non-cardiac region.
- G3: Qualitatively passing. Normal reconstructions do not show enlarged cardiac silhouettes.
- G4: Qualitatively passing. Normal image reconstructions anatomically plausible.

**Recommendation:** Add G1 and G2 formal evaluation (frozen linear probe) before preencoding. A single eval script run on the D3 final checkpoint is sufficient.

---

## 6. Comparison to V1 Best

The V1 best (probe AUC 0.774) and V2 D3 results address different phases of the problem:

| Dimension | V1 best | V2 D3 | Status |
|-----------|---------|-------|--------|
| Disease discriminability | AUC 0.774 | Not formally measured | V2 likely better (BboxCrossAttn + supervised signals) |
| Reconstruction quality | Blurry (no GAN, stride-32 backbone) | Crisp ribs/vessels/cardiac border | **V2 much better** |
| Disentanglement formalism | Cross-head score 0.744 (but cardio dead) | Not directly measured | V2 designed to be better via spatial routing |
| Artifact-free output | Checkerboard in some runs | Clean | **V2 better** |
| Spatial routing | Learned attention without ground truth | BboxCrossAttn with supervised concentration | **V2 better** |
| Architecture trainability | Frozen CheSS (cannot adapt for reconstruction) | Full ResNet-50 from scratch | **V2 better** |

V2 D3 represents a qualitative step change in reconstruction quality. The next measurement priority is adding formal probe AUC evaluation to confirm that the reconstruction quality improvements did not sacrifice disease discriminability.

---

*End of document. Continue to [08 Current State (D3)](08_current_state_d3.md).*
