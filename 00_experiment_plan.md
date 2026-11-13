# Comparative Disentanglement Study — Experiment Plan
## `/home-mscluster/mmolefe/Playground/PhD/sepVAE`

**Goal:** Systematically isolate the contribution of each loss component and architectural choice
to disease-factor disentanglement, on synthetic ground-truth datasets where composition quality
is measurable.

**Date drafted:** 2026-03-14
**Branch:** sepVAEIndependet (mirrors the VinBig CXR main project)

---

## 1. Datasets (Tier Hierarchy)

| Tier | Name | Channels | Size | Common factor | Disease-1 | Disease-2 | Comorbid test |
|------|------|----------|------|---------------|-----------|-----------|---------------|
| T1 | **SynDis** | 1 | 64×64 | Gaussian-field background | Blob (centre-left) | Band (bottom) | ✓ |
| T2 | **SpyMNIST** | 1 | 64×64 | Digit class (0-9) | Circle top-left | Square bottom-right | ✓ |
| T3 | **ColoredMNIST+Boxes** | 3 | 64×64 | Digit identity + digit colour | Red box top-left | Blue box bottom-right | ✓ |
| T4 | **3DShapes-Disease** | 3 | 64×64 | Wall/floor hue, object size | Object-1 shape present | Object-2 colour present | ✓ |

**Protocol:** Develop and tune on T1. Transfer (no retuning) to T2, T3 without modification.
T4 is used only for MIG / DCI_d metrics (ground-truth factor labels available).

### Comorbid split (all tiers)
- Train: Normal + Disease-1-only + Disease-2-only (no comorbid)
- Val:   Normal + Disease-1-only + Disease-2-only + comorbid (150 each)
- Test:  Normal + Disease-1-only + Disease-2-only + comorbid (150 each)

### Paired protocol (Abid / Severson experiments)
Background set **B** = Normal examples.
Salient set **S** = Disease-1 or Disease-2 examples.
Pairs are **matched by global seed** (same background blob pattern, different disease label) to
eliminate confounding background variation from the contrastive signal.
At runtime `SynDisDataset(…, global_seed=42)` seeds Background and Disease images identically
so seed `42 * 1_000_000 + i` appears in both Normal and Disease-1; the pairing is positional.

---

## 2. Method Ladder — 7 Experiments (E0–E7)

All experiments share the same encoder/decoder architecture (A1 CNN by default).
The only differences are the loss stack.

### E0 — Vanilla VAE
**Paper ref:** Kingma & Welling (2013)
**Loss:** `L = MSE(x, x̂) + β·KL(q(z|x) ∥ p(z))`
**β = 1**; single latent z of dimension 8 (=4+2+2 to match E2–E7 total).
**Expected behaviour:** z encodes everything (common + both diseases) entangled.
**Composition test:** impossible — no factorised heads.
**Role:** reconstruction quality baseline.

### E1 — β-VAE
**Paper ref:** Higgins et al. (2017)
**Loss:** Same as E0 but β ∈ {2, 4, 6, 8, 10} (sweep).
**Expected behaviour:** higher β → more axis-aligned but blurrier reconstructions.
**Fixed run for comparisons:** β = 4.
**Role:** disentanglement-without-labels baseline.

### E2 — Split-Head VAE (ablation baseline)
**No paper.** Our own ablation.
**Architecture:** z = [z_c (4-dim), z_s1 (2-dim), z_s2 (2-dim)]; three independent encoder heads.
**Loss:** `L = MSE + KL(z_c) + KL(z_s1) + KL(z_s2)`
No cross-head loss, no nulling, no contrastive.
**Expected behaviour:** heads trained independently will all encode everything; shows baseline
entanglement with split structure alone.
**Role:** isolates the effect of cross-head losses in E3–E7.

### E3 — Contrastive VAE / cVAE (Abid)
**Paper:** Abid et al., *Contrastive Variational Autoencoder Enhances Salient Features* (2019)
**Loss:**
```
L = MSE(x, x̂)
  + KL(q(z_c|x) ∥ p(z_c))
  + KL(q(z_s|x) ∥ p(z_s))
  − λ · KL(q(z_s|x_B) ∥ p(z_s))   ← nulling on background z_s
```
Background set B = Normal images.
**Implementation note:** Abid has one salient head z_s shared across both diseases.
We run two variants: E3a (one z_s head, all disease pooled) and E3b (two separate z_s1, z_s2
heads with independent nulling).
**Expected behaviour:** z_s will be pushed away from background variance; some disease separation.

### E4 — CLVM (Severson)
**Paper:** Severson et al., *Unsupervised Learning with Contrastive Latent Variable Models* (2019)
**Loss:**
```
L = MSE(x, x̂)
  + KL(q(z_c|x) ∥ p(z_c))
  + KL(q(z_s|x) ∥ p(z_s))
  + α · [KL(q(z_c|x_B) ∥ q(z_c|x_S)) + KL(q(z_c|x_S) ∥ q(z_c|x_B))]
  − λ · KL(q(z_s|x_B) ∥ p(z_s))
```
Requires explicit (B, S) pairs.
**Expected behaviour:** z_c marginals for B and S aligned → z_s absorbs all disease variance.

### E5 — MM-cLVM (Weinberger)
**Paper:** Weinberger et al., *Moment Matching Deep Contrastive Latent Variable Models* (2023)
**Loss:** As E4 but replaces KL-alignment with moment matching (mean + covariance):
```
MM(z_c; B, S) = ‖μ_c^B − μ_c^S‖² + ‖Σ_c^B − Σ_c^S‖_F²
```
**Expected behaviour:** more stable alignment than KL-based; may preserve z_c expressivity better.

### E6 — Joint Disentanglement (Zou)
**Paper:** Zou et al., *Joint Disentanglement of Labels and Their Features with VAE* (2022)
**Loss:**
```
L = MSE(x, x̂)
  + KL(q(z_c|x) ∥ p(z_c))
  + KL(q(z_s|x) ∥ p(z_s|y))        ← label-conditioned prior
  + γ · CE(p(y|z_s), y)             ← classification on z_s
  + δ · MI(z_c, z_s)                ← mutual information penalty
```
Key ingredient NOT in our codebase: `p(z_s|y)` = N(μ_y, I) where μ_y is learned per label.
For y=normal: μ_y = 0 (standard Gaussian); for y=disease: μ_y = learnable.
**Role:** tests whether label-conditioned priors outperform our nulling loss.

### E7 — SepVAE (Ours)
**Full loss stack:**
```
L = MSE
  + KL_z_c + KL_z_d1 + KL_z_d2
  + λ_null  · ‖μ_d1[normal]‖²  + ‖μ_d2[normal]‖²      (R3 nulling)
  + λ_orth  · |cos(μ_d1, μ_d2)|                         (R3 orthogonality)
  + λ_cont  · contrastive_margin_loss(z_d1, z_d2)        (R5a)
  + λ_adv   · cross_head_adversarial(z_d1→z_d2, z_d2→z_d1)  (R5b)
  + λ_clf   · BCE(clf_d1(GAP(z_d1)), has_d1)             (R11)
  + λ_clf   · BCE(clf_d2(GAP(z_d2)), has_d2)             (R11)
```
With spatial label-attention encoder heads (R7) as variant E7+attn.
**Role:** our proposed method. Should outperform E0–E6 on spec_ratio and comp_score.

---

## 3. Architecture Ladder (A0–A3)

Run each architecture with E7 (full loss stack) on T1 (SynDis), then on T2 (SpyMNIST).

| ID | Name | Encoder | Decoder | Params (approx) |
|----|------|---------|---------|-----------------|
| A0 | MLP | Flatten → Linear(4096,512) → ReLU → Linear(512,256) → ReLU → 3× Linear(256, 2ch) | 3× Linear(2ch,256) → ReLU → Linear(256,512) → ReLU → Linear(512,4096) | ~4 M |
| A1 | CNN | 3× Conv2d(stride=2) → 3 heads | 3× ConvTranspose2d → Tanh | ~1.5 M |
| A2 | CNN+SelfAttn | A1 + 1 MHSA layer before heads | A1 decoder | ~2 M |
| A3 | CNN+LabelAttn | A1 + LabelAttentionHead (R7 analog) per disease head | A1 decoder | ~2.2 M |

**Primary comparison:** A1 vs A3 with E7 loss to isolate spatial attention gain.

---

## 4. Unified Metric Protocol

Compute after every run on the **test** split. All metrics logged to W&B and written to
`results/{exp_name}/metrics.json`.

### Reconstruction quality
| Metric | Symbol | Notes |
|--------|--------|-------|
| MSE | `rec_mse` | Mean over test set |
| KL per dim | `kl_per_dim` | total KL / (8 × H × W) for spatial, 8 for MLP |

### Encoder discriminability
| Metric | Symbol | Notes |
|--------|--------|-------|
| AUC d1→d1 | `auc_d1_d1` | LR probe on GAP(z_d1), target = has_disease1 |
| AUC d2→d2 | `auc_d2_d2` | LR probe on GAP(z_d2), target = has_disease2 |
| AUC d1→d2 | `auc_d1_d2` | LR probe on GAP(z_d1), target = has_disease2 (leakage) |
| AUC d2→d1 | `auc_d2_d1` | LR probe on GAP(z_d2), target = has_disease1 (leakage) |
| Specificity ratio d1 | `spec_ratio_d1` | auc_d1_d1 / auc_d2_d1 (target > 2.0) |
| Specificity ratio d2 | `spec_ratio_d2` | auc_d2_d2 / auc_d1_d2 (target > 2.0) |
| AUC common→d1 | `auc_c_d1` | LR probe on GAP(z_c), target = has_disease1 (should be ~0.5) |
| AUC common→d2 | `auc_c_d2` | LR probe on GAP(z_c), target = has_disease2 (should be ~0.5) |

### Disentanglement (T4 only — ground-truth factors available)
| Metric | Symbol | Notes |
|--------|--------|-------|
| MIG | `mig` | Mutual Information Gap (Chen et al. 2018) |
| DCI Disentanglement | `dci_d` | Eastwood & Williams 2018 |

### Composition quality (all tiers)
Compose x̂_composed = Decoder(z_c ~ Normal, z_d1 ~ Disease-1, z_d2 ~ Disease-2)
where z_c, z_d1, z_d2 come from **different** samples (cross-composition).

| Metric | Symbol | Formula |
|--------|--------|---------|
| Blob delta | `comp_blob_delta` | mean(x̂_composed[BLOB_ROWS, BLOB_COLS]) − mean(x̂_normal[same]) |
| Stripe delta | `comp_stripe_delta` | mean(x̂_composed[STRIPE_ROWS, STRIPE_COLS]) − mean(x̂_normal[same]) |
| Composition score | `comp_score` | (blob_delta + stripe_delta) / 2 (higher = better signal localisation) |
| Composition AUC | `comp_auc` | Train LR classifier on {composed, normal}; AUC (higher = more realistic) |

**Gate:** proceed to cross-dataset transfer only if `spec_ratio_d1 > 2.0` AND `spec_ratio_d2 > 2.0`.

---

## 5. Code Structure

```
sepVAE/
├── 00_experiment_plan.md          ← this file
├── datasets/
│   ├── __init__.py
│   ├── SynDis.py                  ← T1 (copied from superdiff-ldm/datasets/)
│   ├── spymnist.py                ← T2 (to implement)
│   ├── colored_mnist_boxes.py     ← T3 (to implement)
│   └── shapes3d_disease.py        ← T4 (to implement; wraps tensorflow_datasets)
│
├── models/
│   ├── __init__.py
│   ├── shared.py                  ← ConvBlock, ConvTransposeBlock, ResBlock, MHSA, LabelAttnHead
│   ├── mlp_vae.py                 ← A0 MLP encoder/decoder
│   ├── cnn_vae.py                 ← A1 CNN encoder/decoder (primary)
│   ├── cnn_attn_vae.py            ← A2 CNN+SelfAttn, A3 CNN+LabelAttn
│   └── disease_classifier.py      ← R11 small per-head classifier (shared across experiments)
│
├── losses/
│   ├── __init__.py
│   ├── base.py                    ← kl_loss, mse_loss, reparameterize
│   ├── nulling.py                 ← nulling_loss, orthogonality_loss (R3)
│   ├── contrastive.py             ← margin_contrastive_loss (R5a)
│   ├── cross_adversarial.py       ← CrossHeadDiscriminator, cross_adv_loss (R5b)
│   ├── moment_matching.py         ← mm_alignment_loss (E5 Weinberger)
│   ├── conditional_prior.py       ← LabelConditionedPrior, kl_to_label_prior (E6 Zou)
│   └── disease_clf_loss.py        ← stable_bce, clf_bce_loss (R11)
│
├── train/
│   ├── __init__.py
│   ├── train_syndis_e7_ref.py     ← reference impl (copied from superdiff-ldm/run/)
│   ├── trainer_base.py            ← shared training loop, eval, save_recon_grid
│   ├── e0_vanilla.py              ← E0: single-head β=1
│   ├── e1_beta_vae.py             ← E1: β sweep
│   ├── e2_split_head.py           ← E2: split-head ablation
│   ├── e3_cvae_abid.py            ← E3: Abid cVAE
│   ├── e4_clvm_severson.py        ← E4: Severson CLVM
│   ├── e5_mmclvm_weinberger.py    ← E5: Weinberger MM-cLVM
│   ├── e6_zou.py                  ← E6: Zou joint disentanglement
│   └── e7_sepvae.py               ← E7: our full SepVAE
│
├── launchers/
│   ├── run_all_syndis.sh           ← run E0–E7 on T1 sequentially
│   ├── run_all_spymnist.sh         ← run E0–E7 on T2
│   └── run_tier3_colored.sh        ← run E7 variants on T3
│
├── eval/
│   ├── __init__.py
│   ├── metrics.py                 ← all metric functions (auc_probe, spec_ratio, mig, dci_d, comp_*)
│   └── report.py                  ← load metrics.json files → pandas DataFrame → latex table
│
├── results/
│   └── (auto-created per run: results/{exp_name}/metrics.json, recon_grid.png, comp_grid.png)
│
└── docs/
    ├── 01_vanilla_vae.md
    ├── 02_beta_vae.md
    ├── 03_split_head.md
    ├── 04_abid_cvae.md
    ├── 05_severson_clvm.md
    ├── 06_weinberger_mmclvm.md
    ├── 07_zou_joint.md
    └── 08_sepvae_ours.md
```

---

## 6. Design Decisions (Recorded)

### 6.1 Paired protocol
Abid and Severson require (Background, Salient) pairs. We use **positional matching by seed**:
`SynDisDataset(global_seed=42)` gives Normal[i] same background pattern as Disease-1[i] because
both use `base + i` within their class-specific seed ranges. This is not truly hard-paired but
provides soft matching; SpyMNIST T2 will also support this by construction.

### 6.2 Architecture for SepVAE (E7) on SynDis
A1 (plain CNN) is the default for all E0–E7 comparison runs to keep architecture identical.
A0–A3 is a separate sweep run only with E7.

### 6.3 β sweep range (E1)
Run β ∈ {1, 2, 4, 6, 8, 10}. Use β = 4 as the fixed single-β comparison point in Method Ladder tables.

### 6.4 SepVAE spatial latent size
SynDis 64×64 images → after 3× stride-2 convolutions → spatial latent at 8×8 (same as
superdiff-ldm). MLP variant A0 uses GAP → fixed-dim vector.

### 6.5 Cross-dataset transfer
After T1 shows spec_ratio > 2.0 for E7 → copy weights (or retrain from scratch) on T2.
No hyperparameter retuning between tiers.

---

## 7. Execution Order

```
Step 1: Implement datasets/spymnist.py, datasets/colored_mnist_boxes.py
Step 2: Implement models/shared.py, models/cnn_vae.py
Step 3: Implement losses/base.py, losses/nulling.py
Step 4: Implement train/trainer_base.py
Step 5: Implement train/e0_vanilla.py → verify runs on T1
Step 6: Implement train/e2_split_head.py → verify ablation
Step 7: Implement train/e3_cvae_abid.py, train/e4_clvm_severson.py (both need nulling loss)
Step 8: Implement train/e5_mmclvm_weinberger.py, train/e6_zou.py (new loss modules)
Step 9: Implement train/e7_sepvae.py (full stack, adapts train_syndis_e7_ref.py)
Step 10: Wire losses into launchers, run comparison sweep
Step 11: eval/report.py → generate comparison table
```

---

## 8. Expected Results Table (Hypothesis)

| Experiment | rec_mse ↓ | spec_ratio_d1 ↑ | spec_ratio_d2 ↑ | comp_score ↑ |
|------------|-----------|-----------------|-----------------|--------------|
| E0 Vanilla | low | ~1.0 | ~1.0 | ~0.0 |
| E1 β-VAE (β=4) | medium | ~1.1 | ~1.1 | ~0.0 |
| E2 Split-head | low | ~1.2 | ~1.2 | ~0.05 |
| E3 cVAE Abid | medium | ~1.4 | ~1.4 | ~0.08 |
| E4 CLVM Severson | medium | ~1.6 | ~1.6 | ~0.10 |
| E5 MM-cLVM | medium | ~1.7 | ~1.7 | ~0.12 |
| E6 Zou Joint | medium | ~1.8 | ~1.8 | ~0.14 |
| **E7 SepVAE** | **medium** | **>2.0** | **>2.0** | **>0.20** |

These are falsifiable predictions. Deviations from this ranking are scientifically interesting.
