# SepVAE Research Discussion: Empirical Findings, Open Questions, and Composition Theory

**Date:** 2026-03-12
**Branch:** `sepVAEIndependet`
**Project:** `sepvae-disentangle` (W&B)
**Authors:** mmolefe

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Empirical Results: Cross-Run Summary](#3-empirical-results-cross-run-summary)
4. [Root Cause Diagnoses](#4-root-cause-diagnoses)
5. [Cross-Run Patterns](#5-cross-run-patterns)
6. [Implemented Recommendations (R1–R5)](#6-implemented-recommendations-r1r5)
7. [Research Assumptions and Their Risks](#7-research-assumptions-and-their-risks)
8. [What Exactly Is Being Composed?](#8-what-exactly-is-being-composed)
9. [Four Verifiability Criteria and Current Gaps](#9-four-verifiability-criteria-and-current-gaps)
10. [Four Proposed Additions](#10-four-proposed-additions)
11. [Revised Research Hypothesis](#11-revised-research-hypothesis)
12. [Run Commands](#12-run-commands)

---

## 1. Background and Motivation

The broader goal of this project is to enable **compositional multi-pathology generation** in chest X-rays. Concretely: given a model that has learned what cardiomegaly looks like and a separate model that has learned what pleural effusion looks like, can we synthesise a realistic image containing *both* without ever having jointly trained on comorbid data?

This is clinically valuable because:
- Comorbid cases are rare and expensive to collect with clean labels
- Controlled synthesis allows studying pathology interactions
- Disentangled representations could allow attribute-level editing for explainability

The SepVAE is the first building block: a variational autoencoder that produces **factorized latent representations** — one shared head $z_{\text{common}}$ capturing anatomy and acquisition, and one head per disease ($z_{\text{cardio}}$, $z_{\text{effusion}}$) capturing only disease-specific variation.

---

## 2. Architecture Overview

```
 Input x (512×512 CXR)
        │
        ▼
 ┌──────────────┐
 │ CheSS encoder│  (frozen ResNet-50, pretrained on CXR)
 │ (backbone)   │
 └──────┬───────┘
        │  feature map f ∈ ℝ^{H×W×C}
        ▼
 ┌──────────────────────────────────────────────┐
 │              Encoder heads                    │
 │                                               │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
 │  │ μ_c, σ_c     │  │ μ_d1, σ_d1  │  │ μ_d2, σ_d2  │  │
 │  │ (z_common)   │  │ (z_cardio)  │  │ (z_effusion)│  │
 │  │ 4 channels   │  │ 2 channels  │  │ 2 channels  │  │
 │  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  │
 └─────────┼─────────────────┼────────────────┼──────────┘
           │                 │                │
           └─────────────────┴────────────────┘
                             │
                    z = cat[z_common, z_cardio, z_effusion]
                             │
                             ▼
                    ┌─────────────────┐
                    │    Decoder      │  (shared conv upsampling)
                    └────────┬────────┘
                             │
                             ▼
                        x̂ (reconstruction)
```

**Disease label routing:**
The triplet dataloader provides `(x_norm, x_disease1[effusion], x_disease2[cardio])` per batch.
At forward pass, concatenated batch has labels $y \in \{0, 1, 2\}$:
- $y=0$ (normal): $z_{\text{cardio}}$ and $z_{\text{effusion}}$ are both **inactive** — penalised toward $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$
- $y=1$ (effusion): $z_{\text{effusion}}$ is **active** — free ELBO; $z_{\text{cardio}}$ is inactive
- $y=2$ (cardio): $z_{\text{cardio}}$ is **active** — free ELBO; $z_{\text{effusion}}$ is inactive

---

## 3. Empirical Results: Cross-Run Summary

### 3.1 Run Inventory (February 2026)

| Run | Wandb ID | Methodology | Epochs | Status |
|-----|----------|-------------|--------|--------|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | disentangle-A | 192/200 | ~complete (old script, no probe AUC) |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | disentangle-B | 1 | **CRASHED** (step 0) |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | disentangle-C | 84 | Stopped — resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | disentangle-D | 101 | Stopped — resumed as E |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | disentangle-E | 200 | **NaN crash at epoch 200** |
| `sepvae_inactivity_driven-20260220-062635` | `99s5jhqe` | inactivity-F | 0 | **CRASHED** (transient GPU) |
| `sepvae_inactivity_driven-20260220-085943` | `9lj20so0` | inactivity-G | 100 | Completed |
| `sepvae_independence_driven-20260220-062742` | `i9cevm1x` | independence-H | 0 | **CRASHED** (transient GPU) |
| `sepvae_independence_driven-20260220-085943` | `41nce8qq` | independence-I | 100 | Completed |

> Runs C, D, E form one continuous training chain across three SLURM jobs.
> Runs F and H crashed during model init from transient GPU conflict; identical configs G and I succeeded 23 minutes later.

### 3.2 Hyperparameter Matrix

| Parameter | disentangle-C/D/E | inactivity-G | independence-I |
|-----------|-------------------|--------------|----------------|
| `free_bits` | 1 | 1 | **2** |
| `sigma_inactive` | 0.1 | **0.05** | 0.1 |
| $\text{KL}_{\text{inactive}}$ (est.) | ~1.8 nats | ~2.5 nats | ~1.8 nats |
| $\text{KL}_{\text{inactive}} > \text{free\_bits}$? | yes | yes | **NO (1.8 < 2.0)** |
| `weight_null` | 0.01 | **0.05** | 0.01 |
| `weight_orthogonality` | 0.03 | 0.05 | **0.1** |
| `weight_mi` | 0.003 | 0.005 | **0.01** |
| `weight_perceptual` | 0.03 | 0.05 | 0.05 |
| `use_fpn` | yes | no | no |
| `batch_size` | 6 | 10 | 10 |
| `kl_warmup_epochs` | 30 | 10 | 10 |
| `half_precision` | fp32 | bf16 | bf16 |

KL of an inactive head estimated analytically:

$$\text{KL}_{\text{inactive}} = \frac{1}{2}\left(\sigma_{\text{inactive}}^2 - 1 - \log \sigma_{\text{inactive}}^2\right) \quad \text{(with } \mu = 0\text{)}$$

| $\sigma_{\text{inactive}}$ | $\text{KL}_{\text{inactive}}$ (nats) |
|---------------------------|--------------------------------------|
| 0.20 | ~1.1 |
| 0.10 | ~1.8 |
| 0.05 | ~2.5 |

### 3.3 Performance Table

| Run | Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Probe AUC (mean) | Cross-head score ↓ |
|-----|-------|--------------------|----------------------|------------------|--------------------|
| disentangle-A | 192 | N/A (old script) | N/A | N/A | N/A |
| disentangle-C/D/E — **best** | ~185 | **0.776** | **0.772** | **0.774** | 0.914 |
| disentangle-C/D/E — ep 101 | 101 | 0.758 | 0.700 | 0.729 | 0.952 |
| disentangle-C/D/E — last valid | 199 | 0.731 | 0.744 | 0.737 | 0.935 |
| inactivity-G — **best** | ~50 | 0.639 | 0.627 | 0.650 | **0.744** |
| inactivity-G — final | 100 | 0.476 | 0.767 | 0.622 | **0.809** |
| independence-I — early peak | ~25 | 0.762 | 0.737 | 0.750 | 0.820 |
| independence-I — final | 100 | 0.532 | 0.812 | 0.672 | 0.871 |

**Metric definitions:**
- **Probe AUC**: linear-probe classifier trained on disease-head latents $z_{\text{cardio}}$ or $z_{\text{effusion}}$; measures how much disease-specific information the head carries. Higher is better (closer to 1.0).
- **Cross-head score**: mean ratio of inactive-head energy to active-head energy across disease samples. 1.0 = total cross-leakage (heads are interchangeable). 0.5 = ideal (no cross-information). **Lower is better.**

---

## 4. Root Cause Diagnoses

### 4.1 Independence-I: Free-bits dead zone

The central failure mechanism:

$$\text{KL}_{\text{inactive}}(\sigma=0.1) \approx 1.8 \text{ nats} < \text{free\_bits} = 2.0$$

The free-bits threshold clips the KL gradient to zero for any channel with $\text{KL} < \lambda_{\text{fb}}$. Because **all disease channels** are held near $\sigma_{\text{inactive}} = 0.1$, they all land below the threshold permanently.

```
KL_channel ≈ 1.8 nats
────────────────────────────────────────────────────
          ◄── dead zone ──►
    0 ──────────────────── 2.0 ──────────── ∞
                           ↑ free_bits = 2.0
          gradient = 0 here
```

The free-bits mechanism was designed to prevent posterior collapse in the *common* head by protecting low-KL channels from being over-penalised. But it applies globally and inadvertently kills disease-head gradients.

**Fix (R1):** Set `free_bits = 0.0` (removed entirely) with `sigma_inactive = 0.05` to achieve tighter inactive priors instead.

### 4.2 Disentangle-E: NaN explosion at epoch 200

Without a learning rate schedule, accumulated small gradient errors over 200 epochs eventually produce a catastrophic update from one unlucky batch. Full fp32 precision increases the risk since there is no implicit gradient scaling. The total loss was stable (0.004–0.006) through epoch 199, then all terms went `nan` simultaneously.

**Fix (R3):** Cosine LR decay from `lr_decay_epochs` onward:

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot (t - t_{\text{decay}})}{T - t_{\text{decay}}}\right)\right)$$

with $\eta_{\min} = 0.1 \cdot \eta_0$.

### 4.3 Inactivity-G: Cardiomegaly head collapse under heavy nulling

Strong `weight_null = 0.05` drives $\mu_{\text{cardio}} \to 0$ faster than the reconstruction loss can maintain the cardiomegaly signal, specifically because:
- Cardiomegaly is a distributed, subtle feature (enlarged cardiac silhouette) vs. the localised bright blobs of effusion
- CheSS backbone, trained primarily on normal anatomy, represents cardiomegaly as a weaker feature
- The nulling loss is symmetric — it cannot distinguish "this head should be inactive here" from "this head should be active here"

Final result: `probe_auc/cardiomegaly = 0.476` (below chance).

**Fix (R2):** Balance `weight_null = 0.02`, `weight_orthogonality = 0.02`, `sigma_inactive = 0.05`. Gentler pressure that does not overwhelm the cardiomegaly signal.

### 4.4 Universal cross-head leakage

Across **all runs**, cross-head score never falls below 0.74. Even the best run ends at 0.809. The current loss portfolio — nulling + orthogonality + MI discriminator — enforces marginal statistical independence but not **conditional independence**. The heads can share information about the other disease without violating any of these constraints as long as their marginal distributions are uncorrelated.

```
What orthogonality enforces:
  𝔼[z_cardio] ⊥ 𝔼[z_effusion]   (prototype cosine penalty)
  Corr(z_cardio_i, z_effusion_j) ≈ 0   (Barlow-style)

What it does NOT enforce:
  z_cardio ⊥ y_effusion | z_common   (conditional independence of heads given anatomy)
```

**Fix (R5a, R5b):** Paired contrastive loss and cross-head adversarial discriminators — see Section 6.

---

## 5. Cross-Run Patterns

**Pattern 1: `free_bits` / `sigma_inactive` conflict is a hard failure mode**
Any configuration where $\text{KL}_{\text{inactive}} < \text{free\_bits}$ results in disease heads permanently in the gradient dead zone. All such runs fail (probe AUC near chance). This is not a soft tradeoff — it is a binary failure condition that must be checked before any other hyperparameter analysis.

**Pattern 2: Cross-head leakage is a universal unsolved problem**
Cross-head score never falls below 0.74 across any run at any epoch. The best leakage suppression (inactivity-G, 0.744) comes at the cost of cardiomegaly head collapse. The current regularisation set does not have a mechanism that directly enforces conditional independence between disease heads.

**Pattern 3: Cardiomegaly head is consistently harder to train**
Across all runs, cardiomegaly probe AUC is lower, more volatile, and more susceptible to collapse than effusion AUC. Likely causes:
- Cardiomegaly is a subtle, globally distributed change; effusion is a local, high-contrast finding
- After `exclude_cross_disease_overlap=true`, effective cardiomegaly examples may be fewer
- CheSS backbone features do not explicitly separate cardiac silhouette from lung fields

**Pattern 4: Probe AUC is noisy — rolling averages required**
The same checkpoint evaluated at consecutive epochs shows probe AUC ranging from 0.53 to 0.77. The 600-sample probe eval has high variance. Single-epoch readings are unreliable; a 3-evaluation rolling average should be the reporting standard.

**Pattern 5: Long training improves peak but destabilises without LR decay**
The 200-epoch chain achieves the highest peak probe AUC (0.774) but ends in NaN. LR decay was absent in all runs. The combination of long training + no decay + no gradient clip tightening creates a slow-moving instability that eventually triggers catastrophic failure.

---

## 6. Implemented Recommendations (R1–R5)

| ID | Change | Rationale | Status |
|----|--------|-----------|--------|
| **R1** | `free_bits=0.0` | Removes gradient dead zone for disease heads | Implemented |
| **R2** | `sigma_inactive=0.05`, `weight_null=0.02`, `weight_ortho=0.02` | Balanced inactivity without cardiomegaly collapse | Implemented |
| **R3** | Cosine LR decay from `lr_decay_epochs=60` | Prevents NaN at long training; preserves peaks | Implemented |
| **R4** | `weight_perceptual=0.05` | Spatial fidelity; prevents blurry reconstructions | Implemented |
| **R5a** | `--use_contrastive`: paired contrastive loss | Structural separation: separate-push + blind-pull | Implemented |
| **R5b** | `--use_cross_adv`: cross-head adversarial MLPs | Conditional independence enforcement | Implemented |

### R5a: Paired Contrastive Loss (structural)

For disease head $k$ with active class $c_k$, let $\bar{z}_k^{(c)}$ denote the L2-normalised centroid over GAP-pooled latents for class $c$:

$$\mathcal{L}_{\text{sep-push}}^{(k)} = \max\left(0,\ \text{margin} - \left(1 - \bar{z}_k^{(c_k)} \cdot \bar{z}_k^{(\text{norm})}\right)\right)$$

$$\mathcal{L}_{\text{blind-pull}}^{(k)} = 1 - \bar{z}_k^{(c_{\text{other}})} \cdot \bar{z}_k^{(\text{norm})}$$

$$\mathcal{L}_{\text{contrastive}} = \mathcal{L}_{\text{sep-push}}^{(\text{cardio})} + \mathcal{L}_{\text{sep-push}}^{(\text{effusion})} + \mathcal{L}_{\text{blind-pull}}^{(\text{cardio})} + \mathcal{L}_{\text{blind-pull}}^{(\text{effusion})}$$

The blind-pull term is the critical addition: it directly penalises the effusion head for encoding cardiomegaly information (and vice versa), forcing the cross-head centroid to collapse onto the normal centroid.

### R5b: Cross-Head Adversarial Discriminators

Two 3-layer MLPs with leaky-ReLU:
- $D_{c \to e}$: predicts `is_cardiomegaly` from $z_{\text{effusion}}$
- $D_{e \to c}$: predicts `is_effusion` from $z_{\text{cardio}}$

**Discriminator loss** (trained to classify correctly on disease samples):

$$\mathcal{L}_{\text{disc}} = \text{BCE}(D_{c \to e}(z_{\text{effusion}}), y_{\text{cardio}}) + \text{BCE}(D_{e \to c}(z_{\text{cardio}}), y_{\text{effusion}})$$

**VAE adversarial penalty** (trained to confuse discriminators):

$$\mathcal{L}_{\text{cross-adv}} = \mathbb{E}\left[ D_{c \to e}(z_{\text{effusion}})^2 + D_{e \to c}(z_{\text{cardio}})^2 \right] \cdot \mathbf{1}[\text{disease sample}]$$

Minimising $\mathcal{L}_{\text{cross-adv}}$ drives discriminator outputs toward $0.5$ (maximum uncertainty), i.e., no cross-head predictability.

---

## 7. Research Assumptions and Their Risks

This section documents the assumptions required for the SepVAE + composition pipeline to work, and the evidence for and against each.

### Assumption A1: Disease factors are approximately independent

$$z_{\text{cardio}} \perp z_{\text{effusion}}$$

or at least sufficiently separable for composition.

**Why it may not hold:** In VinBigData, cardiomegaly and pleural effusion co-occur via shared pathology (e.g., heart failure causes both), shared appearance patterns (enlarged cardiac silhouette pushes against the pleural space), projection effects, and dataset sampling bias.

**Mitigation implemented:** `exclude_cross_disease_overlap=true` removes all co-occurring patients from the triplet dataset, so the VAE never sees both simultaneously. However, this creates a distribution mismatch at inference when we want to synthesise a comorbid image.

**Evidence from runs:** Cross-head score consistently above 0.74, suggesting the representations are not fully independent even after training.

### Assumption A2: Radiologist labels correspond to clean generative factors

Cardiomegaly is not a primitive visual atom. It is a high-level clinical finding whose appearance varies with patient position, exposure settings, AP vs PA view, body habitus, cardiac rotation, and co-morbid findings. The model may place some cardiomegaly-relevant variation into $z_{\text{common}}$ simply because it looks like "normal anatomy variation" to the backbone.

**Evidence from runs:** `probe_auc/cardiomegaly` is consistently lower and more volatile than `probe_auc/effusion`. The backbone (CheSS) was trained on CXR but not with a supervision signal that separates cardiac pathology from normal cardiac anatomy.

**Mitigation:** This is partially structural. The cross-head adversarial discriminator (R5b) attacks the *symptom* (cross-head leakage) rather than the cause. A longer-term fix would be to fine-tune the backbone with a cardiac-specific contrastive objective, or use a dedicated cardiomegaly-supervised feature extractor.

### Assumption A3: Factorised representation implies factorised score fields

Even if the SepVAE gives clean $z_{\text{cardio}}$ and $z_{\text{effusion}}$, it does not automatically follow that separately trained LDMs over those factors will combine cleanly by score addition.

Score addition works as a product of experts:

$$p(z \mid c, e) \propto \frac{p(z \mid c) \cdot p(z \mid e)}{p(z)}$$

This identity holds under the conditional independence assumption $p(z \mid c, e) = \frac{p(z \mid c) p(z \mid e)}{p(z)}$, which requires:

$$p(z \mid c) \perp p(z \mid e) \mid z_{\text{common}}$$

If each disease LDM is trained on the *full* z (including $z_{\text{common}}$), they will both encode anatomical structure, and naïvely adding their scores double-counts the shared content:

$$\nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) \approx \nabla_z \log p(z \mid c, e) + \nabla_z \log p(z)$$

The extra $\nabla_z \log p(z)$ term over-reinforces common structure, producing anatomically over-sharpened or physically inconsistent compositions.

**This is the central mathematical risk of the project.** Section 8 addresses the composition strategy that avoids it.

---

## 8. What Exactly Is Being Composed?

The SepVAE is the representation-learning step. The downstream composition pipeline is not yet implemented, and the strategy chosen determines what the SepVAE must provide. There are three candidate strategies with different mathematical properties.

### Strategy A — Sub-block conditional LDMs (recommended)

The most principled design, given the current architecture:

```
z = [ z_common (4ch) | z_cardio (2ch) | z_effusion (2ch) ]
      └───────────────┘  └─────────────┘  └──────────────┘
         base LDM            LDM_cardio        LDM_effusion
                              ↑ conditioned      ↑ conditioned
                              on z_common        on z_common
```

**Factorised generative process:**

$$p(z) = p(z_{\text{common}}) \cdot p(z_{\text{cardio}} \mid z_{\text{common}}) \cdot p(z_{\text{effusion}} \mid z_{\text{common}})$$

**Inference (both diseases simultaneously):**

1. Sample $z_{\text{common}} \sim \text{LDM}_{\text{common}}$
2. Sample $z_{\text{cardio}} \sim \text{LDM}_{\text{cardio}}(z_{\text{common}})$
3. Sample $z_{\text{effusion}} \sim \text{LDM}_{\text{effusion}}(z_{\text{common}})$
4. Decode $\hat{x} = \text{Dec}(z_{\text{common}}, z_{\text{cardio}}, z_{\text{effusion}})$

```
Strategy A — Inference diagram:

    [LDM_common]
         │
    z_common ──────────────────────────┐
         │                             │
         ├──→ [LDM_cardio(z_common)] → z_cardio    │
         │                             │
         └──→ [LDM_effusion(z_common)] → z_effusion │
                                        │
              ┌─────────────────────────┘
              ↓
     Decoder(z_common, z_cardio, z_effusion)
              │
              ↓
       Composed CXR (cardio + effusion)
```

**Why this avoids the double-counting problem:** Each LDM operates on a *different* subspace. There is no score addition; the composition is a sequential ancestral sample from the factorised joint. The conditional independence assumption $z_{\text{cardio}} \perp z_{\text{effusion}} \mid z_{\text{common}}$ is what the SepVAE training must enforce — and this is exactly what R5a and R5b are designed to provide.

**What the SepVAE must provide for this to work:**

$$z_{\text{cardio}} \perp\!\!\!\perp z_{\text{effusion}} \mid z_{\text{common}}$$

The cross-head adversarial discriminator (R5b) is the direct empirical test of this: if $D_{c \to e}(z_{\text{effusion}}) \approx 0.5$, then $z_{\text{effusion}}$ contains no cardiomegaly information beyond what is in $z_{\text{common}}$.

---

### Strategy B — Full-z LDMs with score composition (product of experts)

Train two full-z conditional LDMs and one unconditional LDM, then compose scores:

$$\nabla_z \log p(z \mid c, e) \approx \nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) - \nabla_z \log p(z)$$

```
Strategy B — Score composition:

  ∇ log p(z|cardio) ──┐
                       ├──(+)──→ ∇ log p(z|both)
  ∇ log p(z|effusion)─┘
          │
  ∇ log p(z) ────────(−)──→ subtract prior once
```

**The mathematical concern:** The identity $p(z \mid c, e) \propto p(z \mid c) p(z \mid e) / p(z)$ only holds when $c \perp e \mid z$. If $z_{\text{common}}$ leaks into $z_{\text{disease}}$ (as observed — cross-head score 0.74–0.87), both disease LDMs encode anatomy, and the prior subtraction does not compensate:

$$\underbrace{\nabla_z \log p(z \mid c)}_{\text{anatomy + cardio}} + \underbrace{\nabla_z \log p(z \mid e)}_{\text{anatomy + effusion}} - \underbrace{\nabla_z \log p(z)}_{\text{anatomy once}} = \underbrace{\nabla_z \log p(z \mid c, e)}_{\text{target}} + \underbrace{\nabla_z \log p(z)}_{\text{extra anatomy term}}$$

The extra $\nabla_z \log p(z)$ over-sharpens anatomical structure and can cause inconsistencies at the cardiac-pleural interface.

**Requires:** Much cleaner disentanglement (cross-head score approaching 0.5) than currently achieved.

---

### Strategy C — Disease sub-block scores only (implicit current assumption)

Assume each disease LDM only acts on its own sub-block and $z_{\text{common}}$ is sampled from a separate base model:

$$\nabla_z \log p(z) = \underbrace{\nabla_{z_c} \log p(z_{\text{common}})}_{\text{base}} + \underbrace{\nabla_{z_d} \log p(z_{\text{cardio}})}_{\text{cardio sub-block}} + \underbrace{\nabla_{z_e} \log p(z_{\text{effusion}})}_{\text{effusion sub-block}}$$

```
Strategy C — Sub-block score decomposition:

   z = [ z_common | z_cardio | z_effusion ]
              ↑          ↑           ↑
         base LDM   cardio LDM  effusion LDM
        (score acts (score acts  (score acts
         on this     on this      on this
         block only) block only)  block only)
```

**Why this is likely broken with the current decoder:** The SepVAE decoder is a shared convolutional network that takes all heads concatenated. It does not treat disease sub-blocks as independent additive perturbations — the decoder mixes all channels before any convolution. Score gradients w.r.t. $z_{\text{cardio}}$ are entangled with $z_{\text{common}}$ and $z_{\text{effusion}}$ through the decoder Jacobian. Sub-block score independence requires a decoder with block-diagonal Jacobian (e.g., independent channel-wise decoders), which is not the current design.

---

### Strategy Comparison

| | Strategy A | Strategy B | Strategy C |
|--|------------|------------|------------|
| **Double-counting risk** | None (different subspaces) | High without near-perfect disentanglement | Medium (decoder entanglement) |
| **Required disentanglement** | $z_c \perp z_e \mid z_{\text{common}}$ | $z_c \perp z_e$ (marginal) | Sub-block independence in decoder |
| **Needs unconditional LDM** | No | Yes | No |
| **Architecturally consistent** | Yes (matches SepVAE design) | Partial | No (decoder mixes blocks) |
| **Implementation complexity** | Moderate | High | Low |
| **Recommended** | **Yes** | After R5b validation | No |

---

## 9. Four Verifiability Criteria and Current Gaps

The following four properties must hold empirically for the research hypothesis to be testable. Without them, composition experiments are premature.

### Criterion V1: Changing $z_{\text{cardio}}$ alters heart enlargement only

**What to measure:** Decode $\hat{x}_{\text{edited}} = \text{Dec}(z_{\text{common}}, z_{\text{cardio}}^*, z_{\text{effusion}})$ where $z_{\text{cardio}}^*$ comes from a cardiomegaly image. Measure pixel-change map $|\hat{x}_{\text{edited}} - \hat{x}_{\text{original}}|$.

**Ideal result:** Change map concentrates in cardiac region (centre, mediastinum); pleural regions (lower lateral) unchanged.

**Current support:** `plot_latent_swap_grid` in `utils/sepvae_diagnostics.py` does this visually.

**Gap:** No quantitative metric. Proposed addition:
```python
cardiac_mask  = image[h*0.2:h*0.7, w*0.3:w*0.7]   # approximate central region
effusion_mask = image[h*0.6:,     :]               # lower third
edit_purity_cardio = mean_change(cardiac_mask) / (mean_change(effusion_mask) + ε)
# Should be >> 1.0 for a clean edit
```

### Criterion V2: Changing $z_{\text{effusion}}$ alters pleural fluid only

Same methodology as V1 but swapping $z_{\text{effusion}}^*$ from an effusion image while keeping $z_{\text{cardio}}$ fixed.

**Ideal result:** Change map concentrates in costophrenic angles and lower lateral lung; cardiac silhouette unchanged.

**Gap:** Same as V1 — visual only, no `edit_purity_effusion` metric.

### Criterion V3: $z_{\text{common}}$ preserves anatomy, pose, and acquisition

**What to measure:** Decode with $z_{\text{common}}$ transferred between two normal images while keeping disease heads fixed. Measure SSIM between transferred and target on non-disease regions.

**Ideal result:** SSIM > 0.85 on lung fields; probe classifiers for cardiomegaly/effusion should give near-chance predictions.

**Gap:** Not implemented. The PCA visualisation of $z_{\text{common}}$ provides qualitative evidence (clusters by acquisition/view), but no SSIM gate.

### Criterion V4: Jointly composed sample resembles real co-morbid images

This is the ultimate test. A composed image from Strategy A should be closer to held-out real co-morbid images than either single-disease synthesis.

**What to measure:**
- FID between composed images and held-out co-morbid split
- Nearest-neighbour distance in VGG feature space
- (Ideal) CheXNet classifier should predict *both* cardiomegaly and effusion with high confidence

**Gap:** No co-morbid evaluation split prepared. No composition pipeline implemented yet (SepVAE training is the prerequisite).

### Summary Table

| Criterion | Qualitative | Quantitative | Gap |
|-----------|-------------|--------------|-----|
| V1: Cardiac edit purity | Swap grid ✓ | Edit purity metric | Not implemented |
| V2: Effusion edit purity | Swap grid ✓ | Edit purity metric | Not implemented |
| V3: z_common anatomy fidelity | PCA ✓ | SSIM on swapped anatomy | Not implemented |
| V4: Composition ≈ real comorbid | — | FID, NN-distance | Neither implemented |

**Recommendation:** Implement V1 and V2 metrics in `utils/sepvae_diagnostics.py` as a gate before committing to LDM training. These can be computed from existing SepVAE checkpoints with no additional training.

---

## 10. Four Proposed Additions

### Addition 1: Explicit shared anatomical expert (base LDM over $z_{\text{common}}$)

$z_{\text{common}}$ already plays this role in the SepVAE. The required addition is to train a dedicated LDM over the $z_{\text{common}}$ embeddings extracted from normal CXRs. This LDM becomes the **base model** in Strategy A — it is sampled first to generate an anatomically consistent background.

**Why this matters:** Without a base model, each disease LDM must implicitly encode anatomy to produce coherent outputs. This creates the double-counting problem. With an explicit base, disease LDMs only need to model the *residual* from healthy anatomy — a much simpler and lower-dimensional distribution.

**Architecture:**
```
LDM_common: p(z_common) — trained on z_common from normal images only
LDM_cardio: p(z_cardio | z_common) — z_common is the conditioning signal
LDM_effusion: p(z_effusion | z_common) — z_common is the conditioning signal
```

**Implementation note:** Conditioning on $z_{\text{common}}$ (a 4-channel spatial map) can be done via cross-attention or concatenation in the UNet denoiser, following standard LDM conditioning patterns already in the codebase.

### Addition 2: Quantitative single-disease editing tests before conjunction

**Block conjunction experiments on this gate.** If single-factor interventions are not clean (V1, V2 edit purity < 2.0), conjunction will fail for fundamental reasons, not implementation reasons.

Proposed gate procedure:
```python
# For each checkpoint to be used for LDM training:
purity_cardio  = eval_edit_purity(model, split="cardio_only")
purity_effusion = eval_edit_purity(model, split="effusion_only")

if purity_cardio < 2.0 or purity_effusion < 2.0:
    print("GATE FAILED: head not clean enough for composition")
    sys.exit(1)
```

Additionally: pass decoded swapped images through a CheXNet-style classifier. An image with $z_{\text{cardio}}^*$ swapped in should increase classifier confidence for cardiomegaly; effusion confidence should be unchanged.

### Addition 3: Jointly trained baseline for comparison

Without this, it is impossible to distinguish:
- **(a)** Composition fails because the semantic assumptions are wrong (diseases are intrinsically entangled)
- **(b)** Composition fails because the SepVAE disentanglement is insufficient

The jointly trained baseline:
```
Single conditional VAE: encode(x, y=[0,1,2]) → z
Single conditional LDM: p(z | y_cardio=1, y_effusion=1) → composed z
```

**Evaluation:** FID between jointly-conditioned synthesis and real co-morbid images, compared against Strategy A composition. If jointly-trained wins, the assumption about score factorisation is wrong. If Strategy A wins, disentangled composition is adding value beyond what a monolithic model can do.

### Addition 4: CFG with unconditional = common expert

An elegant reformulation of Strategy B that avoids explicit score subtraction. Train one conditional LDM with disease labels, where the **unconditional guidance** direction starts from $z_{\text{common}}$-only (disease heads set to their prior $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$):

$$\nabla_z \log p_\theta^{\text{guided}}(z \mid c, e) = \nabla_z \log p_\theta(z \mid \varnothing) + \lambda \left(\nabla_z \log p_\theta(z \mid c) - \nabla_z \log p_\theta(z \mid \varnothing)\right) + \lambda \left(\nabla_z \log p_\theta(z \mid e) - \nabla_z \log p_\theta(z \mid \varnothing)\right)$$

where $\varnothing$ denotes the unconditional (common-only) direction. Expanding:

$$= (1 - 2\lambda) \nabla_z \log p_\theta(z \mid \varnothing) + \lambda \nabla_z \log p_\theta(z \mid c) + \lambda \nabla_z \log p_\theta(z \mid e)$$

**Why this is natural given the architecture:** The SepVAE already defines what "unconditional" means — it is the output when disease heads are clamped to their inactive prior. The common expert is not an abstract baseline; it is the decoder output with $z_{\text{cardio}} = z_{\text{effusion}} = \mathbf{0}$.

**Relationship to Strategy A:** If the SepVAE is well-disentangled, Strategy A and CFG-composition should give approximately equivalent results. If they diverge, the divergence is diagnostic — it reveals how much cross-head information the disease LDMs learned.

---

## 11. Revised Research Hypothesis

### Original hypothesis (from initial framing):
> "If disease-related variation can be approximately factorized from shared anatomical and acquisition variation, then score composition in latent space should better approximate multi-pathology generation than composition in a fully entangled latent space."

### Revised, falsifiable hypothesis:

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
**Comparison baseline:** Jointly-trained conditional model (Addition 3).
**Variant:** CFG-guided composition (Addition 4) as a comparison within composition strategies.

### What the results so far tell us about the gates:

| Gate | Best result to date | Threshold | Status |
|------|---------------------|-----------|--------|
| G1 (Probe AUC) | 0.774 mean (C/D/E ep185) | > 0.75 | **Borderline — not stable** |
| G2 (Cross-head) | 0.744 (inactivity-G ep50) | < 0.65 | **Not met** |
| G3 (Edit purity) | Not measured | > 2.0 | **Not measured** |
| G4 (z_common SSIM) | Not measured | > 0.85 | **Not measured** |

Current training phase is focused on closing G2 (R5a, R5b) and stabilising G1 (R1–R3).

---

## 12. Run Commands

### SLURM launchers (recommended)

```bash
# R1+R2+R3+R4 only (baseline, no structural disentanglement)
bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh

# R1+R2+R3+R4 + R5a (paired contrastive)
bash launchers/single_runs/vae/train_sep_vae_contrastive.sh

# R1+R2+R3+R4 + R5b (cross-head adversarial)
bash launchers/single_runs/vae/train_sep_vae_cross_adv.sh

# All recommendations (R1+R2+R3+R4+R5a+R5b)
bash launchers/single_runs/vae/train_sep_vae_full.sh
```

Override options (apply to any launcher):
```bash
# Change partition
bash launchers/single_runs/vae/train_sep_vae_full.sh --partition gpuq

# Disable W&B
bash launchers/single_runs/vae/train_sep_vae_full.sh --no_wandb

# Resume from checkpoint
bash launchers/single_runs/vae/train_sep_vae_full.sh \
  --resume runs_sepvae/sepvae_full-20260312-xxxxxx/checkpoints/checkpoint_epoch0050.pkl

# Adjust contrastive weight
bash launchers/single_runs/vae/train_sep_vae_full.sh \
  --weight_contrastive 0.2 --contrastive_margin 0.3
```

### Direct Python invocations

```bash
# Baseline fixed (R1–R4)
python run/train_sep_vae.py \
  --exp_name sepvae_baseline_fixed \
  --batch_size 10 --epochs 100 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --weight_perceptual 0.05 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Add R5a (contrastive)
python run/train_sep_vae.py \
  ... \
  --use_contrastive \
  --weight_contrastive 0.1 \
  --contrastive_margin 0.5 \
  --exp_name sepvae_contrastive

# Add R5b (cross-head adversarial)
python run/train_sep_vae.py \
  ... \
  --use_cross_adv \
  --weight_cross_adv 0.05 \
  --lr_cross_disc 1e-4 \
  --cross_disc_hidden_dim 256 \
  --exp_name sepvae_cross_adv

# Full (R5a + R5b)
python run/train_sep_vae.py \
  ... \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --exp_name sepvae_full
```

### Diagnostic commands (post-training)

```bash
# Analyse a completed run directory
python run/analyze_supervised_disease_axes.py \
  --run_dir runs_sepvae/sepvae_full-20260312-xxxxxx \
  --checkpoint checkpoint_epoch0100.pkl

# Generate swap grid manually (without launching a full training run)
python -c "
from utils.sepvae_diagnostics import plot_latent_swap_grid
# ... load model, call function
"
```

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `run/train_sep_vae.py` | Main training script with all R1–R5 changes |
| `losses/sep_vae_losses.py` | Loss functions including `paired_contrastive_loss`, `CrossHeadDiscriminator`, `cross_head_disc_loss` |
| `utils/sepvae_diagnostics.py` | Diagnostic visualisations (swap grid, KL heatmap, PCA) |
| `run/analyze_supervised_disease_axes.py` | Post-hoc analysis script |
| `results/sepvae_disentangle_analysis.md` | Full cross-run comparison report (Feb 2026 runs) |
| `slurm_scripts/sep_vae.slurm` | SLURM job script (reads env vars, builds Python args) |
| `launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh` | R1+R2+R3+R4 launcher |
| `launchers/single_runs/vae/train_sep_vae_contrastive.sh` | R1–R4 + R5a launcher |
| `launchers/single_runs/vae/train_sep_vae_cross_adv.sh` | R1–R4 + R5b launcher |
| `launchers/single_runs/vae/train_sep_vae_full.sh` | All recommendations launcher |
