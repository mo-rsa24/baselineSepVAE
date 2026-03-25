# SepVAE Research Log — Compositional Multi-Pathology Synthesis

**Project:** Separable VAE for compositional chest X-ray generation
**W&B project group:** `sepvae-disentangle`
**Branch:** `sepVAEIndependet`
**Log started:** 2026-02-16
**Last updated:** 2026-03-13

> **How to read this document.**
> Each phase begins with *"What we saw"* (observations), continues with *"Why we changed something"* (scientific rationale), lists the exact hyperparameters or code changes made, and closes with *"What we learned"* (findings that motivated the next phase). Nothing is here without a reason. If you are returning to this project and need to reconstruct why a particular sweep was run or why a piece of code was added, start at the phase that corresponds to the run timestamp.

---

## Table of Contents

1. [Research Objective and Core Hypothesis](#1-research-objective-and-core-hypothesis)
2. [Architecture: What We Built and Why](#2-architecture-what-we-built-and-why)
3. [Phase 1 — Initial Disentangle Runs (Feb 16–17, 2026)](#3-phase-1--initial-disentangle-runs-feb-1617-2026)
4. [Phase 2 — Targeted Hyperparameter Sweeps (Feb 20, 2026)](#4-phase-2--targeted-hyperparameter-sweeps-feb-20-2026)
5. [Phase 3 — Root Cause Diagnoses and Fixes (R1–R6)](#5-phase-3--root-cause-diagnoses-and-fixes-r1r6)
6. [Phase 4 — LDM Proof-of-Concept: Unconditional Sub-Block LDMs](#6-phase-4--ldm-proof-of-concept-unconditional-sub-block-ldms)
7. [Phase 5 — Reconstruction Sharpness Investigation](#7-phase-5--reconstruction-sharpness-investigation)
8. [Phase 6 — Spatial Attention for Disease Routing (R7)](#8-phase-6--spatial-attention-for-disease-routing-r7)
9. [Phase 7 — Full SepVAE Sweeps with All Fixes](#9-phase-7--full-sepvae-sweeps-with-all-fixes)
10. [Research Theory: Assumptions and Composition Strategies](#10-research-theory-assumptions-and-composition-strategies)
11. [Verification Gates Before Composition](#11-verification-gates-before-composition)
12. [Future Roadmap: Phases 8–11](#12-future-roadmap-phases-811)
13. [File and Checkpoint Registry](#13-file-and-checkpoint-registry)

---

## 1. Research Objective and Core Hypothesis

### What we are trying to do

We want to synthesise a realistic chest X-ray containing **both** cardiomegaly and pleural effusion without ever having trained a model jointly on comorbid data. Instead, we train separate generative models — one per disease — and **compose** them at inference time.

This is clinically valuable because:
- Comorbid cases (both diseases present) are rare and expensive to label cleanly
- Controlled synthesis of combined pathologies is useful for data augmentation and robustness testing
- Disentangled latent representations allow attribute-level editing, which supports radiological explainability

### The SepVAE as a prerequisite

Before any composition can happen, we need a latent space where disease variation is separated from shared anatomy. The **Separable VAE (SepVAE)** is the first building block: a VAE with a frozen pretrained backbone (CheSS, ResNet-50) and three separate encoder heads producing:

$$z = [z_{\text{common}} \;|\; z_{\text{cardio}} \;|\; z_{\text{effusion}}]$$

- $z_{\text{common}}$ (4 channels): captures shared anatomy, patient position, acquisition settings
- $z_{\text{cardio}}$ (2 channels): should capture **only** cardiomegaly-specific variation
- $z_{\text{effusion}}$ (2 channels): should capture **only** effusion-specific variation

Everything downstream — LDM training, score composition, comorbid synthesis — depends on these heads being genuinely disentangled. If $z_{\text{cardio}}$ leaks effusion information or $z_{\text{common}}$ absorbs cardiomegaly variation, composition will produce globally inconsistent images.

### Informal hypothesis (starting point)

> "If disease-related variation can be approximately factorised from shared anatomical and acquisition variation, then score composition in latent space should better approximate multi-pathology generation than composition in a fully entangled latent space."

This hypothesis is refined into a falsifiable form in [Section 10](#10-research-theory-assumptions-and-composition-strategies) after we identified what "composition" concretely means and what must be measured to test it.

---

## 2. Architecture: What We Built and Why

### 2.1 Encoder

```
Input x (512×512 CXR, 1 channel)
        │
        ▼
┌────────────────┐
│ CheSS backbone │  ResNet-50 pretrained on CXR classification (frozen)
│   (stride 32)  │  Output: feature map f ∈ ℝ^{16×16×2048}
└───────┬────────┘
        │  (optionally bilinear-upsampled to 64×64 if use_fpn=False)
        │  (or FPN-fused from layer2/3/4 to 64×64 if use_fpn=True)
        ▼
┌────────────────────────────────────────────────────┐
│                 Three encoder heads                 │
│                                                     │
│  head_common:     ConvHead → μ_c, σ_c   (4 ch)    │
│  head_cardiomegaly: ConvHead → μ_d1, σ_d1 (2 ch)  │
│  head_effusion:   ConvHead → μ_d2, σ_d2 (2 ch)    │
└────────────────────────────────────────────────────┘
        │
        ▼
z = cat[z_common (4ch), z_cardio (2ch), z_effusion (2ch)]  @  64×64 spatial
```

**Why frozen backbone:** CheSS provides rich, CXR-specific features without needing to train feature extraction from scratch. Freezing it avoids catastrophic forgetting and keeps training stable. The encoder heads are lightweight ConvHead modules trained on top.

**Why 64×64 spatial latents:** Spatial latents preserve where in the image the disease information lives. Global pooled vectors (standard VAE) discard spatial structure that is needed for region-selective editing.

**Why 4 + 2 + 2 channel split:** Empirically small disease heads discourage the model from routing common structure into them (the reconstruction gradient is small relative to the KL penalty). The common head has 4 channels to give it enough capacity for anatomy, pose, and acquisition variation.

### 2.2 Disease label routing

The dataloader provides triplets: $(x_{\text{norm}},\, x_{\text{eff}},\, x_{\text{cardio}})$. Within each batch, labels $y \in \{0, 1, 2\}$ control which heads are "active":

| Label | Meaning | Active head | Inactive heads |
|-------|---------|-------------|----------------|
| $y=0$ | Normal | none | $z_{\text{cardio}}$, $z_{\text{effusion}}$ |
| $y=1$ | Effusion | $z_{\text{effusion}}$ | $z_{\text{cardio}}$ |
| $y=2$ | Cardiomegaly | $z_{\text{cardio}}$ | $z_{\text{effusion}}$ |

Inactive heads are penalised toward $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$ — a narrow prior that forces the mean to zero and prevents the head from encoding anything for images of the wrong disease class.

### 2.3 Decoder

```
z (64×64 × 8 channels)
        │
        ▼  initial conv 8ch → 256ch
        │
        ▼  ResBlocks + SmoothUp ×3:
        │     64×64 → 128×128 → 256×256 → 512×512
        │     (bilinear-upsample or subpixel-shuffle, configurable)
        ▼
        Conv 1×1 → Sigmoid
        │
        ▼
x̂  (512×512 × 1 channel)
```

All three heads are concatenated before the decoder. The decoder is shared — it mixes all channels in its first convolution. This has important implications for composition strategies (discussed in Section 10).

### 2.4 Loss portfolio (initial)

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \mathcal{L}_{\text{KL}} + w_{\text{perc}} \mathcal{L}_{\text{perc}} + w_{\text{null}} \mathcal{L}_{\text{null}} + w_{\text{orth}} \mathcal{L}_{\text{orth}} + w_{\text{MI}} \mathcal{L}_{\text{MI}}$$

Where:
- $\mathcal{L}_{\text{rec}}$: L2 pixel reconstruction
- $\mathcal{L}_{\text{KL}}$: per-channel KL with free-bits threshold and warmup
- $\mathcal{L}_{\text{perc}}$: VGG perceptual loss
- $\mathcal{L}_{\text{null}}$: penalises non-zero $\mu$ on inactive heads — $\mathbb{E}[\mu_{\text{inactive}}^2]$
- $\mathcal{L}_{\text{orth}}$: Barlow-style cross-correlation + prototype cosine penalty
- $\mathcal{L}_{\text{MI}}$: MI discriminator (joint vs shuffled pair, trained adversarially)

### 2.5 Key metrics we track

- **Probe AUC**: linear classifier trained on disease head latents; higher = head captures more disease information
- **Cross-head score**: mean ratio of inactive-head energy to active-head energy; 0.5 = ideal (no leakage), 1.0 = total leakage
- **KL per head**: to detect dead zones and collapse

---

## 3. Phase 1 — Initial Disentangle Runs (Feb 16–17, 2026)

### 3.1 What we ran

Five runs under the `sepvae_disentangle` name, forming one continuous training chain split across SLURM jobs:

| Run ID | W&B | Epochs | Notes |
|--------|-----|--------|-------|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | 192 | Old script — no probe AUC logged |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | 1 | **CRASHED** at step 0 |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | 84 | Stopped; resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | 101 | Stopped; resumed as E |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | 200 | **NaN crash at epoch 200** |

> Runs C, D, E are the same training, checkpointed and resubmitted. Together they represent one 200-epoch run (the disentangle-E chain).

**Configuration (disentangle-C/D/E):**

```bash
--free_bits 1.0
--sigma_inactive 0.1         # → KL_inactive ≈ 1.8 nats (above free_bits)
--weight_null 0.01
--weight_orthogonality 0.03
--weight_mi 0.003
--weight_perceptual 0.03
--use_fpn true
--batch_size 6
--lr_vae 6.7e-5
--kl_warmup_epochs 30
--half_precision fp32
--epochs 200
```

### 3.2 What we saw

**Good:** Disease heads learned something useful. Peak probe AUC ≈ 0.774 mean (cardiomegaly: 0.776, effusion: 0.772) at around epoch 185. This is the best any run has achieved.

**Problem 1 — NaN at epoch 200.** Total loss was stable (0.004–0.006) through epoch 199, then all terms simultaneously became `nan`. PCA failed with `ValueError: Input X contains NaN`. The run had no learning rate schedule. After 200 epochs with constant LR, one catastrophic gradient update from an unlucky batch cascaded through the network.

**Problem 2 — Persistent cross-head leakage.** Cross_head_score fluctuated between 0.85 and 1.01 throughout all 200 epochs. The heads never achieved genuine independence. The MI discriminator and orthogonality loss reduce marginal correlation but do not enforce **conditional** independence — a head can still encode the other disease without violating orthogonality or MI constraints.

**Problem 3 — High probe AUC variance.** The 600-sample probe eval produced noisy estimates — the same checkpoint read anywhere from 0.53 to 0.77 on adjacent evaluations. Epoch 185 "best" is partly noise; rolling averages were not implemented.

**Problem 4 — No discrimination between active/inactive KL regions.** The free_bits mechanism clips KL gradients for any channel below $\lambda_{\text{fb}} = 1.0$ nats. Since $\sigma_{\text{inactive}} = 0.1$ gives $\text{KL}_{\text{inactive}} \approx 1.8 > 1.0$, disease channels here are above the threshold and receive gradients — this works. But we had not yet systematically checked what happens when free_bits is raised.

### 3.3 What we learned / questions raised

- Long training (200 epochs) does improve peak performance — but **without LR decay it is unstable**. We need a cosine schedule.
- The current regularisation set ($\mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{MI}} + \mathcal{L}_{\text{null}}$) is insufficient to eliminate cross-head leakage. A stronger structural mechanism is needed.
- The cardiomegaly head seems harder to train — its probe AUC is more volatile and lower than effusion's. Is this a feature of the disease (distributed, subtle signal) or a regularisation imbalance?

**Questions these findings raised that drove Phase 2:**
1. What if we increase the inactivity pressure (tighter $\sigma_{\text{inactive}}$, stronger nulling)?
2. What if we increase the independence pressure (stronger orthogonality, add explicit MI weight)?
3. What is the effect of `free_bits` on the disease heads specifically?

---

## 4. Phase 2 — Targeted Hyperparameter Sweeps (Feb 20, 2026)

We ran two new experiments, each isolating a different hypothesis from the Phase 1 findings. Both used a simplified setup: no FPN, bf16, batch size 10, 100 epochs.

### 4.1 Sweep 1: Inactivity-driven (`sepvae_inactivity_driven`)

**Scientific rationale:** Phase 1 showed leakage despite moderate null/ortho weights. Hypothesis: if the inactive heads are held even tighter to the prior ($\sigma_{\text{inactive}} = 0.05$ instead of 0.1), and the nulling weight is increased 5×, the inactive head will have less capacity to represent the other disease.

$$\text{KL}_{\text{inactive}}(\sigma=0.05) = \frac{1}{2}(0.05^2 - 1 - \log 0.05^2) \approx 2.5 \text{ nats}$$

This is above `free_bits=1.0`, so gradients still flow.

**Run:** `sepvae_inactivity_driven-20260220-085943` | W&B `9lj20so0`
(Note: `62635` / W&B `99s5jhqe` crashed at ~93s from transient GPU conflict; the config was identical, relaunched 23 minutes later.)

```bash
--free_bits 1.0
--sigma_inactive 0.05        # tighter inactive prior (KL_inactive ≈ 2.5)
--weight_null 0.05           # 5× stronger than Phase 1
--weight_orthogonality 0.05
--weight_mi 0.005
--weight_perceptual 0.05
--use_fpn false
--batch_size 10
--lr_vae 1e-4
--kl_warmup_epochs 10
--half_precision bf16
--epochs 100
```

**Results:**

| Metric | Best (~ep50) | Final (ep100) |
|--------|-------------|---------------|
| Probe AUC (cardio) | 0.639 | **0.476** (below chance) |
| Probe AUC (effusion) | 0.627 | 0.767 |
| Cross-head score | **0.744** | 0.809 |

**What we saw:** This run achieved the **best cross-head leakage suppression** of all runs — cross_head_score = 0.744 at its best. However, by epoch 100 the cardiomegaly probe AUC had collapsed to 0.476 (sub-random). The strong nulling pressure drove $\mu_{\text{cardio}} \to 0$ faster than the reconstruction gradient could maintain the cardiomegaly signal.

**Why cardiomegaly specifically?** Cardiomegaly is a distributed, low-contrast change (enlarged cardiac silhouette, subtle mediastinal widening). Effusion is a localised, high-contrast finding (bright pleural fluid). The nulling loss applies equal pressure to both heads regardless of how strong the disease signal is. For cardiomegaly, the signal is weaker — the nulling pressure wins over the reconstruction gradient that tries to preserve it.

**What we learned:** Reducing leakage at the cost of one head dying is not a solution. The regularisation balance needs to be asymmetric or the minimum active KL needs to be floored. This is captured in R4.

---

### 4.2 Sweep 2: Independence-driven (`sepvae_independence_driven`)

**Scientific rationale:** Orthogonality and MI enforce marginal independence. What if we push these much harder (10× stronger orthogonality) while relaxing nulling? And what happens to `free_bits` if we raise it to 2.0?

**Run:** `sepvae_independence_driven-20260220-085943` | W&B `41nce8qq`
(Note: `62742` / W&B `i9cevm1x` crashed at ~91s; same transient GPU issue.)

```bash
--free_bits 2.0              # ← THIS IS THE CRITICAL MISTAKE
--sigma_inactive 0.1         # → KL_inactive ≈ 1.8 nats (below free_bits=2.0!)
--weight_null 0.01
--weight_orthogonality 0.1   # 3× stronger than Phase 1
--weight_mi 0.01             # 3× stronger than Phase 1
--weight_perceptual 0.05
--use_fpn false
--batch_size 10
--lr_vae 1e-4
--kl_warmup_epochs 10
--half_precision bf16
--epochs 100
```

**Results:**

| Metric | Early peak (~ep25) | Final (ep100) |
|--------|--------------------|---------------|
| Probe AUC (cardio) | 0.762 | 0.532 |
| Probe AUC (effusion) | 0.737 | 0.812 |
| Cross-head score | 0.820 | 0.871 |
| Disease head $\mu$-norms | moderate | **≈ 0.025–0.030** (collapsed) |
| KL (disease channels) | variable | **uniformly 1.7–1.8 nats** |

**Root cause — the free_bits dead zone:**

The free-bits mechanism clips KL gradients to zero for any channel with KL below the threshold:

```
KL_inactive ≈ 1.8 nats
─────────────────────────────────────────────────────────
          ◄── gradient = 0 in this zone ──►
    0 ───────────────────────── 2.0 ──────── ∞
                                 ↑ free_bits = 2.0
              disease heads land here permanently
```

Because $\text{KL}_{\text{inactive}}({\sigma=0.1}) \approx 1.8 < \text{free\_bits} = 2.0$, all disease channels are permanently below the threshold. They receive **no KL gradient**. The nulling loss drives $\mu \to 0$, the channels collapse, and the probe AUC degrades to near-chance.

There is a brief window around epoch 20–25 where the channels haven't fully collapsed yet (probe AUC 0.75), but without KL gradients to maintain structure, the heads degrade over the remaining 75 epochs.

**What we learned:** The free_bits mechanism was designed to protect the common head from over-penalisation on low-information channels. Raising it to 2.0 inadvertently killed the disease heads. This is a binary failure condition, not a soft tradeoff. **Any configuration with $\text{KL}_{\text{inactive}} < \text{free\_bits}$ will fail.**

---

### 4.3 Cross-run patterns (summary after Phase 2)

These five patterns were identified by comparing all runs:

**Pattern 1 — `free_bits` / `sigma_inactive` conflict is a hard failure mode**
Any run with $\text{KL}_{\text{inactive}} < \text{free\_bits}$ results in disease heads in a permanent gradient dead zone. All such runs fail. This must be checked before any other hyperparameter analysis.

$$\text{KL}_{\text{inactive}} = \frac{1}{2}\left(\sigma_{\text{inactive}}^2 - 1 - \log \sigma_{\text{inactive}}^2\right)$$

| $\sigma_{\text{inactive}}$ | $\text{KL}_{\text{inactive}}$ | Must keep `free_bits` below |
|--------------------------|-------------------------------|------------------------------|
| 0.20 | ~1.1 nats | 1.0 |
| 0.10 | ~1.8 nats | 1.5 |
| 0.05 | ~2.5 nats | 2.0 (safe up to 2.4) |

**Pattern 2 — Cross-head leakage is universally unsolved**
No run ever achieved cross_head_score < 0.74. The orthogonality + MI + nulling portfolio enforces marginal independence but not conditional independence. The heads can share information about each other's disease without violating any current loss.

**Pattern 3 — Cardiomegaly is consistently harder to train**
Across all runs, cardiomegaly probe AUC is lower, more volatile, and more prone to collapse. Likely because: (a) cardiomegaly is a subtle, distributed feature; (b) CheSS backbone was not trained to separate cardiac pathology from anatomy; (c) the nulling loss applies equal pressure regardless of signal strength.

**Pattern 4 — Probe AUC is too noisy to use single-epoch readings**
Same checkpoint read at adjacent epochs can differ by 0.2+ AUC. The 600-sample probe eval has high variance. Rolling averages or best-of-5 should be the reporting standard.

**Pattern 5 — Long training improves peak but destabilises without LR decay**
200-epoch chain achieves highest peak (0.774) but ends in NaN. No LR schedule in any run.

---

## 5. Phase 3 — Root Cause Diagnoses and Fixes (R1–R6)

After Phase 2, we systematically implemented fixes for each diagnosed failure. These are the code and configuration changes made before the next round of training.

### R1 — Fix the free_bits / sigma_inactive conflict

**Problem:** `free_bits=2.0` kills disease-head gradients when `sigma_inactive=0.1`.
**Fix:** Set `free_bits=0.0` (removed entirely). Control inactivity through `sigma_inactive` alone — tighter $\sigma$ means tighter inactive prior, no artificial gradient clipping.
**Rationale:** Free-bits was originally protecting the common head; with careful $\sigma_{\text{inactive}}$ tuning it is not needed and creates more problems than it solves.

```bash
# Before (broken): free_bits=2.0, sigma_inactive=0.1 → dead zone
# After (fixed):   free_bits=0.0, sigma_inactive=0.05 → gradients flow everywhere
--free_bits 0.0
--sigma_inactive 0.05
```

---

### R2 — Rebalance regularisation weights

**Problem:** `weight_null=0.05` collapses cardiomegaly head; `weight_null=0.01` is insufficient.
**Fix:** Use `weight_null=0.02`, `weight_orthogonality=0.02` — moderate, balanced pressure. Equal null/ortho weighting (unlike the 10:1 orthogonality:null ratio in independence-I).
**Rationale:** The inactivity-G run showed that leakage improves with stronger pressure, but the cardiomegaly head cannot sustain heavy nulling. A gentler equilibrium is needed.

```bash
--weight_null 0.02
--weight_orthogonality 0.02
--weight_mi 0.005
```

---

### R3 — Add cosine LR decay (prevent NaN at long training)

**Problem:** Constant LR over 200 epochs caused NaN explosion at the final epoch.
**Fix:** Cosine decay starting from `lr_decay_epochs` onward, decaying to 10% of initial LR.

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi (t - t_{\text{decay}})}{T - t_{\text{decay}}}\right)\right), \quad \eta_{\min} = 0.1\eta_0$$

**Implementation:** `optax.join_schedules` with a constant phase followed by `optax.cosine_decay_schedule`.

```bash
--lr_decay_epochs 60    # start decaying at epoch 60 of a 100-epoch run
--lr_vae 1e-4           # decays to 1e-5 by final epoch
```

---

### R4 — Floor active-head KL to prevent cardiomegaly collapse

**Problem:** Nulling loss applies to inactive samples but cannot distinguish "this head should be active here." The cardiomegaly head receives so much nulling pressure that it collapses even on cardiomegaly images.
**Fix:** For active-label samples, add a penalty that drives KL above a minimum floor (e.g., 2.0 nats), so the head cannot collapse even under strong nulling.

```bash
--min_active_kl 2.0     # active head must maintain KL ≥ 2.0 nats
```

---

### R5a — Paired contrastive loss (structural disentanglement)

**Problem:** Orthogonality and MI act on marginals/geometry but do not force the inactive head to be blind to the other disease.
**Scientific rationale:** We need a loss that directly says: "when you see a cardiomegaly image, the effusion head should look exactly like it does for a normal image."
**Implementation:** Prototype-based centroid loss.

For disease head $k$ with active class $c_k$, let $\bar{z}_k^{(c)}$ be the L2-normalised GAP-pooled centroid over class $c$:

$$\mathcal{L}_{\text{sep-push}}^{(k)} = \max\left(0,\; \text{margin} - \left(1 - \bar{z}_k^{(c_k)} \cdot \bar{z}_k^{(\text{norm})}\right)\right)$$

$$\mathcal{L}_{\text{blind-pull}}^{(k)} = 1 - \bar{z}_k^{(c_{\text{other}})} \cdot \bar{z}_k^{(\text{norm})}$$

$$\mathcal{L}_{\text{contrastive}} = \mathcal{L}_{\text{sep-push}}^{(\text{cardio})} + \mathcal{L}_{\text{sep-push}}^{(\text{effusion})} + \mathcal{L}_{\text{blind-pull}}^{(\text{cardio})} + \mathcal{L}_{\text{blind-pull}}^{(\text{effusion})}$$

The **blind-pull** term is the key addition: it pulls the cardiomegaly-image centroid in the effusion head toward the normal centroid — the effusion head must not respond to cardiomegaly.

**Flag:** `--use_contrastive` | **Weight:** `--weight_contrastive 0.1`

---

### R5b — Cross-head adversarial discriminators (conditional independence)

**Problem:** Even if marginal distributions are orthogonal, the heads can still encode cross-disease information in their conditional structure. We need to test and penalise **conditional predictability**.
**Scientific rationale:** Two adversarial classifiers directly test whether $z_{\text{effusion}}$ encodes cardiomegaly information (and vice versa). The VAE is penalised if they succeed.

Two 3-layer MLPs:
- $D_{c \to e}$: predicts cardiomegaly label from $z_{\text{effusion}}$
- $D_{e \to c}$: predicts effusion label from $z_{\text{cardio}}$

$$\mathcal{L}_{\text{disc}} = \text{BCE}(D_{c \to e}(z_{\text{effusion}}), y_{\text{cardio}}) + \text{BCE}(D_{e \to c}(z_{\text{cardio}}), y_{\text{effusion}})$$

$$\mathcal{L}_{\text{cross-adv}} = \mathbb{E}\left[D_{c \to e}(z_{\text{effusion}})^2 + D_{e \to c}(z_{\text{cardio}})^2\right] \cdot \mathbf{1}[\text{disease sample}]$$

When $D_{c \to e}(z_{\text{effusion}}) \to 0.5$, the effusion head contains no recoverable cardiomegaly information.

**Flag:** `--use_cross_adv` | **Weights:** `--weight_cross_adv 0.05 --lr_cross_disc 1e-4`

---

### R6 — Remove FPN

**Problem:** FPN (Feature Pyramid Network) was used in the older disentangle runs but not in the Feb 20 runs. The Feb 20 runs reached comparable or better probe AUC without FPN.
**Fix:** Default to `use_fpn=False`. FPN adds parameters and peak memory with no measurable disentanglement benefit.

```bash
# Remove: --use_fpn true
# Default is now False
```

---

### Phase 3 recommended run (baseline_fixed, implements R1–R4, R6)

```bash
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
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle \
  --exp_name sepvae_baseline_fixed
```

With R5a (contrastive):
```bash
python run/train_sep_vae.py ... \
  --use_contrastive --weight_contrastive 0.1 --contrastive_margin 0.5 \
  --exp_name sepvae_contrastive
```

With R5b (cross-adversarial):
```bash
python run/train_sep_vae.py ... \
  --use_cross_adv --weight_cross_adv 0.05 --lr_cross_disc 1e-4 \
  --exp_name sepvae_cross_adv
```

Full stack (R5a + R5b):
```bash
python run/train_sep_vae.py ... \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv --weight_cross_adv 0.05 \
  --exp_name sepvae_full
```

SLURM launchers (preferred — handles staging, env setup):
```bash
bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh
bash launchers/single_runs/vae/train_sep_vae_contrastive.sh
bash launchers/single_runs/vae/train_sep_vae_cross_adv.sh
bash launchers/single_runs/vae/train_sep_vae_full.sh
```

---

## 6. Phase 4 — LDM Proof-of-Concept: Unconditional Sub-Block LDMs

### 6.1 Why we ran LDMs before the SepVAE was fully fixed

The full composition pipeline (Strategy A, described in Section 10) requires:
1. Phase 1 SepVAE quality gate to pass (`specificity_ratio > 2.0`)
2. `LDM_common` trained on $z_{\text{common}}$ with $z_{\text{disease}}$ LDMs conditioned on $z_{\text{common}}$

However, before investing engineering effort in conditional LDM architecture, there is a more basic question: **Can a diffusion model learn to generate the disease sub-block latents at all?** Specifically:
- Is the marginal distribution of $z_{\text{cardio}}$ (2 channels × 64×64 spatial) well-shaped for VP-SDE diffusion?
- Does sampling from $p(z_{\text{disease}})$ and decoding produce recognisable disease morphology?
- Are the latent scale statistics reasonable after pre-encoding?

If these marginal LDMs fail, investing in conditional Strategy A architecture is premature. These runs are a **feasibility gate**, not a detour.

### 6.2 Checkpoint selection — why disentangle-E ep180

We had three completed checkpoints:
- `disentangle-E ep180–185` (best, W&B `laikh8dr`)
- `inactivity-G ep100` (W&B `9lj20so0`) — cardiomegaly head collapsed (AUC 0.476)
- `independence-I ep100` (W&B `41nce8qq`) — both heads in KL dead zone

The only checkpoint where **both** disease heads carry meaningful class information is disentangle-E. Despite its leakage issues (cross_head_score = 0.935), it is the only viable starting point.

We use **epoch 180** rather than epoch 185 (stated best in the results table) because `save_every=10` — checkpoints exist at ep170, ep180, ep190, ep200. Epoch 185 was not saved. ep190 and ep199 have lower probe AUC than the ep185 peak. ep180 is the nearest clean checkpoint.

| Epoch | Mean probe AUC | Status |
|-------|---------------|--------|
| ep170 | ~0.75 (est.) | Available |
| **ep180** | **~0.77 (est.)** | **Selected** |
| ep190 | ~0.74 | Degraded |
| ep199 | 0.737 | Last valid |
| ep200 | NaN | Unusable |

### 6.3 Pre-encoding pipeline

Pre-encoding extracts the disease-head latents ($\mu$ only, not a reparameterised sample) from the SepVAE encoder and writes them to `.npy` files indexed by `manifest.jsonl`. A scale factor $s = 1/\text{std}(\mu)$ is computed over the full dataset and stored in `latent_meta.json`; all `.npy` files are rescaled to unit variance so the LDM receives standard-normal inputs.

**Submit both simultaneously (CPU-only, no GPU contention):**

```bash
# Cardiomegaly latents
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio \
  slurm_scripts/preencode_sepvae.slurm

# Effusion latents
sbatch --nodelist=mscluster76 \
  --job-name=preencode-effusion \
  --export=ALL,\
DISEASE=effusion,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_effusion \
  slurm_scripts/preencode_sepvae.slurm
```

If interrupted, add `RESUME=1` to skip already-encoded image IDs:
```bash
sbatch ... --export=ALL,...,RESUME=1 slurm_scripts/preencode_sepvae.slurm
```

Read scale factor after job completes:
```bash
python -c "import json; d=json.load(open('preencoded_latents/disentangle_cardio/latent_meta.json')); print(d['latent_scale_factor'])"
```

### 6.4 LDM training

After pre-encoding, launch LDMs. `SAMPLE_EVERY=9999` disables in-loop sampling (no VAE decoder is loaded — training runs on stored latents only). Key parameters match the SepVAE disease sub-block: `latent_size=64`, `vae_z_channels=2`, `ldm_z_channels=2`.

```bash
./launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh full_train
./launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh full_train
```

### 6.5 What these runs are NOT

These are **not** the final composition LDMs. They do not condition on $z_{\text{common}}$. They do not account for the double-counting problem (Section 10). They will produce imperfect compositions because:
- $z_{\text{common}}$ must be provided externally (fixed from a real image) — no generation of anatomy
- The leakage in disentangle-E means each disease head partially encodes the other disease

They exist to answer the feasibility question and provide a quick visualisation of whether the learned latent subspace is useful for disease-specific generation at all.

---

## 7. Phase 5 — Reconstruction Sharpness Investigation

### 7.1 What we observed

Looking at the reconstruction grid (originals on top, reconstructions on bottom), the reconstructed CXRs are consistently soft and blurry. Fine detail — rib cortex edges, vessel walls, air-bronchogram texture — is absent from all reconstructions regardless of which run produced the checkpoint.

### 7.2 Root cause analysis

Blurriness has four compounding sources in this specific architecture:

```
Input x  (512×512, full radiographic detail)
     │
     ▼  Frozen ResNet-50 backbone (cumulative stride = 32)
Feature map  (16×16, ~2048ch)
     │   ← HIGH-FREQUENCY TEXTURE LOST HERE
     │     backbone trained for classification → discards texture at stride-32
     ▼
8-channel spatial bottleneck (64×64×8 after encoder heads)
     │   ← COMPRESSION RATIO ≈ 0.78% of input
     │
     ▼  Decoder: 3 × (ResBlocks → SmoothUp)
     │     Each SmoothUp = bilinear-resize + conv × 2 (6 smoothing convs total)
     │
     ▼  L2 reconstruction loss
     │   ← optimal under L2 is the posterior mean = blurred average
     │
     ▼  No discriminator (weight_adversarial = 0.0)
     │   ← nothing penalises statistically implausible smooth outputs
     ▼
  x̂  (blurry)
```

**Compression ratio:**
$$\text{compression ratio} = \frac{8 \times 16 \times 16}{512 \times 512} = \frac{2048}{262144} \approx 0.78\%$$

**Why L2 produces blur:** The ELBO reconstruction term is:
$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{q(z|x)}\left[\|x - \hat{x}_\theta(z)\|_2^2\right]$$
The optimal decoder under L2 is $\hat{x}_\theta(z) = \mathbb{E}_{p(x|z)}[x]$. For any region with ambiguity across plausible completions, the mean is a blurred average. The sharper the texture, the more it blurs.

**Why `bilinear` + double-smoothing makes it worse:** Each `SmoothUp` block:
```python
h = jax.image.resize(x, target_shape, method='bilinear')  # low-pass filter
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 1
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 2
```
Three upsampling stages × two smoothing convolutions = **6 successive low-pass operations** before the final sigmoid. High-frequency energy is progressively destroyed.

### 7.3 Fixes — ordered by expected impact

#### Fix 1: Enable PatchGAN adversarial loss (highest impact — already implemented, just disabled)

The PatchGAN discriminator is fully implemented in `losses/sep_vae_losses.py`. It has been `weight_adversarial=0.0` in all runs to date. Setting it to 0.1 with `disc_start_epoch=10` forces the decoder to produce sharp, realistic patches instead of blurred averages.

```bash
--weight_adversarial 0.1
--disc_start_epoch 10      # GAN activates after 10 epoch warmup
```

**Scientific rationale:** Averaged blurry textures look statistically unlike real X-ray patches. The discriminator rejects them; the generator (decoder) must commit to a single sharp realisation. This is why VAE-GANs (VQ-GAN, etc.) are sharper than plain VAEs.

#### Fix 2: Switch to subpixel (pixel-shuffle) upsampling (zero-cost — already implemented)

`--upsample_method subpixel` replaces fixed bilinear + two smoothing convs with a learned pixel-shuffle that can amplify high-frequency components. Already implemented in `models/sep_vae_jax.py:68–82`. Requires one flag change.

```bash
--upsample_method subpixel
```

#### Fix 3: Increase perceptual loss weight (conservative improvement)

VGG perceptual loss biases the reconstruction toward matching feature statistics that correlate with human perception of sharpness. Current `weight_perceptual=0.05` is conservative.

```bash
--weight_perceptual 0.1
```

#### Fix 4 (optional): Partially unfreeze backbone

Allows the encoder to re-learn what texture to preserve for reconstruction. Improves the highest-frequency detail that stride-32 pooling discards.

```bash
--unfreeze_from layer3    # or layer4 for a lighter change
```

Use cautiously — unfreezing the backbone can destabilise the disentanglement objectives and increase memory significantly.

### 7.4 Recommended sharpness sweep

```bash
python run/train_sep_vae.py \
  --exp_name sepvae_sharp_test \
  --upsample_method subpixel \
  --weight_adversarial 0.1 \
  --weight_perceptual 0.1 \
  --disc_start_epoch 10 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --batch_size 10 --epochs 100 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

> Note on `weight_null=0.02` and `weight_orthogonality=0.02`: these are deliberately lightened relative to the full recommended values (0.05) to balance against the new adversarial loss. The discriminator introduces a competing gradient signal; over-regularising disentanglement simultaneously can destabilise training.

### 7.5 Fundamental limit

Even with all fixes applied, sub-rib-level sharpness will not be fully recovered. The stride-32 ResNet discards that information before the encoder heads ever see it. The frozen backbone is the hard ceiling on reconstruction quality. The fixes above are about closing the gap between the ceiling and current quality (which is far below the ceiling due to L2 + bilinear + no GAN).

---

## 8. Phase 6 — Spatial Attention for Disease Routing (R7)

### 8.1 What we observed — Pattern 3 revisited

After Phase 2, Pattern 3 became a central problem: cardiomegaly probe AUC consistently underperforms and collapses under pressure. The root cause we identified in Phase 3 is:

> Both the cardiomegaly and effusion ConvHeads see the **full** 64×64 backbone feature map and must learn, through gradient pressure from orthogonality and MI losses alone, to selectively ignore the spatial regions belonging to the other disease.

This is an ill-posed implicit learning problem. Cardiomegaly occupies the central ~30% of the image (enlarged cardiac silhouette — low contrast, distributed). Effusion occupies the lower lateral 10–15% (bright pleural fluid — high contrast, localised). A ConvHead with no spatial routing bias applies equal weight to cardiac and pleural regions simultaneously. The cardiomegaly head must suppress its response to the pleural region purely through weight tuning — which is why it consistently fails under strong regularisation.

### 8.2 Scientific rationale for attention

If we give each disease head an explicit, **learnable spatial routing mechanism**, the model can learn to attend to the region where its disease lives, structurally preventing cross-head leakage at the source (encoder input) rather than at the output (latent geometry, which is what orthogonality and MI address).

### 8.3 Mechanism — learned disease prototype query

Each disease head learns a prototype vector $q_d \in \mathbb{R}^{D}$ (default $D=256$). This query is used to compute a spatial attention map over all $HW = 64 \times 64$ positions in the backbone feature map:

$$A_d(i) = \text{softmax}\!\left(\frac{K_i^\top q_d}{\sqrt{D}}\right), \quad K = \text{Dense}(h_\text{flat})$$

The attention map is rescaled by $HW$ (so that at initialisation, when $q_d \approx 0$, $A_d \approx 1$ everywhere and the attended features equal the unattended input) and used to gate the backbone features:

$$h_\text{attended} = h \odot (A_d \cdot HW)$$

**Key properties:**
- **At initialisation:** $h_\text{attended} \approx h$ — identical to the baseline ConvHead, no training instability
- **Over training:** $q_\text{cardio}$ learns to concentrate $A_\text{cardio}$ on the cardiac silhouette; $q_\text{effusion}$ concentrates on the pleural angles
- **Free diagnostic:** $A_d$ is a 64×64 spatial heatmap — logged to W&B under `diagnostics/attn_maps` every `sample_every` epochs. These maps directly proxy the verifiability criteria V1/V2: if $A_\text{cardio}$ concentrates over the cardiac region, spatial selectivity is structurally enforced

### 8.4 Connection to existing problems

**Pattern 2 (persistent leakage):** Orthogonality and MI penalise the *output* of the heads (the latent vectors). Attention addresses leakage at the *input* — if the cardiomegaly head attends only to cardiac regions, it never has access to pleural features, so there is nothing to suppress through regularisation.

**Pattern 3 (cardiomegaly collapse under nulling):** Without attention, the cardiomegaly signal is spatially diffuse across the full 64×64 map — strong nulling pressure drives the mean to zero everywhere. With attention, the active signal is concentrated in a spatial subregion. The same nulling pressure is absorbed by fewer, higher-weight positions that also receive strong reconstruction gradient. Cardiomegaly head robustness under regularisation should improve.

### 8.5 Sweep

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_label_attention \
  --use_label_attention \
  --attn_query_dim 256 \
  --batch_size 10 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

Full stack (R5a + R5b + R7):
```bash
python -m run.train_sep_vae \
  --exp_name sepvae_full \
  --use_label_attention --attn_query_dim 256 \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv --weight_cross_adv 0.05 \
  --batch_size 10 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 --lr_cross_disc 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

**What to watch in W&B:**
- `diagnostics/attn_maps` — attention heatmaps logged every `sample_every` epochs. Early epochs: uniform (expected). By epoch 20–30: cardiomegaly maps should begin concentrating over the central cardiac region; effusion maps over the lower lateral pleural angles.
- If both maps remain diffuse after epoch 50: disease signal is too weak — reduce `--weight_null` or `--sigma_inactive`.
- `cross_head_score` — should decrease faster than in non-attention runs due to structural routing

**Caveat — checkpoint incompatibility:** `--use_label_attention` changes the `head_cardiomegaly` and `head_effusion` parameter trees (adds `disease_query` and `key_proj`). Resuming from a non-attention checkpoint will warn of architecture mismatch. Do not resume; start fresh.

---

## 9. Phase 7 — Full SepVAE Sweeps with All Fixes

This is the current recommended training configuration combining all improvements from Phases 3–6. The goal is to satisfy the verification gates (Section 11) before proceeding to Strategy A LDM composition (Phase 8).

### Sweep matrix

| Config name | R1–R4 | R5a | R5b | R7 | GAN | Subpixel | Purpose |
|-------------|-------|-----|-----|----|-----|----------|---------|
| `baseline_fixed` | ✓ | — | — | — | — | — | Establish corrected baseline |
| `contrastive` | ✓ | ✓ | — | — | — | — | Test paired contrastive alone |
| `cross_adv` | ✓ | — | ✓ | — | — | — | Test cross-adversarial alone |
| `sharp_test` | ✓ | — | — | — | ✓ | ✓ | Test sharpness fixes alone |
| `attention` | ✓ | — | — | ✓ | — | — | Test spatial routing alone |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full combined stack |

Each run logs to W&B project `sepvae-disentangle`. After 100 epochs, compare:
- `probe_auc/cardiomegaly` and `probe_auc/effusion` (rolling 3-epoch average)
- `cross_head_score` at epoch 50 and epoch 100
- Attention heatmaps (for runs with R7)
- Reconstruction quality (visual + SSIM to be added)

---

## 10. Research Theory: Assumptions and Composition Strategies

*This section documents the fundamental assumptions of the project and the three candidate strategies for downstream composition. Read this before designing any LDM training experiment.*

### 10.1 Three core assumptions

#### A1 — Disease factors are approximately separable

$$z_{\text{cardiomegaly}} \perp z_{\text{effusion}}$$

In chest X-rays this is not strictly true. Cardiomegaly and effusion co-occur through shared pathology (right heart failure causes both), shared appearance patterns, and dataset sampling bias. `exclude_cross_disease_overlap=True` removes co-occurring patients from training but creates distribution mismatch at inference when we compose both.

**Evidence from runs:** Cross_head_score never below 0.74. Assumption is violated to a significant degree.

#### A2 — Radiologist labels correspond to clean generative factors

Cardiomegaly is not a primitive visual atom — it is a high-level finding that shifts the mediastinum, displaces the lungs, and creates global shape changes that look like "anatomy variation" to any encoder. The model routes some of this into $z_{\text{common}}$.

**Evidence from runs:** Cardiomegaly probe AUC is consistently lower and more volatile than effusion AUC.

#### A3 — Separated representation implies separable score fields

Even with clean factorised latents, separately trained LDMs will double-count shared content unless the composition strategy accounts for it:

$$p(z \mid c, e) \propto \frac{p(z \mid c)\, p(z \mid e)}{p(z)}$$

If each disease LDM was trained on the full $z$ (including $z_{\text{common}}$), both carry anatomical structure in their scores. Naive addition double-counts anatomy:

$$\nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) = \nabla_z \log p(z \mid c, e) + \nabla_z \log p(z)$$

The extra $\nabla_z \log p(z)$ over-reinforces common structure and produces physically inconsistent compositions.

---

### 10.2 Three candidate composition strategies

#### Strategy A — Sub-block conditional LDMs (RECOMMENDED)

$$p(z) = p(z_{\text{common}}) \cdot p(z_{\text{cardio}} \mid z_{\text{common}}) \cdot p(z_{\text{effusion}} \mid z_{\text{common}})$$

```
Inference (both diseases):

    LDM_common  ──→  z_common ──────────────────────┐
                         │                           │
                         ├──→ LDM_cardio(z_common) ──→ z_cardio   ──┐
                         │                                           │
                         └──→ LDM_effusion(z_common) → z_effusion ──┤
                                                                     │
                    Decoder(z_common, z_cardio, z_effusion) ←────────┘
                         │
                         ▼
                  Composed CXR (cardio + effusion)
```

Each LDM acts on a **different disjoint subspace** — no score addition, no double-counting. Composition is a sequential ancestral sample. Requires: $z_{\text{cardio}} \perp\!\!\!\perp z_{\text{effusion}} \mid z_{\text{common}}$ — exactly what R5b tests and enforces.

#### Strategy B — Full-z score composition (fallback)

$$\nabla_z \log p(z \mid c, e) \approx \nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) - \nabla_z \log p(z)$$

Requires three LDMs and near-perfect disentanglement. The prior subtraction corrects double-counting only if the disease LDMs are truly independent of each other's content in $z$.

#### Strategy C — Sub-block scores only (broken with current decoder)

Strategy C assumes the decoder treats each sub-block as an independent additive perturbation. The current SepVAE decoder concatenates all heads and applies shared convolutions — it mixes all channels. The decoder Jacobian is not block-diagonal. Strategy C is geometrically unsound for this architecture.

#### CFG with unconditional = common expert (Addition P4)

If a single conditional LDM is trained with CFG dropout, the unconditional direction = $z_{\text{common}}$-only output. Composition becomes:

$$\nabla_z \log p^{\text{guided}}(z \mid c, e) = \nabla_z \log p(z \mid \varnothing) + \lambda[\nabla_z \log p(z \mid c) - \nabla_z \log p(z \mid \varnothing)] + \lambda[\nabla_z \log p(z \mid e) - \nabla_z \log p(z \mid \varnothing)]$$

Double-counting is prevented: both guidance directions are subtracted against the same common base. Architecturally simpler than Strategy A (one LDM to train). Natural extension if a joint conditional LDM is trained as the ablation baseline.

| | Strategy A | Strategy B | Strategy C | CFG (P4) |
|--|------------|------------|------------|----------|
| Double-counts anatomy? | No | Requires prior subtraction | No (if block-diagonal decoder) | No |
| Compatible with current decoder | Yes | Yes | **No** | Yes |
| Recommended | **Yes** | Fallback | No | Yes (as variant) |

---

### 10.3 Revised, falsifiable research hypothesis

> **If disease-related variation can be approximately factorised from shared anatomical and acquisition variation in the SepVAE latent space** — as measured by probe AUC > 0.75 per head, cross-head score < 0.65, and edit purity ratio > 2.0 — **then conditional sequential sampling from disease-specific sub-block LDMs conditioned on $z_{\text{common}}$ (Strategy A) will better approximate multi-pathology comorbid synthesis**, as measured by:
> 1. Lower FID against held-out comorbid images vs. a jointly-trained conditional baseline
> 2. Higher dual-disease classifier confidence on synthesised images
> 3. Higher anatomical consistency (SSIM on non-disease regions vs. single-disease reference)

**Fallback claim (if cardiomegaly remains unfactorisable):**
Score composition achieves plausible multi-pathology generation even when factorisation is imperfect, whereas direct joint conditioning fails to generalise to the comorbid case due to data sparsity.

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

## 12. Future Roadmap: Phases 8–11

### Phase 8 — Strategy A LDM training (conditional on G1–G4 passing)

```
Train:
  LDM_common   — unconditional VP-SDE on z_common (4ch × 64×64)
  LDM_cardio   — VP-SDE on z_cardio (2ch × 64×64), conditioned on z_common
  LDM_effusion — VP-SDE on z_effusion (2ch × 64×64), conditioned on z_common
```

Conditioning on $z_{\text{common}}$: cross-attention in the ScoreNet UNet denoiser, or concatenation along the channel axis. Cross-attention is preferred (allows attending to specific spatial positions of $z_{\text{common}}$ that are relevant to disease localisation).

### Phase 9 — Composition evaluation

```bash
python run/compose_diseases.py \
  --strategy a \
  --vae_checkpoint runs_sepvae/sepvae_full/best_checkpoint \
  --ldm_common_checkpoint runs_ldm/common \
  --ldm_cardio_checkpoint runs_ldm/cardio \
  --ldm_effusion_checkpoint runs_ldm/effusion \
  --n_samples 1000 \
  --out_dir results/composed_samples/

python run/eval_fid.py \
  --generated results/composed_samples/ \
  --reference data/vinbig/comorbid_holdout/
```

### Phase 10 — Ablation baseline

Train a single jointly-conditional LDM (or VAE) with a 4-class label (normal, cardio, effusion, both). Compare FID and region SSIM against Strategy A. If jointly-conditional wins, the factorisation assumption (A1) is the bottleneck. If Strategy A wins, disentangled composition is adding value.

### Phase 11 — CFG composition variant (Addition P4)

Train one conditional LDM with CFG dropout (`p_uncond=0.1`). At inference, apply dual-condition CFG guidance. Compare against Strategy A — divergence is diagnostic of how much cross-head information the disease LDMs learn.

---

## 13. File and Checkpoint Registry

### Code files

| File | Role |
|------|------|
| `run/train_sep_vae.py` | Main SepVAE training script — all R1–R7 flags |
| `losses/sep_vae_losses.py` | All loss functions: `paired_contrastive_loss`, `CrossHeadDiscriminator`, `cross_head_disc_loss`, `sepvae_loss` |
| `models/sep_vae_jax.py` | SepVAE architecture: `SepVAEEncoder`, `SepVAEDecoder`, `SmoothUp`, `DiseaseAttentionHead` |
| `datasets/VinBigData.py` | Triplet dataloader with `exclude_cross_disease_overlap` |
| `utils/sepvae_diagnostics.py` | Diagnostics: `plot_latent_swap_grid`, `plot_per_channel_kl_heatmap`, PCA, attention maps |
| `scripts/preencode_sepvae_latents.py` | Pre-encoding pipeline for LDM training |
| `run/analyze_supervised_disease_axes.py` | Post-hoc analysis: probe classifier, cross_head_score |
| `slurm_scripts/preencode_sepvae.slurm` | SLURM script for pre-encoding |
| `slurm_scripts/sep_vae.slurm` | SLURM script for SepVAE training |
| `launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh` | R1+R2+R3+R4 launcher |
| `launchers/single_runs/vae/train_sep_vae_contrastive.sh` | R1–R4 + R5a launcher |
| `launchers/single_runs/vae/train_sep_vae_cross_adv.sh` | R1–R4 + R5b launcher |
| `launchers/single_runs/vae/train_sep_vae_full.sh` | All recommendations launcher |
| `launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh` | LDM (cardiomegaly sub-block) launcher |
| `launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh` | LDM (effusion sub-block) launcher |

### Analysis documents (this folder)

| File | Content |
|------|---------|
| `results/research_log/RESEARCH_LOG.md` | **This file** — master ordered research log |
| `results/sepvae_disentangle_analysis.md` | Full cross-run comparison table, per-group consensus, recommendations, sharpness and research theory (detailed supplementary reference) |
| `results/research_discussion.md` | Theory-first reference: architecture, loss formulations, strategy comparison, hypothesis — structured as a standalone document for writing/presentation |

### Key checkpoints

| Checkpoint | W&B | Description | Use |
|-----------|-----|-------------|-----|
| `runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl` | `laikh8dr` | disentangle-E ep180 — best usable checkpoint | LDM pre-encoding (Phase 4) |
| *(to be added)* | — | sepvae_full ep100 | Gate evaluation (G1–G4) |
| `preencoded_latents/disentangle_cardio/` | — | Pre-encoded $z_{\text{cardio}}$ from ep180 | LDM_cardio training |
| `preencoded_latents/disentangle_effusion/` | — | Pre-encoded $z_{\text{effusion}}$ from ep180 | LDM_effusion training |

---

*End of research log. Update this document whenever a new training run is launched or a significant finding is made. The goal is that anyone returning to this project after a gap — including the original authors — can reconstruct exactly why each decision was made and what it led to.*
