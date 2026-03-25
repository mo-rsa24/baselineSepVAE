# SepVAE — D3 Milestone Documentation

**Status:** Stop point. Reconstructions exceed expectations. This document captures everything that led to this result so work can resume from a fully understood state.

**Winning run:** `d3_gan_fix-20260325-143813`
**W&B:** https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk
**Git state:** `git checkout -b "d3_gan_fix-20260325-143813" 97f678113b2770417db1c126b839905103f1766c`

---

## Table of Contents

1. [What this model does](#1-what-this-model-does)
2. [Architecture](#2-architecture)
3. [Loss stack](#3-loss-stack)
4. [Curriculum — phase by phase](#4-curriculum--phase-by-phase)
   - [D0 — Smoke test](#d0--smoke-test)
   - [D1 — Reconstruction + bbox cross-attention](#d1--reconstruction--bbox-cross-attention)
   - [D2 — Perceptual sharpening + bbox attention loss](#d2--perceptual-sharpening--bbox-attention-loss)
   - [D3 — PatchGAN + TV (current milestone)](#d3--patchgan--tv-current-milestone)
5. [What failed before D3](#5-what-failed-before-d3)
6. [Hyperparameter rationale](#6-hyperparameter-rationale)
7. [Monitoring metrics and kill conditions](#7-monitoring-metrics-and-kill-conditions)
8. [Checkpoint registry](#8-checkpoint-registry)
9. [Recommendations for resuming](#9-recommendations-for-resuming)

---

## 1. What this model does

SepVAE is a disentangled variational autoencoder for chest X-rays (CXRs). It takes a paired batch of Normal and Cardiomegaly images and encodes each into two independent latent maps:

- **z_common** (16 channels × 16×16 spatial): anatomy shared by all images — lung texture, rib geometry, vertebral structure, soft tissue.
- **z_disease** (16 channels × 16×16 spatial): cardiac-specific shape information present only in Cardiomegaly images.

For Normal images, z_disease is **hard-zeroed** before decoding — the decoder is forced to reconstruct the full image from z_common alone, proving z_common has absorbed no cardiac-size information.

The goal is that z_disease encodes exactly and only the enlarged-heart signal, enabling:
- Counterfactual generation: swap z_disease between images to make a Normal CXR look cardiomegalic and vice-versa
- Downstream conditional diffusion modelling on a clean, disentangled latent space

---

## 2. Architecture

**File:** [models/sep_vae_v2.py](models/sep_vae_v2.py)

### 2.1 Encoder

```
Input (256×256×1 grayscale CXR)
    │
    ▼
ResNet50Scratch  — trained from scratch, GroupNorm throughout
    ├── Stem: 7×7 conv stride-2 → (128×128×64)
    ├── MaxPool stride-2         → (64×64×64)
    ├── Layer1: 3 BottleneckBlockGN, out=256  → (64×64×256)
    ├── Layer2: 4 BottleneckBlockGN, out=512  → (32×32×512)  ← h_layer2 for D7 skip
    └── Layer3: 6 BottleneckBlockGN, out=1024 → (16×16×1024)
                └── SelfAttention2D (4 heads) → h_shared (16×16×1024)
    │                                              │
    ▼                                              ▼
bg_branch (Layer4BranchGN)           tg_branch (Layer4BranchGN)
    3× BottleneckBlockGN stride=1        3× BottleneckBlockGN stride=1
    → (16×16×2048)                       → (16×16×2048)
    │                                              │
    ▼                                              ▼
ConvHeadGN                           BboxCrossAttnHead (D1+)
    → (μ_c, logvar_c)                    → (μ_d, logvar_d, attn_map)
    z_common (16ch × 16×16)              z_disease (16ch × 16×16)
```

**CBAM in every BottleneckBlockGN:**
Every residual block runs channel attention (avg+max pool → shared MLP → sigmoid) then spatial attention (channel-wise avg+max → 7×7 conv → sigmoid) before the residual add. This progressively focuses spatial features on cardiac contours across layers 1→4, so features reaching the disease head are already biased toward the heart boundary.

**SelfAttention2D at layer3 bottleneck (16×16):**
256 tokens — captures long-range cardiac-to-lung-field ratio that is the defining signal for cardiomegaly (CTR > 0.5). O(256²) = negligible cost.

**Layer4BranchGN — stride=1 throughout:**
Both branches stay at 16×16. The first block uses a projection shortcut (1024→2048) without spatial downsampling. This avoids the stride-2→bilinear-upsample round-trip that introduced aliasing into the latent space in earlier runs (feature frequencies above Nyquist are aliased by stride-2, and bilinear upsample cannot recover them).

**BboxCrossAttnHead (D1+ mode):**
For Cardiomegaly images, the ground-truth bbox [x0,y0,x1,y1] from VinBigData annotations defines a 2D Gaussian spatial prior centred on the heart (σ = bbox_width/4). This prior weights the encoder key vectors to form a spatial query Q_bbox, blended with a learned fallback query Q_learned via `bbox_query_mix`. The cross-attention map tells the disease head where to look from epoch 1 — bypassing the slow convergence of a purely learned query. Normal images always use Q_learned; stop_gradient prevents Normal images from training the key projection toward border-salient features and contaminating the disease head.

At D3: `bbox_query_mix=1.0` — pure Gaussian prior, the encoder is mature enough to not need the learned blend.

**Hard-zero nulling:**
After sampling z_c and z_d, z_d is multiplied by a label mask: 1.0 for Cardiomegaly, 0.0 for Normal. The decoder receives `concat(z_c, z_d_masked)` — for Normal images this is simply z_c padded with zeros.

### 2.2 Decoder

```
z_concat (16×16×32) = concat(z_c, z_d_masked)
    │
    ▼
z_proj: Conv(512, 3×3) → (16×16×512)
    │
    ▼ ── i=4: 16×16×512,  num_res_blocks ResBlockSE
    │
    SmoothUp → (32×32×512)   [bilinear resize + 2×conv3×3, no checkerboard]
    │
    ▼ ── i=3: 32×32×512,  num_res_blocks ResBlockSE
    │         SelfAttention2D (4 heads) — global cardiac silhouette coordination
    │
    SmoothUp → (64×64×256)
    │
    ▼ ── i=2: 64×64×256,  num_res_blocks ResBlockSE
    │
    SmoothUp → (128×128×128)
    │
    ▼ ── i=1: 128×128×128, num_res_blocks ResBlockSE
    │
    SmoothUp → (256×256×128)
    │
    ▼ ── i=0: 256×256×128, num_res_blocks ResBlockSE
    │
    GN → swish → Conv(1, 3×3) → sigmoid
    │
    ▼
Output (256×256×1)
```

**Channel schedule (finest→coarsest):** `(128, 128, 256, 512, 512)`
At 256×256 there are now 128 channels (doubled from the original 64). Earlier channel schedules produced blurry reconstructions because 64 channels at full resolution gave only 4 conv operations to generate the 1-channel output — a hard information bottleneck.

**ResBlockSE:** Each block runs GN→swish→conv3×3→GN→swish→conv3×3→SE gate. The Squeeze-and-Excitation gate (avg pool → FC → relu → FC → sigmoid, `se_reduction=8`) recalibrates channels before the residual add. When z_disease is non-zero, SE learns to amplify cardiac-silhouette channels and suppress lung/bone channels at every decoder level, making the disease latent→feature mapping explicit.

**SelfAttention2D at 32×32 in decoder:** Added at D5 (present in D3 codebase). 1024 tokens at 32×32. Lets the decoder coordinate the cardiac silhouette globally before upsampling to finer scales, preventing inconsistent left/right cardiac border rendering that local convolutions alone cannot enforce.

**SmoothUp:** bilinear resize followed by 2 conv3×3 layers. Eliminates the checkerboard artifacts that transposed convolutions produce.

**num_res_blocks:** Set to 3 (added in D2 from D1's 2). Flax names blocks by loop index so ResBlockSE_0 and ResBlockSE_1 loaded from the D1 checkpoint cleanly; ResBlockSE_2 initialised near-identity.

---

## 3. Loss stack

**File:** [losses/sep_vae_losses.py](losses/sep_vae_losses.py)

Three separate optimizers run alternately each iteration:

```
Total VAE loss:
  L_vae = w_rec·L_rec + β_c·KL_c + β_d·KL_d + κ·L_mi
        + w_bbox·L_bbox + w_perceptual·L_perceptual
        + w_gan·L_gan + w_tv·L_tv + w_masked_rec·L_masked_rec
        + w_supcon·L_supcon

FactorDisc loss (alternating):
  L_factor = BCE(D_factor(z_c, z_d), joint=1) + BCE(D_factor(z_c, z_d[perm]), marginal=0)

PatchGAN disc loss (alternating, phase-local start):
  L_patch = hinge(D_patch(x_real), D_patch(x_rec_stale))
```

### 3.1 Objective 1 — Orthogonal latent separation

| Loss | Formula | Purpose |
|------|---------|---------|
| **KL_common** | KL(q(z_c\|x) \|\| N(0,I)) | Regularise z_common toward standard prior |
| **KL_disease** | KL conditional on label: active→N(0,I), inactive→N(0,0.1²·I) | Tight prior for Normal images drives z_d→0, complementing hard-zero |
| **L_mi_factor** | E_q[logit(D_factor(z_c, z_d))] | FactorVAE-style: minimise TC(z_c; z_d), pushes toward product of marginals |
| **L_bbox_attn** | fraction of attn mass outside bbox (Cardio only) | Force disease attention head to stay inside annotated cardiac region |
| **L_supcon** | supervised contrastive on pooled z_d means | Pull same-class disease latents together, push Normal/Cardio apart |
| **L_masked_rec** | MSE outside bbox, z_d=0, Cardio only | Force z_common to reconstruct non-cardiac region without disease info |

**KL free bits = 0.5:** Per-dimension KL floor. Floors inactive dims preventing collapse, caps gradient of runaway dims — together bounds the step-to-step KL variance that caused the KL→11,600 spikes observed without it.

**sigma_inactive = 0.1:** For Normal images, the disease prior is N(0, 0.01·I). A 100× tighter prior strongly penalises μ_d away from zero. Combined with hard-zero nulling this creates two independent pressures on z_d for Normal images.

**D_acc = 0.50 (FactorVAE discriminator):** When the FactorVAE discriminator cannot distinguish the joint distribution from the product of marginals, it is at chance accuracy. This is the desired equilibrium — z_c and z_d are independent.

### 3.2 Objective 2 — Crisp reconstructions

| Loss | Formula | Purpose |
|------|---------|---------|
| **L_rec** | MSE(x_01, x_rec) | Pixel-level fidelity; hard-zero nulling makes this a clean signal for Normal |
| **L_perceptual** | L1 in frozen CheSS feature space (layers 1–2 only) | Mid-frequency texture sharpening without aliasing |
| **L_gan** | -mean(D_patch(x_rec)) hinge generator | PatchGAN adversarial pressure — sharpens fine texture (ribs, vessels, cardiac border) |
| **L_tv** | mean(|∂x/∂h| + |∂x/∂w|) anisotropic | Suppresses horizontal stripe artifacts from strided perceptual backbone gradients |

**Why layers 1–2 only for perceptual (not layers 3–4):**
Layer3 (stride=16 for 256px input) injects 16px-period gradients that manifest as visible banding. Layer4 (stride=32) is worse. Layer1 (stride=4) and Layer2 (stride=8) provide mid-frequency texture guidance without significant aliasing. Layer3 was the primary source of the grid/banding artifacts observed from D2 onward.

**PatchGAN (NLayerDiscriminator):**
The discriminator sees local patches rather than the full image. Its gradients sharpen local texture (rib cortex sharpness, vessel wall definition, cardiac border crispness) without globally pushing Normal anatomy toward the Cardiomegaly silhouette. Applied to the full batch (Normal + Cardiomegaly) because both are equally blurry.

---

## 4. Curriculum — phase by phase

The curriculum introduces objectives incrementally. Each stage adds exactly one new pressure to an already-stable model, so failures are diagnosable and regressions are traceable.

### D0 — Smoke test

**Purpose:** Verify the V2 pipeline runs end-to-end. No orthogonality pressure, no bbox, no perceptual.

**Launcher:** `PHASE=d0 sbatch slurm_scripts/sep_vae_v2.slurm`

**Epochs:** 5

| Parameter | Value |
|-----------|-------|
| use_bbox_cross_attn | False (DiseaseAttnHeadV2) |
| weight_rec | 1.0 |
| weight_kl_common | 1e-4 |
| weight_kl_disease | 5e-5 |
| weight_mi_factor | 0.0 |
| weight_bbox_attn | 0.0 |
| weight_perceptual | 0.0 |
| batch_size | 16 |
| kl_warmup_epochs | 0 |

**Checkpoint:** `runs_sepvae/d0_smoke_v2-20260324-063142/` _(most recent smoke run)_

**What to look for:** Loss decreasing from epoch 1. No NaN. Samples recognisably CXR-shaped within 3 epochs.

---

### D1 — Reconstruction + bbox cross-attention

**Purpose:** Train the backbone from scratch and learn to encode CXR features. Introduce BboxCrossAttnHead so the disease head localises the heart from epoch 1 via the Gaussian prior. No MI pressure yet — the discriminator needs a stable encoder to train against.

**Launcher:** `PHASE=d1 sbatch slurm_scripts/sep_vae_v2.slurm`

**Epochs:** 30 (KL warmup for first 5 epochs)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| use_bbox_cross_attn | True | Gaussian prior from bbox label active from epoch 1 |
| weight_rec | 1.0 | primary signal |
| weight_kl_common | 1e-4 | standard |
| weight_kl_disease | 5e-5 | lighter than common: disease head needs room to learn |
| weight_mi_factor | 0.0 | not yet — encoder not stable enough for discriminator |
| weight_bbox_attn | 0.0 | not yet — attention maps not reliable enough to supervise |
| weight_perceptual | 0.0 | not yet |
| batch_size | 16 | maximum possible without perceptual backbone in memory |
| kl_warmup_epochs | 5 | ramp KL from 0→1 over 5 epochs, prevents KL spike at startup |
| bbox_query_mix | 0.7 | 70% Gaussian prior, 30% learned fallback |
| decoder_res_blocks | 2 | baseline depth |

**Canonical checkpoint:** `runs_sepvae/d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl`
_(epoch 30)_

**What to look for:** Reconstructions anatomically correct by epoch 10. Attention maps beginning to localise over the cardiac region for Cardiomegaly by epoch 20.

---

### D2 — Perceptual sharpening + bbox attention loss

**Purpose:** Sharpen reconstructions with CheSS perceptual loss. Activate bbox attention supervision so the disease head is explicitly penalised for attending outside the annotated cardiac region. Add a 3rd ResBlockSE to the decoder for extra capacity.

**Launcher:** `sbatch slurm_scripts/d2_perceptual_bbox.slurm`

**Resumes from:** D1 final (`d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl`)

**Epochs:** 30 → 55 (+25 epochs)

| Parameter | D1 | D2 | Rationale |
|-----------|----|----|-----------|
| weight_bbox_attn | 0.0 | 0.05 | Light: first time active, don't shock the encoder |
| weight_perceptual | 0.0 | 0.15 | CheSS layers 1–3 at first; layer3 later excluded |
| weight_mi_factor | 0.0 | 1.0 | FactorVAE discriminator introduced |
| weight_kl_disease | 5e-5 | 5e-5 | keep |
| decoder_res_blocks | 2 | 3 | extra depth; ResBlockSE_2 initialised near-identity |
| batch_size | 16 | 6 | perceptual backbone + decoder depth raises peak memory |
| bbox_query_mix | 0.7 | 0.7 | keep |
| weight_masked_rec | 0.0 | 0.3 | outside-bbox MSE with z_d=0 forces z_common purity |
| weight_cardio_supcon | 0.0 | 0.05 | contrastive on pooled z_d means |

**Note on perceptual layers:** D2 originally used layers 1–3. Layer3 was later identified as the primary source of stripe artifacts (16px-period gradients at stride=16). The D3 launcher uses `--perceptual_only` which restricts to layers 1–2.

**Canonical checkpoint:** `runs_sepvae/d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl`
_(epoch 55)_

**What to look for:** Loss/perceptual decreasing. Attention mass inside bbox increasing (loss/bbox_attn decreasing toward 0.2). FactorDisc accuracy settling near 0.50. Samples sharper than D1 but may show faint banding from layer3 perceptual gradients.

---

### D3 — PatchGAN + TV (current milestone)

**W&B:** https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk
**Git:** `97f678113b2770417db1c126b839905103f1766c`
**Launcher:** `slurm_scripts/d3_gan_fix.slurm`

**Resumes from:** D2 final (`d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl`)

**Epochs:** 55 → 120 (+65 epochs)

**New at D3:**
- PatchGAN adversarial training (NLayerDiscriminator)
- Anisotropic TV loss
- Phase-local `gan_start_step` (critical bug fix)
- `bbox_query_mix=1.0` (pure Gaussian prior)
- Perceptual restricted to layers 1–2 only

#### Complete hyperparameter table

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | | |
| img_size | 256 | |
| z_channels_common | 16 | 16ch × 16×16 spatial |
| z_channels_disease | 16 | |
| attn_query_dim | 256 | |
| attn_heads | 4 | encoder bottleneck and decoder 32×32 |
| decoder_res_blocks | 3 | set in D2, carried forward |
| bbox_query_mix | 1.0 | pure Gaussian prior — encoder mature enough |
| bbox_dropout_prob | 0.3 | randomly drop bbox during training for robustness |
| **Loss weights** | | |
| weight_rec | 1.0 | MSE |
| weight_kl_common | 1e-4 | |
| weight_kl_disease | 5e-5 | reverted from 1e-4 (doubling caused stripe artifacts) |
| kl_free_bits | 0.5 | per-dim KL floor |
| weight_mi_factor | 1.0 | FactorVAE |
| weight_bbox_attn | 0.10 | raised from 0.05 — 120 ep of guidance, now stronger |
| weight_cardio_supcon | 0.05 | contrastive on z_d pooled means |
| weight_perceptual | 0.05 | layers 1–2 only; reduced from D2's 0.15 |
| weight_gan | **0.1** | PatchGAN hinge generator (was 0.5 in failed runs) |
| weight_tv | **0.005** | anisotropic TV (was 0.001 in failed runs) |
| weight_masked_rec | 0.3 | outside-bbox MSE with z_d=0 |
| sigma_inactive | 0.1 | tight prior for Normal z_disease |
| **GAN** | | |
| gan_start_step | 2000 | **phase-local** (critical fix — see §5) |
| lr_patch_disc | 1e-4 | |
| disc_r1_penalty | 0.0 | removed (was 10.0 in failed runs) |
| **Training** | | |
| batch_size | 6 | OOM constraint with decoder_res_blocks=3 |
| lr_vae | 1e-4 | |
| lr_disc | 1e-4 | FactorVAE discriminator |
| weight_decay | 1e-4 | |
| grad_clip | 1.0 | |
| seed | 0 | |

#### Timing of objectives within D3

```
Epoch 55 (resumed from D2):
    Active: L_rec, KL (β_c=1e-4, β_d=5e-5), L_mi, L_bbox (0.10), L_perceptual (0.05),
            L_masked_rec (0.3), L_supcon (0.05)
    Dormant: L_gan (weight=0.1 but gan_start_step not yet reached), L_tv (active from step 1)

Phase-local step ~2000 (≈ 5 epochs into D3, epoch ~60):
    → GAN activates: L_gan (weight=0.1) + L_patch disc (lr=1e-4)
    → Full loss stack now active

Epoch 120 (D3 end):
    All losses active. Discriminator at ~0.55–0.65 accuracy (healthy balance).
```

**What made this work (see §5 for what failed):**
- `weight_gan=0.1` keeps the GAN contribution below reconstruction at all times
- Phase-local `gan_start_step` ensures the discriminator starts fresh in each phase, not from a restored global step that immediately exceeds the threshold
- `disc_r1_penalty=0.0` lets the discriminator bootstrap normally before the generator has adapted
- `weight_tv=0.005` (5× previous) provides enough suppression to counteract perceptual stripe gradients
- Perceptual layers 1–2 only removes the dominant stripe source (layer3 stride-16 gradients)

**Checkpoint:** `runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/`
_(checkpoints saved every 5 epochs: epoch0060, 0065, 0070 ... in progress)_

---

## 5. What failed before D3

Understanding what was tried and why it failed is as important as knowing what works.

### Failed run: `d5_gan-20260323-042442` (catastrophic, epoch 4)

**Bug 1 — weight_gan=0.5:**
At epoch 140 of that run: `0.5 × 1.25 = 0.625` GAN contribution vs `1.0 × 0.135 = 0.135` reconstruction. A 4.6× imbalance. Adversarial gradients overwhelmed reconstruction and corrupted the decoder in 4 epochs. Samples devolved into textured noise.

**Bug 2 — global_step used for gan_start_step:**
The checkpoint restored `global_step ≈ 57,196` from D4. With `gan_start_step=2000`, the condition `global_step >= gan_start_step` was immediately True from step 1 of the new phase. The fresh discriminator (random initialisation) fired at full strength from the very first batch, reached 88% accuracy within 4 epochs, and overwhelmed the generator before it could adapt.

**Fix:** Train script now uses `phase_local_step = global_step - phase_start_global_step`. The 2000-step warmup is counted from the beginning of each phase, regardless of the accumulated global step.

### Failed run: `d5_gan_v2-20260323-085311` (stalled)

**Bug 3 — disc_r1_penalty=10.0 + lr_patch_disc=3e-5:**
R1 gradient penalty = γ/2 · E[||∇_x D(x_real)||²] prevents the discriminator from fitting high-frequency decision boundaries on real samples. At γ=10 and lr=3e-5, the discriminator was so constrained it could not escape its random-init loss regime. Generator loss was noisy with no consistent adversarial signal for 80+ epochs.

**Fix:** `disc_r1_penalty=0.0` and `lr_patch_disc=1e-4`.

### Artifact source: stripe banding (D2–D4)

CheSS perceptual loss at layers 1–3 with `weight_perceptual=0.15` injected 16px-period stripe gradients from layer3 (stride=16 for 256px input) with a gradient power that `weight_tv=0.001` could not suppress (estimated 300× power imbalance). The stripes were baked into the decoder weights across D2–D4.

D3 addresses this with:
1. Perceptual layers 1–2 only (remove layer3 entirely)
2. `weight_perceptual` reduced from 0.15 → 0.05
3. `weight_tv` raised from 0.001 → 0.005

Resuming from D2 (before the CheSS layer3-corrupted D4/D5 runs) was the correct choice — it avoided loading weights that had stripe artifacts baked in.

---

## 6. Hyperparameter rationale

### The load-bearing parameters (do not change without understanding this)

**`weight_gan = 0.1`**
The GAN contribution must remain smaller than reconstruction at all times during training. At D3 equilibrium: `0.1 × |l_gan| ≈ 0.1 × 0.5 = 0.05` vs `1.0 × l_rec ≈ 0.13`. GAN is a sharpening refinement, not a dominant objective. Going above 0.2 risks the imbalance that failed at 0.5.

**`gan_start_step = 2000` (phase-local)**
2000 steps ≈ 5 epochs at batch=6. This gives the VAE encoder/decoder time to stabilise before adversarial gradients arrive. The phase-local counting is essential — the fix in `train_sep_vae.py` subtracts `phase_start_global_step` so each resumed phase restarts this counter.

**`weight_tv = 0.005`**
Anisotropic TV penalises pixel-to-pixel differences. At 0.001 it was insufficient to counteract CheSS layer3 stripe gradients (estimated 300× imbalance). At 0.005 with layer3 disabled it provides adequate suppression without blurring fine detail.

**`weight_kl_disease = 5e-5`**
At 1e-4 (doubling) the disease KL weight was observed to produce stripe artifacts — likely by increasing pressure on z_d to be more diffuse, causing the decoder to spread disease information spatially. Reverted and kept at 5e-5.

**`sigma_inactive = 0.1`**
The tight prior N(0, 0.01·I) for Normal images is 100× tighter than the standard prior. Combined with hard-zero nulling, Normal images see two independent zero-driving pressures on z_d: (a) the prior penalises μ_d away from zero, (b) z_d is zeroed before decoding regardless. This double enforcement is what gives the model clean Normal reconstructions.

**`weight_kl_common = 1e-4`, `weight_kl_disease = 5e-5`**
Kept very small throughout the curriculum. The reconstructive and adversarial losses are orders of magnitude larger. KL is regularisation only — it should not be the dominant loss. Free bits at 0.5 prevent KL collapse without needing higher weights.

### The safe parameters (reasonable to adjust)

| Parameter | Current | Safe range | Notes |
|-----------|---------|-----------|-------|
| weight_bbox_attn | 0.10 | 0.05–0.20 | Higher → stronger spatial constraint but may conflict with GAN |
| weight_perceptual | 0.05 | 0.01–0.10 | Layers 1–2 only; above 0.10 may reintroduce stripe risk |
| weight_masked_rec | 0.3 | 0.1–0.5 | Raises decoder memory (extra forward pass with z_d=0) |
| weight_cardio_supcon | 0.05 | 0.01–0.1 | Contrastive is stable but low priority |
| lr_vae | 1e-4 | 5e-5–2e-4 | Lower for architectural changes (D7); keep for hyperparameter-only stages |

---

## 7. Monitoring metrics and kill conditions

| Metric | Healthy range | Kill if |
|--------|--------------|---------|
| `loss/reconstruction` | ≤ 0.005 and decreasing | > 0.010 and rising for 10 epochs |
| `loss/gan_g` | slowly decreasing (more negative) | strongly negative while recon is rising |
| `loss/tv` | decreasing over epochs | flat/rising → stripes persisting |
| `loss/bbox_attn` | ≤ 0.25 | rising > 0.40 → attention ignoring bbox |
| `metrics/patch_disc_acc` | 0.55–0.65 | > 0.80 sustained → VAE losing the adversarial game |
| `metrics/z_cardio_norm_ratio` | > 2.0 (active/inactive z_d norm) | < 1.5 → disentanglement degrading |
| `metrics/z_common_norm_ratio` | near 1.0 | > 1.5 or < 0.7 → z_common absorbing disease signal |
| Visual recon grid | sharp ribs/vessels, clean cardiac border | new artifact types (stripes, checkerboard, mode collapse) |

---

## 8. Checkpoint registry

| Phase | Exp name | Run dir | Final checkpoint | Epochs |
|-------|----------|---------|-----------------|--------|
| D0 | d0_smoke_v2 | `runs_sepvae/d0_smoke_v2-20260324-063142/` | `checkpoint_final.pkl` | 5 |
| D1 | d1_recon_bbox_xattn | `runs_sepvae/d1_recon_bbox_xattn-20260321-004241/` | `checkpoints/checkpoint_final.pkl` | 30 |
| D2 | d2_perceptual_bbox | `runs_sepvae/d2_perceptual_bbox-20260324-105108/` | `checkpoints/checkpoint_final.pkl` | 55 |
| D3 | d3_gan_fix | `runs_sepvae/d3_gan_fix-20260325-143813/` | in progress (every 5 ep) | 55→120 |

### Checkpoint chain

```
d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl  (D2 final, epoch 55)
    └── resumes into ──▶
d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch00XX.pkl       (D3 current)
```

**D1 is the canonical clean starting point.** If any future stage needs to restart the GAN curriculum from scratch, resume from D1 final (not D2), add perceptual layers 1–2 at weight 0.05, and proceed directly to D3 hyperparameters.

**D2 final is the correct resume for all further stages.** The D4/D5 checkpoints that exist in `runs_sepvae/` (e.g. `d4_mi_percep-20260322-222345`, `d5_recon-20260318-*`) were trained with layer3 perceptual at weight 0.15 and have stripe artifacts baked into the decoder weights. Do not resume from those.

---

## 9. Recommendations for resuming

### If continuing from D3 (adding D4, D5, D6, D7)

**Preserve these three parameters exactly:**
1. `weight_gan = 0.1` — do not increase without monitoring recon loss for 10 epochs
2. `gan_start_step` must remain **phase-local** in the training script — verify this before each new phase
3. `weight_tv = 0.005` — minimum required to suppress perceptual stripe gradients

**D4 considerations:**
- If adding loss weight adjustments only: low risk. Keep rec/GAN balance as above.
- If increasing `weight_perceptual` back above 0.05: run layer1–2 only and monitor TV loss.
- Do not increase `weight_kl_disease` above 5e-5 — 1e-4 was empirically linked to stripe artifacts.

**D5/D6 considerations:**
- The architecture fixes described in PLAN_D5.md are already implemented in the codebase at git `97f6781`. No architectural changes are needed for a hyperparameter-only D5.
- If activating a higher `weight_gan` (e.g. 0.2): monitor `metrics/patch_disc_acc` — if it exceeds 0.75 for more than 5 epochs, reduce immediately.

**D7 (UNet skip connections + z_common=32):**
- Highest regression risk of all planned stages
- `head_common` and `decoder/z_proj` will be re-initialised (z_common shape changes 16→32)
- Lower `lr_vae` to 5e-5 for the architectural transition
- `weight_perceptual = 0.0` is sensible for D7 (skip connections replace it)
- Expect 10–20 epoch transient regression before skip connections stabilise
- Do not judge D7 quality before epoch 20 from resume

### If starting fresh from this codebase

```bash
# Phase 1: D1 (train from scratch)
PHASE=d1 sbatch slurm_scripts/sep_vae_v2.slurm

# Phase 2: D2 (resume D1, add perceptual + bbox supervision + MI disc)
sbatch slurm_scripts/d2_perceptual_bbox.slurm

# Phase 3: D3 (resume D2, add PatchGAN + TV)
sbatch slurm_scripts/d3_gan_fix.slurm
```

### Git snapshot

The complete working state is:

```bash
git stash           # if any uncommitted local changes
git checkout 97f678113b2770417db1c126b839905103f1766c
```

Key files at this commit:
- [models/sep_vae_v2.py](models/sep_vae_v2.py) — full architecture
- [losses/sep_vae_losses.py](losses/sep_vae_losses.py) — all loss functions
- [run/train_sep_vae.py](run/train_sep_vae.py) — training loop with phase-local GAN start
- [slurm_scripts/d3_gan_fix.slurm](slurm_scripts/d3_gan_fix.slurm) — D3 launcher
- [slurm_scripts/d2_perceptual_bbox.slurm](slurm_scripts/d2_perceptual_bbox.slurm) — D2 launcher
- [slurm_scripts/sep_vae_v2.slurm](slurm_scripts/sep_vae_v2.slurm) — D0/D1 launcher
