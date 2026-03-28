# Training Curriculum

**Related documents:** [04 Model Architecture](04_model_architecture.md) | [05 Objective Functions](05_objective_functions.md) | [06 Failures & Debugging](06_failures_debugging.md) | [08 Current State (D3)](08_current_state_d3.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25

---

## 1. Curriculum Philosophy

The SepVAE training curriculum follows one rule: **introduce exactly one new pressure per stage, applied to an already-stable model.**

This is the direct lesson from V1's failures. When all objectives are added simultaneously (as in V1 Phase 1), a failure in any one objective produces ambiguous diagnostics — it is impossible to determine whether the NaN, the head collapse, or the cross-head leakage was caused by the nulling loss, the orthogonality loss, the MI discriminator, or their interaction. In contrast, when each stage adds a single new component:

1. Failures are unambiguous in their source
2. Regressions are traced to the one change made
3. The model that enters each stage is already verified to be stable

The cost of this approach is extra training time — 3 stages (D1, D2, D3) where V1 would train a single run. The benefit is diagnosable, reversible failures instead of total-loss situations.

### The transition criterion

Before advancing from stage N to stage N+1:
- The primary objective for stage N must be converging (loss decreasing over 5+ epochs)
- No new artifact types from stage N
- No metric regression beyond the expected transient (< 5 epochs at resumed training start)

If the transition criterion fails, the stage is debugged in isolation before proceeding. This is what happened with D3 — two failed GAN runs (B1, B2 in [Failures document](06_failures_debugging.md)) were diagnosed and fixed before the canonical D3 run was stable.

---

## 2. Curriculum Map

| Stage | Name | Epochs | What's new | Resume from |
|-------|------|--------|-----------|-------------|
| D0 | Smoke test | 5 | Nothing — pipeline verification | Fresh start |
| D1 | Reconstruction + bbox | 0→30 | BboxCrossAttnHead, KL warmup | Fresh start |
| D2 | Perceptual + MI + bbox attn | 30→55 | CheSS perceptual, FactorDisc, bbox_attn_loss, masked_rec, supcon | D1 final |
| D3 | PatchGAN + TV | 55→120 | PatchGAN adversarial, TV loss | D2 final |
| D4 (planned) | TBD | 120→? | Refinement of D3 objectives | D3 final |
| D5 (planned) | TBD | ?→? | Hyperparameter / objective tuning | D4 final |
| D6 (planned) | TBD | ?→? | TBD | D5 final |
| D7 (planned) | UNet skip + z_common=32 | ?→? | Skip connections, wider z_common | D6 final |

---

## 3. Stage-by-Stage Specification

### D0 — Smoke Test

**Purpose:** Verify the V2 pipeline runs end-to-end without crashing. Confirm data loading, model forward pass, loss computation, and checkpoint saving work on the target hardware.

**Not a training run** — 5 epochs is insufficient to learn anything useful. The checkpoint is not reused.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| use_bbox_cross_attn | False | Simpler architecture for initial test |
| weight_rec | 1.0 | Only reconstruction active |
| weight_kl_common | 1e-4 | Standard |
| weight_kl_disease | 5e-5 | Standard |
| weight_mi_factor | 0.0 | Not yet |
| weight_bbox_attn | 0.0 | Not yet |
| weight_perceptual | 0.0 | Not yet |
| batch_size | 16 | Maximum without perceptual backbone in memory |
| kl_warmup_epochs | 0 | Smoke only |

**Success criteria:** Loss decreases from epoch 1. No NaN. Samples are CXR-shaped (not random noise) within 3 epochs. W&B logs appear.

**Canonical run:** `d0_smoke_v2-20260324-063142`

---

### D1 — Reconstruction + BboxCrossAttn

**Purpose:** Train the full backbone (ResNet-50 from scratch) and establish the reconstruction baseline. Introduce BboxCrossAttnHead so the disease head localises the heart from epoch 1 via the Gaussian prior. No MI pressure yet — the FactorDisc needs a stable encoder to train against.

**What's added vs D0:**
- BboxCrossAttnHead (replaces placeholder DiseaseAttnHeadV2)
- KL warmup for 5 epochs (prevents KL spike at startup)
- `bbox_query_mix = 0.7` (70% Gaussian prior, 30% learned)

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| use_bbox_cross_attn | True | Gaussian prior from epoch 1 |
| weight_rec | 1.0 | Primary signal |
| weight_kl_common | 1e-4 | Standard |
| weight_kl_disease | 5e-5 | Lighter than common — disease head needs room |
| weight_mi_factor | 0.0 | Not yet — encoder not stable enough |
| weight_bbox_attn | 0.0 | Not yet — attention maps not reliable enough to supervise |
| weight_perceptual | 0.0 | Not yet |
| weight_masked_rec | 0.0 | Not yet |
| weight_cardio_supcon | 0.0 | Not yet |
| kl_free_bits | 0.5 | Per-dim KL floor — prevents collapse without blocking gradients |
| batch_size | 16 | Maximum without perceptual backbone in memory |
| kl_warmup_epochs | 5 | Ramp KL from 0→1 over 5 epochs |
| bbox_query_mix | 0.7 | 70% Gaussian prior, 30% learned fallback |
| decoder_res_blocks | 2 | Baseline depth |
| epochs | 30 | Sufficient to establish stable reconstruction |

**Why not activate bbox_attn_loss in D1?**
The attention maps need to be trained before they can be supervised. In epoch 1, the cross-attention map is driven primarily by the Gaussian prior (bbox_query_mix=0.7), which means the maps are approximately correct. But the key projection hasn't learned to produce useful cardiac-region features yet. Supervising random-initialisation maps with a loss would penalise the model for noise, not for meaningful spatial drift. D2 is the right time to add supervision once the key projection has seen 30 epochs of reconstruction gradient.

**Why not activate MI pressure in D1?**
The FactorVAE discriminator trains alongside the encoder. If the encoder is still in the rapid early-learning phase (epochs 0–10 of training from scratch), the discriminator cannot learn a stable decision boundary — the encoder features are changing too fast. Starting MI pressure at D2 (epoch 30) gives the encoder time to settle.

**Success criteria:**
- Reconstruction quality: anatomically correct at epoch 10 (lungs, ribs, cardiac silhouette shape plausible)
- Attention maps: partial cardiac concentration visible by epoch 20
- No NaN, no OOM

**Canonical checkpoint:** `runs_sepvae/d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl`

---

### D2 — Perceptual Sharpening + Bbox Attn + MI

**Purpose:** Sharpen reconstructions using CheSS perceptual loss (layers 1–2 only, not layer3). Activate bbox attention supervision so the disease head is explicitly penalised for wandering. Introduce MI discriminator (FactorVAE) to push z_common and z_disease toward independence. Add a 3rd ResBlockSE to the decoder for extra capacity. Add masked reconstruction for z_common purity. Add supervised contrastive loss for z_disease discriminability.

**Resumes from:** D1 final (`d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl`, epoch 30)

**What's added vs D1:**

| Added objective | Weight | Purpose |
|----------------|--------|---------|
| L_perceptual (CheSS layers 1–2) | 0.05 (was 0.15 originally → caused stripes; see [Failure C1](06_failures_debugging.md#c1--horizontal-16px-stripe-banding-d2-d4)) | Mid-frequency texture sharpening |
| L_mi_factor (FactorVAE disc) | 1.0 | Push z_common ⊥ z_disease |
| L_bbox_attn | 0.05 (raised to 0.10 in D3) | Force disease attention to cardiac region |
| L_masked_rec | 0.3 | Verify z_common reconstructs non-cardiac correctly with z_d=0 |
| L_supcon | 0.05 | Supervised contrastive on pooled z_d means |
| decoder_res_blocks: 2 → 3 | — | Extra decoder capacity; block_2 initialised near-identity |

**Configuration (full D2 hyperparameter table):**

| Parameter | D1 | D2 | Rationale for change |
|-----------|----|----|---------------------|
| weight_bbox_attn | 0.0 | **0.05** | First time active; light to avoid shocking encoder |
| weight_perceptual | 0.0 | **0.05** | CheSS layers 1–2 only; reduced from original 0.15 that caused stripes |
| weight_mi_factor | 0.0 | **1.0** | FactorVAE discriminator introduced |
| weight_masked_rec | 0.0 | **0.3** | Outside-bbox MSE with z_d=0 |
| weight_cardio_supcon | 0.0 | **0.05** | Contrastive on z_d pooled means |
| decoder_res_blocks | 2 | **3** | Extra depth; block_2 initialised near-identity for clean resume |
| batch_size | 16 | **6** | Perceptual backbone + extra decoder depth raises peak memory |
| bbox_query_mix | 0.7 | 0.7 | Unchanged |
| kl_warmup_epochs | 5 | **0** | KL already warmed up from D1 |
| epochs | 30 | **55** (+25 epochs) | |

**Why decoder_res_blocks increases from 2 to 3 here?**
The new perceptual + MI objectives make the decoder generate more complex features. At res_blocks=2, the 16×16 stage has only 4 conv operations total — insufficient to decode the additional structure. ResBlockSE_2 is freshly initialised (near-identity) — its parameters are small, so it doesn't disrupt the weights loaded from D1.

**Note on D2 perceptual layers:**
The original D2 launcher used CheSS layers 1–3. Layer3 was subsequently identified as the dominant source of stripe artifacts (see [Failure C1](06_failures_debugging.md)). The canonical D2 configuration listed here uses layers 1–2 only (`--perceptual_only`), which is what the D3 launcher also uses when resuming.

**Success criteria:**
- loss/perceptual decreasing
- loss/bbox_attn decreasing from ~0.65 toward ≤ 0.30 by epoch 55
- FactorDisc accuracy settling near 0.52–0.58 (near-ideal 0.50)
- loss/masked_rec decreasing from ~0.08 toward ~0.02
- No new artifact types (stripes, checkerboard)

**Canonical checkpoint:** `runs_sepvae/d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl`

---

### D3 — PatchGAN + TV (Current Milestone)

**Purpose:** Add adversarial sharpening via PatchGAN discriminator and anisotropic TV loss to suppress residual stripe artifacts. Set `bbox_query_mix=1.0` (pure Gaussian prior — the encoder is now mature enough). Restrict perceptual loss to layers 1–2 only and reduce its weight.

**Resumes from:** D2 final (epoch 55)

**What's added vs D2:**

| Added/changed | Value | Purpose |
|--------------|-------|---------|
| L_gan (PatchGAN, hinge) | weight=0.1 | Adversarial sharpening — ribs, vessels, cardiac border |
| L_tv (anisotropic TV) | weight=0.005 | Suppress stripe artifacts from perceptual gradients |
| bbox_query_mix | 0.7 → **1.0** | Pure Gaussian prior — encoder mature, learned blend not needed |
| weight_perceptual | 0.05 → **0.05** (layers 1–2 only via `--perceptual_only`) | Reduce aliasing risk |
| weight_bbox_attn | 0.05 → **0.10** | 120 epochs of guidance; stronger spatial constraint now |
| weight_kl_disease | 5e-5 → **5e-5** | Unchanged (1e-4 tested and caused stripes — see [Failure C4](06_failures_debugging.md)) |
| batch_size | 6 → **6** | Unchanged |

**GAN timing (phase-local — critical):**
```
Epoch 55 (D3 start):
    Active: L_rec, KL, L_mi, L_bbox, L_perceptual, L_masked_rec, L_supcon, L_tv
    Dormant: L_gan (weight=0.1 but phase-local step < 2000)

Phase-local step ~2000 (≈ epoch 60):
    → GAN activates: NLayerDiscriminator fires
    → Full loss stack active
    → Discriminator warm-up period begins
```

`gan_start_step=2000` at batch_size=6 ≈ `2000 / (len(dataset)/6)` = approximately 5 epochs of warm-up in D3. During this warm-up, the encoder/decoder stabilise with the new TV loss before adversarial gradients arrive.

**Complete hyperparameter table (D3 final configuration):**

| Parameter | Value |
|-----------|-------|
| img_size | 256 |
| z_channels_common | 16 (16ch × 16×16) |
| z_channels_disease | 16 |
| attn_query_dim | 256 |
| attn_heads | 4 |
| decoder_res_blocks | 3 |
| bbox_query_mix | **1.0** |
| bbox_dropout_prob | 0.3 |
| **Loss weights** | |
| weight_rec | 1.0 |
| weight_kl_common | 1e-4 |
| weight_kl_disease | **5e-5** |
| kl_free_bits | 0.5 |
| weight_mi_factor | 1.0 |
| weight_bbox_attn | **0.10** |
| weight_cardio_supcon | 0.05 |
| weight_perceptual | 0.05 (layers 1–2 only) |
| weight_gan | **0.1** |
| weight_tv | **0.005** |
| weight_masked_rec | 0.3 |
| sigma_inactive | 0.1 |
| **GAN** | |
| gan_start_step | 2000 (phase-local) |
| lr_patch_disc | 1e-4 |
| disc_r1_penalty | 0.0 |
| **Training** | |
| batch_size | 6 |
| epochs | 120 (55→120) |
| lr_vae | 1e-4 |
| lr_disc | 1e-4 (FactorVAE discriminator) |
| weight_decay | 1e-4 |
| grad_clip | 1.0 |
| seed | 0 |

**Launcher:** `slurm_scripts/d3_gan_fix.slurm`
**W&B:** https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk
**Canonical checkpoint:** `runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/` (every 5 epochs)

**What not to change when resuming from D3 for future phases:**
1. `weight_gan = 0.1` — do not increase without monitoring recon loss for 10 epochs
2. `gan_start_step` must remain **phase-local** — verify in training script before each new phase
3. `weight_tv = 0.005` — minimum for stripe suppression
4. `weight_kl_disease = 5e-5` — empirically linked to stripe artifacts at 1e-4

---

## 4. How Objectives Are Accumulated

The diagram below shows which objectives are active at each stage:

```
                  D0   D1   D2   D3
                  ─────────────────
L_rec             ✓    ✓    ✓    ✓
KL (common)       ✓    ✓    ✓    ✓
KL (disease)      ✓    ✓    ✓    ✓
BboxCrossAttn     –    ✓    ✓    ✓     (architecture, not a loss)
L_mi_factor       –    –    ✓    ✓
L_bbox_attn       –    –    ✓    ✓
L_perceptual      –    –    ✓    ✓     (layers 1-2)
L_masked_rec      –    –    ✓    ✓
L_supcon          –    –    ✓    ✓
L_gan             –    –    –    ✓     (phase-local start step)
L_tv              –    –    –    ✓
```

Each row added at a stage remains active for all subsequent stages. The curriculum is strictly additive.

---

## 5. Stage Transition Criteria

Use these criteria before advancing to the next stage. Do not advance until all criteria are met.

### D0 → D1 criteria
- [ ] Loss decreases from epoch 1 (not flat or rising)
- [ ] No NaN in 5 epochs
- [ ] Samples recognisably CXR-shaped (not random noise)

### D1 → D2 criteria
- [ ] Reconstruction at epoch 30 is anatomically correct (lungs, ribs, cardiac outline)
- [ ] Attention maps show partial cardiac concentration for Cardiomegaly images by epoch 20
- [ ] No OOM, no NaN

### D2 → D3 criteria
- [ ] loss/perceptual decreasing
- [ ] loss/bbox_attn < 0.30 at epoch 55
- [ ] FactorDisc accuracy in 0.50–0.60 range
- [ ] loss/masked_rec < 0.03 at epoch 55
- [ ] No new artifact types vs D1 (stripe banding was introduced in the original D2 — see [Failure C1](06_failures_debugging.md))

### D3 → D4 criteria
- [ ] All losses stable (non-increasing) at epoch 120
- [ ] patch_disc_acc in 0.55–0.65 range
- [ ] loss/bbox_attn ≤ 0.20
- [ ] No new artifact types
- [ ] Visual inspection: ribs, vessels, cardiac border all sharper than D2

---

## 6. Checkpoint Integrity Notes

**Resuming between stages:** Each stage resume loads the previous stage's `checkpoint_final.pkl` via `--resume`. The checkpoint system merges parameters by name — any new parameter (e.g., `ResBlockSE_2` added in D2, `NLayerDiscriminator` added in D3) that does not exist in the previous checkpoint is initialised near-identity.

**Do not resume from D4/D5 stale runs:** The checkpoint directory contains runs from failed D4/D5 experiments (`d4_mi_percep-20260322-222345`, `d5_recon-20260318-*`) that have stripe artifacts baked into the decoder weights from CheSS layer3 perceptual gradients. The D2 canonical checkpoint (`d2_perceptual_bbox-20260324-105108`) is the clean starting point — all D3+ phases resume from it.

**The clean checkpoint chain:**
```
d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl  (epoch 30)
    └── resumes ──▶
d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl   (epoch 55)
    └── resumes ──▶
d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch01XX.pkl        (epoch 55→120)
```

If D4+ needs to restart from scratch, use D1 final — not D2 — if the objective stack differs significantly from the D2→D3 curriculum (to avoid carrying any D2 artifact bias).

---

*End of document. Continue to [04 Model Architecture](04_model_architecture.md).*
