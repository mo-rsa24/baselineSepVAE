# 01 — Experiment Timeline

**Cross-references:** [06 Failures](06_failures_debugging.md) · [07 Results](07_results_evaluation.md) · [08 Current State](08_current_state_d3.md)
**Last updated:** 2026-03-25

---

## How to read this document

Each entry follows the structure:
- **What changed** (relative to previous phase)
- **Why** (hypothesis or decision rationale)
- **Outcome** (what was observed)
- **What it drove** (what decision came next)

Entries are grouped into two eras separated by the V2 architectural rewrite.

---

## Era 1 — V1 Architecture (CheSS Backbone, Feb–Mar 2026)

### Context

The V1 architecture used a **frozen CheSS backbone** (a ResNet-50 pretrained on chest X-rays for pathology detection) as the shared encoder trunk. Two disease-specific ConvHeads projected from the 64×64 feature map to z_cardio and z_effusion. The model operated on 512×512 images (later changed to 256×256 in V2).

The V1 work addressed a binary task: disentangle cardiomegaly and pleural effusion into separate latent codes so that each could be independently manipulated.

---

### Phase 1 — Initial Disentangle Runs (Feb 16–17, 2026)

**W&B group:** `sepvae-disentangle`
**Runs:**

| Run ID | W&B | Epochs | Status |
|--------|-----|--------|--------|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | 192 | Complete (old script, no probe AUC) |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | 1 | CRASHED at step 0 |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | 84 | Stopped → resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | 101 | Stopped → resumed as E |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | 200 | **NaN crash at final epoch** |

> Runs C, D, E are the same continuous training, checkpointed and resubmitted across three SLURM jobs. Together they represent one 200-epoch chain.

**Key configuration:**
```
free_bits=1.0, sigma_inactive=0.1, weight_null=0.01, weight_orthogonality=0.03
weight_mi=0.003, weight_perceptual=0.03, use_fpn=True
batch_size=6, lr_vae=6.7e-5, kl_warmup_epochs=30, epochs=200
```

**What changed:** First full run of the V1 architecture from scratch.

**Outcome:**
- Peak probe AUC ≈ 0.774 (mean cardiomegaly + effusion) at epoch ~185 — best of all V1 runs
- Cross-head leakage score: 0.85–1.01 throughout — leakage never solved
- NaN explosion at epoch 200 — total loss was stable (0.004–0.006) through epoch 199, then all terms simultaneously became NaN
- No learning rate schedule → accumulated gradient errors + unlucky batch = catastrophic update

**What it drove:** Need for LR decay (→ R3), diagnosis of universal leakage problem (→ R5a, R5b)

---

### Phase 2 — Targeted Hyperparameter Sweeps (Feb 20, 2026)

Two sweeps designed to isolate specific hypotheses from Phase 1 failures.

#### Sweep A: Inactivity-Driven

**Run:** `sepvae_inactivity_driven-20260220-085943` | W&B `9lj20so0`
*(Run `62635` / W&B `99s5jhqe` crashed at ~93s — transient GPU conflict; relaunched 23 min later)*

**Hypothesis:** Tighter inactive prior (σ=0.05 → KL_inactive ≈ 2.5 nats) + stronger nulling (0.01→0.05) will reduce cross-head leakage.

**Key config diff from Phase 1:**
```
sigma_inactive: 0.1 → 0.05    (KL_inactive 1.8 → 2.5 nats)
weight_null:    0.01 → 0.05   (5× stronger)
use_fpn:        True → False
batch_size:     6 → 10
```

**Outcome:**
- Best leakage suppression of all V1 runs: cross_head_score = 0.744 at best
- Cardiomegaly probe AUC COLLAPSED to 0.476 (sub-random) by epoch 100
- Root cause: strong nulling drives μ_cardio → 0 faster than reconstruction gradient can maintain cardiomegaly signal; cardiomegaly is subtler/more distributed than effusion

**Lesson:** Cannot reduce leakage at the cost of one head dying. → R4 (floor active KL)

#### Sweep B: Independence-Driven

**Run:** `sepvae_independence_driven-20260220-085943` | W&B `41nce8qq`
*(Run `62742` / W&B `i9cevm1x` crashed at ~91s; same transient GPU issue)*

**Hypothesis:** Stronger orthogonality (0.03→0.1) + MI (0.003→0.01) will enforce independence without head collapse.

**Critical mistake:** `free_bits=2.0` with `sigma_inactive=0.1` → KL_inactive ≈ 1.8 < free_bits=2.0 → **disease heads in permanent gradient dead zone**.

**Outcome:**
- Disease head μ-norms collapsed to 0.025–0.030 (essentially zero) by epoch 100
- KL uniformly 1.7–1.8 nats (no class differentiation)
- Cardiomegaly probe AUC = 0.532 (near chance)
- Worst configuration of all V1 runs

**Lesson:** `KL_inactive < free_bits` is a hard failure mode — not a soft tradeoff. → R1 (remove free_bits)

#### Cross-run patterns identified after Phase 2

1. **free_bits / sigma_inactive conflict** is a hard binary failure: if KL_inactive < free_bits, all disease heads die
2. **Cross-head leakage** is universally unsolved: no run achieves cross_head_score < 0.74
3. **Cardiomegaly is consistently harder** than effusion across all configurations
4. **Probe AUC is too noisy** for single-epoch readings: same checkpoint varies 0.53–0.77 across adjacent evaluations
5. **Long training improves peak but destabilises** without LR decay

---

### Phase 3 — Systematic Fixes R1–R7 (Mar 5–15, 2026)

Seven targeted fixes derived from Phase 1–2 diagnoses. Implemented cumulatively.

| Fix | Problem | Change |
|-----|---------|--------|
| R1 | free_bits dead zone | `free_bits=0.0` + `sigma_inactive=0.05` |
| R2 | Regularisation imbalance | `weight_null=0.02`, `weight_orthogonality=0.02` |
| R3 | NaN at long training | Cosine LR decay from epoch 60 onward (→10% of initial) |
| R4 | Cardiomegaly head collapse | Floor active-head KL ≥ 2.0 nats for active-label samples |
| R5a | Marginal-only independence | Paired contrastive loss (sep-push + blind-pull) |
| R5b | Conditional leakage | Cross-head adversarial MLPs (D_{c→e}, D_{e→c}) |
| R6 | FPN overhead | Remove FPN — same AUC without the compute cost |
| R7 | Undirected spatial attention | BboxCrossAttnHead — bbox Gaussian prior guides disease head from epoch 1 |

See [06 Failures](06_failures_debugging.md) for full diagnosis of each and [05 Objective Functions](05_objective_functions.md) for mathematical formulations.

---

### Phase 4 — LDM Proof of Concept (Mar 15–17, 2026)

**Goal:** Verify that unconditional latent diffusion on z_common and z_disease sub-blocks was feasible before investing in a full conditional LDM.

**Approach:** Train small VP-SDE score networks on pre-encoded latents from a V1 checkpoint. Two sub-block LDMs trained independently (z_common, z_cardio).

**Outcome:** POC succeeded — LDMs converged on both sub-blocks. This confirmed that the latent space was structured enough to be modelled by diffusion. See [research_log/07_ldm_proof_of_concept.md](../research_log/07_ldm_proof_of_concept.md) for details.

---

### Phase 5 — Reconstruction Sharpness Investigation (Mar 15–19, 2026)

**Observation:** All V1 reconstructions were consistently soft and blurry — fine rib cortex, vessel walls, air-bronchogram texture absent.

**Root causes identified:**
1. Frozen CheSS backbone (stride=32) discards high-frequency texture before encoder heads
2. 8-channel bottleneck = 0.78% compression ratio → decoder hallucinating most content
3. L2 loss optimal solution = posterior mean = blurred average
4. SmoothUp = bilinear + 2×conv3×3 = 6 successive low-pass operations
5. No discriminator (weight_adversarial=0.0)

**Fixes identified (but not yet applied in V1):**
- PatchGAN adversarial loss (highest impact)
- Subpixel upsampling
- Higher perceptual weight

**Critical realisation:** The frozen CheSS backbone (stride=32) is a hard ceiling on reconstruction quality. Even with all sharpening fixes, sub-rib-level detail cannot be recovered. This led directly to the V2 rewrite decision.

---

### Phases 6–7 — Full Sweeps with All Fixes (Mar 15–19, 2026)

**Runs:** `sepvae_baseline_fixed` (R1–R4), `sepvae_full` (R1–R7), `sepvae_full_ortho-20260315-220524`

Best V1 result with all fixes:
- Silhouette score (disease-only PCA): 0.40–0.50
- FactorVAE discriminator accuracy: ~0.50 (equilibrium)
- Reconstruction still blurry (backbone ceiling)

See [research_log/08_full_sweeps.md](../research_log/08_full_sweeps.md) for detailed sweep tables.

---

## V2 Rewrite Decision (Mar 19, 2026)

**Decision:** Replace frozen CheSS backbone with ResNet-50 trained from scratch.

**Reasons:**
1. CheSS stride=32 is an architectural ceiling on reconstruction quality — cannot recover sub-rib detail regardless of loss changes
2. CheSS BatchNorm is incompatible with small generative batches (batch=6) — produces unstable running statistics
3. CheSS was trained for classification, not for separating cardiac from anatomical features — its features are not structured around the cardiomegaly/anatomy distinction
4. CheSS perceptual loss introduced 16px-period stripe artifacts (stride=16 layer3 gradients) when used at weight ≥ 0.1
5. GroupNorm is strictly better for small-batch generative training than BatchNorm

**Key architectural changes at V2:**
- ResNet-50 from scratch (GroupNorm throughout, no pretrained weights)
- CBAM in every BottleneckBlockGN
- Self-attention at layer3 bottleneck (16×16 for 256px input)
- BboxCrossAttnHead replaces DiseaseAttnHeadV1 (bbox Gaussian prior from day 1)
- SE-gated decoder (ResBlockSE) replaces plain ResBlock
- SmoothUp retained (no checkerboard)
- Hard-zero nulling retained

**V1 → V2 scope change:** From binary (cardiomegaly + effusion) to binary V2 (Normal vs. Cardiomegaly only) for focused iteration. Effusion head to be re-added once V2 architecture is validated.

---

## Era 2 — V2 Architecture (ResNet-50 from Scratch, Mar 2026)

### D0 — Smoke Tests (Mar 22–24, 2026)

**Purpose:** Verify V2 pipeline end-to-end. No orthogonality pressure, no bbox, no perceptual.

**Runs (selected):**

| Run | Date | Status |
|-----|------|--------|
| `d0_smoke_v2-20260322-075129` | Mar 22 | OOM at first JIT step |
| `d0_smoke_v2-20260322-080041` | Mar 22 | Batch size fixed — passed |
| `d0_smoke_v2-20260324-045917` | Mar 24 | Confirmed full pipeline |
| `d0_smoke_v2-20260324-063142` | Mar 24 | **Canonical D0** |

**Config:**
```
use_bbox_cross_attn=False (DiseaseAttnHeadV2)
weight_rec=1.0, weight_kl_common=1e-4, weight_kl_disease=5e-5
weight_mi=0.0, weight_bbox_attn=0.0, weight_perceptual=0.0
batch_size=16, epochs=5
```

**Outcome:** Pipeline confirmed. Loss decreasing from epoch 1. Samples recognisably CXR-shaped within 3 epochs.

---

### D1 — Reconstruction + BboxCrossAttn (Mar 21, 2026)

**Canonical run:** `d1_recon_bbox_xattn-20260321-004241`
**Checkpoint:** `runs_sepvae/d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl`

> Note: D1 predates some D0 smoke runs due to parallel exploration. The canonical D0 and D1 runs were confirmed in sequence; the date ordering reflects separate execution streams.

**Preceding failed D1 attempts:**

| Run | Date | Failure reason |
|-----|------|----------------|
| `d1_recon_bbox_xattn-20260318-184639` | Mar 18 | OOM |
| `d1_recon_bbox_xattn-20260318-185502` | Mar 18 | SLURM node issue |
| `d1_recon_bbox_xattn-20260319-074124` | Mar 19 | Config mismatch |
| `d1_recon_bbox_xattn-20260321-000421` | Mar 21 | Checkpoint key mismatch |
| `d1_recon_bbox_xattn-20260321-000738` | Mar 21 | Resume path error |
| `d1_recon_bbox_xattn-20260321-002801` | Mar 21 | BatchNorm/GroupNorm conflict in resume |

**What changed from D0:**
- `use_bbox_cross_attn=True` (BboxCrossAttnHead with Gaussian prior)
- `kl_warmup_epochs=5`
- `decoder_res_blocks=2` (default)
- `bbox_query_mix=0.7` (70% Gaussian prior, 30% learned fallback)

**Config:**
```
epochs=30, kl_warmup_epochs=5, batch_size=16
weight_rec=1.0, weight_kl_common=1e-4, weight_kl_disease=5e-5
weight_mi=0.0, weight_bbox_attn=0.0, weight_perceptual=0.0
```

**Outcome:** Reconstruction loss converging. Attention maps beginning to localise over cardiac region by epoch 20. No MI pressure yet — discriminator not trained.

**What it drove:** Need to add perceptual sharpening, MI disc, and bbox supervision (→ D2)

---

### D2 — Perceptual + MI Disc + Bbox Attention Loss (Mar 24, 2026)

**Canonical run:** `d2_perceptual_bbox-20260324-105108`
**Checkpoint:** `runs_sepvae/d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl`
**Epoch range:** 30 → 55 (+25 epochs from D1 final)

**Preceding D2 attempts:**

| Run | Date | Issue |
|-----|------|-------|
| `d2_perceptual_bbox-20260322-045255` | Mar 22 | weight_perceptual=0.15 (too high), layer3 stripes appear |
| `d2_perceptual_bbox-20260322-045648` | Mar 22 | OOM after adding perceptual |
| `d2_perceptual_bbox-20260322-050230` | Mar 22 | Batch size fix, ran but stripes worsened |
| `d2_perceptual_bbox-20260322-074027` | Mar 22 | MI disc destabilised recon |
| `d2_perceptual_bbox-20260322-120754` | Mar 22 | Perceptual weight reduced to 0.10 — stripes persist |
| `d2_perceptual_bbox-20260323-043904` | Mar 23 | Layer3 excluded — stripes reduced |
| `d2_perceptual_bbox-20260324-084547` | Mar 24 | Marginal stripe remaining |
| `d2_perceptual_bbox-20260324-105108` | Mar 24 | **Canonical D2** — acceptable quality |

**What changed from D1:**
```
decoder_res_blocks: 2 → 3    (extra depth; ResBlockSE_2 freshly init'd near-identity)
weight_bbox_attn:   0.0 → 0.05
weight_perceptual:  0.0 → 0.15 (early) → 0.05 (canonical)
weight_mi_factor:   0.0 → 1.0   (FactorVAE disc introduced)
weight_masked_rec:  0.0 → 0.3
weight_cardio_supcon: 0.0 → 0.05
batch_size: 16 → 6  (perceptual backbone + decoder depth raises peak memory)
```

**Outcome:** Reconstructions sharper than D1. Attention mass inside bbox increasing. FactorVAE discriminator reaching ~0.50 accuracy. Faint stripe banding still visible (CheSS layer3 residual gradient).

**What it drove:** Need to activate GAN for further sharpening while suppressing remaining stripe artifacts → D3

---

### D3 — PatchGAN + TV (Mar 25, 2026 — ongoing)

**Run:** `d3_gan_fix-20260325-143813`
**W&B:** https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk
**Epoch range:** 55 → 120 (+65 epochs from D2 final)
**Checkpoint in progress:** `runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/` (saved every 5 epochs)

**Preceding failed GAN attempts:**

| Run | Date | Failure |
|-----|------|---------|
| `d5_gan-20260323-042442` | Mar 23 | Catastrophic collapse at epoch 4 (weight_gan=0.5, global gan_start_step) |
| `d5_gan_v2-20260323-085311` | Mar 23 | Stalled 80+ epochs (R1 penalty trap) |
| `d5_gan-20260325-041409` | Mar 25 | Wrong resume checkpoint |
| `d5_gan-20260325-044647` | Mar 25 | Config error |
| `d5_gan_v2-20260325-142348` | Mar 25 | Batch size OOM |
| `d5_gan_v2-20260325-142729` | Mar 25 | Wrong checkpoint path |

**What changed from D2 (critical fixes):**

| Parameter | D2 | D3 | Fix rationale |
|-----------|----|----|---------------|
| `weight_gan` | 0.0 | **0.1** | Conservative: was 0.5 in failed runs (4.6× GAN:rec imbalance) |
| `gan_start_step` | — | **2000 (phase-local)** | Phase-local = restarts from 0 each resumed phase; global fired immediately at step 1 |
| `disc_r1_penalty` | — | **0.0** | Was 10.0 — prevented discriminator bootstrap |
| `lr_patch_disc` | — | **1e-4** | Was 3e-5 — too slow to escape random-init regime |
| `weight_tv` | 0.0 | **0.005** | 5× from failed runs' 0.001 — suppresses stripe artifacts |
| `weight_perceptual` | 0.05 | **0.05** | Keep; layers 1–2 only (layer3 disabled) |
| `weight_kl_disease` | 1e-4 | **5e-5** | 1e-4 empirically linked to stripe artifacts |
| `bbox_query_mix` | 0.7 | **1.0** | Pure Gaussian prior — encoder mature enough |
| `weight_bbox_attn` | 0.05 | **0.10** | Raised after 120 ep of guidance |

**GAN timing within D3:**
- Steps 0–2000 (≈ epochs 55–60): GAN dormant, all other losses active
- Step 2000+ (≈ epoch 60+): Full loss stack active including GAN + patch disc

**Outcome:** Results exceed expectations. Reconstructions described as "incredibly clean." Stripe artifacts eliminated. Cardiac border crisp. Declared D3 milestone. See [08 Current State](08_current_state_d3.md) for full analysis.

---

## Checkpoint Chain

```
D1 final (epoch 30)
  runs_sepvae/d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl
    └─ resumed into ─▶
D2 final (epoch 55)
  runs_sepvae/d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl
    └─ resumed into ─▶
D3 (epoch 55 → 120, ongoing)
  runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch01XX.pkl
```

**Warning:** Several `d4_mi_percep`, `d4_perceptual`, and `d5_recon` checkpoints exist in `runs_sepvae/`. These were trained with CheSS layer3 perceptual at weight=0.15 and have stripe artifacts baked into decoder weights. **Do not resume from these.** The canonical D3 chain traces through D2 → D3 only.

---

## Key Decision Points

| Date | Decision | Alternative rejected | Reason |
|------|----------|---------------------|--------|
| Feb 20 | Remove free_bits | Keep free_bits, tune threshold | Hard failure mode; no safe threshold with sigma_inactive=0.1 |
| Mar 19 | V2 rewrite (from scratch) | Unfreeze CheSS backbone partially | Stride=32 ceiling; BatchNorm instability; feature misalignment |
| Mar 24 | Perceptual layers 1–2 only | Include layer3 | Layer3 stride=16 → 16px-period stripes at weight≥0.05 |
| Mar 25 | weight_gan=0.1 | weight_gan=0.5 | 0.5 failed catastrophically in <4 epochs |
| Mar 25 | Phase-local gan_start_step | Global gan_start_step | Global fired from step 1 with restored global_step=57196 |
| Mar 25 | disc_r1_penalty=0.0 | disc_r1_penalty=10.0 | 10.0 prevented discriminator from bootstrapping |
| Mar 25 | Resume from D2 (not D4/D5) | Resume from d4/d5 checkpoints | D4/D5 had stripe artifacts baked in |
