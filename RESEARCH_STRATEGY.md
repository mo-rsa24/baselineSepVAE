# SepVAE Research Strategy
**Last updated:** 2026-03-16
**Status:** Active — replaces the execution order in `00_experiment_plan.md`

---

## 1. Situation Assessment — What We Know From Running Experiments

### 1.1 Baseline Run Results (baselineSepVAE, 512×512, frozen CheSS)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| KL at epoch 1 | 742,833 | Posterior catastrophically far from prior |
| KL at epoch 65 | ~8,600 | Still 86× higher than a healthy VAE |
| rec (MSE) | 0.013–0.015 | Numerically small, visually catastrophic |
| bbox attn loss | 0.95 → 0.45 | Attention learning slowly but not anatomical |
| silhouette (all heads) | 0.022 | No meaningful class separation |
| silhouette (disease only) | 0.216 | Marginal disease separation |
| Prototype angle | 179.7° | Prototypes orthogonal ✓ |
| Cross-head correlation |r| | 0.375 | Latent activations still correlated ✗ |
| Reconstruction visual quality | Blurry, structureless | Clinically useless |

### 1.2 Root Cause Diagnosis

**The frozen backbone is the primary failure mode.**

The frozen CheSS ResNet-50 was pretrained for discriminative classification, not generative encoding. Its features are optimised to separate classes, not to preserve the pixel-level information that a decoder needs to reconstruct. Crucially:

- The backbone outputs fixed features regardless of what the decoder needs.
- The three encoder heads map these frozen features into a posterior `q(z|x)`.
- The posterior is so far from `N(0,I)` that the KL term dominates every other loss by 2–3 orders of magnitude (KL=742k vs rec=0.07 at step 0).
- The huge KL gradient dominates training and pushes the latents to a near-zero regime where the decoder receives near-zero signal → structureless grey reconstructions.
- The backbone cannot be updated to resolve this because it is frozen.

**Secondary failure: too many objectives active before reconstruction is stable.**

bbox + ortho + KL_disease + perceptual + MSE all active from epoch 1 with warmup only on KL. The separation losses have sharp gradient signal and win early; reconstruction never gets to stabilise.

**Secondary failure: spatial resolution mismatch.**

512×512 input → 64×64 frozen feature map → 8ch × 64×64 latent → decode to 512×512. The 8× compression from latent to output is being asked to recover all high-frequency detail that was never encoded in the frozen features. The decoder is not capable of inverting a frozen discriminative encoder.

### 1.3 What Did Work

- The prototype angle (179.7°) shows the orthogonality loss successfully pushes the learned prototypes apart. The *direction* of the disease prototypes separates.
- The bbox loss is showing consistent decay — attention IS learning spatial structure, just slowly.
- The loss architecture and metric logging infrastructure is complete and correct.
- EMA, checkpoint/resume, W&B integration all function correctly.

### 1.4 The Fundamental Tension

> Crisp VAE reconstructions require the encoder to preserve pixel-level information in the latent.
> Disentanglement objectives require the latent to be compressed and structured.
> A frozen discriminative backbone serves neither.

You cannot resolve this tension by tuning loss weights. The backbone must be trainable (or replaced).

---

## 2. Strategic Reorientation

### Core Principle: Ground Every Improvement on a Working Baseline

Never add a new complexity (new loss, new architecture component, new dataset) until you have a confirmed working baseline at the current complexity level. Each phase has a defined success criterion. Do not proceed to the next phase until the criterion is met.

### The Ladder

```
Phase 0  SynDis 64×64    — Prove the loss stack works on synthetic data (miniSepVAE / sepVAE codebase)
Phase 1  miniCXR 128×128 — Prove reconstruction quality on real CXR with a trainable encoder
Phase 2  Single disease  — Prove separation (cardiomegaly only) without breaking reconstruction
Phase 3  Two diseases    — Extend to cardiomegaly + effusion with nulling
Phase 4  Orthogonality   — Add spatial + statistical decorrelation on proven reconstruction baseline
Phase 5  Full scale      — 512×512 with partial backbone unfreezing if needed
```

---

## 3. Phase 0 — SynDis Synthetic Validation (64×64, PyTorch)

**Goal:** Confirm that the complete loss stack can achieve clean separation AND good reconstruction on a dataset with known ground truth.

**Codebase:** `miniSepVAE/run/train_syndis_sepvae.py` (already exists, PyTorch, CPU-runnable)
**Dataset:** `miniSepVAE/datasets/SynDis.py` — Gaussian background + blob (cardio analog) + stripe (effusion analog)

### What to run

```bash
# E0: Vanilla VAE — can we reconstruct SynDis cleanly at all?
python run/train_syndis_sepvae.py --epochs 100 \
    --weight_null 0 --weight_orth 0 --weight_clf 0 \
    --exp_name e0_vanilla

# E2: Split-head VAE — do heads separate naturally without any separation loss?
python run/train_syndis_sepvae.py --epochs 100 \
    --weight_null 0 --weight_orth 0 --weight_clf 0 \
    --split_heads --exp_name e2_split_head

# E7: Full SepVAE — does the full loss stack preserve reconstruction?
python run/train_syndis_sepvae.py --epochs 100 \
    --weight_null 0.2 --weight_orth 0.1 --weight_clf 1.0 \
    --exp_name e7_sepvae_full
```

### Success criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| rec_mse (E0) | < 0.005 | Architecture can reconstruct cleanly |
| spec_ratio_d1 (E7) | > 2.0 | Disease-1 head is specific to disease-1 |
| spec_ratio_d2 (E7) | > 2.0 | Disease-2 head is specific to disease-2 |
| comp_score (E7) | > 0.15 | Composed samples contain both disease signals |
| rec_mse (E7) | ≤ 2× E0 | Separation doesn't destroy reconstruction |

If E0 fails reconstruction: fix the architecture before continuing.
If E7 fails spec_ratio but E2 passes: the loss stack is the problem.
If E7 passes all criteria: proceed to Phase 1.

### Key diagnostic: compare E0 vs E7 reconstruction quality side by side

This tells you the cost of adding the separation objective in isolation of CXR complexity.

---

## 4. Phase 1 — Paper Reference Implementation

**Goal:** Obtain a confirmed working SepVAE from the published paper to use as a reference implementation. Adapt VinBigData to run through it for single disease (cardiomegaly).

### Step 4.1 — Clone the paper's codebase

Provide the paper's GitHub URL. Steps:
```bash
# In /home-mscluster/mmolefe/Playground/PhD/
git clone <paper_repo_url> reference_sepvae
cd reference_sepvae
git remote remove origin     # strip remote tracking
```

The reference implementation is the ground truth for:
- What a healthy training loss curve should look like
- What the reconstruction quality should be at convergence
- What the expected separation metrics are on the paper's dataset

### Step 4.2 — Adapt VinBigData cardiomegaly-only to the paper's dataloader

Create a minimal adapter:
- Normal vs Cardiomegaly only (2 classes)
- 128×128 or paper's native resolution
- Match the paper's normalisation range exactly
- Reuse `miniSepVAE/datasets/VinBigData.py` logic, strip effusion, strip triplet structure

### Step 4.3 — Run and confirm reconstruction quality

**Success criterion:** The paper's model produces visually interpretable CXR reconstructions at cardiomegaly-only setting. If not, document why (data mismatch, resolution mismatch, etc.) and fix.

---

## 5. Phase 2 — miniSepVAE on CXR, Single Disease (128×128, Trainable Encoder)

**Goal:** First working CXR SepVAE with crisp reconstructions. One disease head (cardiomegaly). Trainable encoder from scratch (no frozen backbone).

**Codebase:** `miniSepVAE/run/train_mini_sepvae.py` + `miniSepVAE/models/mini_sep_vae.py`
**Architecture:** `MiniCNNBackbone` (3 stride-2 stages → 16×16×128) — fully trainable
**Resolution:** 128×128 (VinBigData resized)

### Why trainable encoder fixes the reconstruction problem

| | Frozen CheSS (failed) | Trainable MiniCNN |
|---|---|---|
| KL regime | 742,000 at step 0 | Expected < 100 at step 0 |
| Gradient flow | Encoder frozen → decoder alone | Encoder + decoder co-adapt |
| Information bottleneck | Discriminative features only | Encoder learns to encode what decoder needs |
| CheSS benefit | Discriminative features | Not used — learn from data |

### Loss schedule — staged activation

Do not activate all losses simultaneously. Use this order:

**Warmup (epochs 1–10):** MSE + KL only (β=0.0001, no warmup needed at small scale)
**Stage 1 (epochs 11–30):** + Nulling loss (weight=0.1). Confirm separation starts without breaking reconstruction.
**Stage 2 (epochs 31–60):** + KL_disease conditional prior. Confirm latent sparsity.
**Ablation:** compare Stage 2 vs Stage 1 reconstruction quality to quantify the cost of each loss.

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Reconstruction | Visually interpretable CXR structure (cardiac silhouette visible) |
| AUC d_cardio → cardio | > 0.75 |
| AUC common → cardio | < 0.65 (common head doesn't encode disease) |
| KL at convergence | < 500 (vs 8,600 in failed baseline) |

---

## 6. Phase 3 — Two Diseases (Cardiomegaly + Effusion, 128×128)

**Goal:** Extend the working single-disease model to two disease heads without losing reconstruction quality.

### What changes from Phase 2

- Add effusion head (z_effusion, 2 channels)
- Return to triplet dataset structure (Normal, Cardiomegaly, Effusion)
- Add nulling for effusion head (identical mechanism to cardiomegaly)
- No orthogonality loss yet — measure natural separation from nulling alone

### Key question to answer

Does nulling alone (+ separate heads) produce meaningful inter-head separation, or is explicit orthogonality required?

Measure:
- Cross-head AUC: `AUC(z_cardio → effusion_label)` and `AUC(z_effusion → cardio_label)` — both should be ~0.5
- Cross-head correlation `|r|` — should be < 0.2 if nulling is sufficient

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Reconstruction quality | Maintained from Phase 2 |
| AUC d_cardio → cardio | > 0.75 |
| AUC d_effusion → effusion | > 0.75 |
| AUC d_cardio → effusion | < 0.65 |
| AUC d_effusion → cardio | < 0.65 |

If cross-head AUCs are already < 0.65 without orthogonality loss: the nulling mechanism is sufficient. Record this as an ablation finding (orthogonality loss may be unnecessary overhead).

---

## 7. Phase 4 — Orthogonality (Only If Phase 3 Shows Leakage)

**Goal:** Add spatial and statistical decorrelation only if Phase 3 shows that nulling alone is insufficient for separation.

### Losses to add, in order of invasiveness

**4a — Prototype cosine penalty** (lowest cost)
Penalise `cos²(mean(z_cardio), mean(z_effusion))` across the batch. No discriminator, no extra parameters.

```python
weight_ortho_proto = 0.05   # start low, increase if needed
```

**4b — Barlow cross-correlation penalty** (moderate cost)
Cross-correlate disease head activations on disease-positive samples only. Penalise off-diagonal correlation.

```python
weight_ortho_barlow = 0.1
```

**4c — Bbox attention supervision** (requires annotation)
Only if attention maps are available and verified to be spatially meaningful after Phase 3.

```python
weight_bbox_attn = 0.1    # not 0.2 — start conservative
```

### Danger signs — stop and investigate if you see these

- Reconstruction MSE increases > 20% after adding any orthogonality loss → the weight is too high or the mechanism is conflicting
- KL increases significantly → orthogonality is pushing latents away from the prior
- Silhouette score drops while cross-head AUC also drops → the model is collapsing, not separating

---

## 8. Phase 5 — Scale to 512×512 (After Phase 4 is Stable)

**Goal:** Bring the validated 128×128 system to full CXR resolution with a partially unfrozen CheSS backbone.

### Architecture strategy at full scale

**Option A — Partially unfreeze CheSS (recommended)**
- Keep CheSS ResNet-50 as encoder backbone
- Unfreeze the last 2 residual stages (layer3, layer4) with LR 1e-5
- Keep the rest frozen (layer1, layer2) with LR 0
- Encoder heads use LR 1e-4 (as before)
- This gives the encoder the ability to adapt its features for reconstruction without destroying all CheSS pretraining

```python
# Differential learning rates
lr_backbone_frozen  = 0.0      # layer1, layer2
lr_backbone_tunable = 1e-5     # layer3, layer4
lr_heads            = 1e-4     # encoder heads
lr_decoder          = 1e-4     # decoder
```

**Option B — Fully trainable encoder (fallback)**
Replace frozen CheSS with a trainable ResNet-50 initialised from ImageNet weights. Simpler, but loses the domain-specific CXR pretraining advantage.

### Perceptual loss at 512×512

Once the backbone is unfrozen, the perceptual loss setup changes:
- Cannot use the training backbone for perceptual loss (would create circular dependency)
- Either: use a separate frozen copy of CheSS for perceptual loss (as currently implemented)
- Or: use LPIPS (VGG-based) — available in `miniSepVAE/losses/lpips_gan.py`

### Expected KL regime at 512×512

With partially unfrozen backbone:
- The encoder heads adapt to produce features compatible with the prior
- KL should fall from the catastrophic 742,000 seen with frozen backbone to < 1,000 within the first few epochs
- If KL is still > 5,000 after epoch 5 with unfreezing: increase LR of unfrozen layers or unfreeze more

---

## 9. Codebase Map — What Exists vs What Needs Building

### Already implemented and validated

| Component | Location | Status |
|-----------|----------|--------|
| SynDis dataset | `miniSepVAE/datasets/SynDis.py` | Complete |
| SynDis PyTorch trainer | `miniSepVAE/run/train_syndis_sepvae.py` | Complete |
| MiniCNNBackbone (128×128, trainable) | `miniSepVAE/models/mini_sep_vae.py` | Complete |
| Full loss library | `miniSepVAE/losses/sep_vae_losses.py` | Complete |
| LPIPS + PatchGAN discriminator | `miniSepVAE/losses/lpips_gan.py` | Complete |
| VinBigData triplet loader | `miniSepVAE/datasets/VinBigData.py` | Complete |
| CheSS ResNet-50 (JAX) | `baselineSepVAE/models/resnet_jax.py` | Complete |
| Full-scale decoder | `baselineSepVAE/models/sep_vae_jax.py` | Complete |
| Diagnostic visualisation | `baselineSepVAE/utils/sepvae_diagnostics.py` | Complete |
| EMA, W&B, checkpoint/resume | `baselineSepVAE/run/train_sep_vae.py` | Complete |
| SepVAE E0-E6 trainers (SynDis) | `sepVAE/train/e0_vanilla.py` … `e6_zou.py` | Complete (needs audit) |

### Needs to be built

| Component | Priority | Phase |
|-----------|----------|-------|
| VinBigData cardiomegaly-only loader (non-triplet) | High | Phase 1–2 |
| Paper reference adaptation scripts | High | Phase 1 |
| miniSepVAE single-disease config | Medium | Phase 2 |
| Differential LR optimizer (partially unfrozen backbone) | Medium | Phase 5 |
| Evaluation script: AUC probes, spec_ratio, comp_score | High | Phase 2 onwards |

---

## 10. Metric Protocol (All Phases)

Use these metrics consistently across all phases to make results comparable.

### Reconstruction quality
- **rec_mse**: Mean squared error over test set (lower is better)
- **Visual inspection**: recon grids saved every `sample_every` epochs — the primary truth signal

### Separation quality
- **spec_ratio_d1** = `AUC(z_d1 → disease1) / AUC(z_d2 → disease1)` — target > 2.0
- **spec_ratio_d2** = `AUC(z_d2 → disease2) / AUC(z_d1 → disease2)` — target > 2.0
- **AUC(z_common → disease)** — should be ~0.5 (common head is disease-agnostic)
- **Cross-head correlation |r|** — should be < 0.2 for good separation

### Composition quality (main scientific claim)
- Build composed image: `Decoder(z_common from Normal, z_d1 from Disease1 image, z_d2 from Disease2 image)`
- **comp_score**: pixel signal in disease regions of composed image vs normal baseline
- **This is the metric that validates the paper's core claim** — high comp_score means the latent factors are truly independent and composable

### Diagnostic checkpoints (logged to W&B)
- Recon grid (every 5 epochs)
- Attention maps with bbox overlays (every 5 epochs)
- Latent manifold PCA + silhouette (every 10 epochs)
- Per-channel KL heatmap (every 10 epochs)
- Cross-head correlation matrix (every 10 epochs)

---

## 11. Lessons Recorded (Do Not Repeat)

1. **Do not freeze the entire backbone for generative tasks.** Discriminative pretraining ≠ generative pretraining. Always allow the encoder to adapt. Partial unfreezing is the minimal intervention.

2. **Do not activate all losses simultaneously.** Add one loss at a time, confirm the model is stable, then add the next. The order matters: reconstruction → KL → nulling → orthogonality.

3. **Do not judge separation quality before reconstruction is stable.** A silhouette score of 0.022 means nothing if the latents are encoding noise because the KL is catastrophically high.

4. **Watch KL at step 0.** It should be < 1000 for a healthy VAE start. If it is > 10,000, the encoder is not producing latents anywhere near the prior — stop and fix the encoder before training further.

5. **MSE numerically small ≠ reconstruction visually good.** `rec=0.013` in log-scale masked the catastrophic visual quality. Always inspect recon grids, not just loss numbers.

6. **Prototype orthogonality ≠ latent activation orthogonality.** The prototype angle being 179.7° only means the learned query vectors are opposite. The actual encoded latents can still be correlated (|r|=0.375) if the backbone features are dominated by shared structure.

7. **The composition test is the real test.** Silhouette scores and AUCs are proxies. The only metric that validates the scientific claim is whether `Decoder(z_c_normal + z_d1_disease1 + z_d2_disease2)` contains both disease signals in the correct anatomical locations.

---

## 12. Decision Gates

```
Phase 0 complete?
  └─ E0 rec_mse < 0.005  AND  E7 spec_ratio > 2.0 → proceed to Phase 1
  └─ E0 fails             → fix architecture (do not proceed)
  └─ E7 fails spec_ratio  → debug loss stack on synthetic data first

Phase 1 complete?
  └─ Reference impl reconstructs CXR cleanly → proceed to Phase 2
  └─ Reference impl fails → document data mismatch, adapt and retry

Phase 2 complete?
  └─ Reconstruction visually good  AND  AUC d_cardio > 0.75 → proceed to Phase 3
  └─ Reconstruction fails → try smaller KL weight, larger decoder, longer warmup

Phase 3 complete?
  └─ Cross-head AUC < 0.65  AND  Phase 2 reconstruction maintained → proceed to Phase 4
  └─ Cross-head AUC > 0.65 only → add Phase 4 orthogonality
  └─ Reconstruction drops  AND  cross-head AUC improves → record trade-off, investigate

Phase 4 complete?
  └─ spec_ratio > 2.0  AND  comp_score > 0.15  AND  reconstruction maintained → proceed to Phase 5
```
