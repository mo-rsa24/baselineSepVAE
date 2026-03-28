# Failures & Debugging Log

**Related documents:** [01 Experiment Timeline](01_experiment_timeline.md) | [03 Training Curriculum](03_training_curriculum.md) | [04 Model Architecture](04_model_architecture.md) | [05 Objective Functions](05_objective_functions.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25

---

## Purpose of This Document

This is the canonical failure log for the SepVAE project. Every failure that materially changed a training decision is recorded here with: the symptom as it appeared, the root cause diagnosis, the fix applied, whether the fix worked, and the lesson for future work.

If you are about to launch a new training phase, read the checklist in §8 first.

---

## Failure Index (quick lookup)

| ID | Name | Era | Category | Outcome |
|----|------|-----|----------|---------|
| [A1](#a1--free-bits--sigma_inactive-conflict) | free_bits / sigma_inactive dead zone | V1 | Gradient blocking | Fixed by R1 |
| [A2](#a2--nan-explosion-at-epoch-200) | NaN explosion at epoch 200 | V1 | Training instability | Fixed by R3 |
| [A3](#a3--cardiomegaly-head-collapse-under-heavy-nulling) | Cardiomegaly head collapse | V1 | Latent collapse | Fixed by R2+R4 |
| [A4](#a4--universal-cross-head-leakage) | Universal cross-head leakage > 0.74 | V1 | Disentanglement failure | Addressed by R5a/R5b/R7; not fully solved |
| [A5](#a5--no-lr-schedule--no-convergence-signal) | No LR schedule / plateau | V1 | Optimisation failure | Fixed by R3 |
| [B1](#b1--d5_gan-20260323-042442--catastrophic-collapse-epoch-4) | d5_gan catastrophic collapse | V2 | Adversarial instability | Fixed for D3 |
| [B2](#b2--d5_gan_v2-20260323-085311--r1-trap-stalled-80-epochs) | d5_gan_v2 R1 trap | V2 | Adversarial instability | Fixed for D3 |
| [B3](#b3--disc-dominance-warning-pattern) | Disc dominance warning pattern | V2 | Adversarial instability | Kill condition defined |
| [C1](#c1--horizontal-16px-stripe-banding-d2-d4) | Stripe banding (16px period) | V2 | Aliasing artifact | Fixed in D3 |
| [C2](#c2--layer4-stride-aliasing-in-latent-space) | Layer4 stride aliasing | V2 | Aliasing artifact | Fixed before D1 |
| [C3](#c3--checkerboard-artifacts) | Checkerboard artifacts | V1/V2 | Upsampling artifact | Fixed by SmoothUp |
| [C4](#c4--stripe-artifacts-from-weight_kl_disease1e-4) | weight_kl_disease stripe artifacts | V2 | KL-induced artifact | Fixed by reverting |
| [D1](#d1--z_disease-not-zeroing-for-normal-images-v1) | z_disease not zeroing for Normal | V1 | Latent routing | Fixed by hard-zero |
| [D2](#d2--z_common-absorbing-disease-signal) | z_common absorbing disease | V1/V2 | Latent routing | Partially fixed by masked_rec |
| [D3](#d3--attention-head-free-drift-d1-d2) | Attention head free drift | V2 | Latent routing | Fixed by bbox_attn loss in D2 |
| [D4](#d4--val_proj-waste-in-bboxcrossattnhead) | val_proj trained but unused | V2 | Architecture waste | Fixed before D3 |
| [D5](#d5--normal-images-contaminating-key_proj) | Normal images contaminating key_proj | V2 | Latent routing | Fixed by stop_gradient |

---

## Category A — Training Instability (V1 Architecture)

### A1 — free_bits / sigma_inactive Conflict

**Era:** V1 (all runs before Phase 3)
**Run(s):** `independence-I-20260218-sweep` (W&B: `9lj20so0`), multiple others

**Symptom:**
Disease head probe AUC flat at 0.50 (random). The cardiomegaly and effusion probes never trained. Inspecting the loss logs showed `kl_cardio ≈ 1.8 nats` throughout — exactly the value expected for $\sigma_{\text{inactive}} = 0.1$.

**Root cause:**
The free-bits mechanism clips the KL gradient to zero for any latent dimension with $\text{KL} < \lambda_{\text{fb}}$ (free_bits threshold). This was designed to protect the common head from over-penalisation at early training, but it applies globally.

For the disease head, inactive samples (Normal images) force $\sigma_{\text{inactive}} = 0.1$, which gives:
$$\text{KL}_{\text{inactive}} = \frac{1}{2}\left(\sigma_{\text{inactive}}^2 - 1 - \log \sigma_{\text{inactive}}^2\right) \approx 1.8 \text{ nats}$$

With `free_bits = 2.0`, every disease-head channel was permanently inside the dead zone. The encoder received zero gradient through the disease head KL — the disease head was effectively cut off from any KL-based training signal.

```
KL_channel ≈ 1.8 nats
────────────────────────────────────────────────────
          ◄── dead zone ──►
    0 ──────────────────── 2.0 ──────────── ∞
                           ↑ free_bits = 2.0
          gradient = 0 here (ALL disease channels permanently)
```

The constraint that causes failure:
$$\text{KL}_{\text{inactive}}(\sigma_{\text{inactive}}) < \text{free\_bits}$$

| $\sigma_{\text{inactive}}$ | $\text{KL}_{\text{inactive}}$ (nats) | Dead zone with free_bits=2.0? |
|---------------------------|--------------------------------------|-------------------------------|
| 0.20 | ~1.1 | **YES** |
| 0.10 | ~1.8 | **YES** |
| 0.05 | ~2.5 | No |

**Fix (R1):** Remove `free_bits` entirely (`free_bits=0.0`). Control inactivity through `sigma_inactive` alone. Tighten to `sigma_inactive=0.05` for V1 (gives $\text{KL} \approx 2.5 \text{ nats}$, safely above any practical threshold).

Note: V2 uses `kl_free_bits=0.5` (not 2.0) and `sigma_inactive=0.1`. With these values: $\text{KL}_{\text{inactive}} \approx 1.8 \text{ nats} > 0.5$ — the dead zone does not apply. The V1 failure required the specific combination of large free_bits + moderate sigma_inactive.

**Outcome:** After R1, disease head KL showed proper gradients. Probe AUC climbed from 0.50 to 0.774 by epoch 200.

**Lesson:** Before any training run, verify: $\text{KL}(\sigma_{\text{inactive}}) > \text{free\_bits}$. This is a binary kill condition. The check takes 5 seconds.

---

### A2 — NaN Explosion at Epoch 200

**Era:** V1 Phase 1
**Run(s):** `disentangle-E-20260217-153031` (W&B: epoch 200 NaN), also affected other long runs

**Symptom:**
Loss stable at 0.004–0.006 through epoch 199. At epoch 200 (final batch), all loss terms went `nan` simultaneously. Full run wasted. No intermediate checkpoints because `save_every=50`.

**Root cause:**
Three compounding factors:

1. **No learning rate schedule.** Fixed `lr_vae = 1e-4` for 200 epochs with no decay. Small gradient errors accumulated over long training until one unlucky batch (likely a batch with many hard examples and high KL simultaneously) produced a catastrophic parameter update.

2. **Full fp32 precision.** No implicit gradient scaling. The gradient norm could grow arbitrarily without the overflow protection that mixed precision provides.

3. **No gradient clipping.** Even a single large gradient update was unchecked.

The NaN was not recoverable — the checkpoint at epoch 150 had valid weights, but epoch 200 was lost.

**Fix (R3):** Cosine LR decay from `lr_decay_epochs` onward:
$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - t_{\text{decay}})}{T - t_{\text{decay}}}\right)\right), \quad \eta_{\min} = 0.1\eta_0$$

Combined with `grad_clip = 1.0` (added in V2). `save_every` reduced from 50 to 5 (V2) or 10 (late V1) to limit checkpoint loss.

**Outcome:** No NaN in any V2 run. Cosine decay also improves final loss quality by allowing finer steps near convergence.

**Lesson:** Never run long training without: (1) LR decay, (2) gradient clipping, (3) frequent checkpoints. The NaN is unrecoverable — the only mitigation is preventing it and having recent saves to fall back to.

---

### A3 — Cardiomegaly Head Collapse Under Heavy Nulling

**Era:** V1 Phase 2
**Run(s):** `inactivity-G-20260220` (W&B: `41nce8qq`)

**Symptom:**
`probe_auc/cardiomegaly = 0.476` (below random chance) by epoch 100. The cardiomegaly head KL collapsed to near-zero. Effusion head was unaffected (`probe_auc/effusion ≈ 0.71`). Paradoxically, this was the run with the strongest regularisation settings.

**Root cause:**
Heavy `weight_null = 0.05` drove $\mu_{\text{cardio}} \to 0$ faster than the reconstruction gradient could maintain the cardiomegaly signal. Three compounding factors:

1. **Cardiomegaly is distributed and subtle.** Enlarged cardiac silhouette is a globally distributed, low-contrast change. Effusion is a localised bright blob in the lower lobes. The reconstruction gradient on the cardiomegaly head is diffuse and weak — the encoder can often get lower total loss by routing cardiomegaly features into z_common than by maintaining them in z_cardio against the nulling pressure.

2. **CheSS backbone features do not separate cardiac from pulmonary features well.** The backbone was pretrained on a classification task that does not require this separation.

3. **Nulling loss applies symmetric pressure regardless of signal strength.** The loss is $\| \mu_d \|^2$ — it does not know whether this head *should* be active for a given image. A strong weight drives both active and inactive cardiomegaly directions toward zero.

**Fix (R2+R4):**
- R2: Reduce to `weight_null=0.02`, `weight_orthogonality=0.02` — gentler, balanced pressure.
- R4: Add `min_active_kl=2.0` floor — for active-label samples, add a penalty that drives KL above a minimum:
  $$\mathcal{L}_{\text{floor}} = \max(0, 2.0 - \text{KL}_{\text{active}})$$
  This prevents the cardiomegaly head from collapsing even under strong nulling, because collapsing on a cardiomegaly-positive sample now incurs a penalty.

**V2 replacement:** V2 uses hard-zero nulling + BboxCrossAttnHead + `weight_kl_disease=5e-5` (very soft). There is no explicit `min_active_kl` in V2 because the supervised contrastive loss (`L_supcon`) and masked reconstruction (`L_masked_rec`) together maintain the disease signal without needing an explicit floor.

**Outcome:** After R2+R4, cardiomegaly probe AUC recovered to 0.67–0.72 range without effusion collapse.

**Lesson:** Nulling pressure and reconstruction pressure are in direct competition for the disease head. If nulling wins, the head collapses. Always monitor `probe_auc/cardiomegaly` separately and kill any run where it falls below 0.55 for 3+ consecutive evaluations.

---

### A4 — Universal Cross-Head Leakage Above 0.74

**Era:** V1 (all phases)
**Run(s):** All runs across Phases 1–6

**Symptom:**
`cross_head_score` never falls below 0.74 in any run. Best result: 0.744 (inactivity-G), at the cost of cardiomegaly collapse. The orthogonality + MI loss portfolio does not achieve conditional independence.

**Root cause:**
The current regularisation enforces marginal statistical independence but not *conditional independence*:

```
What orthogonality enforces:
  E[z_cardio] ⊥ E[z_effusion]       (prototype cosine penalty)
  Corr(z_cardio_i, z_effusion_j) ≈ 0    (Barlow-style)

What it does NOT enforce:
  z_cardio ⊥ y_effusion | z_common   (conditional independence)
```

Both ConvHeads see the same full 64×64 backbone feature map and must learn, through gradient pressure, to ignore the spatial region belonging to the other disease. This is an ill-posed implicit learning problem. The cardiomegaly head can encode pleural-region features that correlate with effusion without violating any marginal independence constraint.

**Fixes (R5a, R5b, R7):**
- **R5a** — paired contrastive loss: blind-pull term forces effusion-head representations of cardiomegaly images toward the normal centroid.
- **R5b** — cross-head adversarial MLPs: directly tests and penalises whether z_effusion predicts cardiomegaly (and vice versa).
- **R7** — label attention routing: gives each disease head a learnable spatial routing mechanism that can structurally prevent it from attending to the other disease's anatomical region.

**V2 approach:** V2 replaces the explicit cross-head discriminators with a combination of BboxCrossAttnHead (structural spatial routing from epoch 1) + bbox_attention_loss (supervised concentration) + masked_rec (z_common purity pressure). Cross-head score in V2 is not separately tracked yet — monitoring via FactorDisc accuracy proxy.

**Outcome:** R5a+R5b+R7 together reduced cross-head score from 0.80 to approximately 0.74–0.76 (marginal improvement) in V1. The problem is unsolved structurally — it is the primary open research challenge.

**Lesson:** Marginal independence losses (orthogonality, MI) are insufficient for conditional independence between disease heads. Structural mechanisms (spatial routing, conditional discriminators) are required. This is an active area of research for V2 D4+ phases.

---

### A5 — No LR Schedule / No Convergence Signal

**Era:** V1 Phases 1–2
**Run(s):** All early runs

**Symptom:**
Probe AUC plateaus between epochs 50–80 with no further improvement, even with extended training to 150–200 epochs. Loss improvement is negligible after the plateau. Training is wasted compute.

**Root cause:**
Fixed learning rate `lr_vae = 1e-4` for all epochs. After the initial rapid training phase, the model oscillates in a loss basin without being able to converge finely. The same gradient step size that enabled fast initial learning prevents fine convergence later.

**Fix (R3):** Cosine LR decay. The schedule starts at `lr_decay_epochs` (default 60) and decays to $0.1 \times \text{lr}_0$ by the final epoch. This allows aggressive early learning and fine convergence late.

**Outcome:** Runs with cosine decay consistently show improvement in the final 20–30 epochs rather than plateauing.

**Lesson:** Always use LR decay for runs longer than 50 epochs. `lr_decay_epochs = 60%–70%` of total epochs is a reliable default.

---

## Category B — Adversarial Instability (V2 PatchGAN)

### B1 — d5_gan-20260323-042442: Catastrophic Collapse (Epoch 4)

**Era:** V2 (between D2 and D3)
**Run:** `d5_gan-20260323-042442` (W&B: `unh2f9vw`)

**Symptom:**
Reconstruction loss spiked from 0.04 to >2.0 within 4 epochs of GAN activation. Samples devolved into textured noise. The run was dead at epoch 144 (global) / epoch 4 of the GAN phase. No recovery.

**Root cause — Bug 1: weight_gan=0.5 (4.6× GAN:rec imbalance):**

At epoch 144 (the epoch of collapse), the approximate loss magnitudes were:
- Reconstruction: `1.0 × 0.135 = 0.135` (weight × value)
- GAN: `0.5 × 1.25 = 0.625` (weight × value)

Ratio: 4.6× in favour of adversarial gradient. The adversarial gradient overwhelmed the reconstruction signal. The decoder could not maintain image structure against the discriminator pressure — it committed to texture patterns that fooled the discriminator but destroyed the spatial structure the reconstruction loss was trying to preserve.

**Root cause — Bug 2: global_step used for gan_start_step:**

The checkpoint restored `global_step ≈ 57,196` from the D4 run. The condition for GAN activation was:
```python
gan_active = global_step >= gan_start_step  # gan_start_step = 2000
```
Since `57,196 >> 2,000`, `gan_active = True` from step 1 of the new phase. The fresh discriminator (randomly initialised weights) received full gradients from the very first batch. It reached 88% accuracy within 4 epochs (the generator made no effort to fool a discriminator that had never fired before in this phase). With an 88%-accurate discriminator and a 4.6× loss weighting, generator loss dominated and corrupted the decoder.

**Fix (both bugs):**

Bug 1: `weight_gan = 0.1` (not 0.5). At healthy training, this gives approximately:
- Reconstruction: `1.0 × 0.08 = 0.080`
- GAN: `0.1 × 1.0 = 0.100`
A ratio of ~1.25×, roughly balanced.

Bug 2: Phase-local `gan_start_step`. The training script now tracks:
```python
phase_start_global_step = global_step  # recorded at phase start
phase_local_step = global_step - phase_start_global_step
gan_active = phase_local_step >= gan_start_step  # 2000 phase-local steps
```
With `gan_start_step=2000` and batch_size=6, 2000 steps ≈ 5 epochs of warm-up in the new phase — giving the discriminator time to adapt before it fires at full strength.

**Outcome:** D3 (`d3_gan_fix`) runs 65 GAN-active epochs with no instability. Discriminator accuracy settles at 0.55–0.65 (healthy range).

**Lesson:** When resuming from a checkpoint: (1) `global_step` will be large; any threshold expressed in global steps may fire immediately. Always use phase-local step counters. (2) Check GAN:rec ratio before launch — both weights and loss magnitudes matter.

---

### B2 — d5_gan_v2-20260323-085311: R1 Trap (Stalled 80+ Epochs)

**Era:** V2 (between D2 and D3, second attempt)
**Run:** `d5_gan_v2-20260323-085311` (W&B: `sqwldbxb`)

**Symptom:**
Generator loss noisy with no consistent adversarial signal for 80+ epochs. Discriminator loss oscillated without a clear training trend. Reconstruction quality did not improve beyond the D2 baseline. No catastrophic collapse, but no GAN benefit either — the GAN was effectively inactive despite being enabled.

**Root cause: disc_r1_penalty=10.0 + lr_patch_disc=3e-5:**

The R1 gradient penalty is:
$$\mathcal{L}_{R1} = \frac{\gamma}{2} \cdot \mathbb{E}\left[\|\nabla_x D(x_{\text{real}})\|^2\right]$$

At $\gamma = 10$ (strong penalty), the discriminator was heavily penalised for any decision boundary that relied on high-frequency gradients with respect to the input. Combined with a very slow learning rate (`lr_patch_disc = 3e-5`), the discriminator could not escape its random-initialisation loss regime:

1. At random init, the discriminator's decision boundary is arbitrary.
2. Any boundary that happens to work would have high-frequency gradients → R1 penalty of 10× immediately drives it toward a flat boundary.
3. At `lr=3e-5`, the discriminator learns so slowly it cannot find a valid low-gradient boundary before the generator has already adapted away from it.

The intended purpose of R1 (prevent the discriminator from overfitting to high-frequency noise) was correct in principle, but the combination of strength=10 and lr=3e-5 was so conservative that the discriminator never meaningfully trained.

**Fix:**
- `disc_r1_penalty = 0.0` (removed entirely)
- `lr_patch_disc = 1e-4` (33× faster than the failed run)

The discriminator can now bootstrap normally. At healthy convergence, the discriminator accuracy is 0.55–0.65 — it has signal but is not dominant. R1 regularisation is not needed at this early stage of adversarial training.

**Outcome:** D3 discriminator bootstraps within the 2000-step warmup and maintains healthy accuracy throughout 65 adversarial epochs.

**Lesson:** R1 gradient penalty is useful for preventing discriminator mode-dropping at later stages of training, but it should not be applied during the initial bootstrap phase. If the discriminator cannot win against a randomly-initialised generator, R1 is too strong. Diagnose: if discriminator accuracy does not exceed 0.60 within 10 epochs of GAN activation, the discriminator is trapped.

---

### B3 — Disc Dominance Warning Pattern

**Era:** V2 (recurring risk pattern, not a single run)

**Symptom:**
`patch_disc_acc > 0.80` sustained for more than 5 consecutive epochs during active GAN training.

**Diagnosis:**
The discriminator has learned a reliable decision rule that the generator cannot evade. This precedes generator collapse: the adversarial gradient is too strong and consistent, driving the decoder toward texture patterns that fool the discriminator at the expense of reconstruction quality.

**Relationship to B1:** B1 was an extreme version of this (88% discriminator accuracy from epoch 1). B3 is the slower version — the discriminator winning gradually over tens of epochs. The outcome is the same if left unaddressed.

**Defined kill condition and response:**

| Condition | Action |
|-----------|--------|
| `patch_disc_acc > 0.80` for ≥5 epochs | Reduce `weight_gan` by 50% |
| `patch_disc_acc > 0.90` for ≥3 epochs | Kill run immediately; resume from last checkpoint with `weight_gan = 0.05` |
| `loss/reconstruction` increasing while `patch_disc_acc > 0.80` | Kill immediately |

**Recovery:** Reload the last checkpoint where disc_acc was below 0.75. Resume with `weight_gan` halved. If still unstable, temporarily set `weight_gan = 0.0` for 5 epochs to let the decoder recover, then re-introduce GAN at the reduced weight.

**Lesson:** Monitor `patch_disc_acc` every logging step during GAN training. Healthy range: 0.55–0.75. Above 0.80 is a warning; above 0.90 is a kill condition.

---

## Category C — Aliasing and Artifacts

### C1 — Horizontal 16px Stripe Banding (D2–D4)

**Era:** V2, D2 through D4
**Run(s):** Visible in all checkpoints of `d2_perceptual_bbox`, `d4_mi_percep`, earlier `d5_gan` runs

**Symptom:**
Visible horizontal banding with a spatial period of approximately 16 pixels in all reconstructions. Most prominent in flat-intensity regions (lungs, soft tissue). The banding was already visible at epoch 55 (end of D2) and progressively worsened through D4.

**Root cause:**
CheSS perceptual loss using layer3 features (stride=16 for 256px input) injects 16px-period gradients into the decoder via backpropagation through the frozen backbone. At `weight_perceptual=0.15` and `weight_tv=0.001`, the estimated gradient power ratio was approximately 300× in favour of perceptual over TV suppression.

Layer3 of the CheSS backbone has an effective receptive field of ~64px but a stride of 16px — meaning the gradient signal is modulated at 16px periods (every stride boundary creates a gradient discontinuity). The decoder learned to produce striped patterns that minimised the L1 distance in layer3 feature space, because stripes at this frequency look similar to layer3 features of real CXRs.

Crucially, **the stripes were baked into the decoder weights across D2–D4** — they were not a transient artifact but a learned pattern.

**Fix (D3):**
Three simultaneous changes:
1. Perceptual layers 1–2 only (remove layer3 entirely): `--perceptual_only` flag restricts to layers 1–2, which have strides 4 and 8 — producing 4px and 8px period gradients that are below the visual threshold.
2. `weight_perceptual`: 0.15 → 0.05 (3× reduction in perceptual gradient magnitude).
3. `weight_tv`: 0.001 → 0.005 (5× increase in TV suppression).

**Why D3 resumed from D2 (not D4):**
The stripe artifacts were baked into weights in D4/D5. D2 was the last checkpoint before layer3 perceptual had significantly corrupted the decoder. Resuming from D2 with the corrected perceptual settings was cleaner than trying to un-bake the stripes from D4 weights.

**Outcome:** Stripe banding eliminated in D3. At epoch 120, reconstructions are clean without visible periodic artifacts.

**Lesson:** When using a pretrained feature extractor for perceptual loss: (1) check the stride of each layer and its implications for gradient periodicity; (2) start with the lowest-stride layers (finer features, less aliasing risk); (3) TV weight must be calibrated against perceptual gradient magnitude. 300× imbalance means TV is irrelevant — the perceptual gradient dominates completely.

---

### C2 — Layer4BranchGN Stride Aliasing in Latent Space

**Era:** V2 (D0 pre-architecture-fix)
**Run(s):** Early D0 smoke runs with original stride=2 in Layer4BranchGN

**Symptom:**
Reconstructions with blurry, low-frequency latent features. Patterns that should have been crisp at the 16×16 scale appeared soft and spectrally impoverished. FactorDisc accuracy was unstable (could not find a stable decision boundary on the latent features).

**Root cause:**
The original Layer4BranchGN used stride=2 in the first bottleneck block, downsampling from 16×16 to 8×8. The output was then bilinearly upsampled back to 16×16 before the encoder head. This stride-2 → bilinear-upsample round-trip:

1. **Aliases spatial frequencies above the 8×8 Nyquist limit** (i.e., any spatial pattern with period < 4 pixels at the 16×16 scale is lost at the 8×8 intermediate — it cannot be recovered by any subsequent upsampling).
2. **Bilinear upsample introduces checkerboard aliasing** of its own — the bilinear kernel is a low-pass filter that smears any remaining high-frequency content.

The result: latent features at 16×16 were actually 8×8 features with false spatial resolution, making them less useful for both reconstruction and disentanglement.

**Fix:**
Stride=1 throughout Layer4BranchGN. The first bottleneck block uses a 1×1 projection shortcut (1024→2048 channels) without spatial downsampling. The branch stays at 16×16 throughout, preserving all spatial frequencies available from layer3.

```python
# Before (aliasing):
#   BottleneckBlockGN(in_channels=1024, out_channels=2048, stride=2)  # 16→8
#   → bilinear upsample back to 16

# After (fixed):
#   BottleneckBlockGN(in_channels=1024, out_channels=2048, stride=1)  # 16→16
#   shortcut: Conv(1x1, 1024→2048) no stride
```

**Outcome:** Latent features at 16×16 have genuine 16×16 spatial resolution. Reconstructions at D1+ show sharper spatial structure than any pre-fix run.

**Lesson:** Stride-2 downsampling followed by upsampling is not information-preserving. If you need to increase channel count without losing spatial resolution, use stride=1 with a projection shortcut. This is standard ResNet practice but easy to overlook when adapting architectures.

---

### C3 — Checkerboard Artifacts

**Era:** V1 and early V2 (before SmoothUp)
**Run(s):** All runs using transposed convolution upsampling

**Symptom:**
High-frequency checkerboard patterns across the full image, most visible in flat regions. Characteristic of transposed convolution with stride=2 — the kernel tiles with overlap, and any uneven kernel weights create a periodic amplitude modulation.

**Root cause:**
Transposed convolution (also called deconvolution) with stride=2: each output pixel is the sum of all kernel positions that contribute to it, but not all output positions receive the same number of contributing kernel positions (edge effects + interior positions differ). This creates a checkerboard modulation with period = stride.

**Fix: SmoothUp (bilinear resize + 2× conv3×3):**
```python
class SmoothUp(nn.Module):
    def __call__(self, x):
        h, w = x.shape[-3], x.shape[-2]
        x = jax.image.resize(x, (..., h*2, w*2, ...), method='bilinear')
        x = nn.Conv(features, (3,3), padding='SAME')(x)
        x = nn.Conv(features, (3,3), padding='SAME')(x)
        return x
```

Bilinear resize is a deterministic low-pass with no overlap artifacts. The subsequent 3×3 convolutions add learnable high-frequency restoration without introducing period-2 artifacts.

**Outcome:** Checkerboard artifacts completely absent from all V2 runs.

**Lesson:** Use bilinear resize + convolution for upsampling in image generation models. Transposed convolution checkerboards are a well-known problem with a well-known fix. The only reason to use transposed convolution is computational — accept checkerboards only as a last resort under memory constraints.

---

### C4 — Stripe Artifacts from weight_kl_disease=1e-4

**Era:** V2 D3-era experiments
**Run(s):** `d6_something` (intermediate experimental run, not canonical)

**Symptom:**
Vertical stripe artifacts appeared in reconstructions when `weight_kl_disease` was doubled from `5e-5` to `1e-4`. The stripes had a different character than the C1 horizontal banding — more irregular, vertical orientation, ~8px period.

**Root cause (hypothesis):**
Increased KL pressure on the disease head forces the encoder to minimise `KL(q(z_d|x) || p(z_d))` more aggressively. For Cardiomegaly images, the natural response is to encode less information in z_d (reducing posterior variance). But the reconstruction loss still requires z_d to be useful — the encoder resolves this by encoding disease features in a more regular, periodic pattern in z_d (stripes in feature space map to stripes in image space via the decoder's linear projection). Vertical period reflects the 8×8 subblock structure of the 16×16 latent map.

This is an unconfirmed hypothesis — the mechanism is inferred from the pattern, not directly observed.

**Fix:**
Revert `weight_kl_disease` to `5e-5`. This is the canonical value from D1 onward.

**Outcome:** Stripes disappeared immediately on reverting.

**Lesson:** The disease KL weight is load-bearing. `5e-5` was established carefully as the right balance between regularisation pressure and disease head capacity. Doubling it exceeded the encoder's ability to maintain smooth feature representations. Do not increase this value without understanding why.

---

## Category D — Latent Space Failures

### D1 — z_disease Not Zeroing for Normal Images (V1)

**Era:** V1
**Run(s):** All V1 runs before explicit hard-zero was enforced

**Symptom:**
Normal images produced reconstructions that occasionally showed traces of cardiac silhouette enlargement. The nulling loss was present (`weight_null=0.02`) but not achieving strict zero.

**Root cause:**
The V1 nulling loss was a soft L2 penalty:
$$\mathcal{L}_{\text{null}} = \| \mu_d \|^2 \cdot \mathbf{1}[\text{label=Normal}]$$

This drives $\mu_d \to 0$ but cannot enforce it perfectly at every sample — gradient descent minimises the *expected* loss, not the maximum. Some normal images with ambiguous anatomy produced $\mu_d \neq 0$ despite the penalty.

**Fix:**
Hard-zero nulling (V2): multiply z_d by the label mask after sampling:
```python
disease_mask = jnp.where(label == DISEASE_CLASS, 1.0, 0.0)
z_d_nulled = z_d * disease_mask[:, None, None, None]
# For Normal images: z_d_nulled = 0 everywhere, exactly
```

This is an architectural constraint, not a loss term. It cannot be violated by any gradient update. The conditional KL loss (tight prior N(0, σ²·I) for Normal images) provides additional gradient-based pressure toward zero that complements the hard zero.

**Outcome:** Hard-zero nulling is strictly enforced in all V2 runs. Normal image reconstructions rely solely on z_common.

**Lesson:** When you need a strict constraint (z_d = 0 for Normal images), enforce it architecturally, not through loss terms. Loss terms enforce it in expectation; architectural constraints enforce it exactly.

---

### D2 — z_common Absorbing Disease Signal

**Era:** V1 and early V2
**Run(s):** Observable across V1 Phases 1–3; partially present in V2 D1–D2

**Symptom:**
When z_disease is nulled for Normal images, the decoder still produces images with subtle cardiomegaly features. A linear probe on z_common achieves AUC > 0.65 for cardiomegaly classification — z_common has absorbed disease signal.

**Root cause:**
The reconstruction loss gradient on z_common (via the cardiomegaly branch) is stronger than the pressure keeping disease information out of z_common. The encoder finds it easier to spread disease signal across both z_common and z_disease (maximising total information available to the decoder) than to restrict it to z_disease. This is the "free-rider" problem for disentanglement: the common head can always improve reconstruction by encoding a little disease signal.

**Fix:**
`L_masked_rec` (masked anatomy reconstruction loss), introduced in D2:
$$\mathcal{L}_{\text{masked}} = \mathbb{E}\left[\| x \cdot (1 - M) - \hat{x} \cdot (1 - M) \|_2^2\right]_{\text{Cardio, z\_d=0}}$$

For Cardiomegaly images, decode using `z_d=0` (regardless of actual z_d) and compute MSE only in the non-cardiac region (outside the bbox mask $M$). If z_common contains cardiomegaly signal, this loss will be high — the non-cardiac lung fields will be wrong when decoded without z_d. Weight: `0.3`.

**Outcome:** `loss/masked_rec` decreased from ~0.08 in D1 to ~0.02 by end of D2, indicating improved z_common purity in the non-cardiac region.

**Lesson:** The reconstruction loss is not sufficient to prevent z_common from absorbing disease features. An explicit purity check (decode with z_d=0 and verify the output matches the disease-free regions) is needed. `L_masked_rec` implements this check directly.

---

### D3 — Attention Head Free Drift (D1–D2)

**Era:** V2, D1 phase
**Run(s):** Multiple D1-equivalent runs before bbox_attn loss was activated

**Symptom:**
Attention maps from BboxCrossAttnHead concentrated in the corners and lower edges of the image rather than the cardiac region. Disease head produced essentially random spatial weighting, making the Gaussian prior guidance ineffective.

**Root cause:**
`weight_bbox_attn=0.0` in D1 — the bbox attention supervision was not active. The BboxCrossAttnHead had a Gaussian prior from the bbox coordinates, but with no explicit loss penalising attention outside the bbox, the key projection could drift to attend to any high-energy region of the feature map. The corners and edges of CXRs often have high-frequency content (image border, text annotations) that the key vectors learned to attend to instead of the cardiac region.

The Gaussian prior blends with the learned query (`bbox_query_mix=0.7`), so 30% of the query was still free to drift.

**Fix:**
`weight_bbox_attn=0.05` (introduced in D2), raised to `0.10` in D3:
$$\mathcal{L}_{\text{bbox}} = \frac{1}{B_{\text{cardio}}} \sum_{b \in \text{cardio}} \frac{1}{HW} \sum_{i,j} A_d^{(b)}(i,j) \cdot (1 - M^{(b)}(i,j))$$

This directly penalises attention mass outside the annotated bbox for Cardiomegaly images. At D3, the attention loss decreases steadily from 0.7 (uniform attention) toward 0.15–0.20 (concentrated in bbox), confirming the attention head is concentrating.

**Outcome:** Attention maps in D3 show clear cardiac region concentration for Cardiomegaly images by epoch 70+. Normal images still use the learned fallback query (unaffected by bbox supervision).

**Lesson:** The Gaussian prior gives a good initialisation but is not sufficient to maintain bbox-confined attention — the learned component of the query will drift toward other salient features without explicit supervision. Always activate `weight_bbox_attn > 0` from the first phase where bbox cross-attention is used.

---

### D4 — val_proj Waste in BboxCrossAttnHead

**Era:** V2, D1–D2 original implementation
**Run(s):** `d1_recon_bbox_xattn-20260321-004241` and early D2 runs

**Symptom:**
The V projection (`val_proj`) inside BboxCrossAttnHead was being trained (consumed 512×512 parameters) but its output was never actually used — the attention-weighted output was discarded in favour of a direct spatial-weighted sum of the input feature map. Pure wasted compute.

**Root cause:**
Architecture implementation error during initial development of BboxCrossAttnHead. The design intended a full cross-attention mechanism (Q, K, V), but the final code used the attention map to weight the input features directly (not the V projection output). The val_proj module was never removed.

**Fix:**
Removed `val_proj` from BboxCrossAttnHead before D3. The forward pass correctly computes:
```python
# Attention map A = softmax(Q_bbox @ K.T / sqrt(D))  [B, HW]
# Disease features = (h_shared * A.reshape(B, H, W, 1)).mean(...)
# val_proj is not in the computational graph
```

**Outcome:** Minor parameter count reduction, cleaner gradient flow. No functional impact on output quality, but eliminates misleading training signal (val_proj was receiving gradients from the encoder loss and learning nothing useful).

**Lesson:** Verify that every trained module contributes to the computational graph. Orphaned modules waste memory, compute, and add misleading loss statistics. Check with a simple gradient inspection (`jax.grad` and verify which parameters have non-zero gradients).

---

### D5 — Normal Images Contaminating key_proj

**Era:** V2 D1–D2 original implementation
**Run(s):** Pre-fix versions of D1

**Symptom:**
The key projection (`key_proj`) in BboxCrossAttnHead was learning to respond to whatever spatial features Normal images had, not just the cardiac region of Cardiomegaly images. Since Normal images contribute the majority of the training batch (roughly equal split, but Normal images provide gradient to key_proj for every batch element), the key projection drifted toward border/edge features that are common in Normal images.

**Root cause:**
For Normal images, the BboxCrossAttnHead used the learned query (`Q_learned`) rather than the Gaussian prior (`Q_bbox`). This is correct. However, the attention computation still involved `key_proj(h_shared)` for Normal images, and the gradient from the reconstruction loss flowed back through `key_proj` for Normal images. The Normal image gradient shaped `key_proj` to project features useful for Normal image reconstruction — not cardiac features.

**Fix:**
`stop_gradient` on the `key_proj` output for Normal images:
```python
# For Normal images, key features are passed through stop_gradient
# before computing the attention map. This prevents Normal image
# reconstruction gradients from shaping the key projection.
keys = key_proj(h_shared)
if label == NORMAL:
    keys = lax.stop_gradient(keys)  # Normal images do not train key_proj
```

With this fix, `key_proj` is trained only by Cardiomegaly image gradients — ensuring it projects features relevant to cardiac spatial routing.

**Outcome:** Attention maps for Cardiomegaly images became more reliable, particularly in early training when the key projection had not yet learned to differentiate cardiac from non-cardiac features.

**Lesson:** In a cross-attention head designed to specialise on a specific anatomical region, ensure that training signal for the projection matrices comes only from examples of that region. Mixed-class gradients can contaminate the learned representations in ways that are difficult to detect without explicit diagnostic visualisation of the attention maps.

---

## Cross-Cutting Lessons

### Lesson 1: Check gradient flow before analysing metric behaviour

Before concluding that a loss is "not working," verify it actually produces gradients. A1 (free_bits dead zone) wasted multiple run cycles before the gradient flow was explicitly checked. The check is:
```python
grads = jax.grad(loss_fn)(params, batch)
print(jax.tree_util.tree_map(lambda g: jnp.any(g != 0), grads))
```

### Lesson 2: Phase-local step counters for all phase-conditional activations

Any time a loss or component is activated based on a step count, that count must be relative to the current phase, not the global step. B1 (catastrophic collapse) was caused by a single line that used `global_step` instead of `phase_local_step`. All threshold conditions in the training script should now use:
```python
phase_local_step = global_step - phase_start_global_step
some_condition = phase_local_step >= threshold
```

### Lesson 3: Save checkpoints every 5 epochs, not every 50

A2 (NaN) would have been survivable with a checkpoint at epoch 190 instead of epoch 150. V2 uses `save_every=5` consistently. The disk cost (each checkpoint ≈ 50MB) is negligible compared to the cost of rerunning 50 epochs.

### Lesson 4: Perceptual gradient power dominates TV

The ratio of perceptual gradient power to TV gradient power must be estimated before training, not inferred post-hoc. C1 (stripe banding) required estimating the per-pixel gradient magnitudes for both losses and discovering a 300× imbalance. A simple check:
```python
# Compute per-pixel gradient magnitude for each loss term
grad_perceptual = jnp.abs(jax.grad(perceptual_loss)(x))
grad_tv = jnp.abs(jax.grad(tv_loss)(x))
print(grad_perceptual.mean() / grad_tv.mean())
```
If the ratio is > 10×, TV is not providing meaningful regularisation.

### Lesson 5: The discriminator accuracy diagnostic

For any adversarial component:
- `disc_acc ≈ 0.50`: discriminator at chance — generator has fooled it (ideal)
- `disc_acc = 0.55–0.75`: healthy balance
- `disc_acc > 0.80`: generator losing — reduce weight_gan or temporarily disable
- `disc_acc < 0.40`: discriminator is failing — generator dominating (less common, but possible if weight_gan is very high)

Log disc_acc at every step during active GAN training.

### Lesson 6: Curriculum staging is essential for diagnosable failures

V1 combined all objectives simultaneously. When a failure occurred, it was impossible to identify which component caused it. V2's D0→D1→D2→D3 curriculum adds exactly one pressure per stage. When a failure occurs (as in B1 and B2), the causal component is unambiguous. The extra training cost of staged introduction is paid back immediately when a failure occurs and only one component needs to be diagnosed.

### Lesson 7: Resume from before the corruption, not from after

When decoder weights are corrupted by a bad objective (C1: layer3 stripe artifacts baked into D4/D5), resume from the last checkpoint before the corruption, not from the corrupted checkpoint with a fixed objective. The corrupted decoder weights carry the learned bad pattern; fixing the objective does not un-learn it.

---

## Checklist: Before Launching a New Training Phase

Use this checklist before every `sbatch` / `bash` execution.

**Invariant checks (failures cause binary kill):**
- [ ] Verify $\text{KL}(\sigma_{\text{inactive}}) > \text{free\_bits}$ (avoids A1)
- [ ] Confirm `phase_start_global_step` is set in training script if resuming (avoids B1)
- [ ] Verify checkpoint path exists and is the intended resume point (avoids resuming from corrupted weights)

**GAN checks (if enabling or changing adversarial training):**
- [ ] Verify `weight_gan` × typical GAN loss value ≤ `weight_rec` × reconstruction loss (avoids B1 imbalance)
- [ ] Set `disc_r1_penalty=0.0` unless explicitly testing R1 regularisation (avoids B2)
- [ ] Set `lr_patch_disc ≥ 1e-4` (avoids B2 discriminator trap)
- [ ] Confirm `gan_start_step` is phase-local, not global (avoids B1 immediate activation)

**Architecture checks (if changing model structure):**
- [ ] Confirm no stride-2 followed by upsample in latent branches (avoids C2)
- [ ] Confirm decoder uses SmoothUp, not transposed convolution (avoids C3)

**Perceptual loss checks (if using CheSS perceptual):**
- [ ] Confirm `perceptual_only=True` (layers 1–2 only) or explicitly checked that layer3 is excluded (avoids C1)
- [ ] Estimate gradient power ratio: `weight_perceptual × |∇perceptual|` / `weight_tv × |∇tv|` < 10× (avoids C1)

**Latent routing checks:**
- [ ] Confirm `stop_gradient` on `key_proj` for Normal images (avoids D5)
- [ ] Confirm `weight_bbox_attn > 0` if bbox cross-attention is enabled (avoids D3 drift)

**Monitoring setup:**
- [ ] W&B enabled
- [ ] `save_every ≤ 5` (avoids large checkpoint loss from A2-type NaN)
- [ ] Kill conditions noted: `patch_disc_acc > 0.90` for ≥3 epochs; `loss/reconstruction` spike > 3× baseline

---

*End of document. Continue to [07 Results & Evaluation](07_results_evaluation.md).*
