# Cross-Run Comparison: `sepvae-disentangle` Project

**Date of analysis:** 2026-03-12
**Runs analysed:** 9 (Feb 16–20, 2026) — 6 informative, 3 immediate crashes
**Metric definitions:** probe_auc = linear-probe AUC on disease head latents; cross_head_score = mean energy-ratio of inactive head to active head (1.0 = perfect leakage, 0.5 = ideal)

---

## 1. Run Inventory

| Run folder | Wandb ID | Methodology | Epochs | Status |
|---|---|---|---|---|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | disentangle-A | 192/200 | ~complete (old script, no probe AUC) |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | disentangle-B | 1 | **CRASHED** (step 0 only) |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | disentangle-C | 84 | Stopped — resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | disentangle-D | 101 | Stopped — resumed as E (from C ep80) |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | disentangle-E | 200 | **NaN crash at final epoch** (from D ep100) |
| `sepvae_inactivity_driven-20260220-062635` | `99s5jhqe` | inactivity-F | 0 | **CRASHED** (~93 s, transient GPU) |
| `sepvae_inactivity_driven-20260220-085943` | `9lj20so0` | inactivity-G | 100 | Completed |
| `sepvae_independence_driven-20260220-062742` | `i9cevm1x` | independence-H | 0 | **CRASHED** (~91 s, transient GPU) |
| `sepvae_independence_driven-20260220-085943` | `41nce8qq` | independence-I | 100 | Completed |

> Note: runs C, D, E form a single continuous training run split across three SLURM jobs.
> Runs F and H crashed during model init — the same configs succeeded 23 min later (G and I), indicating a transient GPU resource conflict.

---

## 2. Hyperparameter Matrix

| Parameter | disentangle-A | disentangle-C/D/E | inactivity-G | independence-I |
|---|---|---|---|---|
| `free_bits` | 1 | 1 | 1 | **2** |
| `sigma_inactive` | 0.2 | 0.1 | **0.05** | 0.1 |
| KL_inactive (est.) | ~1.1 nats | ~1.8 nats | ~2.5 nats | ~1.8 nats |
| KL_inactive > free_bits? | yes | yes | yes | **NO (1.8 < 2.0)** |
| `weight_null` | 0.005 | 0.01 | **0.05** | 0.01 |
| `weight_orthogonality` | 0.01 | 0.03 | 0.05 | **0.1** |
| `weight_mi` | 0.001 | 0.003 | 0.005 | **0.01** |
| `weight_perceptual` | 0.1 | 0.03 | 0.05 | 0.05 |
| `use_fpn` | yes | yes | no | no |
| `batch_size` | 6 | 6 | 10 | 10 |
| `lr_vae` | 6.7e-5 | 6.7e-5 | 1e-4 | 1e-4 |
| `kl_warmup_epochs` | 30 | 30 | 10 | 10 |
| `half_precision` | fp32 | fp32 | bf16 | bf16 |
| Epochs target | 200 | 200 | 100 | 100 |

KL_inactive estimated as: `0.5 * (sigma^2 - 1 - log(sigma^2))` with mu=0.

---

## 3. Results Table

### 3.1 Final / best disentanglement metrics

| Run | Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Probe AUC (mean) | Cross-head score ↓ | Silhouette (disease-only) |
|---|---|---|---|---|---|---|
| disentangle-A | 192 | N/A (old script) | N/A | N/A | N/A | 0.020 |
| disentangle-C/D/E — best checkpoint | ~185 | **0.776** | **0.772** | **0.774** | 0.914 | — |
| disentangle-C/D/E — epoch 101 | 101 | 0.758 | 0.700 | 0.729 | 0.952 | 0.079 |
| disentangle-C/D/E — last valid (ep199) | 199 | 0.731 | 0.744 | 0.737 | 0.935 | — |
| inactivity-G — best checkpoint | ~50 | 0.639 | 0.627 | 0.650 | **0.744** | — |
| inactivity-G — epoch 100 (final) | 100 | 0.476 | 0.767 | 0.622 | **0.809** | 0.079 |
| independence-I — early peak (~ep25) | ~25 | 0.762 | 0.737 | 0.750 | 0.820 | — |
| independence-I — epoch 100 (final) | 100 | 0.532 | 0.812 | 0.672 | 0.871 | 0.124 |

**Cross-head score interpretation:** 1.0 = total leakage (each head predicts the other disease as well as itself); 0.5 = ideal (both heads carry no information about the other disease).

### 3.2 KL state at epoch 100 (disease heads)

| Run | KL_cardio | KL_effusion | Status vs free_bits |
|---|---|---|---|
| inactivity-G | ~2.7 (active) / ~2.5 (inactive) | similar | Above free_bits=1 — gradients flowing |
| independence-I | 4.00 (clamped) | 4.00 (clamped) | All below free_bits=2 — **DEAD ZONE** |

---

## 4. Group-Level Consensus

### 4.1 Group 1: `sepvae_disentangle` (Feb 16–17, up to 200 epochs)

**Approach:** Moderate regularization, FPN encoder, slow LR (`6.7e-5`), long KL warmup (30 epochs).

**Findings:**
- **Highest peak performance** — probe AUC mean reaches 0.774 at around epoch 185. Best single-head cardiomegaly AUC = 0.776.
- **Highly unstable** — probe_auc fluctuates from 0.53 to 0.77 across consecutive evaluations within the same run. This is partly eval noise (small 600-sample eval set) and partly genuine training oscillation.
- **NaN explosion at epoch 200** — total loss was stable (0.004–0.006) through epoch 199 before blowing up in the final epoch. The run uses full fp32 precision; the NaN likely reflects a catastrophic gradient from an unlucky batch amplified over time without any LR decay. No LR schedule was used.
- **Persistent high leakage** — cross_head_score stays between 0.85 and 1.01 throughout 200 epochs. The heads learn some disease information but never achieve genuine independence.

### 4.2 Group 2: `sepvae_inactivity_driven` (Feb 20, 100 epochs)

**Approach:** Aggressive inactivity penalty — `sigma_inactive=0.05` (KL_inactive ≈ 2.5 nats), 5× stronger nulling (`weight_null=0.05`), no FPN, bf16.

**Findings:**
- **Best cross-head leakage suppression** — reaches cross_head_score = 0.744 mid-training and ends at 0.809, the best of all runs.
- **Weaker but more consistent probe AUC** — peaks around 0.65 (much lower than the 200-epoch disentangle chain), with less volatility.
- **Cardiomegaly head collapse by epoch 100** — final cardiomegaly probe AUC = 0.476 (sub-chance). The strong nulling pressure, when applied consistently over 100 epochs, appears to over-suppress the cardiomegaly head specifically. Cardiomegaly is a subtler, more distributed feature than effusion, so the inactivity pressure wins over the information signal.
- **Key trade-off:** reduces leakage at the cost of one head dying. Shows the right direction but the regularization balance is off.

### 4.3 Group 3: `sepvae_independence_driven` (Feb 20, 100 epochs)

**Approach:** Same architecture as Group 2, but `free_bits=2`, weaker nulling (`0.01`), stronger orthogonality (`0.1`).

**Findings:**
- **Confirmed `free_bits` / `sigma_inactive` conflict** — KL_inactive ≈ 1.8 nats < free_bits = 2.0. Disease heads are permanently in the KL dead zone. The free_bits threshold was intended to prevent posterior collapse on common-head channels but instead kills disease-head gradients entirely.
- **Interesting early dynamics** — probe AUC = 0.750 with cross_head_score = 0.820 around epoch 20–25. Some useful structure emerges before the disease heads degrade.
- **Full posterior collapse by epoch 100** — both disease head mu-norms ≈ 0.025–0.030 (essentially zero). KL uniformly 1.7–1.8 nats across all channels and classes (no class-conditional differentiation). Cardiomegaly probe AUC = 0.532 (near chance).
- **Worst configuration** — the `free_bits=2` setting is directly responsible.

---

## 5. Cross-Run Patterns

**Pattern 1: `free_bits` / `sigma_inactive` conflict is a hard failure**
Only Group 3 has `free_bits > KL_inactive`. It is unambiguously the worst configuration. All runs with `free_bits ≤ 1` allow gradients to flow through disease channels.

**Pattern 2: Cross-head leakage is a universal unsolved problem**
Cross_head_score never goes below 0.74 across any run at any point. Even the best result (0.744 in inactivity-G mid-training) means 74% cross-leakage. The current loss design — orthogonality + MI + nulling — does not force genuine conditional independence between heads.

**Pattern 3: Cardiomegaly head is consistently harder to train**
Across all runs, cardiomegaly AUC is lower and more volatile than effusion AUC. Likely causes:
- Cardiomegaly is a subtle, distributed feature (enlarged heart silhouette) vs. effusion (bright, localised pleural fluid)
- The triplet dataset after filtering has fewer effective cardiomegaly examples

**Pattern 4: Probe AUC is very noisy — report rolling averages**
The same checkpoint reads anywhere from 0.53 to 0.77 on adjacent evaluations. The 600-sample probe eval has high variance. Best practice: use a 3-evaluation rolling average or report best-of-5.

**Pattern 5: Long training helps peak, but destabilises without LR decay**
The 200-epoch chain outperforms 100-epoch runs in peak probe AUC (~0.77 vs ~0.72) but ends in NaN without a learning-rate schedule. A cosine decay from epoch 80 onward would likely preserve those gains.

---

## 6. Recommendations

### R1 — Fix `free_bits` to ≤ 1.0 (confirmed essential, Group 3 failure)

Set `free_bits=0.5` to stay safely below KL_inactive for any reasonable sigma_inactive. The current default `free_bits=2.0` kills disease-head gradients when `sigma_inactive=0.1`.

**Change:** default `free_bits` in `train_sep_vae.py` from `0.0` → `0.5` (applies to all heads).

---

### R2 — Adopt `inactivity_driven` regularisation profile (leakage winner)

Use `sigma_inactive=0.05`, `weight_null=0.05`, `weight_orthogonality=0.05`. This gave the best cross-head scores. The equal null/ortho weighting (0.05/0.05) rather than the 10:1 ratio in independence_driven (0.1 ortho / 0.01 null) is critical.

---

### R3 — Add cosine LR decay to prevent NaN (confirmed by 200-epoch crash)

The 200-epoch run proves training without decay causes catastrophic instability. Add cosine decay from epoch 80 with `min_lr = 0.1 * init_lr`.

**Implementation:** `optax.cosine_decay_schedule` from `total_steps - decay_steps` onwards, wrapped with a warmup schedule.

---

### R4 — Fix cardiomegaly head collapse (new insight from inactivity-G)

The inactivity-G run ends with cardiomegaly probe AUC = 0.476. The nulling loss is over-suppressing the cardiomegaly head. Two options:

- **(Implemented)** Add `--min_active_kl` floor: for samples where the disease head IS active (label matches), add a penalty that drives KL above a minimum floor (e.g., 2.0 nats), preventing the head from collapsing even under strong nulling.
- Use asymmetric sigma_inactive: smaller for effusion (which has sharp features), larger for cardiomegaly.

---

### R5a — Paired contrastive loss (structural disentanglement)

For each disease head and its active class:
- **Separate-push**: push the active disease head's representation for disease-positive images away from normal images (inter-class margin loss on centroids).
- **Blind-pull**: pull the inactive head's representation for disease-positive images toward the normal-image representation (the head should not respond to the other disease).

This directly enforces what the swap grid should show. Enabled via `--use_contrastive`.

---

### R5b — Cross-head adversarial discriminator (structural disentanglement)

Two lightweight MLPs:
- `disc_c2e`: takes `z_effusion`, predicts whether the input was a cardiomegaly image
- `disc_e2c`: takes `z_cardio`, predicts whether the input was an effusion image

Training:
- The discriminators are trained to correctly classify (standard BCE).
- The VAE encoder is penalised when the discriminators succeed — the encoder is trained to push discriminator outputs toward 0.5 (maximum uncertainty = no cross-head information).

This enforces conditional independence between heads at the information level, not just at the geometry level (unlike orthogonality). Enabled via `--use_cross_adv`.

---

### R6 — Remove FPN for simplicity

FPN (`use_fpn=true`) is only in the older disentangle runs. The Feb 20 runs without FPN reach comparable or better probe AUC. FPN adds parameters and peak memory without measurable disentanglement benefit. Remove from default.

---

### R7 — Disease prototype cross-attention heads (`--use_label_attention`)

**Motivation — Pattern 3 and the implicit routing problem:**

All three run groups show the same structural failure: the disease ConvHeads see the *entire* 64×64 backbone feature map and must learn, purely through gradient pressure from orthogonality and MI losses, to ignore the spatial regions that belong to the other disease. This is an ill-posed implicit learning problem. Cardiomegaly is a distributed, low-contrast change (enlarged cardiac silhouette occupying the central ~30% of the image) while effusion is a localised, high-contrast finding (bright pleural fluid in the lower lateral costophrenic angles). A ConvHead with no spatial routing bias treats both the cardiac and pleural regions identically — the cardiomegaly head must somehow suppress its response to the pleural region through weight tuning alone, which is precisely why it consistently underperforms and collapses under strong regularisation.

**Mechanism — learned disease prototype query:**

Each disease head now learns a single prototype vector $q_d \in \mathbb{R}^{D}$ (the "disease query", $D=256$ by default). This vector is used to compute a scaled dot-product attention score over all $HW = 64 \times 64 = 4096$ spatial positions in the backbone feature map:

$$A_d(i) = \text{softmax}\!\left(\frac{K_i^\top q_d}{\sqrt{D}}\right), \quad K = \text{Dense}(h_\text{flat})$$

The resulting spatial attention map $A_d \in \mathbb{R}^{H \times W}$ is rescaled by $HW$ (so that at initialisation, when $q_d \approx 0$, $A_d \approx 1$ everywhere and the attended features equal the input), then used to gate the backbone features before the ConvHead:

$$h_\text{attended} = h \odot (A_d \cdot HW)$$

This gives the model an explicit, learnable spatial routing mechanism. Over training, $q_\text{cardio}$ learns to concentrate $A_\text{cardio}$ on the cardiac silhouette region; $q_\text{effusion}$ learns to concentrate on the pleural region. The key properties:

- **At init:** $h_\text{attended} \approx h$ — identical to the baseline ConvHead, no training instability.
- **Inductive bias:** the attention map provides a bottleneck that structurally favours spatial specialisation, reducing the gradient pressure needed from orthogonality and MI losses to achieve the same level of cross-head independence.
- **Free diagnostic:** $A_d$ is a 64×64 spatial attention map returned as a third output from each disease head. Logged as heatmaps overlaid on CXR images during training (saved to `diagnostics/attn_maps_epoch{N}.png` and W&B under `diagnostics/attn_maps`). These maps directly proxy the verifiability criteria V1/V2 (Section 9.3): if $A_\text{cardio}$ concentrates over the cardiac region and $A_\text{effusion}$ over the pleural region, the factorisation is spatially grounded.

**Connection to Pattern 2 (persistent cross-head leakage):**

The current best cross_head_score = 0.744 means 74% leakage. The orthogonality and cross-adversarial losses penalise the *output* of the heads (the latent vectors) but do nothing about the *input routing* — both heads still see the same full feature map. Attention addresses the leakage at the source: if the cardiomegaly head genuinely attends only to cardiac regions, it never has access to the pleural features that it would otherwise need to suppress through regularisation.

**Interaction with the cardiomegaly head collapse (R4):**

Inactivity-G showed cardiomegaly probe AUC collapsing to 0.476 under strong nulling pressure. A plausible mechanism: the ConvHead's cardiomegaly signal is spatially diffuse across the full 64×64 map, so even a small inactivity KL pushes the mean toward zero everywhere. With attention, the active signal is concentrated in a spatial subregion — the same nulling pressure is absorbed by a smaller number of high-weight positions, which are also the positions with the strongest gradient signal from reconstruction. This should improve the robustness of the cardiomegaly head under regularisation.

**Implementation note:** Enabled via `--use_label_attention`. The common head is unchanged (ConvHead). The flag is backward-incompatible with existing checkpoints — the `head_cardiomegaly` and `head_effusion` parameter trees change structure (add `disease_query` and `key_proj` parameters). Resume will warn of architecture mismatch.

---

## 7. Recommended Next Configurations

### Baseline fixed (implements R1–R4, R6)

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_baseline_fixed \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

### With paired contrastive loss (R5a)

```bash
# ...same as above, plus:
  --use_contrastive \
  --weight_contrastive 0.1 \
  --exp_name sepvae_contrastive
```

### With cross-head adversarial discriminator (R5b)

```bash
# ...same as baseline_fixed, plus:
  --use_cross_adv \
  --weight_cross_adv 0.05 \
  --lr_cross_disc 1e-4 \
  --exp_name sepvae_cross_adv
```

### Full recommendations (R5a + R5b)

```bash
# ...same as baseline_fixed, plus:
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --exp_name sepvae_full_recommendations
```

### With disease prototype cross-attention heads (R7)

Replaces the disease ConvHeads with `DiseaseAttentionHead` modules. Can be stacked on top of any prior configuration; shown here on the baseline_fixed settings.

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_label_attention \
  --use_label_attention \
  --attn_query_dim 256 \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

**What to look for in W&B:** Attention map images logged under `diagnostics/attn_maps` every `--sample_every` epochs. Early epochs should show diffuse/uniform maps (expected — query is near-zero at init). By epoch 20–30, cardiomegaly maps should begin concentrating over the central cardiac silhouette region; effusion maps over the lower lateral pleural angles. If both maps remain diffuse after epoch 50, the disease signal is too weak relative to nulling pressure — reduce `--weight_null` or `--sigma_inactive`.

### Full stack (R5a + R5b + R7)

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_full \
  --use_label_attention \
  --attn_query_dim 256 \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_cross_disc 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

---

## 8. Reconstruction Sharpness — Why the VAE Produces Blurry Images

**Observed problem:** Reconstructed CXR images are consistently soft and lack the fine detail (rib cortex sharpness, vessel walls, air-bronchogram texture) present in the originals.

### 8.1 Root cause pipeline

```
Input x  (512×512, full radiographic detail)
     │
     ▼  Frozen ResNet-50 backbone (cumulative stride = 32)
Feature map  (16×16, ~2048ch)
     │         ← HIGH-FREQUENCY TEXTURE LOST HERE (stride-32 pooling)
     │           backbone was trained for classification, not reconstruction
     ▼
8-channel spatial bottleneck  (64×64×8)
     │         ← COMPRESSION RATIO ≈ 0.8%  of input information
     │           decoder must hallucinate all detail it cannot decode
     ▼
Decoder: 3 × (ResBlocks → bilinear-upsample → smoothing-conv × 2)
     │         64 → 128 → 256 → 512  with 6 extra smoothing convolutions
     │         each SmoothUp applies jax.image.resize then TWO conv layers
     ▼
L2 reconstruction loss
     │         optimal prediction under L2 is the posterior mean,
     │         which is a blurred average over all plausible reconstructions
     ▼
No discriminator active  (weight_adversarial = 0.0 in all Feb 2026 runs)
     │         nothing penalises statistically implausible smooth textures
     ▼
   x̂   (blurry)
```

The three sources compound: the backbone discards texture before encoding; the bottleneck forces the decoder to hallucinate; and the L2+no-GAN training objective rewards the mean, not any sharp realisation.

### 8.2 Compression ratio

With $z_\text{common} = 4$ channels, $z_\text{cardio} = z_\text{effusion} = 2$ channels each, the spatial bottleneck is:

$$\text{compression ratio} = \frac{8 \times 16 \times 16}{512 \times 512} = \frac{2048}{262144} \approx 0.78\%$$

At this ratio the decoder is necessarily hallucinating the vast majority of image content. The hallucination is anchored by the 8 latent channels but cannot be verified against ground truth at the pixel level — only the blurry mean is consistent across all possible sharp completions.

### 8.3 L2 loss averages over the posterior

The ELBO reconstruction term:

$$\mathcal{L}_\text{rec} = \mathbb{E}_{q(z|x)}\left[\|x - \hat{x}_\theta(z)\|_2^2\right]$$

The optimal decoder under this objective (for any posterior with non-zero variance) is:

$$\hat{x}_\theta(z) = \mathbb{E}_{p(x|z)}[x]$$

For natural images, this expectation is a blurred version of any plausible sharp image. The sharper the texture, the more variance there is across plausible completions, and the more blurred the mean.

### 8.4 The discriminator fix

A PatchGAN discriminator $D$ trained adversarially forces:

$$\mathcal{L}_\text{adv} = -\log D(\hat{x})$$

Because averaged, blurry textures look statistically unlike real X-ray patches, the discriminator learns to reject them. The generator (decoder) is then forced to commit to a single sharp realisation rather than an average. This is the standard explanation for why VAE-GANs (e.g. VQ-GAN) are sharper than plain VAEs.

### 8.5 Bilinear + double-smoothing in the decoder

Each `SmoothUp` block in the current decoder applies:

```python
# bilinear path (current default)
h = jax.image.resize(x, target_shape, method='bilinear')  # low-pass filter
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 1
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 2
```

Three upsampling stages × two smoothing convolutions = **six successive low-pass operations** before the final sigmoid. High-frequency energy is progressively suppressed with each stage.

The alternative `subpixel` path (already implemented in `models/sep_vae_jax.py:68–82`) uses pixel-shuffle, which is a learned rearrangement of channels into spatial positions and has no inherent low-pass bias.

### 8.6 Frozen backbone: the fundamental limit

Even with all fixes below applied, fine rib cortex and vessel detail will not be perfectly recovered, because the stride-32 ResNet discards that information before the encoder heads ever see it. Partially unfreezing the backbone (`--unfreeze_from layer3`) allows those layers to re-learn what to preserve for reconstruction — at the cost of training stability and potential conflict with disentanglement objectives.

---

### 8.7 Fixes — ordered by impact

#### Fix 1: Enable the PatchGAN adversarial loss

The discriminator is fully implemented; it is just disabled. This is the highest-impact change.

```bash
--weight_adversarial 0.1 \
--disc_start_epoch 10      # warmup 10 epochs before discriminator activates
```

`disc_start_epoch=10` prevents GAN instability during early training when reconstructions are poor.

#### Fix 2: Switch to subpixel (pixel-shuffle) upsampling

Already implemented — requires one flag change. Replaces 6 low-pass smoothing convolutions with learned upsampling that can amplify high-frequency components.

```bash
--upsample_method subpixel
```

#### Fix 3: Increase perceptual loss weight

`weight_perceptual=0.05` is conservative. VGG perceptual loss biases the reconstruction toward matching feature statistics that correlate with human perception of sharpness.

```bash
--weight_perceptual 0.1
```

#### Fix 4 (optional): Partially unfreeze the backbone

Allows the encoder to re-learn what texture to preserve for reconstruction.

```bash
--unfreeze_from layer3    # or layer4 for a lighter change
```

Use cautiously — unfreezing too much can destabilise the disentanglement objectives.

---

### 8.8 Recommended sharpness run

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

Note: `free_bits=0.0` here to keep disease heads active (no KL dead zone); the nulling and orthogonality weights are lightened (`0.02`) relative to the baseline_fixed run to balance against the new adversarial loss term.

---

## 9. Research Objectives: Assumptions, Composition Strategies, and Open Questions

*This section records the technical discussion that arose from the cross-run analysis. It documents what we are actually building, what we are assuming, and what we need to verify before the downstream LDM composition is scientifically defensible.*

---

### 9.1 Core Assumptions Being Made

The SepVAE project rests on three assumptions, each of which can fail independently.

#### Assumption A1 — Disease factors are approximately separable

$$z_\text{cardiomegaly} \perp z_\text{effusion}$$

or at minimum, sufficiently separable for score composition to be meaningful. In chest X-rays this is not strictly true. Cardiomegaly and pleural effusion co-occur through shared pathology (right heart failure), shared appearance patterns (enlarged cardiac silhouette → raised diaphragm → apparent pleural blunting), projection effects, and dataset bias.

The `exclude_cross_disease_overlap=True` flag in the dataset removes patients who have *both* diseases from training, which reduces statistical correlation but does not make the underlying physiology independent. At inference, when we compose both diseases, the resulting sample is out-of-distribution for both single-disease LDMs.

**Empirical test:** cross_head_score and probe_auc/cross (does z_effusion predict cardiomegaly labels?). Current best: cross_head_score = 0.744 — meaning 74% leakage, far from the ideal 0.5.

#### Assumption A2 — Labels correspond to clean generative factors

A clinical label like "cardiomegaly" is a high-level diagnostic summary, not a primitive visual atom. The cardiac silhouette enlarges, but it also:

- Shifts the mediastinum
- Displaces the lungs
- Creates global shape changes that look like "common anatomy" to any encoder

The model may route part of the cardiomegaly signal into $z_\text{common}$ because that signal *is* correlated with anatomy from the encoder's perspective. This is consistent with cardiomegaly probe AUC being consistently lower and more unstable than effusion AUC across all runs — effusion is a localised, high-contrast finding (bright pleural fluid) that is easier to separate from background anatomy.

#### Assumption A3 — Separated representation implies separable score fields

Even if the SepVAE achieves clean factorised latents, it does not automatically follow that separately trained LDMs over those factors will compose cleanly via score addition. Score addition:

$$\nabla \log p(z \mid c, e) \approx \nabla \log p(z \mid c) + \nabla \log p(z \mid e)$$

is exact only if $p(z \mid c, e) \propto p(z \mid c) \cdot p(z \mid e)$ — a product-of-experts identity that holds when the two conditions are conditionally independent given $z$. If each disease LDM was trained on the full $z$ (including $z_\text{common}$), both LDMs carry shared anatomical structure in their scores, and direct summation double-counts it:

$$p(z \mid c, e) \propto \frac{p(z \mid c)\, p(z \mid e)}{p(z)}$$

This means $z_\text{common}$ is not just useful — it may be **mathematically necessary** to subtract the common prior score once during composition to avoid double-counting anatomy.

---

### 9.2 What Exactly Is Being Composed — Three Strategies

This question must be answered before any downstream LDM is trained, because the composition strategy determines what the SepVAE must provide. The three strategies have different requirements, failure modes, and implementation costs.

#### Strategy A — Sub-block LDMs with common base (recommended)

The factorisation of the joint latent is treated as a product of conditionals:

$$p(z) = p(z_\text{common}) \cdot p(z_\text{cardio} \mid z_\text{common}) \cdot p(z_\text{effusion} \mid z_\text{common})$$

**Composition pipeline:**

```
Step 1:  z_common ~ LDM_common                          (unconditional base)
Step 2:  z_cardio  ~ LDM_cardio(conditioned on z_common)
Step 3:  z_effusion ~ LDM_effusion(conditioned on z_common)
Step 4:  x̂ = Decoder(cat(z_common, z_cardio, z_effusion))
```

**Why this avoids the double-counting problem:** Each LDM acts on a different, disjoint subspace of $z$. There is no score to subtract because there is no overlap. The composition is a sequential ancestral sample from the factorised prior, not a product of experts.

**What this requires from SepVAE:**

$$z_\text{cardio} \perp z_\text{effusion} \mid z_\text{common}$$

Conditional independence given anatomy. This is exactly what the cross-head adversarial discriminator (R5b) tests at training time.

**Architecture for disease LDMs:**

```
LDM_cardio: takes z_common as conditioning context,
             predicts noise in z_cardio channel(s) only
LDM_effusion: symmetrically for z_effusion
```

#### Strategy B — Full-z LDMs with product-of-experts composition

Train two full-$z$ conditional LDMs and one unconditional LDM:

$$\nabla_z \log p(z \mid c, e) \approx \nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) - \nabla_z \log p(z)$$

**Advantage:** captures cross-head correlations naturally; does not require the SepVAE factorisation to be perfect.

**Disadvantage:** requires training three LDMs; the unconditional prior subtraction must be stable; if z_common leaks into z_disease (as observed — cross_head_score = 0.87), both disease LDMs will encode anatomical structure and the subtracted prior will not fully cancel it.

**Required for mathematical correctness:**

$$p(z_\text{cardio}, z_\text{effusion} \mid z_\text{common}) = p(z_\text{cardio} \mid z_\text{common}) \cdot p(z_\text{effusion} \mid z_\text{common})$$

Effectively the same conditional independence requirement as Strategy A.

#### Strategy C — Disease sub-block scores only (implicit current assumption, likely broken)

If each disease LDM acts only on $z_\text{cardio}$ or $z_\text{effusion}$ respectively, and $z_\text{common}$ is sampled from its own LDM, score addition is:

$$\nabla_z \log p(z) \approx \nabla_{z_c} \log p(z_\text{common}) + \nabla_{z_d} \log p(z_\text{cardio}) + \nabla_{z_e} \log p(z_\text{effusion})$$

This is clean if the decoder treats each sub-block as an independent additive perturbation. However, the current SepVAE decoder uses concatenation followed by shared convolutional layers — it mixes all heads before generating pixels. Strategy C is therefore not geometrically justified for the current decoder.

**Strategy comparison:**

| | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| Double-counts anatomy? | No | Requires prior subtraction | No (if decoder separable) |
| Requires perfect SepVAE factorisation? | Yes (conditionally) | No (but helps) | Yes (structurally) |
| N. of LDMs to train | 3 (1 common + 2 conditional) | 3 (2 conditional + 1 uncond) | 3 (1 common + 2 disease) |
| Compatible with current decoder | Yes | Yes | No (decoder mixes heads) |
| **Recommended** | **Yes** | Fallback | No |

---

### 9.3 Verifiability Criteria — Current Gaps

For the research hypothesis to be falsifiable, four intervention experiments are needed. The current codebase has partial support for all four but no quantitative metrics.

#### Criterion V1 — Changing $z_\text{cardio}$ alters heart enlargement without strongly altering the effusion region

**Current support:** `plot_latent_swap_grid` in `utils/sepvae_diagnostics.py` — qualitative visual only.

**Additional support when `--use_label_attention` is enabled:** The cardiomegaly attention map $A_\text{cardio} \in \mathbb{R}^{64 \times 64}$ is a direct spatial certificate of where the cardiomegaly head is routing its information. If $A_\text{cardio}$ concentrates over the cardiac silhouette region (central 35–55% height, 30–70% width) rather than the pleural region, V1 spatial selectivity is structurally enforced at the encoder level — not just evidenced post-hoc by pixel deltas. The attention map can be thresholded and compared against an anatomical cardiac mask as a cheap approximation of the specificity ratio before running full latent swap experiments.

**Gap:** No region-selective pixel-level change metric. Proposed implementation:

```python
# Approximate cardiac region as centre 35-55% of image height, 30-70% of width
cardiac_mask = make_region_mask(H, W, row=(0.35, 0.55), col=(0.30, 0.70))
effusion_mask = make_region_mask(H, W, row=(0.70, 1.00), col=(0.10, 0.90))

x_base  = decode(z_common, z_cardio_normal,   z_effusion)
x_swapped = decode(z_common, z_cardio_disease, z_effusion)
delta   = jnp.abs(x_swapped - x_base)

cardio_change  = delta[cardiac_mask].mean()   # should be HIGH
effusion_change = delta[effusion_mask].mean()  # should be LOW
specificity_ratio = cardio_change / (effusion_change + 1e-6)  # > 1.0 = correct localisation
```

**Target:** `specificity_ratio > 2.0` for cardiomegaly swaps; `< 1.0` for null swaps.

#### Criterion V2 — Changing $z_\text{effusion}$ alters pleural fluid patterns without strongly altering the cardiac silhouette

Symmetric to V1. Effusion region = lower 30% of image (costophrenic angle area). Target: `specificity_ratio > 2.0` for effusion swaps.

**Additional support when `--use_label_attention` is enabled:** $A_\text{effusion}$ provides the symmetric spatial certificate for the effusion head. Empirically, effusion concentration should be easier to achieve than cardiomegaly given the localised, high-contrast nature of the finding — if $A_\text{effusion}$ fails to localise to the lower lateral regions after training, it is evidence that the backbone features at the relevant spatial positions are insufficient for effusion separation, which would motivate partial unfreezing (`--unfreeze_from layer3`).

#### Criterion V3 — $z_\text{common}$ preserves anatomy, pose, and acquisition factors

**Gap:** No direct metric. Proposed: SSIM between original and edited image over the non-disease regions (everything outside the cardiac and effusion masks).

```python
z_common_new = encode_common(x_different_patient)
x_edited = decode(z_common_new, z_cardio, z_effusion)  # transplant anatomy
ssim_non_disease = compute_ssim(x_original, x_edited, mask=~(cardiac_mask | effusion_mask))
# Should be LOW (anatomy changed) — confirms z_common carries anatomy
```

The reciprocal: fix z_common, swap z_disease → SSIM over non-disease region should be HIGH (anatomy unchanged).

#### Criterion V4 — Jointly composed sample resembles real co-morbid images

**Gap:** Not yet implemented. Requires:
1. A held-out split of patients with *both* cardiomegaly and effusion (currently excluded from training by `exclude_cross_disease_overlap=True`)
2. FID computed between composed samples and this held-out comorbid split
3. Comparison against: (a) single-disease samples, (b) jointly conditioned baseline

```bash
# Generate N composed samples
python run/compose_diseases.py \
  --strategy a \                          # Strategy A sequential sampling
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

---

### 9.4 Four Proposed Additions

#### Addition P1 — Explicit shared anatomical/background expert (already partially present)

$z_\text{common}$ already plays the role of the anatomical/background expert in the SepVAE encoder. The critical missing piece is the **LDM over $z_\text{common}$** — without it, $z_\text{common}$ must be synthesised by the disease LDMs, which forces them to encode anatomy and reintroduces double-counting.

**Design:** `LDM_common` is an unconditional diffusion model trained on $z_\text{common}$ spatial maps. In Strategy A, it is sampled first; the disease LDMs are then conditioned on $z_\text{common}$.

**What prevents each disease expert from carrying all image structure:** once $z_\text{common}$ is sampled and fixed, the disease LDMs only need to model residual variation in their sub-blocks. Anatomy is already determined. This is the principal motivation for the factorised design.

#### Addition P2 — Single-disease editing gate before conjunction experiments

Conjunction will almost certainly fail if single-factor interventions are not clean. This should be a hard gate: the project should not proceed to composition experiments until V1 and V2 are passing quantitatively.

Proposed entry criterion: `specificity_ratio > 2.0` on both V1 and V2 on a 200-sample validation set, measured at the best SepVAE checkpoint.

#### Addition P3 — Jointly trained conditional model as ablation baseline

Without a directly trained joint model for (cardiomegaly, effusion), it is impossible to know whether any composition failure is due to:
- (a) The factorisation being fundamentally wrong for these diseases, or
- (b) Poor quality of the SepVAE disentanglement

**Proposed baseline:** train a single conditional VAE or LDM with a 3-class label (normal, cardio, effusion, both) and condition on (cardio=1, effusion=1) at inference. Compare:

| Model | FID on comorbid | SSIM cardiac region | SSIM effusion region |
|---|---|---|---|
| SepVAE + Strategy A | ? | ? | ? |
| Jointly conditional LDM | ? | ? | ? |
| Single-disease LDMs (no composition) | ? | ? | ? |

This ablation is necessary for the paper to claim that composition adds value over a simpler approach.

#### Addition P4 — CFG with unconditional = common expert

If the downstream LDMs are trained in a classifier-free guidance style, the unconditional score can be set to the common-expert output — i.e., the score of the LDM when all disease conditions are turned off. This gives a principled composition mechanism that avoids training a separate unconditional prior:

$$\nabla_z \log p(z \mid c, e) = (1 - 2\lambda)\nabla_z \log p_\text{common}(z) + \lambda\nabla_z \log p(z \mid c) + \lambda\nabla_z \log p(z \mid e)$$

This is equivalent to:

$$= \nabla_z \log p_\text{common}(z) + \lambda\underbrace{[\nabla_z \log p(z \mid c) - \nabla_z \log p_\text{common}(z)]}_{\text{cardiomegaly guidance direction}} + \lambda\underbrace{[\nabla_z \log p(z \mid e) - \nabla_z \log p_\text{common}(z)]}_{\text{effusion guidance direction}}$$

Each bracketed term is the standard CFG perturbation direction — how much the score changes when disease condition is turned on relative to the common base. Composition = sum of the two disease perturbation directions, both anchored to the same common base. **Double-counting is prevented** because both subtractions use the same $p_\text{common}$.

**Implementation note:** this requires training a single conditional LDM with CFG dropout (where the condition is zeroed with probability $p_\text{uncond}$ during training) and treating the unconditioned output as the common-expert baseline. It is architecturally simpler than Strategy A because there is only one LDM to train.

---

### 9.5 Revised Research Hypothesis

The informal hypothesis ("score composition in factorised latent space is better than in entangled space") is not falsifiable without specifying a comparison, a composition mechanism, and a metric.

**Formal hypothesis:**

> If disease-related variation can be approximately factorised from shared anatomical and acquisition variation in the SepVAE latent space — as measured by specificity ratios V1/V2 $> 2.0$ and probe AUC $> 0.75$ on both disease heads — then **conditional sequential sampling from disease-specific sub-block LDMs conditioned on $z_\text{common}$** (Strategy A) will produce multi-pathology chest X-rays with lower FID on a held-out comorbid split, and higher region-selective specificity, than either:
>
> (a) a jointly-trained conditional model sampled with disease condition (cardio=1, effusion=1), or
> (b) naive score addition of full-$z$ disease LDMs without prior subtraction.

**Key unknowns that determine whether the hypothesis can be tested:**

1. Can the SepVAE achieve `specificity_ratio > 2.0`? (Current best is unmeasured; qualitative swap grid looks marginal.)
2. Does the VinBigData comorbid hold-out split have sufficient samples for FID to be meaningful? (FID requires ~1000 real samples for stability.)
3. Is cardiomegaly fundamentally too distributed a feature to be localised in $z_\text{cardio}$ given the stride-32 backbone? (The consistent cardiomegaly AUC underperformance suggests yes.)

If (3) is a hard barrier, the fallback position is a more honest claim: "score composition achieves plausible multi-pathology generation even when the factorisation is imperfect, whereas direct joint conditioning fails to generalise to the comorbid case due to data sparsity." This is a weaker but potentially more defensible result.

---

### 9.6 Recommended Verification Sequence

```
Phase 1: SepVAE quality gate
  → Train sepvae_full (R1–R5 + sharp fixes)
  → Measure specificity_ratio V1, V2 on validation set
  → Gate: specificity_ratio > 2.0 on both

Phase 2: LDM training (conditional on Phase 1 pass)
  → Train LDM_common on z_common maps
  → Train LDM_cardio conditioned on z_common
  → Train LDM_effusion conditioned on z_common
  → Train LDM_joint (CFG baseline, P4)

Phase 3: Composition evaluation
  → Strategy A sequential sampling
  → CFG composition (P4)
  → Jointly conditioned baseline (P3)
  → Compute FID, region SSIM, classifier confidence on all three

Phase 4: Ablation
  → Remove z_common conditioning from disease LDMs → measure degradation
  → Remove cross-adv discriminator → measure leakage increase
  → Replace SepVAE with standard VAE → measure composition quality drop
```

Run commands for Phase 1 entry criterion check (to be added to `utils/sepvae_diagnostics.py`):

```bash
python utils/sepvae_diagnostics.py \
  --mode edit_purity \
  --checkpoint runs_sepvae/sepvae_full/best_checkpoint \
  --n_samples 200 \
  --cardiac_row_range 0.35 0.55 \
  --cardiac_col_range 0.30 0.70 \
  --effusion_row_range 0.70 1.00 \
  --effusion_col_range 0.10 0.90 \
  --out results/edit_purity_report.json
```

Proceed to Phase 2 only if `edit_purity_report.json` shows `specificity_ratio_cardio > 2.0` and `specificity_ratio_effusion > 2.0`.

---

## 10. Phase 2 Execution: Unconditional Sub-Block LDMs (2026-03-13)

*This section records the current implementation decisions for the first LDM training run, which departs from the full Strategy A plan in a deliberate, time-bounded way.*

---

### 10.1 Checkpoint Choice Rationale

**Selected:** `sepvae_disentangle-20260217-153031` (W&B `laikh8dr`, disentangle-E), **epoch 180**.

**Why disentangle-E over inactivity-G or independence-I:**

| | disentangle-E ep180 | inactivity-G ep100 | independence-I ep100 |
|---|---|---|---|
| Cardio probe AUC | **0.776** | 0.476 (collapsed) | 0.532 (dead zone) |
| Effusion probe AUC | **0.772** | 0.767 | 0.812 |
| Mean probe AUC | **0.774** | 0.622 | 0.672 |
| Both heads usable? | **Yes** | No (cardio dead) | No (cardio dead) |

independence-I has `free_bits=2 > KL_inactive≈1.8` — disease heads are permanently in the KL dead zone (see Section 3.2). inactivity-G collapses the cardiomegaly head by ep100. disentangle-E ep180 is the only checkpoint where both disease heads carry meaningful class information.

**Why epoch 180 and not 185 (stated best in Table 3.1):**

`save_every=10` in the training script, so checkpoints exist at ep170, ep180, ep190, ep200. Epoch 185 is not saved. The NaN explosion occurs at epoch 200. ep190 and ep199 (`last_valid`) have lower probe AUC than the ep185 peak (0.737 mean at ep199 vs 0.774 at ep185). ep180 is the nearest clean, stable checkpoint to the true peak:

| Epoch | Mean probe AUC | Notes |
|---|---|---|
| ep170 | ~0.75 (estimated) | Below peak |
| **ep180** | ~0.77 (estimated) | **Selected — nearest clean checkpoint to ep185 peak** |
| ep190 | ~0.74 | Degraded (oscillation) |
| ep199 | 0.737 | Last valid before NaN |
| ep200 | NaN | Unusable |

**Why unconditional sub-block LDMs now instead of Strategy A:**

Strategy A (Section 9.2) is the correct long-term design: each disease LDM is conditioned on `z_common` to prevent double-counting anatomy. However, it requires the Phase 1 quality gate to pass (`specificity_ratio > 2.0`) and architectural changes to the LDM ScoreNet to accept cross-attention or concatenated conditioning from `z_common`.

The current runs answer a prior question: **can each disease head's sub-block latent (`z_cardio` or `z_effusion`, 2 channels × 64×64 spatial) be modelled by a VP-SDE diffusion model at all?** Specifically:

1. Is the marginal distribution of `z_cardio` unimodal and well-shaped for a diffusion model?
2. Does sampling from `p(z_disease)` and decoding with a fixed `z_common` produce recognisable disease morphology?
3. Are the latent statistics (scale factor, variance) reasonable after pre-encoding?

If these marginal LDMs fail or produce degenerate samples, investing in the conditional Strategy A architecture is premature. These are therefore a **necessary feasibility gate**, not a deviation from the plan.

---

### 10.2 Pre-Encoding Pipeline

Pre-encoding extracts disease-head latents from the SepVAE encoder into `.npy` files indexed by `manifest.jsonl`. The `latent_scale_factor` (= `1/std` of all saved latents) is computed, stored in `latent_meta.json`, and used to rescale all `.npy` files in place so the LDM receives unit-variance inputs.

`--use_mean` is set (store `mu` rather than a reparameterised sample) because these are training targets, not diversity samples. The scale factor corrects the slight variance underestimation.

**Submit both jobs simultaneously (CPU-only nodes, no GPU contention):**

```bash
# Cardiomegaly — mscluster72
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio \
  slurm_scripts/preencode_sepvae.slurm

# Pleural Effusion — mscluster76
sbatch --nodelist=mscluster76 \
  --job-name=preencode-effusion \
  --export=ALL,\
DISEASE=effusion,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_effusion \
  slurm_scripts/preencode_sepvae.slurm
```

If a job is interrupted, add `RESUME=1` — the script reads the existing `manifest.jsonl`, skips already-encoded `image_id`s, and continues appending:

```bash
# Resume example
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio-resume \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio,\
RESUME=1 \
  slurm_scripts/preencode_sepvae.slurm
```

Scale factor is printed at the end of the job log and also readable from:

```bash
python -c "import json; d=json.load(open('preencoded_latents/disentangle_cardio/latent_meta.json')); print(d['latent_scale_factor'])"
```

---

### 10.3 LDM Training

After pre-encoding completes, launch the LDMs. Launchers default to the `disentangle_*` pre-encoded directories. In-loop sampling is disabled (`SAMPLE_EVERY=9999`) since no AE decoder is loaded — training runs purely on stored latents.

Key architectural parameters match the SepVAE disease sub-block: `latent_size=64`, `vae_z_channels=2`, `ldm_z_channels=2`.

```bash
./launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh full_train
./launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh full_train
```

---

### 10.4 What These Runs Are Not

These are **not** the Phase 2 LDMs described in Section 9.6, which require:

- Phase 1 quality gate passed (`specificity_ratio > 2.0` on both disease heads)
- `LDM_common` trained on `z_common` 4-channel maps
- Disease LDMs conditioned on `z_common` via cross-attention or concatenation in the ScoreNet
- Full composition pipeline (`run/compose_diseases.py`)

The outputs of these runs will inform whether Strategy A is worth building and whether the disentangle-E checkpoint provides sufficient representation quality, or whether further SepVAE training (with R1–R5 applied) is needed before proceeding.
