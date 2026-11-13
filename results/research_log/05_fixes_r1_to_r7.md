# Chapter 05 — Fixes R1–R7: Diagnoses, Changes, and Rationale

**Previous chapter:** [04 Diagnoses and Patterns](04_diagnoses_and_patterns.md)
**Next chapter:** [06 Reconstruction Sharpness](06_reconstruction_sharpness.md)

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

An additional option for preventing cardiomegaly head collapse (implemented as `--min_active_kl`): for samples where the disease head IS active (label matches), add a penalty that drives KL above a minimum floor (e.g., 2.0 nats), preventing the head from collapsing even under strong nulling. Use asymmetric sigma_inactive as an alternative: smaller for effusion (which has sharp features), larger for cardiomegaly.

```bash
--min_active_kl 2.0     # active head must maintain KL ≥ 2.0 nats
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

**Discriminator loss** (trained to classify correctly on disease samples):

$$\mathcal{L}_{\text{disc}} = \text{BCE}(D_{c \to e}(z_{\text{effusion}}), y_{\text{cardio}}) + \text{BCE}(D_{e \to c}(z_{\text{cardio}}), y_{\text{effusion}})$$

**VAE adversarial penalty** (trained to confuse discriminators):

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

## Supplementary: Implementation Status Summary (R1–R7)

| ID | Change | Rationale | Flag / Parameter |
|----|--------|-----------|-----------------|
| **R1** | `free_bits=0.0` | Removes gradient dead zone for disease heads | `--free_bits 0.0` |
| **R2** | `sigma_inactive=0.05`, `weight_null=0.02`, `weight_ortho=0.02` | Balanced inactivity without cardiomegaly collapse | see §R2 |
| **R3** | Cosine LR decay from `lr_decay_epochs=60` | Prevents NaN at long training; preserves peaks | `--lr_decay_epochs 60` |
| **R4** | `min_active_kl=2.0` | Floors active-head KL; prevents collapse under heavy nulling | `--min_active_kl 2.0` |
| **R5a** | `--use_contrastive`: paired contrastive loss | Structural separation: separate-push + blind-pull | `--use_contrastive --weight_contrastive 0.1` |
| **R5b** | `--use_cross_adv`: cross-head adversarial MLPs | Conditional independence enforcement | `--use_cross_adv --weight_cross_adv 0.05` |
| **R6** | Remove FPN | No benefit; reduces parameters and memory | (omit `--use_fpn`) |
| **R7** | `--use_label_attention`: disease prototype query | Spatial routing; addresses Pattern 2 and 3 at source | `--use_label_attention --attn_query_dim 256` |

---

## R8–R10: Proposed Next Fixes (not yet implemented)

The following three proposals address the two remaining structural gaps that R1–R7 do not close. They are ranked by expected ROI.

### Remaining gaps after R1–R7

| Gap | Description | Current symptom |
|-----|-------------|-----------------|
| **Gap 1** | Nulling loss loophole: the model can satisfy `z_d ≈ 0` by routing disease signal into `z_common` rather than genuinely suppressing it | Cross-head score remains > 0.74; cardiomegaly probe AUC volatile |
| **Gap 2** | GAP-level MI discrimination: the cross-head discriminator (R5b) compresses the 64×64 spatial latent to a single vector via global average pooling before the MLP. A spatially varying disease pattern can survive this compression undetected | Cross-head score above threshold even after R5b |

---

### R8 — Bbox-guided spatial KL supervision (Highest ROI)

**What the nulling loss misses:** `weight_null` penalises the KL of the inactive head toward `N(0, σ_inactive²)` globally. It applies the same pressure everywhere in the 64×64 spatial map. A disease signal concentrated in the cardiac or pleural region can persist because the global KL is diluted by the large normal-tissue area.

**Fix:** Use the VinBigData bounding box annotations (already available in the dataset) to apply *spatially differentiated* KL targets.

```python
# Inside the disease region (bbox projected to 64×64):
#   active head  → standard KL toward N(0, 1)   (encouraged to encode)
#   inactive head → strong KL toward N(0, σ_inactive²)  (suppressed)
#
# Outside the disease region:
#   active head  → mild KL toward N(0, 1)        (should be quiet outside bbox)
#   inactive head → standard inactive KL          (always suppressed)

bbox_mask = project_bbox_to_latent(bbox_xyxy, latent_h=64, latent_w=64)

kl_active_inside  = kl_divergence(z_d_mu[bbox_mask],  z_d_logvar[bbox_mask],  prior_mu=0, prior_var=1.0)
kl_active_outside = kl_divergence(z_d_mu[~bbox_mask], z_d_logvar[~bbox_mask], prior_mu=0, prior_var=1.0)
kl_inactive       = kl_divergence(z_d_inactive_mu,    z_d_inactive_logvar,    prior_mu=0, prior_var=sigma_inactive**2)

spatial_kl_loss = (
    weight_kl_active_inside  * kl_active_inside.mean() +
    weight_kl_active_outside * kl_active_outside.mean() +  # small weight, discourages outside encoding
    weight_null              * kl_inactive.mean()
)
```

**Files to modify:**
- `datasets/VinBigData.py`: return `bbox_xyxy` tensor alongside `(anchor, positive, negative)` triplet
- `losses/sep_vae_losses.py`: add `bbox_spatial_kl_loss` using the mask logic above
- `run/train_sep_vae.py`: pass `--use_bbox_kl`, `--weight_bbox_kl_inside`, `--weight_bbox_kl_outside` flags

**Why this has highest ROI:** VinBigData bounding boxes are already in the dataset annotations. No new labels, no new models. This directly encodes anatomical knowledge about *where* each disease should appear into the KL geometry, closing Gap 1 at the source.

**Launcher flag:**
```bash
--use_bbox_kl \
--weight_bbox_kl_inside 1.0 \
--weight_bbox_kl_outside 0.1
```

---

### R9 — Neutral-decode consistency loss (Gap 1 fix)

**What the nulling loss misses:** `weight_null` only penalises the *posterior parameters* of the inactive head. It does not check whether decoding with `z_d = 0` produces a normal-looking image. A model that routes disease signal into `z_common` will satisfy the nulling loss in `z_d` while still producing a diseased reconstruction.

**Fix:** For each disease batch element, decode with the inactive head zeroed and measure how far the result is from a normal image centroid in perceptual space.

```python
# During training of a cardiomegaly batch element:
z_cardio_zeroed = jnp.zeros_like(z_cardio)
x_neutral = decode(z_common, z_cardio_zeroed, z_effusion)

# Perceptual distance to normal centroid (running mean of backbone features of normal images)
feat_neutral = backbone(x_neutral)
feat_normal_centroid = lax.stop_gradient(normal_centroid_ema)
neutral_consistency_loss = jnp.mean((feat_neutral - feat_normal_centroid) ** 2)
```

**Why this closes Gap 1:** If `z_common` contains cardiomegaly signal, then `decode(z_common, 0, z_effusion)` will still produce an enlarged cardiac silhouette, which will be far from the normal centroid. This directly penalises the loophole.

**Files to modify:**
- `losses/sep_vae_losses.py`: add `neutral_decode_consistency_loss`
- `run/train_sep_vae.py`: add `--use_neutral_consistency`, `--weight_neutral_consistency` flags; maintain `normal_centroid_ema` as an EMA buffer updated from normal-only batch elements

**Launcher flag:**
```bash
--use_neutral_consistency \
--weight_neutral_consistency 0.05
```

---

### R10 — Spatial patch-level MI discriminator (Gap 2 fix)

**What R5b misses:** The cross-head adversarial discriminator (R5b) takes `z_c` and `z_d` (each `2ch × 64×64`), applies global average pooling to get a `2`-dimensional vector, then feeds it into an MLP. A disease pattern that is spatially localised (e.g., 8×8 patch in the cardiac region) will be averaged away by GAP and the discriminator will miss it.

**Fix:** Replace GAP with spatial patch sampling. Sample `K` spatial locations from the 64×64 map and concatenate the local feature vectors before the MLP.

```python
# Instead of:
#   z_c_vec = z_c.mean(axis=(-2, -1))   # GAP → [B, 4]
#   z_d_vec = z_d.mean(axis=(-2, -1))   # GAP → [B, 2]

# Use:
K = 16  # number of spatial patches sampled per image
h_idx = jax.random.randint(key, (K,), 0, 64)
w_idx = jax.random.randint(key, (K,), 0, 64)

z_c_patches = z_c[:, :, h_idx, w_idx]   # [B, 4, K]
z_d_patches = z_d[:, :, h_idx, w_idx]   # [B, 2, K]
z_c_vec = z_c_patches.reshape(B, -1)    # [B, 4K]
z_d_vec = z_d_patches.reshape(B, -1)    # [B, 2K]
# then feed into the same MLP discriminator
```

**Files to modify:**
- `losses/sep_vae_losses.py`: modify `CrossHeadDiscriminator` to accept `use_patch_sampling=True, K=16` and replace GAP with random patch extraction
- `run/train_sep_vae.py`: add `--cross_disc_patch_sampling`, `--cross_disc_n_patches` flags

**Launcher flag:**
```bash
--cross_disc_patch_sampling \
--cross_disc_n_patches 16
```

---

### Proposal comparison

| Proposal | Gap addressed | Data required | Files changed | Estimated ROI |
|----------|--------------|---------------|---------------|---------------|
| **R8: Bbox-guided spatial KL** | Gap 1 (nulling loophole) | VinBigData bboxes (already available) | `VinBigData.py`, `sep_vae_losses.py`, `train_sep_vae.py` | **Highest** |
| **R9: Neutral-decode consistency** | Gap 1 (nulling loophole) | Normal image centroid (online EMA) | `sep_vae_losses.py`, `train_sep_vae.py` | Medium |
| **R10: Spatial MI discriminator** | Gap 2 (GAP compression) | None (architectural change only) | `sep_vae_losses.py`, `train_sep_vae.py` | Medium |

R8 is recommended first because it uses supervision signal already present in VinBigData and makes the spatial prior explicit rather than emergent. R9 and R10 are complementary and can be combined with R8.

---

## R11 — Disease Discriminability Classifier (Implemented)

**What all prior losses miss:** R1–R10 address leakage, collapse under nulling, and spatial routing. None of them enforce that `z_disease_k` is *positively informative* about disease_k when it is present.

**The free-rider problem:** The training signal on `z_disease_k` for a disease-positive sample is:

| Loss | Effect on `z_disease_k` (positive sample) |
|------|-------------------------------------------|
| Reconstruction | "encode *something* useful" — but `z_common` can do this instead |
| KL disease | collapse toward N(0, I) — purely destructive |
| Null loss | only fires for the *other* head (inactive head), not `z_disease_k` itself |
| Orthogonality | decorrelates cardio↔effusion directions — structural, not informative |

`z_common` has a much stronger reconstruction gradient (more channels, full image structure) and will greedily absorb disease-relevant texture if it reduces total reconstruction loss. `z_disease_k` can coast at near-prior with the KL satisfied — posterior collapse in the disease head. This is never caught because the null loss only watches the disease head being active for *wrong* samples; it says nothing about disease-positive samples.

**Fix:** Add an auxiliary binary classifier per disease head that predicts disease_k presence from `z_disease_k.μ` (spatially averaged). Train it jointly with the encoder via BCE. The gradient flows back into the encoder through the classifier, forcing `z_disease_k` to pack disease-discriminative information.

$$\mathcal{L}_{\text{clf}} = \text{BCE}\!\left(\hat{y}_k,\, \mathbf{1}[\text{label} = k]\right), \quad \hat{y}_k = \sigma\!\left(\text{MLP}(\text{GAP}(\mu_k))\right)$$

**Why this is cooperative, not adversarial:** Both the classifier and the encoder minimise the same BCE. In `vae_step` the classifier params are treated as frozen and the gradient reaches only the encoder (making `z_disease_k` more discriminative). In `clf_step` the encoder is treated as fixed and the gradient reaches only the classifier (making the classifier better). There is no gradient reversal.

**Architecture:** `Linear(C → 16) → ReLU → Linear(16 → 1)`. With `z_channels_disease=2` this is 49 parameters — negligible memory overhead.

**Connection to verification gate G1:** G1 requires that a frozen linear probe on `z_disease_k` achieves AUC > 0.75 post-training. R11 directly trains the encoder to satisfy this criterion during training, rather than hoping it emerges from the reconstruction signal alone. If G1 fails, the disease head has posterior collapsed — R11 prevents this from happening silently.

**Connection to counterfactual quality:** If `z_disease_k` carries no information (collapsed), nulling it produces visually identical output. The counterfactual appears to "work" trivially but is meaningless. R11 ensures `z_disease_k` genuinely encodes pathology, so nulling produces a real change. Without R11, the swap grid in Chapter 08 is uninterpretable.

**Connection to Strategy A LDM:** The LDM learns `p(z_cardio | z_common)`. If `z_cardio` collapses during SepVAE training, the LDM learns a trivial near-prior and composition generates nothing disease-specific. R11 ensures `z_disease_k` has enough variance and semantic structure for the LDM to model.

### R11 — Implementation details

**Files changed:**
- `losses/sep_vae_losses.py`: added `DiseaseClassifier` module and `disease_discriminability_loss()` function; added `weight_disease_clf` field to `SepVAELossConfig`; wired as step 10 in `sepvae_loss()`
- `run/train_sep_vae.py`: added `--use_disease_clf`, `--weight_disease_clf`, `--lr_disease_clf`; initialised `clf_cardio_state` and `clf_effusion_state` with AdamW; added `clf_step()` update function; wired into checkpoint save/restore and W&B logging

**New log keys** (W&B and per-step stdout):
- `loss/disease_clf` — combined BCE for both heads
- `loss/disease_clf_cardio` — cardiomegaly head BCE
- `loss/disease_clf_effusion` — effusion head BCE
- `loss/disease_clf_update` — classifier-side update (for monitoring classifier accuracy separately from encoder signal)

**Run command — biggpu dual-GPU (mscluster106, both GPUs, recommended):**

```bash
# From repo root on the login node. R11 is always enabled for both GPUs.
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh
```

This submits one SLURM job (partition `biggpu`, nodelist `mscluster106`) that runs two Python processes internally via `CUDA_VISIBLE_DEVICES` pinning:

| GPU | Experiment | Config | Batch size |
|-----|-----------|--------|-----------|
| 0 | `sepvae_baseline_fixed` | R1+R2+R3+R4+**R11** | 8 |
| 1 | `sepvae_full` | R1+R2+R3+R4+R5a+R5b+R7+**R11** | 6 |

Both batch sizes were tuned down from 20 after OOM at the first JIT step (36GB allocation on GPU 1 with the full stack at bs=20). GPU 0 (baseline) OOM'd at the same batch size due to JIT overhead, not steady-state usage. Sizes 8 and 6 give ample headroom on the 49GB RTX 8000 cards.

**Run command — single GPU (bigbatch, any available node):**

```bash
# R1+R2+R3+R4+R11 only
bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh

# R1+R2+R3+R4+R5a+R5b+R7+R11 (full stack)
bash launchers/single_runs/vae/train_sep_vae_full.sh
```

**Direct Python invocation — R11 flags:**

```bash
python run/train_sep_vae.py \
  --exp_name sepvae_baseline_fixed_r11 \
  --batch_size 8 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --weight_perceptual 0.05 \
  --use_disease_clf \
  --weight_disease_clf 0.1 \
  --lr_disease_clf 1e-4 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Full stack + R11
python run/train_sep_vae.py \
  --exp_name sepvae_full_r11 \
  --batch_size 6 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --weight_perceptual 0.05 \
  --use_contrastive --weight_contrastive 0.1 --contrastive_margin 0.5 \
  --use_cross_adv --weight_cross_adv 0.05 --lr_cross_disc 1e-4 --cross_disc_hidden_dim 256 \
  --use_label_attention --attn_query_dim 256 \
  --use_disease_clf --weight_disease_clf 0.1 --lr_disease_clf 1e-4 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

**W&B: what to watch for R11:**
- `loss/disease_clf_cardio` and `loss/disease_clf_effusion` should decrease steadily from `log(2) ≈ 0.693` (random classifier) toward ~0.2 by epoch 10–20. If either plateaus above 0.6, the corresponding disease head is collapsing.
- `loss/nulling` should remain low and stable; if it spikes after R11 is added, the encoder is fighting the nulling pressure with the classification signal — reduce `weight_disease_clf` slightly.
- `loss/kl_cardiomegaly` and `loss/kl_effusion` should have lower variance than runs without R11 (R11 provides consistent gradient signal to the disease heads, stabilising their KL).

---

## R12 — Bbox Spatial Attention Supervision (Implemented)

**Problem:** The `DiseaseAttentionHead` (R7) learns soft spatial attention without any structural signal about *where* each disease should appear. The attention map is initialised uniformly and, if the supervision signal is weak, can remain diffuse or drift to the wrong anatomical region — reintroducing spatial leakage.

**Fix:** Penalise attention *outside* the ground-truth bounding box for active-disease samples. For each disease head, compute:

$$\mathcal{L}_{\text{spatial-attn}} = \frac{1}{B_{\text{active}}} \sum_{b \in \text{active}} \frac{1}{HW} \sum_{i,j} A_d^{(b)}(i,j) \cdot (1 - M^{(b)}(i,j))$$

where $M^{(b)}(i,j) \in \{0,1\}$ is the binary mask at the $64 \times 64$ latent resolution derived from the VinBigData bbox annotation for sample $b$, and $A_d$ is the softmax attention map produced by `DiseaseAttentionHead`.

**Why attention supervision is complementary to R7:** R7 gives the model the *capacity* to attend spatially; R12 gives it the *ground-truth target* for where attention should concentrate. R7 alone may learn to attend to any predictive region, including regions shared with the other disease. R12 forces the attention map to stay inside the annotated anatomical region, structurally preventing the two disease heads from attending to the same spatial area.

### R12 — Implementation details

**Files changed:**

- `datasets/VinBigData.py`:
  - `_build_bbox_lookup(df, class_id)`: groups by `image_id`, takes union (min/max) over all radiologist annotations, returns `{image_id: (x0, y0, x1, y1)}` normalised to pixel coordinates. Returns `{}` if bbox columns are absent.
  - `_load_disease_image(pool, bbox_lookup, ...)`: loads DICOM, retrieves original `(H_orig, W_orig)`, normalises bbox to `[0, 1]` range. Returns `(image_tensor, bbox_tensor_4)` where `bbox_tensor_4 = [x_min, y_min, x_max, y_max]` ∈ [0,1]⁴.
  - `__getitem__`: returns `bbox_disease1` (effusion, 4-float) and `bbox_disease2` (cardio, 4-float) alongside the image triplet.
  - `jax_collate_fn`: stacks into `(B, 4)` arrays and includes them in the batch dict as `bbox_disease1` and `bbox_disease2`.

- `losses/sep_vae_losses.py`:
  - `_bbox_to_mask(bboxes, H, W)`: pure JAX; builds `(B, H, W)` binary float mask via `jnp.arange` broadcast. JIT-safe (no dynamic shapes).
  - `spatial_attention_loss(attn_maps, bbox_disease1, bbox_disease2, latent_hw=64)`: slices the triplet batch (`[B:2B]` = effusion, `[2B:3B]` = cardio), builds masks, computes `mean(attn*(1−mask))` per sample, applies validity gate (all-zero bbox → sample excluded from average).
  - `weight_spatial_attn: float = 0.0` added to `SepVAELossConfig`.
  - Step 11 in `sepvae_loss()`: guarded by `cfg.weight_spatial_attn > 0 and 'attn_maps' in latents_dict and 'bbox_disease1' in batch`.
  - New log key: `loss/spatial_attn`.

- `run/train_sep_vae.py`:
  - `--weight_spatial_attn` argparse argument (default `0.0`).
  - Wired into `SepVAELossConfig` construction: set to `0.0` if `--use_label_attention` is absent.
  - `bbox_disease1` / `bbox_disease2` converted from torch tensors to `jnp.array` in the training loop batch dict.
  - `SpatialAttn=` added to step log string and epoch summary.

- `launchers/single_runs/vae/train_sep_vae_biggpu.sh`:
  - Default `WEIGHT_SPATIAL_ATTN="0.05"`.
  - `--weight_spatial_attn` CLI override accepted by the launcher arg parser.
  - Forwarded via `--export=ALL,...,WEIGHT_SPATIAL_ATTN="$WEIGHT_SPATIAL_ATTN"` to the SLURM script.
  - Status line `SpatialAttn: w=... (full run only)` printed in submission summary.

- `slurm_scripts/sep_vae_biggpu.slurm`:
  - `WEIGHT_SPATIAL_ATTN="${WEIGHT_SPATIAL_ATTN:-0.05}"` env var.
  - `--weight_spatial_attn "$WEIGHT_SPATIAL_ATTN"` passed to `ARGS_B` (GPU 1 / full run only, alongside `--use_label_attention`).

**Run command:**
```bash
# Default weight (0.05)
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh

# Stronger supervision
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --weight_spatial_attn 0.1

# Disable (revert to unsupervised attention)
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --weight_spatial_attn 0.0
```

**W&B: what to watch for R12:**
- `loss/spatial_attn` should decrease from ~0.5 (uniform attention covers ~50% outside any typical bbox) toward ~0.05–0.15 by epoch 20. Stagnation above 0.4 means the attention head is not concentrating.
- `diagnostics/attn_maps` — cardiomegaly map should focus on the central cardiac region (rows 0.35–0.55, cols 0.30–0.70 at 512px); effusion map on the lower lateral pleural angles (rows 0.70–1.00, cols 0.10–0.90).
- `cross_head_score` — the intended effect: if attention concentration succeeds, cross-head score should drop faster than in the R7-only run.

---

## Supplementary: Full Implementation Status (R1–R12)

| ID | Change | Status | Gap addressed |
|----|--------|--------|---------------|
| R1 | `free_bits=0.0` | ✓ Implemented | Posterior dead zone |
| R2 | Balanced null/ortho weights | ✓ Implemented | Cardiomegaly collapse |
| R3 | Cosine LR decay | ✓ Implemented | NaN at late epochs |
| R4 | `min_active_kl` floor | ✓ Implemented | KL collapse under heavy nulling |
| R5a | Paired contrastive loss | ✓ Implemented | Cross-head leakage (latent push) |
| R5b | Cross-head adversarial MLPs | ✓ Implemented | Cross-head leakage (adversarial) |
| R6 | Remove FPN | ✓ Implemented | Memory / parameter reduction |
| R7 | Label attention routing | ✓ Implemented | Spatial routing (Pattern 2+3) |
| **R8** | **Bbox-guided spatial KL** | **Proposed** | Gap 1: nulling loophole |
| **R9** | **Neutral-decode consistency** | **Proposed** | Gap 1: nulling loophole |
| **R10** | **Spatial MI discriminator** | **Proposed** | Gap 2: GAP compression |
| **R11** | **Disease discriminability classifier** | ✓ **Implemented** | Posterior collapse in disease heads (missing positive pressure) |
| **R12** | **Bbox spatial attention supervision** | ✓ **Implemented** | Spatial leakage at encoder input; attention without anatomical grounding |

---

*End of Chapter 05. Continue to [Chapter 06: Reconstruction Sharpness](06_reconstruction_sharpness.md).*
