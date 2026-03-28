# Objective Functions

**Related documents:** [04 Model Architecture](04_model_architecture.md) | [03 Training Curriculum](03_training_curriculum.md) | [06 Failures & Debugging](06_failures_debugging.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25
**Implementation:** [losses/sep_vae_losses.py](../losses/sep_vae_losses.py)

---

## 1. Training Structure

Three separate optimisers run alternately at each training step:

```
Step 1: factor_disc_step
    Update FactorVAE discriminator (joint vs. marginal classification)
    Inputs: z_c_stale, z_d_stale (from previous VAE step, stop_gradient'd)
    Outputs: updated factor_disc_state

Step 2: patch_disc_step (D3+)
    Update PatchGAN discriminator (real vs. reconstructed patches)
    Inputs: x_real, x_rec_stale (from previous VAE step, stop_gradient'd)
    Outputs: updated patch_disc_state

Step 3: vae_step
    Update VAE (encoder + decoder) against frozen discriminators
    Inputs: batch, factor_disc_params (frozen), patch_disc_params (frozen)
    Outputs: updated vae_state, fresh z_c, z_d, x_rec (become stale for next step)
```

The stale-input pattern prevents gradient coupling between the VAE and the discriminators within a single step. Discriminator parameters are frozen when computing VAE gradients; VAE parameters are frozen when computing discriminator gradients.

---

## 2. Total VAE Loss

$$\mathcal{L}_{\text{VAE}} = w_{\text{rec}}\mathcal{L}_{\text{rec}} + \beta_c \mathcal{L}_{\text{KL}_c} + \beta_d \mathcal{L}_{\text{KL}_d} + \kappa\mathcal{L}_{\text{MI}} + w_{\text{bbox}}\mathcal{L}_{\text{bbox}} + w_{\text{perc}}\mathcal{L}_{\text{perc}} + w_{\text{gan}}\mathcal{L}_{\text{gan}} + w_{\text{tv}}\mathcal{L}_{\text{tv}} + w_{\text{masked}}\mathcal{L}_{\text{masked}} + w_{\text{supcon}}\mathcal{L}_{\text{supcon}}$$

### D3 weights

| Symbol | Value | Loss name |
|--------|-------|-----------|
| $w_{\text{rec}}$ | 1.0 | Reconstruction (MSE) |
| $\beta_c$ | 1e-4 | KL (common head) |
| $\beta_d$ | 5e-5 | KL (disease head, conditional) |
| $\kappa$ | 1.0 | MI factor (FactorVAE) |
| $w_{\text{bbox}}$ | 0.10 | Bbox attention |
| $w_{\text{perc}}$ | 0.05 | Perceptual (CheSS layers 1–2) |
| $w_{\text{gan}}$ | 0.10 | GAN (PatchGAN hinge generator) |
| $w_{\text{tv}}$ | 0.005 | Total variation |
| $w_{\text{masked}}$ | 0.3 | Masked reconstruction |
| $w_{\text{supcon}}$ | 0.05 | Supervised contrastive |

---

## 3. Objective-by-Objective Specification

### 3.1 Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)

**Purpose:** Pixel-level fidelity. The primary signal driving the encoder to encode everything needed for reconstruction.

**Formula:**
$$\mathcal{L}_{\text{rec}} = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} (x_{b,h,w} - \hat{x}_{b,h,w})^2$$

where $x \in [-1, 1]$ (input image) and $\hat{x} \in [0, 1]$ after sigmoid (the reconstruction). Note: the target is `(x + 1) / 2` — centred to [0,1] before MSE.

**Why MSE blurs:** The optimal decoder under MSE is the posterior mean $\mathbb{E}[p(x|z)]$. For any region with ambiguity across plausible completions, this mean is a blurred average. This is why GAN and perceptual losses are added — they penalise the model for choosing the blurred average.

**Implementation:** `losses/sep_vae_losses.py: reconstruction_loss()`

---

### 3.2 KL Divergence — Common Head ($\mathcal{L}_{\text{KL}_c}$)

**Purpose:** Regularise z_common toward a standard Normal prior $\mathcal{N}(0, I)$. Prevents z_common from encoding arbitrary information with no prior constraint (posterior collapse in reverse: unbounded expansion).

**Formula (standard ELBO KL):**
$$\mathcal{L}_{\text{KL}_c} = \text{KL}(q(z_c | x) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_i \left( \mu_{c,i}^2 + \sigma_{c,i}^2 - 1 - \log \sigma_{c,i}^2 \right)$$

Summed over all latent dimensions $(16 \times 16 \times 16 = 4096)$ and averaged over the batch.

**KL free bits ($\lambda = 0.5$):** Per-dimension KL is floored at $\lambda$:
$$\tilde{\mathcal{L}}_{\text{KL}_c} = \frac{1}{D}\sum_i \max(\lambda, \text{KL}_i)$$

This prevents posterior collapse (any dimension with very small KL would receive zero gradient, collapsing toward the prior). It also bounds the maximum gradient contribution from a single high-KL dimension (cap effect). The specific value 0.5 was chosen empirically — it provides stable training without creating the dead zone observed at free_bits=2.0 in V1 (see [Failure A1](06_failures_debugging.md#a1--free-bits--sigma_inactive-conflict)).

**Implementation:** `losses/sep_vae_losses.py: kl_divergence_standard()`

---

### 3.3 KL Divergence — Disease Head ($\mathcal{L}_{\text{KL}_d}$)

**Purpose:** Conditional regularisation of z_disease. Active samples (Cardiomegaly) are pushed toward $\mathcal{N}(0, I)$; inactive samples (Normal) are pushed toward $\mathcal{N}(0, \sigma_{\text{inactive}}^2 I)$ — a tight prior 100× narrower than the standard prior.

**Formula (conditional tight prior):**
$$\mathcal{L}_{\text{KL}_d} = \begin{cases}
\text{KL}(q(z_d | x) \| \mathcal{N}(0, I)) & \text{if label = Cardiomegaly (active)} \\
\text{KL}(q(z_d | x) \| \mathcal{N}(0, \sigma_{\text{inactive}}^2 I)) & \text{if label = Normal (inactive)}
\end{cases}$$

For the inactive case:
$$\text{KL}(q \| \mathcal{N}(0, \sigma^2 I)) = \frac{1}{2} \sum_i \left( \frac{\mu_{d,i}^2 + \sigma_{d,i}^2}{\sigma_{\text{inactive}}^2} - 1 - \log \frac{\sigma_{d,i}^2}{\sigma_{\text{inactive}}^2} \right)$$

With $\sigma_{\text{inactive}} = 0.1$, the prior is $\mathcal{N}(0, 0.01 \cdot I)$ — penalising any encoder mean $\mu_d$ away from zero for Normal images with 100× the sensitivity of the standard prior.

**Two independent zero-driving pressures for Normal images:**
1. This conditional KL loss penalises $\mu_d \neq 0$ at the posterior level
2. Hard-zero nulling (architectural) sets $z_d = 0$ at the decoder input regardless

This double enforcement means the model cannot satisfy the reconstruction loss for Normal images using disease-head information — the decoder never sees z_d for Normal images, and the encoder is penalised for even having a non-zero posterior mean.

**Note on weight_kl_disease = 5e-5:** This is deliberately lower than weight_kl_common (1e-4). The disease head needs more room to manoeuvre — a heavier KL penalty would compress the z_disease posterior too aggressively, conflicting with the reconstruction gradient that needs z_disease to encode cardiac information. At 1e-4 (double the current value), stripe artifacts were observed in decoder outputs (see [Failure C4](06_failures_debugging.md#c4--stripe-artifacts-from-weight_kl_disease1e-4)).

**Implementation:** `losses/sep_vae_losses.py: kl_divergence_conditional()`

---

### 3.4 FactorVAE MI Loss ($\mathcal{L}_{\text{MI}}$)

**Purpose:** Push z_common and z_disease toward a **product of marginals** distribution (statistical independence). Penalises total correlation between the two latent heads.

**Mechanism (FactorVAE):**
A small discriminator $D_{\text{factor}}$ learns to distinguish:
- The **joint** distribution $(z_c, z_d)$ from actual encoder outputs
- The **marginal** product $(z_c, \tilde{z}_d)$ where $\tilde{z}_d$ is the z_d from a **permuted** batch element

If the VAE has achieved independence ($z_c \perp z_d$), the joint and product distributions are identical, and $D_{\text{factor}}$ cannot distinguish them — accuracy = 0.50.

**Discriminator training loss (step 1, factor_disc_step):**
$$\mathcal{L}_{\text{disc}} = -\mathbb{E}\left[\log D_{\text{factor}}(z_c, z_d)\right] - \mathbb{E}\left[\log(1 - D_{\text{factor}}(z_c, \tilde{z}_d))\right]$$

Binary cross-entropy on joint (label=1) vs. permuted-marginal (label=0).

**VAE adversarial penalty (step 3, vae_step):**
$$\mathcal{L}_{\text{MI}} = \mathbb{E}\left[\log \frac{D_{\text{factor}}(z_c, z_d)}{1 - D_{\text{factor}}(z_c, z_d)}\right]$$

The VAE is penalised when $D_{\text{factor}}$ succeeds at distinguishing joint from marginal — the logit diverges positively when the discriminator is confident. This directly minimises total correlation TC($z_c$; $z_d$).

**Healthy equilibrium:** $D_{\text{factor}}$ accuracy = 0.50. At D3, the discriminator accuracy is monitored and settles near 0.52–0.55, indicating near-independence.

**FactorDiscriminator architecture:**
- Input: GAP(z_c) concatenated with GAP(z_d) → (16+16=32)-dimensional vector
- 4 fully connected layers: 32→256→256→256→2 (logit for joint vs. marginal)
- LeakyReLU activations

**Implementation:** `losses/sep_vae_losses.py: FactorDiscriminator, factor_disc_loss(), factor_vae_mi_loss()`

---

### 3.5 Bbox Attention Loss ($\mathcal{L}_{\text{bbox}}$)

**Purpose:** Force the disease head attention map (from BboxCrossAttnHead) to stay inside the annotated cardiac bounding box for Cardiomegaly images. Prevents attention from drifting to high-energy but non-cardiac regions (image borders, vertebrae, etc.).

**Formula:**
$$\mathcal{L}_{\text{bbox}} = \frac{1}{B_{\text{cardio}}} \sum_{b \in \text{cardio}} \frac{1}{HW} \sum_{i,j} A_d^{(b)}(i,j) \cdot (1 - M^{(b)}(i,j))$$

where $M^{(b)}(i,j) \in \{0, 1\}$ is the binary mask at the 16×16 latent resolution derived from the normalised bbox coordinates, and $A_d$ is the softmax attention map from BboxCrossAttnHead.

**Interpretation:** This is the average attention weight outside the cardiac bbox — the penalty is zero when all attention mass is inside the bbox (perfect localisation) and 1.0 when attention is uniform across the full map.

**Only for Cardiomegaly:** Normal images have no bbox, so `L_bbox` is computed on the `[B:2B]` slice of the batch (cardiomegaly triplet position).

**Healthy values:**
- Epoch 55 (D3 start): ~0.30 (attention partially concentrated but not tight)
- Epoch 100 (D3 mid): ~0.20–0.25 (good concentration)
- Target at D3 end (epoch 120): ≤ 0.20

**Relationship to BboxCrossAttnHead:** The Gaussian prior gives the attention head a good starting point; this loss provides the gradient pressure to maintain it. Without the loss (as in D1 where weight_bbox_attn=0.0), the learned component of the query drifts toward high-energy non-cardiac features over training.

**Implementation:** `losses/sep_vae_losses.py: bbox_attention_loss()`

---

### 3.6 Perceptual Loss ($\mathcal{L}_{\text{perc}}$)

**Purpose:** Mid-frequency texture sharpening. Penalises the decoder for producing images with different feature statistics from real CXRs in the CheSS feature space, even if the pixel-level MSE is comparable.

**Formula:**
$$\mathcal{L}_{\text{perc}} = \frac{1}{L} \sum_{\ell \in \{1,2\}} \| \phi_\ell(x) - \phi_\ell(\hat{x}) \|_1$$

where $\phi_\ell$ denotes the feature map at layer $\ell$ of the frozen CheSS backbone, and L is the number of layers used.

**Frozen CheSS backbone:** CheSS is a CXR-pretrained ResNet-50. Its layer features encode CXR-specific texture statistics (rib edge profiles, vessel boundary profiles, lung parenchyma texture). L1 distance in this feature space measures perceptual dissimilarity from a CXR perspective.

**Layers 1–2 only (`--perceptual_only` flag):**

| Layer | Stride (256px input) | Gradient period | Used? |
|-------|---------------------|-----------------|-------|
| Layer 1 | 4 | 4px | Yes |
| Layer 2 | 8 | 8px | Yes |
| Layer 3 | 16 | **16px** | **No** |
| Layer 4 | 32 | 32px | No |

Layer3 was the primary source of the 16px-period horizontal stripe banding observed in D2–D4 (see [Failure C1](06_failures_debugging.md)). At `weight_perceptual=0.15` and `weight_tv=0.001`, the layer3 perceptual gradient dominated TV by ~300×. Restricting to layers 1–2 eliminates the 16px aliasing while retaining the mid-frequency sharpening benefit.

**Implementation:** `losses/sep_vae_losses.py: backbone_perceptual_loss()`

---

### 3.7 PatchGAN Adversarial Loss ($\mathcal{L}_{\text{gan}}$)

**Purpose:** Force the decoder to produce sharp, photorealistic local patches rather than blurry MSE-minimising averages. The discriminator evaluates local patches rather than the full image, making the gradient local and spatially consistent.

**Architecture — NLayerDiscriminator:**
- 4 layers: Conv(64, stride=2), Conv(128, stride=2), Conv(256, stride=2), Conv(512, stride=1), Conv(1, stride=1)
- LeakyReLU, Instance Normalization (not GroupNorm — standard in PatchGAN)
- Output: a spatial map of "real vs. fake" logits — each output position corresponds to a ~70×70px receptive field of the input

**Generator loss (hinge formulation):**
$$\mathcal{L}_{\text{gan}} = -\mathbb{E}\left[\text{mean}(D_{\text{patch}}(\hat{x}))\right]$$

The VAE (generator) is penalised when $D_{\text{patch}}$ assigns low scores to the reconstruction.

**Discriminator loss (hinge, step 2):**
$$\mathcal{L}_{\text{disc}} = \mathbb{E}[\text{mean}(\text{relu}(1 - D_{\text{patch}}(x_{\text{real}})))] + \mathbb{E}[\text{mean}(\text{relu}(1 + D_{\text{patch}}(\hat{x}_{\text{stale}})))]$$

Real images should produce logits > 1; reconstructions should produce logits < −1 (the discriminator aims for a margin of ±1 in hinge loss).

**Stale reconstruction:** The discriminator update uses `x_rec_stale` — the reconstruction from the previous VAE step (stored after step 3, frozen via stop_gradient for step 2). This avoids gradient coupling within a step.

**Phase-local start:** `gan_start_step = 2000`. The GAN activates 2000 phase-local steps after the phase starts, giving the VAE ~5 epochs of TV-only sharpening before adversarial gradients arrive. Phase-local counting is critical — a restored checkpoint will have `global_step ≈ 57,000+`, which would immediately exceed any global-step threshold (see [Failure B1](06_failures_debugging.md#b1--d5_gan-20260323-042442--catastrophic-collapse-epoch-4)).

**Weight calibration:** `weight_gan = 0.1`. At D3 equilibrium:
- Reconstruction: `1.0 × 0.08 ≈ 0.080`
- GAN: `0.1 × 0.5–1.0 ≈ 0.050–0.100`
- Ratio: ~1.0–1.25× (GAN ≈ reconstruction, not dominant)

At `weight_gan = 0.5` (the failed D5 run): GAN was 4.6× larger than reconstruction — catastrophic collapse in 4 epochs.

**Implementation:** `losses/sep_vae_losses.py: NLayerDiscriminator`, training loop in `run/train_sep_vae.py`

---

### 3.8 Total Variation Loss ($\mathcal{L}_{\text{tv}}$)

**Purpose:** Suppress residual stripe artifacts from CheSS perceptual gradients. Provides a spatial smoothness regulariser on the reconstruction.

**Formula (anisotropic TV):**
$$\mathcal{L}_{\text{tv}} = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} \left( |\hat{x}_{b,h+1,w} - \hat{x}_{b,h,w}| + |\hat{x}_{b,h,w+1} - \hat{x}_{b,h,w}| \right)$$

Anisotropic (separate horizontal and vertical terms, not the L2 norm) — preserves diagonal features better than the isotropic variant.

**Weight calibration:** `weight_tv = 0.005`. This is 5× the value used in failed earlier runs (`0.001`). The increase was necessary because at `weight_tv = 0.001`, the TV gradient was ~300× smaller than the CheSS layer3 perceptual gradient (before layer3 was excluded). Even after layer3 exclusion, `0.001` was insufficient to counteract the combined layer1+2 gradient magnitude. At `0.005`, stripe suppression is effective without over-smoothing fine detail.

**Interaction with perceptual loss:** TV and perceptual are in tension. TV smooths everything; perceptual sharpens features. The balance `weight_tv=0.005, weight_perc=0.05` keeps perceptual 10× stronger than TV, allowing sharpening at medium frequencies (ribs, vessels) while TV suppresses the high-frequency stripe artifacts below the rib-edge scale.

**Implementation:** `losses/sep_vae_losses.py: total_variation_loss()`

---

### 3.9 Masked Reconstruction Loss ($\mathcal{L}_{\text{masked}}$)

**Purpose:** Verify that z_common can reconstruct the non-cardiac region without any z_disease information. Prevents z_common from routing cardiac features through it (the "common head pollution" problem).

**Formula:**
$$\mathcal{L}_{\text{masked}} = \frac{1}{\text{pixels outside bbox}} \sum_{b \in \text{cardio}} \sum_{(h,w) \notin M_b} (x_{b,h,w} - \hat{x}_{b,h,w}^{(z_d=0)})^2$$

The reconstruction $\hat{x}^{(z_d=0)}$ is computed with z_d hard-zeroed (regardless of the actual label). This requires a **second forward pass** through the decoder for Cardiomegaly images in each step — increasing memory by ~30% for the decoder activation, but providing a direct purity check.

**Why outside the bbox?** If we measured MSE everywhere, the loss would penalise the decoder for not reconstructing the cardiac silhouette using z_common — but that's exactly what we want z_disease to do. We only care that z_common correctly reconstructs the non-cardiac regions (lungs, bones, soft tissue) when z_disease is absent.

**Healthy values:** `loss/masked_rec ≈ 0.02` at D3 end. Much higher values indicate z_common is leaking disease information — the non-cardiac regions are being incorrectly reconstructed without z_disease, meaning z_common must have absorbed some disease signal.

**Implementation:** `losses/sep_vae_losses.py: masked_anatomy_reconstruction_loss()`

---

### 3.10 Supervised Contrastive Loss ($\mathcal{L}_{\text{supcon}}$)

**Purpose:** Pull same-class z_disease representations together and push Normal/Cardiomegaly apart. Encourages z_disease to form tight, separable clusters — making it discriminative for downstream classifiers and stable for LDM training.

**Formula:**
$$\mathcal{L}_{\text{supcon}} = \frac{1}{B} \sum_b \frac{-1}{|P(b)|} \sum_{p \in P(b)} \log \frac{\exp(\mathbf{z}_b \cdot \mathbf{z}_p / \tau)}{\sum_{a \neq b} \exp(\mathbf{z}_b \cdot \mathbf{z}_a / \tau)}$$

where $\mathbf{z}$ is the L2-normalised GAP of z_disease, $P(b)$ is the set of positive pairs for sample $b$ (same label), and $\tau$ is temperature (default 0.07).

This is applied to **pooled z_d means** (not sampled z_d), making it a deterministic loss without noise from the reparameterisation trick.

**Implementation:** `losses/sep_vae_losses.py: supervised_contrastive_loss()`

---

## 4. Discriminator Losses (not part of VAE gradient)

### 4.1 FactorVAE Discriminator Loss

Trained in step 1 against stale z_c/z_d from the previous VAE step:
$$\mathcal{L}_{\text{factor\_disc}} = -\mathbb{E}_{q(z_c, z_d)}\left[\log D(z_c, z_d)\right] - \mathbb{E}_{q(z_c)\bar{q}(z_d)}\left[\log(1 - D(z_c, \tilde{z}_d))\right]$$

Standard binary cross-entropy. The discriminator reaches 0.50 accuracy when the VAE has achieved independence between z_c and z_d.

### 4.2 PatchGAN Discriminator Loss

Trained in step 2 against stale x_rec:
$$\mathcal{L}_{\text{patch\_disc}} = \mathbb{E}\left[\text{relu}(1 - D_{\text{patch}}(x_{\text{real}}))\right] + \mathbb{E}\left[\text{relu}(1 + D_{\text{patch}}(\hat{x}_{\text{stale}}))\right]$$

Hinge loss. Real images target logit > +1; reconstructions target logit < −1.

---

## 5. Loss Interaction Map

The objectives interact through the encoder and decoder gradient paths:

```
Objective          → Encoder gradient     → Decoder gradient
─────────────────────────────────────────────────────────────
L_rec              → z_c, z_d informative → all decoder layers
L_KL_c             → regularise z_c       → none (prior-only)
L_KL_d             → regularise z_d       → none
L_mi               → z_c ⊥ z_d           → none (latent-space only)
L_bbox             → attn in cardiac bbox → none
L_perc             → texture features     → all decoder layers
L_gan              → generator pressure   → all decoder layers
L_tv               → none                 → final decoder layers
L_masked_rec       → z_c purity          → decoder (z_d=0 path)
L_supcon           → z_d discriminability → none
```

**Key tension:** L_rec (encoder: encode everything) vs. L_KL (encoder: forget everything → collapse). KL weight 1e-4 and 5e-5 keep KL as a mild regulariser, not a dominant force.

**Key tension:** L_perc (decoder: match CXR feature statistics) vs. L_tv (decoder: smooth output). Calibration: `weight_perc / weight_tv = 0.05 / 0.005 = 10×` — perceptual dominates as intended.

**Key tension:** L_rec (decoder: accurate reconstruction) vs. L_gan (decoder: photorealistic patches). At D3 equilibrium: GAN ≈ reconstruction in magnitude. If GAN dominates (> 2× reconstruction), the decoder sacrifices accuracy for texture → catastrophic collapse.

---

## 6. Objective Addition History

| Objective | Introduced | Stage | Rationale |
|-----------|-----------|-------|-----------|
| L_rec | D0 | Smoke test | Primary reconstruction signal |
| KL (common + disease) | D0 | Smoke test | Latent regularisation |
| BboxCrossAttnHead | D1 | Architecture change | Spatial prior from epoch 1 |
| L_mi (FactorVAE) | D2 | After stable reconstruction | Needs stable encoder to converge |
| L_bbox_attn | D2 | After attention maps stable | Supervise what we can verify |
| L_perc (layers 1–2) | D2 | After stable reconstruction | Texture sharpening |
| L_masked_rec | D2 | After stable reconstruction | Verify z_common purity |
| L_supcon | D2 | After stable z_d | Cluster disease latents |
| L_gan (PatchGAN) | D3 | After perceptual + MI stable | Fine texture sharpening |
| L_tv | D3 | Simultaneously with L_gan | Artifact suppression |

Each objective was added only when the model was stable without it — the curriculum principle (see [Training Curriculum](03_training_curriculum.md#1-curriculum-philosophy)).

---

*End of document. Continue to [06 Failures & Debugging](06_failures_debugging.md).*
