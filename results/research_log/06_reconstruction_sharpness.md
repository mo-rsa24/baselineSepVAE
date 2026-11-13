# Chapter 06 — Reconstruction Sharpness Investigation

**Previous chapter:** [05 Fixes R1–R7](05_fixes_r1_to_r7.md)
**Next chapter:** [07 LDM Proof of Concept](07_ldm_proof_of_concept.md)

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

## Supplementary: Mathematical Detail on Blurriness Causes

*The following provides additional mathematical context on why each source of blurriness compounds.*

### Compression ratio detail

With $z_\text{common} = 4$ channels, $z_\text{cardio} = z_\text{effusion} = 2$ channels each, the spatial bottleneck is:

$$\text{compression ratio} = \frac{8 \times 16 \times 16}{512 \times 512} = \frac{2048}{262144} \approx 0.78\%$$

At this ratio the decoder is necessarily hallucinating the vast majority of image content. The hallucination is anchored by the 8 latent channels but cannot be verified against ground truth at the pixel level — only the blurry mean is consistent across all possible sharp completions.

### L2 loss averages over the posterior

The ELBO reconstruction term:

$$\mathcal{L}_\text{rec} = \mathbb{E}_{q(z|x)}\left[\|x - \hat{x}_\theta(z)\|_2^2\right]$$

The optimal decoder under this objective (for any posterior with non-zero variance) is:

$$\hat{x}_\theta(z) = \mathbb{E}_{p(x|z)}[x]$$

For natural images, this expectation is a blurred version of any plausible sharp image. The sharper the texture, the more variance there is across plausible completions, and the more blurred the mean.

### The discriminator fix

A PatchGAN discriminator $D$ trained adversarially forces:

$$\mathcal{L}_\text{adv} = -\log D(\hat{x})$$

Because averaged, blurry textures look statistically unlike real X-ray patches, the discriminator learns to reject them. The generator (decoder) is then forced to commit to a single sharp realisation rather than an average. This is the standard explanation for why VAE-GANs (e.g. VQ-GAN) are sharper than plain VAEs.

### Subpixel path vs bilinear path

The current default `bilinear` path in `SmoothUp`:
```python
# bilinear path (current default)
h = jax.image.resize(x, target_shape, method='bilinear')  # low-pass filter
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 1
h = nn.Conv(ch, (3,3), ...)(h)                             # smoothing conv 2
```

The alternative `subpixel` path (already implemented in `models/sep_vae_jax.py:68–82`) uses pixel-shuffle, which is a learned rearrangement of channels into spatial positions and has no inherent low-pass bias. Three upsampling stages × two smoothing convolutions = **six successive low-pass operations** in the bilinear path before the final sigmoid. High-frequency energy is progressively suppressed with each stage.

---

*End of Chapter 06. Continue to [Chapter 07: LDM Proof of Concept](07_ldm_proof_of_concept.md).*
