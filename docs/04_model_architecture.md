# Model Architecture

**Related documents:** [03 Training Curriculum](03_training_curriculum.md) | [05 Objective Functions](05_objective_functions.md) | [02 Data Preprocessing](02_data_preprocessing.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25
**Implementation:** [models/sep_vae_v2.py](../models/sep_vae_v2.py)
**Approximate parameter count:** ~73M

---

## 1. Overview

SepVAE V2 is a variational autoencoder that produces **two independent spatial latent codes** from a 256×256 chest X-ray:

- **z_common** (16ch × 16×16): anatomy shared by all patients — lung texture, rib geometry, vertebral structure, cardiac shape for Normal images
- **z_disease** (16ch × 16×16): pathology-specific variation — the enlarged cardiac silhouette of Cardiomegaly, absent for Normal images

For Normal images, z_disease is hard-zeroed before reaching the decoder. This structural constraint is the foundation for all downstream work — it forces z_common to encode a complete reconstruction without any disease information.

```
Input (256×256×1 grayscale CXR)
    │
    ▼  ResNet50Scratch (layers 1–3, GroupNorm, CBAM at every block)
    │
    ▼  SelfAttention2D @ 16×16 bottleneck (256 tokens, 4 heads)
    │
    ├──────────────────────┬────────────────────────
    │                      │
    ▼ bg_branch            ▼ tg_branch
    Layer4BranchGN         Layer4BranchGN
    (16×16×2048)           (16×16×2048)
    │                      │
    ▼                      ▼
    ConvHeadGN             BboxCrossAttnHead
    z_common               z_disease
    (μ_c, logvar_c)        (μ_d, logvar_d, attn_map)
    16ch × 16×16           16ch × 16×16
    │                      │
    │    hard-zero z_d for Normal images
    │                      │
    └────── concat ────────┘
                    │
                    ▼  concat → (16×16×32)
                    │
                    ▼  SepVAEDecoderV2
                    │  16→32→64→128→256 (4× SmoothUp)
                    │
                    ▼
             Output (256×256×1)
```

---

## 2. Encoder — Shared Trunk

### 2.1 ResNet50Scratch

A ResNet-50 trained **from scratch** with GroupNorm throughout. No pretrained weights. GroupNorm is essential for two reasons: (1) stable gradients at small batch sizes (batch_size=6), (2) fully differentiable encoder — unlike a frozen CheSS backbone, all parameters receive gradient signal from the reconstruction loss.

| Stage | Output shape | Block count |
|-------|-------------|-------------|
| Stem (7×7 conv, stride=2) | 128×128×64 | 1 |
| MaxPool (3×3, stride=2) | 64×64×64 | — |
| Layer 1 (filters=64) | 64×64×256 | 3 |
| Layer 2 (filters=128, stride-2 at block0) | 32×32×512 | 4 |
| Layer 3 (filters=256, stride-2 at block0) | 16×16×1024 | 6 |
| SelfAttention2D (4 heads) | 16×16×1024 | — |

**Why not use a pretrained backbone?**
The V1 architecture used a frozen CheSS backbone trained for CXR classification. Its stride=32 compressed the spatial feature map to 16×16 with features optimised for global classification (pooling out spatial detail). The features were incompatible with high-quality reconstruction: the backbone discarded the exact texture information the decoder needed to produce sharp CXRs. V2 trains from scratch so every layer develops reconstruction-relevant features.

### 2.2 CBAM — Convolutional Block Attention Module

Every `BottleneckBlockGN` includes a CBAM on the main branch before the residual add:

**Channel attention:**
```python
avg_pool = h.mean(axis=(-2,-1))   # (B, C)
max_pool = h.max(axis=(-2,-1))    # (B, C)
# shared MLP: C → C/16 → C
gate = sigmoid(MLP(avg_pool) + MLP(max_pool))   # (B, C)
h = h * gate[..., None, None]
```

**Spatial attention:**
```python
# Compress channel axis: mean + max
avg_ch = h.mean(axis=-1, keepdims=True)     # (B, H, W, 1)
max_ch = h.max(axis=-1, keepdims=True)      # (B, H, W, 1)
concat = jnp.concatenate([avg_ch, max_ch], axis=-1)   # (B, H, W, 2)
gate = sigmoid(Conv(1, (7,7))(concat))      # (B, H, W, 1)
h = h * gate
```

CBAM operates at every scale from Layer 1 through Layer 4 branches. The progressive channel attention across layers suppresses lung-parenchyma channels and amplifies cardiac-boundary channels at the bottleneck.

### 2.3 SelfAttention2D at Layer3 Bottleneck

After layer3's 6 bottleneck blocks, a SelfAttention2D module operates on the 16×16 feature map:
- 256 spatial tokens (16×16 positions)
- 4 heads, head dimension = 1024/4 = 256
- O(256²) = 65,536 attention operations — negligible compute cost

This captures the cardiac silhouette-to-lung-field ratio that is the defining signal for cardiomegaly (cardiothoracic ratio > 0.5). A local convolutional field cannot detect this global property; the attention module provides it explicitly.

---

## 3. Encoder — Diverging Branches

### 3.1 Layer4BranchGN

After the shared trunk, two independent `Layer4BranchGN` branches diverge:
- `bg_branch` → common head
- `tg_branch` → disease head

Each is a 3-block `BottleneckBlockGN` stack with **stride=1 throughout**. Output: **(16×16×2048)**.

**Critical: stride=1 (not stride=2)**

An early implementation used stride=2 at the first block (16×16 → 8×8), followed by bilinear upsample back to 16×16. This introduced frequency aliasing — spatial frequencies above the 8×8 Nyquist limit were irreversibly lost. The bilinear upsample added its own low-pass distortion. Result: latent features with false spatial resolution.

The fix: stride=1 throughout with a projection shortcut (1024→2048 channels) at block0:
```python
class Layer4BranchGN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # block0: 1024 → 2048 channels, stride=1 (no spatial downsampling)
        h = BottleneckBlockGN(out_channels=2048, stride=1, use_projection=True)(x)
        # blocks 1-2: 2048 → 2048, no change
        h = BottleneckBlockGN(out_channels=2048, stride=1, use_projection=False)(h)
        h = BottleneckBlockGN(out_channels=2048, stride=1, use_projection=False)(h)
        return h
```

Output stays at 16×16 throughout. All spatial frequencies from layer3 are preserved into the encoder heads.

### 3.2 ConvHeadGN (common head)

Three conv layers (2048→256→128→32) with GroupNorm, producing `(μ_c, log σ²_c)` each **(16×16×16)**.

The common head receives `bg_branch` output (which sees the same anatomy as `tg_branch`, but via an independent set of weights). This independence prevents gradients from the disease head from contaminating common-head features — the branches can specialise.

### 3.3 BboxCrossAttnHead (disease head, D1+)

Produces `(μ_d, log σ²_d)` and an attention map for the bbox supervision loss.

**Mechanism:**

1. **Key projection:** `K = key_proj(h_tg)` where `h_tg` is the tg_branch output. Projected from 2048→256, reshaped to `(B, HW, 256)`. `stop_gradient` for Normal images — prevents Normal image reconstruction gradients from shaping the key projection toward non-cardiac features.

2. **Gaussian spatial prior from bbox:**
   ```python
   cx = (x0 + x1) / 2      # bbox centre (normalised coords)
   cy = (y0 + y1) / 2
   sigma = (x1 - x0) / 4   # quarter-width

   # Create 16×16 grid of positions
   grid_x, grid_y = jnp.meshgrid(jnp.linspace(0, 1, H), jnp.linspace(0, 1, W))
   prior = jnp.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * sigma**2))
   prior = prior / prior.sum()   # normalise to probability distribution
   ```

3. **Bbox-weighted query:** Aggregate K under the Gaussian prior → `Q_bbox ∈ (B, 1, 256)`.

4. **Query blending:** `Q = bbox_query_mix * Q_bbox + (1 - bbox_query_mix) * Q_learned`
   - D1/D2: `bbox_query_mix = 0.7` (70% prior, 30% learned fallback)
   - D3+: `bbox_query_mix = 1.0` (pure prior — encoder is mature)
   - Normal images: always use `Q_learned` (no bbox available)

5. **Cross-attention map:** `A = softmax(Q @ K.T / sqrt(256))` → `(B, H, W)`

6. **Feature gating:** `h_attended = h_tg * (A * HW)` — near-identity at uniform attention (HW = 256, so uniform A = 1/256 → A*HW = 1.0 everywhere), focused when attention localises.

7. **ConvHeadGN on gated features** → `(μ_d, log σ²_d)` — same architecture as common head.

**Why this design?**
The Gaussian prior gives the disease head a strong spatial starting point from epoch 1. Without it, the disease head sees the full 16×16 feature map with uniform attention, and must learn purely from gradient descent that cardiac features (not lung or rib features) should be encoded in z_disease. This takes 30–50 epochs of training time. With the prior, the disease head sees cardiac-weighted features from the very first batch.

---

## 4. Latent Space

### 4.1 Sampling

```python
# Encoder outputs (μ, logvar) for each head
z_c = μ_c + ε * exp(0.5 * logvar_c)   # ε ~ N(0, I)
z_d = μ_d + ε * exp(0.5 * logvar_d)
```

### 4.2 Hard-zero nulling

```python
# disease_mask: 1.0 for Cardiomegaly, 0.0 for Normal
disease_mask = jnp.where(label == CARDIOMEGALY, 1.0, 0.0)
z_d_masked = z_d * disease_mask[:, None, None, None]
```

This is a deterministic architectural operation — not a loss term. It cannot be violated by any gradient update. For Normal images, the decoder input is exactly `concat(z_c, zeros(16×16×16))`.

**Why hard-zero rather than soft nulling loss?**
V1 used a soft nulling loss (`L_null = || μ_d ||² for Normal images`). This drives `μ_d → 0` in expectation but cannot enforce it exactly — some Normal images still produced non-zero z_d. Hard-zero nulling enforces the constraint exactly and allows z_d to remain non-zero in the prior (the conditional KL loss provides the prior pressure), reducing gradient conflict.

### 4.3 Decoder input

```python
z_concat = jnp.concatenate([z_c, z_d_masked], axis=-1)   # (B, 16, 16, 32)
```

For Normal images: `(B, 16, 16, 32)` = 16 channels of z_common + 16 channels of zeros.
For Cardiomegaly: `(B, 16, 16, 32)` = 16 channels of z_common + 16 channels of z_disease.

**Pooled versions** for FactorVAE MI discriminator:
```python
z_c_pooled = z_c.mean(axis=(-2,-1))   # (B, 16) — global average pooled
z_d_pooled = z_d.mean(axis=(-2,-1))   # (B, 16)
```

---

## 5. Decoder

### 5.1 Architecture overview

```
z_concat (16×16×32)
    │
    ▼  z_proj: Conv(32→512, 3×3, GN)
    │
    ├── Stage i=4: 16×16×512, decoder_res_blocks × ResBlockSE
    │
    ├── SmoothUp → 32×32×512
    │
    ├── Stage i=3: 32×32×512, decoder_res_blocks × ResBlockSE
    │              SelfAttention2D (4 heads, 1024 tokens)
    │
    ├── SmoothUp → 64×64×256
    │
    ├── Stage i=2: 64×64×256, decoder_res_blocks × ResBlockSE
    │
    ├── SmoothUp → 128×128×128
    │
    ├── Stage i=1: 128×128×128, decoder_res_blocks × ResBlockSE
    │
    ├── SmoothUp → 256×256×128
    │
    ├── Stage i=0: 256×256×128, decoder_res_blocks × ResBlockSE
    │
    └── GN → swish → Conv(128→1, 3×3) → sigmoid
         Output (256×256×1)
```

**Channel schedule (coarse→fine):** `(512, 512, 256, 128, 128)` — i.e., `ch_mults=(128,128,256,512,512)`.
At 256×256 there are 128 channels. Earlier schedules used 64 channels at full resolution — only 4 conv operations to generate the 1-channel output, an information bottleneck that produced blurry reconstructions. 128 channels provide adequate capacity.

`decoder_res_blocks = 3` (introduced in D2). Flax names blocks by loop index, so checkpoint resumption is clean — `ResBlockSE_0`, `ResBlockSE_1` load from previous checkpoint, `ResBlockSE_2` initialises near-identity.

### 5.2 ResBlockSE

The decoder's core building block:

```python
class ResBlockSE(nn.Module):
    features: int
    se_reduction: int = 8

    @nn.compact
    def __call__(self, x):
        h = nn.GroupNorm()(x)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3,3), padding='SAME')(h)
        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3,3), padding='SAME')(h)

        # Squeeze-and-Excitation gate
        s = h.mean(axis=(-3,-2))              # (B, C) — global average pool
        s = nn.Dense(self.features // self.se_reduction)(s)
        s = nn.relu(s)
        s = nn.Dense(self.features)(s)
        s = nn.sigmoid(s)                     # (B, C)
        h = h * s[:, None, None, :]           # channel-wise scaling

        # Residual
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1,1))(x)
        return x + h
```

**SE gate rationale:** When z_disease is non-zero, the SE gate learns to amplify cardiac-silhouette channels and suppress lung/bone channels at every decoder level. This makes the disease latent→feature mapping explicit — the SE gate acts as a "which channels matter for cardiomegaly reconstruction" selector.

`se_reduction=8`: with 128 channels at fine resolution, hidden dim = 16 — adequate for independent excitation of 16 channel combinations.

### 5.3 SmoothUp

```python
class SmoothUp(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # Bilinear resize (no transposed conv → no checkerboard)
        x = jax.image.resize(x, (B, H*2, W*2, C), method='bilinear')
        x = nn.Conv(self.features, (3,3), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.Conv(self.features, (3,3), padding='SAME')(x)
        x = nn.swish(x)
        return x
```

**Why bilinear + conv rather than transposed convolution?** Transposed convolution with stride=2 produces checkerboard artifacts — every output pixel receives contributions from a different number of kernel positions depending on its spatial location. The periodic variation manifests as a visual grid pattern. Bilinear resize has no such overlap artifacts. The subsequent 3×3 convolutions add learnable high-frequency content recovery without the period-2 aliasing.

### 5.4 SelfAttention2D at 32×32 (decoder)

Added at stage i=3 (32×32), after the ResBlockSE processing and before SmoothUp:
- 1024 spatial tokens (32×32 positions)
- 4 heads
- ~1M attention operations — cheap relative to conv layers

**Why at 32×32 specifically?** The cardiac silhouette spans roughly 30–50% of image width. At 32×32, each position corresponds to 8×8 pixels — the spatial scale at which the silhouette boundary is most salient. The decoder must coordinate the left and right cardiac borders consistently (they are spatially separated by the full heart width) — local convolutions cannot enforce this consistency; attention can.

---

## 6. V1 vs V2 Architecture Comparison

| Aspect | V1 (CheSS backbone) | V2 (ResNet50Scratch) |
|--------|--------------------|--------------------|
| Backbone | CheSS (frozen, pretrained for CXR classification) | ResNet-50, trained from scratch |
| Normalisation | BatchNorm (frozen stats, unstable at small batch) | GroupNorm throughout (stable at batch=6) |
| Feature stride | 32 (16×16 features from 512×512 input) | 16 (16×16 features from 256×256 input) |
| Reconstruction quality | Blurry (backbone discards texture) | Sharp ribs/vessels/cardiac border |
| Trainability | Frozen — cannot adapt for reconstruction | Fully trainable |
| Disease head | ConvHead with learned attention | BboxCrossAttnHead with Gaussian prior |
| Decoder | 4 SmoothUp blocks, ch=(64,128,256,512,512) | 4 SmoothUp blocks, ch=(128,128,256,512,512), SE gates |
| Self-attention | Encoder bottleneck only | Encoder bottleneck + decoder 32×32 |

---

## 7. Parameter Breakdown

| Component | Parameters |
|-----------|-----------|
| ResNet50Scratch (shared trunk, layers 1–3) | ~23M |
| bg_branch (Layer4BranchGN, 3 blocks) | ~11M |
| tg_branch (Layer4BranchGN, 3 blocks) | ~11M |
| ConvHeadGN (common) | ~1.5M |
| BboxCrossAttnHead (disease) | ~1.5M |
| SepVAEDecoderV2 | ~25M |
| **Total** | **~73M** |
| FactorVAE discriminator (separate) | ~0.5M |
| NLayerDiscriminator / PatchGAN (D3+) | ~2M |

---

## 8. Forward Pass Summary

```python
# 1. Encode
h_shared = resnet50_scratch(x)                          # (B, 16, 16, 1024)
h_shared = self_attn_encoder(h_shared)                   # same shape

h_bg = bg_branch(h_shared)                              # (B, 16, 16, 2048)
h_tg = tg_branch(h_shared)                              # (B, 16, 16, 2048)

μ_c, logvar_c = conv_head_common(h_bg)                  # (B, 16, 16, 16) each
μ_d, logvar_d, attn_map = bbox_cross_attn_head(h_tg, bbox)  # (B, 16, 16, 16), (B, 16, 16)

# 2. Sample
z_c = μ_c + ε_c * exp(0.5 * logvar_c)
z_d = μ_d + ε_d * exp(0.5 * logvar_d)

# 3. Hard-zero nulling
z_d_masked = z_d * label_mask[:, None, None, None]

# 4. Decode
z_concat = concat([z_c, z_d_masked], axis=-1)           # (B, 16, 16, 32)
x_rec = decoder(z_concat)                               # (B, 256, 256, 1)

# 5. Pool for MI discriminator
z_c_pooled = z_c.mean(axis=(-2,-1))                     # (B, 16)
z_d_pooled = z_d.mean(axis=(-2,-1))                     # (B, 16)
```

---

*End of document. Continue to [05 Objective Functions](05_objective_functions.md).*
