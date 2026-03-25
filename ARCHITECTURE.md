# SepVAE V2 — Architecture Description

> See also: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | [PLAN_D.md](PLAN_D.md) | [PLAN_D5.md](PLAN_D5.md)
> Implementation: [models/sep_vae_v2.py](models/sep_vae_v2.py)

---

## Overview

SepVAE V2 is a variational autoencoder designed to produce **factorised spatial latent codes** from 256×256 chest X-rays. The encoder produces two separate latent heads:

- `z_common` — shared anatomy, patient position, acquisition settings
- `z_disease` — disease-specific variation (cardiomegaly or pleural thickening)

For Normal images, `z_disease` is hard-zeroed at the decoder input, forcing the model to reconstruct anatomy using only `z_common`. This structural constraint is the foundation for downstream composition.

```
Input (256×256×1)
    │
    ▼  ResNet50Scratch (layers 1–3, GroupNorm, CBAM at every block)
    │
    ▼  SelfAttention2D @ 16×16 bottleneck (256 tokens)
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
    (16×16×16)             (16×16×16)
    │                      │
    └──── hard-zero z_disease for Normal ───┘
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

## Encoder

### Shared Trunk — `ResNet50Scratch`

A ResNet-50 trained **from scratch** (no pretrained weights). GroupNorm throughout — no frozen BatchNorm statistics, stable at small batch sizes.

| Stage | Output shape | Notes |
|---|---|---|
| Stem (7×7, stride 2) | 128×128×64 | GN + ReLU |
| Max pool (3×3, stride 2) | 64×64×64 | |
| Layer 1 (3 blocks, filters=64) | 64×64×256 | |
| Layer 2 (4 blocks, filters=128, stride-2 at b0) | 32×32×512 | |
| Layer 3 (6 blocks, filters=256, stride-2 at b0) | 16×16×1024 | |
| SelfAttention2D (4 heads) | 16×16×1024 | 256 tokens, O(65k) ops |

Every `BottleneckBlockGN` block runs **CBAM** (Convolutional Block Attention Module) on its main branch before the residual add:
- **Channel attention**: shared MLP on avg-pooled + max-pooled descriptors → per-channel sigmoid scale. Suppresses lung-parenchyma channels, amplifies cardiac-boundary channels.
- **Spatial attention**: 7×7 conv on avg+max channel descriptors → spatial sigmoid map. Progressively focuses the spatial field-of-view on the heart boundary across layers 1→4.

The **SelfAttention2D** at the end of layer 3 captures long-range dependencies (cardiac silhouette vs. surrounding lung fields) at a cost of only 256² = 65k attention operations.

### Diverging Layer4 Branches — `Layer4BranchGN`

After the shared trunk, two **independent** layer4 branches diverge:

- `bg_branch` → feeds the common head
- `tg_branch` → feeds the disease head

Each is a 3-block `BottleneckBlockGN` stack with `stride=1` throughout. The first block uses a projection shortcut (1024→2048 channels) with no spatial downsampling. This avoids the stride-2 → bilinear-upsample round-trip that caused aliasing in earlier versions.

Output of each branch: **(16×16×2048)**.

### Common Head — `ConvHeadGN`

Three conv layers (2048→256→128→32) producing `(μ_c, log σ²_c)`, each **(16×16×16)**.

### Disease Head — `BboxCrossAttnHead` (D1+)

Produces `(μ_d, log σ²_d)` and an **attention map** for the bbox supervision loss.

**Mechanism:**

1. Project branch features to key vectors: `K ∈ (B, HW, 256)`
2. Compute a Gaussian spatial prior from the ground-truth bbox:
   - Center: `(cx, cy) = ((x0+x2)/2, (y0+y1)/2)`
   - σ = bbox_width/4 (quarter-width; prior drops to ~14% at the bbox boundary, stays inside the heart)
3. Weighted aggregate of `K` under the Gaussian → bbox-driven query `Q_bbox ∈ (B, 1, 256)`
4. For Normal images (`has_bbox=0`): use a learned fallback query `Q_learned` instead
5. Cross-attention: `Q × K^T → softmax → attn_map (B, H, W)`
6. Gate encoder features: `h * (attn_map * HW)` — near-identity at uniform attention, focused when attention localises
7. `ConvHeadGN` on gated features → `(μ_d, log σ²_d)`

This gives the disease head a strong spatial prior from **epoch 1**, bypassing the slow convergence of a purely learned query.

---

## Latent Space & Head Nulling

After sampling `z_c ~ N(μ_c, σ_c²)` and `z_d ~ N(μ_d, σ_d²)`:

| Label | `z_disease` passed to decoder | Rationale |
|---|---|---|
| Normal (0) | **zeros** | Forces `z_common` to reconstruct anatomy alone |
| Disease (1) | `z_d` (sampled) | Disease-specific features can contribute |

The hard-zero mask is: `z_d_decode = z_d * (label == 1)`.

Concatenated latent passed to decoder: **(B, 16×16, 32)** — 16ch common + 16ch disease.

Global-pooled versions `z_c_pooled` and `z_d_pooled` are used by the **FactorVAE MI discriminator** to enforce `z_common ⊥ z_disease` in feature space.

---

## Decoder — `SepVAEDecoderV2`

SE-gated decoder: **16×16×32 → 256×256×1** via 4 bilinear upsamples.

```
z (16×16×32)
    │  Conv 3×3 → 512ch
    │
    ├── 2× ResBlockSE (512ch) @ 16×16
    ├── SmoothUp → 32×32×256
    ├── 2× ResBlockSE (256ch) @ 32×32
    ├── SelfAttention2D (4 heads) @ 32×32        ← 1024 tokens
    ├── SmoothUp → 64×64×128
    ├── 2× ResBlockSE (128ch) @ 64×64
    ├── SmoothUp → 128×128×128
    ├── 2× ResBlockSE (128ch) @ 128×128
    ├── SmoothUp → 256×256×128 (no ch change)
    ├── 2× ResBlockSE (128ch) @ 256×256
    │
    ▼  GN → swish → Conv 3×3 → sigmoid
    Output (256×256×1)
```

**Channel schedule** (fine→coarse): `(128, 128, 256, 512, 512)`. The 256×256 level uses 128 channels (doubled from 64 in D0–D4) to prevent a high-frequency information bottleneck.

**`ResBlockSE`**: standard ResBlock (GN→swish→conv→GN→swish→conv) with a **Squeeze-and-Excitation** gate on the main branch before the residual add. SE uses global average pool only (not max pool) — the decoder is generative, needing mean channel activation levels rather than peak detection. `se_reduction=8` (hidden dim = C/8) gives adequate capacity at fine scales.

**`SmoothUp`**: bilinear resize + 2× conv3×3. No transposed convolutions → no checkerboard artifacts.

**`SelfAttention2D` at 32×32**: coordinates the cardiac silhouette globally before upsampling to finer scales. 1024 tokens, ~1M attention ops — cheap relative to the conv layers.

---

## Loss Portfolio

| Loss | Weight (D5) | Purpose |
|---|---|---|
| MSE reconstruction | 2.0 | Pixel-level accuracy |
| KL common | 1e-4 | Regularise `z_common` → N(0,I) |
| KL disease | 1e-4 | Regularise `z_disease`; tight prior (σ=0.1) for inactive heads |
| FactorVAE MI | 1.0 | `z_common ⊥ z_disease` in feature space |
| Perceptual (CheSS) | 0.05 | Texture sharpness via frozen CheSS L1 feature distances |
| Bbox attention | 0.1 | Force attn map to stay inside ground-truth bbox |
| PatchGAN hinge | 0.5 | Activates at step 5000; enforces photorealistic texture |
| Total variation | 1e-3 | Suppresses stripe artifacts |

---

## Parameter Count (approximate)

| Component | Parameters |
|---|---|
| ResNet50Scratch (shared trunk) | ~23M |
| Two Layer4 branches | ~2× 11M = 22M |
| Encoder heads (common + disease) | ~3M |
| Decoder (SepVAEDecoderV2) | ~25M |
| **Total** | **~73M** |
