# Chapter 02 — Architecture: What We Built and Why

**Previous chapter:** [01 Project Overview](01_project_overview_and_motivation.md)
**Next chapter:** [03 Empirical Results](03_empirical_results.md)

---

## 2. Architecture: What We Built and Why

### 2.1 Encoder

```
Input x (512×512 CXR, 1 channel)
        │
        ▼
┌────────────────┐
│ CheSS backbone │  ResNet-50 pretrained on CXR classification (frozen)
│   (stride 32)  │  Output: feature map f ∈ ℝ^{16×16×2048}
└───────┬────────┘
        │  (optionally bilinear-upsampled to 64×64 if use_fpn=False)
        │  (or FPN-fused from layer2/3/4 to 64×64 if use_fpn=True)
        ▼
┌────────────────────────────────────────────────────┐
│                 Three encoder heads                 │
│                                                     │
│  head_common:     ConvHead → μ_c, σ_c   (4 ch)    │
│  head_cardiomegaly: ConvHead → μ_d1, σ_d1 (2 ch)  │
│  head_effusion:   ConvHead → μ_d2, σ_d2 (2 ch)    │
└────────────────────────────────────────────────────┘
        │
        ▼
z = cat[z_common (4ch), z_cardio (2ch), z_effusion (2ch)]  @  64×64 spatial
```

**Why frozen backbone:** CheSS provides rich, CXR-specific features without needing to train feature extraction from scratch. Freezing it avoids catastrophic forgetting and keeps training stable. The encoder heads are lightweight ConvHead modules trained on top.

**Why 64×64 spatial latents:** Spatial latents preserve where in the image the disease information lives. Global pooled vectors (standard VAE) discard spatial structure that is needed for region-selective editing.

**Why 4 + 2 + 2 channel split:** Empirically small disease heads discourage the model from routing common structure into them (the reconstruction gradient is small relative to the KL penalty). The common head has 4 channels to give it enough capacity for anatomy, pose, and acquisition variation.

### 2.2 Disease label routing

The dataloader provides triplets: $(x_{\text{norm}},\, x_{\text{eff}},\, x_{\text{cardio}})$. Within each batch, labels $y \in \{0, 1, 2\}$ control which heads are "active":

| Label | Meaning | Active head | Inactive heads |
|-------|---------|-------------|----------------|
| $y=0$ | Normal | none | $z_{\text{cardio}}$, $z_{\text{effusion}}$ |
| $y=1$ | Effusion | $z_{\text{effusion}}$ | $z_{\text{cardio}}$ |
| $y=2$ | Cardiomegaly | $z_{\text{cardio}}$ | $z_{\text{effusion}}$ |

Inactive heads are penalised toward $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$ — a narrow prior that forces the mean to zero and prevents the head from encoding anything for images of the wrong disease class.

### 2.3 Decoder

```
z (64×64 × 8 channels)
        │
        ▼  initial conv 8ch → 256ch
        │
        ▼  ResBlocks + SmoothUp ×3:
        │     64×64 → 128×128 → 256×256 → 512×512
        │     (bilinear-upsample or subpixel-shuffle, configurable)
        ▼
        Conv 1×1 → Sigmoid
        │
        ▼
x̂  (512×512 × 1 channel)
```

All three heads are concatenated before the decoder. The decoder is shared — it mixes all channels in its first convolution. This has important implications for composition strategies (discussed in [Chapter 09](09_composition_theory.md)).

### 2.4 Loss portfolio (initial)

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \mathcal{L}_{\text{KL}} + w_{\text{perc}} \mathcal{L}_{\text{perc}} + w_{\text{null}} \mathcal{L}_{\text{null}} + w_{\text{orth}} \mathcal{L}_{\text{orth}} + w_{\text{MI}} \mathcal{L}_{\text{MI}}$$

Where:
- $\mathcal{L}_{\text{rec}}$: L2 pixel reconstruction
- $\mathcal{L}_{\text{KL}}$: per-channel KL with free-bits threshold and warmup
- $\mathcal{L}_{\text{perc}}$: VGG perceptual loss
- $\mathcal{L}_{\text{null}}$: penalises non-zero $\mu$ on inactive heads — $\mathbb{E}[\mu_{\text{inactive}}^2]$
- $\mathcal{L}_{\text{orth}}$: Barlow-style cross-correlation + prototype cosine penalty
- $\mathcal{L}_{\text{MI}}$: MI discriminator (joint vs shuffled pair, trained adversarially)

### 2.5 Key metrics we track

- **Probe AUC**: linear classifier trained on disease head latents; higher = head captures more disease information
- **Cross-head score**: mean ratio of inactive-head energy to active-head energy; 0.5 = ideal (no leakage), 1.0 = total leakage
- **KL per head**: to detect dead zones and collapse

---

## Supplementary: Architecture Standalone Overview

*The following provides a condensed standalone reference for the architecture that is useful for writing and presentation.*

```
 Input x (512×512 CXR)
        │
        ▼
 ┌──────────────┐
 │ CheSS encoder│  (frozen ResNet-50, pretrained on CXR)
 │ (backbone)   │
 └──────┬───────┘
        │  feature map f ∈ ℝ^{H×W×C}
        ▼
 ┌──────────────────────────────────────────────┐
 │              Encoder heads                    │
 │                                               │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
 │  │ μ_c, σ_c     │  │ μ_d1, σ_d1  │  │ μ_d2, σ_d2  │  │
 │  │ (z_common)   │  │ (z_cardio)  │  │ (z_effusion)│  │
 │  │ 4 channels   │  │ 2 channels  │  │ 2 channels  │  │
 │  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  │
 └─────────┼─────────────────┼────────────────┼──────────┘
           │                 │                │
           └─────────────────┴────────────────┘
                             │
                    z = cat[z_common, z_cardio, z_effusion]
                             │
                             ▼
                    ┌─────────────────┐
                    │    Decoder      │  (shared conv upsampling)
                    └────────┬────────┘
                             │
                             ▼
                        x̂ (reconstruction)
```

**Disease label routing (detailed):**
The triplet dataloader provides `(x_norm, x_disease1[effusion], x_disease2[cardio])` per batch.
At forward pass, concatenated batch has labels $y \in \{0, 1, 2\}$:
- $y=0$ (normal): $z_{\text{cardio}}$ and $z_{\text{effusion}}$ are both **inactive** — penalised toward $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$
- $y=1$ (effusion): $z_{\text{effusion}}$ is **active** — free ELBO; $z_{\text{cardio}}$ is inactive
- $y=2$ (cardio): $z_{\text{cardio}}$ is **active** — free ELBO; $z_{\text{effusion}}$ is inactive

---

*End of Chapter 02. Continue to [Chapter 03: Empirical Results](03_empirical_results.md).*
