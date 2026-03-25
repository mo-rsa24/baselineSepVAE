# Project Overview — baselineSepVAE vs. superdiff-ldm

> See also: [PLAN_D.md](PLAN_D.md) | [PLAN_D5.md](PLAN_D5.md) | [PREPROCESSING.md](PREPROCESSING.md)
> Research log (shared): [research_log/00_INDEX.md](research_log/00_INDEX.md)

---

## Shared Goal

Both repos are the **same research project**: disentangled chest X-ray generation for compositional multi-pathology synthesis (cardiomegaly + pleural thickening without joint training on comorbid data). The `research_log/` in this repo was copied verbatim from `superdiff-ldm/results/research_log/`.

The ultimate aim is: given a model that has learned what cardiomegaly looks like and a separate model that has learned what pleural thickening looks like, synthesise a realistic image containing *both* without ever having jointly trained on comorbid data. The SepVAE is the **first building block** — everything downstream (LDM training, score composition) depends on its latents being genuinely disentangled.

---

## The Key Difference: Encoder Architecture

| | superdiff-ldm (V1) | baselineSepVAE (V2 / Plan D) |
|---|---|---|
| **Backbone** | CheSS ResNet-50 **frozen in encoder** | ResNet-50 trained **from scratch** |
| **Latent spatial size** | 64×64×8 | 16×16×32 |
| **Training resolution** | 512×512 | 256×256 (D0–D4); 512×512 planned for D5 |
| **CheSS role** | Backbone (encoder feature extractor) | **Perceptual loss only** (frozen, D4+) |
| **Encoder norm** | BatchNorm (frozen CheSS stats, small-batch unstable) | **GroupNorm throughout** |
| **Disease head** | `ConvHead` (plain conv) | `BboxCrossAttnHead` (Gaussian-prior spatial attention from bbox) |
| **Self-attention** | None in encoder | `SelfAttention2D` at layer3 bottleneck (16×16) |

---

## Why the Rewrite

The CheSS backbone was trained discriminatively. Its feature space is incompatible with N(0,I), causing catastrophically high KL at initialisation and hazy reconstructions that are slow to recover. **baselineSepVAE removes CheSS from the encoder entirely**, which eliminates the root cause.

CheSS re-enters only in D4 as a **frozen perceptual loss** — extracting L1 feature distances at ResNet layers 2/3/4 to sharpen reconstruction texture without touching the encoder KL. This is the role it is actually suited for.

---

## Phase Progression (baselineSepVAE)

Defined in [PLAN_D.md](PLAN_D.md). Each phase adds one component and must pass a gate before the next activates.

| Phase | New component | Gate |
|---|---|---|
| D0 | ResNet-50 from scratch + GroupNorm + self-attn | No OOM; KL < 50k at step 0; structure visible by epoch 3 |
| D1 | `BboxCrossAttnHead` in disease head | Attn maps localise to cardiac region by epoch 5 |
| D2 | FactorVAE MI discriminator | MI acc → 0.5; z_common ⊥ z_cardio confirmed |
| D3 | Pleural Thickening + second disease head | Both heads respond only to their class; MI(z_ca, z_pt) ≤ 0.55 |
| D4 | CheSS perceptual loss (frozen, loss only) | Recon visually sharper; O1 metrics stable (< 10% regression) |
| D5 | Reconstruction quality fixes (current) | See [PLAN_D5.md](PLAN_D5.md) |

> **Note:** D5 in the original PLAN_D.md was "scale to 512×512". In practice, D5 was repurposed to address reconstruction quality issues first (PatchGAN, stride aliasing, decoder channel width, bbox supervision). The scale-up to 512×512 follows after quality is confirmed at 256×256.

---

## D5 — Current Focus

Defined in [PLAN_D5.md](PLAN_D5.md). All architecture (Phase 1) and training script (Phase 2) changes are complete. D5 resumes from the D4 checkpoint at epoch 160.

Key additions over D4:
- **PatchGAN discriminator** (hinge loss, activates at step 5000)
- **TV loss** (suppresses stripe artifacts)
- **Bbox attention loss re-enabled** (`weight_bbox_attn=0.1` — was 0.0 in D2/D3/D4)
- **Decoder capacity increase** (`ch_mults` 64→128 at 256×256 level)
- **Decoder self-attention** at 32×32 level
- **Layer4 branch aliasing fix** (stride=1, no round-trip downsample/upsample)
- **MSE rebalanced** (`weight_rec=2.0` to compensate for perceptual dominance)

---

## Downstream Plan (superdiff-ldm)

Once baselineSepVAE's latents are clean and disentangled (gates G1–G4 in `research_log/10_verification_and_hypothesis.md`), the downstream work in `superdiff-ldm` takes over:

1. Pre-encode the training set to `(z_common, z_cardio, z_plthick)` latents
2. Train conditional LDMs on each disease sub-block separately
3. Compose scores at inference time (Strategy A: conditional independence assumed) to synthesise comorbid images
4. Evaluate composition against a jointly-trained oracle

See `superdiff-ldm/results/research_log/07_ldm_proof_of_concept.md` and `09_composition_theory.md` for the full plan.
