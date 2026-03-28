# Forward Plan — D4 through LDM

**Related documents:** [08 Current State (D3)](08_current_state_d3.md) | [07 Results & Evaluation](07_results_evaluation.md) | [03 Training Curriculum](03_training_curriculum.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25

---

## 1. Overview

From D3, the project has three parallel tracks:

**Track 1 — SepVAE refinement (D4–D7):** Continue improving reconstruction quality and latent disentanglement. Each stage adds architectural or objective changes.

**Track 2 — LDM training (Phase 8+):** Once SepVAE passes the verification gates G1–G4, pre-encode the dataset into (z_common, z_disease) pairs and train conditional LDMs.

**Track 3 — Mask supervision (feature/mask-supervision branch):** Replace flat bbox supervision with physician-validated cardiac masks from CheXmask (VinDr-CXR, 13,970 images). See [10 Mask Supervision Journey](10_mask_supervision_journey.md) for the full investigation. This track feeds back into D4 if loss-weight disambiguation (§8) confirms signal quality is the bottleneck.

The tracks are not strictly sequential — if G1–G4 pass at D3 final, LDM preencoding can start in parallel with D4 SepVAE training.

---

## 2. SepVAE Refinement Track (D4–D7)

### 2.1 D4 — TBD (Objective Tuning)

**Status:** Not planned in detail. Depends on D3 final checkpoint quality assessment.

**Candidates:**
- Raise `weight_bbox_attn` from 0.10 to 0.15–0.20 if attention maps are not yet sufficiently concentrated at D3 end
- Add probe AUC evaluation to confirm G1 before proceeding
- Adjust `weight_mi_factor` if FactorDisc accuracy drifts above 0.60 at D3 end

**Principle:** D4 should only change one thing vs. D3. If D3 is already passing all G1–G4 gates, D4 can be skipped.

**Resume from:** D3 final checkpoint (`runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_final.pkl`)

---

### 2.2 D5 — TBD (Objective or Architecture Tuning)

**Status:** Not planned in detail.

**Possible direction:** If probe AUC for z_disease is below G1 threshold after D4, D5 could add the disease discriminability classifier (R11 from V1 framework):
$$\mathcal{L}_{\text{clf}} = \text{BCE}\left(\hat{y}_k, \mathbf{1}[\text{label} = k]\right), \quad \hat{y}_k = \sigma\left(\text{MLP}(\text{GAP}(\mu_k))\right)$$

This directly trains the encoder to satisfy G1 during training, rather than hoping it emerges from reconstruction alone.

---

### 2.3 D6 — TBD

Reserved. Depends on D4–D5 outcomes.

---

### 2.4 D7 — UNet Skip Connections + Wider z_common

**Status:** Planned. Architectural change with highest regression risk.
**Launcher:** `slurm_scripts/d7_skip.slurm`
**Resume from:** D6 final checkpoint

#### Architectural changes vs D6

**Change 1 — UNet skip connections:**

```
Current (D1–D6):
    Encoder: ResNet50 → h_shared → branches → z_c, z_d
    Decoder: z_concat → decoder stages (no skip)

D7:
    Encoder: ResNet50 → h_shared (16×16×1024) → branches → z_c, z_d
             Also exposes: h_layer3 (16×16×1024), h_layer2 (32×32×512)
    Decoder: z_concat → decoder
             At i=4 (16×16): fuse h_layer3 with decoder h via concat + 1×1 conv
             At i=3 (32×32): fuse h_layer2 with decoder h via concat + 1×1 conv
```

Injection mechanism:
```python
# At decoder stage i=4 (16×16, 512ch):
h_skip3 = skip3_fuse(concat([h_decoder, h_layer3], axis=-1))  # 512+1024 → 512

# At decoder stage i=3 (32×32, 512ch):
h_skip2 = skip2_fuse(concat([h_decoder, h_layer2], axis=-1))  # 512+512 → 512
```

`skip3_fuse` and `skip2_fuse` are `Conv(512, 1×1)` — freshly initialised, so all other weights load cleanly from D6.

**Why skip connections?**
The ResNet encoder compresses spatial information into the 16×16 bottleneck. Fine-grained spatial details (the exact edge location of a rib, the vessel diameter at a specific position) are partially lost in this compression even at 16×16. Skip connections bring encoder intermediate features directly into the decoder, bypassing the bottleneck and recovering this spatial detail. This is the standard U-Net design principle.

**Change 2 — z_channels_common 16 → 32:**

Doubles the common latent capacity: 16×16×16 = 4,096 values → 16×16×32 = 8,192 values.

**Implication:** The common head encoder (`ConvHeadGN`) and the decoder z_proj (`Conv(32→512, 3×3)`) are re-initialised (their shapes change). All other encoder and decoder weights are restored from D6. This is a partial re-initialisation — expect a ~10–20 epoch transient regression before the new parameters learn.

#### D7 configuration

| Parameter | D6 | D7 | Rationale |
|-----------|----|----|-----------|
| z_channels_common | 16 | **32** | Doubles common latent capacity |
| skip connections | None | **layer3 (16×16) + layer2 (32×32)** | Recover fine spatial detail |
| weight_perceptual | 0.05 | **0.0** | Skip connections + GAN replace CheSS perceptual |
| lr_vae | 1e-4 | **5e-5** | Lower for architectural transition |
| batch_size | 6 | **4** | Skip connections increase peak memory |
| gan_start_step | 2000 | **500** | Skip connections accelerate decoder convergence |

**Why weight_perceptual=0.0 in D7?**
Skip connections bring encoder features directly into the decoder, providing the mid-frequency texture guidance that perceptual loss was supplying externally. Disabling perceptual loss at D7 reduces aliasing risk (since CheSS perceptual was the source of stripe artifacts) and reduces the interaction between two competing sharpness signals.

**What to watch in D7:**
- Do not judge D7 quality before epoch 20 from resume — the new skip connections and z_common head are randomly initialised
- Monitor `loss/reconstruction` — expect a transient increase in the first 10 epochs as the new skip projection weights adapt
- Monitor `metrics/z_cardio_norm_ratio` — if skip connections cause z_common to absorb more cardiac information, this ratio will drop

**Risk:** D7 is the highest-risk planned stage because two things change simultaneously (skip connections + z_common width). The safer version is D7-A (skip connections only, z_common=16) followed by D7-B (z_common=32). If D7 combined regresses badly, split into two stages.

---

## 3. Verification Gates (Gate All LDM Work)

Before starting LDM preencoding (Track 2), the following four gates must all pass. These can be evaluated on the D3 final checkpoint — LDM training does not need to wait for D7.

| Gate | Criterion | How to measure |
|------|-----------|---------------|
| **G1: Disease discriminability** | Frozen linear probe on GAP(z_d_mu) achieves AUC > 0.75 for cardiomegaly | `scripts/eval_probe_auc.py --ckpt D3_final --head disease` |
| **G2: Common purity** | Frozen linear probe on GAP(z_c_mu) achieves AUC < 0.65 for cardiomegaly | `scripts/eval_probe_auc.py --ckpt D3_final --head common` |
| **G3: Edit purity** | Nulling z_d for a Cardiomegaly image produces a reconstruction without enlarged cardiac silhouette (qualitative + bbox-masked pixel delta) | Visual inspection + `scripts/eval_counterfactual.py` |
| **G4: Normal fidelity** | Normal image reconstructions are anatomically plausible without disease traces | Visual inspection |

If G1–G4 pass at D3: proceed to preencoding immediately. D4–D7 can run in parallel.
If G1 fails: add disease discriminability classifier (R11) in D4 before preencoding.
If G2 fails: investigate z_common leakage — increase `weight_masked_rec` or add neutral-decode consistency loss.
If G3 fails: attention concentration is insufficient — increase `weight_bbox_attn`.

---

## 4. LDM Training Track (Phase 8+)

### 4.1 Preencoding (prerequisite)

Pre-encode the full training set into (z_common, z_disease) pairs using the verified SepVAE checkpoint:

```bash
python scripts/preencode_sepvae_v2_latents.py \
    --checkpoint runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_final.pkl \
    --csv_path /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
    --output_dir preencoded_latents/d3_final \
    --batch_size 16 \
    --num_workers 8
```

Output: one `.npy` file per image with shape `(2, 16, 16, 16)` = (z_common, z_disease), plus `latent_meta.json` with statistics (mean, std, scale factor).

The scale factor is critical for LDM normalisation:
```python
import json
meta = json.load(open('preencoded_latents/d3_final/latent_meta.json'))
latent_scale_factor = meta['latent_scale_factor']
# Apply to all z_common / z_disease before LDM training
```

---

### 4.2 Phase 8 — LDM Training (Strategy A)

Train three conditional LDMs:

| LDM | Input to score function | Conditioning | Purpose |
|-----|------------------------|-------------|---------|
| LDM_common | z_common (4ch × 64×64, remapped) | Unconditioned | Model distribution of normal anatomy |
| LDM_cardio | z_cardio (2ch × 64×64) | z_common via cross-attention | Sample cardiac-specific latents conditioned on anatomy |
| LDM_effusion | z_effusion (2ch × 64×64) | z_common via cross-attention | Sample effusion-specific latents (future) |

Note: In the current V2 training, z_disease has 16 channels split between cardiomegaly and normal disease heads. The LDM training splits these into the appropriate semantic subsets based on the labels.

**VP-SDE diffusion process:**
$$dz = -\frac{1}{2}\beta(t)z\,dt + \sqrt{\beta(t)}\,dW$$
$$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min}), \quad t \in [0, 1]$$

**Score network architecture (ScoreNet UNet):**
- Input: noisy z, time embedding t
- Conditioning: cross-attention on z_common tokens for disease LDMs
- Architecture: U-Net with residual blocks, multi-head attention at 8×8 bottleneck

---

### 4.3 Phase 9 — Composition Evaluation

Test composition by jointly sampling z_common (from LDM_common) and z_disease (from LDM_cardio conditioned on z_common):

```
Strategy A — Sequential:
    1. z_common ~ LDM_common()
    2. z_cardio ~ LDM_cardio(z_common)
    3. x = SepVAE_decoder(concat(z_common, z_cardio))

Evaluation:
    - FID against held-out cardiomegaly images
    - Region SSIM in cardiac region
    - Classifier confidence (cardiomegaly probability)
    - Counterfactual coherence (does the same z_common produce anatomically consistent Normal and Cardio variants?)
```

---

### 4.4 Phase 10 — Ablation Baseline

Train a jointly-conditional LDM with a 4-class label (normal, cardio, effusion, both) — the standard approach without disentanglement.

Compare against Strategy A on FID, region SSIM, classifier confidence. If the jointly-conditional baseline wins: the factorisation assumption (A1) is the bottleneck. If Strategy A wins: disentangled composition adds measurable value.

---

### 4.5 Phase 11 — CFG Composition Variant

Train one conditional LDM with CFG dropout (`p_uncond=0.1`). At inference, apply dual-condition CFG guidance:

```
x_pred = x_uncond + w_cardio * (x_pred_cardio - x_uncond)
                  + w_effusion * (x_pred_effusion - x_uncond)
```

Compare against Strategy A. Divergence is diagnostic of how much cross-head information the disease LDMs learn — if CFG wins, the LDMs are learning joint rather than marginal distributions.

---

## 5. Decision Tree

```
D3 final checkpoint
    │
    ├── Evaluate G1–G4
    │       │
    │       ├── All pass → Preencoding (parallel to D4+)
    │       │
    │       └── G1 fails → D4 adds R11 classifier
    │           G2 fails → D4 increases weight_masked_rec
    │           G3 fails → D4 increases weight_bbox_attn
    │
    ├── D4 → D5 → D6 → D7 (incremental refinement)
    │
    └── LDM track (after G1–G4 pass)
            │
            ├── Phase 8: Train LDM_common, LDM_cardio
            ├── Phase 9: Strategy A composition evaluation
            ├── Phase 10: Jointly-conditional baseline
            └── Phase 11: CFG variant
```

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| G1 fails (z_d not discriminative) | Low | High — blocks LDM | R11 disease classifier; re-evaluate after 20 epochs |
| D7 skip connections regress quality | Medium | Medium | D7-A (skip only) before D7-B (z_common=32) |
| LDM_cardio fails to capture cardiac shape | Medium | High | Increase LDM conditioning strength; check z_cardio quality |
| Strategy A underperforms jointly-conditional | Possible | Affects thesis claim | Thoroughly evaluate Phase 10 ablation; document gap |
| D3→D4 GAN instability (disc dominance) | Low | Medium | Monitor disc_acc; kill if > 0.80 for 5 epochs |

---

## 7. Immediate Next Actions

1. **Let D3 complete to epoch 120.** Monitor W&B for kill conditions (disc_acc > 0.80, recon spike).

2. **After D3 final checkpoint:** Run G1–G4 evaluation on the final checkpoint before deciding D4 scope.
   ```bash
   python scripts/eval_counterfactual.py \
     --checkpoint runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_final.pkl \
     --csv_path /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
     --output results/d3_gate_evaluation/
   ```

3. **If G1–G4 pass:** Start preencoding immediately. Begin drafting LDM architecture.

4. **D4 planning:** Based on G1–G4 results and D3 final qualitative assessment, specify the single change for D4. Do not plan D4 before seeing D3 final.

---

## 8. Track 3 — Mask Supervision (feature/mask-supervision)

**Status:** Exploration complete (2026-03-28). Integration pending root cause disambiguation.

**Finding:** Latent traversal at ep120 showed z_disease mask area Δ = ±7 px across the full
α = 0 → 2.0 range — near-collapsed. Three hypotheses for why (H1: noisy bbox signal,
H2: loss weight too low, H3: capacity). Full investigation in [10 Mask Supervision Journey](10_mask_supervision_journey.md).

**CheXmask dataset acquired:**
- `/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv` — 309 MB, 18,000 rows
- 14,029 / 14,029 training images matched (100% hit rate)
- 13,970 usable after Dice RCA ≥ 0.70 filter
- Heart + lung masks at 1024×1024, resize to 512×512 for training

**Next action — H2 test (cost: ~2 hours GPU):**
```bash
# Resume from D3 ep120. Raise weight_bbox_attn: 0.10 → 1.0. Train 20 epochs.
# Then: conda run -n jaxstack python scripts/latent_traversal_medsam.py \
#   --checkpoint <new_ckpt> --output results/traversal_bbox_weight_test/
```
If Δarea > ±50 px → H2 confirmed → keep bbox supervision, raise weight in D4.
If still flat → H1/H3 → integrate CheXmask CTR regression head (see §5.3 of doc 10).

**Branch:** `feature/mask-supervision` | **Tag:** `explore/mask-supervision-v1`

---

*End of document. Return to [Index](INDEX.md).*
