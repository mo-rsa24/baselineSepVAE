# Experiment Plan D — From-Scratch SepVAE (No CheSS Encoder)

**Primary research objectives (unchanged):**
- **O1 — Orthogonal latent separation**: z_common ⊥ z_cardio ⊥ z_plthick; each disease head responds only to its disease
- **O2 — Clean and crisp reconstructions**: visually sharp CXR output

**Why a new plan:**
The CheSS backbone was trained discriminatively. Its feature space is incompatible with
N(0,I), causing high KL at initialisation and hazy reconstructions that are slow to recover.
Removing it from the encoder eliminates the root cause. CheSS is reintroduced in D4 as a
**frozen perceptual loss only** — the role it is actually suited for.

**Architecture: `SepVAEV2` (`models/sep_vae_v2.py`)**
- ResNet-50 trained from scratch, GroupNorm throughout (no frozen stats, small-batch stable)
- Self-attention at layer3 bottleneck (16×16 = 256 tokens for 256px — negligible cost)
- Two learnable Layer4 branches diverge after the shared trunk
- Disease head: `DiseaseAttnHeadV2` (D0) → `BboxCrossAttnHead` (D1+)
- Latent: (B, 16, 16, 32) — z_common (16ch) + z_disease (16ch)
- Decoder: 16×16 → 32 → 64 → 128 → 256 (4 bilinear upsamples)
- **Loss functions unchanged** — `sep_vae_losses.py` requires no modifications

**Resolution: 256×256**
- 4× fewer pixels than 512×512; morphological features (cardiac silhouette, pleural border) remain visible
- Passes 128×128's speed advantage without its diagnostic limitation
- Once the architecture is validated at 256×256, D5 re-runs at 512×512

Each phase adds exactly one new component. A phase **must pass its gate** before activating the next.

---

## Summary table

| Phase | New component | Losses active | Gate to proceed |
|---|---|---|---|
| D0 | ResNet-50 from scratch + self-attn | MSE + KL (defaults) | No OOM; KL manageable at step 0; recon grid shows structure by epoch 3 |
| D1 | Bbox cross-attention in disease head | MSE + KL only | Attn maps localise to cardiac region by epoch 5 (without MI pressure) |
| D2 | FactorVAE MI discriminator | D1 losses + MI | MI accuracy → 0.5; z_common/z_disease norms diverge by class |
| D3 | Pleural Thickening + second disease head | D2 losses | Both disease heads respond only to their class; MI (z_ca, z_pt) ≤ 0.55 |
| D4 | CheSS perceptual loss (frozen, loss only) | D3 losses + perceptual | Recon sharper; O1 metrics from D3 stable (< 10% regression) |
| D5 | Scale to 512×512 | D4 losses | Architecture confirmed at full resolution |

---

## D0 — Smoke test (2–3 epochs)

**Model:** `SepVAEV2(use_bbox_cross_attn=False)`
**Purpose:** Confirm no OOM, KL behaves normally (not catastrophically high), reconstruction
grid shows recognisable image structure, W&B connects. This is the key diagnostic: if KL is
manageable here (vs. catastrophically high in V1 at step 0), the CheSS backbone was the culprit.

```bash
cd /workspace/baselineSepVAE
conda activate jaxstack

python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --img_size         256 \
    --model_version    v2 \
    --batch_size       8 \
    --epochs           3 \
    --num_workers      4 \
    --lr_disc          1e-4 \
    --sample_every     1 \
    --manifold_every   -1 \
    --exp_name         d0_smoke_v2 \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate:** No OOM. KL < 50 000 at step 0 (vs. V1 catastrophic blowup). Recon grid shows
recognisable lung/chest structure by epoch 3. W&B run appears.

**Diagnostic split:**
- KL still catastrophically high (> 100k) → backbone architecture issue, not CheSS; check GroupNorm groups
- KL manageable but recon is grey → decoder issue or weight_rec too low
- Both fine → proceed to D1

---

## D1 — Reconstruction baseline + bbox cross-attention (20 epochs)

**Model:** `SepVAEV2(use_bbox_cross_attn=True)`
**New component:** Bbox cross-attention in the disease head. The bbox Gaussian prior
drives the disease head's attention from epoch 1 — no waiting for the learned query
to converge. MSE + KL only; MI and perceptual off.

**Key change from V1-C1:** Bbox supervision is architectural (baked in from the start),
not a late-phase loss. If attn maps are not localising by epoch 10 even with this strong
spatial prior, the architecture needs rethinking — you find this out cheaply.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --img_size         256 \
    --model_version    v2 \
    --use_bbox_cross_attn \
    --batch_size       8 \
    --epochs           20 \
    --num_workers      8 \
    --kl_warmup_epochs 5 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     5e-5 \
    --weight_mi_factor      0.0 \
    --weight_bbox_attn      0.0 \
    --weight_perceptual     0.0 \
    --lr_disc          1e-4 \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         d1_recon_bbox_xattn \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O2 foundation + O1 spatial prior check):**
- KL_total < 5 000 by epoch 10
- Recon grid shows recognisable lung fields and cardiac silhouette
- Attention maps (logged to W&B) show the disease head *already* concentrating on the
  cardiac region for Cardiomegaly images — before any MI pressure (cross-attention should
  make this visible early, unlike the learned-query approach in V1)

**Kill if:** Attn maps uniformly diffuse for all classes at epoch 10 (cross-attention not working).

---

## D2 — FactorVAE MI discriminator (O1 step 1, 25 epochs)

**New component:** FactorVAE MI discriminator. With cross-attention already grounding
the disease head spatially, the discriminator's job is easier — it reinforces latent
independence after spatial separation has already begun.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --img_size         256 \
    --model_version    v2 \
    --use_bbox_cross_attn \
    --batch_size       8 \
    --epochs           25 \
    --num_workers      8 \
    --kl_warmup_epochs 0 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     1e-4 \
    --weight_mi_factor      1.0 \
    --weight_bbox_attn      0.0 \
    --weight_perceptual     0.0 \
    --lr_disc          1e-4 \
    --resume           /workspace/runs_sepvae/d1_recon_bbox_xattn/checkpoint_final.pkl \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         d2_mi_disc \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O1 step 1):**
- MI discriminator accuracy → ~0.5 (chance); confirms z_common ⊥ z_cardio in feature space
- ‖z_cardio‖ larger for Cardiomegaly than Normal
- z_common mean does not shift by disease label

**Kill if:** MI loss diverges or MSE degrades > 2× D1.

---

## D3 — Three-class extension: Pleural Thickening (O1 complete, 30 epochs)

**New component:** Add Pleural Thickening class. Requires code changes:
1. Add `pt_branch = Layer4BranchGN()` and `head_plthick = BboxCrossAttnHead()` to `SepVAEEncoderV2`
2. Add `'plthick'` to `apply_head_nulling_v2` (hard-zero z_pt for Normal/Cardio)
3. Switch dataset: `VinBigDataTripletDataset` (Normal + Cardiomegaly + Pleural Thickening)
4. MI discriminator input: `jnp.concatenate([z_ca_pooled, z_pt_pooled], axis=-1)` (32D)
   — the two disease heads must now be independent of *each other*

**Gate (O1 complete):**
- Latent swap grid: swapping z_cardio changes cardiac silhouette only; swapping z_plthick
  changes pleural surface only; neither affects the other
- MI discriminator accuracy for (z_cardio, z_plthick) pair ≤ 0.55

**Kill if:** Either disease head collapses (near-zero norms across all classes).

---

## D4 — CheSS perceptual loss (O2 complete, 15 epochs)

**New component:** Reintroduce CheSS — but as a **frozen perceptual loss only**.
The backbone is not in the encoder. It extracts L1 feature distances at layers 2, 3, 4
to sharpen reconstruction texture without adding parameters or touching the encoder KL.

```bash
python run/train_sep_vae.py \
    ...                                          # same data/model args as D3 \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --weight_perceptual     0.05 \
    --resume           /workspace/runs_sepvae/d3_three_class/checkpoint_final.pkl \
    --epochs           15 \
    --exp_name         d4_perceptual \
    ...
```

**Gate (O2 complete):**
- Recon grid visually sharper than D3 (finer rib/vessel/pleural border detail)
- KL, MI, and attn metrics do not regress > 10% from D3

**Kill if:** KL_common spikes > 2× D3 (perceptual loss fighting the prior); reduce weight_perceptual.

---

## D5 — Scale to 512×512

Retrain D1–D4 at 512×512 with the validated architecture and hyperparameters from 256×256.
This is confirmation, not exploration. The 256×256 phase ladder gives you a calibrated
starting point for loss weights, KL warmup, and attention behaviour.

---

## Training script changes required

The existing `run/train_sep_vae.py` imports `SepVAE` from `models/sep_vae_jax.py`.
To use V2, add:

```python
# In train_sep_vae.py
if args.model_version == 'v2':
    from models.sep_vae_v2 import SepVAEV2
    model = SepVAEV2(
        use_bbox_cross_attn=args.use_bbox_cross_attn,
        z_channels_common=16,
        z_channels_disease=16,
    )
else:
    from models.sep_vae_jax import SepVAE
    model = SepVAE(...)
```

New CLI args to add:
```
--model_version    {v1, v2}    (default: v1 for backward compat)
--use_bbox_cross_attn          (flag, activates BboxCrossAttnHead)
--img_size         INT         (256 for D0–D4, 512 for D5)
```

For D1+, the dataset collate function must pass `bbox` and `has_bbox` tensors.
`has_bbox[i] = 1.0` if sample i is a disease image with a valid bbox, else `0.0`.

---

## Notes

### Note 1 — GroupNorm group count
All ResNet-50 channel sizes (64, 128, 256, 512 → expanded: 256, 512, 1024, 2048) are
divisible by 32. `GroupNorm(num_groups=32)` is valid throughout. No special casing needed.

### Note 2 — Self-attention token count by resolution
| Input resolution | Layer3 spatial | Tokens | Attention ops |
|---|---|---|---|
| 128×128 | 8×8   |   64 | 4 096    |
| 256×256 | 16×16 |  256 | 65 536   |
| 512×512 | 32×32 | 1024 | 1 048 576 |

At 512×512, bottleneck self-attention over 1024 tokens is still manageable on an A100
(~1M ops vs billions in the conv layers). Gradient checkpointing keeps memory flat.

### Note 3 — Latent resolution change
V1: 64×64×32 latent (from CheSS layer4 upsampled to 64×64)
V2: 16×16×32 latent (from layer4 branches upsampled to 16×16)

The decoder is adjusted accordingly (5 ch_mults levels, 4 upsamples for 256px).
Checkpoint files from V1 are not compatible with V2 (different latent size + no CheSS params).

### Note 4 — KL diagnostic at D0 step 0
Expected range with from-scratch ResNet-50:
- Healthy: KL_total 1 000–20 000 at step 0 (random init, no prior mismatch)
- Warning: > 50 000 → check GroupNorm is not misconfigured (num_groups must divide channels)
- Kill: > 200 000 → something is wrong with the encoder output scale
