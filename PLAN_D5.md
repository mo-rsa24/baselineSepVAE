# D5 Plan — Reconstruction Quality at 256×256

**Goal:** Completely remove artifacts, ensure crisp reconstructions, maintain clear latent separation between Normal and Cardiomegaly. Stay at 256×256.

**Resumes from:** `/workspace/runs_sepvae/d4_perceptual-20260317-043603/checkpoints/checkpoint_epoch0160.pkl`

---

## Issues and Fixes

---

### Issue A — Layer4 branches stride-then-upsample aliasing

**Verbatim:** The bg_branch and tg_branch go 16×16 → 8×8 (stride-2) → 16×16 (bilinear upsample). This create-then-destroy pattern introduces aliasing: stride-2 creates aliasing in the feature space (frequencies above Nyquist are aliased), bilinear upsample tries to recover but introduces smooth interpolation artifacts. The features that drive the latent space are "corrupted" by this stride-upsample round-trip.

**Fix A:** Change `Layer4BranchGN` to use stride=1 throughout (3 blocks, all stride=1). Output stays (16×16×2048), no upsample needed in the encoder. The main cost: this halves the compute in the branch (fewer MFLOPs at the 3×3 conv step), which is actually a win.

```
Before: 16×16 → stride-2 → 8×8 → bilinear upsample → 16×16
After:  16×16 → stride-1 → 16×16 (stays)
```

**File:** `models/sep_vae_v2.py` — `Layer4BranchGN` and `SepVAEEncoderV2`

---

### Issue B — V projection in BboxCrossAttnHead is wasted

**Verbatim:** `V` is projected but then never used. The cross-attention computation uses Q×K^T → softmax → attn_map, and then gates the original features `h` (not the value projections V) by the attention map. The V projection is completely wasted — `val_proj` has `query_dim × C` parameters that contribute zero gradient.

**Fix (Issue B):** Remove the `val_proj` Dense layer entirely from `BboxCrossAttnHead`. The design intent is spatial masking, not cross-attention in the classical sense — the fix makes the code honest about that.

**File:** `models/sep_vae_v2.py` — `BboxCrossAttnHead`

---

### Issue C — Bbox supervision disabled the whole time

**Verbatim:** `weight_bbox_attn = 0.0` in D2, D3, D4. The `BboxCrossAttnHead` computes a Gaussian-prior-driven query from epoch 1, but there is zero loss pressure to keep the attention inside the bbox. The attention learns to attend wherever benefits reconstruction, which need not be the cardiac region. This is why the attention maps aren't localizing. The bbox cross-attention provides the query initialisation from the prior, but the attention map is still free to drift. The `bbox_attention_loss` is the penalty that forces it to stay.

**Fix (Issue C):** Set `weight_bbox_attn = 0.1` in the D5 launcher. Small but non-zero; activates the full pipeline that was wired but unused.

**File:** `launchers/runpod/run_d5_recon.sh`

---

### Issue D — Too few channels at fine resolution

**Verbatim:** Channel schedule (fine-to-coarse): **(64, 128, 256, 512, 512)**. At 256×256, only **64 channels**. At 128×128, only **128 channels**. These are both too thin for generating crisp high-frequency detail. LDM/VQGAN decoders use at minimum 128 channels at full output resolution. With 64 channels at 256×256 and only 2 ResBlocks, there are effectively **4 conv operations** (2 ResBlocks × 2 convs each) to generate the final 1-channel output. This is a hard information bottleneck that directly limits reconstruction crispness.

**Fix D:** Change to **(128, 128, 256, 512, 512)**. At 256×256 we now have 128 channels (doubling capacity), and 128×128 stays at 128. Memory cost is manageable — the batch size is 16 and 128×128×128 activations at float32 = 8MB per image in the batch.

**File:** `models/sep_vae_v2.py` — `SepVAEDecoderV2` default `ch_mults` and `SepVAEV2.setup()`

---

### Issue E — No self-attention in decoder

**Verbatim:** The encoder has self-attention at the 16×16 bottleneck. The decoder has SE gates (channel attention) but **no spatial self-attention anywhere**. Without global attention in the decoder: the cardiac silhouette is reconstructed locally by each spatial position without coordination; long-range spatial coherence (e.g., the left/right cardiac border being consistent) is enforced only weakly via the convolutional receptive field. The 32×32 level is the right place to add self-attention: 1024 tokens, O(1M) attention ops, manageable.

**Fix E:** Add `SelfAttention2D` at the 32×32 decoder level (after the ResBlockSE processing for i=3, before the SmoothUp to 64×64).

**File:** `models/sep_vae_v2.py` — `SepVAEDecoderV2.__call__`

---

### Issue F — SE reduction ratio too aggressive at fine scales

**Verbatim:** `se_reduction=16`: 64 channels → SE hidden dim = 4 (barely any capacity for excitation); 128 channels → SE hidden dim = 8. At 64 channels, 4-dim excitation is minimal. SE with only 4 hidden units can only learn 4 independent channel combinations, which is inadequate for a 64-channel activation map.

**Fix F:** Use `se_reduction=8` so 64-channel levels get hidden=8 and 128-channel levels get hidden=16.

**File:** `models/sep_vae_v2.py` — `SepVAEDecoderV2` default `se_reduction` and `SepVAEV2.setup()`

---

### Issue G — MSE + perceptual imbalance

**Verbatim:** In D4's loss at epoch 160: `weight_rec=1.0`, `l_rec≈0.005` → contribution: **0.005**; `weight_perceptual=0.05`, estimated `l_perceptual≈0.55` → contribution: **0.028**; Weighted KL≈175×(1e-4+1e-4) = **0.035**. MSE contributes only 0.005 to the total ~0.045 loss — it's the **smallest signal**. This means the encoder/decoder are primarily optimized for KL regularization and perceptual feature matching, not pixel-level reconstruction accuracy. MSE being too small relative to perceptual can allow the decoder to sacrifice pixel accuracy for perceptual similarity, which is fine for sharpness but bad for structural faithfulness.

**Fix G:** Small increase in `weight_rec` to 2.0 to rebalance. Do not increase more — MSE still tends to blur.

**File:** `launchers/runpod/run_d5_recon.sh`

---

### Issue H — PatchGAN generator loss: needs wiring, discriminator update step absent

**Verbatim:** The `NLayerDiscriminator` output is `(N, H', W', 1)` — a patch map. `-jnp.mean()` averages over batch AND spatial dims, which is correct for hinge generator loss. However, the PatchGAN discriminator update step does not yet exist in the training script.

**Fix (Issue H):** Wire `NLayerDiscriminator` fully: initialize it, create its optimizer and `TrainState`, add a `patch_disc_step` JIT function, pass `patch_disc_params` into `sepvae_loss` via `vae_step`, and add the alternating update to the train loop. Expose `--weight_gan`, `--gan_start_step`, and `--lr_patch_disc` as CLI args.

**File:** `run/train_sep_vae.py`

---

### Issue I — `vae_step` unpacks 4 values but `sepvae_loss` returns 5

**Verbatim:** `vae_step`'s inner `loss_fn` does `total_loss, logs, z_c, z_ca = sepvae_loss(...)` but `sepvae_loss` returns five values: `(total_loss, logs, z_c_pooled, z_ca_pooled, x_rec)`. This was added to the loss function as D5 prep but the train script wasn't updated. This will raise `ValueError: too many values to unpack` at runtime. `x_rec` needs to be captured and passed to the PatchGAN discriminator step.

**Fix (Issue I):** Update `vae_step`'s `loss_fn` to unpack all 5 return values and include `x_rec` in the auxiliary output. Update the caller to receive `x_rec` from `vae_step`.

**File:** `run/train_sep_vae.py` — `vae_step`

---

### Issue J — PatchGAN discriminator update needs stale `x_rec`

**Verbatim:** Following the same stale-latent pattern used for the FactorVAE discriminator, the PatchGAN needs stale reconstructions: (1) `patch_disc_step(patch_disc_state, x_real, x_rec_stale)` — update discriminator; (2) `vae_step(...)` with frozen patch disc params — returns fresh `x_rec_stale` for next step. This avoids back-propagation through both the VAE and the discriminator simultaneously.

**Fix (Issue J):** Initialise `x_rec_stale` to zeros before the train loop (mirroring `z_c_stale`/`z_ca_stale`). In each iteration: first run `patch_disc_step` with `x_rec_stale`, then run `vae_step` which returns fresh `x_rec` that becomes `x_rec_stale` for the next iteration.

**File:** `run/train_sep_vae.py` — train loop

---

## D5 Plan — What to Change, What to Keep

### What to keep (unchanged)
- `SepVAELossConfig` loss function implementations — all correct
- `FactorDiscriminator` and `factor_disc_loss` — working (D_acc=0.50 confirmed)
- `BboxCrossAttnHead` Gaussian prior construction (σ = bbox_width/4) — correct
- `bbox_attention_loss` implementation — correct
- `SmoothUp` (bilinear + 2×conv3×3) — correct, no checkerboard
- `SelfAttention2D` in encoder at layer3 bottleneck — correct placement
- CBAM in every `BottleneckBlockGN` — correct
- Hard-zero nulling of z_disease for Normal images — correct
- `sigma_inactive=0.1` tight prior for inactive disease head — working
- `weight_kl_common=1e-4`, `weight_kl_disease=1e-4` — stable, do not increase
- `weight_mi_factor=1.0` — working
- `weight_perceptual=0.05` — keep
- EMA decay — keep
- GroupNorm throughout — correct

### Architecture changes (`models/sep_vae_v2.py`)

| Change | Location | Before | After |
|---|---|---|---|
| Fix A | `Layer4BranchGN.block0` | `stride=2, use_projection=True` | `stride=1, use_projection=False` |
| Fix A | `SepVAEEncoderV2.__call__` | `jax.image.resize(h_bg/h_tg, ..., 16×16)` | remove resize calls (already 16×16) |
| Fix B | `BboxCrossAttnHead` | `V = nn.Dense(query_dim, name='val_proj')(h_flat)` | remove entirely |
| Fix D | `SepVAEDecoderV2` | `ch_mults=(64,128,256,512,512)` | `ch_mults=(128,128,256,512,512)` |
| Fix D | `SepVAEV2.setup()` | `ch_mults=(64,128,256,512,512)` | `ch_mults=(128,128,256,512,512)` |
| Fix E | `SepVAEDecoderV2.__call__` | no attention in decoder | add `SelfAttention2D` after i=3 ResBlockSE, before SmoothUp |
| Fix F | `SepVAEDecoderV2` | `se_reduction=16` | `se_reduction=8` |
| Fix F | `SepVAEV2.setup()` | (default) | pass `se_reduction=8` |

### Training script changes (`run/train_sep_vae.py`)

| Change | Location | Description |
|---|---|---|
| Fix I | `vae_step` → `loss_fn` | Unpack 5 return values; include `x_rec` in auxiliary output |
| Fix I | `vae_step` return | Return `x_rec` alongside `logs, z_c, z_ca` |
| Fix H | `parse_args()` | Add `--weight_gan` (float, default 0.0) |
| Fix H | `parse_args()` | Add `--weight_tv` (float, default 0.0) |
| Fix H | `parse_args()` | Add `--gan_start_step` (int, default 5000) |
| Fix H | `parse_args()` | Add `--lr_patch_disc` (float, default 1e-4) |
| Fix H | after FactorDisc init | Initialize `NLayerDiscriminator`, `patch_disc_state` with `tx_patch_disc` optimizer |
| Fix H | `SepVAELossConfig` construction | Pass `weight_gan=args.weight_gan`, `weight_tv=args.weight_tv` |
| Fix H | `vae_step` | Accept `patch_disc_params` and `patch_discriminator`; pass to `sepvae_loss` |
| Fix H | JIT functions | Add `patch_disc_step` JIT (hinge_d_loss on real vs stale_rec, frozen by stop_gradient) |
| Fix J | before train loop | Initialize `x_rec_stale = jnp.zeros((2B, H, W, 1))` |
| Fix J | train loop | Run `patch_disc_step(patch_disc_state, x_real, x_rec_stale)` before `vae_step` |
| Fix J | train loop | Capture `x_rec_stale` from `vae_step` return each iteration |
| Fix H | checkpoint save | Add `patch_disc_params`, `patch_disc_opt_state` to `ckpt_data` |
| Fix H | checkpoint restore | Restore `patch_disc_state` from checkpoint if key present |
| Fix H | logging | Add `loss/gan_g`, `loss/tv`, `metrics/patch_disc_acc` to step and epoch summary logs |

### New launcher (`launchers/runpod/run_d5_recon.sh`)

Key hyperparameters (all others inherited from D4 launcher defaults):

| Parameter | Value | Rationale |
|---|---|---|
| `WEIGHT_REC` | 2.0 | Fix G: rebalance MSE vs perceptual |
| `WEIGHT_PERCEPTUAL` | 0.05 | keep |
| `WEIGHT_GAN` | 0.5 | Fix H: PatchGAN hinge generator |
| `WEIGHT_TV` | 1e-3 | suppresses stripe artifacts |
| `WEIGHT_BBOX_ATTN` | 0.1 | Fix C: re-enable bbox supervision |
| `GAN_START_STEP` | 5000 | Fix H: VAE stabilises before GAN activates |
| `LR_PATCH_DISC` | 1e-4 | Fix H |
| `EPOCHS` | 60 | top-up from D4 checkpoint |
| `RESUME` | d4_perceptual-20260317-043603/checkpoints/checkpoint_epoch0160.pkl | |

---

## Monitoring in D5

| Metric | Expected | Kill condition |
|---|---|---|
| `loss/reconstruction` | down → ≤ 0.003 | up > 0.010 |
| `loss/gan_g` | down (more negative) | strongly negative + recon rising |
| `loss/tv` | down over epochs | stays high → stripes persist |
| `loss/bbox_attn` | down → ≤ 0.2 | up → attention ignoring bbox |
| `silhouette_disease_only_pca` | stable ≥ 0.50 | drops below 0.40 |
| `metrics/patch_disc_acc` | 0.55–0.65 | > 0.80 (VAE losing) |
| Recon grid (visual) | sharp ribs/vessels, no stripes | new artifact types |

---

## TODOs (ordered by dependency)

### Phase 1 — Architecture (`models/sep_vae_v2.py`)
These are independent of each other and can be done in any order within Phase 1.

- [ ] **A1** Fix A: Change `Layer4BranchGN.block0` from `stride=2, use_projection=True` to `stride=1, use_projection=False`
- [ ] **A2** Fix A: Remove `jax.image.resize` upsample calls for `h_bg` and `h_tg` in `SepVAEEncoderV2.__call__` (branches now stay at 16×16)
- [ ] **A3** Fix B: Remove unused `V = nn.Dense(query_dim, name='val_proj')(h_flat)` from `BboxCrossAttnHead`
- [ ] **A4** Fix D: Change `ch_mults=(64,128,256,512,512)` → `(128,128,256,512,512)` in `SepVAEDecoderV2` default and in `SepVAEV2.setup()`
- [ ] **A5** Fix E: Add `SelfAttention2D` at 32×32 level in `SepVAEDecoderV2.__call__` (after i=3 ResBlockSE blocks, before SmoothUp)
- [ ] **A6** Fix F: Change `se_reduction=16` → `se_reduction=8` in `SepVAEDecoderV2` and pass it explicitly in `SepVAEV2.setup()`

### Phase 2 — Training script (`run/train_sep_vae.py`)
Must be done in order due to dependencies.

- [ ] **T1** Fix I: Update `vae_step` → `loss_fn` to unpack all 5 return values from `sepvae_loss`; return `x_rec` from `vae_step`
- [ ] **T2** Fix H: Add CLI args `--weight_gan`, `--weight_tv`, `--gan_start_step`, `--lr_patch_disc` to `parse_args()`
- [ ] **T3** Fix H: Initialize `NLayerDiscriminator`, create `tx_patch_disc` optimizer, create `patch_disc_state` (after existing FactorDisc init block)
- [ ] **T4** Fix J: Initialize `x_rec_stale = jnp.zeros((2*args.batch_size, args.img_size, args.img_size, 1))` before the train loop
- [ ] **T5** Fix H: Add `patch_disc_step` JIT function (hinge_d_loss real vs stale, stop_gradient on stale rec)
- [ ] **T6** Fix H: Update `vae_step` to accept `patch_disc_params` and `patch_discriminator`; pass to `sepvae_loss`
- [ ] **T7** Fix H: Update `SepVAELossConfig` construction to pass `weight_gan=args.weight_gan`, `weight_tv=args.weight_tv`
- [ ] **T8** Fix J: Update train loop to run `patch_disc_step` before `vae_step`; capture `x_rec_stale` from `vae_step`
- [ ] **T9** Fix H: Update checkpoint save/restore to include `patch_disc_params` / `patch_disc_opt_state`
- [ ] **T10** Fix H: Update step and epoch summary console logging to show `loss/gan_g`, `loss/tv`, `metrics/patch_disc_acc`

### Phase 3 — Launcher
Depends on Phase 1 and Phase 2 being complete.

- [x] **L1** Fix C + Fix G: Create `launchers/runpod/run_d5_recon.sh` with hyperparameters above (weight_rec=2.0, weight_bbox_attn=0.1, weight_gan=0.5, weight_tv=1e-3, gan_start_step=5000)
