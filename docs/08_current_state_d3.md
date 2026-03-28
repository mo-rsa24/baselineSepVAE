# Current State — D3 Milestone

**Related documents:** [07 Results & Evaluation](07_results_evaluation.md) | [09 Forward Plan](09_forward_plan.md) | [03 Training Curriculum](03_training_curriculum.md) | [Index](INDEX.md)

**Last updated:** 2026-03-25
**Status:** Active training run. D3 is in progress at epoch 55→120.

---

## 1. Winning Run Identity

| Property | Value |
|----------|-------|
| Experiment name | `d3_gan_fix` |
| Run directory | `runs_sepvae/d3_gan_fix-20260325-143813/` |
| W&B | https://wandb.ai/prime_lab/baseline-sepvae/runs/vamoroxk |
| Git state | `97f678113b2770417db1c126b839905103f1766c` |
| Branch | `snapshot/d3-gan-fix` (immutable) / `main` (current) |
| Started | 2026-03-25 14:38:13 |
| Target epochs | 55 → 120 (+65 epochs) |
| Checkpoints | `checkpoints/checkpoint_epoch00XX.pkl`, every 5 epochs |

**To restore this exact state:**
```bash
git checkout 97f678113b2770417db1c126b839905103f1766c
# or
git checkout snapshot/d3-gan-fix
```

---

## 2. What This Model Can Do Now

**Reconstruction quality:**
- Crisp rib cortex edges visible in reconstructions
- Sharp cardiac border (right and left cardiac margins defined by a bright-dark edge pair)
- Pulmonary vessels visible in hilar region as branching linear structures
- Lung parenchyma texture (faint lung markings) present
- No horizontal stripe banding (CheSS layer3 excluded, TV weight raised)
- No checkerboard artifacts (SmoothUp throughout)
- No mode collapse

**Latent disentanglement (qualitative):**
- Hard-zero nulling: Normal image reconstructions do not show enlarged cardiac silhouettes
- z_cardio_norm_ratio > 2.0: Cardiomegaly images produce z_d with larger norm than Normal images
- loss/bbox_attn ≈ 0.20–0.25: Disease attention map concentrated in cardiac region
- loss/masked_rec ≈ 0.02: z_common reconstructs non-cardiac region without z_d information
- FactorDisc accuracy ≈ 0.52–0.58: Near-ideal MI (z_common and z_disease near-independent)

---

## 3. Complete Hyperparameter State at D3

This is the complete, authoritative configuration. Copy verbatim when designing downstream phases.

### 3.1 Model architecture

| Parameter | Value |
|-----------|-------|
| model_version | v2 |
| img_size | 256 |
| z_channels_common | 16 |
| z_channels_disease | 16 |
| attn_query_dim | 256 |
| attn_heads | 4 |
| decoder_res_blocks | 3 |
| use_bbox_cross_attn | True |
| bbox_query_mix | 1.0 (pure Gaussian prior) |
| bbox_dropout_prob | 0.3 |
| perceptual_only | True (CheSS layers 1–2) |

### 3.2 Loss weights

| Parameter | Value | Notes |
|-----------|-------|-------|
| weight_rec | 1.0 | MSE |
| weight_kl_common | 1e-4 | |
| weight_kl_disease | **5e-5** | Do not raise to 1e-4 |
| kl_free_bits | 0.5 | Per-dim KL floor |
| weight_mi_factor | 1.0 | FactorVAE |
| weight_bbox_attn | 0.10 | |
| weight_cardio_supcon | 0.05 | |
| weight_perceptual | 0.05 | Layers 1–2 only |
| weight_gan | **0.1** | Do not raise above 0.2 |
| weight_tv | **0.005** | Do not lower |
| weight_masked_rec | 0.3 | |
| sigma_inactive | 0.1 | N(0, 0.01·I) tight prior for Normal |

### 3.3 GAN configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| gan_start_step | 2000 | **Phase-local** — critical |
| lr_patch_disc | 1e-4 | |
| disc_r1_penalty | 0.0 | |

### 3.4 Training

| Parameter | Value |
|-----------|-------|
| batch_size | 6 |
| num_workers | 8 |
| eval_num_workers | 0 |
| seed | 0 |
| lr_vae | 1e-4 |
| lr_disc (FactorVAE) | 1e-4 |
| weight_decay | 1e-4 |
| grad_clip | 1.0 |

### 3.5 Monitoring / logging

| Parameter | Value |
|-----------|-------|
| save_every | 5 |
| sample_every | 5 |
| manifold_every | 5 |
| eval_subset_size | 1024 |
| manifold_bbox_mode | both |
| manifold_max_samples | 1024 |

---

## 4. Checkpoint Chain

All checkpoints are on the cluster at `runs_sepvae/`.

```
d0_smoke_v2-20260324-063142/checkpoint_final.pkl          (epoch 5,  smoke only)
    — not used as a resume point —

d1_recon_bbox_xattn-20260321-004241/checkpoint_final.pkl  (epoch 30)
    └── resumes ──▶
d2_perceptual_bbox-20260324-105108/checkpoint_final.pkl   (epoch 55)
    └── resumes ──▶
d3_gan_fix-20260325-143813/checkpoints/
    checkpoint_epoch0060.pkl   ← first post-GAN-activation checkpoint
    checkpoint_epoch0065.pkl
    checkpoint_epoch0070.pkl
    ... (every 5 epochs)
    checkpoint_final.pkl       ← written at epoch 120 completion
```

**D1 is the canonical clean starting point.** If any future stage needs to restart from scratch (e.g., D7's z_common=32 requires re-initialising the common head), resume from D1 final — not D2 — to avoid carrying any D2 artifact bias from the perceptual-layer3 banding.

**D2 final is the correct resume for all D3→D6 stages.** Do not resume from the stale runs in `runs_sepvae/d4_mi_percep-*` or `runs_sepvae/d5_recon-*` — those checkpoints have stripe artifacts baked into the decoder weights.

---

## 5. What Was Fixed to Get Here

Three bugs in the failed D5 GAN runs were diagnosed and corrected before D3 was stable:

| Bug | Failed run | Symptom | Fix |
|-----|-----------|---------|-----|
| weight_gan=0.5 | d5_gan-20260323-042442 | Collapsed epoch 4 (4.6× GAN:rec) | weight_gan=0.1 |
| global_step gan_start | d5_gan-20260323-042442 | Discriminator fired from step 1 | Phase-local step counter |
| disc_r1_penalty=10.0 | d5_gan_v2-20260323-085311 | Stalled 80+ epochs (discriminator trapped) | disc_r1_penalty=0.0 |

The D3 launcher (`slurm_scripts/d3_gan_fix.slurm`) documents all three fixes inline. The training script (`run/train_sep_vae.py`) implements the phase-local step counter fix.

---

## 6. Active Monitoring Thresholds

Reference these when watching the W&B run.

| Metric | Healthy range | Warning | Kill condition |
|--------|-------------|---------|---------------|
| `loss/reconstruction` | 0.004–0.008 | > 0.008 for 5 epochs | > 0.010 and rising |
| `loss/gan_g` | Slowly more negative | Flat for 10+ epochs | Strongly negative while recon is rising |
| `loss/tv` | Decreasing trend | Flat | Rising (stripes re-emerging) |
| `loss/bbox_attn` | ≤ 0.25 | > 0.30 | Rising > 0.40 |
| `metrics/patch_disc_acc` | 0.55–0.65 | > 0.75 | > 0.80 sustained for 5 epochs |
| `metrics/z_cardio_norm_ratio` | > 2.0 | 1.5–2.0 | < 1.5 |
| `metrics/z_common_norm_ratio` | 0.7–1.5 | > 1.5 | > 2.0 (z_common absorbing disease) |

---

## 7. What Is Not Yet Measured (V2 Gaps)

The following metrics from V1 are not yet implemented in the V2 training loop:

| Missing metric | Priority | How to add |
|---------------|----------|------------|
| Frozen linear probe AUC on z_d | **High** | Eval step in train loop; 10-epoch linear probe on held-out set |
| Cross-head score | Medium | Probe z_c for cardiomegaly label; probe z_d for normal label |
| Counterfactual swap grid | Medium | Swap z_d between Normal/Cardio pairs; inspect visual output |
| FID / LPIPS | Low | Requires batched generation at scale |

The absence of probe AUC is the largest gap. Before LDM preencoding (Phase 8), a one-off probe AUC evaluation on the D3 final checkpoint should be run to confirm G1 (disease discriminability > 0.75) and G2 (z_common purity < 0.65).

---

## 8. Project Position

D3 is the end of the reconstruction quality track — all three sharpness objectives (perceptual, PatchGAN, TV) are now active and stable. The model produces high-quality CXR reconstructions with meaningful latent separation.

The curriculum positions D4–D7 as refinement and scaling phases:
- D4–D6: hyperparameter and objective tuning on the D3 foundation
- D7: architectural change (UNet skip connections + wider z_common)

For the LDM track: SepVAE at D3 is not yet ready for preencoding. The verification gates G1–G4 (see [Results document §5](07_results_evaluation.md#5-verification-gates)) need formal evaluation first. If G1–G4 pass at D3 final, preencoding can proceed in parallel with D4 training.

---

*End of document. Continue to [09 Forward Plan](09_forward_plan.md).*
