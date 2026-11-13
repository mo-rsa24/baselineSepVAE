#!/usr/bin/env bash
# =============================================================================
# Stage 1 — Reconstruction baseline
#
# Goal: Verify that MiniSepVAE can reconstruct all three image types at 128×128
#       using only L_rec + L_KL (very small KL weight so the model does not
#       collapse to prior).  No routing, no disentanglement pressure.
#
# Pass/fail criterion: visual inspection — both disease textures should be
#       visible in x_rec_disease* reconstructions by epoch 10.
#
# Expected runtime: ~5 min on a single GPU with SyntheticCXR.
# =============================================================================
set -euo pipefail

RUN_NAME="mini_stage1_recon_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."   # repo root = miniSepVAE/

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         synthetic \
  --n_samples       2000 \
  --img_size        128 \
  --epochs          20 \
  --batch_size      64 \
  --lr_vae          3e-4 \
  --weight_kl_common   1e-4 \
  --weight_kl_disease  1e-4 \
  --weight_null        0.0 \
  --weight_orthogonality 0.0 \
  --weight_mi          0.0 \
  --weight_contrastive 0.0 \
  --weight_spatial_attn 0.0 \
  --sigma_inactive     0.1 \
  --seed            42

echo "Stage 1 done — check WandB run: ${RUN_NAME}"
