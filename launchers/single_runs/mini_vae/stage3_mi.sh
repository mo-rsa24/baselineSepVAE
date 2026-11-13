#!/usr/bin/env bash
# =============================================================================
# Stage 3 — Mutual information minimisation between common and disease heads
#
# Adds on top of Stage 2:
#   L_MI  (CLUB-style MI upper bound between z_common and z_disease)
#   MI discriminator trained jointly (--lr_disc controls its LR)
#
# Purpose: prevent z_common from absorbing disease information that should
#          live in the disease heads.  R4 finding: without this, common head
#          acts as a leak sink.
#
# Gate G2: cross_head_score < 0.65  (tighter after adding MI)
# Expected runtime: ~8 min on a single GPU with SyntheticCXR.
# =============================================================================
set -euo pipefail

RUN_NAME="mini_stage3_mi_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         synthetic \
  --n_samples       2000 \
  --img_size        128 \
  --epochs          25 \
  --batch_size      64 \
  --lr_vae          3e-4 \
  --lr_disc         1e-4 \
  --weight_kl_common   1e-4 \
  --weight_kl_disease  1e-4 \
  --weight_null        0.02 \
  --sigma_inactive     0.05 \
  --weight_orthogonality 0.02 \
  --use_disease_clf \
  --weight_disease_clf 0.1 \
  --weight_mi          0.005 \
  --weight_contrastive 0.0 \
  --weight_spatial_attn 0.0 \
  --seed            42

echo "Stage 3 done — check WandB run: ${RUN_NAME}"
echo "Gates: probe_auc/cardiomegaly > 0.75  &&  cross_head_score < 0.65"
