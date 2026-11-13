#!/usr/bin/env bash
# =============================================================================
# Stage 4 — Cross-adversarial head independence (R5b)
#
# Adds on top of Stage 3:
#   L_cross_adv  (adversarial discriminator trained to classify which disease
#                 head a latent came from; encoder trained to fool it)
#
# Purpose: while L_ortho + L_MI address linear/statistical dependence,
#          L_cross_adv catches nonlinear between-disease leakage.  R5b showed
#          cross_head_score continued to drop after MI was saturated.
#
# Gate G2: cross_head_score < 0.60  (tighter target now)
# Expected runtime: ~9 min on a single GPU with SyntheticCXR.
# =============================================================================
set -euo pipefail

RUN_NAME="mini_stage4_cross_adv_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         synthetic \
  --n_samples       2000 \
  --img_size        128 \
  --epochs          30 \
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
  --use_cross_adv \
  --weight_cross_adv   0.05 \
  --cross_disc_hidden_dim 128 \
  --weight_contrastive 0.0 \
  --weight_spatial_attn 0.0 \
  --seed            42

echo "Stage 4 done — check WandB run: ${RUN_NAME}"
echo "Gates: probe_auc/cardiomegaly > 0.75  &&  cross_head_score < 0.60"
