#!/usr/bin/env bash
# =============================================================================
# Stage 5 — Spatial attention + contrastive (R7 + R12)
#
# Adds on top of Stage 4:
#   DiseaseAttentionHead (R7): learnable query that spatially attends over
#       the feature map — each disease head learns WHERE to look
#   L_spatial_attn (R12): supervision from ground-truth bboxes; attention
#       map must concentrate inside the disease bounding box
#   L_contrastive: blind-pull loss — pull z_disease closer to disease samples
#       and push away from normal/other-disease samples in latent space
#
# This is the FULL recommended architecture.  If G1 + G2 are both met here,
# Stage 6 can move to real VinBigData images.
#
# Gate G1: probe_auc/cardiomegaly > 0.80  (tighter now that we have attn)
# Gate G2: cross_head_score < 0.60
# Gate G3: edit_purity > 2.0  (if computed)
#
# Expected runtime: ~12 min on a single GPU with SyntheticCXR.
# =============================================================================
set -euo pipefail

RUN_NAME="mini_stage5_attn_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         synthetic \
  --n_samples       2000 \
  --img_size        128 \
  --epochs          35 \
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
  --use_label_attention \
  --attn_query_dim     64 \
  --use_contrastive \
  --weight_contrastive 0.1 \
  --weight_spatial_attn 0.05 \
  --seed            42

echo "Stage 5 done — check WandB run: ${RUN_NAME}"
echo "Gates: probe_auc/cardiomegaly > 0.80  &&  cross_head_score < 0.60  &&  edit_purity > 2.0"
