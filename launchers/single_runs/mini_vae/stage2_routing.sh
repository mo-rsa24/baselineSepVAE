#!/usr/bin/env bash
# =============================================================================
# Stage 2 — Routing + disease classification pressure
#
# Adds on top of Stage 1:
#   L_null  (σ_inactive=0.05 — tighter nulling)
#   L_ortho (between-disease orthogonality)
#   L_disease_clf (positive pressure: each head must be discriminative for
#                  its own disease label via a linear classifier)
#
# Gate G1: probe_auc/cardiomegaly > 0.75
# Gate G2: cross_head_score < 0.65
#
# Expected runtime: ~6 min on a single GPU with SyntheticCXR.
# =============================================================================
set -euo pipefail

RUN_NAME="mini_stage2_routing_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         synthetic \
  --n_samples       2000 \
  --img_size        128 \
  --epochs          25 \
  --batch_size      64 \
  --lr_vae          3e-4 \
  --weight_kl_common   1e-4 \
  --weight_kl_disease  1e-4 \
  --weight_null        0.02 \
  --sigma_inactive     0.05 \
  --weight_orthogonality 0.02 \
  --use_disease_clf \
  --weight_disease_clf 0.1 \
  --weight_mi          0.0 \
  --weight_contrastive 0.0 \
  --weight_spatial_attn 0.0 \
  --seed            42

echo "Stage 2 done — check WandB run: ${RUN_NAME}"
echo "Gates: probe_auc/cardiomegaly > 0.75  &&  cross_head_score < 0.65"
