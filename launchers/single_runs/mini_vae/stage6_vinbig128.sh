#!/usr/bin/env bash
# =============================================================================
# Stage 6 — Full stack on real VinBigData @ 128×128
#
# Identical hypers to Stage 5 but switches to real chest X-rays:
#   --use_vinbig         (loads VinBigDataTripletDataset resized to 128px)
#   --img_size 128
#   --n_samples 0        (0 = use full dataset)
#   --epochs 50          (real data needs more epochs)
#   --batch_size 32      (DICOM loading is heavier; reduce if OOM)
#
# Prerequisites:
#   - Stage 5 gates must be satisfied on SyntheticCXR
#   - VinBigData processed CSVs + DICOM root in VINBIG_ROOT env var
#
# Gate G1: probe_auc/cardiomegaly > 0.75  (harder on real data)
# Gate G2: cross_head_score < 0.65
# Gate G4: SSIM > 0.85 on held-out reconstructions
#
# Expected runtime: ~45 min / epoch on a single GPU.
#   Use slurm_scripts/mini_sep_vae.slurm for cluster submission.
# =============================================================================
set -euo pipefail

: "${VINBIG_ROOT:?Please set VINBIG_ROOT to the VinBigData root directory}"

RUN_NAME="mini_stage6_vinbig128_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/../../.."

python run/train_mini_sepvae.py \
  --run_name        "${RUN_NAME}" \
  --dataset         vinbig \
  --vinbig_root     "${VINBIG_ROOT}" \
  --img_size        128 \
  --epochs          50 \
  --batch_size      32 \
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

echo "Stage 6 done — check WandB run: ${RUN_NAME}"
echo "Gates: probe_auc/cardiomegaly > 0.75  &&  cross_head_score < 0.65  &&  SSIM > 0.85"
