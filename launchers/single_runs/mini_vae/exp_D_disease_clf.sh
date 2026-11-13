#!/usr/bin/env bash
# =============================================================================
# Experiment D — stronger disease discriminability classifier (R11, weight=0.20)
#
# In the biggpu run, DiseaseClf BCE stalled at 0.63 after 85 epochs
# (random baseline = 0.69, target < 0.40). The R11 gradient at weight=0.1 is
# insufficient to prevent z_common from absorbing disease signal. Doubling to
# 0.2 increases the positive pressure on the disease heads to encode class-
# discriminative information rather than letting z_common free-ride.
#
# Full stack: R1+R2+R3+R4+R5a+R5b+R7+R11(weight=0.20)+R12
#
# Gate (judge at epoch 25):
#   DiseaseClf BCE < 0.50 by epoch 20  (vs. 0.63 plateau in biggpu run)
#   probe_auc/cardiomegaly > 0.75  AND  probe_auc/effusion > 0.75
#
# Usage:
#   bash launchers/single_runs/mini_vae/exp_D_disease_clf.sh
#   PARTITION=gpu bash launchers/single_runs/mini_vae/exp_D_disease_clf.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-bigbatch}"
EXP_NAME="mini_D_disease_clf"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  python run/train_mini_sepvae.py \
    --exp_name         "${EXP_NAME}" \
    --use_vinbig \
    --data_dir         /datasets/mmolefe/vinbigdata/train \
    --csv_path         /datasets/mmolefe/vinbigdata/train.csv \
    --img_size         128 \
    --batch_size       32 \
    --epochs           50 \
    --lr_vae           3e-4 \
    --lr_disc          1e-4 \
    --lr_cross_disc    1e-4 \
    --lr_decay_epochs  30 \
    --kl_warmup_epochs 5 \
    --weight_kl_common    1e-4 \
    --weight_kl_disease   1e-4 \
    --weight_null         0.05 \
    --sigma_inactive      0.05 \
    --weight_orthogonality 0.05 \
    --weight_mi           0.005 \
    --use_contrastive \
    --weight_contrastive  0.1 \
    --contrastive_margin  0.5 \
    --use_cross_adv \
    --weight_cross_adv    0.05 \
    --cross_disc_hidden_dim 128 \
    --use_disease_clf \
    --weight_disease_clf  0.2 \
    --lr_disease_clf      1e-4 \
    --use_label_attention \
    --attn_query_dim      64 \
    --weight_spatial_attn 0.05 \
    --seed 42
  exit $?
fi

sbatch \
  --partition="${PARTITION}" \
  --job-name="${EXP_NAME}" \
  --output="results/slurm_logs/${EXP_NAME}_%j.out" \
  --error="results/slurm_logs/${EXP_NAME}_%j.err" \
  slurm_scripts/mini_sep_vae.slurm "$(realpath "$0")"

echo "Submitted: ${EXP_NAME}  →  partition=${PARTITION}"
echo "Watch:     tail -f results/slurm_logs/${EXP_NAME}_<JOB_ID>.out"
