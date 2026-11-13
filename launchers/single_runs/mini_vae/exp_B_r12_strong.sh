#!/usr/bin/env bash
# =============================================================================
# Experiment B — R12 stronger spatial supervision (weight=0.10)
#
# Run if Experiment A still shows cross_head_score oscillation past epoch 25.
# Doubles the spatial attention supervision weight to 0.10 to assert a stronger
# spatial prior against the adversarial gradient from R5b.
#
# Full stack: R1+R2+R3+R4+R5a+R5b+R7+R11+R12 (weight_spatial_attn=0.10)
#
# Gate (judge at epoch 25):
#   cross_head_score: std < 0.15 AND mean < 0.65 over evals at ep10/15/20/25
#   loss/spatial_attn < 0.15 by epoch 15  (stronger weight → tighter target)
#
# Usage:
#   bash launchers/single_runs/mini_vae/exp_B_r12_strong.sh
#   PARTITION=gpu bash launchers/single_runs/mini_vae/exp_B_r12_strong.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-bigbatch}"
EXP_NAME="mini_B_r12_strong"

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
    --weight_disease_clf  0.1 \
    --lr_disease_clf      1e-4 \
    --use_label_attention \
    --attn_query_dim      64 \
    --weight_spatial_attn 0.10 \
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
