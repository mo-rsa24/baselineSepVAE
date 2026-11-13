#!/usr/bin/env bash
# =============================================================================
# Experiment C — reduced contrastive margin (0.5 → 0.3)
#
# In the biggpu run, contrastive loss stalled at 2.1 after 85 epochs (should
# converge toward 0). A margin of 0.5 is too large for the 2-channel latent
# geometry — the model cannot push centroids 0.5 apart in cosine space while
# satisfying reconstruction and KL constraints simultaneously.
# Reducing to 0.3 allows the contrastive loss to reach its minimum consistently,
# providing a more stable separation gradient throughout training.
#
# Full stack: R1+R2+R3+R4+R5a(margin=0.3)+R5b+R7+R11+R12
#
# Gate (judge at epoch 25):
#   Contrastive loss < 0.8 by epoch 20  (vs. 2.1 in the biggpu run)
#   cross_head_score: std < 0.15 AND mean < 0.65
#
# Usage:
#   bash launchers/single_runs/mini_vae/exp_C_contrastive_margin.sh
#   PARTITION=gpu bash launchers/single_runs/mini_vae/exp_C_contrastive_margin.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-bigbatch}"
EXP_NAME="mini_C_contrastive_margin"

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
    --contrastive_margin  0.3 \
    --use_cross_adv \
    --weight_cross_adv    0.05 \
    --cross_disc_hidden_dim 128 \
    --use_disease_clf \
    --weight_disease_clf  0.1 \
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
