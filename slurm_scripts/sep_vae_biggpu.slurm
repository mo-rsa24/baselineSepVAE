#!/usr/bin/env bash
# Array job: two experiments on separate nodes.
#   Task 0 — full config  (with ortho, weight=0.1)
#   Task 1 — no ortho     (weight_orthogonality=0)
#
# Submit:  sbatch slurm_scripts/sep_vae_biggpu.slurm
# One job: sbatch --array=0 slurm_scripts/sep_vae_biggpu.slurm

#SBATCH --job-name=sepvae-ablation
#SBATCH --partition=biggpu
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-ablation-%A_%a.out
#SBATCH --error=logs/sepvae-ablation-%A_%a.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
RED=$(printf '\033[31m'); GRN=$(printf '\033[32m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-26s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}** %s${RST}\n" "$*"; }

# ----------------------------
# Per-task config
# ----------------------------
case "$SLURM_ARRAY_TASK_ID" in
  0)
    EXP_NAME="sepvae_full_ortho"
    WEIGHT_ORTHOGONALITY="0.1"
    ;;
  1)
    EXP_NAME="sepvae_no_ortho"
    WEIGHT_ORTHOGONALITY="0.0"
    ;;
esac

# ----------------------------
# Shared defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
ENV_NAME="${ENV_NAME:-jaxstack}"

DICOM_DIR="${DICOM_DIR:-/datasets/mmolefe/vinbigdata/train}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/train.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
CHESS_CONVERTED="${CHESS_CONVERTED:-}"

# Model
Z_COMMON="${Z_COMMON:-4}"
Z_DISEASE="${Z_DISEASE:-2}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"

# Training — batch_size=2 to avoid OOM at 512×512 (effective batch=6 with triplets)
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-5}"

# Loss weights
WEIGHT_REC="${WEIGHT_REC:-1.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-5e-5}"
WEIGHT_BBOX_ATTN="${WEIGHT_BBOX_ATTN:-0.2}"
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.2}"
EMA_DECAY="${EMA_DECAY:-0.999}"

# Logging
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
MANIFOLD_MAX="${MANIFOLD_MAX:-600}"
MANIFOLD_METHOD="${MANIFOLD_METHOD:-pca}"
N_SAMPLES="${N_SAMPLES:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs_sepvae}"

# W&B
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ----------------------------
# Environment
# ----------------------------
banner "Setup [task=${SLURM_ARRAY_TASK_ID}] ${EXP_NAME}"
cd "$WORKDIR"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}"

kv "Node"          "$(hostname)"
kv "Task ID"       "$SLURM_ARRAY_TASK_ID"
kv "Exp name"      "$EXP_NAME"
kv "Batch size"    "$BATCH_SIZE  (effective=$(( BATCH_SIZE * 3 )) with triplets)"
kv "Epochs"        "$EPOCHS"
kv "LR"            "$LR_VAE"
kv "KL warmup"     "$KL_WARMUP_EPOCHS epochs"
kv "z_common"      "$Z_COMMON  z_disease=$Z_DISEASE  sigma_inactive=$SIGMA_INACTIVE"
kv "w_rec"         "$WEIGHT_REC  w_perceptual=$WEIGHT_PERCEPTUAL"
kv "w_ortho"       "$WEIGHT_ORTHOGONALITY  w_bbox=$WEIGHT_BBOX_ATTN"
kv "ema_decay"     "$EMA_DECAY"
kv "W&B"           "${WANDB_PROJECT} (wandb=${WANDB})"

# ----------------------------
# Build args
# ----------------------------
ARGS=(
  run/train_sep_vae.py
  --dicom_dir          "$DICOM_DIR"
  --csv_path           "$CSV_PATH"
  --chess_checkpoint   "$CHESS_CHECKPOINT"
  --img_size           512

  --z_channels_common  "$Z_COMMON"
  --z_channels_disease "$Z_DISEASE"
  --sigma_inactive     "$SIGMA_INACTIVE"
  --attn_query_dim     "$ATTN_QUERY_DIM"

  --batch_size         "$BATCH_SIZE"
  --epochs             "$EPOCHS"
  --num_workers        "$NUM_WORKERS"
  --seed               "$SEED"

  --lr_vae             "$LR_VAE"
  --weight_decay       "$WEIGHT_DECAY"
  --grad_clip          "$GRAD_CLIP"
  --kl_warmup_epochs   "$KL_WARMUP_EPOCHS"

  --weight_rec             "$WEIGHT_REC"
  --weight_kl_common       "$WEIGHT_KL_COMMON"
  --weight_kl_disease      "$WEIGHT_KL_DISEASE"
  --weight_orthogonality   "$WEIGHT_ORTHOGONALITY"
  --weight_bbox_attn       "$WEIGHT_BBOX_ATTN"
  --weight_perceptual      "$WEIGHT_PERCEPTUAL"
  --ema_decay              "$EMA_DECAY"

  --output_root        "$OUTPUT_ROOT"
  --exp_name           "$EXP_NAME"
  --log_every          "$LOG_EVERY"
  --save_every         "$SAVE_EVERY"
  --sample_every       "$SAMPLE_EVERY"
  --manifold_every     "$MANIFOLD_EVERY"
  --manifold_max_samples "$MANIFOLD_MAX"
  --manifold_method    "$MANIFOLD_METHOD"
  --n_samples_per_class "$N_SAMPLES"
)

[[ -n "$CHESS_CONVERTED" ]] && ARGS+=( --chess_converted "$CHESS_CONVERTED" )

if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ----------------------------
# Launch
# ----------------------------
banner "Launching training [${EXP_NAME}]"
python -u "${ARGS[@]}"
ok "Training complete [${EXP_NAME}]"
