#!/usr/bin/env bash
# =============================================================================
# Attention (R1 + R2 + R3 + R4 + R7)
# ─────────────────────────────────────────────────────────────────────────────
# Adds disease prototype label attention (R7) on top of the baseline_fixed
# config, without contrastive or cross-adversarial losses.
# R7 attaches a learned query vector per disease head that cross-attends over
# backbone spatial features, giving the encoder an explicit spatial routing
# mechanism for each disease.
#
# Usage:
#   bash launchers/single_runs/vae/train_sep_vae_attention.sh
#   bash launchers/single_runs/vae/train_sep_vae_attention.sh --partition gpuq
# =============================================================================
set -euo pipefail

if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi
status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-22s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..55}; printf "\n"; }

# ---------------------------------------------------------
# Defaults — same as baseline_fixed, plus R7
# ---------------------------------------------------------
export ENV_NAME="jaxstack"
export DICOM_DIR="/datasets/mmolefe/vinbigdata/train"
export CSV_PATH="/datasets/mmolefe/vinbigdata/train.csv"
export IMG_SIZE="512"
export Z_CHANNELS_COMMON="4"
export Z_CHANNELS_DISEASE="2"
export FROZEN_BACKBONE="1"
export USE_FPN="0"
export FPN_CHANNELS="512"
export UNFREEZE_FROM=""
export CHESS_CHECKPOINT="/datasets/mmolefe/chess/pretrained_weights.pth.tar"
export CHESS_CONVERTED=""

# Loss weights (same as baseline_fixed)
export WEIGHT_REC="1.0"
export WEIGHT_KL_COMMON="1e-4"
export WEIGHT_KL_DISEASE="1e-4"
export WEIGHT_NULL="0.05"
export WEIGHT_MI="0.005"
export WEIGHT_ORTHOGONALITY="0.05"
export SIGMA_INACTIVE="0.05"
export FREE_BITS="0.0"
export WEIGHT_PERCEPTUAL="0.05"
export WEIGHT_ADVERSARIAL="0.0"
export DISC_START_EPOCH="10"
export KL_WARMUP_EPOCHS="10"
export LR_DECAY_EPOCHS="80"       # longer decay for attention runs

# R5a/b: disabled
export USE_CONTRASTIVE="0"
export WEIGHT_CONTRASTIVE="0.1"
export CONTRASTIVE_MARGIN="0.5"
export USE_CROSS_ADV="0"
export WEIGHT_CROSS_ADV="0.05"
export LR_CROSS_DISC="1e-4"
export CROSS_DISC_HIDDEN_DIM="256"

# R7: Disease prototype label attention — ENABLED
export USE_LABEL_ATTENTION="1"
export ATTN_QUERY_DIM="256"

# Optimizer
export LR_VAE="1e-4"
export LR_DISC="1e-4"
export LR_BACKBONE="1e-5"
export LR_PATCH_DISC="4e-4"
export WEIGHT_DECAY="1e-4"
export GRAD_CLIP="1.0"

# Training
export BATCH_SIZE="10"
export EPOCHS="150"
export NUM_WORKERS="8"
export SEED="0"

# Logging & Checkpoints
export OUTPUT_ROOT="runs_sepvae"
export EXP_NAME="sepvae_attention"
export LOG_EVERY="100"
export SAVE_EVERY="1"

# Verbose diagnostics
export VERBOSE_BACKBONE="0"
export VERBOSE_N_SAMPLES="12"

# WandB
export WANDB="1"
export WANDB_PROJECT="sepvae-disentangle"
export WANDB_ENTITY=""
export WANDB_RUN_GROUP="recommendations"

# SLURM
export SLURM_PARTITION="${SLURM_PARTITION:-bigbatch}"
export SLURM_JOB_NAME="sepvae-attention"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"

# ---------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)         export SLURM_PARTITION="$2";     shift 2 ;;
    --job-name)          export SLURM_JOB_NAME="$2";      shift 2 ;;
    --time)              export TIME_LIMIT="$2";           shift 2 ;;
    --exp_name)          export EXP_NAME="$2";             shift 2 ;;
    --batch_size)        export BATCH_SIZE="$2";           shift 2 ;;
    --epochs)            export EPOCHS="$2";               shift 2 ;;
    --attn_query_dim)    export ATTN_QUERY_DIM="$2";       shift 2 ;;
    --weight_adversarial) export WEIGHT_ADVERSARIAL="$2"; shift 2 ;;
    --weight_perceptual)  export WEIGHT_PERCEPTUAL="$2";  shift 2 ;;
    --resume)            OTHER_ARGS+=("--resume" "$2");    shift 2 ;;
    --no_wandb)          export WANDB="0";                 shift ;;
    *) OTHER_ARGS+=("$1"); shift ;;
  esac
done

# ---------------------------------------------------------
# Run naming
# ---------------------------------------------------------
REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${GIT_HASH}-${TIMESTAMP}}"
export WANDB_TAGS="${WANDB_TAGS:-sepvae,attention,R1,R2,R3,R4,R7,bs${BATCH_SIZE},${GIT_BRANCH}}"

JOB_NAME="${SLURM_JOB_NAME}-${GIT_HASH}"
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${TIMESTAMP}"

# ---------------------------------------------------------
# Stage repo snapshot
# ---------------------------------------------------------
echo "Staging repo to ${STAGING_DIR}"
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' --exclude 'runs' --exclude 'runs_sepvae' \
  --exclude 'runs_sepvae_debug' --exclude 'runs_ldm' \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'wandb' --exclude 'playground' \
  --exclude 'preencoded_latents' \
  "$REPO_ROOT/" "$STAGING_DIR/"

mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/runs_sepvae"
export WORKDIR="${WORKDIR:-$STAGING_DIR}"

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
rule
status_line "Experiment"    "$EXP_NAME  [R1+R2+R3+R4+R7]"
status_line "Workdir"       "$WORKDIR"
status_line "Commit"        "$GIT_HASH  ($GIT_BRANCH)"
status_line "W&B name"      "$WANDB_NAME"
status_line "Batch size"    "$BATCH_SIZE (x3 = $((BATCH_SIZE * 3)) images)"
status_line "Epochs"        "$EPOCHS  (LR decay from ep $LR_DECAY_EPOCHS)"
status_line "Free-bits"     "$FREE_BITS  (R1)"
status_line "sigma_inactive" "$SIGMA_INACTIVE  (R2)"
status_line "null/ortho"    "$WEIGHT_NULL / $WEIGHT_ORTHOGONALITY  (R2)"
status_line "perceptual"    "$WEIGHT_PERCEPTUAL  (R4)"
status_line "adversarial"   "${WEIGHT_ADVERSARIAL}  (PatchGAN, start ep ${DISC_START_EPOCH})"
status_line "Contrastive"   "disabled"
status_line "Cross-adv"     "disabled"
status_line "Label-attn"    "ENABLED  query_dim=$ATTN_QUERY_DIM  (R7)"
rule

# ---------------------------------------------------------
# Submit to SLURM
# ---------------------------------------------------------
echo "Submitting ${EXP_NAME}..."
JOB_ID=$(sbatch \
  --partition="$SLURM_PARTITION" \
  --job-name="$SLURM_JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL,\
ENV_NAME="$ENV_NAME",WORKDIR="$WORKDIR",\
GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",\
WANDB_NAME="$WANDB_NAME",WANDB_TAGS="$WANDB_TAGS",\
WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_PROJECT="$WANDB_PROJECT",\
DICOM_DIR="$DICOM_DIR",CSV_PATH="$CSV_PATH",IMG_SIZE="$IMG_SIZE",\
Z_CHANNELS_COMMON="$Z_CHANNELS_COMMON",Z_CHANNELS_DISEASE="$Z_CHANNELS_DISEASE",\
FROZEN_BACKBONE="$FROZEN_BACKBONE",USE_FPN="$USE_FPN",\
FPN_CHANNELS="$FPN_CHANNELS",UNFREEZE_FROM="$UNFREEZE_FROM",\
CHESS_CHECKPOINT="$CHESS_CHECKPOINT",CHESS_CONVERTED="$CHESS_CONVERTED",\
WEIGHT_REC="$WEIGHT_REC",WEIGHT_KL_COMMON="$WEIGHT_KL_COMMON",\
WEIGHT_KL_DISEASE="$WEIGHT_KL_DISEASE",WEIGHT_NULL="$WEIGHT_NULL",\
WEIGHT_MI="$WEIGHT_MI",WEIGHT_ORTHOGONALITY="$WEIGHT_ORTHOGONALITY",\
SIGMA_INACTIVE="$SIGMA_INACTIVE",FREE_BITS="$FREE_BITS",\
WEIGHT_PERCEPTUAL="$WEIGHT_PERCEPTUAL",WEIGHT_ADVERSARIAL="$WEIGHT_ADVERSARIAL",\
DISC_START_EPOCH="$DISC_START_EPOCH",KL_WARMUP_EPOCHS="$KL_WARMUP_EPOCHS",\
LR_DECAY_EPOCHS="$LR_DECAY_EPOCHS",\
USE_CONTRASTIVE="$USE_CONTRASTIVE",WEIGHT_CONTRASTIVE="$WEIGHT_CONTRASTIVE",\
CONTRASTIVE_MARGIN="$CONTRASTIVE_MARGIN",\
USE_CROSS_ADV="$USE_CROSS_ADV",WEIGHT_CROSS_ADV="$WEIGHT_CROSS_ADV",\
LR_CROSS_DISC="$LR_CROSS_DISC",CROSS_DISC_HIDDEN_DIM="$CROSS_DISC_HIDDEN_DIM",\
USE_LABEL_ATTENTION="$USE_LABEL_ATTENTION",ATTN_QUERY_DIM="$ATTN_QUERY_DIM",\
LR_VAE="$LR_VAE",LR_DISC="$LR_DISC",LR_BACKBONE="$LR_BACKBONE",\
LR_PATCH_DISC="$LR_PATCH_DISC",WEIGHT_DECAY="$WEIGHT_DECAY",\
GRAD_CLIP="$GRAD_CLIP",BATCH_SIZE="$BATCH_SIZE",EPOCHS="$EPOCHS",\
NUM_WORKERS="$NUM_WORKERS",SEED="$SEED",\
OUTPUT_ROOT="$OUTPUT_ROOT",EXP_NAME="$EXP_NAME",\
LOG_EVERY="$LOG_EVERY",SAVE_EVERY="$SAVE_EVERY",\
VERBOSE_BACKBONE="$VERBOSE_BACKBONE",VERBOSE_N_SAMPLES="$VERBOSE_N_SAMPLES",\
WANDB="$WANDB",WANDB_ENTITY="$WANDB_ENTITY" \
  slurm_scripts/sep_vae.slurm "${OTHER_ARGS[@]}" | awk '{print $4}')

status_line "Submitted" "Job ID: $JOB_ID"
status_line "Logs at"   "${REPO_ROOT}/logs/${SLURM_JOB_NAME}-${JOB_ID}.out"
