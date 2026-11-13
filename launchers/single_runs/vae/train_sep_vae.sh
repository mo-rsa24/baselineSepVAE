#!/usr/bin/env bash
set -euo pipefail

# Using tput for better compatibility and cleaner syntax
if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }

# ---------------------------------------------------------
# Defaults — RTX 4090 24GB configuration
# Override any of these via command-line args (see below)
# ---------------------------------------------------------

# Environment
export ENV_NAME="jaxstack"

# Data
export DICOM_DIR="/datasets/mmolefe/vinbigdata/train"
export CSV_PATH="/datasets/mmolefe/vinbigdata/train.csv"
export IMG_SIZE="512"

# Model
export Z_CHANNELS_COMMON="4"
export Z_CHANNELS_DISEASE="2"
export FROZEN_BACKBONE="1"
export USE_FPN="0"
export FPN_CHANNELS="512"
export UNFREEZE_FROM=""

# CheSS weights
export CHESS_CHECKPOINT="/datasets/mmolefe/chess/pretrained_weights.pth.tar"
export CHESS_CONVERTED=""

# Loss weights
export WEIGHT_REC="1.0"
export WEIGHT_KL_COMMON="1e-4"
export WEIGHT_KL_DISEASE="1e-4"
export WEIGHT_NULL="1e-3"
export WEIGHT_MI="1e-3"
export SIGMA_INACTIVE="0.1"
export FREE_BITS="0.0"
export WEIGHT_PERCEPTUAL="0.0"
export WEIGHT_ADVERSARIAL="0.0"
export DISC_START_EPOCH="10"
export KL_WARMUP_EPOCHS="0"

# Optimizer — batch_size=4, lr=1e-4 (base rate)
export LR_VAE="1e-4"
export LR_DISC="1e-4"
export LR_BACKBONE="1e-5"
export LR_PATCH_DISC="4e-4"
export WEIGHT_DECAY="1e-4"
export GRAD_CLIP="1.0"

# Training
export BATCH_SIZE="4"
export EPOCHS="100"
export NUM_WORKERS="8"
export SEED="0"

# Logging & Checkpoints
export OUTPUT_ROOT="runs_sepvae"
export EXP_NAME="sepvae_chess"
export LOG_EVERY="100"
export SAVE_EVERY="10"

# Verbose diagnostics
export VERBOSE_BACKBONE="1"
export VERBOSE_N_SAMPLES="12"

# WandB
export WANDB="1"
export WANDB_PROJECT="sepvae-chess"
export WANDB_ENTITY=""
export WANDB_RUN_GROUP="sepvae"

# SLURM
export SLURM_PARTITION="${SLURM_PARTITION:-bigbatch}"
export SLURM_JOB_NAME="sep-vae"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"

# ---------------------------------------------------------
# Parse command-line arguments
# Recognised args are consumed; everything else is forwarded
# to the python script via the slurm script's "$@"
# ---------------------------------------------------------
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    # --- SLURM ---
    --partition)       export SLURM_PARTITION="$2";     shift 2 ;;
    --job-name)        export SLURM_JOB_NAME="$2";      shift 2 ;;
    --time)            export TIME_LIMIT="$2";           shift 2 ;;
    --workdir)         export WORKDIR="$2";              shift 2 ;;
    # --- Data ---
    --dicom_dir)       export DICOM_DIR="$2";            shift 2 ;;
    --csv_path)        export CSV_PATH="$2";             shift 2 ;;
    --img_size)        export IMG_SIZE="$2";             shift 2 ;;
    # --- Model ---
    --z_channels_common)  export Z_CHANNELS_COMMON="$2"; shift 2 ;;
    --z_channels_disease) export Z_CHANNELS_DISEASE="$2"; shift 2 ;;
    --use_fpn)            export USE_FPN="1";               shift ;;
    --no_fpn)             export USE_FPN="0";               shift ;;
    --fpn_channels)       export FPN_CHANNELS="$2";         shift 2 ;;
    --unfreeze_from)      export UNFREEZE_FROM="$2";        shift 2 ;;
    # --- CheSS ---
    --chess_checkpoint) export CHESS_CHECKPOINT="$2";    shift 2 ;;
    --chess_converted)  export CHESS_CONVERTED="$2";     shift 2 ;;
    # --- Loss ---
    --weight_rec)      export WEIGHT_REC="$2";           shift 2 ;;
    --weight_kl_common) export WEIGHT_KL_COMMON="$2";   shift 2 ;;
    --weight_kl_disease) export WEIGHT_KL_DISEASE="$2"; shift 2 ;;
    --weight_null)     export WEIGHT_NULL="$2";          shift 2 ;;
    --weight_mi)       export WEIGHT_MI="$2";            shift 2 ;;
    --sigma_inactive)  export SIGMA_INACTIVE="$2";       shift 2 ;;
    --free_bits)       export FREE_BITS="$2";            shift 2 ;;
    --weight_perceptual) export WEIGHT_PERCEPTUAL="$2";  shift 2 ;;
    --weight_adversarial) export WEIGHT_ADVERSARIAL="$2"; shift 2 ;;
    --disc_start_epoch) export DISC_START_EPOCH="$2";    shift 2 ;;
    --kl_warmup_epochs) export KL_WARMUP_EPOCHS="$2";   shift 2 ;;
    # --- Optimizer ---
    --lr_vae)          export LR_VAE="$2";               shift 2 ;;
    --lr_disc)         export LR_DISC="$2";              shift 2 ;;
    --lr_backbone)     export LR_BACKBONE="$2";          shift 2 ;;
    --lr_patch_disc)   export LR_PATCH_DISC="$2";        shift 2 ;;
    --weight_decay)    export WEIGHT_DECAY="$2";         shift 2 ;;
    --grad_clip)       export GRAD_CLIP="$2";            shift 2 ;;
    # --- Training ---
    --batch_size)      export BATCH_SIZE="$2";           shift 2 ;;
    --epochs)          export EPOCHS="$2";               shift 2 ;;
    --num_workers)     export NUM_WORKERS="$2";          shift 2 ;;
    --seed)            export SEED="$2";                 shift 2 ;;
    # --- Logging ---
    --output_root)     export OUTPUT_ROOT="$2";          shift 2 ;;
    --exp_name)        export EXP_NAME="$2";             shift 2 ;;
    --log_every)       export LOG_EVERY="$2";            shift 2 ;;
    --save_every)      export SAVE_EVERY="$2";           shift 2 ;;
    # --- Diagnostics ---
    --verbose_backbone) export VERBOSE_BACKBONE="1";     shift ;;
    --no_verbose_backbone) export VERBOSE_BACKBONE="0";  shift ;;
    --verbose_n_samples) export VERBOSE_N_SAMPLES="$2";  shift 2 ;;
    # --- WandB ---
    --wandb_project)   export WANDB_PROJECT="$2";        shift 2 ;;
    --wandb_name)      export WANDB_NAME="$2";           shift 2 ;;
    --wandb_entity)    export WANDB_ENTITY="$2";         shift 2 ;;
    --wandb_group)     export WANDB_RUN_GROUP="$2";      shift 2 ;;
    --no_wandb)        export WANDB="0";                 shift ;;
    # --- Passthrough ---
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

export WANDB_NAME="${WANDB_NAME:-${SLURM_JOB_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-$WANDB_NAME}"
export WANDB_TAGS="${WANDB_TAGS:-sepvae,chess,bs${BATCH_SIZE},${GIT_BRANCH},${GIT_HASH}}"

JOB_NAME="${SLURM_JOB_NAME}-${GIT_HASH}"
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${TIMESTAMP}"

# ---------------------------------------------------------
# Stage repo snapshot
# ---------------------------------------------------------
echo "Staging repo to ${STAGING_DIR}"
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' \
  --exclude 'runs' \
  --exclude 'runs_sepvae' \
  --exclude 'runs_sepvae_debug' \
  --exclude 'runs_ldm' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'wandb' \
  --exclude 'playground' \
  --exclude 'preencoded_latents' \
  --exclude 'composed_output*' \
  --exclude 'and_out' \
  --exclude 'superdiff_and_output' \
  "$REPO_ROOT/" "$STAGING_DIR/"

mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/runs_sepvae"
export WORKDIR="${WORKDIR:-$STAGING_DIR}"

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
rule
status_line "Workdir"       "$WORKDIR"
status_line "Commit"        "$GIT_HASH"
status_line "Branch"        "$GIT_BRANCH"
status_line "W&B name"      "$WANDB_NAME"
status_line "Batch size"    "$BATCH_SIZE (x3 = $((BATCH_SIZE * 3)) images)"
status_line "LR (VAE)"      "$LR_VAE"
status_line "LR (disc)"     "$LR_DISC"
status_line "Epochs"        "$EPOCHS"
status_line "Save every"    "$SAVE_EVERY"
rule

# ---------------------------------------------------------
# Submit to SLURM
# ---------------------------------------------------------
echo "Submitting SepVAE Training..."
JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
  --job-name="$SLURM_JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL,ENV_NAME="$ENV_NAME",WORKDIR="$WORKDIR",\
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
WEIGHT_MI="$WEIGHT_MI",SIGMA_INACTIVE="$SIGMA_INACTIVE",\
FREE_BITS="$FREE_BITS",WEIGHT_PERCEPTUAL="$WEIGHT_PERCEPTUAL",\
WEIGHT_ADVERSARIAL="$WEIGHT_ADVERSARIAL",DISC_START_EPOCH="$DISC_START_EPOCH",\
KL_WARMUP_EPOCHS="$KL_WARMUP_EPOCHS",\
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
status_line "Logs at" "${REPO_ROOT}/logs/${SLURM_JOB_NAME}-${JOB_ID}.out"

# ---------------------------------------------------------
# Example usage:
#
# RTX 4090 (24GB) — default config:
#   bash launchers/single_runs/vae/train_sep_vae.sh
#
# A100 (80GB) — larger batch, scaled LR:
#   bash launchers/single_runs/vae/train_sep_vae.sh \
#     --batch_size 24 --lr_vae 3e-4 --lr_disc 3e-4 \
#     --save_every 10 --verbose_n_samples 18 \
#     --exp_name sepvae_a100
#
# V100 (32GB) — moderate batch:
#   bash launchers/single_runs/vae/train_sep_vae.sh \
#     --batch_size 8 --lr_vae 1e-4 --lr_disc 1e-4 \
#     --exp_name sepvae_v100
#
# Skip pre-training conversion (reuse .npy):
#   bash launchers/single_runs/vae/train_sep_vae.sh \
#     --chess_converted /path/to/chess_jax_params.npy
#
# Disable W&B:
#   bash launchers/single_runs/vae/train_sep_vae.sh --no_wandb
# ---------------------------------------------------------
