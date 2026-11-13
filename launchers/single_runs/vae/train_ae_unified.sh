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

# --- Defaults (can be overridden by command-line args) ---
export ENV_NAME="jaxstack"
export DATA_ROOT="../datasets/cleaned"
export TASK="All_CXR"
export IMG_SIZE="128"
export CLASS_FILTER="None"
export BASE_CH="128"
export CH_MULTS="1,2,4"
export Z_CHANNELS="1"
export EMBED_DIM="None"
export NUM_RES_BLOCKS="3"
export ATTN_RES="16,8"
export LR="2e-4"
export KL_WEIGHT="1.0e-6"
export EPOCHS="100"
export BATCH_PER_DEVICE="8"
export SAMPLE_EVERY="10"
export WANDB="1"
export WANDB_PROJECT="unified-cxr-vae"
export WANDB_RUN_GROUP="unified-ae"
# SLURM Defaults
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="unified-ae-proto"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"
# --- Parse Command-Line Arguments ---
# This loop processes arguments like --partition, --job-name, etc.
# Any other arguments (e.g., --img_size, --base_ch) are passed to the python script via $@
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)
      export SLURM_PARTITION="$2"
      shift 2
      ;;
    --job-name)
      export SLURM_JOB_NAME="$2"
      shift 2
      ;;
    --time)
      export TIME_LIMIT="$2"
      shift 2
      ;;
    --workdir)
      export WORKDIR="$2"
      shift 2
      ;;
    --wandb_project)
      export WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_name)
      export WANDB_NAME="$2"
      shift 2
      ;;
    --wandb_tags)
      export WANDB_TAGS="$2"
      shift 2
      ;;
    --wandb_group)
      export WANDB_RUN_GROUP="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1") # save unrecognized arg
      shift
      ;;
  esac
done

# --- Run Naming (uses final values) ---
REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

export WANDB_NAME="${WANDB_NAME:-${SLURM_JOB_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-$WANDB_NAME}"
export WANDB_TAGS="${WANDB_TAGS:-unified-ae,all-cxr,z${Z_CHANNELS},${GIT_BRANCH},${GIT_HASH}}"

JOB_NAME="${SLURM_JOB_NAME}-${GIT_HASH}"
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${TIMESTAMP}"

echo "Staging repo to ${STAGING_DIR}"
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' \
  --exclude 'runs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'wandb' \
  --exclude 'playground' \
  --exclude 'composed_output_sweep' \
  --exclude 'composed_output_single' \
  --exclude 'composed_output' \
  --exclude 'runs_ldm' \
  --exclude 'preencoded_latents' \
  --exclude 'runs' \
  --exclude 'and_out' \
  --exclude 'superdiff_and_output' \
  "$REPO_ROOT/" "$STAGING_DIR/"

status_line "--------------------------------------------------------"
mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/runs"
mkdir -p "${STAGING_DIR}/runs_ldm"
export WORKDIR="${WORKDIR:-$STAGING_DIR}"
status_line "--------------------------------------------------------"
status_line "📂 Workdir" "$WORKDIR"
status_line "📌 Commit"  "$GIT_HASH"
status_line "🏷️  Branch"  "$GIT_BRANCH"
status_line "🪪 W&B"     "$WANDB_NAME"
# --- Submit to SLURM ---
echo "Submitting Unified Autoencoder Training..."
JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
  --job-name="$SLURM_JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL,ENV_NAME="$ENV_NAME",WORKDIR="$WORKDIR",GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",WANDB_NAME="$WANDB_NAME",WANDB_TAGS="$WANDB_TAGS",WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_PROJECT="$WANDB_PROJECT" \
  slurm_scripts/cxr_ae.slurm "${OTHER_ARGS[@]}" | awk '{print $4}')
status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at" "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"

#bash launchers/single_runs/vae/train_ae_unified.sh \
#  --data_root /datasets/mmolefe/cleaned \
#  --task All_CXR \
#  --split train \
#  --img_size 256 \
#  --base_ch 128 \
#  --ch_mults 1,2,4 \
#  --num_res_blocks 3 \
#  --attn_res 16,8 \
#  --z_channels 1 \
#  --lr 2e-4 \
#  --weight_decay 1e-4 \
#  --epochs 100 \
#  --batch_per_device 8 \
#  --sample_every 10 \
#  --log_every 1 \
#  --output_root runs \
#  --exp_name cxr_ae \
#  --wandb \
#  --wandb_project unified-cxr-vae \
#  --wandb_tags unified-ae