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
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

export ENV_NAME="jaxstack"
export DATA_ROOT="/datasets/mmolefe/cleaned"
export TASK="TB"
export IMG_SIZE="256"
export Z_CHANNELS="${1:-128}"

# --- Training ---
export LR="2e-4"
export KL_WEIGHT="1e-6"
export EPOCHS="100"
export BATCH_PER_DEVICE="2"
export WANDB_PROJECT="cxr-vae"
export WANDB_RUN_GROUP="tb-ae"
export WANDB="1"
# SLURM Defaults
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="tb-ae-proto"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"

# --- Parse Command-Line Arguments ---
OTHER_ARGS=()
shift || true
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
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

export WANDB_NAME="${WANDB_NAME:-${SLURM_JOB_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-$WANDB_NAME}"
export WANDB_TAGS="${WANDB_TAGS:-vae,tb,z${Z_CHANNELS},${GIT_BRANCH},${GIT_HASH}}"

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
  --exclude 'runs' \
  --exclude 'and_out' \
  --exclude 'superdiff_and_output' \
  "$REPO_ROOT/" "$STAGING_DIR/"

status_line "✅ Snapshot" "Complete"


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


echo "Submitting VAE Training for TB with Z_CHANNELS=${Z_CHANNELS}"
JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
  --job-name="$SLURM_JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL,ENV_NAME="$ENV_NAME",WORKDIR="$WORKDIR",GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",WANDB_NAME="$WANDB_NAME",WANDB_TAGS="$WANDB_TAGS",WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_PROJECT="$WANDB_PROJECT" \
  slurm_scripts/cxr_ae.slurm "${OTHER_ARGS[@]}" | awk '{print $4}')
status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at" "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"