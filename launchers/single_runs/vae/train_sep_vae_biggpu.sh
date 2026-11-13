#!/usr/bin/env bash
# =============================================================================
# Dual-GPU SLURM Launcher — mscluster106 (2x Quadro RTX 8000, 49 GB each)
# =============================================================================
# Run from the login node. Submits ONE sbatch job to the biggpu partition on
# mscluster106. The job runs two Python processes in parallel internally:
#
#   GPU 0  sepvae_baseline_fixed  R1+R2+R3+R4             (corrected baseline)
#   GPU 1  sepvae_full            R1+R2+R3+R4+R5a+R5b+R7  (gate-check candidate)
#
# OverSubscribe=NO on biggpu means two separate jobs cannot run simultaneously
# on the same node, so both training runs live inside a single SLURM job with
# internal CUDA_VISIBLE_DEVICES pinning.
#
# Usage (from repo root on the login node):
#   bash launchers/single_runs/vae/train_sep_vae_biggpu.sh
#   bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --no_wandb
#   bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --batch_size_baseline 24 --batch_size_full 10
#   bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --weight_spatial_attn 0.1
#
# Monitor (after submission):
#   tail -f logs/biggpu-<JOB_ID>.out          # SLURM wrapper log
#   tail -f <WORKDIR>/logs/biggpu-baseline-<JOB_ID>.log  # baseline training log
#   tail -f <WORKDIR>/logs/biggpu-full-<JOB_ID>.log      # full-stack training log
# =============================================================================
set -euo pipefail

if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); GREEN=$(tput setaf 2); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; GREEN=""; RESET=""
fi
status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-24s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s─${RESET}" {1..60}; printf "\n"; }

# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------
ENV_NAME="${ENV_NAME:-jaxstack}"
DICOM_DIR="${DICOM_DIR:-/datasets/mmolefe/vinbigdata/train}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/train.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
CHESS_CONVERTED="${CHESS_CONVERTED:-}"

BATCH_SIZE_BASELINE="4"    # baseline_fixed: R1+R2+R3+R4 only; bs=8 OOM'd (17.9GB alloc), bs=6 matches full run image count (18 imgs)
BATCH_SIZE_FULL="4"        # sepvae_full: R5a+R5b+R7; bs=8 OOM'd at ~21GB, bs=6 gives headroom
EPOCHS="150"
NUM_WORKERS="4"
HALF_PRECISION="bf16"
GRADIENT_CHECKPOINTING="1"
WANDB="1"
WANDB_PROJECT="sepvae-disentangle"
WANDB_ENTITY=""
SLURM_PARTITION="biggpu"
TARGET_NODE="mscluster106"
TIME_LIMIT="72:00:00"
STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"
WEIGHT_SPATIAL_ATTN="0.05"

# ---------------------------------------------------------
# Parse overrides
# ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch_size_baseline)  BATCH_SIZE_BASELINE="$2";  shift 2 ;;
    --batch_size_full)      BATCH_SIZE_FULL="$2";      shift 2 ;;
    --epochs)               EPOCHS="$2";               shift 2 ;;
    --node)                 TARGET_NODE="$2";           shift 2 ;;
    --weight_spatial_attn)  WEIGHT_SPATIAL_ATTN="$2";  shift 2 ;;
    --no_wandb)             WANDB="0";                 shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------
# Run metadata
# ---------------------------------------------------------
REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

mkdir -p "${REPO_ROOT}/logs"

# ---------------------------------------------------------
# Stage repo snapshot
# ---------------------------------------------------------
STAGING_DIR="${STAGING_ROOT}/biggpu-${GIT_HASH}-${TIMESTAMP}"
echo "Staging repo to ${STAGING_DIR} ..."
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' --exclude 'runs' --exclude 'runs_sepvae' \
  --exclude 'runs_sepvae_debug' --exclude 'runs_ldm' \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'wandb' --exclude 'playground' \
  --exclude 'preencoded_latents' \
  "$REPO_ROOT/" "$STAGING_DIR/"
mkdir -p "${STAGING_DIR}/runs_sepvae" "${STAGING_DIR}/logs"

# ---------------------------------------------------------
# Submit single job (both GPUs live inside it)
# ---------------------------------------------------------
JOB_ID=$(sbatch \
  --partition="$SLURM_PARTITION" \
  --nodelist="$TARGET_NODE" \
  --job-name="sepvae-biggpu" \
  --time="$TIME_LIMIT" \
  --ntasks=1 \
  --output="${REPO_ROOT}/logs/biggpu-%j.out" \
  --error="${REPO_ROOT}/logs/biggpu-%j.err" \
  --export=ALL,\
ENV_NAME="$ENV_NAME",\
WORKDIR="$STAGING_DIR",\
GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",\
WANDB_RUN_GROUP="biggpu",WANDB_PROJECT="$WANDB_PROJECT",\
DICOM_DIR="$DICOM_DIR",CSV_PATH="$CSV_PATH",\
CHESS_CHECKPOINT="$CHESS_CHECKPOINT",CHESS_CONVERTED="$CHESS_CONVERTED",\
BATCH_SIZE_BASELINE="$BATCH_SIZE_BASELINE",BATCH_SIZE_FULL="$BATCH_SIZE_FULL",\
EPOCHS="$EPOCHS",NUM_WORKERS="$NUM_WORKERS",\
HALF_PRECISION="$HALF_PRECISION",GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING",\
WANDB="$WANDB",WANDB_ENTITY="$WANDB_ENTITY",\
WEIGHT_SPATIAL_ATTN="$WEIGHT_SPATIAL_ATTN" \
  slurm_scripts/sep_vae_biggpu.slurm \
  | awk '{print $4}')

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
rule
status_line "Submitted"   "Job ID: ${JOB_ID}"
status_line "Node"        "${TARGET_NODE}  (partition: ${SLURM_PARTITION})"
status_line "Commit"      "${GIT_HASH}  (${GIT_BRANCH})"
status_line "Staging dir" "${STAGING_DIR}"
status_line "GPU 0 batch" "${BATCH_SIZE_BASELINE}  (×3 = $((BATCH_SIZE_BASELINE * 3)) images/step)"
status_line "GPU 1 batch" "${BATCH_SIZE_FULL}  (×3 = $((BATCH_SIZE_FULL * 3)) images/step)"
status_line "Precision"   "${HALF_PRECISION} + gradient_checkpointing"
status_line "Epochs"       "${EPOCHS}"
status_line "SpatialAttn"  "w=${WEIGHT_SPATIAL_ATTN}  (full run only, requires --use_label_attention)"
rule
printf "${GREEN}▶${RESET} ${BOLD}GPU 0${RESET}  sepvae_baseline_fixed  [R1+R2+R3+R4+R11]                   bs=${BATCH_SIZE_BASELINE}\n"
printf "${GREEN}▶${RESET} ${BOLD}GPU 1${RESET}  sepvae_full            [R1+R2+R3+R4+R5a+R7+R11+R12+FPN]   bs=${BATCH_SIZE_FULL}\n"
rule
printf "Monitor SLURM wrapper:  tail -f ${REPO_ROOT}/logs/biggpu-%s.out\n" "$JOB_ID"
printf "Monitor baseline:       tail -f ${STAGING_DIR}/logs/biggpu-baseline-%s.log\n" "$JOB_ID"
printf "Monitor full:           tail -f ${STAGING_DIR}/logs/biggpu-full-%s.log\n" "$JOB_ID"
printf "Cancel:                 scancel %s\n" "$JOB_ID"
