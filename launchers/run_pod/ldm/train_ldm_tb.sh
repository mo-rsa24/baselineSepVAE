#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

# --- Defaults (override with CLI flags) ---
TASK="TB"
CLASS_FILTER="1"
DISEASE_NAME="tb"
IMG_SIZE="256"
SPLIT="train"
DATA_ROOT="/workspace/datasets/cleaned"

AE_CKPT_PATH="/workspace/superdiff-ldm/runs/unified-ae-proto-increase-ae-autoencoder-eb7c6d6-20260112-063726/20260112-063740/ckpts/last.flax"
AE_CONFIG_PATH="/workspace/superdiff-ldm/runs/unified-ae-proto-increase-ae-autoencoder-eb7c6d6-20260112-063726/20260112-063740/run_meta.json"
PREENCODED_LATENTS_DIR="/workspace/superdiff-ldm/preencode_latents/vae"

LDM_CH_MULTS="1,2,4,4"
LDM_NUM_RES_BLOCKS="3"
LR="1e-4"
WEIGHT_DECAY="1e-4"
GRAD_CLIP="1.0"
EPOCHS="1500"
LOG_EVERY="100"
SAMPLE_EVERY="300"

# Conservative default due to latent stats instability at high batch sizes.
# Safe scaling plan: 4 -> 8 -> 16 (only if stats remain stable). Gradient accumulation is not supported here.
BATCH_PER_DEVICE="8"
SAMPLE_BATCH_SIZE="8"

OUTPUT_ROOT="runs_ldm"
EXP_NAME="cxr_ldm"

WANDB_PROJECT="cxr-ldm-composition-test"
WANDB_ENTITY=""
WANDB_RUN_GROUP="ldm-tb"
WANDB_TAGS="ldm,256,a100,tb"

USE_EMA="1"

OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root)            DATA_ROOT="$2"; shift 2 ;;
    --img_size)             IMG_SIZE="$2"; shift 2 ;;
    --split)                SPLIT="$2"; shift 2 ;;
    --lr)                   LR="$2"; shift 2 ;;
    --weight_decay)         WEIGHT_DECAY="$2"; shift 2 ;;
    --grad_clip)            GRAD_CLIP="$2"; shift 2 ;;
    --epochs)               EPOCHS="$2"; shift 2 ;;
    --batch_per_device)     BATCH_PER_DEVICE="$2"; shift 2 ;;
    --sample_batch_size)    SAMPLE_BATCH_SIZE="$2"; shift 2 ;;
    --log_every)            LOG_EVERY="$2"; shift 2 ;;
    --sample_every)         SAMPLE_EVERY="$2"; shift 2 ;;
    --ae_ckpt_path)         AE_CKPT_PATH="$2"; shift 2 ;;
    --ae_config_path)       AE_CONFIG_PATH="$2"; shift 2 ;;
    --preencoded_latents_dir) PREENCODED_LATENTS_DIR="$2"; shift 2 ;;
    --ldm_ch_mults)         LDM_CH_MULTS="$2"; shift 2 ;;
    --ldm_num_res_blocks)   LDM_NUM_RES_BLOCKS="$2"; shift 2 ;;
    --wandb_project)        WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_group)          WANDB_RUN_GROUP="$2"; shift 2 ;;
    --wandb_tags)           WANDB_TAGS="$2"; shift 2 ;;
    --wandb_entity)         WANDB_ENTITY="$2"; shift 2 ;;
    --output_root)          OUTPUT_ROOT="$2"; shift 2 ;;
    --exp_name)             EXP_NAME="$2"; shift 2 ;;
    *)                      OTHER_ARGS+=("$1"); shift ;;
  esac
done

REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_PARENT=$(git rev-parse --short HEAD^ 2>/dev/null || echo "none")

WANDB_NAME="${WANDB_NAME:-ldm-${DISEASE_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-$WANDB_NAME}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/ldm-${DISEASE_NAME}-${TIMESTAMP}.log"

header "LDM ${DISEASE_NAME}"
status_line "Repo" "$REPO_ROOT"
status_line "Commit" "$GIT_HASH"
status_line "Branch" "$GIT_BRANCH"
status_line "Parent" "$GIT_PARENT"
status_line "Task" "$TASK"
status_line "Class Filter" "$CLASS_FILTER"
status_line "Batch/Device" "$BATCH_PER_DEVICE"
status_line "W&B" "$WANDB_NAME"
status_line "Log" "$LOG_FILE"

python run/ldm.py \
  --data_root "$DATA_ROOT" \
  --task "$TASK" \
  --split "$SPLIT" \
  --class_filter "$CLASS_FILTER" \
  --img_size "$IMG_SIZE" \
  --ae_ckpt_path "$AE_CKPT_PATH" \
  --ae_config_path "$AE_CONFIG_PATH" \
  --preencoded_latents_dir "$PREENCODED_LATENTS_DIR" \
  --ldm_ch_mults "$LDM_CH_MULTS" \
  --ldm_num_res_blocks "$LDM_NUM_RES_BLOCKS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --epochs "$EPOCHS" \
  --batch_per_device "$BATCH_PER_DEVICE" \
  --log_every "$LOG_EVERY" \
  --sample_every "$SAMPLE_EVERY" \
  --sample_batch_size "$SAMPLE_BATCH_SIZE" \
  --output_root "$OUTPUT_ROOT" \
  --exp_name "$EXP_NAME" \
  --run_name "$RUN_NAME" \
  --use_ema \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_tags "$WANDB_TAGS" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_name "$WANDB_NAME" \
  --git_hash "$GIT_HASH" \
  --git_branch "$GIT_BRANCH" \
  --git_parent "$GIT_PARENT" \
  "${OTHER_ARGS[@]}" | tee "$LOG_FILE"
