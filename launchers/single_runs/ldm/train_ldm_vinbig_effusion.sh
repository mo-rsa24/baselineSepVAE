#!/usr/bin/env bash
# Launcher: Unconditional LDM on SepVAE Pleural Effusion latents
#
# Pre-requisite: run the pre-encoding step first —
#
#   python scripts/preencode_sepvae_latents.py \
#     --sepvae_ckpt runs_sepvae/sepvae_independence_driven-20260220-085943/checkpoints/checkpoint_final.pkl \
#     --disease effusion \
#     --output_dir preencoded_latents/sepvae_effusion
#
# Then set LATENT_SCALE_FACTOR to the value printed by the script above,
# and call this launcher:
#
#   ./launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh full_train \
#     --latent_scale_factor <value from pre-encoding>
#
set -euo pipefail

if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

# --- Training mode (first positional arg) ---
export TRAINING_MODE="${1:-full_train}"
export ENV_NAME="jaxstack"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1

# --- Pre-encoded latents (no AE checkpoint needed) ---
export PREENCODED_LATENTS_DIR="${PREENCODED_LATENTS_DIR:-preencoded_latents/disentangle_effusion}"
export PREENCODED_MANIFEST="${PREENCODED_MANIFEST:-manifest.jsonl}"

# SepVAE latent shape overrides (2-ch effusion head, 64×64 spatial)
export LATENT_SIZE="64"
export VAE_Z_CHANNELS="2"
export LDM_Z_CHANNELS="2"

# ❗ Set this to the value printed by preencode_sepvae_latents.py
export LATENT_SCALE_FACTOR="${LATENT_SCALE_FACTOR:-1.0}"

# --- LDM Architecture ---
export LDM_BASE_CH="128"
export LDM_CH_MULTS="1,2,4,4"
export LDM_NUM_RES_BLOCKS="3"
export LDM_ATTN_RES="16"

# --- Optimizer ---
export LR="1e-4"
export WEIGHT_DECAY="1e-4"
export GRAD_CLIP="1.0"

# --- Training duration ---
export EPOCHS="1500"
export BATCH_PER_DEVICE="32"
export LOG_EVERY="100"
export SAMPLE_EVERY="9999"   # no sampling visuals (no AE decoder available)
export SAMPLE_BATCH_SIZE="16"

# --- EMA / precision ---
export USE_EMA="1"
export EMA_DECAY="0.999"
export USE_BFLOAT16="1"
export USE_REMAT="1"

# --- W&B ---
export WANDB="1"
export WANDB_PROJECT="cxr-ldm-composition"
export WANDB_ENTITY=""
export WANDB_RUN_GROUP="ldm-vinbig-effusion"
export WANDB_TAGS="ldm,effusion,sepvae,vinbig"

# --- SLURM ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-vinbig-effusion"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"

# --- Argument parsing loop ---
OTHER_ARGS=()
shift || true

while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)              export SLURM_PARTITION="$2"; shift 2 ;;
    --job-name)               export SLURM_JOB_NAME="$2"; shift 2 ;;
    --lr)                     export LR="$2"; shift 2 ;;
    --weight_decay)           export WEIGHT_DECAY="$2"; shift 2 ;;
    --ldm_base_ch)            export LDM_BASE_CH="$2"; shift 2 ;;
    --grad_clip)              export GRAD_CLIP="$2"; shift 2 ;;
    --epochs)                 export EPOCHS="$2"; shift 2 ;;
    --batch_per_device)       export BATCH_PER_DEVICE="$2"; shift 2 ;;
    --ldm_ch_mults)           export LDM_CH_MULTS="$2"; shift 2 ;;
    --ldm_num_res_blocks)     export LDM_NUM_RES_BLOCKS="$2"; shift 2 ;;
    --ldm_attn_res)           export LDM_ATTN_RES="$2"; shift 2 ;;
    --log_every)              export LOG_EVERY="$2"; shift 2 ;;
    --sample_every)           export SAMPLE_EVERY="$2"; shift 2 ;;
    --sample_batch_size)      export SAMPLE_BATCH_SIZE="$2"; shift 2 ;;
    --latent_scale_factor)    export LATENT_SCALE_FACTOR="$2"; shift 2 ;;
    --preencoded_latents_dir) export PREENCODED_LATENTS_DIR="$2"; shift 2 ;;
    --preencoded_manifest)    export PREENCODED_MANIFEST="$2"; shift 2 ;;
    --wandb_project)          export WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_name)             export WANDB_NAME="$2"; shift 2 ;;
    --wandb_tags)             export WANDB_TAGS="$2"; shift 2 ;;
    --wandb_group)            export WANDB_RUN_GROUP="$2"; shift 2 ;;
    --wandb_entity)           export WANDB_ENTITY="$2"; shift 2 ;;
    --time)                   export TIME_LIMIT="$2"; shift 2 ;;
    --use_bfloat16)           export USE_BFLOAT16="$2"; shift 2 ;;
    --use_remat)              export USE_REMAT="$2"; shift 2 ;;
    --overfit_one)            export OVERFIT_ONE="1"; shift ;;
    --overfit_k)              export OVERFIT_K="$2"; shift 2 ;;
    --repeat_len)             export REPEAT_LEN="$2"; shift 2 ;;
    *)                        OTHER_ARGS+=("$1"); shift ;;
  esac
done

REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_PARENT=$(git rev-parse --short HEAD^ 2>/dev/null || echo "none")

export WANDB_NAME="${WANDB_NAME:-${SLURM_JOB_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-$WANDB_NAME}"
export WANDB_TAGS="${WANDB_TAGS:-ldm,effusion,sepvae,${GIT_BRANCH},${GIT_HASH},parent-${GIT_PARENT}}"

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
  --exclude 'runs_ldm' \
  --exclude 'runs_sepvae' \
  --exclude 'preencoded_latents' \
  "$REPO_ROOT/" "$STAGING_DIR/"

status_line "✅ Snapshot" "Complete"

mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/runs_ldm"
export WORKDIR="${WORKDIR:-$STAGING_DIR}"

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
kv(){ printf "  ${CYN}%-26s${RST} %s\n" "$1" "$2"; }
rule(){ printf "${BLU}%.0s=" $(seq 1 60); printf "${RST}\n"; }

rule
printf "${BLD}${BLU}🫁 Submitting LDM-PleuralEffusion Training Job${RST}\n"
rule
kv "SLURM Job Name"       "${SLURM_JOB_NAME}"
kv "SLURM Partition"      "${SLURM_PARTITION}"
printf "\n"
kv "Preencoded Latents"   "${PREENCODED_LATENTS_DIR}"
kv "Disease"              "Pleural Effusion (SepVAE effusion head)"
kv "Latent Shape"         "64×64×2"
kv "Latent Scale Factor"  "${LATENT_SCALE_FACTOR}"
printf "\n"
kv "Model Base CH"        "${LDM_BASE_CH}"
kv "CH Multipliers"       "${LDM_CH_MULTS}"
kv "Attention Resolutions" "${LDM_ATTN_RES}"
printf "\n"
kv "Learning Rate"        "${LR}"
kv "Epochs"               "${EPOCHS}"
kv "Batch/Device"         "${BATCH_PER_DEVICE}"
kv "EMA"                  "${USE_EMA} (decay=${EMA_DECAY})"
kv "W&B Group"            "${WANDB_RUN_GROUP}"
rule

JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
  --job-name="$JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL \
  slurm_scripts/cxr_ldm.slurm "${OTHER_ARGS[@]}" | awk '{print $4}')

status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at"   "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"
