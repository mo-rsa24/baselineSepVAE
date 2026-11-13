#!/usr/bin/env bash
#SBATCH --exclude=mscluster65,mscluster76,mscluster59,mscluster48,mscluster82,mscluster75,mscluster72
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1             # Run a single main process (your python script)
#SBATCH --cpus-per-task=8      # Give that process 4 CPU cores

set -euo pipefail

# --- Pretty printing helpers ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m');
RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2";
}

# ----------------------------
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-run/ldm.py}"
DATA_ROOT="/home-mscluster/mmolefe/Playground/PhD/datasets/cleaned"
LATENT_SCALE_FACTOR="${LATENT_SCALE_FACTOR:-1.0}"
PREENCODED_LATENTS_DIR="${PREENCODED_LATENTS_DIR:-}"
PREENCODED_MANIFEST="${PREENCODED_MANIFEST:-}"
# Data & Debug
TASK="${TASK:-TB}"
SPLIT="${SPLIT:-train}"
IMG_SIZE="${IMG_SIZE:-256}"
DISEASE="${DISEASE:-0}" # Use 1 for diseased, 0 for normal
OVERFIT_ONE="${OVERFIT_ONE:-0}"
LDM_Z_CHANNELS="${LDM_Z_CHANNELS:-}"
OVERFIT_K="${OVERFIT_K:-0}"
REPEAT_LEN="${REPEAT_LEN:-100}"
LOG_EVERY="${LOG_EVERY:-100}"
# Autoencoder paths — required unless PREENCODED_LATENTS_DIR is set
AE_CKPT_PATH="${AE_CKPT_PATH:-}"
AE_CONFIG_PATH="${AE_CONFIG_PATH:-}"
if [[ -z "$PREENCODED_LATENTS_DIR" ]]; then
    : "${AE_CKPT_PATH:?Must set AE_CKPT_PATH when not using preencoded_latents_dir}"
    : "${AE_CONFIG_PATH:?Must set AE_CONFIG_PATH when not using preencoded_latents_dir}"
fi

# SepVAE latent overrides (used when AE paths are not provided)
LATENT_SIZE="${LATENT_SIZE:-}"
VAE_Z_CHANNELS="${VAE_Z_CHANNELS:-}"

# LDM Arch
LDM_BASE_CH="${LDM_BASE_CH:-128}"
LDM_CH_MULTS="${LDM_CH_MULTS:-1,2,4,4}"
LDM_NUM_RES_BLOCKS="${LDM_NUM_RES_BLOCKS:-3}"
LDM_ATTN_RES="${LDM_ATTN_RES:-16}"

# Training
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-1500}"
USE_EMA="${USE_EMA:-1}" # Default to off if not set
EMA_DECAY="${EMA_DECAY:-0.999}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-32}"
SAMPLE_EVERY="${SAMPLE_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-100}"
SEED="${SEED:-42}"
USE_BFLOAT16="${USE_BFLOAT16:-1}"
USE_REMAT="${USE_REMAT:-1}"

# Experiment naming / output
OUTPUT_ROOT="${OUTPUT_ROOT:-runs_ldm}"
EXP_NAME="${EXP_NAME:-cxr_ldm}"
RUN_NAME="${RUN_NAME:-}"
RESUME_DIR="${RESUME_DIR:-}"

# WandB
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cxr-ldm}"
WANDB_TAGS="${WANDB_TAGS:-slurm}"

# ----------------------------
# Environment & paths
# ----------------------------
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CPP_MIN_LOG_LEVEL=3
set +u
source ~/.bashrc
mamba activate "${ENV_NAME:?Must set ENV_NAME}"
set -u
# ----------------------------
# Normalize list-like vars
# ----------------------------
LDM_CH_MULTS="${LDM_CH_MULTS//:/,}"
LDM_ATTN_RES="${LDM_ATTN_RES//:/,}"
WANDB_TAGS="${WANDB_TAGS//:/,}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
WANDB_NAME="${WANDB_NAME:-}"
GIT_HASH="${GIT_HASH:-}"
GIT_BRANCH="${GIT_BRANCH:-}"
GIT_PARENT="${GIT_PARENT:-}"

# ----------------------------
# Build args array safely
# ----------------------------
ARGS=(
  --data_root "$DATA_ROOT" --task "$TASK" --split "$SPLIT" --img_size "$IMG_SIZE"
  --class_filter "$DISEASE"
  --latent_scale_factor "$LATENT_SCALE_FACTOR"
  --ldm_base_ch "$LDM_BASE_CH" --ldm_ch_mults "$LDM_CH_MULTS" --ldm_num_res_blocks "$LDM_NUM_RES_BLOCKS" --ldm_attn_res "$LDM_ATTN_RES"
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --grad_clip "$GRAD_CLIP"
  --epochs "$EPOCHS" --batch_per_device "$BATCH_PER_DEVICE"
  --seed "$SEED" --sample_every "$SAMPLE_EVERY" --log_every "$LOG_EVERY" --output_root "$OUTPUT_ROOT" --exp_name "$EXP_NAME"
)

# AE paths: only add when provided
[[ -n "$AE_CKPT_PATH" ]]   && ARGS+=( --ae_ckpt_path   "$AE_CKPT_PATH" )
[[ -n "$AE_CONFIG_PATH" ]] && ARGS+=( --ae_config_path "$AE_CONFIG_PATH" )

# SepVAE latent overrides: only add when provided
[[ -n "$LATENT_SIZE" ]]    && ARGS+=( --latent_size    "$LATENT_SIZE" )
[[ -n "$VAE_Z_CHANNELS" ]] && ARGS+=( --vae_z_channels "$VAE_Z_CHANNELS" )

if [[ -n "$LDM_Z_CHANNELS" ]]; then
  ARGS+=( --ldm_z_channels "$LDM_Z_CHANNELS" )
fi



# --- Data Staging to Local SSD ---
LOCAL_SCRATCH="/tmp/${SLURM_JOB_ID}_latents"
mkdir -p "$LOCAL_SCRATCH"

if [[ -n "$PREENCODED_LATENTS_DIR" ]]; then
    echo "🚚 Staging: Rsyncing latents to local SSD ($LOCAL_SCRATCH)..."
    rsync -aq "$PREENCODED_LATENTS_DIR/" "$LOCAL_SCRATCH/"
    export PREENCODED_LATENTS_DIR="$LOCAL_SCRATCH"
fi
if [[ -n "$PREENCODED_MANIFEST" ]]; then
  ARGS+=( --preencoded_manifest "$PREENCODED_MANIFEST" )
fi

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=true --xla_gpu_graph_level=2"
# Debugging
if [[ "$OVERFIT_ONE" == "1" || "$OVERFIT_ONE" == "true" ]]; then ARGS+=( --overfit_one --repeat_len "$REPEAT_LEN" );
elif (( OVERFIT_K > 0 )); then ARGS+=( --overfit_k "$OVERFIT_K" );
fi

# --- Add EMA arguments if enabled ---
if [[ "$USE_EMA" == "1" ]]; then
  ARGS+=(--use_ema --ema_decay "$EMA_DECAY")
fi
if [[ "$USE_BFLOAT16" == "1" ]]; then
  ARGS+=(--use_bfloat16)
fi
if [[ "$USE_REMAT" == "1" ]]; then
  ARGS+=(--use_remat)
fi


# Optional run name / resume
[[ -n "$RUN_NAME" ]] && ARGS+=( --run_name "$RUN_NAME" )
[[ -n "$RESUME_DIR" ]] && ARGS+=( --resume_dir "$RESUME_DIR" )

# WandB
if [[ "$WANDB" == "1" ]];
then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_TAGS" ]] && ARGS+=( --wandb_tags "$WANDB_TAGS" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "$WANDB_RUN_GROUP" ]] && ARGS+=( --wandb_group "$WANDB_RUN_GROUP" )
  [[ -n "$WANDB_NAME" ]] && ARGS+=( --wandb_name "$WANDB_NAME" )
  [[ -n "$GIT_HASH" ]] && ARGS+=( --git_hash "$GIT_HASH" )
  [[ -n "$GIT_BRANCH" ]] && ARGS+=( --git_branch "$GIT_BRANCH" )
  [[ -n "$GIT_PARENT" ]] && ARGS+=( --git_parent "$GIT_PARENT" )
fi

# ----------------------------
# Info & Run
# ----------------------------
banner "SLURM Job Info"
kv "JOB_ID"         "${SLURM_JOB_ID:-<none>}"
kv "Node"           "$(hostname)"

banner "LDM Training Info"
kv "PYFILE"         "$PYFILE"
kv "TASK"           "$TASK"
kv "DATASET"        "$([[ "$DISEASE" == 1 ]] && echo "Diseased" || echo "Normal")"
kv "EPOCHS"         "$EPOCHS"
kv "BATCHxDEVS"     "$BATCH_PER_DEVICE x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
kv "AE_CHECKPOINT"  "${AE_CKPT_PATH:-<none — pre-encoded latents mode>}"
kv "Z_CHANNELS"     "${LDM_Z_CHANNELS:-Default}"
kv "LATENT_SIZE"    "${LATENT_SIZE:-<from AE config>}"
kv "VAE_Z_CHANNELS" "${VAE_Z_CHANNELS:-<from AE config>}"
kv "OVERFIT"        "(of1=$OVERFIT_ONE, tinyK=$OVERFIT_K)"
kv "W&B"            "$WANDB (${WANDB_PROJECT}) tags=[$WANDB_TAGS]"
kv "EMA Enabled" "${USE_EMA}"
if [[ "$USE_EMA" == "1" ]]; then
  kv "EMA Decay" "${EMA_DECAY}"
fi
kv "bfloat16 Enabled" "${USE_BFLOAT16}"
kv "Remat Enabled" "${USE_REMAT}"
banner "Run"
kv "Preencoded Dir" "${PREENCODED_LATENTS_DIR:-<none>}"
kv "Preencoded Manifest" "${PREENCODED_MANIFEST:-<none>}"
kv "CPUs Allocated:" "${SLURM_CPUS_PER_TASK}"
srun python -u "$PYFILE" "${ARGS[@]}"

# --- Cleanup ---
echo "🧹 Cleaning up local SSD..."
rm -rf "$LOCAL_SCRATCH"