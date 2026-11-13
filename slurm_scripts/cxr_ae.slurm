#!/usr/bin/env bash
#SBATCH --job-name=cxr-ae
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster65,mscluster76
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1 # Total number of tasks

set -euo pipefail

# --- Pretty printing helpers ---
CYN=$(printf '\\033[36m'); BLU=$(printf '\\033[34m'); BLD=$(printf '\\033[1m');
RED=$(printf '\\033[31m'); GRN=$(printf '\\033[32m');
RST=$(printf '\\033[0m')
banner(){ printf "\\n${BLU}${BLD}== %s ==${RST}\\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\\n" "$1" "$2"; }
err(){ printf "${RED}!! %s${RST}\\n" "$*"; }
ok(){ printf "${GRN}** %s${RST}\\n" "$*"; }

# ----------------------------
# Defaults (override via en)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-run/autoencoder.py}"
DATA_ROOT="/datasets/mmolefe/cleaned"
ENV_NAME="${ENV_NAME:-jaxstack}" # Environment name for mamba

# Data & Debug
TASK="${TASK:-TB}"
SPLIT="${SPLIT:-train}"
IMG_SIZE="${IMG_SIZE:-256}"
CLASS_FILTER="${CLASS_FILTER:-None}"
OVERFIT_ONE="${OVERFIT_ONE:-0}"
OVERFIT_K="${OVERFIT_K:-0}"
REPEAT_LEN="${REPEAT_LEN:-500}"

# Architecture
BASE_CH="${BASE_CH:-128}"
CH_MULTS="${CH_MULTS:-1,2,4}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-3}"
ATTN_RES="${ATTN_RES:-16,8}"
Z_CHANNELS="${Z_CHANNELS:-1}"
EMBED_DIM="${EMBED_DIM:-None}"
DROPOUT="${DROPOUT:-0.0}"

# Loss
KL_WEIGHT="${KL_WEIGHT:-1.0e-6}"
PIXEL_WEIGHT="${PIXEL_WEIGHT:-1.0}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.05}"
DISC_START="${DISC_START:-5000}"
DISC_FACTOR="${DISC_FACTOR:-1.0}"
DISC_WEIGHT="${DISC_WEIGHT:-0.1}"
DISC_LAYERS="${DISC_LAYERS:-3}"
DISC_LOSS="${DISC_LOSS:-hinge}"

# Optimizer
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-100}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
SEED="${SEED:-0}"

# Logging & Checkpoints
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
EXP_NAME="${EXP_NAME:-cxr_ae}"
RUN_NAME="${RUN_NAME:-}"
RESUME_DIR="${RESUME_DIR:-}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-100}"

# WandB
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cxr-ae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_ID="${WANDB_ID:-}"

# ----------------------------
# Environment Setup
# ----------------------------
banner "Setup"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3



kv "Working directory" "$WORKDIR"
kv "Python path" "$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
echo "Commit: ${GIT_COMMIT_SHORT:-unknown} | Branch: ${GIT_BRANCH:-unknown} | W&B Name: ${WANDB_NAME:-unset}"
# Activate Conda/Mamba Environment
set +u
source ~/.bashrc
mamba activate "${ENV_NAME:?Must set ENV_NAME}"
set -u
ok "Activated micromamba environment: ${ENV_NAME}"


# ----------------------------
# Arguments for Python script
# ----------------------------
ARGS=(
  --data_root "$DATA_ROOT"
  --task "$TASK"
  --split "$SPLIT"
  --img_size "$IMG_SIZE"
  --class_filter "$CLASS_FILTER"
  --base_ch "$BASE_CH"
  --ch_mults "$CH_MULTS"
  --num_res_blocks "$NUM_RES_BLOCKS"
  --attn_res "$ATTN_RES"
  --z_channels "$Z_CHANNELS"
  --embed_dim "$EMBED_DIM"
  --dropout "$DROPOUT"
  --kl_weight "$KL_WEIGHT"
  --pixel_weight "$PIXEL_WEIGHT"
  --perceptual_weight "$PERCEPTUAL_WEIGHT"
  --disc_start "$DISC_START"
  --disc_factor "$DISC_FACTOR"
  --disc_weight "$DISC_WEIGHT"
  --disc_layers "$DISC_LAYERS"
  --disc_loss "$DISC_LOSS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --epochs "$EPOCHS"
  --batch_per_device "$BATCH_PER_DEVICE"
  --seed "$SEED"
  --sample_every "$SAMPLE_EVERY"
  --log_every "$LOG_EVERY"
  --output_root "$OUTPUT_ROOT"
  --exp_name "$EXP_NAME"
)

# Debugging
if [[ "$OVERFIT_ONE" == "1" ]]; then ARGS+=( --overfit_one --repeat_len "$REPEAT_LEN" );
elif (( OVERFIT_K > 0 )); then ARGS+=( --overfit_k "$OVERFIT_K" );
fi

# Optional run name / resume
[[ -n "$RUN_NAME" ]] && ARGS+=( --run_name "$RUN_NAME" )
[[ -n "$RESUME_DIR" ]] && ARGS+=( --resume_dir "$RESUME_DIR" )

# WandB
if [[ "$WANDB" == "1" ]];
then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "$WANDB_TAGS" ]] && ARGS+=( --wandb_tags "$WANDB_TAGS" )
  [[ -n "$WANDB_ID" ]] && ARGS+=( --wandb_id "$WANDB_ID" )
fi

# ----------------------------
# Info & Run
# ----------------------------
banner "AE Training Info"
kv "Python script"  "$PYFILE"
kv "Run name"       "$RUN_NAME"
kv "Task"           "$TASK"
kv "Image size"     "$IMG_SIZE"
kv "Base channels"  "$BASE_CH"
kv "Channel mults"  "\"$CH_MULTS\""
kv "Z channels"     "$Z_CHANNELS"
kv "KL weight"      "$KL_WEIGHT"
kv "Batch/device"   "$BATCH_PER_DEVICE"
kv "Epochs"         "$EPOCHS"
kv "W&B Enabled"    "$WANDB"

banner "AE Training Run"
srun python -u "$PYFILE" "$@" "${ARGS[@]}"
RC=$?

banner "Done (ExitCode: $RC)"
[[ $RC -eq 0 ]] && ok "Success." || { err "Failure (exit $RC)"; exit "$RC"; }