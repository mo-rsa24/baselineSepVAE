#!/usr/bin/env bash
#SBATCH --job-name=sep-vae
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster65,mscluster76,mscluster59,mscluster48,mscluster82,mscluster75,mscluster72
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
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-run/train_sep_vae.py}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# Data
DICOM_DIR="${DICOM_DIR:-/datasets/mmolefe/vinbigdata/train}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/train.csv}"
IMG_SIZE="${IMG_SIZE:-512}"

# Model architecture
Z_CHANNELS_COMMON="${Z_CHANNELS_COMMON:-4}"
Z_CHANNELS_DISEASE="${Z_CHANNELS_DISEASE:-2}"
FROZEN_BACKBONE="${FROZEN_BACKBONE:-1}"
USE_FPN="${USE_FPN:-0}"
FPN_CHANNELS="${FPN_CHANNELS:-512}"
UNFREEZE_FROM="${UNFREEZE_FROM:-}"

# CheSS weights
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
CHESS_CONVERTED="${CHESS_CONVERTED:-}"

# Loss weights
WEIGHT_REC="${WEIGHT_REC:-1.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-1e-4}"
WEIGHT_NULL="${WEIGHT_NULL:-1e-3}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"
FREE_BITS="${FREE_BITS:-0.0}"
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.0}"
WEIGHT_ADVERSARIAL="${WEIGHT_ADVERSARIAL:-0.0}"
DISC_START_EPOCH="${DISC_START_EPOCH:-10}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-0}"
WEIGHT_ORTHOGONALITY="${WEIGHT_ORTHOGONALITY:-0.0}"
LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-0}"

# R5a: Paired contrastive loss
USE_CONTRASTIVE="${USE_CONTRASTIVE:-0}"
WEIGHT_CONTRASTIVE="${WEIGHT_CONTRASTIVE:-0.1}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.5}"

# R7: Disease prototype label attention
USE_LABEL_ATTENTION="${USE_LABEL_ATTENTION:-0}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"

# R11: Disease discriminability classifiers
USE_DISEASE_CLF="${USE_DISEASE_CLF:-0}"
WEIGHT_DISEASE_CLF="${WEIGHT_DISEASE_CLF:-0.1}"
LR_DISEASE_CLF="${LR_DISEASE_CLF:-1e-4}"

# Optimizer
LR_VAE="${LR_VAE:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_PATCH_DISC="${LR_PATCH_DISC:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Training (RTX 4090 24GB defaults)
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"

# Logging & Checkpoints
OUTPUT_ROOT="${OUTPUT_ROOT:-runs_sepvae}"
EXP_NAME="${EXP_NAME:-sepvae_chess}"
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"

# Verbose diagnostics
VERBOSE_BACKBONE="${VERBOSE_BACKBONE:-1}"
VERBOSE_N_SAMPLES="${VERBOSE_N_SAMPLES:-12}"

# WandB
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sepvae-chess}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

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
  --dicom_dir "$DICOM_DIR"
  --csv_path "$CSV_PATH"
  --img_size "$IMG_SIZE"
  --z_channels_common "$Z_CHANNELS_COMMON"
  --z_channels_disease "$Z_CHANNELS_DISEASE"
  --chess_checkpoint "$CHESS_CHECKPOINT"
  --weight_rec "$WEIGHT_REC"
  --weight_kl_common "$WEIGHT_KL_COMMON"
  --weight_kl_disease "$WEIGHT_KL_DISEASE"
  --weight_null "$WEIGHT_NULL"
  --sigma_inactive "$SIGMA_INACTIVE"
  --free_bits "$FREE_BITS"
  --weight_perceptual "$WEIGHT_PERCEPTUAL"
  --weight_adversarial "$WEIGHT_ADVERSARIAL"
  --disc_start_epoch "$DISC_START_EPOCH"
  --kl_warmup_epochs "$KL_WARMUP_EPOCHS"
  --weight_orthogonality "$WEIGHT_ORTHOGONALITY"
  --lr_decay_epochs "$LR_DECAY_EPOCHS"
  --weight_contrastive "$WEIGHT_CONTRASTIVE"
  --contrastive_margin "$CONTRASTIVE_MARGIN"
  --fpn_channels "$FPN_CHANNELS"
  --lr_vae "$LR_VAE"
  --lr_backbone "$LR_BACKBONE"
  --lr_patch_disc "$LR_PATCH_DISC"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --output_root "$OUTPUT_ROOT"
  --exp_name "$EXP_NAME"
  --log_every "$LOG_EVERY"
  --save_every "$SAVE_EVERY"
  --verbose_n_samples "$VERBOSE_N_SAMPLES"
)

# Frozen backbone (default on)
[[ "$FROZEN_BACKBONE" == "1" ]] && ARGS+=( --frozen_backbone )

# FPN and partial unfreezing
[[ "$USE_FPN" == "1" ]] && ARGS+=( --use_fpn )
[[ -n "$UNFREEZE_FROM" ]] && ARGS+=( --unfreeze_from "$UNFREEZE_FROM" )

# Pre-converted CheSS weights (skip PyTorch conversion)
[[ -n "$CHESS_CONVERTED" ]] && ARGS+=( --chess_converted "$CHESS_CONVERTED" )

# R5a: Paired contrastive loss
[[ "$USE_CONTRASTIVE" == "1" ]] && ARGS+=( --use_contrastive )

# R7: Disease prototype label attention
[[ "$USE_LABEL_ATTENTION" == "1" ]] && ARGS+=( --use_label_attention --attn_query_dim "$ATTN_QUERY_DIM" )

# R11: Disease discriminability classifiers
[[ "$USE_DISEASE_CLF" == "1" ]] && ARGS+=( --use_disease_clf --weight_disease_clf "$WEIGHT_DISEASE_CLF" --lr_disease_clf "$LR_DISEASE_CLF" )

# Verbose backbone diagnostics
[[ "$VERBOSE_BACKBONE" == "1" ]] && ARGS+=( --verbose_backbone )

# WandB
if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ----------------------------
# Info & Run
# ----------------------------
banner "SepVAE Training Info"
kv "Python script"    "$PYFILE"
kv "DICOM dir"        "$DICOM_DIR"
kv "Image size"       "$IMG_SIZE"
kv "Batch size"       "$BATCH_SIZE (x3 = $((BATCH_SIZE * 3)) images)"
kv "z common"         "$Z_CHANNELS_COMMON"
kv "z disease"        "$Z_CHANNELS_DISEASE"
kv "LR (VAE)"         "$LR_VAE"
kv "Loss: rec"        "$WEIGHT_REC"
kv "Loss: KL common"  "$WEIGHT_KL_COMMON"
kv "Loss: KL disease"  "$WEIGHT_KL_DISEASE"
kv "Loss: null"       "$WEIGHT_NULL"
kv "Loss: perceptual" "$WEIGHT_PERCEPTUAL"
kv "Loss: adversarial" "$WEIGHT_ADVERSARIAL"
kv "Free-bits"        "$FREE_BITS"
kv "FPN"              "$USE_FPN (channels=$FPN_CHANNELS)"
kv "Unfreeze from"    "${UNFREEZE_FROM:-none}"
kv "Epochs"           "$EPOCHS"
kv "Save every"       "$SAVE_EVERY"
kv "W&B Enabled"      "$WANDB"
kv "Verbose backbone"  "$VERBOSE_BACKBONE"
kv "LR decay from ep" "$LR_DECAY_EPOCHS"
kv "R5a contrastive"  "$USE_CONTRASTIVE (w=$WEIGHT_CONTRASTIVE, margin=$CONTRASTIVE_MARGIN)"
kv "R7 label-attn"    "$USE_LABEL_ATTENTION (query_dim=$ATTN_QUERY_DIM)"
kv "R11 disease-clf"  "$USE_DISEASE_CLF (w=$WEIGHT_DISEASE_CLF, lr=$LR_DISEASE_CLF)"

banner "SepVAE Training Run"
srun python -u "$PYFILE" "$@" "${ARGS[@]}"
RC=$?

banner "Done (ExitCode: $RC)"
[[ $RC -eq 0 ]] && ok "Success." || { err "Failure (exit $RC)"; exit "$RC"; }
