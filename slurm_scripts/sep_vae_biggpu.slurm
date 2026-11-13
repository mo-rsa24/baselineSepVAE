#!/usr/bin/env bash
#SBATCH --job-name=sepvae-biggpu
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster106
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/biggpu-%j.out
#SBATCH --error=logs/biggpu-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
RED=$(printf '\033[31m'); GRN=$(printf '\033[32m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-24s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}** %s${RST}\n" "$*"; }
err(){ printf "${RED}!! %s${RST}\n" "$*"; }

# ----------------------------
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
ENV_NAME="${ENV_NAME:-jaxstack}"

DICOM_DIR="${DICOM_DIR:-/datasets/mmolefe/vinbigdata/train}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/train.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
CHESS_CONVERTED="${CHESS_CONVERTED:-}"

BATCH_SIZE_BASELINE="${BATCH_SIZE_BASELINE:-6}"
BATCH_SIZE_FULL="${BATCH_SIZE_FULL:-6}"
EPOCHS="${EPOCHS:-150}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
HALF_PRECISION="${HALF_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SAVE_EVERY="${SAVE_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-100}"

WEIGHT_REC="${WEIGHT_REC:-1.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-5e-5}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.05}"
FREE_BITS="${FREE_BITS:-0.0}"
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.1}"
WEIGHT_ADVERSARIAL="${WEIGHT_ADVERSARIAL:-0.05}"
DISC_START_EPOCH="${DISC_START_EPOCH:-5}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-10}"

# R11: Disease discriminability classifiers (always on in biggpu runs)
WEIGHT_DISEASE_CLF="${WEIGHT_DISEASE_CLF:-0.4}"
LR_DISEASE_CLF="${LR_DISEASE_CLF:-1e-4}"

# R12: Spatial attention bbox supervision (full run only, requires --use_label_attention)
WEIGHT_SPATIAL_ATTN="${WEIGHT_SPATIAL_ATTN:-0.2}"

LR_VAE="${LR_VAE:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_PATCH_DISC="${LR_PATCH_DISC:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sepvae-disentangle}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-biggpu}"

# ----------------------------
# Environment
# ----------------------------
banner "Setup"
cd "$WORKDIR"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}"
kv "Working dir" "$WORKDIR"
kv "Commit"      "${GIT_COMMIT_SHORT:-unknown} (${GIT_BRANCH:-unknown})"

# ----------------------------
# Common args builder
# ----------------------------
common_args() {
  local exp=$1 null=$2 ortho=$3 lr_decay=$4 bs=$5
  local -a args=(
    run/train_sep_vae.py
    --dicom_dir        "$DICOM_DIR"
    --csv_path         "$CSV_PATH"
    --chess_checkpoint "$CHESS_CHECKPOINT"
    --img_size         512
    --z_channels_common  4
    --z_channels_disease 4
    --frozen_backbone
    --batch_size       "$bs"
    --epochs           "$EPOCHS"
    --num_workers      "$NUM_WORKERS"
    --seed             "$SEED"
    --weight_rec       "$WEIGHT_REC"
    --weight_kl_common "$WEIGHT_KL_COMMON"
    --weight_kl_disease "$WEIGHT_KL_DISEASE"
    --weight_null      "$null"
    --weight_orthogonality "$ortho"
    --sigma_inactive   "$SIGMA_INACTIVE"
    --free_bits        "$FREE_BITS"
    --weight_perceptual "$WEIGHT_PERCEPTUAL"
    --weight_adversarial "$WEIGHT_ADVERSARIAL"
    --disc_start_epoch "$DISC_START_EPOCH"
    --kl_warmup_epochs "$KL_WARMUP_EPOCHS"
    --lr_vae           "$LR_VAE"
    --lr_decay_epochs  "$lr_decay"
    --lr_backbone      "$LR_BACKBONE"
    --lr_patch_disc    "$LR_PATCH_DISC"
    --weight_decay     "$WEIGHT_DECAY"
    --grad_clip        "$GRAD_CLIP"
    --output_root      runs_sepvae
    --exp_name         "$exp"
    --log_every        "$LOG_EVERY"
    --save_every       "$SAVE_EVERY"
    --half_precision   "$HALF_PRECISION"
  )
  [[ "$GRADIENT_CHECKPOINTING" == "1" ]] && args+=( --gradient_checkpointing )
  [[ -n "$CHESS_CONVERTED" ]]            && args+=( --chess_converted "$CHESS_CONVERTED" )
  # R11: always enabled in biggpu runs
  args+=( --use_disease_clf --weight_disease_clf "$WEIGHT_DISEASE_CLF" --lr_disease_clf "$LR_DISEASE_CLF" )
  if [[ "$WANDB" == "1" ]]; then
    args+=( --wandb --wandb_project "$WANDB_PROJECT" )
    [[ -n "$WANDB_ENTITY" ]] && args+=( --wandb_entity "$WANDB_ENTITY" )
  fi
  printf '%s\n' "${args[@]}"
}

# ----------------------------
# Launch
# ----------------------------
banner "Launching two training jobs"
kv "GPU 0" "sepvae_baseline_fixed  [R1+R2+R3+R4+R11]             bs=${BATCH_SIZE_BASELINE}"
kv "GPU 1" "sepvae_full            [R1+R2+R3+R4+R5a+R7+R11]     bs=${BATCH_SIZE_FULL}"
kv "R11 disease-clf" "w=${WEIGHT_DISEASE_CLF}  lr=${LR_DISEASE_CLF}"
kv "Precision"   "$HALF_PRECISION + gradient_checkpointing=$GRADIENT_CHECKPOINTING"
kv "Epochs"      "$EPOCHS"

# Job A — baseline_fixed on GPU 0
mapfile -t ARGS_A < <(common_args "sepvae_baseline_fixed" "0.02" "0.15" "60" "$BATCH_SIZE_BASELINE")
CUDA_VISIBLE_DEVICES=0 python -u "${ARGS_A[@]}" \
  > logs/biggpu-baseline-${SLURM_JOB_ID}.log 2>&1 &
PID_A=$!
ok "GPU 0  PID $PID_A  →  sepvae_baseline_fixed  (bs=${BATCH_SIZE_BASELINE})"

# Job B — sepvae_full on GPU 1
mapfile -t ARGS_B < <(common_args "sepvae_full" "0.05" "0.15" "80" "$BATCH_SIZE_FULL")
ARGS_B+=(
  --use_contrastive  --weight_contrastive 0.05  --contrastive_margin 0.5
  --use_label_attention --attn_query_dim 256
  --weight_spatial_attn "$WEIGHT_SPATIAL_ATTN"
  --use_fpn --fpn_channels 512 --unfreeze_from layer4
)
CUDA_VISIBLE_DEVICES=1 python -u "${ARGS_B[@]}" \
  > logs/biggpu-full-${SLURM_JOB_ID}.log 2>&1 &
PID_B=$!
ok "GPU 1  PID $PID_B  →  sepvae_full  (bs=${BATCH_SIZE_FULL})"

banner "Waiting for both jobs (Job ID: ${SLURM_JOB_ID})"
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?

banner "Results"
[[ $RC_A -eq 0 ]] && ok "baseline_fixed  PASSED (exit 0)" || err "baseline_fixed  FAILED (exit $RC_A)"
[[ $RC_B -eq 0 ]] && ok "sepvae_full     PASSED (exit 0)" || err "sepvae_full     FAILED (exit $RC_B)"

[[ $RC_A -eq 0 && $RC_B -eq 0 ]] && exit 0 || exit 1
