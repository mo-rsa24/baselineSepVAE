#!/usr/bin/env bash
# D4 — CheSS perceptual loss (15 epochs, resumes from D2 or D3)
#
# Purpose : Sharpen reconstruction texture using frozen CheSS feature L1.
#           CheSS is NOT in the encoder — only used as a perceptual feature
#           extractor. O1 metrics from D2/D3 must remain stable.
#
# NOTE: If you ran D3 (Pleural Thickening), resume from d3_three_class.
#       If skipping D3 (binary only), resume from d2_mi_disc.
#
# Prerequisite: D2 (or D3) gate passed.
# Resumes:      /workspace/runs_sepvae/d2_mi_disc/checkpoint_final.pkl  (default)
#
# Usage:
#   ./launchers/runpod/run_d4_perceptual.sh
#   RESUME=/workspace/runs_sepvae/d3_three_class/checkpoint_final.pkl \
#     ./launchers/runpod/run_d4_perceptual.sh

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv()    { printf "  ${CYN}%-26s${RST} %s\n" "$1" "$2"; }
ok()    { printf "${GRN}** %s${RST}\n" "$*"; }
die()   { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-/workspace/baselineSepVAE}"
DATA_DIR="${DATA_DIR:-/workspace/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/workspace/vinbigdata/cache_npy/train_filtered.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/runs_sepvae}"
RESUME="${RESUME:-${OUTPUT_ROOT}/d2_mi_disc/checkpoint_final.pkl}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/workspace/chess/pretrained_weights.pth.tar}"

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-15}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ── Loss weights ──────────────────────────────────────────────────────────────
WEIGHT_REC="${WEIGHT_REC:-1.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-1e-4}"
WEIGHT_MI_FACTOR="${WEIGHT_MI_FACTOR:-1.0}"
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.05}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"

# ── Logging ───────────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-d4_perceptual}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -d "$WORKDIR"          ]] || die "WORKDIR not found: $WORKDIR"
[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$RESUME"           ]] || die "Checkpoint not found: $RESUME — run D2/D3 first"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Environment ───────────────────────────────────────────────────────────────
banner "Environment"
cd "$WORKDIR"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0

set +u
source ~/.bashrc
conda activate "$ENV_NAME" 2>/dev/null || true
set -u
ok "Activated: ${ENV_NAME}"

kv "Phase"             "D4 — CheSS perceptual loss (O2 completion)"
kv "CheSS role"        "frozen perceptual loss ONLY — not in encoder"
kv "Perceptual weight" "$WEIGHT_PERCEPTUAL"
kv "Losses"            "MSE + KL + MI(${WEIGHT_MI_FACTOR}) + perceptual(${WEIGHT_PERCEPTUAL})"
kv "Batch / Epochs"    "${BATCH_SIZE} / ${EPOCHS}"
kv "Resume"            "$RESUME"
kv "Exp name"          "$EXP_NAME"
kv "Gate"              "Recon visually sharper; KL/MI stable (<10% regression from D2/D3)"

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
  run/train_sep_vae.py
  --dicom_dir            "$DATA_DIR"
  --csv_path             "$CSV_PATH"
  --use_cache

  --model_version        v2
  --use_bbox_cross_attn
  --img_size             "$IMG_SIZE"
  --z_channels_common    "$Z_COMMON"
  --z_channels_disease   "$Z_DISEASE"
  --attn_query_dim       "$ATTN_QUERY_DIM"
  --attn_heads           "$ATTN_HEADS"

  --batch_size           "$BATCH_SIZE"
  --epochs               "$EPOCHS"
  --num_workers          "$NUM_WORKERS"
  --seed                 "$SEED"
  --kl_warmup_epochs     0

  --lr_vae               "$LR_VAE"
  --lr_disc              "$LR_DISC"
  --weight_decay         "$WEIGHT_DECAY"
  --grad_clip            "$GRAD_CLIP"

  --weight_rec           "$WEIGHT_REC"
  --weight_kl_common     "$WEIGHT_KL_COMMON"
  --weight_kl_disease    "$WEIGHT_KL_DISEASE"
  --weight_mi_factor     "$WEIGHT_MI_FACTOR"
  --weight_bbox_attn     0.0
  --weight_perceptual    "$WEIGHT_PERCEPTUAL"
  --sigma_inactive       "$SIGMA_INACTIVE"

  --chess_checkpoint     "$CHESS_CHECKPOINT"
  --perceptual_only                          # use CheSS for perceptual loss only, not encoder

  --resume               "$RESUME"
  --output_root          "$OUTPUT_ROOT"
  --exp_name             "$EXP_NAME"
  --sample_every         "$SAMPLE_EVERY"
  --manifold_every       "$MANIFOLD_EVERY"
)

if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ── Launch ────────────────────────────────────────────────────────────────────
banner "Launching D4"
python -u "${ARGS[@]}"
ok "D4 complete — compare recon sharpness vs D2/D3 in W&B image grids"
