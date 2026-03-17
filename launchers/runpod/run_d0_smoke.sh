#!/usr/bin/env bash
# D0 — Smoke test (2–3 epochs)
#
# Purpose : Confirm no OOM, KL is manageable at step 0 (from-scratch backbone),
#           recon grid shows image structure, W&B connects.
#
# Key diagnostic: if KL_total is < 50 000 at step 0, the CheSS backbone was
# the root cause of the V1 blowup. Proceed to D1 if gate passes.
#
# Usage (RunPod pod shell):
#   chmod +x launchers/runpod/run_d0_smoke.sh
#   ./launchers/runpod/run_d0_smoke.sh
#
# Override any default with an env var before calling, e.g.:
#   BATCH_SIZE=4 ./launchers/runpod/run_d0_smoke.sh

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

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ── Loss weights (defaults — all extras off for smoke test) ───────────────────
WEIGHT_REC="${WEIGHT_REC:-1.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-5e-5}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"

# ── Logging ───────────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-d0_smoke_v2}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:--1}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# ── Validate paths ────────────────────────────────────────────────────────────
[[ -d "$WORKDIR"   ]] || die "WORKDIR not found: $WORKDIR"
[[ -d "$DATA_DIR"  ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"  ]] || die "CSV not found: $CSV_PATH"

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

kv "Host"              "$(hostname)"
kv "GPU"               "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
kv "Phase"             "D0 — smoke test (no CheSS encoder)"
kv "Model"             "SepVAEV2  img=${IMG_SIZE}px  z_common=${Z_COMMON}  z_disease=${Z_DISEASE}"
kv "Backbone"          "ResNet-50 from scratch + self-attn (no bbox cross-attn yet)"
kv "Batch / Epochs"    "${BATCH_SIZE} / ${EPOCHS}"
kv "Exp name"          "$EXP_NAME"
kv "Output"            "$OUTPUT_ROOT"
kv "W&B"               "${WANDB_PROJECT} (enabled=${WANDB})"

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
  run/train_sep_vae.py
  --dicom_dir            "$DATA_DIR"
  --csv_path             "$CSV_PATH"
  --use_cache

  --model_version        v2
  --img_size             "$IMG_SIZE"
  --z_channels_common    "$Z_COMMON"
  --z_channels_disease   "$Z_DISEASE"
  --attn_query_dim       "$ATTN_QUERY_DIM"
  --attn_heads           "$ATTN_HEADS"

  --batch_size           "$BATCH_SIZE"
  --epochs               "$EPOCHS"
  --num_workers          "$NUM_WORKERS"
  --seed                 "$SEED"

  --lr_vae               "$LR_VAE"
  --lr_disc              "$LR_DISC"
  --weight_decay         "$WEIGHT_DECAY"
  --grad_clip            "$GRAD_CLIP"

  --weight_rec           "$WEIGHT_REC"
  --weight_kl_common     "$WEIGHT_KL_COMMON"
  --weight_kl_disease    "$WEIGHT_KL_DISEASE"
  --weight_mi_factor     0.0
  --weight_bbox_attn     0.0
  --weight_perceptual    0.0
  --sigma_inactive       "$SIGMA_INACTIVE"

  --output_root          "$OUTPUT_ROOT"
  --exp_name             "$EXP_NAME"
  --save_every           "$SAVE_EVERY"
  --sample_every         "$SAMPLE_EVERY"
  --manifold_every       "$MANIFOLD_EVERY"
)

if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ── Launch ────────────────────────────────────────────────────────────────────
banner "Launching D0 smoke test"
python -u "${ARGS[@]}"
ok "D0 complete — check KL at step 0 and recon grid in W&B before proceeding to D1"
