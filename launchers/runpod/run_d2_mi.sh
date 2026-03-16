#!/usr/bin/env bash
# D2 — FactorVAE MI discriminator (25 epochs, resumes from D1)
#
# Purpose : Push z_common ⊥ z_cardio via FactorVAE density-ratio estimation.
#           With bbox cross-attention already grounding the disease head
#           spatially, the discriminator reinforces latent independence.
#
# Prerequisite: D1 gate passed (KL stable, attn maps localising).
# Resumes:      d1_recon_bbox_xattn/checkpoint_final.pkl
#
# Usage:
#   ./launchers/runpod/run_d2_mi.sh
#   RESUME=/workspace/runs_sepvae/my_d1_run/checkpoint_final.pkl \
#     ./launchers/runpod/run_d2_mi.sh

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
RESUME="${RESUME:-${OUTPUT_ROOT}/d1_recon_bbox_xattn/checkpoint_final.pkl}"

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-25}"
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
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"

# ── Logging ───────────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-d2_mi_disc}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -d "$WORKDIR"  ]] || die "WORKDIR not found: $WORKDIR"
[[ -d "$DATA_DIR" ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH" ]] || die "CSV not found: $CSV_PATH"
[[ -f "$RESUME"   ]] || die "D1 checkpoint not found: $RESUME — run D1 first"

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

kv "Phase"             "D2 — FactorVAE MI discriminator (O1 step 1)"
kv "Losses"            "MSE + KL + MI(w=${WEIGHT_MI_FACTOR})  (bbox_loss=0, perceptual=0)"
kv "Batch / Epochs"    "${BATCH_SIZE} / ${EPOCHS}"
kv "Resume"            "$RESUME"
kv "Exp name"          "$EXP_NAME"
kv "Gate"              "MI disc accuracy → 0.5; ||z_disease|| larger for Cardiomegaly"

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
  --weight_perceptual    0.0
  --sigma_inactive       "$SIGMA_INACTIVE"

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
banner "Launching D2"
python -u "${ARGS[@]}"
ok "D2 complete — check MI disc accuracy and manifold separation in W&B"
