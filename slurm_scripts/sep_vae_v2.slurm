#!/usr/bin/env bash
# SepVAE V2 — parameterised phase runner for the cluster.
#
# Runs any phase D0–D4 via the PHASE environment variable.
# Each phase is a distinct SBATCH submission so the checkpoint chain is
# explicit and auditable.
#
# Submit examples:
#   PHASE=d0 sbatch slurm_scripts/sep_vae_v2.slurm
#   PHASE=d1 sbatch slurm_scripts/sep_vae_v2.slurm
#   PHASE=d2 sbatch slurm_scripts/sep_vae_v2.slurm
#   PHASE=d4 WEIGHT_PERCEPTUAL=0.05 sbatch slurm_scripts/sep_vae_v2.slurm
#
# Override any default by exporting the variable before sbatch:
#   export BATCH_SIZE=16; PHASE=d1 sbatch slurm_scripts/sep_vae_v2.slurm

#SBATCH --job-name=sepvae-v2
#SBATCH --exclude=mscluster44,mscluster65,mscluster76,mscluster59,mscluster48,mscluster82,mscluster75,mscluster72
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-v2-%j.out
#SBATCH --error=logs/sepvae-v2-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv()    { printf "  ${CYN}%-26s${RST} %s\n" "$1" "$2"; }
ok()    { printf "${GRN}** %s${RST}\n" "$*"; }
die()   { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Phase selection ───────────────────────────────────────────────────────────
PHASE="${PHASE:-d0}"

case "$PHASE" in
  d0|d1|d2|d4|d5) ;;
  *) die "Unknown PHASE='${PHASE}'. Valid: d0 d1 d2 d4 d5" ;;
esac

# ── Shared paths ──────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$PWD}"
ENV_NAME="${ENV_NAME:-jaxstack}"
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"

# ── Shared model defaults ─────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"

# ── Shared training defaults ──────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"

# ── Logging defaults ──────────────────────────────────────────────────────────
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Per-phase config ──────────────────────────────────────────────────────────
case "$PHASE" in
  d0)
    # Smoke test — verify pipeline end-to-end, no orthogonality pressure
    EXP_NAME="${EXP_NAME:-d0_smoke_v2}"
    EPOCHS="${EPOCHS:-5}"
    KL_WARMUP_EPOCHS=0
    WEIGHT_REC=1.0; WEIGHT_KL_COMMON=1e-4; WEIGHT_KL_DISEASE=5e-5
    WEIGHT_MI_FACTOR=0.0; WEIGHT_BBOX_ATTN=0.0; WEIGHT_PERCEPTUAL=0.0
    BATCH_SIZE="${BATCH_SIZE:-16}"
    USE_BBOX_CROSS_ATTN=0
    RESUME=""
    SAMPLE_EVERY=1; MANIFOLD_EVERY=-1
    ;;
  d1)
    # Reconstruction + KL warm-up + bbox cross-attn alignment; no MI yet.
    # 30 epochs gives the encoder time to learn CXR features from scratch and the
    # BboxCrossAttnHead to reliably localise cardiomegaly before the MI disc is added.
    EXP_NAME="${EXP_NAME:-d1_recon_bbox_xattn}"
    EPOCHS="${EPOCHS:-30}"
    KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-5}"
    WEIGHT_REC=1.0; WEIGHT_KL_COMMON=1e-4; WEIGHT_KL_DISEASE=5e-5
    WEIGHT_MI_FACTOR=0.0; WEIGHT_BBOX_ATTN=0.0; WEIGHT_PERCEPTUAL=0.0
    BATCH_SIZE="${BATCH_SIZE:-16}"
    USE_BBOX_CROSS_ATTN=1
    RESUME="${RESUME:-}"
    ;;
  d2)
    # Add FactorVAE MI pressure.  35 epochs lets the discriminator stabilise and
    # orthogonal separation emerge before we add the perceptual loss in D4.
    EXP_NAME="${EXP_NAME:-d2_mi_disc}"
    EPOCHS="${EPOCHS:-35}"
    KL_WARMUP_EPOCHS=0
    WEIGHT_REC=1.0; WEIGHT_KL_COMMON=1e-4; WEIGHT_KL_DISEASE=1e-4
    WEIGHT_MI_FACTOR="${WEIGHT_MI_FACTOR:-1.0}"
    WEIGHT_BBOX_ATTN=0.0; WEIGHT_PERCEPTUAL=0.0
    BATCH_SIZE="${BATCH_SIZE:-16}"
    USE_BBOX_CROSS_ATTN=1
    RESUME="${RESUME:-}"   # set by run_curriculum.sh; no hardcoded stale path
    ;;
  d4)
    # CheSS perceptual loss refines reconstruction sharpness.
    # 20 epochs is enough since the encoder/decoder are already well-trained.
    # batch=12: perceptual backbone adds ~1 GB vs D1/D2.
    EXP_NAME="${EXP_NAME:-d4_perceptual}"
    EPOCHS="${EPOCHS:-20}"
    KL_WARMUP_EPOCHS=0
    WEIGHT_REC=1.0; WEIGHT_KL_COMMON=1e-4; WEIGHT_KL_DISEASE=1e-4
    WEIGHT_MI_FACTOR="${WEIGHT_MI_FACTOR:-1.0}"
    WEIGHT_BBOX_ATTN=0.0
    WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.05}"
    BATCH_SIZE="${BATCH_SIZE:-12}"
    USE_BBOX_CROSS_ATTN=1
    RESUME="${RESUME:-}"   # set by run_curriculum.sh
    ;;
  d5)
    EXP_NAME="${EXP_NAME:-d5_recon}"
    EPOCHS="${EPOCHS:-60}"
    KL_WARMUP_EPOCHS=0
    WEIGHT_REC="${WEIGHT_REC:-2.0}"
    WEIGHT_KL_COMMON=1e-4; WEIGHT_KL_DISEASE=1e-4
    WEIGHT_MI_FACTOR="${WEIGHT_MI_FACTOR:-1.0}"
    WEIGHT_BBOX_ATTN="${WEIGHT_BBOX_ATTN:-0.1}"
    WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.05}"
    WEIGHT_GAN="${WEIGHT_GAN:-0.5}"
    WEIGHT_TV="${WEIGHT_TV:-1e-3}"
    GAN_START_STEP="${GAN_START_STEP:-5000}"
    LR_PATCH_DISC="${LR_PATCH_DISC:-1e-4}"
    # 24 GB (RTX 3090): batch=12 keeps decoder 256×256×128 activations + perceptual
    # backbone within ~20 GB. Use BATCH_SIZE=16 override only if first steps stay < 22 GB.
    BATCH_SIZE="${BATCH_SIZE:-12}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    USE_BBOX_CROSS_ATTN=1
    RESUME="${RESUME:-${OUTPUT_ROOT}/sepvae_full_ortho-20260315-220524/checkpoints/checkpoint_final.pkl}"
    ;;
esac

# ── Validate required files ───────────────────────────────────────────────────
[[ -d "$DATA_DIR"  ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"  ]] || die "CSV not found: $CSV_PATH"
[[ -z "$RESUME" || -f "$RESUME" ]] || die "Checkpoint not found: $RESUME"
if [[ "$PHASE" == "d4" || "$PHASE" == "d5" ]]; then
  [[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT (needed for D4/D5 perceptual loss)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
banner "Setup [phase=${PHASE}]"
cd "$WORKDIR"
mkdir -p logs

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}"

kv "Node"              "$(hostname)"
kv "Phase"             "$PHASE  →  $EXP_NAME"
kv "Model"             "SepVAEV2  img=${IMG_SIZE}px  z_c=${Z_COMMON}  z_d=${Z_DISEASE}"
kv "Bbox cross-attn"   "$USE_BBOX_CROSS_ATTN"
kv "Batch / Epochs"    "${BATCH_SIZE} / ${EPOCHS}  (KL warmup: ${KL_WARMUP_EPOCHS})"
kv "Loss weights"      "rec=${WEIGHT_REC}  kl_c=${WEIGHT_KL_COMMON}  kl_d=${WEIGHT_KL_DISEASE}"
kv "               "   "mi=${WEIGHT_MI_FACTOR}  bbox=${WEIGHT_BBOX_ATTN}  perceptual=${WEIGHT_PERCEPTUAL}"
kv "Resume"            "${RESUME:-none}"
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
  --kl_warmup_epochs     "$KL_WARMUP_EPOCHS"

  --lr_vae               "$LR_VAE"
  --lr_disc              "$LR_DISC"
  --weight_decay         "$WEIGHT_DECAY"
  --grad_clip            "$GRAD_CLIP"

  --weight_rec           "$WEIGHT_REC"
  --weight_kl_common     "$WEIGHT_KL_COMMON"
  --weight_kl_disease    "$WEIGHT_KL_DISEASE"
  --weight_mi_factor     "$WEIGHT_MI_FACTOR"
  --weight_bbox_attn     "$WEIGHT_BBOX_ATTN"
  --weight_perceptual    "$WEIGHT_PERCEPTUAL"
  --sigma_inactive       "$SIGMA_INACTIVE"
)

if [[ "$PHASE" == "d5" ]]; then
  ARGS+=(
    --weight_gan       "${WEIGHT_GAN:-0.0}"
    --weight_tv        "${WEIGHT_TV:-0.0}"
    --gan_start_step   "${GAN_START_STEP:-5000}"
    --lr_patch_disc    "${LR_PATCH_DISC:-1e-4}"
  )
fi

ARGS+=(
  --output_root          "$OUTPUT_ROOT"
  --exp_name             "$EXP_NAME"
  --sample_every         "$SAMPLE_EVERY"
  --manifold_every       "$MANIFOLD_EVERY"
)

[[ "$USE_BBOX_CROSS_ATTN" == "1" ]] && ARGS+=( --use_bbox_cross_attn )
[[ -n "$RESUME"           ]]        && ARGS+=( --resume "$RESUME" )

if [[ "$PHASE" == "d4" || "$PHASE" == "d5" ]]; then
  ARGS+=( --chess_checkpoint "$CHESS_CHECKPOINT" --perceptual_only )
fi

if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ── Launch ────────────────────────────────────────────────────────────────────
banner "Launching phase ${PHASE} — ${EXP_NAME}"
python -u "${ARGS[@]}"
ok "Phase ${PHASE} complete"
