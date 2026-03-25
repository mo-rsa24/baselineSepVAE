#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_baseline_ldm}"
RUNS_SEPVAE_ROOT="${RUNS_SEPVAE_ROOT:-${WORKDIR}/runs_sepvae}"
ENV_NAME="${ENV_NAME:-jaxstack}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"

# Phase-aware SepVAE checkpoint selection.
# Supported phases:
#   d0 -> d0_smoke_v2
#   d1 -> d1_recon_bbox_xattn
#   d2 -> d2_mi_disc
#   d3 -> d3_*            (if/when such runs exist)
#   d4 -> d4_perceptual
#   d5 -> d5_recon
#   auto -> d5,d4,d3,d2,d1,d0 fallback
SEPVAE_PHASE="${SEPVAE_PHASE:-auto}"
SEPVAE_CKPT="${SEPVAE_CKPT:-}"

# Latent export.
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
LATENTS_SPLIT="${LATENTS_SPLIT:-train}"
LATENT_MODE="${LATENT_MODE:-mean}"
USE_CACHE="${USE_CACHE:-1}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
MAX_EXPORT_SAMPLES="${MAX_EXPORT_SAMPLES:-0}"

# 24GB VRAM profile.
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LDM_BASE_CH="${LDM_BASE_CH:-128}"
LDM_CH_MULTS="${LDM_CH_MULTS:-1,2,4}"
LDM_NUM_RES_BLOCKS="${LDM_NUM_RES_BLOCKS:-2}"
LDM_ATTN_RES="${LDM_ATTN_RES:-16}"
USE_BFLOAT16="${USE_BFLOAT16:-1}"
USE_REMAT="${USE_REMAT:-1}"
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SAMPLE_STEPS="${SAMPLE_STEPS:-250}"
SEED="${SEED:-42}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae-ldm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

phase_prefix() {
  case "$1" in
    d0) printf '%s\n' 'd0_smoke_v2-' ;;
    d1) printf '%s\n' 'd1_recon_bbox_xattn-' ;;
    d2) printf '%s\n' 'd2_mi_disc-' ;;
    d3) printf '%s\n' 'd3_' ;;
    d4) printf '%s\n' 'd4_perceptual-' ;;
    d5) printf '%s\n' 'd5_recon-' ;;
    *) return 1 ;;
  esac
}

find_latest_phase_ckpt() {
  local phase="$1"
  local prefix
  prefix="$(phase_prefix "$phase")" || return 1
  find "${RUNS_SEPVAE_ROOT}" -maxdepth 3 -name 'checkpoint_final.pkl' | grep "/${prefix}" | sort | tail -1 || true
}

resolve_sepvae_ckpt() {
  if [[ -n "${SEPVAE_CKPT}" ]]; then
    printf '%s\n' "${SEPVAE_CKPT}"
    return
  fi

  if [[ "${SEPVAE_PHASE}" != "auto" ]]; then
    find_latest_phase_ckpt "${SEPVAE_PHASE}"
    return
  fi

  local phase
  for phase in d5 d4 d3 d2 d1 d0; do
    local hit
    hit="$(find_latest_phase_ckpt "${phase}")"
    if [[ -n "${hit}" ]]; then
      printf '%s\n' "${hit}"
      return
    fi
  done
}

SEPVAE_CKPT="$(resolve_sepvae_ckpt)"
[[ -n "${SEPVAE_CKPT}" ]] || { echo "No SepVAE checkpoint found for phase=${SEPVAE_PHASE} under ${RUNS_SEPVAE_ROOT}"; exit 1; }
[[ -f "${SEPVAE_CKPT}" ]] || { echo "SepVAE checkpoint not found: ${SEPVAE_CKPT}"; exit 1; }
[[ -d "${DATA_DIR}" ]] || { echo "DATA_DIR not found: ${DATA_DIR}"; exit 1; }
[[ -f "${CSV_PATH}" ]] || { echo "CSV_PATH not found: ${CSV_PATH}"; exit 1; }

PHASE_TAG="${SEPVAE_PHASE}"
if [[ "${PHASE_TAG}" == "auto" ]]; then
  case "${SEPVAE_CKPT}" in
    */d0_smoke_v2-*) PHASE_TAG="d0" ;;
    */d1_recon_bbox_xattn-*) PHASE_TAG="d1" ;;
    */d2_mi_disc-*) PHASE_TAG="d2" ;;
    */d3_*) PHASE_TAG="d3" ;;
    */d4_perceptual-*) PHASE_TAG="d4" ;;
    */d5_recon-*) PHASE_TAG="d5" ;;
    *) PHASE_TAG="custom" ;;
  esac
fi

LATENTS_DIR="${LATENTS_DIR:-${OUTPUT_ROOT}/latents/${PHASE_TAG}_cardio_${LATENTS_SPLIT}}"
EXP_NAME="${EXP_NAME:-baseline_ldm_cardio_${PHASE_TAG}}"

cd "${WORKDIR}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORM}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u

echo "== baselineLDM launcher =="
echo "  SepVAE phase:        ${PHASE_TAG}"
echo "  SepVAE checkpoint:   ${SEPVAE_CKPT}"
echo "  Latents dir:         ${LATENTS_DIR}"
echo "  24GB profile:        batch=${BATCH_SIZE} base_ch=${LDM_BASE_CH} ch_mults=${LDM_CH_MULTS} res_blocks=${LDM_NUM_RES_BLOCKS}"
echo "  Precision/remat:     bfloat16=${USE_BFLOAT16} remat=${USE_REMAT}"

if [[ "${FORCE_EXPORT}" == "1" || ! -f "${LATENTS_DIR}/manifest.jsonl" ]]; then
  EXPORT_ARGS=(
    scripts/preencode_sepvae_v2_latents.py
    --sepvae_ckpt "${SEPVAE_CKPT}"
    --csv_path "${CSV_PATH}"
    --dicom_dir "${DATA_DIR}"
    --output_dir "${LATENTS_DIR}"
    --split "${LATENTS_SPLIT}"
    --latent_mode "${LATENT_MODE}"
    --batch_size 8
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
  )
  [[ "${USE_CACHE}" == "1" ]] && EXPORT_ARGS+=( --use_cache )
  [[ "${MAX_EXPORT_SAMPLES}" != "0" ]] && EXPORT_ARGS+=( --max_samples "${MAX_EXPORT_SAMPLES}" )
  echo "Exporting paired latents..."
  python -u "${EXPORT_ARGS[@]}"
fi

ARGS=(
  run/train_baseline_ldm.py
  --preencoded_latents_dir "${LATENTS_DIR}"
  --sepvae_ckpt "${SEPVAE_CKPT}"
  --output_root "${OUTPUT_ROOT}"
  --exp_name "${EXP_NAME}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --ldm_base_ch "${LDM_BASE_CH}"
  --ldm_ch_mults "${LDM_CH_MULTS}"
  --ldm_num_res_blocks "${LDM_NUM_RES_BLOCKS}"
  --ldm_attn_res "${LDM_ATTN_RES}"
  --log_every "${LOG_EVERY}"
  --save_every "${SAVE_EVERY}"
  --sample_every "${SAMPLE_EVERY}"
  --sample_batch_size "${SAMPLE_BATCH_SIZE}"
  --sample_steps "${SAMPLE_STEPS}"
  --seed "${SEED}"
  --ema_decay "${EMA_DECAY}"
)

[[ "${USE_BFLOAT16}" == "1" ]] && ARGS+=( --use_bfloat16 )
[[ "${USE_REMAT}" == "1" ]] && ARGS+=( --use_remat )
[[ "${USE_EMA}" == "1" ]] && ARGS+=( --use_ema )

if [[ "${WANDB}" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "${WANDB_PROJECT}" )
  [[ -n "${WANDB_ENTITY}" ]] && ARGS+=( --wandb_entity "${WANDB_ENTITY}" )
fi

python -u "${ARGS[@]}"
