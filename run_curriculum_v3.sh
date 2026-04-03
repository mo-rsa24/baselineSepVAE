#!/usr/bin/env bash
# run_curriculum_v3.sh — SepVAEV3 compositional run with internal staged curriculum
#
# Unlike the older D0→D5 / M0→M7 launchers, V3 does not chain multiple jobs.
# The curriculum now lives inside train_sep_vae.py:
#   M0  rec + KL
#   M1  + alpha mask
#   M2  + common_out + heart_in
#   M3  + CTR supervision
#
# Usage:
#   bash run_curriculum_v3.sh
#   sbatch run_curriculum_v3.sh
#
# Override any default before launching:
#   EPOCHS=180 BATCH_SIZE=6 CUDA_VISIBLE_DEVICES=0 bash run_curriculum_v3.sh
#   WANDB_PROJECT=my-project sbatch run_curriculum_v3.sh

#SBATCH --job-name=sepvae-v3
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/curriculum-v3-%j.out
#SBATCH --error=logs/curriculum-v3-%j.err

set -euo pipefail

resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then
        printf '%s\n' "$WORKDIR"; return
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local job_cmd
        job_cmd=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
                  | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
        [[ -n "$job_cmd" && -f "$job_cmd" ]] && { cd "$(dirname "$job_cmd")" && pwd; return; }
    fi
    [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]] \
        && { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; return; }
    pwd
}

WORKDIR="$(resolve_workdir)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"
ENV_NAME="${ENV_NAME:-jaxstack}"

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner() { printf "\n${BLU}${BLD}======  %s  ======${RST}\n\n" "$*"; }
kv()     { printf "  ${CYN}%-28s${RST} %s\n" "$1" "$2"; }
ok()     { printf "${GRN}** %s${RST}\n" "$*"; }
die()    { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHEXMASK_CSV="${CHEXMASK_CSV:-/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv}"

[[ -d "$DATA_DIR"     ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"     ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHEXMASK_CSV" ]] || die "CheXmask CSV not found: $CHEXMASK_CSV"

# ── Optional resume ───────────────────────────────────────────────────────────
RESUME="${RESUME:-}"

# ── Model settings ────────────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_HEART="${Z_HEART:-16}"
ATTN_HEADS="${ATTN_HEADS:-4}"
DECODER_RES_BLOCKS="${DECODER_RES_BLOCKS:-2}"

# ── Training settings ─────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-6}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-1024}"
DETERMINISTIC_DATA="${DETERMINISTIC_DATA:-1}"

LR_VAE="${LR_VAE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
KL_FREE_BITS="${KL_FREE_BITS:-0.5}"

# ── V3 base loss weights ──────────────────────────────────────────────────────
W_REC="${W_REC:-1.0}"
W_KL_C="${W_KL_C:-1e-4}"
W_KL_H="${W_KL_H:-1e-4}"
W_ALPHA="${W_ALPHA:-1.0}"
W_COMMON_OUT="${W_COMMON_OUT:-1.0}"
W_HEART_IN="${W_HEART_IN:-1.0}"
W_CTR="${W_CTR:-1.0}"

# ── V3 curriculum schedule ────────────────────────────────────────────────────
V3_CURRICULUM="${V3_CURRICULUM:-1}"
V3_ALPHA_START_FRAC="${V3_ALPHA_START_FRAC:-0.05}"
V3_PARTS_START_FRAC="${V3_PARTS_START_FRAC:-0.15}"
V3_CTR_START_FRAC="${V3_CTR_START_FRAC:-0.30}"

# ── Logging ───────────────────────────────────────────────────────────────────
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-5}"
MANIFOLD_METHOD="${MANIFOLD_METHOD:-pca}"
MANIFOLD_MAX_SAMPLES="${MANIFOLD_MAX_SAMPLES:-512}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXP_NAME="${EXP_NAME:-sepvae_v3_curriculum}"

banner "SepVAEV3 Compositional Run"
kv "Output root"         "$OUTPUT_ROOT"
kv "Resume"              "${RESUME:-<fresh init>}"
kv "Data dir"            "$DATA_DIR"
kv "CSV"                 "$CSV_PATH"
kv "CheXmask CSV"        "$CHEXMASK_CSV"
kv "Epochs / batch"      "${EPOCHS} / ${BATCH_SIZE}"
kv "Latents"             "z_common=${Z_COMMON}  z_heart=${Z_HEART}"
kv "Decoder blocks"      "$DECODER_RES_BLOCKS"
kv "Loss base"           "rec=${W_REC} kl_c=${W_KL_C} kl_h=${W_KL_H}"
kv "Branch losses"       "alpha=${W_ALPHA} common_out=${W_COMMON_OUT} heart_in=${W_HEART_IN} ctr=${W_CTR}"
kv "Curriculum"          "enabled=${V3_CURRICULUM} alpha=${V3_ALPHA_START_FRAC} parts=${V3_PARTS_START_FRAC} ctr=${V3_CTR_START_FRAC}"
kv "Diagnostics"         "recon + alpha + branch grid + CTR scatter + manifold + scaffolding + traversal + composition"
kv "W&B project"         "${WANDB_PROJECT}  (enabled=${WANDB})"
printf "\n"

ARGS=(
    run/train_sep_vae.py
    --dicom_dir "$DATA_DIR"
    --csv_path "$CSV_PATH"
    --use_cache
    --chexmask_csv "$CHEXMASK_CSV"

    --model_version v3
    --img_size "$IMG_SIZE"
    --z_channels_common "$Z_COMMON"
    --z_channels_disease "$Z_HEART"
    --attn_heads "$ATTN_HEADS"
    --decoder_res_blocks "$DECODER_RES_BLOCKS"

    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --num_workers "$NUM_WORKERS"
    --eval_num_workers "$EVAL_NUM_WORKERS"
    --eval_subset_size "$EVAL_SUBSET_SIZE"
    --seed "$SEED"

    --lr_vae "$LR_VAE"
    --weight_decay "$WEIGHT_DECAY"
    --grad_clip "$GRAD_CLIP"
    --ema_decay "$EMA_DECAY"
    --kl_free_bits "$KL_FREE_BITS"

    --weight_rec "$W_REC"
    --weight_kl_common "$W_KL_C"
    --weight_kl_disease "$W_KL_H"
    --weight_alpha_mask "$W_ALPHA"
    --weight_common_out "$W_COMMON_OUT"
    --weight_heart_in "$W_HEART_IN"
    --weight_ctr_reg "$W_CTR"

    --sample_every "$SAMPLE_EVERY"
    --save_every "$SAVE_EVERY"
    --manifold_every "$MANIFOLD_EVERY"
    --manifold_method "$MANIFOLD_METHOD"
    --manifold_max_samples "$MANIFOLD_MAX_SAMPLES"

    --output_root "$OUTPUT_ROOT"
    --exp_name "$EXP_NAME"
)

if [[ "$DETERMINISTIC_DATA" == "1" ]]; then
    ARGS+=( --deterministic_data )
else
    ARGS+=( --no-deterministic_data )
fi

if [[ "$V3_CURRICULUM" == "1" ]]; then
    ARGS+=(
        --v3_curriculum
        --v3_alpha_start_frac "$V3_ALPHA_START_FRAC"
        --v3_parts_start_frac "$V3_PARTS_START_FRAC"
        --v3_ctr_start_frac "$V3_CTR_START_FRAC"
    )
else
    ARGS+=( --no-v3_curriculum )
fi

[[ -n "$RESUME" ]] && ARGS+=( --resume "$RESUME" )

if [[ "$WANDB" == "1" ]]; then
    ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
    [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

python -u "${ARGS[@]}"
