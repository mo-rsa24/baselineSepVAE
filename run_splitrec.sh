#!/usr/bin/env bash
# run_splitrec.sh — SepVAEV3 split-reconstruction run
#
# Resumes from sepvae_v3_sep-20260404-121321 (epoch 200, silhouette=0.920).
# Extends to 300 epochs while introducing Fix 2 (split reconstruction):
#
#   Epochs 201-209  M5 quality, split_rec_weight = 0   (pure soft-composite baseline)
#   Epochs 210-229  M5 quality, split_rec_weight 0→1   (linear warmup, sigma=8px boundary)
#   Epochs 230-300  M5 quality, split_rec_weight = 1   (fully split, absolute branch encoding)
#
# Expected effect: heart-only branch transitions from ring/outline artefact
# to full cardiac content, enabling ecologically valid compositional synthesis.
#
# Usage:
#   bash run_splitrec.sh
#   sbatch run_splitrec.sh

#SBATCH --job-name=sepvae-splitrec
#SBATCH --nodelist=mscluster106
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/splitrec-%j.out
#SBATCH --error=logs/splitrec-%j.err

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
kv()     { printf "  ${CYN}%-32s${RST} %s\n" "$1" "$2"; }
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

banner "GPU Preflight"
printf "  SLURM_JOB_ID=%s\n" "${SLURM_JOB_ID:-unset}"
printf "  CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_VISIBLE_DEVICES"
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi -L \
    || die "nvidia-smi not found — no GPU environment detected."

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHEXMASK_CSV="${CHEXMASK_CSV:-/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv}"

[[ -d "$DATA_DIR"     ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"     ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHEXMASK_CSV" ]] || die "CheXmask CSV not found: $CHEXMASK_CSV"

# ── Resume checkpoint ─────────────────────────────────────────────────────────
RESUME="${RESUME:-${OUTPUT_ROOT}/sepvae_v3_sep-20260404-121321/checkpoints/checkpoint_final.pkl}"
[[ -f "$RESUME" ]] || die "Resume checkpoint not found: $RESUME"

# ── Model (unchanged from previous run) ───────────────────────────────────────
IMG_SIZE=256
Z_COMMON=16
Z_HEART=16
ATTN_HEADS=4
DECODER_RES_BLOCKS=2

# ── Training ──────────────────────────────────────────────────────────────────
# 300 total epochs: resumes at 200, runs 100 more.
EPOCHS=300
BATCH_SIZE=6
SEED=0
NUM_WORKERS=0
EVAL_NUM_WORKERS=0
EVAL_SUBSET_SIZE=1024

LR_VAE=1e-4
LR_DISC=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
EMA_DECAY=0.999
KL_FREE_BITS=0.5

# ── Base loss weights (identical to previous run) ─────────────────────────────
W_REC=1.0
W_KL_C=1e-4
W_KL_H=1e-4
W_ALPHA=1.0
W_COMMON_OUT=1.0
W_HEART_IN=1.0
W_CTR=1.0
W_MI=1.0
W_SUPCON=0.05
SUPCON_TEMP=0.1
SIGMA_INACTIVE=0.3
W_CTR_ADV=0.1

# ── Curriculum (fractions reference total 300 epochs) ─────────────────────────
# All stages below were already fully active at epoch 200 — no regression.
V3_ALPHA_START_FRAC=0.01      # epoch 3   (already active)
V3_PARTS_START_FRAC=0.02      # epoch 6   (already active)
V3_CTR_START_FRAC=0.02        # epoch 6   (already active)
V3_SEP_START_FRAC=0.05        # epoch 15  (already active)
V3_QUALITY_START_FRAC=0.60    # epoch 180 (already active)

# ── Split reconstruction warmup ───────────────────────────────────────────────
# Start fraction 0.70 → epoch ceil(300 * 0.70) = 210  (10 epochs after resuming)
# Warmup 20 epochs → split_rec_weight reaches 1.0 at epoch 230
# Remaining 70 epochs (231-300) train with fully split reconstruction.
V3_SPLIT_REC_START_FRAC=0.70
V3_SPLIT_REC_WARMUP_EPOCHS=20
V3_BOUNDARY_SIGMA=8.0         # ~16px soft boundary at heart edge

# ── Logging ───────────────────────────────────────────────────────────────────
SAMPLE_EVERY=5
SAVE_EVERY=5
MANIFOLD_EVERY=5
MANIFOLD_METHOD=pca
MANIFOLD_MAX_SAMPLES=512
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXP_NAME="${EXP_NAME:-sepvae_v3_splitrec}"

banner "SepVAEV3 Split-Reconstruction Run"
kv "Resume"                    "$RESUME"
kv "Output root"               "$OUTPUT_ROOT"
kv "Epochs"                    "${EPOCHS}  (resumes at 200, runs 100 more)"
kv "Batch / LR"                "${BATCH_SIZE} / ${LR_VAE}"
kv "Split rec start"           "epoch ceil(${EPOCHS}*${V3_SPLIT_REC_START_FRAC})=210"
kv "Split rec warmup"          "${V3_SPLIT_REC_WARMUP_EPOCHS} epochs  (fully active ep 230)"
kv "Boundary sigma"            "${V3_BOUNDARY_SIGMA} px  (~$(echo "2*$V3_BOUNDARY_SIGMA" | bc) px blend zone)"
kv "M4 separation"             "mi=${W_MI} supcon=${W_SUPCON} sigma_inactive=${SIGMA_INACTIVE} ctr_adv=${W_CTR_ADV}"
kv "W&B project"               "${WANDB_PROJECT}  (enabled=${WANDB})"
printf "\n"

ARGS=(
    run/train_sep_vae.py
    --dicom_dir        "$DATA_DIR"
    --csv_path         "$CSV_PATH"
    --use_cache
    --chexmask_csv     "$CHEXMASK_CSV"

    --model_version    v3
    --img_size         "$IMG_SIZE"
    --z_channels_common  "$Z_COMMON"
    --z_channels_disease "$Z_HEART"
    --attn_heads         "$ATTN_HEADS"
    --decoder_res_blocks "$DECODER_RES_BLOCKS"

    --batch_size       "$BATCH_SIZE"
    --epochs           "$EPOCHS"
    --num_workers      "$NUM_WORKERS"
    --eval_num_workers "$EVAL_NUM_WORKERS"
    --eval_subset_size "$EVAL_SUBSET_SIZE"
    --seed             "$SEED"
    --deterministic_data

    --lr_vae           "$LR_VAE"
    --lr_disc          "$LR_DISC"
    --weight_decay     "$WEIGHT_DECAY"
    --grad_clip        "$GRAD_CLIP"
    --ema_decay        "$EMA_DECAY"
    --kl_free_bits     "$KL_FREE_BITS"

    --weight_rec           "$W_REC"
    --weight_kl_common     "$W_KL_C"
    --weight_kl_disease    "$W_KL_H"
    --weight_alpha_mask    "$W_ALPHA"
    --weight_common_out    "$W_COMMON_OUT"
    --weight_heart_in      "$W_HEART_IN"
    --weight_ctr_reg       "$W_CTR"
    --weight_mi_factor     "$W_MI"
    --weight_cardio_supcon "$W_SUPCON"
    --supcon_temperature   "$SUPCON_TEMP"
    --sigma_inactive       "$SIGMA_INACTIVE"
    --weight_ctr_adv       "$W_CTR_ADV"

    --v3_curriculum
    --v3_alpha_start_frac    "$V3_ALPHA_START_FRAC"
    --v3_parts_start_frac    "$V3_PARTS_START_FRAC"
    --v3_ctr_start_frac      "$V3_CTR_START_FRAC"
    --v3_sep_start_frac      "$V3_SEP_START_FRAC"
    --v3_quality_start_frac  "$V3_QUALITY_START_FRAC"

    --v3_split_rec_start_frac     "$V3_SPLIT_REC_START_FRAC"
    --v3_split_rec_warmup_epochs  "$V3_SPLIT_REC_WARMUP_EPOCHS"
    --v3_boundary_sigma           "$V3_BOUNDARY_SIGMA"

    --sample_every         "$SAMPLE_EVERY"
    --save_every           "$SAVE_EVERY"
    --manifold_every       "$MANIFOLD_EVERY"
    --manifold_method      "$MANIFOLD_METHOD"
    --manifold_max_samples "$MANIFOLD_MAX_SAMPLES"

    --output_root "$OUTPUT_ROOT"
    --exp_name    "$EXP_NAME"
    --resume      "$RESUME"
)

if [[ "$WANDB" == "1" ]]; then
    ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
    [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

banner "Launching"
python -u "${ARGS[@]}"
ok "Run complete"
