#!/usr/bin/env bash
# run_mask_sep.sh — Full spatial factorisation: heart_out_zc + heart_in_zd
#
# Architecture change from h1_hout_attn:
#   REMOVED: BboxCrossAttnHead (cross-attention + learned query)
#   ADDED:   heart_in_zd — tg_branch sees ONLY cardiac pixels (complement masking)
#            ConvHeadGN for z_d (same as z_c head; attention not needed)
#
# Spatial factorisation:
#   z_c = encode(h_shared * (1 - mask))  → background anatomy only
#   z_d = encode(h_shared * mask)         → cardiac silhouette only
#   z_c + z_d → decode → full CXR
#
# This is the structural prerequisite for the LDM composition rule:
#   ε_AND = ε(z_c,∅) + w1·[ε(z_c,c1)−ε(z_c,∅)] + w2·[ε(z_c,c2)−ε(z_c,∅)]
# which requires c1 ⊥ c2 | z_c — holds iff spatial supports are disjoint.
#
# Diagnostic convergence goals (check W&B scaffolding panel every 5 epochs):
#   col 05 (z_d decode)  → should look like col 02 (GT CheXmask)
#   col 06 (diff ×5)     → cardiac silhouette should sharpen over epochs
#   z_cardio_norm_ratio  → should rise above 1.5 by epoch 20
#
# Warm-started from h1_hout_attn ep35 checkpoint:
#   backbone, bg_branch, head_common, decoder → restored
#   head_disease (BboxCrossAttnHead → ConvHeadGN) → fresh init (shape mismatch)
#   tg_branch → restored but now fed cardiac-only pixels (adapts quickly)
#
# Usage:
#   bash run_mask_sep.sh          # interactive
#   sbatch run_mask_sep.sh        # SLURM

#SBATCH --job-name=sepvae-mask-sep
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/mask-sep-%j.out
#SBATCH --error=logs/mask-sep-%j.err

set -euo pipefail

# ── Working directory ─────────────────────────────────────────────────────────
resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then printf '%s\n' "$WORKDIR"; return; fi
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DICOM_DIR="/datasets/mmolefe/vinbigdata/cache_npy"
CSV_PATH="/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv"
CHEXMASK_CSV="/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv"

# Smoke test: fresh init, no resume.
# Set RESUME to a checkpoint path to warm-start from h1:
#   RESUME="/home-mscluster/mmolefe/Playground/PhD/baselineSepVAE/runs_sepvae/h1_hout_attn-20260331-221954/checkpoints/checkpoint_final.pkl"
RESUME=""

WANDB_PROJECT="baseline-sepvae"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS=3            # smoke test: 3 epochs to validate full pipeline
BATCH_SIZE=6
SEED=0

# Loss weights
W_REC=1.0
W_KL_C=1e-4
W_KL_D=5e-5
W_KL_FREE=0.5
W_MI=1.0
W_SUPCON=0.05
W_MASKED_REC=0.3   # outside-bbox MSE with z_d=0 — forces z_c to handle background
W_CTR=0.5          # CTR regression anchors z_d to cardiac size (scalar dial)
W_BBOX_ATTN=0.0    # not needed: spatial support is structural, not learned

# ── Environment ───────────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" || true
fi

banner "mask-sep  heart_out_zc + heart_in_zd  (full spatial factorisation)"
kv "Resume from"          "${RESUME:-<fresh init>}"
kv "Epochs"               "$EPOCHS"
kv "heart_out_zc"         "ENABLED — z_c sees background only"
kv "heart_in_zd"          "ENABLED — z_d sees cardiac only (complement masking)"
kv "head_disease"         "ConvHeadGN  (no cross-attention)"
kv "CheXmask"             "$CHEXMASK_CSV"
kv "weight_masked_rec"    "$W_MASKED_REC"
kv "weight_ctr_reg"       "$W_CTR"
kv "Diagnostic goals"     "col05≈col02 (cardiac), col06 sharpens to silhouette"
printf "\n"

python run/train_sep_vae.py \
    --dicom_dir          "$DICOM_DIR" \
    --csv_path           "$CSV_PATH" \
    --use_cache \
    --deterministic_data \
    --chexmask_csv       "$CHEXMASK_CSV" \
    \
    --heart_out_zc \
    --heart_in_zd \
    \
    --model_version      v2 \
    --img_size           256 \
    --z_channels_common  16 \
    --z_channels_disease 16 \
    --attn_heads         4 \
    --decoder_res_blocks 3 \
    \
    --batch_size         $BATCH_SIZE \
    --epochs             $EPOCHS \
    --num_workers        8 \
    --eval_num_workers   0 \
    --seed               $SEED \
    \
    --kl_warmup_epochs   0 \
    --lr_vae             1e-4 \
    --lr_disc            1e-4 \
    --weight_decay       1e-4 \
    --grad_clip          1.0 \
    \
    --weight_rec         $W_REC \
    --weight_kl_common   $W_KL_C \
    --weight_kl_disease  $W_KL_D \
    --kl_free_bits       $W_KL_FREE \
    --weight_mi_factor   $W_MI \
    --weight_cardio_supcon $W_SUPCON \
    --weight_masked_rec  $W_MASKED_REC \
    --weight_ctr_reg     $W_CTR \
    --weight_bbox_attn   $W_BBOX_ATTN \
    --weight_perceptual  0.0 \
    --weight_gan         0.0 \
    --weight_tv          0.0 \
    --gan_start_step     99999 \
    --lr_patch_disc      1e-4 \
    --disc_r1_penalty    0.0 \
    \
    --supcon_temperature 0.1 \
    --sigma_inactive     0.1 \
    \
    --output_root        "$OUTPUT_ROOT" \
    --exp_name           h2_mask_sep \
    --sample_every       1 \
    --save_every         1 \
    --manifold_every     1 \
    --manifold_bbox_mode bbox_free \
    --eval_subset_size   256 \
    --manifold_max_samples 256 \
    \
    ${RESUME:+--resume "$RESUME"} \
    --wandb \
    --wandb_project      "$WANDB_PROJECT"
