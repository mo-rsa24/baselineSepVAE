#!/usr/bin/env bash
# M1 — Mask attention + CTR regression  (mask curriculum, stage 1)
#
# Resumes from M0 final checkpoint.
# Parallel to D1 (bbox cross-attention) in the original curriculum.
#
# What this stage adds vs M0:
#   - weight_ctr_reg=0.5    CTR regression — forces z_cardio to encode cardiac size
#   - weight_masked_rec=0.3 z_common must reconstruct outside-cardiac from z_cardio=0
#   - weight_mi_factor=1.0  FactorVAE MI — pushes z_common ⊥ z_cardio
#   - weight_cardio_supcon=0.05  pulls same-class z_cardio together
#
# The mask prior in BboxCrossAttnHead is already active (supplied by chexmask_csv).
# weight_bbox_attn=0.0 throughout this entire curriculum — bbox never supervised.
#
# CTR weight 0.5 (not full 1.0): the encoder is still learning basic anatomy.
# Going straight to 1.0 can over-constrain z_cardio before reconstruction is stable.
# M2 raises to 1.0 once perceptual sharpening is active.
#
# Gate before moving to M2:
#   - loss/ctr_reg < 0.05
#   - metrics/z_cardio_norm_ratio > 1.5 (z_cardio more active on Cardiomegaly)
#   - loss/reconstruction < 0.02
#
# Submit:
#   sbatch slurm_scripts/m1_mask_attn.slurm
#
# Override M0 checkpoint:
#   M0_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/m1_mask_attn.slurm

#SBATCH --job-name=sepvae-m1-attn
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-m1-attn-%j.out
#SBATCH --error=logs/sepvae-m1-attn-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv()    { printf "  ${CYN}%-28s${RST} %s\n" "$1" "$2"; }
ok()    { printf "${GRN}** %s${RST}\n" "$*"; }
die()   { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$PWD}"
ENV_NAME="${ENV_NAME:-jaxstack}"
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHEXMASK_CSV="${CHEXMASK_CSV:-/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"

M0_CKPT="${M0_CKPT:-}"  # Must be set — no hardcoded default (timestamp unknown)
[[ -n "$M0_CKPT" ]] || die "M0_CKPT not set. Example: M0_CKPT=runs_sepvae/m0_smoke-<ts>/checkpoints/checkpoint_final.pkl"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="m1_mask_attn"
EPOCHS=35              # M0 ended at 5, add 30 epochs for mask attention to stabilise
KL_WARMUP_EPOCHS=0

# ── Model ──────────────────────────────────────────────────────────────────────
IMG_SIZE=256
Z_COMMON=16
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0
BBOX_DROPOUT_PROB=0.0

# ── Loss weights — add CTR + MI + masked_rec ──────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0       # FactorVAE MI introduced
WEIGHT_BBOX_ATTN=0.0       # No bbox supervision ever
WEIGHT_CTR_REG=0.5         # CTR regression introduced (conservative — encoder still learning)
WEIGHT_CARDIO_SUPCON=0.05  # Supcon introduced
WEIGHT_PERCEPTUAL=0.0      # Perceptual added in M2
WEIGHT_GAN=0.0             # GAN added in M3
WEIGHT_TV=0.0
WEIGHT_MASKED_REC=0.3      # z_common must own outside-cardiac region
SIGMA_INACTIVE=0.1

GAN_START_STEP=99999
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE=6
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=1e-4
LR_DISC=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0

# ── Logging ────────────────────────────────────────────────────────────────────
SAMPLE_EVERY=5
MANIFOLD_EVERY=5
SAVE_EVERY=5
EVAL_SUBSET_SIZE=1024
WANDB="${WANDB:-1}"
WANDB_PROJECT="baseline-sepvae"

# ── Validate ───────────────────────────────────────────────────────────────────
[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"
[[ -f "$CHEXMASK_CSV"     ]] || die "CheXmask CSV not found: $CHEXMASK_CSV"
[[ -f "$M0_CKPT"          ]] || die "M0 checkpoint not found: $M0_CKPT"

banner "M1 Mask Attention + CTR Regression"
cd "$WORKDIR"
mkdir -p logs

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=1

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}"

kv "Node / GPU"       "$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
kv "Exp"              "$EXP_NAME  (ep5 → ${EPOCHS})"
kv "M0 resume"        "$M0_CKPT"
kv "CheXmask CSV"     "$CHEXMASK_CSV"
kv "NEW: weight_ctr_reg"   "${WEIGHT_CTR_REG}  (CTR regression introduced)"
kv "NEW: weight_mi_factor" "${WEIGHT_MI_FACTOR}  (FactorVAE MI introduced)"
kv "weight_masked_rec"     "${WEIGHT_MASKED_REC}"
kv "weight_bbox_attn"      "${WEIGHT_BBOX_ATTN}  (always 0 in mask curriculum)"

banner "Launching ${EXP_NAME}"
python -u run/train_sep_vae.py \
  --dicom_dir              "$DATA_DIR" \
  --csv_path               "$CSV_PATH" \
  --use_cache \
  --deterministic_data \
  --chexmask_csv           "$CHEXMASK_CSV" \
  \
  --model_version          v2 \
  --img_size               "$IMG_SIZE" \
  --z_channels_common      "$Z_COMMON" \
  --z_channels_disease     "$Z_DISEASE" \
  --attn_query_dim         "$ATTN_QUERY_DIM" \
  --attn_heads             "$ATTN_HEADS" \
  --decoder_res_blocks     "$DECODER_RES_BLOCKS" \
  --use_bbox_cross_attn \
  --bbox_query_mix         "$BBOX_QUERY_MIX" \
  --bbox_dropout_prob      "$BBOX_DROPOUT_PROB" \
  \
  --chess_checkpoint       "$CHESS_CHECKPOINT" \
  --perceptual_only \
  \
  --batch_size             "$BATCH_SIZE" \
  --epochs                 "$EPOCHS" \
  --num_workers            "$NUM_WORKERS" \
  --eval_num_workers       "$EVAL_NUM_WORKERS" \
  --seed                   "$SEED" \
  --kl_warmup_epochs       "$KL_WARMUP_EPOCHS" \
  \
  --lr_vae                 "$LR_VAE" \
  --lr_disc                "$LR_DISC" \
  --weight_decay           "$WEIGHT_DECAY" \
  --grad_clip              "$GRAD_CLIP" \
  \
  --weight_rec             "$WEIGHT_REC" \
  --weight_kl_common       "$WEIGHT_KL_COMMON" \
  --weight_kl_disease      "$WEIGHT_KL_DISEASE" \
  --kl_free_bits           "$KL_FREE_BITS" \
  --weight_mi_factor       "$WEIGHT_MI_FACTOR" \
  --weight_bbox_attn       "$WEIGHT_BBOX_ATTN" \
  --weight_ctr_reg         "$WEIGHT_CTR_REG" \
  --weight_cardio_supcon   "$WEIGHT_CARDIO_SUPCON" \
  --weight_perceptual      "$WEIGHT_PERCEPTUAL" \
  --weight_gan             "$WEIGHT_GAN" \
  --weight_tv              "$WEIGHT_TV" \
  --weight_masked_rec      "$WEIGHT_MASKED_REC" \
  --sigma_inactive         "$SIGMA_INACTIVE" \
  \
  --gan_start_step         "$GAN_START_STEP" \
  --lr_patch_disc          "$LR_PATCH_DISC" \
  --disc_r1_penalty        "$DISC_R1_PENALTY" \
  \
  --output_root            "$OUTPUT_ROOT" \
  --exp_name               "$EXP_NAME" \
  --save_every             "$SAVE_EVERY" \
  --sample_every           "$SAMPLE_EVERY" \
  --manifold_every         "$MANIFOLD_EVERY" \
  --manifold_bbox_mode     both \
  --eval_subset_size       "$EVAL_SUBSET_SIZE" \
  --manifold_max_samples   "$EVAL_SUBSET_SIZE" \
  \
  --resume                 "$M0_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "M1 complete"
ok "Gate check: loss/ctr_reg < 0.05, z_cardio_norm_ratio > 1.5, loss/reconstruction < 0.02"
ok "Resume path for M2: runs_sepvae/m1_mask_attn-<timestamp>/checkpoints/checkpoint_final.pkl"
