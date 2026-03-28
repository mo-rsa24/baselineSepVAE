#!/usr/bin/env bash
# D0-mask — Full mask supervision from scratch (no resume, no bbox supervision)
#
# Purpose:
#   Train SepVAEV2 from random initialisation with CheXmask supervision active
#   from epoch 1 and bbox attention loss disabled. Tests whether mask supervision
#   alone (without bbox ever being used) is sufficient for z_cardio disentanglement.
#
# Key differences vs the D0→D3 curriculum:
#   - No --resume (fresh encoder, decoder, discriminators)
#   - weight_bbox_attn=0.0 — bbox never used as a supervision signal
#   - weight_ctr_reg=1.0   — CTR regression active from epoch 1
#   - kl_warmup_epochs=10  — prevents posterior collapse when z_cardio is random
#   - lr_vae=1e-4          — full rate (not the conservative 5e-5 for mature weights)
#   - EPOCHS=200           — encoder/decoder learn from scratch, need more time
#   - GAN starts at step 5000 — reconstruction must stabilise first
#
# What to watch on W&B:
#   loss/ctr_reg        — should drop from ~0.15 toward ~0.02 within 20 epochs
#   metrics/ctr_pred_mean — should converge toward dataset mean CTR (~0.47)
#   metrics/z_cardio_norm_ratio — active/inactive ratio, target > 2.0
#   Manifold plots      — Normal / Cardiomegaly should separate along a CTR axis
#
# Success criterion (run after ep160+):
#   conda run -n jaxstack python scripts/latent_traversal_medsam.py \
#     --checkpoint runs_sepvae/d0_mask_from_scratch-*/checkpoints/checkpoint_final.pkl \
#     --output results/traversal_d0_mask/
#   Target: Δarea > ±500px across α=0→2 traversal
#
# Submit:
#   sbatch slurm_scripts/d0_mask_from_scratch.slurm

#SBATCH --job-name=sepvae-d0-mask
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-d0-mask-%j.out
#SBATCH --error=logs/sepvae-d0-mask-%j.err

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

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="d0_mask_from_scratch"
EPOCHS=200
KL_WARMUP_EPOCHS=10        # Ramp KL 0→1 over first 10 epochs — prevents early collapse

# ── Model ──────────────────────────────────────────────────────────────────────
IMG_SIZE=256
Z_COMMON=16
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0
BBOX_DROPOUT_PROB=0.0      # No bbox dropout — mask supervision covers all images

# ── Loss weights ──────────────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=0.0       # Pure mask — bbox attention loss disabled
WEIGHT_CTR_REG=1.0         # CTR regression active from epoch 1
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.05
WEIGHT_GAN=0.1
WEIGHT_TV=0.005
WEIGHT_MASKED_REC=0.3      # Start conservative — decoder learning from scratch
SIGMA_INACTIVE=0.1

# ── GAN — delayed start ────────────────────────────────────────────────────────
# step 5000 ≈ 12 epochs at batch=6 on ~14k pairs → reconstruction stabilises first
GAN_START_STEP=5000
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE=6
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=1e-4                # Full rate — training from scratch, not fine-tuning
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

# ── Environment ────────────────────────────────────────────────────────────────
banner "D0 Mask From Scratch — pure mask supervision, no bbox, no resume"
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

kv "Node / GPU"          "$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
kv "Exp"                 "$EXP_NAME  (fresh weights, epoch 1 → ${EPOCHS})"
kv "CheXmask CSV"        "$CHEXMASK_CSV"
kv "PARADIGM"            "pure mask supervision — bbox attention loss DISABLED"
kv "weight_bbox_attn"    "${WEIGHT_BBOX_ATTN}  (disabled)"
kv "weight_ctr_reg"      "${WEIGHT_CTR_REG}  (active from epoch 1)"
kv "weight_masked_rec"   "${WEIGHT_MASKED_REC}"
kv "kl_warmup_epochs"    "${KL_WARMUP_EPOCHS}"
kv "gan_start_step"      "${GAN_START_STEP}  (~12 epochs warm-up)"
kv "Batch / lr_vae"      "${BATCH_SIZE} / ${LR_VAE}"

# ── Launch ─────────────────────────────────────────────────────────────────────
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
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "D0 mask from scratch complete"
ok "Next: run latent_traversal_medsam.py — success if Δarea > ±500px"
