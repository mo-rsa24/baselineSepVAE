#!/usr/bin/env bash
# D7 — UNet skip connections + wider z_common
#
# Resumes from D6 checkpoint (d5_gan_v2 run).
#
# Architectural changes vs D6:
#
#   Change 1 — UNet skip connections (models/sep_vae_v2.py)
#       ResNet50Scratch now returns (h_layer3, h_layer2).
#       SepVAEEncoderV2 exposes skip_feats in latents_dict.
#       SepVAEDecoderV2 injects at two scales via concat + 1×1 conv:
#           i=4 (16×16): h_shared (1024ch) fused with decoder h (512ch)
#           i=3 (32×32): h_layer2  (512ch) fused with decoder h (512ch)
#       New parameters (skip3_fuse, skip2_fuse) are freshly initialised
#       by _merge_params — all other weights restored from D6 checkpoint.
#
#   Change 2 — z_channels_common 16 → 32  (D7-B)
#       Doubles common-latent capacity: 4096 → 8192 values for Normal images.
#       Encoder head_common and decoder z_proj are re-initialised (shape change).
#       All other encoder/decoder weights restored from D6.
#
# Submit:
#   sbatch slurm_scripts/d7_skip.slurm
#
# Override D6 checkpoint:
#   D6_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/d7_skip.slurm

#SBATCH --job-name=sepvae-d7
#SBATCH --nodelist=mscluster106
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-d7-%j.out
#SBATCH --error=logs/sepvae-d7-%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"

# D6 final checkpoint — set this after D6 completes
D6_CKPT="${D6_CKPT:-${WORKDIR}/runs_sepvae/d5_gan_v2-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl}"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="d7_skip"
EPOCHS=200          # final epoch; D6 ends at 160, so this adds 40 epochs
KL_WARMUP_EPOCHS=0

# ── Model ──────────────────────────────────────────────────────────────────────
IMG_SIZE=256
Z_COMMON=32         # D7-B: doubled from 16 — encoder head_common re-initialised
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0
BBOX_DROPOUT_PROB=0.3

# ── Loss weights ───────────────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=0.10
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.0       # D7: perceptual off — skip connections + GAN replace it; eliminates CheSS aliasing
WEIGHT_GAN=0.1
WEIGHT_TV=0.005
WEIGHT_MASKED_REC=0.3
SIGMA_INACTIVE=0.1

# ── GAN / discriminator ────────────────────────────────────────────────────────
GAN_START_STEP=500              # shorter warm-up: skip connections accelerate decoder
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE=4                    # skip connections increase peak memory; reduce if OOM
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=5e-5                     # lower LR: mature encoder, only skip projections are new
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
[[ -f "$D6_CKPT"          ]] || die "D6 checkpoint not found: $D6_CKPT"

# ── Environment ────────────────────────────────────────────────────────────────
banner "D7 skip — UNet skip connections + z_common=32"
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
kv "Exp"                 "$EXP_NAME  (epochs 160 → ${EPOCHS})"
kv "D6 resume"           "$D6_CKPT"
kv "z_common"            "${Z_COMMON}  (D7-B: doubled from 16)"
kv "Skip connections"    "layer3 (16×16×1024) + layer2 (32×32×512) → decoder"
kv "Batch / lr_vae"      "${BATCH_SIZE} / ${LR_VAE}"
kv "decoder_res_blocks"  "$DECODER_RES_BLOCKS"
kv "weight_gan"          "$WEIGHT_GAN"
kv "gan_start_step"      "$GAN_START_STEP  (phase-local, shorter warm-up)"
kv "weight_rec"          "$WEIGHT_REC"
kv "weight_perceptual"   "0.0 (disabled — skip connections + GAN replace CheSS perceptual)"
kv "weight_tv"           "$WEIGHT_TV"
kv "Output"              "$OUTPUT_ROOT"

# ── Launch ─────────────────────────────────────────────────────────────────────
banner "Launching ${EXP_NAME}"
python -u run/train_sep_vae.py \
  --dicom_dir              "$DATA_DIR" \
  --csv_path               "$CSV_PATH" \
  --use_cache \
  --deterministic_data \
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
  --resume                 "$D6_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "D7 skip complete"
