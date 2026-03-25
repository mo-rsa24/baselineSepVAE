#!/usr/bin/env bash
# D2 — Perceptual + bbox attention loss top-up
#
# Resumes from d1_recon_bbox_xattn checkpoint (epoch 30).
# Adds CheSS perceptual loss and a light bbox attention loss to sharpen
# reconstructions and push free-running silhouette higher.
# No MI disc, no GAN — keep it simple.
#
# Submit:
#   sbatch slurm_scripts/d2_perceptual_bbox.slurm
#
# Override resume path:
#   RESUME=/path/to/ckpt.pkl sbatch slurm_scripts/d2_perceptual_bbox.slurm

#SBATCH --job-name=sepvae-d2
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-d2-%j.out
#SBATCH --error=logs/sepvae-d2-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv()    { printf "  ${CYN}%-26s${RST} %s\n" "$1" "$2"; }
ok()    { printf "${GRN}** %s${RST}\n" "$*"; }
die()   { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$PWD}"
ENV_NAME="${ENV_NAME:-jaxstack}"
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"

RESUME="${RESUME:-${WORKDIR}/runs_sepvae/d1_recon_bbox_xattn-20260321-004241/checkpoints/checkpoint_final.pkl}"

# ── Experiment ────────────────────────────────────────────────────────────────
EXP_NAME="d2_perceptual_bbox"
EPOCHS=55   # final epoch number (resumes from epoch 30, so +25 epochs)
KL_WARMUP_EPOCHS=0

# ── Model ─────────────────────────────────────────────────────────────────────
# Encoder must match d1 exactly (same arch = same checkpoint keys).
# Decoder gains a 3rd ResBlockSE per level: Flax names blocks by loop index so
# ResBlockSE_0 and _1 load from d1 cleanly; ResBlockSE_2 is freshly init'd.
IMG_SIZE=256
Z_COMMON=16
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3        # was 2 in d1 (adds depth at every scale, near-identity init)

# ── Loss weights ──────────────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5            # per-dim KL floor: bounds step variance (was 0 → 33–11600 spikes)
WEIGHT_MI_FACTOR=0.0        # no MI disc yet
WEIGHT_BBOX_ATTN=0.05       # light supervision — was 0 in d1
WEIGHT_PERCEPTUAL=0.15      # increased from 0.05: 3× stronger texture signal
WEIGHT_TV=0.001             # total variation: suppresses stride-aliased perceptual gradients
WEIGHT_CARDIO_SUPCON=0.05   # same as d1 default
SIGMA_INACTIVE=0.1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE=8
NUM_WORKERS=8
SEED=0
LR_VAE=1e-4
LR_BACKBONE=1e-5
LR_DISC=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0

# ── Logging ───────────────────────────────────────────────────────────────────
SAMPLE_EVERY=5
MANIFOLD_EVERY=5            # more frequent — watching silhouette closely
SAVE_EVERY=5
WANDB="${WANDB:-1}"
WANDB_PROJECT="baseline-sepvae"

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$RESUME"           ]] || die "Resume checkpoint not found: $RESUME"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Environment ───────────────────────────────────────────────────────────────
banner "D2 — perceptual + bbox attn"
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
kv "Exp"               "$EXP_NAME  (${EPOCHS} epochs)"
kv "Resume"            "$RESUME"
kv "Batch / LR_vae"    "${BATCH_SIZE} / ${LR_VAE}"
kv "decoder_res_blocks" "$DECODER_RES_BLOCKS  (was 2, ResBlockSE_2 freshly init'd)"
kv "weight_perceptual" "$WEIGHT_PERCEPTUAL  (CheSS, was 0.05)"
kv "weight_tv"         "$WEIGHT_TV          (suppresses perceptual stride artifacts)"
kv "kl_free_bits"      "$KL_FREE_BITS       (per-dim KL floor, was 0)"
kv "weight_bbox_attn"  "$WEIGHT_BBOX_ATTN   (was 0 in d1)"
kv "weight_mi_factor"  "$WEIGHT_MI_FACTOR"
kv "Output"            "$OUTPUT_ROOT"

# ── Launch ────────────────────────────────────────────────────────────────────
banner "Launching ${EXP_NAME}"
python -u run/train_sep_vae.py \
  --dicom_dir            "$DATA_DIR" \
  --csv_path             "$CSV_PATH" \
  --use_cache \
  \
  --model_version        v2 \
  --img_size             "$IMG_SIZE" \
  --z_channels_common    "$Z_COMMON" \
  --z_channels_disease   "$Z_DISEASE" \
  --attn_query_dim       "$ATTN_QUERY_DIM" \
  --attn_heads           "$ATTN_HEADS" \
  --decoder_res_blocks   "$DECODER_RES_BLOCKS" \
  --use_bbox_cross_attn \
  \
  --chess_checkpoint     "$CHESS_CHECKPOINT" \
  --perceptual_only \
  \
  --batch_size           "$BATCH_SIZE" \
  --epochs               "$EPOCHS" \
  --num_workers          "$NUM_WORKERS" \
  --seed                 "$SEED" \
  --kl_warmup_epochs     "$KL_WARMUP_EPOCHS" \
  \
  --lr_vae               "$LR_VAE" \
  --lr_backbone          "$LR_BACKBONE" \
  --lr_disc              "$LR_DISC" \
  --weight_decay         "$WEIGHT_DECAY" \
  --grad_clip            "$GRAD_CLIP" \
  \
  --weight_rec           "$WEIGHT_REC" \
  --weight_kl_common     "$WEIGHT_KL_COMMON" \
  --weight_kl_disease    "$WEIGHT_KL_DISEASE" \
  --kl_free_bits         "$KL_FREE_BITS" \
  --weight_mi_factor     "$WEIGHT_MI_FACTOR" \
  --weight_bbox_attn     "$WEIGHT_BBOX_ATTN" \
  --weight_perceptual    "$WEIGHT_PERCEPTUAL" \
  --weight_tv            "$WEIGHT_TV" \
  --weight_cardio_supcon "$WEIGHT_CARDIO_SUPCON" \
  --sigma_inactive       "$SIGMA_INACTIVE" \
  \
  --output_root          "$OUTPUT_ROOT" \
  --exp_name             "$EXP_NAME" \
  --save_every           "$SAVE_EVERY" \
  --sample_every         "$SAMPLE_EVERY" \
  --manifold_every       "$MANIFOLD_EVERY" \
  --manifold_bbox_mode   both \
  \
  --resume               "$RESUME" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "D2 complete"
