#!/usr/bin/env bash
# D4 — Weight Amplification Test
#
# Resumes from D3 ep120 (d3_gan_fix-20260325-143813).
#
# Purpose: H2 disambiguation — test whether the z_disease traversal collapse
# (Δarea = ±7px at D3/ep120) was caused by loss weights being too low, rather
# than by the bbox supervision signal being too noisy.
#
# D3 ran with: weight_bbox_attn=0.10, weight_masked_rec=0.3
# D4 raises:   weight_bbox_attn=2.0  (20× increase)
#              weight_masked_rec=1.5  (5× increase)
#
# No architectural changes. No mask supervision. Pure weight test.
#
# Diagnostic gate:
#   After 20 epochs, run: conda run -n jaxstack python scripts/latent_traversal_medsam.py \
#     --checkpoint runs_sepvae/d4_weight_amp-*/checkpoints/checkpoint_final.pkl \
#     --output results/traversal_d4_weight_test/
#
#   If Δarea > ±50px → H2 confirmed. Loss weights were the issue. Proceed to
#     standard D5 (raise weights, keep bbox supervision).
#   If Δarea ≤ ±50px → H1/H3. Signal quality is the bottleneck. Run D5_mask.
#
# Submit:
#   sbatch slurm_scripts/d4_weight_amp.slurm
#
# Override D3 checkpoint:
#   D3_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/d4_weight_amp.slurm

#SBATCH --job-name=sepvae-d4-wamp
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-d4-wamp-%j.out
#SBATCH --error=logs/sepvae-d4-wamp-%j.err

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
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"

D3_CKPT="${D3_CKPT:-${WORKDIR}/runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch0120.pkl}"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="d4_weight_amp"
EPOCHS=140          # D3 ended at 120, add 20 diagnostic epochs
KL_WARMUP_EPOCHS=0

# ── Model — identical to D3 ───────────────────────────────────────────────────
IMG_SIZE=256
Z_COMMON=16
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0
BBOX_DROPOUT_PROB=0.3

# ── Loss weights — KEY CHANGE: 20× bbox_attn, 5× masked_rec ──────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=2.0        # D3: 0.10 → D4: 2.0  (20× increase — H2 test)
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.05
WEIGHT_GAN=0.1
WEIGHT_TV=0.005
WEIGHT_MASKED_REC=1.5       # D3: 0.30 → D4: 1.5  (5× increase)
SIGMA_INACTIVE=0.1

# ── GAN — same as D3 ──────────────────────────────────────────────────────────
GAN_START_STEP=2000
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE=6
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=5e-5             # Lower than D3 (1e-4) — mature weights, small perturbation
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
[[ -f "$D3_CKPT"          ]] || die "D3 checkpoint not found: $D3_CKPT"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Environment ────────────────────────────────────────────────────────────────
banner "D4 Weight Amplification — H2 diagnostic"
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
kv "Exp"                 "$EXP_NAME  (ep120 → ${EPOCHS}, +20 diagnostic epochs)"
kv "D3 resume"           "$D3_CKPT"
kv "DIAGNOSTIC"          "H2 weight test: bbox_attn 0.10→2.0, masked_rec 0.30→1.50"
kv "weight_bbox_attn"    "${WEIGHT_BBOX_ATTN}  (was 0.10 in D3)"
kv "weight_masked_rec"   "${WEIGHT_MASKED_REC}  (was 0.30 in D3)"
kv "Batch / lr_vae"      "${BATCH_SIZE} / ${LR_VAE}"

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
  --resume                 "$D3_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "D4 weight amplification complete"
ok "Next: run latent_traversal_medsam.py to measure Δarea and check H2"
