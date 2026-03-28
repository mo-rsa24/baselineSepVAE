#!/usr/bin/env bash
# M3 — PatchGAN adversarial sharpening  (mask curriculum, stage 3)
#
# Resumes from M2 final checkpoint.
# Parallel to D3 (d3_gan_fix) in the original curriculum.
#
# What this stage adds vs M2:
#   - weight_gan=0.1         PatchGAN hinge generator loss
#   - gan_start_step=2000    Phase-local warm-up (~5 epochs before GAN activates)
#   - weight_masked_rec 1.0→1.5  Final push on non-cardiac region constraint
#
# GAN lessons from D3 (applied here from the start):
#   - weight_gan=0.1 (NOT 0.5 — 0.5 caused catastrophic decoder collapse in D5v1)
#   - gan_start_step is PHASE-LOCAL (counts from this run's start, not global_step)
#   - disc_r1_penalty=0.0 and lr_patch_disc=1e-4 (R1 penalty caused bootstrap trap)
#
# Kill conditions (same as D3):
#   - disc_acc > 0.80 for 5 consecutive epochs → discriminator dominating
#   - loss/reconstruction sudden spike (> 3× recent baseline) → GAN collapse
#
# Gate: traversal diagnostic
#   conda run -n jaxstack python scripts/latent_traversal_medsam.py \
#     --checkpoint runs_sepvae/m3_gan-<ts>/checkpoints/checkpoint_final.pkl \
#     --output results/traversal_m3/
#   Success: Δarea > ±500px across α=0→2
#
# Submit:
#   sbatch slurm_scripts/m3_gan.slurm
#
# Override M2 checkpoint:
#   M2_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/m3_gan.slurm

#SBATCH --job-name=sepvae-m3-gan
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-m3-gan-%j.out
#SBATCH --error=logs/sepvae-m3-gan-%j.err

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

M2_CKPT="${M2_CKPT:-}"
[[ -n "$M2_CKPT" ]] || die "M2_CKPT not set. Example: M2_CKPT=runs_sepvae/m2_perceptual-<ts>/checkpoints/checkpoint_final.pkl"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="m3_gan"
EPOCHS=130             # M2 ended at 65, add 65 epochs — matches D3's 65-epoch GAN phase
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

# ── Loss weights — add GAN ────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=0.0       # No bbox supervision ever
WEIGHT_CTR_REG=1.0
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.05
WEIGHT_GAN=0.1             # PatchGAN hinge generator loss introduced
WEIGHT_TV=0.005
WEIGHT_MASKED_REC=1.5      # Final level — matches D4/D5 weight
SIGMA_INACTIVE=0.1

# ── GAN — phase-local warm-up (lessons from D3 failure) ──────────────────────
GAN_START_STEP=2000        # Phase-local: ~5 epochs before adversarial activates
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0        # R1 off — caused bootstrap trap in D3 debugging

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
[[ -f "$M2_CKPT"          ]] || die "M2 checkpoint not found: $M2_CKPT"

banner "M3 PatchGAN Adversarial Sharpening"
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

kv "Node / GPU"        "$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
kv "Exp"               "$EXP_NAME  (ep65 → ${EPOCHS})"
kv "M2 resume"         "$M2_CKPT"
kv "NEW: weight_gan"     "${WEIGHT_GAN}  (PatchGAN, phase-local start at step ${GAN_START_STEP})"
kv "weight_masked_rec"   "${WEIGHT_MASKED_REC}  (raised from M2's 1.0)"
kv "KILL CONDITIONS"     "disc_acc > 0.80 for 5ep  |  recon spike > 3× baseline"

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
  --resume                 "$M2_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "M3 complete — mask curriculum finished"
ok "Traversal diagnostic: conda run -n jaxstack python scripts/latent_traversal_medsam.py \\"
ok "  --checkpoint runs_sepvae/m3_gan-<ts>/checkpoints/checkpoint_final.pkl \\"
ok "  --output results/traversal_m3/"
ok "Success criterion: Δarea > ±500px across α=0→2"
