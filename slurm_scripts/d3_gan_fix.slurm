#!/usr/bin/env bash
# D5 — PatchGAN sharpening  (v2: bug-fixed re-run)
#
# Resumes from d4_mi_percep-20260322-222345 (epoch 120).
#
# Bug fixes vs d5_gan-20260323-042442 (post-mortem of catastrophic failure):
#
#   Bug 1 — weight_gan 0.5 → 0.1
#       At epoch 140 of the failed run: 0.5×1.25=0.625 GAN vs 1.0×0.135=0.135 rec
#       (4.6× imbalance). Adversarial gradient overwhelmed reconstruction and
#       corrupted the decoder within 4 epochs.
#
#   Bug 2 — gan_start_step is now PHASE-LOCAL (fix in train_sep_vae.py)
#       global_step was restored from D4 ckpt (~57196), immediately exceeding
#       the old gan_start_step=2000. The fresh discriminator fired from step 1
#       of D5, reached 88% accuracy in 4 epochs, overwhelmed the generator.
#       train_sep_vae.py now uses:
#           phase_local_step = global_step - phase_start_global_step
#           gan_active = phase_local_step >= gan_start_step
#
#   Bug 3 — lr_patch_disc 1e-4 → 3e-5  +  disc_r1_penalty=10.0
#       Slower discriminator convergence + R1 gradient penalty on real samples
#       (γ/2 · E[||∇_x D(x_real)||²]) prevents discriminator from overfitting
#       freely before the generator can adapt.
#
# Submit:
#   sbatch slurm_scripts/d5_gan_v2.slurm
#
# Override D4 checkpoint:
#   D4_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/d5_gan_v2.slurm

#SBATCH --job-name=sepvae-d5v2
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-d5v2-%j.out
#SBATCH --error=logs/sepvae-d5v2-%j.err

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

# D2 final checkpoint — starting from before CheSS perceptual was introduced (avoids stripe-baked weights)
D4_CKPT="${D4_CKPT:-${WORKDIR}/runs_sepvae/d2_perceptual_bbox-20260324-105108/checkpoints/checkpoint_final.pkl}"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="d3_gan_fix"
EPOCHS=120          # final epoch; D2 ended at 55, so this adds 65 epochs
KL_WARMUP_EPOCHS=0

# ── Model ──────────────────────────────────────────────────────────────────────
IMG_SIZE=256
Z_COMMON=16
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0          # D5: pure Gaussian prior — encoder mature enough
BBOX_DROPOUT_PROB=0.3

# ── Loss weights ───────────────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5      # D6 FIX: reverted from 1e-4 — doubling caused stripe artifacts
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=0.10       # raised from 0.05: 120 ep of guidance, now stronger
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.05      # D6 FIX: reduced from 0.15 — layer3 excluded, lower weight balances TV
WEIGHT_GAN=0.1              # BUG FIX: was 0.5 → GAN dominated rec (4.6× ratio)
WEIGHT_TV=0.005             # D6 FIX: raised from 0.001 — suppresses periodic artifacts (5×)
WEIGHT_MASKED_REC=0.3
SIGMA_INACTIVE=0.1

# ── GAN / discriminator ────────────────────────────────────────────────────────
# gan_start_step is now PHASE-LOCAL (train_sep_vae.py fix):
#   uses global_step - phase_start_global_step, not raw global_step.
#   2000 steps = ~5 epochs of warm-up before adversarial training activates.
GAN_START_STEP=2000
LR_PATCH_DISC=1e-4          # D6 FIX: raised from 3e-5 — too slow to escape R1 trap
DISC_R1_PENALTY=0.0         # D6 FIX: removed (was 10.0) — prevented discriminator bootstrap

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE=6                # OOM fix: decoder_res_blocks=3 raises peak memory
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=1e-4
LR_DISC=1e-4                # FactorVAE discriminator LR
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
[[ -f "$D4_CKPT"          ]] || die "D4 checkpoint not found: $D4_CKPT"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Environment ────────────────────────────────────────────────────────────────
banner "D5 GAN v2 — bug-fixed re-run"
cd "$WORKDIR"
mkdir -p logs

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # contiguous block avoids fragmentation OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # ~3.6 GB headroom on 24 GB card
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=1               # GPU 1 on mscluster106 (GPU 0 in use, GPU 1 free — 48 GB RTX 8000)

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}"

kv "Node / GPU"          "$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
kv "Exp"                 "$EXP_NAME  (epochs 120 → ${EPOCHS})"
kv "D4 resume"           "$D4_CKPT"
kv "Batch / lr_vae"      "${BATCH_SIZE} / ${LR_VAE}"
kv "decoder_res_blocks"  "$DECODER_RES_BLOCKS"
kv "weight_gan"          "${WEIGHT_GAN}  (BUG FIX: was 0.5 → 4.6× imbalance)"
kv "gan_start_step"      "${GAN_START_STEP}  (phase-local, BUG FIX: was global_step)"
kv "lr_patch_disc"       "${LR_PATCH_DISC}  (BUG FIX: was 1e-4)"
kv "disc_r1_penalty"     "${DISC_R1_PENALTY}  (BUG FIX: R1 penalty, was 0)"
kv "weight_rec"          "$WEIGHT_REC"
kv "weight_perceptual"   "$WEIGHT_PERCEPTUAL  tv=$WEIGHT_TV"
kv "weight_mi_factor"    "$WEIGHT_MI_FACTOR  masked_rec=$WEIGHT_MASKED_REC"
kv "weight_bbox_attn"    "$WEIGHT_BBOX_ATTN  (raised from 0.05)"
kv "bbox_query_mix"      "$BBOX_QUERY_MIX  (pure Gaussian prior, D5)"
kv "kl_free_bits"        "$KL_FREE_BITS"
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
  --resume                 "$D4_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "D5 v2 complete"
