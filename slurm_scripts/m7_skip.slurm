#!/usr/bin/env bash
# M7 — UNet skip connections + wider z_common  (mask curriculum, stage 7)
#
# Resumes from M3 final checkpoint.
# Parallel to D7 in the original bbox curriculum.
#
# Architectural changes vs M3:
#
#   Change 1 — UNet skip connections (already in sep_vae_v2.py since D7)
#       enc_layer3 (16×16×1024) injected at decoder scale i=4
#       enc_layer2 (32×32×512)  injected at decoder scale i=3
#       New params: skip3_fuse, skip2_fuse (Conv 1×1) — freshly initialised
#       All other encoder/decoder weights restore from M3.
#
#   Change 2 — z_channels_common 16 → 32
#       Doubles common-latent capacity: 4096 → 8192 values.
#       Encoder head_common and decoder z_proj are re-initialised (shape change).
#       All other encoder/decoder weights restore from M3.
#
# CRITICAL — skip connection bypass hazard (unique to mask curriculum):
#   enc_layer3 comes from the SHARED encoder (before z_common/z_disease split).
#   At 16×16, it carries cardiac structure directly into the decoder.
#   Without mitigation, the decoder uses this skip to reconstruct the cardiac
#   region, ignoring z_cardio — z_cardio variation has no visual effect.
#
#   Mitigation implemented in sep_vae_v2.py (SepVAEDecoderV2):
#       cardiac_gate = 1.0 - heart_mask[..., None]   # 0 in cardiac region
#       layer3_gated = enc_layer3 * cardiac_gate
#   This zeros the cardiac region in enc_layer3 before fusion, forcing the
#   decoder to use z_cardio for that region while non-cardiac anatomy still
#   benefits from the skip connection.
#
#   The gate is ONLY active when heart_mask is provided (--chexmask_csv set).
#   Without chexmask_csv the gate is off — equivalent to standard D7.
#
# What to watch in M7 (same as D7, plus one extra):
#   - Do not judge M7 quality before epoch 20 from resume — skip3_fuse,
#     skip2_fuse, and z_common head are randomly initialised → transient regression
#   - Monitor loss/reconstruction — expect a transient increase in first 10 epochs
#   - Monitor metrics/z_cardio_norm_ratio — if skip connections absorb cardiac
#     info despite gating, this ratio will drop; investigate if < 1.5
#   - Monitor loss/ctr_reg — should stay < 0.05; a spike means z_cardio is
#     losing its scalar representation during the architectural transition
#
# Submit:
#   sbatch slurm_scripts/m7_skip.slurm
#
# Override M3 checkpoint:
#   M3_CKPT=/path/to/ckpt.pkl sbatch slurm_scripts/m7_skip.slurm

#SBATCH --job-name=sepvae-m7-skip
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/sepvae-m7-skip-%j.out
#SBATCH --error=logs/sepvae-m7-skip-%j.err

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

M3_CKPT="${M3_CKPT:-}"
[[ -n "$M3_CKPT" ]] || die "M3_CKPT not set. Example: M3_CKPT=runs_sepvae/m3_gan-<ts>/checkpoints/checkpoint_final.pkl"

# ── Experiment ─────────────────────────────────────────────────────────────────
EXP_NAME="m7_skip"
EPOCHS=200             # M3 ends at 130, add 70 epochs for skip + z_common=32 to adapt
KL_WARMUP_EPOCHS=0

# ── Model — architectural changes vs M3 ───────────────────────────────────────
IMG_SIZE=256
Z_COMMON=32            # Doubled from 16 — encoder head_common re-initialised
Z_DISEASE=16
ATTN_QUERY_DIM=256
ATTN_HEADS=4
DECODER_RES_BLOCKS=3
BBOX_QUERY_MIX=1.0
BBOX_DROPOUT_PROB=0.0

# ── Loss weights ──────────────────────────────────────────────────────────────
WEIGHT_REC=1.0
WEIGHT_KL_COMMON=1e-4
WEIGHT_KL_DISEASE=5e-5
KL_FREE_BITS=0.5
WEIGHT_MI_FACTOR=1.0
WEIGHT_BBOX_ATTN=0.0       # No bbox supervision ever
WEIGHT_CTR_REG=1.0
WEIGHT_CARDIO_SUPCON=0.05
WEIGHT_PERCEPTUAL=0.0       # Disabled: skip connections + GAN replace perceptual;
                             # also eliminates CheSS aliasing at the transition point
WEIGHT_GAN=0.1
WEIGHT_TV=0.005
WEIGHT_MASKED_REC=1.5
SIGMA_INACTIVE=0.1

# ── GAN — shorter phase-local warm-up (skip connections accelerate convergence) ─
GAN_START_STEP=500
LR_PATCH_DISC=1e-4
DISC_R1_PENALTY=0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE=4           # Skip connections raise peak memory — reduce from 6
NUM_WORKERS=8
EVAL_NUM_WORKERS=0
SEED=0
LR_VAE=5e-5            # Conservative — only skip projections + z_common head are new
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
[[ -f "$M3_CKPT"          ]] || die "M3 checkpoint not found: $M3_CKPT"

banner "M7 Skip Connections + z_common=32  (mask curriculum)"
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
kv "Exp"                 "$EXP_NAME  (ep130 → ${EPOCHS})"
kv "M3 resume"           "$M3_CKPT"
kv "CheXmask CSV"        "$CHEXMASK_CSV"
kv "z_common"            "${Z_COMMON}  (doubled from 16 — head_common re-init)"
kv "Skip connections"    "layer3 (16×16) + layer2 (32×32) → decoder"
kv "BYPASS MITIGATION"   "cardiac mask-gate on layer3 skip (zeros heart region)"
kv "weight_perceptual"   "0.0  (skip+GAN replace CheSS perceptual)"
kv "weight_ctr_reg"      "${WEIGHT_CTR_REG}  (active — z_cardio must keep CTR signal)"
kv "Batch / lr_vae"      "${BATCH_SIZE} / ${LR_VAE}  (batch reduced for skip memory)"
kv "WATCH"               "z_cardio_norm_ratio must stay > 1.5 after ep20"

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
  --resume                 "$M3_CKPT" \
  \
  $( [[ "$WANDB" == "1" ]] && echo "--wandb --wandb_project $WANDB_PROJECT" )

ok "M7 complete — mask curriculum with skip connections finished"
ok "Traversal: conda run -n jaxstack python scripts/latent_traversal_medsam.py \\"
ok "  --checkpoint runs_sepvae/m7_skip-<ts>/checkpoints/checkpoint_final.pkl \\"
ok "  --output results/traversal_m7/"
