#!/usr/bin/env bash
# D5 — Reconstruction quality at 256×256
#
# Purpose : Completely remove artifacts and ensure crisp reconstructions while
#           maintaining clear latent separation between Normal and Cardiomegaly.
#
# Changes from D4:
#   Architecture (sep_vae_v2.py):
#     Fix A — Layer4BranchGN stride=1 throughout (no stride-2 aliasing round-trip)
#     Fix B — val_proj removed from BboxCrossAttnHead (was wasted, never used)
#     Fix D — decoder ch_mults (64→128, 128, 256, 512, 512): 128ch at 256×256
#     Fix E — SelfAttention2D at 32×32 in decoder (global cardiac silhouette coherence)
#     Fix F — SE reduction 16→8 (hidden=8 at 128ch, not 4 at 64ch)
#   Loss (Fix C, Fix G, Fix H, Fix J):
#     Fix C — weight_bbox_attn=0.1  re-enables bbox attention supervision
#     Fix G — weight_rec=2.0        rebalances MSE vs perceptual (was ~0.005 contribution)
#     Fix H — weight_gan=0.5        PatchGAN hinge generator loss, start at step 5000
#     Fix H — weight_tv=1e-3        total variation suppresses stripe artifacts
#
# Prerequisite: D4 gate passed.
# Resumes:      d4_perceptual-20260317-043603/checkpoints/checkpoint_epoch0160.pkl (default)
#
# Usage:
#   ./launchers/runpod/run_d5_recon.sh
#   RESUME=/workspace/runs_sepvae/d4_perceptual-XXXXXXXX/checkpoints/checkpoint_epoch0160.pkl \
#     ./launchers/runpod/run_d5_recon.sh

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv()    { printf "  ${CYN}%-28s${RST} %s\n" "$1" "$2"; }
ok()    { printf "${GRN}** %s${RST}\n" "$*"; }
die()   { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-/workspace/baselineSepVAE}"
DATA_DIR="${DATA_DIR:-/workspace/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/workspace/vinbigdata/cache_npy/train_filtered.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/runs_sepvae}"
RESUME="${RESUME:-${OUTPUT_ROOT}/d4_perceptual-20260317-043603/checkpoints/checkpoint_epoch0160.pkl}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/workspace/chess/pretrained_weights.pth.tar}"

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
LR_DISC="${LR_DISC:-1e-4}"
LR_PATCH_DISC="${LR_PATCH_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ── Loss weights ──────────────────────────────────────────────────────────────
# Fix G: weight_rec raised from 1.0 → 2.0 to rebalance MSE vs perceptual.
#        In D4, MSE contributed ~0.005 vs perceptual ~0.028 — MSE was the
#        smallest signal. 2.0 doubles its relative weight without over-blurring.
WEIGHT_REC="${WEIGHT_REC:-2.0}"
WEIGHT_KL_COMMON="${WEIGHT_KL_COMMON:-1e-4}"
WEIGHT_KL_DISEASE="${WEIGHT_KL_DISEASE:-1e-4}"
WEIGHT_MI_FACTOR="${WEIGHT_MI_FACTOR:-1.0}"
# Fix C: bbox attention supervision re-enabled.
#        Was 0.0 throughout D2/D3/D4 — attention maps were free to drift.
#        0.1 is small enough to not destabilise MI separation.
WEIGHT_BBOX_ATTN="${WEIGHT_BBOX_ATTN:-0.1}"
# Perceptual: keep D4 weight unchanged (layers 1–3 only, layer4 excluded).
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.05}"
# Fix H: PatchGAN hinge generator loss.
#        Sharpens textures and suppresses blurriness that MSE+perceptual alone
#        cannot fix. Activated after gan_start_step to let VAE stabilise first.
WEIGHT_GAN="${WEIGHT_GAN:-0.5}"
# Fix H: Total variation loss.
#        Directly suppresses horizontal stripe artifacts from strided perceptual
#        gradients. Small weight — just enough to damp the stripes.
WEIGHT_TV="${WEIGHT_TV:-1e-3}"
# Fix H: GAN warmup. PatchGAN activates after this many global steps.
#        Lets the VAE establish a reasonable reconstruction before adversarial
#        pressure begins. ~5000 steps ≈ first few epochs at batch_size=16.
GAN_START_STEP="${GAN_START_STEP:-5000}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"

# ── Logging ───────────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-d5_recon}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -d "$WORKDIR"          ]] || die "WORKDIR not found: $WORKDIR"
[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$RESUME"           ]] || die "Checkpoint not found: $RESUME — run D4 first"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Environment ───────────────────────────────────────────────────────────────
banner "Environment"
cd "$WORKDIR"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0

set +u
source ~/.bashrc
conda activate "$ENV_NAME" 2>/dev/null || true
set -u
ok "Activated: ${ENV_NAME}"

kv "Phase"               "D5 — Reconstruction quality at 256×256"
kv "Architecture fixes"  "stride=1 branches, 128ch decoder, attn@32×32, SE=8, no val_proj"
kv "Losses"              "MSE(${WEIGHT_REC}) + KL + MI(${WEIGHT_MI_FACTOR}) + percep(${WEIGHT_PERCEPTUAL}) + GAN(${WEIGHT_GAN}) + TV(${WEIGHT_TV}) + bbox(${WEIGHT_BBOX_ATTN})"
kv "GAN start step"      "$GAN_START_STEP"
kv "Batch / Epochs"      "${BATCH_SIZE} / ${EPOCHS}"
kv "Resume"              "$RESUME"
kv "Exp name"            "$EXP_NAME"
kv "Gate (O2 complete)"  "No stripes; sharp ribs/vessels; sil_cardio >= 0.50; PD_acc 0.55-0.65"

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
  run/train_sep_vae.py
  --dicom_dir            "$DATA_DIR"
  --csv_path             "$CSV_PATH"
  --use_cache

  --model_version        v2
  --use_bbox_cross_attn
  --img_size             "$IMG_SIZE"
  --z_channels_common    "$Z_COMMON"
  --z_channels_disease   "$Z_DISEASE"
  --attn_query_dim       "$ATTN_QUERY_DIM"
  --attn_heads           "$ATTN_HEADS"

  --batch_size           "$BATCH_SIZE"
  --epochs               "$EPOCHS"
  --num_workers          "$NUM_WORKERS"
  --seed                 "$SEED"
  --kl_warmup_epochs     0

  --lr_vae               "$LR_VAE"
  --lr_disc              "$LR_DISC"
  --lr_patch_disc        "$LR_PATCH_DISC"
  --weight_decay         "$WEIGHT_DECAY"
  --grad_clip            "$GRAD_CLIP"

  --weight_rec           "$WEIGHT_REC"
  --weight_kl_common     "$WEIGHT_KL_COMMON"
  --weight_kl_disease    "$WEIGHT_KL_DISEASE"
  --weight_mi_factor     "$WEIGHT_MI_FACTOR"
  --weight_bbox_attn     "$WEIGHT_BBOX_ATTN"
  --weight_perceptual    "$WEIGHT_PERCEPTUAL"
  --weight_gan           "$WEIGHT_GAN"
  --weight_tv            "$WEIGHT_TV"
  --gan_start_step       "$GAN_START_STEP"
  --sigma_inactive       "$SIGMA_INACTIVE"

  --chess_checkpoint     "$CHESS_CHECKPOINT"
  --perceptual_only                          # CheSS for perceptual loss only, not encoder

  --resume               "$RESUME"
  --output_root          "$OUTPUT_ROOT"
  --exp_name             "$EXP_NAME"
  --save_every           "$SAVE_EVERY"
  --sample_every         "$SAMPLE_EVERY"
  --manifold_every       "$MANIFOLD_EVERY"
)

if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
fi

# ── Launch ────────────────────────────────────────────────────────────────────
banner "Launching D5"
python -u "${ARGS[@]}"
ok "D5 complete — check recon grid for stripe removal and sharpness vs D4 in W&B"
