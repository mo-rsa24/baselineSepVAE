#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }

# ----------------------------
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
DISEASE="${DISEASE:?Must set DISEASE (cardiomegaly or effusion)}"
OUTPUT_DIR="${OUTPUT_DIR:?Must set OUTPUT_DIR}"
SEPVAE_CKPT="${SEPVAE_CKPT:?Must set SEPVAE_CKPT}"
DICOM_DIR="${DICOM_DIR:-/datasets/mmolefe/vinbigdata/train}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/train.csv}"
IMG_SIZE="${IMG_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
EXCLUDE_OVERLAP="${EXCLUDE_OVERLAP:-0}"
USE_MEAN="${USE_MEAN:-0}"
RESUME="${RESUME:-0}"
ENV_NAME="${ENV_NAME:-jaxstack}"

# ----------------------------
# Environment
# ----------------------------
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Pre-encoding runs on CPU — keep JAX off the GPU so it doesn't fight LDM jobs
export JAX_PLATFORMS=cpu
export TF_CPP_MIN_LOG_LEVEL=3

set +u
source ~/.bashrc
mamba activate "$ENV_NAME"
set -u

# ----------------------------
# Info
# ----------------------------
banner "SepVAE Pre-encoding"
kv "JOB_ID"      "${SLURM_JOB_ID:-<none>}"
kv "Node"        "$(hostname)"
kv "DISEASE"     "$DISEASE"
kv "SEPVAE_CKPT" "$SEPVAE_CKPT"
kv "OUTPUT_DIR"  "$OUTPUT_DIR"
kv "IMG_SIZE"    "$IMG_SIZE"
kv "BATCH_SIZE"  "$BATCH_SIZE"

# ----------------------------
# Build args
# ----------------------------
ARGS=(
  --sepvae_ckpt  "$SEPVAE_CKPT"
  --disease      "$DISEASE"
  --output_dir   "$OUTPUT_DIR"
  --dicom_dir    "$DICOM_DIR"
  --csv_path     "$CSV_PATH"
  --img_size     "$IMG_SIZE"
  --batch_size   "$BATCH_SIZE"
  --num_workers  "$NUM_WORKERS"
  --seed         "$SEED"
)

[[ "$EXCLUDE_OVERLAP" == "1" ]] && ARGS+=( --exclude_cross_disease_overlap )
[[ "$USE_MEAN"        == "1" ]] && ARGS+=( --use_mean )
[[ "$RESUME"          == "1" ]] && ARGS+=( --resume )

banner "Run"
srun python -u scripts/preencode_sepvae_latents.py "${ARGS[@]}"
