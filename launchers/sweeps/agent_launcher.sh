#!/usr/bin/env bash
set -euo pipefail

# This script is called by the `wandb agent`.
# It parses hyperparameters, generates tags, and submits a SLURM job.

# --- Fixed Defaults ---
export ENV_NAME="jax115"
# NOTE: DATA_ROOT is now expected from the sweep .yaml file

# --- Parse W&B agent args and export as env vars ---
for arg in "$@"; do
    if [[ "$arg" == --program* ]]; then
        PY_PROGRAM="${arg#*=}"
        continue
    fi
    VAR_NAME=$(echo "$arg" | cut -d'=' -f1 | sed 's/--//' | tr '[:lower:]' '[:upper:]')
    VAR_VALUE="${arg#*=}"
    export "$VAR_NAME"="$VAR_VALUE"
done

# --- Determine SLURM script to use ---
SLURM_SCRIPT=""
if [[ "$PY_PROGRAM" == "run/autoencoder.py" ]]; then
    SLURM_SCRIPT="slurm_scripts/cxr_ae.slurm"
    export WANDB_PROJECT="cxr-vae-sweeps"
elif [[ "$PY_PROGRAM" == "run/ldm.py" ]]; then
    SLURM_SCRIPT="slurm_scripts/cxr_ldm.slurm"
    export WANDB_PROJECT="cxr-ldm-sweeps"
else
    echo "ERROR: Unknown program '$PY_PROGRAM'" >&2
    exit 1
fi

# --- Generate W&B Tags from Sweep Parameters ---
# This section automatically creates descriptive tags for better organization.
TAGS="sweep"
TRAINING_MODE="full_train" # Default mode

# Determine training mode from overfit flags
if [[ "${OVERFIT_ONE}" == "true" ]]; then TRAINING_MODE="overfit_one";
elif [[ "${OVERFIT_K}" -gt 0 ]]; then TRAINING_MODE="overfit_${OVERFIT_K}";
fi

# Add task and mode to tags
[[ -n "${TASK}" ]] && TAGS+=",${TASK,,}" # Add task, e.g., ",tb"
TAGS+=",${TRAINING_MODE}"

# If this is a z_channels sweep, add that tag too
[[ -n "${Z_CHANNELS}" ]] && TAGS+=",z${Z_CHANNELS}"

export WANDB_TAGS="$TAGS"
# --- End of Tag Generation ---


# --- SPECIAL LOGIC for LDM Z_CHANNELS SWEEP ---
# (This part remains unchanged)
# If Z_CHANNELS and VAE_RUNS_ROOT are set, find the corresponding VAE checkpoint.
if [[ -n "${Z_CHANNELS:-}" && -n "${VAE_RUNS_ROOT:-}" && "$PY_PROGRAM" == "run/ldm.py" ]]; then
    echo ">>> LDM z_channels sweep detected. Finding VAE artifacts..."
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Find the most recent run directory for this task and z_channels
    # Example search pattern: "runs/ae_tb_z1_*"
    SEARCH_PATTERN="${VAE_RUNS_ROOT}/ae_${TASK_LOWER}_full_kl_*_zchannels_${Z_CHANNELS}/*"

    # Find the latest directory matching the pattern
    VAE_RUN_DIR=$(find $VAE_RUNS_ROOT -type d -name "ae_${TASK_LOWER}_full_kl_*_zchannels_${Z_CHANNELS}" | sort -r | head -n 1)

    if [[ -z "$VAE_RUN_DIR" || ! -d "$VAE_RUN_DIR" ]]; then
        echo "ERROR: Could not find a VAE run directory for TASK=${TASK} and Z_CHANNELS=${Z_CHANNELS}" >&2
        echo "Searched with pattern: ${SEARCH_PATTERN}" >&2
        exit 1
    fi

    export AE_CKPT_PATH="${VAE_RUN_DIR}/ckpts/last.flax"
    export AE_CONFIG_PATH="${VAE_RUN_DIR}/run_meta.json"
    SCALE_FACTOR_FILE="${VAE_RUN_DIR}/latent_scale_factor.txt"

    if [[ ! -f "$SCALE_FACTOR_FILE" ]]; then
        echo "ERROR: latent_scale_factor.txt not found in ${VAE_RUN_DIR}" >&2
        echo "Please run scripts/find_latent_scale.py for this VAE first." >&2
        exit 1
    fi
    export LATENT_SCALE_FACTOR=$(cat "$SCALE_FACTOR_FILE")

    echo "Found VAE Checkpoint: $AE_CKPT_PATH"
    echo "Found Latent Scale Factor: $LATENT_SCALE_FACTOR"
fi

# --- Generate a Unique Run Name for W&B ---
export RUN_NAME="sweep-${TASK,,}-${TRAINING_MODE}-$(date +%s)"

# --- Submit Job ---
echo "LAUNCHING SWEEP RUN with TAGS: ${WANDB_TAGS}"
sbatch "$SLURM_SCRIPT"
