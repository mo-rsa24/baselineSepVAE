#!/bin/bash
# =============================================================================
# mini_sep_vae.slurm — SLURM job script for MiniSepVAE ablation experiments
#
# Usage:
#   # Stage 1-5 (synthetic, fast):
#   sbatch slurm_scripts/mini_sep_vae.slurm launchers/single_runs/mini_vae/stage1_recon.sh
#
#   # Stage 6 (VinBigData, longer):
#   sbatch --time=04:00:00 slurm_scripts/mini_sep_vae.slurm launchers/single_runs/mini_vae/stage6_vinbig128.sh
#
# The first positional argument is the launcher script to run.
# If omitted, stage1_recon.sh is used as a smoke test.
# =============================================================================

#SBATCH --job-name=mini_sepvae
#SBATCH --exclude=mscluster65,mscluster76,mscluster59,mscluster48,mscluster82,mscluster75,mscluster72
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=results/slurm_logs/%x_%j.out
#SBATCH --error=results/slurm_logs/%x_%j.err

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
# Suspend strict mode around conda activation — conda deactivation scripts
# reference CUDAARCHS_BACKUP before it is ever set, which trips -u.
set +eu
source ~/.bashrc
conda activate jaxstack
set -euo pipefail

# Navigate to repo root (script is submitted from repo root, but be safe)
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "Repo root: ${REPO_ROOT}"

# Ensure results/slurm_logs exists
# mkdir -p results/slurm_logs

# Make sure the miniSepVAE package is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --------------------------------------------------------------------------- #
# Pick launcher script
# --------------------------------------------------------------------------- #
LAUNCHER="${1:-launchers/single_runs/mini_vae/stage1_recon.sh}"

if [[ ! -f "${LAUNCHER}" ]]; then
    echo "ERROR: launcher script not found: ${LAUNCHER}"
    exit 1
fi

echo "========================================"
echo "SLURM job  : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Launcher   : ${LAUNCHER}"
echo "Start time : $(date)"
echo "========================================"

chmod +x "${LAUNCHER}"
bash "${LAUNCHER}"

echo "========================================"
echo "Finished   : $(date)"
echo "========================================"
