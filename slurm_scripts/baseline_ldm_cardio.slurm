#!/usr/bin/env bash
# baselineLDM cardio SLURM job
#
# Submit directly:
#   SEPVAE_PHASE=d4 sbatch --nodelist=mscluster107 slurm_scripts/baseline_ldm_cardio.slurm
#
# Or use the wrapper:
#   NODELIST=mscluster107 SEPVAE_PHASE=d4 \
#     ./launchers/single_runs/ldm/sbatch_train_baseline_ldm_cardio.sh

#SBATCH --job-name=baseline-ldm-cardio
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/baseline-ldm-cardio-%j.out
#SBATCH --error=logs/baseline-ldm-cardio-%j.err

set -euo pipefail

resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then
        printf '%s\n' "${WORKDIR}"
        return
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local job_cmd
        job_cmd=$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
        if [[ -n "${job_cmd}" && -f "${job_cmd}" ]]; then
            cd "$(dirname "${job_cmd}")/.." && pwd
            return
        fi
    fi

    pwd
}

WORKDIR="$(resolve_workdir)"
cd "${WORKDIR}"
mkdir -p logs

echo "== baselineLDM cardio SLURM job =="
echo "  host:           $(hostname)"
echo "  workdir:        ${WORKDIR}"
echo "  phase:          ${SEPVAE_PHASE:-auto}"
echo "  partition:      ${SLURM_JOB_PARTITION:-unknown}"
echo "  nodelist:       ${SLURM_JOB_NODELIST:-unknown}"
echo "  output_root:    ${OUTPUT_ROOT:-${WORKDIR}/runs_baseline_ldm}"

exec ./launchers/single_runs/ldm/train_baseline_ldm_cardio.sh
