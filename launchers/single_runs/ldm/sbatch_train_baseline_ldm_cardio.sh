#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${WORKDIR}/slurm_scripts/baseline_ldm_cardio.slurm}"
PARTITION="${PARTITION:-bigbatch}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
JOB_NAME="${JOB_NAME:-baseline-ldm-cardio}"
NODELIST="${NODELIST:-}"
SEPVAE_PHASE="${SEPVAE_PHASE:-d4}"

[[ -f "${SLURM_SCRIPT}" ]] || { echo "SLURM script not found: ${SLURM_SCRIPT}"; exit 1; }

mkdir -p "${WORKDIR}/logs"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --job-name="${JOB_NAME}"
  --output="${WORKDIR}/logs/${JOB_NAME}-%j.out"
  --error="${WORKDIR}/logs/${JOB_NAME}-%j.err"
  --export=ALL,WORKDIR="${WORKDIR}",SEPVAE_PHASE="${SEPVAE_PHASE}"
)

if [[ -n "${NODELIST}" ]]; then
  SBATCH_ARGS+=( --nodelist="${NODELIST}" )
fi

echo "== sbatch baselineLDM cardio =="
echo "  workdir:     ${WORKDIR}"
echo "  phase:       ${SEPVAE_PHASE}"
echo "  partition:   ${PARTITION}"
echo "  nodelist:    ${NODELIST:-<scheduler choice>}"
echo "  job name:    ${JOB_NAME}"

sbatch "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}"
