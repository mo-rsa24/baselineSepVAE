#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_baseline_ldm}"
ENV_NAME="${ENV_NAME:-jaxstack}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"

LDM_CKPT="${LDM_CKPT:-$(find "${OUTPUT_ROOT}" -maxdepth 3 -name 'checkpoint_final.pkl' | sort | tail -1)}"
[[ -n "${LDM_CKPT}" ]] || { echo "No baselineLDM checkpoint found under ${OUTPUT_ROOT}"; exit 1; }
[[ -f "${LDM_CKPT}" ]] || { echo "LDM checkpoint not found: ${LDM_CKPT}"; exit 1; }

RUN_DIR="$(cd "$(dirname "${LDM_CKPT}")/.." && pwd)"
RUN_META="${RUN_DIR}/run_meta.json"
SEPVAE_CKPT="${SEPVAE_CKPT:-}"
if [[ -z "${SEPVAE_CKPT}" && -f "${RUN_META}" ]]; then
  SEPVAE_CKPT="$(python - <<'PY' "${RUN_META}"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    meta = json.load(f)
print(meta.get('sepvae_ckpt', ''))
PY
)"
fi
[[ -n "${SEPVAE_CKPT}" ]] || { echo "Could not infer SepVAE checkpoint. Set SEPVAE_CKPT explicitly."; exit 1; }
[[ -f "${SEPVAE_CKPT}" ]] || { echo "SepVAE checkpoint not found: ${SEPVAE_CKPT}"; exit 1; }

LATENTS_DIR="${LATENTS_DIR:-}"
if [[ -z "${LATENTS_DIR}" && -f "${RUN_META}" ]]; then
  LATENTS_DIR="$(python - <<'PY' "${RUN_META}"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    meta = json.load(f)
print(meta.get('preencoded_latents_dir', ''))
PY
)"
fi

SAMPLE_OUT="${SAMPLE_OUT:-${OUTPUT_ROOT}/sample_preview}"
MODE="${MODE:-sample}"
NUM_RECORDS="${NUM_RECORDS:-8}"
NSTEPS="${NSTEPS:-250}"

cd "${WORKDIR}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORM}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u

ARGS=(
  run/sample_baseline_ldm.py
  --ldm_ckpt "${LDM_CKPT}"
  --sepvae_ckpt "${SEPVAE_CKPT}"
  --output_dir "${SAMPLE_OUT}"
  --mode "${MODE}"
  --num_records "${NUM_RECORDS}"
  --n_steps "${NSTEPS}"
)

if [[ -n "${LATENTS_DIR}" ]]; then
  ARGS+=( --preencoded_latents_dir "${LATENTS_DIR}" )
fi

python -u "${ARGS[@]}"
