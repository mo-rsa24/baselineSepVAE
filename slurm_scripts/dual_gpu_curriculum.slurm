#!/usr/bin/env bash
# dual_gpu_curriculum.slurm — Run two independent D1→D2→D4 curriculum experiments
# simultaneously on the two GPUs of mscluster107 (biggpu).
#
# Why two runs?  The D1 KL-explosion failure was caused by non-deterministic data
# ordering hitting an unlucky first-epoch basin.  We test robustness across two
# seeds (0 and 1) with the fixed weight_kl_common=5e-4 to confirm the fix is
# seed-independent.  Both outputs are gated by the D2 silhouette gate; whichever
# passes gate first gives us the D4 checkpoint to resume D5.
#
# Configs:
#   GPU 0 — seed=0, output: ${OUTPUT_ROOT}/run_s0
#   GPU 1 — seed=1, output: ${OUTPUT_ROOT}/run_s1
#
# Submit:
#   sbatch slurm_scripts/dual_gpu_curriculum.slurm
#
# Override start phase (e.g. already have D1, resume from D2):
#   START_PHASE=d2 sbatch slurm_scripts/dual_gpu_curriculum.slurm

#SBATCH --job-name=sepvae-dual
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/dual-curriculum-%j.out
#SBATCH --error=logs/dual-curriculum-%j.err

set -euo pipefail

CYN=$(printf '\033[36m'); GRN=$(printf '\033[32m')
RED=$(printf '\033[31m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
banner() { printf "\n${CYN}${BLD}======  %s  ======${RST}\n\n" "$*"; }
ok()     { printf "${GRN}** %s${RST}\n" "$*"; }
die()    { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Resolve working directory ─────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    job_cmd=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
              | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
    [[ -n "$job_cmd" && -f "$job_cmd" ]] && cd "$(dirname "$job_cmd")/.."
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
WORKDIR="${WORKDIR:-$PWD}"
cd "$WORKDIR"

# ── Environment ───────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-jaxstack}"
mkdir -p logs

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

# ── GPU check ────────────────────────────────────────────────────────────────
nvidia-smi -L || die "No GPUs visible"
GPU_COUNT=$(nvidia-smi -L | wc -l)
[[ "$GPU_COUNT" -ge 2 ]] || die "Expected ≥2 GPUs on mscluster107, got ${GPU_COUNT}"
ok "GPUs on this node: ${GPU_COUNT}"

# ── Shared settings ───────────────────────────────────────────────────────────
START_PHASE="${START_PHASE:-d1}"
END_PHASE="${END_PHASE:-d4}"
BASE_OUTPUT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"
WANDB="${WANDB:-1}"
JOB_TAG="${SLURM_JOB_ID:-local}"

banner "Launching dual curriculum  (${START_PHASE^^} → ${END_PHASE^^})"
printf "  GPU 0  seed=0  output=%s/run_s0\n" "$BASE_OUTPUT"
printf "  GPU 1  seed=1  output=%s/run_s1\n" "$BASE_OUTPUT"
printf "  Logs:  logs/dual-%s-gpu{0,1}.out\n" "$JOB_TAG"

# ── Run A — GPU 0, seed=0 ────────────────────────────────────────────────────
(
    export CUDA_VISIBLE_DEVICES=0
    export OUTPUT_ROOT="${BASE_OUTPUT}/run_s0"
    export SEED=0
    export WANDB="$WANDB"
    export WORKDIR="$WORKDIR"
    export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_CPP_MIN_LOG_LEVEL=3
    bash run_curriculum.sh "$START_PHASE" "$END_PHASE"
) > "logs/dual-${JOB_TAG}-gpu0.out" 2>&1 &
PID_A=$!
ok "GPU 0 (seed=0) started — PID ${PID_A}"

# Small stagger: avoids both processes hitting dataset validation and W&B init
# at the exact same moment.
sleep 10

# ── Run B — GPU 1, seed=1 ────────────────────────────────────────────────────
(
    export CUDA_VISIBLE_DEVICES=1
    export OUTPUT_ROOT="${BASE_OUTPUT}/run_s1"
    export SEED=1
    export WANDB="$WANDB"
    export WORKDIR="$WORKDIR"
    export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_CPP_MIN_LOG_LEVEL=3
    bash run_curriculum.sh "$START_PHASE" "$END_PHASE"
) > "logs/dual-${JOB_TAG}-gpu1.out" 2>&1 &
PID_B=$!
ok "GPU 1 (seed=1) started — PID ${PID_B}"

# ── Wait and report ───────────────────────────────────────────────────────────
banner "Both runs in flight — waiting"

STATUS_A=0; STATUS_B=0
wait "$PID_A" || STATUS_A=$?
ok "GPU 0 finished (exit=${STATUS_A})"
wait "$PID_B" || STATUS_B=$?
ok "GPU 1 finished (exit=${STATUS_B})"

banner "Final results"
if [[ "$STATUS_A" -eq 0 ]]; then
    ok "seed=0: PASSED — D4 checkpoint in ${BASE_OUTPUT}/run_s0"
else
    printf "${RED}!! seed=0: FAILED (exit %d) — logs/dual-%s-gpu0.out${RST}\n" \
        "$STATUS_A" "$JOB_TAG" >&2
fi
if [[ "$STATUS_B" -eq 0 ]]; then
    ok "seed=1: PASSED — D4 checkpoint in ${BASE_OUTPUT}/run_s1"
else
    printf "${RED}!! seed=1: FAILED (exit %d) — logs/dual-%s-gpu1.out${RST}\n" \
        "$STATUS_B" "$JOB_TAG" >&2
fi

[[ "$STATUS_A" -eq 0 || "$STATUS_B" -eq 0 ]] \
    || die "Both runs failed — re-examine D1 KL fix before running D5"

ok "Dual curriculum complete — at least one D4 checkpoint is ready for D5"
