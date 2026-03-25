#!/usr/bin/env bash
# Reproduction curriculum for the known-good SepVAE run sequence on biggpu.
#
# This script mirrors the successful RunPod hyperparameters as closely as
# possible. The intended phases are:
#   d0 -> d1 -> d2 -> d4
#
# Differences from the original environment:
# - cluster paths instead of /workspace/...
# - SLURM / biggpu runtime
#
# Submit:
#   sbatch run_curriculum_e1eca19_biggpu.sh
#   sbatch run_curriculum_e1eca19_biggpu.sh d1
#   sbatch run_curriculum_e1eca19_biggpu.sh d2
#   sbatch run_curriculum_e1eca19_biggpu.sh d4

#SBATCH --job-name=sepvae-e1eca19
#SBATCH --partition=biggpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/e1eca19-curriculum-%j.out
#SBATCH --error=logs/e1eca19-curriculum-%j.err

set -euo pipefail

resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then
        printf '%s\n' "$WORKDIR"
        return
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local job_cmd
        job_cmd=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
        if [[ -n "$job_cmd" && -f "$job_cmd" ]]; then
            cd "$(dirname "$job_cmd")" && pwd
            return
        fi
    fi
    if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
        cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
        return
    fi
    pwd
}

WORKDIR="$(resolve_workdir)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae_e1eca19}"
ENV_NAME="${ENV_NAME:-jaxstack}"

CYN=$(printf '\033[36m'); GRN=$(printf '\033[32m')
RED=$(printf '\033[31m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
banner() { printf "\n${CYN}${BLD}======  %s  ======${RST}\n\n" "$*"; }
ok()     { printf "${GRN}** %s${RST}\n" "$*"; }
die()    { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

mkdir -p "${WORKDIR}/logs"
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
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

banner "GPU Preflight"
printf "  SLURM_JOB_ID=%s\n" "${SLURM_JOB_ID:-unset}"
printf "  CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || die "nvidia-smi is present but no GPU is visible to this job. Re-submit with --gres=gpu:1."
else
    die "nvidia-smi not found on this node. The job does not appear to have a usable GPU environment."
fi

DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"
SEED="${SEED:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

[[ -d "$DATA_DIR" ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH" ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS checkpoint not found: $CHESS_CHECKPOINT"

START_PHASE="${1:-d0}"
END_PHASE="${2:-d4}"
ALL_PHASES=(d0 d1 d2 d4)

phase_ge() {
    local -A rank=([d0]=0 [d1]=1 [d2]=2 [d4]=3)
    [[ "${rank[$1]}" -ge "${rank[$2]}" ]]
}

phase_le() {
    local -A rank=([d0]=0 [d1]=1 [d2]=2 [d4]=3)
    [[ "${rank[$1]}" -le "${rank[$2]}" ]]
}

find_latest_ckpt() {
    local hit
    hit=$(find "$OUTPUT_ROOT" -maxdepth 3 -name "checkpoint_final.pkl" | grep "/${1}-" | sort | tail -1 || true)
    echo "$hit"
}

require_ckpt() {
    local ckpt
    ckpt=$(find_latest_ckpt "$1")
    [[ -n "$ckpt" ]] || die "No checkpoint_final.pkl found for prefix '${1}' under ${OUTPUT_ROOT}"
    echo "$ckpt"
}

run_phase() {
    local phase="$1"
    shift

    banner "Phase ${phase^^}"
    python -u run/train_sep_vae.py "$@"
    ok "Phase ${phase^^} complete"
}

append_wandb_args() {
    if [[ "$WANDB" == "1" ]]; then
        ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
        [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
    fi
    return 0
}

for phase in "${ALL_PHASES[@]}"; do
    phase_ge "$phase" "$START_PHASE" || continue
    phase_le "$phase" "$END_PHASE" || break

    case "$phase" in
        d0)
            ARGS=(
                --dicom_dir "$DATA_DIR"
                --csv_path "$CSV_PATH"
                --use_cache
                --model_version v2
                --img_size "$IMG_SIZE"
                --z_channels_common "$Z_COMMON"
                --z_channels_disease "$Z_DISEASE"
                --attn_query_dim "$ATTN_QUERY_DIM"
                --attn_heads "$ATTN_HEADS"
                --batch_size 8
                --epochs 3
                --num_workers 4
                --seed "$SEED"
                --lr_vae 1e-4
                --lr_disc 1e-4
                --weight_decay 1e-4
                --grad_clip 1.0
                --weight_rec 1.0
                --weight_kl_common 1e-4
                --weight_kl_disease 5e-5
                --weight_mi_factor 0.0
                --weight_bbox_attn 0.0
                --weight_perceptual 0.0
                --sigma_inactive 0.1
                --output_root "$OUTPUT_ROOT"
                --exp_name d0_smoke_v2
                --sample_every 1
                --manifold_every -1
            )
            append_wandb_args
            run_phase d0 "${ARGS[@]}"
            ;;
        d1)
            ARGS=(
                --dicom_dir "$DATA_DIR"
                --csv_path "$CSV_PATH"
                --use_cache
                --model_version v2
                --use_bbox_cross_attn
                --img_size "$IMG_SIZE"
                --z_channels_common "$Z_COMMON"
                --z_channels_disease "$Z_DISEASE"
                --attn_query_dim "$ATTN_QUERY_DIM"
                --attn_heads "$ATTN_HEADS"
                --batch_size 16
                --epochs 20
                --num_workers 8
                --seed "$SEED"
                --kl_warmup_epochs 5
                --lr_vae 2e-4
                --lr_disc 1e-4
                --weight_decay 1e-4
                --grad_clip 1.0
                --weight_rec 1.0
                --weight_kl_common 1e-4
                --weight_kl_disease 5e-5
                --weight_mi_factor 0.0
                --weight_bbox_attn 0.0
                --weight_perceptual 0.0
                --sigma_inactive 0.1
                --output_root "$OUTPUT_ROOT"
                --exp_name d1_recon_bbox_xattn
                --sample_every 5
                --manifold_every 10
            )
            append_wandb_args
            run_phase d1 "${ARGS[@]}"
            ;;
        d2)
            D1_CKPT=$(require_ckpt "d1_recon_bbox_xattn")
            printf "  D1 checkpoint: %s\n" "$D1_CKPT"
            ARGS=(
                --dicom_dir "$DATA_DIR"
                --csv_path "$CSV_PATH"
                --use_cache
                --model_version v2
                --use_bbox_cross_attn
                --img_size "$IMG_SIZE"
                --z_channels_common "$Z_COMMON"
                --z_channels_disease "$Z_DISEASE"
                --attn_query_dim "$ATTN_QUERY_DIM"
                --attn_heads "$ATTN_HEADS"
                --batch_size 16
                --epochs 120
                --num_workers 8
                --seed "$SEED"
                --kl_warmup_epochs 0
                --lr_vae 1e-4
                --lr_disc 1e-4
                --weight_decay 1e-4
                --grad_clip 1.0
                --weight_rec 1.0
                --weight_kl_common 1e-4
                --weight_kl_disease 1e-4
                --weight_mi_factor 1.0
                --weight_bbox_attn 0.0
                --weight_perceptual 0.0
                --sigma_inactive 0.1
                --resume "$D1_CKPT"
                --output_root "$OUTPUT_ROOT"
                --exp_name d2_mi_disc
                --save_every 30
                --sample_every 5
                --manifold_every 10
            )
            append_wandb_args
            run_phase d2 "${ARGS[@]}"
            ;;
        d4)
            D2_CKPT=$(require_ckpt "d2_mi_disc")
            printf "  D2 checkpoint: %s\n" "$D2_CKPT"
            ARGS=(
                --dicom_dir "$DATA_DIR"
                --csv_path "$CSV_PATH"
                --use_cache
                --model_version v2
                --use_bbox_cross_attn
                --img_size "$IMG_SIZE"
                --z_channels_common "$Z_COMMON"
                --z_channels_disease "$Z_DISEASE"
                --attn_query_dim "$ATTN_QUERY_DIM"
                --attn_heads "$ATTN_HEADS"
                --batch_size 16
                --epochs 160
                --num_workers 8
                --seed "$SEED"
                --kl_warmup_epochs 0
                --lr_vae 1e-4
                --lr_disc 1e-4
                --weight_decay 1e-4
                --grad_clip 1.0
                --weight_rec 1.0
                --weight_kl_common 1e-4
                --weight_kl_disease 1e-4
                --weight_mi_factor 1.0
                --weight_bbox_attn 0.0
                --weight_perceptual 0.3
                --sigma_inactive 0.1
                --chess_checkpoint "$CHESS_CHECKPOINT"
                --perceptual_only
                --resume "$D2_CKPT"
                --output_root "$OUTPUT_ROOT"
                --exp_name d4_perceptual
                --save_every 5
                --sample_every 5
                --manifold_every 10
            )
            append_wandb_args
            run_phase d4 "${ARGS[@]}"
            ;;
    esac
done

banner "Reproduction curriculum complete (${START_PHASE^^} -> ${END_PHASE^^})"
