#!/usr/bin/env bash
# Reproducibility-focused D0 → D1 → D2 → D4 curriculum.
#
# Phases run sequentially inside one job; checkpoints are chained automatically.
# D4 resumes from the best saved D2 checkpoint, not checkpoint_final.pkl, and the
# curriculum aborts before D4 if D2 fails the latent-geometry gate.
#
# Submit (cluster):
#   sbatch run_curriculum.sh                      # D1 → D2 → D4
#   sbatch run_curriculum.sh d2                   # start from D2 (D1 done)
#   sbatch run_curriculum.sh d4                   # start from D4 (D2 done)
#   sbatch run_curriculum.sh d0 d4                # smoke-test first, then full run
#
# Run locally (single GPU node):
#   bash run_curriculum.sh
#
# Override hyperparams as usual:
#   OUTPUT_ROOT=/scratch/my_sepvae_runs sbatch run_curriculum.sh d2

#SBATCH --job-name=sepvae-curriculum
#SBATCH --exclude=mscluster44,mscluster65,mscluster76,mscluster59,mscluster48,mscluster82,mscluster75,mscluster72
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/curriculum-%j.out
#SBATCH --error=logs/curriculum-%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"
ENV_NAME="${ENV_NAME:-jaxstack}"

CYN=$(printf '\033[36m'); GRN=$(printf '\033[32m')
RED=$(printf '\033[31m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
banner() { printf "\n${CYN}${BLD}======  %s  ======${RST}\n\n" "$*"; }
ok()     { printf "${GRN}** %s${RST}\n" "$*"; }
die()    { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Environment (once, for the whole job) ─────────────────────────────────────
mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

# ── Shared config (override via environment before sbatch) ────────────────────
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"; Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"; ATTN_HEADS="${ATTN_HEADS:-4}"
LR_VAE="${LR_VAE:-1e-4}"; LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"; GRAD_CLIP="${GRAD_CLIP:-1.0}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"; SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"; MANIFOLD_EVERY="${MANIFOLD_EVERY:-10}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-1024}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
MANIFOLD_BBOX_MODE="${MANIFOLD_BBOX_MODE:-both}"
BBOX_QUERY_MIX="${BBOX_QUERY_MIX:-0.7}"
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.3}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.1}"
DETERMINISTIC_DATA="${DETERMINISTIC_DATA:-1}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

[[ -d "$DATA_DIR"        ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"        ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Argument parsing ──────────────────────────────────────────────────────────
START_PHASE="${1:-d1}"
END_PHASE="${2:-d4}"

ALL_PHASES=(d0 d1 d2 d4)

phase_ge() {
    local -A rank=([d0]=0 [d1]=1 [d2]=2 [d4]=3 [d5]=4)
    [[ "${rank[$1]}" -ge "${rank[$2]}" ]]
}
phase_le() {
    local -A rank=([d0]=0 [d1]=1 [d2]=2 [d4]=3 [d5]=4)
    [[ "${rank[$1]}" -le "${rank[$2]}" ]]
}

[[ "$START_PHASE" != "d5" && "$END_PHASE" != "d5" ]] || \
    die "D5 is out of scope for the reproducibility curriculum."

# ── Checkpoint discovery ──────────────────────────────────────────────────────
find_latest_run_dir() {
    local hit
    hit=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${1}-*" | sort | tail -1 || true)
    echo "$hit"
}

find_latest_ckpt() {
    local hit
    hit=$(find "$OUTPUT_ROOT" -maxdepth 3 -name "checkpoint_final.pkl" \
          | grep "/${1}-" | sort | tail -1 || true)
    echo "$hit"
}

require_ckpt() {
    local ckpt
    ckpt=$(find_latest_ckpt "$1")
    [[ -n "$ckpt" ]] || die "No checkpoint_final.pkl found for prefix '${1}' under ${OUTPUT_ROOT} — did phase ${2} complete?"
    echo "$ckpt"
}

select_best_d2_checkpoint() {
    local run_dir="$1"
    local selection_path="$run_dir/d2_selection.json"
    python - "$run_dir" "$selection_path" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
history_path = run_dir / "metrics_history.jsonl"
if not history_path.exists():
    raise SystemExit(f"metrics history not found: {history_path}")

records = []
with open(history_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

candidates = []
for rec in records:
    ckpt = rec.get("checkpoint_path")
    if not ckpt:
        continue
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        continue
    free = rec.get("silhouette_disease_only_pca_bbox_free", float("-inf"))
    guided = rec.get("silhouette_disease_only_pca_bbox_guided", float("-inf"))
    rec_loss = rec.get("loss/reconstruction", float("inf"))
    if not math.isfinite(free):
        continue
    candidates.append((free, guided, -rec_loss, rec))

if not candidates:
    raise SystemExit("no saved D2 checkpoints with manifold metrics were found")

_, _, _, best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
free = float(best.get("silhouette_disease_only_pca_bbox_free", float("nan")))
guided = float(best.get("silhouette_disease_only_pca_bbox_guided", float("nan")))
ratio = float(best.get("z_cardio_norm_ratio_bbox_free", float("nan")))
passed = (
    math.isfinite(free) and free >= 0.20 and
    math.isfinite(guided) and guided >= 0.35 and
    math.isfinite(ratio) and ratio >= 2.0
)

payload = {
    "run_dir": str(run_dir),
    "selected_checkpoint": best["checkpoint_path"],
    "epoch": int(best["epoch"]),
    "silhouette_disease_only_pca_bbox_free": free,
    "silhouette_disease_only_pca_bbox_guided": guided,
    "z_cardio_norm_ratio_bbox_free": ratio,
    "passed_gate": passed,
}
with open(selection_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(best["checkpoint_path"])
sys.exit(0 if passed else 2)
PY
}

checkpoint_epoch() {
    local ckpt="$1"
    python - "$ckpt" <<'PY'
import sys
from flax.serialization import msgpack_restore

with open(sys.argv[1], "rb") as f:
    data = msgpack_restore(f.read())

print(int(data["epoch"]))
PY
}

# ── Phase runner ──────────────────────────────────────────────────────────────
# Builds the python argv for one phase and launches it directly (no sub-shell).
run_phase() {
    local phase="$1" exp_name="$2" final_epoch="$3" batch="$4" kl_warmup="$5"
    local w_rec="$6" w_kl_c="$7" w_kl_d="$8" w_mi="$9"
    local w_bbox="${10}" w_supcon="${11}" w_perc="${12}" w_gan="${13}" w_tv="${14}"
    local bbox_xattn="${15}" resume="${16}" save_every="${17}"
    local gan_start="${18:-5000}" lr_pd="${19:-1e-4}"
    local phase_lr_vae="${20:-$LR_VAE}" phase_lr_disc="${21:-$LR_DISC}"
    local w_masked_rec="${22:-0.0}"
    local resume_epoch=""

    if [[ -n "$resume" ]]; then
        resume_epoch=$(checkpoint_epoch "$resume")
        [[ "$final_epoch" -gt "$resume_epoch" ]] || \
            die "Phase ${phase}: final epoch ${final_epoch} must be > resume epoch ${resume_epoch}"
    fi

    banner "Phase ${phase^^} — ${exp_name}  (final_epoch=${final_epoch}  batch=${batch})"
    if [[ -n "$resume" ]]; then
        printf "  Warm-start: %s\n" "$resume"
        printf "  Resume epoch: %s  →  target epoch: %s\n" "$resume_epoch" "$final_epoch"
    fi

    local ARGS=(
        run/train_sep_vae.py
        --dicom_dir "$DATA_DIR" --csv_path "$CSV_PATH" --use_cache
        --deterministic_data
        --model_version v2
        --img_size "$IMG_SIZE" --z_channels_common "$Z_COMMON"
        --z_channels_disease "$Z_DISEASE" --attn_query_dim "$ATTN_QUERY_DIM"
        --attn_heads "$ATTN_HEADS"
        --batch_size "$batch" --epochs "$final_epoch"
        --num_workers "$NUM_WORKERS" --eval_num_workers "$EVAL_NUM_WORKERS" --seed "$SEED"
        --kl_warmup_epochs "$kl_warmup"
        --lr_vae "$phase_lr_vae" --lr_disc "$phase_lr_disc"
        --weight_decay "$WEIGHT_DECAY" --grad_clip "$GRAD_CLIP"
        --weight_rec "$w_rec"
        --weight_kl_common "$w_kl_c" --weight_kl_disease "$w_kl_d"
        --weight_mi_factor "$w_mi"
        --weight_masked_rec "$w_masked_rec"
        --weight_bbox_attn "$w_bbox"
        --weight_cardio_supcon "$w_supcon" --supcon_temperature "$SUPCON_TEMPERATURE"
        --weight_perceptual "$w_perc"
        --weight_gan "$w_gan" --weight_tv "$w_tv"
        --gan_start_step "$gan_start" --lr_patch_disc "$lr_pd"
        --bbox_query_mix "$BBOX_QUERY_MIX" --bbox_dropout_prob "$BBOX_DROPOUT_PROB"
        --sigma_inactive "$SIGMA_INACTIVE"
        --output_root "$OUTPUT_ROOT" --exp_name "$exp_name"
        --sample_every "$SAMPLE_EVERY" --save_every "$save_every"
        --manifold_every "$MANIFOLD_EVERY" --eval_subset_size "$EVAL_SUBSET_SIZE"
        --manifold_max_samples "$EVAL_SUBSET_SIZE"
        --manifold_bbox_mode "$MANIFOLD_BBOX_MODE"
    )

    [[ "$bbox_xattn" == "1" ]] && ARGS+=( --use_bbox_cross_attn )
    [[ -n "$resume"          ]] && ARGS+=( --resume "$resume" )
    [[ "$DETERMINISTIC_DATA" != "1" ]] && ARGS+=( --no-deterministic_data )

    # CheSS needed whenever perceptual or GAN loss is active
    if [[ "$w_perc" != "0.0" || "$w_gan" != "0.0" ]]; then
        ARGS+=( --chess_checkpoint "$CHESS_CHECKPOINT" --perceptual_only )
    fi

    if [[ "$WANDB" == "1" ]]; then
        ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
        [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
    fi

    python -u "${ARGS[@]}"
    ok "Phase ${phase^^} complete"
}

# ── Curriculum ────────────────────────────────────────────────────────────────
# run_phase args:
#   phase  exp_name  final_epoch  batch  kl_warmup
#   w_rec  w_kl_c  w_kl_d  w_mi  w_bbox  w_supcon  w_perc  w_gan  w_tv
#   bbox_xattn  resume  save_every  [gan_start]  [lr_pd]  [lr_vae]  [lr_disc]  [w_masked_rec]

for phase in "${ALL_PHASES[@]}"; do
    phase_ge "$phase" "$START_PHASE" || continue
    phase_le "$phase" "$END_PHASE"   || break

    case "$phase" in
      d0)
        #                                        ep  bs  klw  rec   kc     kd     mi   bb    sc    pe   gn   tv   xa  resume  save
        run_phase d0 d0_smoke_v2                3   8   0  1.0  1e-4  5e-5  0.0  0.0   0.0   0.0  0.0  0.0  0  ""       1
        ;;

      d1)
        # weight_kl_common raised 1e-4 → 5e-4 to prevent KL explosion on local-cluster
        # hardware.  The original 1e-4 allowed the optimizer to find a high-KL / low-rec
        # basin when data ordering was non-deterministic; 5x stronger KL regularisation
        # keeps the posterior collapsed toward the prior regardless of batch ordering.
        # D2 restores 1e-4 via its own hyperparameters after the checkpoint is loaded.
        # w_masked_rec=0.0: encoder not yet warm — no benefit penalising outside-bbox
        # until the bbox attention head is tracking the cardiac region.
        run_phase d1 d1_recon_bbox_xattn       20   8   5  1.0  5e-4  5e-5  0.0  0.05  0.0   0.0  0.0  0.0  1  ""       5  5000  1e-4  2e-4  1e-4  0.0
        ;;

      d2)
        D1_CKPT=$(require_ckpt "d1_recon_bbox_xattn" "D1")
        printf "  D1 checkpoint: %s\n" "$D1_CKPT"
        # w_masked_rec=0.3: outside-bbox MSE with z_cardio=0 forces z_common to explain
        # everything except the heart, complementing bbox_attn on the encoder side.
        run_phase d2 d2_mi_disc               120   8   0  1.0  1e-4  1e-4  1.0  0.10  0.05  0.0  0.0  0.0  1  "$D1_CKPT" 10  5000  1e-4  1e-4  1e-4  0.3
        ;;

      d4)
        D2_RUN_DIR=$(find_latest_run_dir "d2_mi_disc")
        [[ -n "$D2_RUN_DIR" ]] || die "No D2 run directory found under ${OUTPUT_ROOT}"
        printf "  D2 run dir: %s\n" "$D2_RUN_DIR"
        if D2_CKPT=$(select_best_d2_checkpoint "$D2_RUN_DIR"); then
            printf "  Selected D2 checkpoint: %s\n" "$D2_CKPT"
        else
            status=$?
            if [[ "$status" -eq 2 ]]; then
                die "D2 failed the reproducibility gate. See ${D2_RUN_DIR}/d2_selection.json"
            fi
            die "Failed to select best D2 checkpoint from ${D2_RUN_DIR}"
        fi
        # w_masked_rec=0.3: same constraint carried into perceptual phase.
        run_phase d4 d4_perceptual            160   8   0  1.0  1e-4  1e-4  1.0  0.10  0.05  0.3  0.0  0.0  1  "$D2_CKPT" 5   5000  1e-4  1e-4  1e-4  0.3
        ;;
    esac
done

banner "Curriculum complete  (${START_PHASE^^} → ${END_PHASE^^})"
