#!/usr/bin/env bash
# run_curriculum_hearthout.sh — H0→H1→H2→H3→H7 heart-out z_c curriculum
#
# Heart-out z_c: the cardiac region is zeroed from h_shared before the z_c
# (bg) encoder branch, so z_c cannot encode cardiac features.  z_d still
# sees the full h_shared.  The decoder mask gate (Path A fix) applies only
# to Cardiomegaly images, so Normal reconstruction uses the skip freely.
#
# This eliminates the z_d collapse caused by the old mask curriculum, where
# the gate was applied to Normal images (z_d=0 + blocked skip = z_c forced
# to inpaint cardiac region, causing z_d to collapse).
#
# Phase schedule (epochs are cumulative final-epoch numbers):
#   H0  smoke      ep  0→5    batch=6   pipeline verification, recon + KL only
#   H1  mask_attn  ep  5→35   batch=6   heart_out_zc + CTR(0.5) + FactorVAE MI
#   H2  perceptual ep 35→65   batch=6   CheSS perceptual + TV, CTR→1.0
#   H3  gan        ep 65→130  batch=6   PatchGAN adversarial sharpening
#   H7  skip       ep130→200  batch=4   UNet skip + z_common 16→32
#
# Gate after H1:
#   z_cardio_norm_ratio_bbox_free >= 1.2  (z_cardio encoding something cardiac)
#   silhouette_disease_only_pca_bbox_free >= 0.05  (minimal class separation)
#   If gate fails → CTR regression may need tuning or heart_out_zc has issues
#
# Submit:
#   sbatch run_curriculum_hearthout.sh              # full H0 → H7
#   sbatch run_curriculum_hearthout.sh h1           # start from H1 (H0 ckpt must exist)
#   sbatch run_curriculum_hearthout.sh h2 h3        # H2 → H3 only
#
# Override any default before sbatch:
#   CHEXMASK_CSV=/path/to/csv sbatch run_curriculum_hearthout.sh
#   CUDA_VISIBLE_DEVICES=0 sbatch run_curriculum_hearthout.sh h1 h3

#SBATCH --job-name=sepvae-hearthout-curriculum
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/curriculum-hearthout-%j.out
#SBATCH --error=logs/curriculum-hearthout-%j.err

set -euo pipefail

# ── Working directory ──────────────────────────────────────────────────────────
resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then printf '%s\n' "$WORKDIR"; return; fi
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local job_cmd
        job_cmd=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
                  | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
        [[ -n "$job_cmd" && -f "$job_cmd" ]] && { cd "$(dirname "$job_cmd")" && pwd; return; }
    fi
    [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]] \
        && { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; return; }
    pwd
}

WORKDIR="$(resolve_workdir)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/runs_sepvae}"
ENV_NAME="${ENV_NAME:-jaxstack}"

CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m')
GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); RST=$(printf '\033[0m')
banner() { printf "\n${BLU}${BLD}======  %s  ======${RST}\n\n" "$*"; }
kv()     { printf "  ${CYN}%-28s${RST} %s\n" "$1" "$2"; }
ok()     { printf "${GRN}** %s${RST}\n" "$*"; }
die()    { printf "${RED}!! %s${RST}\n" "$*" >&2; exit 1; }

# ── Environment ────────────────────────────────────────────────────────────────
mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

# ── Shared paths ───────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHEXMASK_CSV="${CHEXMASK_CSV:-/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"

[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"
[[ -f "$CHEXMASK_CSV"     ]] || die "CheXmask CSV not found: $CHEXMASK_CSV (run scripts/download_chexmask.sh)"

# ── Shared model settings ──────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"     # H7 overrides to 32
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"
DECODER_RES_BLOCKS="${DECODER_RES_BLOCKS:-3}"
BBOX_QUERY_MIX="${BBOX_QUERY_MIX:-1.0}"
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.0}"

# ── Shared training settings ───────────────────────────────────────────────────
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"
LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-1024}"
DETERMINISTIC_DATA="${DETERMINISTIC_DATA:-1}"
KL_FREE_BITS="${KL_FREE_BITS:-0.5}"

# ── Logging ────────────────────────────────────────────────────────────────────
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-5}"
MANIFOLD_BBOX_MODE="${MANIFOLD_BBOX_MODE:-both}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Phase selection ────────────────────────────────────────────────────────────
START_PHASE="${1:-h0}"
END_PHASE="${2:-h7}"
ALL_PHASES=(h0 h1 h2 h3 h7)

declare -A PHASE_RANK=([h0]=0 [h1]=1 [h2]=2 [h3]=3 [h7]=4)
phase_ge() { [[ "${PHASE_RANK[$1]}" -ge "${PHASE_RANK[$2]}" ]]; }
phase_le() { [[ "${PHASE_RANK[$1]}" -le "${PHASE_RANK[$2]}" ]]; }

[[ -v "PHASE_RANK[$START_PHASE]" ]] || die "Unknown START_PHASE='${START_PHASE}'. Valid: ${ALL_PHASES[*]}"
[[ -v "PHASE_RANK[$END_PHASE]"   ]] || die "Unknown END_PHASE='${END_PHASE}'. Valid: ${ALL_PHASES[*]}"

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
find_latest_ckpt() {
    find "$OUTPUT_ROOT" -maxdepth 3 -name "checkpoint_final.pkl" \
        | grep "/${1}-" | sort | tail -1 || true
}

require_ckpt() {
    local ckpt
    ckpt=$(find_latest_ckpt "$1")
    [[ -n "$ckpt" ]] || die "No checkpoint_final.pkl for prefix '${1}' under ${OUTPUT_ROOT} — did phase ${2} complete?"
    echo "$ckpt"
}

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
from flax.serialization import msgpack_restore
with open(sys.argv[1], "rb") as f:
    data = msgpack_restore(f.read())
print(int(data["epoch"]))
PY
}

# ── Gate: checks H1 latent geometry before allowing H2 to start ───────────────
gate_h1() {
    local run_dir="$1"
    local selection_path="$run_dir/h1_gate.json"
    python - "$run_dir" "$selection_path" <<'PY'
import json, math, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
history = run_dir / "metrics_history.jsonl"
if not history.exists():
    raise SystemExit(f"metrics_history.jsonl not found: {history}")

records = []
with open(history) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

candidates = []
for rec in records:
    ckpt = rec.get("checkpoint_path")
    if not ckpt or not Path(ckpt).exists():
        continue
    ratio = rec.get("z_cardio_norm_ratio_bbox_free", float("-inf"))
    sil   = rec.get("silhouette_disease_only_pca_bbox_free", float("-inf"))
    if not math.isfinite(ratio):
        continue
    candidates.append((ratio, sil, rec))

if not candidates:
    raise SystemExit("No H1 checkpoints with manifold metrics found — ensure manifold_every > 0")

_, _, best = max(candidates, key=lambda x: (x[0], x[1]))
ratio = float(best.get("z_cardio_norm_ratio_bbox_free",            float("nan")))
sil   = float(best.get("silhouette_disease_only_pca_bbox_free",   float("nan")))

passed = (
    math.isfinite(ratio) and ratio >= 1.2 and
    math.isfinite(sil)   and sil   >= 0.05
)

payload = {
    "selected_checkpoint":          best["checkpoint_path"],
    "epoch":                        int(best.get("epoch", -1)),
    "z_cardio_norm_ratio_bbox_free": ratio,
    "silhouette_bbox_free":          sil,
    "passed_gate":                   passed,
    "thresholds": {"z_cardio_norm_ratio": 1.2, "silhouette_bbox_free": 0.05},
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(best["checkpoint_path"])
sys.exit(0 if passed else 2)
PY
}

# ── Phase runner ───────────────────────────────────────────────────────────────
# Args (positional): same as run_curriculum_mask.sh run_phase
run_phase() {
    local phase="$1"   exp_name="$2"   final_epoch="$3"  batch="$4"
    local kl_warmup="$5"
    local w_rec="$6"   w_kl_c="$7"    w_kl_d="$8"       w_mi="$9"
    local w_bbox="${10}"  w_supcon="${11}"  w_perc="${12}"
    local w_gan="${13}"   w_tv="${14}"
    local bbox_xattn="${15}"  resume="${16}"  save_every="${17}"
    local gan_start="${18:-99999}" lr_pd="${19:-1e-4}"
    local phase_lr_vae="${20:-$LR_VAE}"  phase_lr_disc="${21:-$LR_DISC}"
    local w_masked_rec="${22:-0.0}"  disc_r1="${23:-0.0}"
    local w_ctr_reg="${24:-0.0}"

    if [[ -n "$resume" ]]; then
        local resume_epoch
        resume_epoch=$(checkpoint_epoch "$resume")
        [[ "$final_epoch" -gt "$resume_epoch" ]] || \
            die "Phase ${phase}: final_epoch=${final_epoch} must be > resume epoch ${resume_epoch}"
    fi

    banner "Phase ${phase^^} — ${exp_name}  (→ epoch ${final_epoch}  batch=${batch})"
    kv "kl_free_bits"     "$KL_FREE_BITS"
    kv "decoder_res_blks" "$DECODER_RES_BLOCKS"
    kv "z_channels_common" "$Z_COMMON"
    kv "ctr_reg"          "$w_ctr_reg  masked_rec=$w_masked_rec"
    kv "perceptual"       "$w_perc  tv=$w_tv  gan=$w_gan"
    kv "mi_factor"        "$w_mi  bbox=$w_bbox  supcon=$w_supcon"
    kv "chexmask_csv"     "$CHEXMASK_CSV"
    kv "heart_out_zc"     "enabled"
    [[ -n "$resume" ]] && kv "resume" "$resume"

    local ARGS=(
        run/train_sep_vae.py
        --dicom_dir "$DATA_DIR" --csv_path "$CSV_PATH" --use_cache
        --deterministic_data
        --chexmask_csv "$CHEXMASK_CSV"
        --heart_out_zc

        --model_version        v2
        --img_size             "$IMG_SIZE"
        --z_channels_common    "$Z_COMMON"
        --z_channels_disease   "$Z_DISEASE"
        --attn_query_dim       "$ATTN_QUERY_DIM"
        --attn_heads           "$ATTN_HEADS"
        --decoder_res_blocks   "$DECODER_RES_BLOCKS"

        --batch_size           "$batch"
        --epochs               "$final_epoch"
        --num_workers          "$NUM_WORKERS"
        --eval_num_workers     "$EVAL_NUM_WORKERS"
        --seed                 "$SEED"
        --kl_warmup_epochs     "$kl_warmup"

        --lr_vae               "$phase_lr_vae"
        --lr_disc              "$phase_lr_disc"
        --weight_decay         "$WEIGHT_DECAY"
        --grad_clip            "$GRAD_CLIP"

        --weight_rec           "$w_rec"
        --weight_kl_common     "$w_kl_c"
        --weight_kl_disease    "$w_kl_d"
        --kl_free_bits         "$KL_FREE_BITS"
        --weight_mi_factor     "$w_mi"
        --weight_bbox_attn     "$w_bbox"
        --weight_ctr_reg       "$w_ctr_reg"
        --weight_cardio_supcon "$w_supcon"
        --weight_perceptual    "$w_perc"
        --weight_gan           "$w_gan"
        --weight_tv            "$w_tv"
        --weight_masked_rec    "$w_masked_rec"
        --gan_start_step       "$gan_start"
        --lr_patch_disc        "$lr_pd"
        --disc_r1_penalty      "$disc_r1"

        --bbox_query_mix       "$BBOX_QUERY_MIX"
        --bbox_dropout_prob    "$BBOX_DROPOUT_PROB"
        --sigma_inactive       "$SIGMA_INACTIVE"
        --supcon_temperature   "$SUPCON_TEMPERATURE"

        --output_root          "$OUTPUT_ROOT"
        --exp_name             "$exp_name"
        --sample_every         "$SAMPLE_EVERY"
        --save_every           "$save_every"
        --manifold_every       "$MANIFOLD_EVERY"
        --manifold_bbox_mode   "$MANIFOLD_BBOX_MODE"
        --eval_subset_size     "$EVAL_SUBSET_SIZE"
        --manifold_max_samples "$EVAL_SUBSET_SIZE"
    )

    [[ "$bbox_xattn"         == "1" ]] && ARGS+=( --use_bbox_cross_attn )
    [[ -n "$resume"                 ]] && ARGS+=( --resume "$resume" )
    [[ "${DETERMINISTIC_DATA}" != "1" ]] && ARGS+=( --no-deterministic_data )

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

# ── Print curriculum overview ──────────────────────────────────────────────────
banner "SepVAE Heart-out z_c Curriculum — ${START_PHASE^^} → ${END_PHASE^^}"
kv "Node / partition"   "$(hostname) / biggpu"
kv "Output"             "$OUTPUT_ROOT"
kv "Seed"               "$SEED"
kv "KL free-bits"       "$KL_FREE_BITS  (all phases)"
kv "decoder_res_blks"   "$DECODER_RES_BLOCKS"
kv "CheXmask CSV"       "$CHEXMASK_CSV"
kv "W&B project"        "$WANDB_PROJECT  (enabled=${WANDB})"
kv "heart_out_zc"       "ENABLED — cardiac region zeroed from z_c encoder branch"
kv "weight_bbox_attn"   "0.0  (never used — spatial grounding via input masking)"
printf "\n"
kv "H0  smoke"      "ep  0→5    batch=6   pipeline verification, recon+KL"
kv "H1  hout_attn"  "ep  5→35   batch=6   heart_out_zc + CTR(0.5) + FactorVAE MI"
kv "  [gate]"       "z_cardio_norm_ratio >= 1.2  |  silhouette >= 0.05"
kv "H2  perceptual" "ep 35→65   batch=6   CheSS perceptual + TV, CTR→1.0"
kv "H3  gan"        "ep 65→130  batch=6   PatchGAN, masked_rec→1.5"
kv "H7  skip"       "ep130→200  batch=4   UNet skip + z_common→32, perceptual off"

# ── Curriculum ────────────────────────────────────────────────────────────────
for phase in "${ALL_PHASES[@]}"; do
    phase_ge "$phase" "$START_PHASE" || continue
    phase_le "$phase" "$END_PHASE"   || break

    case "$phase" in

      h0)
        # ── H0: smoke test — verify heart_out_zc pipeline end-to-end ──────────
        run_phase h0  h0_smoke              5  6  0  1.0  1e-4  5e-5  0.0  0.0  0.0  0.0  0.0  0.0  1  ""  5  99999  1e-4  1e-4  1e-4  0.0  0.0  0.0
        ;;

      h1)
        # ── H1: heart-out z_c + CTR regression ───────────────────────────────
        # CTR weight=0.5 (conservative — encoder still learning basic anatomy).
        # FactorVAE MI and supcon introduced.  No perceptual yet.
        H0_CKPT=$(require_ckpt "h0_smoke" "H0")
        printf "  H0 → %s\n" "$H0_CKPT"
        run_phase h1  h1_hout_attn         35  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.0  0.0  0.0  1  "$H0_CKPT"  5  99999  1e-4  1e-4  1e-4  0.3  0.0  0.5
        ;;

      h2)
        # ── H2: perceptual sharpening ─────────────────────────────────────────
        H1_RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "h1_hout_attn-*" \
                     | sort | tail -1)
        [[ -n "$H1_RUN_DIR" ]] || die "No H1 run directory found under $OUTPUT_ROOT"
        printf "  H1 run dir: %s\n" "$H1_RUN_DIR"

        if H2_RESUME=$(gate_h1 "$H1_RUN_DIR"); then
            ok "H1 gate PASSED — proceeding to H2"
            printf "  Best H1 checkpoint: %s\n" "$H2_RESUME"
        else
            gate_exit=$?
            if [[ "$gate_exit" -eq 2 ]]; then
                die "H1 failed the latent-geometry gate — inspect ${H1_RUN_DIR}/h1_gate.json before running H2"
            fi
            die "gate_h1 failed unexpectedly (exit ${gate_exit})"
        fi

        run_phase h2  h2_perceptual        65  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.05  0.0  0.005  1  "$H2_RESUME"  5  99999  1e-4  1e-4  1e-4  1.0  0.0  1.0
        ;;

      h3)
        # ── H3: PatchGAN adversarial sharpening ───────────────────────────────
        H2_CKPT=$(require_ckpt "h2_perceptual" "H2")
        printf "  H2 → %s\n" "$H2_CKPT"
        run_phase h3  h3_gan              130  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.05  0.1  0.005  1  "$H2_CKPT"  5  2000  1e-4  1e-4  1e-4  1.5  0.0  1.0
        ;;

      h7)
        # ── H7: UNet skip connections + z_common=32 ───────────────────────────
        H3_CKPT=$(require_ckpt "h3_gan" "H3")
        printf "  H3 → %s\n" "$H3_CKPT"
        _saved_z_common="$Z_COMMON"
        Z_COMMON=32
        run_phase h7  h7_skip             200  4  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.0  0.1  0.005  1  "$H3_CKPT"  5  500  1e-4  5e-5  1e-4  1.5  0.0  1.0
        Z_COMMON="$_saved_z_common"
        ;;

    esac
done

banner "Heart-out z_c curriculum complete  (${START_PHASE^^} → ${END_PHASE^^})"
ok "All phases done — final checkpoint in ${OUTPUT_ROOT}"
ok "Verification: interpolate z_d magnitude with z_c fixed → varying heart size"
ok "Expected: panel 06 diff shows cardiac silhouette only (non-black)"
ok "Expected: panel 04 anatomy-only shows z_c reconstruction with inpainted heart"
