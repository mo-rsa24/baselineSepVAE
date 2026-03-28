#!/usr/bin/env bash
# run_curriculum_mask.sh — M0→M1→M2→M3→M7 mask supervision curriculum
#
# Pure mask supervision from scratch — bbox attention loss never used.
# Parallel to run_curriculum_v2.sh (D0→D5), same chaining pattern.
#
# Phase schedule (epochs are cumulative final-epoch numbers):
#   M0  smoke      ep  0→5    batch=6   pipeline verification, recon + KL only
#   M1  mask_attn  ep  5→35   batch=6   mask prior + CTR regression + FactorVAE MI
#   M2  perceptual ep 35→65   batch=6   CheSS perceptual + TV, CTR raised to 1.0
#   M3  gan        ep 65→130  batch=6   PatchGAN adversarial sharpening
#   M7  skip       ep130→200  batch=4   UNet skip connections + z_common 16→32
#
# Gate after M1:
#   z_cardio_norm_ratio_bbox_free >= 1.2  (z_cardio encoding something cardiac)
#   silhouette_disease_only_pca_bbox_free >= 0.05  (minimal class separation)
#   If gate fails → inspect M1 before running M2; CTR regression may need tuning
#
# Submit:
#   sbatch run_curriculum_mask.sh              # full M0 → M7
#   sbatch run_curriculum_mask.sh m1           # start from M1 (M0 checkpoint must exist)
#   sbatch run_curriculum_mask.sh m2 m3        # M2 → M3 only
#
# Override any default before sbatch:
#   CHEXMASK_CSV=/path/to/csv sbatch run_curriculum_mask.sh
#   CUDA_VISIBLE_DEVICES=0 sbatch run_curriculum_mask.sh m1 m3

#SBATCH --job-name=sepvae-mask-curriculum
#SBATCH --nodelist=mscluster107
#SBATCH --partition=biggpu
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/curriculum-mask-%j.out
#SBATCH --error=logs/curriculum-mask-%j.err

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
Z_COMMON="${Z_COMMON:-16}"     # M7 overrides to 32
Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"
ATTN_HEADS="${ATTN_HEADS:-4}"
DECODER_RES_BLOCKS="${DECODER_RES_BLOCKS:-3}"
BBOX_QUERY_MIX="${BBOX_QUERY_MIX:-1.0}"
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.0}"   # No bbox dropout — mask covers all images

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
START_PHASE="${1:-m0}"
END_PHASE="${2:-m7}"
ALL_PHASES=(m0 m1 m2 m3 m7)

declare -A PHASE_RANK=([m0]=0 [m1]=1 [m2]=2 [m3]=3 [m7]=4)
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

# ── Gate: checks M1 latent geometry before allowing M2 to start ───────────────
gate_m1() {
    local run_dir="$1"
    local selection_path="$run_dir/m1_gate.json"
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
    raise SystemExit("No M1 checkpoints with manifold metrics found — ensure manifold_every > 0")

_, _, best = max(candidates, key=lambda x: (x[0], x[1]))
ratio = float(best.get("z_cardio_norm_ratio_bbox_free",            float("nan")))
sil   = float(best.get("silhouette_disease_only_pca_bbox_free",   float("nan")))

# Thresholds: lenient — M1 only has 30 epochs and no GAN yet.
# The key signal is that z_cardio is MORE active for Cardiomegaly (ratio > 1.2).
# If ratio < 1.2 after 30 epochs of CTR regression, z_cardio is not learning.
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
# Args (positional):
#  1  phase        2  exp_name      3  final_epoch   4  batch
#  5  kl_warmup    6  w_rec         7  w_kl_c        8  w_kl_d
#  9  w_mi        10  w_bbox       11  w_supcon      12  w_perc
# 13  w_gan       14  w_tv         15  bbox_xattn    16  resume
# 17  save_every  18  gan_start    19  lr_pd         20  lr_vae
# 21  lr_disc     22  w_masked_rec 23  disc_r1       24  w_ctr_reg
#
# Globals read: KL_FREE_BITS, DECODER_RES_BLOCKS, SIGMA_INACTIVE, SEED,
#               NUM_WORKERS, EVAL_NUM_WORKERS, WEIGHT_DECAY, GRAD_CLIP,
#               SUPCON_TEMPERATURE, BBOX_QUERY_MIX, BBOX_DROPOUT_PROB,
#               EVAL_SUBSET_SIZE, MANIFOLD_BBOX_MODE, SAMPLE_EVERY,
#               MANIFOLD_EVERY, DATA_DIR, CSV_PATH, CHEXMASK_CSV,
#               OUTPUT_ROOT, CHESS_CHECKPOINT, Z_COMMON,
#               WANDB, WANDB_PROJECT, WANDB_ENTITY

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
    [[ -n "$resume" ]] && kv "resume" "$resume"

    local ARGS=(
        run/train_sep_vae.py
        --dicom_dir "$DATA_DIR" --csv_path "$CSV_PATH" --use_cache
        --deterministic_data
        --chexmask_csv "$CHEXMASK_CSV"

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

    # CheSS perceptual backbone needed whenever perceptual or GAN loss is active
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
banner "SepVAE Mask Curriculum — ${START_PHASE^^} → ${END_PHASE^^}"
kv "Node / partition"   "$(hostname) / biggpu"
kv "Output"             "$OUTPUT_ROOT"
kv "Seed"               "$SEED"
kv "KL free-bits"       "$KL_FREE_BITS  (all phases)"
kv "decoder_res_blks"   "$DECODER_RES_BLOCKS"
kv "CheXmask CSV"       "$CHEXMASK_CSV"
kv "W&B project"        "$WANDB_PROJECT  (enabled=${WANDB})"
kv "weight_bbox_attn"   "0.0  (never used — pure mask supervision)"
printf "\n"
kv "M0  smoke"      "ep  0→5    batch=6   pipeline verification, recon+KL"
kv "M1  mask_attn"  "ep  5→35   batch=6   mask prior + CTR(0.5) + FactorVAE MI"
kv "  [gate]"       "z_cardio_norm_ratio >= 1.2  |  silhouette >= 0.05"
kv "M2  perceptual" "ep 35→65   batch=6   CheSS perceptual + TV, CTR→1.0"
kv "M3  gan"        "ep 65→130  batch=6   PatchGAN, masked_rec→1.5"
kv "M7  skip"       "ep130→200  batch=4   UNet skip + z_common→32, perceptual off"

# ── Curriculum ────────────────────────────────────────────────────────────────
# run_phase positional map:
#   phase  exp_name           ep   bs  klw  rec   kc     kd     mi    bb     sc     pe     gn    tv     xa  resume        save  gan_start  lr_pd  lr_vae  lr_disc  w_mrec  disc_r1  w_ctr
for phase in "${ALL_PHASES[@]}"; do
    phase_ge "$phase" "$START_PHASE" || continue
    phase_le "$phase" "$END_PHASE"   || break

    case "$phase" in

      m0)
        # ── M0: smoke test — verify CheXmask pipeline end-to-end ─────────────
        run_phase m0  m0_smoke              5  6  0  1.0  1e-4  5e-5  0.0  0.0  0.0  0.0  0.0  0.0  1  ""  5  99999  1e-4  1e-4  1e-4  0.0  0.0  0.0
        ;;

      m1)
        # ── M1: mask attention + CTR regression ───────────────────────────────
        # CTR weight=0.5 (conservative — encoder still learning basic anatomy).
        # FactorVAE MI and supcon introduced. No perceptual yet.
        M0_CKPT=$(require_ckpt "m0_smoke" "M0")
        printf "  M0 → %s\n" "$M0_CKPT"
        run_phase m1  m1_mask_attn         35  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.0  0.0  0.0  1  "$M0_CKPT"  5  99999  1e-4  1e-4  1e-4  0.3  0.0  0.5
        ;;

      m2)
        # ── M2: perceptual sharpening ─────────────────────────────────────────
        # Gate: verify M1 has started encoding cardiac signal before perceptual
        # gradients start competing.
        M1_RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "m1_mask_attn-*" \
                     | sort | tail -1)
        [[ -n "$M1_RUN_DIR" ]] || die "No M1 run directory found under $OUTPUT_ROOT"
        printf "  M1 run dir: %s\n" "$M1_RUN_DIR"

        if M2_RESUME=$(gate_m1 "$M1_RUN_DIR"); then
            ok "M1 gate PASSED — proceeding to M2"
            printf "  Best M1 checkpoint: %s\n" "$M2_RESUME"
        else
            gate_exit=$?
            if [[ "$gate_exit" -eq 2 ]]; then
                die "M1 failed the latent-geometry gate — inspect ${M1_RUN_DIR}/m1_gate.json before running M2"
            fi
            die "gate_m1 failed unexpectedly (exit ${gate_exit})"
        fi

        # CTR raised to 1.0 — encoder stable enough after M1.
        # masked_rec raised to 1.0.
        run_phase m2  m2_perceptual        65  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.05  0.0  0.005  1  "$M2_RESUME"  5  99999  1e-4  1e-4  1e-4  1.0  0.0  1.0
        ;;

      m3)
        # ── M3: PatchGAN adversarial sharpening ───────────────────────────────
        # GAN lessons from D3 applied: weight_gan=0.1, phase-local start,
        # no R1 penalty. masked_rec raised to final level (1.5).
        M2_CKPT=$(require_ckpt "m2_perceptual" "M2")
        printf "  M2 → %s\n" "$M2_CKPT"
        run_phase m3  m3_gan              130  6  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.05  0.1  0.005  1  "$M2_CKPT"  5  2000  1e-4  1e-4  1e-4  1.5  0.0  1.0
        ;;

      m7)
        # ── M7: UNet skip connections + z_common=32 ───────────────────────────
        # Architectural changes: skip3_fuse + skip2_fuse (freshly init),
        # z_common head re-init (16→32). All other weights restored from M3.
        #
        # Mask-gate on enc_layer3 skip is active (heart_mask threaded to decoder)
        # — zeros cardiac region in skip features to prevent z_cardio bypass.
        #
        # weight_perceptual=0.0: skip+GAN replace CheSS perceptual.
        # batch=4: skip connections raise peak memory.
        # lr_vae=5e-5: conservative — only skip projections + z_common head new.
        # gan_start=500: skip connections accelerate decoder convergence.
        M3_CKPT=$(require_ckpt "m3_gan" "M3")
        printf "  M3 → %s\n" "$M3_CKPT"
        _saved_z_common="$Z_COMMON"
        Z_COMMON=32          # Override for M7 only — z_common head re-initialised
        run_phase m7  m7_skip             200  4  0  1.0  1e-4  5e-5  1.0  0.0  0.05  0.0  0.1  0.005  1  "$M3_CKPT"  5  500  1e-4  5e-5  1e-4  1.5  0.0  1.0
        Z_COMMON="$_saved_z_common"
        ;;

    esac
done

banner "Mask curriculum complete  (${START_PHASE^^} → ${END_PHASE^^})"
ok "All phases done — final checkpoint in ${OUTPUT_ROOT}"
ok "Traversal diagnostic: conda run -n jaxstack python scripts/latent_traversal_medsam.py \\"
ok "  --checkpoint \$(ls -t ${OUTPUT_ROOT}/m7_skip-*/checkpoints/checkpoint_final.pkl | head -1) \\"
ok "  --output results/traversal_m7_curriculum/"
ok "Success criterion: Δarea > ±500px across α=0→2"
