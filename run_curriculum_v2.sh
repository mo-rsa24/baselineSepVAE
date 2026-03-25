#!/usr/bin/env bash
# run_curriculum_v2.sh — Full D0→D5 curriculum with all P1–P4 fixes baked in.
#
# Fixes vs previous curriculum (run_curriculum.sh):
#   P1  KL free-bits = 0.5   per-dim KL floor from D0 — eliminates 33→11,600 KL spikes
#   P2  TV loss = 0.001       active whenever perceptual loss is on (D2+) — kills stride-8 grid
#   P3  Perceptual weight 0.15 (was 0.05) — 3× stronger texture signal
#   P4  decoder_res_blocks = 3 (was 2) — extra ResBlockSE per decoder level from D0
#
# Phase schedule (epochs are cumulative final-epoch numbers):
#   D0  smoke      ep  0→3    batch=4   pipeline sanity, no MI/percep
#   D1  recon      ep  0→30   batch=4   bbox cross-attn + KL warmup
#   D2  percep     ep 30→55   batch=4   CheSS perceptual + TV + bbox supervision
#   D3  mi         ep 55→90   batch=4   FactorVAE MI discriminator added
#   D4  mi+percep  ep 90→120  batch=4   MI + perceptual + masked recon
#   D5  gan        ep120→160  batch=6   PatchGAN sharpening
#
# Gate: silhouette check after D2 — aborts if latent geometry is degenerate
#       (bbox-free sil ≥ 0.20, bbox-guided sil ≥ 0.35, norm ratio ≥ 2.0)
#
# Submit:
#   sbatch run_curriculum_v2.sh              # full D0 → D5
#   sbatch run_curriculum_v2.sh d2           # skip D0/D1, start from D2
#   sbatch run_curriculum_v2.sh d3 d5        # D3 → D5 only
#
# Override any default before sbatch:
#   OUTPUT_ROOT=/scratch/my_run sbatch run_curriculum_v2.sh d1

#SBATCH --job-name=sepvae-v2-curriculum
#SBATCH --nodelist=mscluster108
#SBATCH --partition=biggpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/curriculum-v2-%j.out
#SBATCH --error=logs/curriculum-v2-%j.err

set -euo pipefail

# ── Working directory ─────────────────────────────────────────────────────────
resolve_workdir() {
    if [[ -n "${WORKDIR:-}" ]]; then
        printf '%s\n' "$WORKDIR"; return
    fi
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

# ── Environment ───────────────────────────────────────────────────────────────
mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +u
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
set -u
ok "Activated: ${ENV_NAME}  node=$(hostname)"

# ── Shared paths ──────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/datasets/mmolefe/vinbigdata/cache_npy}"
CSV_PATH="${CSV_PATH:-/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv}"
CHESS_CHECKPOINT="${CHESS_CHECKPOINT:-/datasets/mmolefe/chess/pretrained_weights.pth.tar}"

[[ -d "$DATA_DIR"         ]] || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CSV_PATH"         ]] || die "CSV not found: $CSV_PATH"
[[ -f "$CHESS_CHECKPOINT" ]] || die "CheSS weights not found: $CHESS_CHECKPOINT"

# ── Shared model settings ─────────────────────────────────────────────────────
IMG_SIZE="${IMG_SIZE:-256}"
Z_COMMON="${Z_COMMON:-16}"; Z_DISEASE="${Z_DISEASE:-16}"
ATTN_QUERY_DIM="${ATTN_QUERY_DIM:-256}"; ATTN_HEADS="${ATTN_HEADS:-4}"
DECODER_RES_BLOCKS="${DECODER_RES_BLOCKS:-3}"   # P4: 3 ResBlockSE per level from D0
BBOX_QUERY_MIX="${BBOX_QUERY_MIX:-0.7}"
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.3}"

# ── Shared training settings ──────────────────────────────────────────────────
SEED="${SEED:-0}"
LR_VAE="${LR_VAE:-1e-4}"; LR_DISC="${LR_DISC:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"; GRAD_CLIP="${GRAD_CLIP:-1.0}"
SIGMA_INACTIVE="${SIGMA_INACTIVE:-0.1}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"; EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-1024}"
DETERMINISTIC_DATA="${DETERMINISTIC_DATA:-1}"

# ── P1: KL free-bits — applied globally across all phases ─────────────────────
KL_FREE_BITS="${KL_FREE_BITS:-0.5}"

# ── Logging ───────────────────────────────────────────────────────────────────
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
MANIFOLD_EVERY="${MANIFOLD_EVERY:-5}"
MANIFOLD_BBOX_MODE="${MANIFOLD_BBOX_MODE:-both}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline-sepvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Phase selection ───────────────────────────────────────────────────────────
START_PHASE="${1:-d0}"
END_PHASE="${2:-d5}"
ALL_PHASES=(d0 d1 d2 d3 d4 d5)

declare -A PHASE_RANK=([d0]=0 [d1]=1 [d2]=2 [d3]=3 [d4]=4 [d5]=5)
phase_ge() { [[ "${PHASE_RANK[$1]}" -ge "${PHASE_RANK[$2]}" ]]; }
phase_le() { [[ "${PHASE_RANK[$1]}" -le "${PHASE_RANK[$2]}" ]]; }

[[ -v "PHASE_RANK[$START_PHASE]" ]] || die "Unknown START_PHASE='${START_PHASE}'. Valid: ${ALL_PHASES[*]}"
[[ -v "PHASE_RANK[$END_PHASE]"   ]] || die "Unknown END_PHASE='${END_PHASE}'. Valid: ${ALL_PHASES[*]}"

# ── Checkpoint helpers ────────────────────────────────────────────────────────
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

# Gate: checks D2 latent geometry before allowing D3 to start
gate_d2() {
    local run_dir="$1"
    local selection_path="$run_dir/d2_gate.json"
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
    free   = rec.get("silhouette_disease_only_pca_bbox_free",   float("-inf"))
    guided = rec.get("silhouette_disease_only_pca_bbox_guided", float("-inf"))
    ratio  = rec.get("z_cardio_norm_ratio_bbox_free",            float("-inf"))
    if not math.isfinite(free):
        continue
    candidates.append((free, guided, rec))

if not candidates:
    raise SystemExit("No D2 checkpoints with manifold metrics found")

_, _, best = max(candidates, key=lambda x: (x[0], x[1]))
free   = float(best.get("silhouette_disease_only_pca_bbox_free",   float("nan")))
guided = float(best.get("silhouette_disease_only_pca_bbox_guided", float("nan")))
ratio  = float(best.get("z_cardio_norm_ratio_bbox_free",            float("nan")))

# Thresholds — D1 already achieved 0.834/0.959 so these are generous
passed = (
    math.isfinite(free)   and free   >= 0.20 and
    math.isfinite(guided) and guided >= 0.35 and
    math.isfinite(ratio)  and ratio  >= 2.0
)

payload = {
    "selected_checkpoint": best["checkpoint_path"],
    "epoch":  int(best.get("epoch", -1)),
    "silhouette_bbox_free":   free,
    "silhouette_bbox_guided": guided,
    "z_cardio_norm_ratio":    ratio,
    "passed_gate": passed,
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(best["checkpoint_path"])
sys.exit(0 if passed else 2)
PY
}

# ── Phase runner ──────────────────────────────────────────────────────────────
# Args (positional):
#  1  phase        2  exp_name      3  final_epoch   4  batch
#  5  kl_warmup    6  w_rec         7  w_kl_c        8  w_kl_d
#  9  w_mi        10  w_bbox       11  w_supcon      12  w_perc
# 13  w_gan       14  w_tv         15  bbox_xattn    16  resume
# 17  save_every  18  gan_start    19  lr_pd         20  lr_vae
# 21  lr_disc     22  w_masked_rec
#
# Globals read: KL_FREE_BITS, DECODER_RES_BLOCKS, SIGMA_INACTIVE, SEED,
#               NUM_WORKERS, EVAL_NUM_WORKERS, WEIGHT_DECAY, GRAD_CLIP,
#               SUPCON_TEMPERATURE, BBOX_QUERY_MIX, BBOX_DROPOUT_PROB,
#               EVAL_SUBSET_SIZE, MANIFOLD_BBOX_MODE, SAMPLE_EVERY,
#               MANIFOLD_EVERY, DATA_DIR, CSV_PATH, OUTPUT_ROOT,
#               CHESS_CHECKPOINT, WANDB, WANDB_PROJECT, WANDB_ENTITY

run_phase() {
    local phase="$1"   exp_name="$2"   final_epoch="$3"  batch="$4"
    local kl_warmup="$5"
    local w_rec="$6"   w_kl_c="$7"    w_kl_d="$8"       w_mi="$9"
    local w_bbox="${10}"  w_supcon="${11}"  w_perc="${12}"
    local w_gan="${13}"   w_tv="${14}"
    local bbox_xattn="${15}"  resume="${16}"  save_every="${17}"
    local gan_start="${18:-5000}"   lr_pd="${19:-1e-4}"
    local phase_lr_vae="${20:-$LR_VAE}"  phase_lr_disc="${21:-$LR_DISC}"
    local w_masked_rec="${22:-0.0}"  disc_r1="${23:-0.0}"

    if [[ -n "$resume" ]]; then
        local resume_epoch
        resume_epoch=$(checkpoint_epoch "$resume")
        [[ "$final_epoch" -gt "$resume_epoch" ]] || \
            die "Phase ${phase}: final_epoch=${final_epoch} must be > resume epoch ${resume_epoch}"
    fi

    banner "Phase ${phase^^} — ${exp_name}  (→ epoch ${final_epoch}  batch=${batch})"
    kv "kl_free_bits"     "$KL_FREE_BITS   (per-dim KL floor)"
    kv "decoder_res_blks" "$DECODER_RES_BLOCKS"
    kv "perceptual"       "$w_perc  tv=$w_tv  gan=$w_gan"
    kv "mi_factor"        "$w_mi  bbox=$w_bbox  supcon=$w_supcon"
    [[ -n "$resume" ]] && kv "resume" "$resume"

    local ARGS=(
        run/train_sep_vae.py
        --dicom_dir "$DATA_DIR" --csv_path "$CSV_PATH" --use_cache
        --deterministic_data

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

    [[ "$bbox_xattn"       == "1" ]] && ARGS+=( --use_bbox_cross_attn )
    [[ -n "$resume"               ]] && ARGS+=( --resume "$resume" )
    [[ "$DETERMINISTIC_DATA" != "1" ]] && ARGS+=( --no-deterministic_data )

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

# ── Print curriculum overview ─────────────────────────────────────────────────
banner "SepVAE V2 Curriculum — ${START_PHASE^^} → ${END_PHASE^^}"
kv "Node / partition"  "$(hostname) / biggpu"
kv "Output"            "$OUTPUT_ROOT"
kv "Seed"              "$SEED"
kv "KL free-bits (P1)" "$KL_FREE_BITS  (all phases)"
kv "decoder_res_blks (P4)" "$DECODER_RES_BLOCKS  (was 2)"
kv "W&B project"       "$WANDB_PROJECT  (enabled=${WANDB})"
printf "\n"
kv "D0  smoke"     "ep  0→3    batch=8   pipeline sanity"
kv "D1  recon"     "ep  0→30   batch=8   bbox cross-attn + KL warmup"
kv "D2  percep"    "ep 30→55   batch=8   perceptual=0.15 + TV=0.001 + bbox (P2,P3)"
kv "D3  mi"        "ep 55→90   batch=8   FactorVAE MI disc + masked recon"
kv "D4  mi+percep" "ep 90→120  batch=8   MI + perceptual + masked recon"
kv "D5  gan"       "ep120→160  batch=12  PatchGAN sharpening"

# ── Curriculum ────────────────────────────────────────────────────────────────
# run_phase:
#  phase  exp_name         ep   bs  klw  rec   kc     kd     mi    bb    sc    pe    gn    tv    xa  resume  save
for phase in "${ALL_PHASES[@]}"; do
    phase_ge "$phase" "$START_PHASE" || continue
    phase_le "$phase" "$END_PHASE"   || break

    case "$phase" in

      d0)
        # ── D0: smoke test — verify pipeline end-to-end ──────────────────────
        run_phase d0  d0_smoke_v2              3  4  0  1.0  1e-4  5e-5  0.0  0.0   0.0  0.0  0.0  0.0  0  ""  1
        ;;

      d1)
        # ── D1: reconstruction foundation + bbox cross-attention ─────────────
        # KL warmup 5 epochs: lets the encoder form basic features before KL
        # pressure kicks in. bbox_attn=0.05 + supcon=0.05 from day 1.
        # Raised kl_c: 1e-4→5e-4 (prevents high-KL basin on first epoch) —
        # free-bits already guards lower bound; this guards the upper.
        run_phase d1  d1_recon_bbox_xattn     30  4  5  1.0  5e-4  5e-5  0.0  0.05  0.05  0.0  0.0  0.0  1  ""  5
        ;;

      d2)
        # ── D2: perceptual sharpening + bbox supervision ─────────────────────
        # P2: TV=0.001 suppresses CheSS stride-8 grid artifacts
        # P3: perceptual=0.15 (was 0.05) — 3× stronger texture signal
        # Restore kl_c to 1e-4 now encoder is stable; free-bits holds the floor
        D1_CKPT=$(require_ckpt "d1_recon_bbox_xattn" "D1")
        printf "  D1 → %s\n" "$D1_CKPT"
        run_phase d2  d2_perceptual_bbox      55  4  0  1.0  1e-4  5e-5  0.0  0.05  0.05  0.15  0.0  0.001  1  "$D1_CKPT"  5
        ;;

      d3)
        # ── D3: FactorVAE MI discriminator ───────────────────────────────────
        # Gate: verify D2 latent geometry before committing to 35 more epochs
        D2_RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "d2_perceptual_bbox-*" \
                     | sort | tail -1)
        [[ -n "$D2_RUN_DIR" ]] || die "No D2 run directory found under $OUTPUT_ROOT"
        printf "  D2 run dir: %s\n" "$D2_RUN_DIR"

        if D3_RESUME=$(gate_d2 "$D2_RUN_DIR"); then
            ok "D2 gate PASSED — proceeding to D3"
            printf "  Best D2 checkpoint: %s\n" "$D3_RESUME"
        else
            gate_exit=$?
            if [[ "$gate_exit" -eq 2 ]]; then
                die "D2 failed the latent-geometry gate — inspect ${D2_RUN_DIR}/d2_gate.json before re-running D3"
            fi
            die "gate_d2 failed unexpectedly (exit ${gate_exit})"
        fi

        # w_masked_rec=0.3: outside-bbox MSE with z_cardio=0 forces z_common to explain
        # non-cardiac anatomy, complementing the bbox attention supervision on z_cardio.
        run_phase d3  d3_mi_disc              90  4  0  1.0  1e-4  5e-5  1.0  0.05  0.05  0.15  0.0  0.001  1  "$D3_RESUME"  5  5000  1e-4  1e-4  1e-4  0.3
        ;;

      d4)
        # ── D4: MI + enhanced perceptual + masked recon ───────────────────────
        # Slightly lower batch=8 → fine on 24 GB GPU with perceptual + MI disc
        D3_CKPT=$(require_ckpt "d3_mi_disc" "D3")
        printf "  D3 → %s\n" "$D3_CKPT"
        run_phase d4  d4_mi_percep           120  4  0  1.0  1e-4  5e-5  1.0  0.05  0.05  0.15  0.0  0.001  1  "$D3_CKPT"  5  5000  1e-4  1e-4  1e-4  0.3
        ;;

      d5)
        # ── D5: PatchGAN sharpening ───────────────────────────────────────────
        # BUG FIXES applied here (from d5_gan-20260323-042442 post-mortem):
        #
        # Bug 1 — weight_gan 0.5→0.1: at epoch 140, 0.5×1.25=0.625 vs rec 0.135
        #   → 4.6× GAN-over-reconstruction imbalance caused catastrophic decoder corruption.
        #
        # Bug 2 — gan_start_step is now PHASE-LOCAL (fix in train_sep_vae.py):
        #   global_step restored from D4 ckpt (~56880) instantly exceeded old
        #   gan_start_step=2000. Freshly-init discriminator fired immediately, reached
        #   88% accuracy in 4 epochs, overwhelmed the generator.
        #   train_sep_vae.py now computes phase_local_step = global_step - phase_start_global_step.
        #
        # Bug 3 — lr_patch_disc 1e-4→3e-5 + disc_r1_penalty=10.0:
        #   Slower discriminator convergence + R1 penalty on real samples prevents
        #   discriminator from growing too powerful before generator can adapt.
        #
        # bbox_query_mix=1.0: pure Gaussian prior by D5 (encoder mature enough).
        # weight_bbox_attn=0.1 (was 0.05): stronger spatial penalty.
        D4_CKPT=$(require_ckpt "d4_mi_percep" "D4")
        printf "  D4 → %s\n" "$D4_CKPT"
        _old_bbox_mix="$BBOX_QUERY_MIX"
        BBOX_QUERY_MIX=1.0
        run_phase d5  d5_gan                160  6   0  1.0  1e-4  1e-4  1.0  0.10  0.05  0.15  0.1  0.001  1  "$D4_CKPT"  5  2000  3e-5  1e-4  1e-4  0.3  10.0
        BBOX_QUERY_MIX="$_old_bbox_mix"
        ;;

    esac
done

banner "Curriculum complete  (${START_PHASE^^} → ${END_PHASE^^})"
ok "All phases done — final checkpoint in ${OUTPUT_ROOT}"
