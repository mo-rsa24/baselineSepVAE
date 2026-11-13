#!/usr/bin/env bash
# run_all_syndis.sh — Run E0–E7 on SynDis (T1) sequentially.
# Usage:
#   bash launchers/run_all_syndis.sh              # all experiments
#   bash launchers/run_all_syndis.sh e0 e7        # specific experiments only
#   EPOCHS=50 bash launchers/run_all_syndis.sh    # override epochs

set -euo pipefail
cd "$(dirname "$0")/.."   # run from sepVAE root

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
DATASET="syndis"
OUT="results"

# Which experiments to run (all by default, or subset from args)
ALL_EXPS=(e0 e1 e2 e3 e4 e7)
if [ $# -gt 0 ]; then
    ALL_EXPS=("$@")
fi

for EXP in "${ALL_EXPS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $EXP  |  dataset=$DATASET  epochs=$EPOCHS"
    echo "════════════════════════════════════════════════════"
    case "$EXP" in
        e0) python train/e0_vanilla.py   --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT ;;
        e1) python train/e1_beta_vae.py  --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT --beta 4 ;;
        e1_sweep) python train/e1_beta_vae.py --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT --beta_sweep ;;
        e2) python train/e2_split_head.py --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT ;;
        e3) python train/e3_cvae_abid.py  --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT ;;
        e4) python train/e4_clvm_severson.py --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT ;;
        e5) python train/e5_mmclvm_weinberger.py ;;
        e6) python train/e6_zou.py ;;
        e7) python train/e7_sepvae.py    --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT ;;
        e7_attn) python train/e7_sepvae.py --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH --out_dir $OUT --use_label_attn --exp_name e7_sepvae_attn ;;
        *) echo "Unknown experiment: $EXP"; exit 1 ;;
    esac
done

echo ""
echo "All done. Results in $OUT/"
echo "Generate comparison table with: python eval/report.py --results_dir $OUT"
