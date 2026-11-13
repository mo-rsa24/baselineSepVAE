#!/usr/bin/env bash
# ============================================================================
# run_all_domains_d0_d5.sh
# ============================================================================
# Runs the full d0–d5 generalisation experiment suite: 4 domains × 6 stages
# = 24 runs. Each run appends a row to results/domain_comparison_table.csv.
#
# Estimated wall time:
#   GPU: ~2–3 min/run × 24 runs ≈ 60–75 min total (sequential)
#   CPU: ~5–8 min/run × 24 runs ≈ 2–3 hours total
#
# Parallelise across domains by running one domain per GPU:
#   bash run_domain.sh A &
#   bash run_domain.sh B &
#   bash run_domain.sh C &
#   bash run_domain.sh D &
#
# Usage:
#   cd /home-mscluster/mmolefe/Playground/PhD/miniSepVAE
#   bash launchers/single_runs/generalize/run_all_domains_d0_d5.sh
#
#   # With W&B logging:
#   WANDB=--wandb bash launchers/single_runs/generalize/run_all_domains_d0_d5.sh
#
# Results:
#   results/domain_comparison_table.csv   — one row per run, all metrics
#   runs_generalize/<domain>_<stage>_v1/  — per-run diagnostics + checkpoints
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
WANDB="${WANDB:-}"      # set to --wandb to enable W&B logging
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
SEED="${SEED:-42}"
EXP_SUFFIX="${EXP_SUFFIX:-v1}"
RESULTS_CSV="${RESULTS_CSV:-results/domain_comparison_table.csv}"

DOMAINS=(A B C D)
STAGES=(d0 d1 d2 d3 d4 d5)

# Correlation rates for domains B and D
CORR_B="${CORR_B:-0.30}"
CORR_D="${CORR_D:-0.50}"

echo "========================================================"
echo " d0–d5 Generalisation Suite"
echo " Domains: ${DOMAINS[*]}"
echo " Stages:  ${STAGES[*]}"
echo " Epochs:  ${EPOCHS}"
echo " Results: ${RESULTS_CSV}"
echo "========================================================"

run_count=0
fail_count=0

for domain in "${DOMAINS[@]}"; do
    # Set correlation rate based on domain
    if [[ "$domain" == "B" ]]; then
        corr_arg="--correlation_rate ${CORR_B}"
    elif [[ "$domain" == "D" ]]; then
        corr_arg="--correlation_rate ${CORR_D}"
    else
        corr_arg="--correlation_rate 0.0"
    fi

    for stage in "${STAGES[@]}"; do
        exp_name="${EXP_SUFFIX}"
        run_label="Domain-${domain} Stage-${stage}"

        echo ""
        echo "──────────────────────────────────────────────────────"
        echo "  Running: ${run_label}  (${run_count}/24 done)"
        echo "──────────────────────────────────────────────────────"

        t_start=$(date +%s)

        if $PYTHON -m run.train_generalize \
            --domain "${domain}" \
            --stage  "${stage}"  \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH}" \
            --seed   "${SEED}"   \
            --exp_name "${exp_name}" \
            --results_csv "${RESULTS_CSV}" \
            $corr_arg \
            $WANDB ; then
            t_end=$(date +%s)
            echo "  ✓ Completed in $((t_end - t_start))s"
            run_count=$((run_count + 1))
        else
            echo "  ✗ FAILED: ${run_label}"
            fail_count=$((fail_count + 1))
            run_count=$((run_count + 1))
        fi
    done
done

echo ""
echo "========================================================"
echo " Suite complete: $((run_count - fail_count)) / ${run_count} runs succeeded"
echo " Results table: ${RESULTS_CSV}"
echo "========================================================"

# Print the comparison table if python is available
if [[ -f "${RESULTS_CSV}" ]]; then
    echo ""
    echo "── Summary table (last ${run_count} rows) ──────────────────"
    $PYTHON - <<'PYEOF'
import csv, sys
from pathlib import Path

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/domain_comparison_table.csv")
if not csv_path.exists():
    print("(CSV not found)")
    sys.exit(0)

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# Print header + all rows with gate summary
cols = ["domain","stage","probe_auc_f1","probe_auc_f2",
        "cross_head_score","norm_ratio_f1","composition_score",
        "mi_disc_acc","G1","G2","G3"]
hdr = f"{'Dom':>4} {'Stage':>5}  {'AUC_f1':>6}  {'AUC_f2':>6}  {'xHead':>6}  {'NR_f1':>5}  {'CompSc':>7}  {'MI_acc':>6}  G1 G2 G3"
print(hdr)
print("─" * len(hdr))
for r in rows:
    g = lambda k: "✓" if r.get(k,"0")=="1" else "✗"
    print(f"{r['domain']:>4} {r['stage']:>5}  "
          f"{float(r['probe_auc_f1']):.4f}  {float(r['probe_auc_f2']):.4f}  "
          f"{float(r['cross_head_score']):.4f}  {float(r['norm_ratio_f1']):.2f}  "
          f"{float(r['composition_score']):+.5f}  {float(r['mi_disc_acc']):.4f}  "
          f"{g('G1')}  {g('G2')}  {g('G3')}")
PYEOF
fi

exit ${fail_count}
