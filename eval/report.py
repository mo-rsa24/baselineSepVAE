"""
report.py — Load all metrics.json files and produce a comparison table.

Usage:
    python eval/report.py --results_dir results/
    python eval/report.py --results_dir results/ --latex
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


METRIC_COLS = [
    'rec_mse', 'kl_per_dim',
    'auc_d1_d1', 'auc_d2_d2', 'auc_d1_d2', 'auc_d2_d1',
    'spec_ratio_d1', 'spec_ratio_d2',
    'auc_c_d1', 'auc_c_d2',
    'comp_blob_delta', 'comp_stripe_delta', 'comp_score',
]


def load_all(results_dir: str) -> dict:
    root = Path(results_dir)
    rows = {}
    for metrics_file in sorted(root.glob('*/metrics.json')):
        exp_name = metrics_file.parent.name
        with open(metrics_file) as f:
            rows[exp_name] = json.load(f)
    return rows


def print_table(rows: dict, latex: bool = False):
    if not rows:
        print("No metrics.json files found.")
        return

    header = ['Experiment'] + METRIC_COLS
    sep = ' | ' if not latex else ' & '
    end = '' if not latex else r' \\'

    if latex:
        print(r'\begin{tabular}{l' + 'r' * len(METRIC_COLS) + '}')
        print(r'\toprule')
        print(sep.join(header) + end)
        print(r'\midrule')
    else:
        col_w = 14
        print('  '.join(f'{h:<{col_w}}' for h in header))
        print('-' * (col_w + 2) * len(header))

    for exp_name, metrics in rows.items():
        vals = [exp_name] + [f'{metrics.get(col, float("nan")):.4f}' for col in METRIC_COLS]
        if latex:
            print(sep.join(vals) + end)
        else:
            print('  '.join(f'{v:<14}' for v in vals))

    if latex:
        print(r'\bottomrule')
        print(r'\end{tabular}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--latex', action='store_true')
    args = p.parse_args()

    rows = load_all(args.results_dir)
    print_table(rows, args.latex)


if __name__ == '__main__':
    main()
