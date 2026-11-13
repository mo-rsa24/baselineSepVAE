"""
E2 — Split-Head VAE (ablation baseline)
========================================
Three independent encoder heads (z_c, z_d1, z_d2) with NO cross-head losses.
Shows baseline entanglement when heads are split but untrained to separate.

Run:
    python train/e2_split_head.py
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# E2 = E7 with all cross-losses disabled. Delegate to e7_sepvae.py.
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='syndis')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--out_dir',    default='results')
    args, extra = p.parse_known_args()

    cmd = [
        sys.executable, str(_ROOT / 'train' / 'e7_sepvae.py'),
        '--no_clf', '--no_contrastive', '--no_cross_adv', '--no_nulling',
        '--dataset',    args.dataset,
        '--epochs',     str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--out_dir',    args.out_dir,
        '--exp_name',   'e2_split_head',
    ] + extra
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
