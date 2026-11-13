"""
E4 — CLVM (Severson et al. 2019)
==================================
Adds KL-alignment between z_c marginals of background and salient sets.
Requires (B, S) pairs — matched by positional index in SynDis.

TODO: Implement full paired KL-alignment loss.
      Stub — calls e7_sepvae.py with nulling only (placeholder).
"""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='syndis')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--out_dir',    default='results')
    args, extra = p.parse_known_args()

    cmd = [
        sys.executable, str(_ROOT / 'train' / 'e7_sepvae.py'),
        '--no_clf', '--no_contrastive', '--no_cross_adv',
        '--dataset',    args.dataset,
        '--epochs',     str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--out_dir',    args.out_dir,
        '--exp_name',   'e4_clvm_severson',
    ] + extra
    print("[E4] CLVM Severson stub — full paired alignment not yet implemented.")
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
