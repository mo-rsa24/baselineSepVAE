"""
E1 — β-VAE
==========
Single latent z (dim=8), same as E0 but β swept over {1,2,4,6,8,10}.
Default comparison β = 4.

Run:
    python train/e1_beta_vae.py --beta 4
    python train/e1_beta_vae.py --beta_sweep  # runs all β values sequentially
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# E1 is identical to E0 with a different β and exp_name.
# Delegate to e0_vanilla.py with overridden args.

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--beta',       type=float, default=4.0)
    p.add_argument('--beta_sweep', action='store_true')
    p.add_argument('--dataset',    default='syndis')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--out_dir',    default='results')
    args, extra = p.parse_known_args()

    betas = [1, 2, 4, 6, 8, 10] if args.beta_sweep else [args.beta]

    for beta in betas:
        cmd = [
            sys.executable, str(_ROOT / 'train' / 'e0_vanilla.py'),
            '--beta',       str(beta),
            '--dataset',    args.dataset,
            '--epochs',     str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--out_dir',    args.out_dir,
            '--exp_name',   f'e1_beta{beta}',
        ] + extra
        print(f"\n{'='*60}")
        print(f"[E1] β={beta}")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
