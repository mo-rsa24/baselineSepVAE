"""
E3 — Contrastive VAE / cVAE (Abid et al. 2019)
================================================
Loss: MSE + KL_c + KL_s − λ · KL_s(background)

Two variants:
  E3a — single salient head z_s (all diseases pooled)
  E3b — two independent salient heads z_d1, z_d2 with per-head nulling

Run:
    python train/e3_cvae_abid.py --variant b

TODO: Implement full training loop.
      Stub — calls e7_sepvae.py with only nulling enabled (approximation).
"""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',    choices=['a','b'], default='b')
    p.add_argument('--dataset',    default='syndis')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--out_dir',    default='results')
    p.add_argument('--weight_null',type=float, default=0.1)
    args, extra = p.parse_known_args()

    # E3 approximation: nulling only (no contrastive, no cross-adv, no clf)
    cmd = [
        sys.executable, str(_ROOT / 'train' / 'e7_sepvae.py'),
        '--no_clf', '--no_contrastive', '--no_cross_adv',
        '--weight_null', str(args.weight_null),
        '--dataset',    args.dataset,
        '--epochs',     str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--out_dir',    args.out_dir,
        '--exp_name',   f'e3_cvae_abid_{args.variant}',
    ] + extra
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
