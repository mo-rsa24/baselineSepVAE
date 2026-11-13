"""
metrics.py — Standalone metric computation (all experiments share this).

Also usable as a CLI to re-evaluate a saved checkpoint.

Usage:
    python eval/metrics.py --checkpoint results/e7_sepvae/model_final.pt \
                           --dataset syndis
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.SynDis import SynDisDataset, BLOB_ROWS, BLOB_COLS, STRIPE_ROWS, STRIPE_COLS
from models.cnn_vae import CNNSepVAE
from train.trainer_base import compute_metrics


def load_model(ckpt_path: str, device: torch.device, **kwargs) -> CNNSepVAE:
    model = CNNSepVAE(**kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset',    default='syndis')
    p.add_argument('--split',      default='test', choices=['val', 'test'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--z_common',   type=int, default=4)
    p.add_argument('--z_disease',  type=int, default=2)
    p.add_argument('--base_ch',    type=int, default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'syndis':
        seed = 99 if args.split == 'val' else 7
        ds = SynDisDataset(n_normal=300, n_disease1=150, n_disease2=150,
                           n_both=150, global_seed=seed)
        region_blob   = (BLOB_ROWS,   BLOB_COLS)
        region_stripe = (STRIPE_ROWS, STRIPE_COLS)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model  = load_model(args.checkpoint, device,
                        in_ch=1, base_ch=args.base_ch,
                        z_common=args.z_common, z_disease=args.z_disease)

    metrics = compute_metrics(model, loader, device, region_blob, region_stripe)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
