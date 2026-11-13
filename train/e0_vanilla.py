"""
E0 — Vanilla VAE (β=1)
======================
Baseline: single latent z (dim=8), ELBO only.
No factorised heads; composition is impossible.

Run:
    python train/e0_vanilla.py --dataset syndis --epochs 100 --batch_size 64
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Repo root on sys.path so datasets/* and models/* resolve
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.SynDis import SynDisDataset
from models.mlp_vae import MLPVAE
from losses.base import kl_loss, mse_loss
from train.trainer_base import save_recon_grid, save_metrics


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def make_datasets(name: str):
    if name == 'syndis':
        train_ds = SynDisDataset(n_normal=1500, n_disease1=750, n_disease2=750,
                                 n_both=0,   global_seed=42)
        val_ds   = SynDisDataset(n_normal=300,  n_disease1=150, n_disease2=150,
                                 n_both=150, global_seed=99)
        return train_ds, val_ds
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    default='syndis')
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--beta',       type=float, default=1.0)
    parser.add_argument('--z_dim',      type=int, default=8)
    parser.add_argument('--exp_name',   default='e0_vanilla')
    parser.add_argument('--out_dir',    default='results')
    parser.add_argument('--eval_every', type=int, default=10)
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = make_datasets(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # E0 uses MLP VAE with a single latent head
    model = MLPVAE(img_shape=(1, 64, 64), z_dim=args.z_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[E0] Vanilla VAE | β={args.beta} | z_dim={args.z_dim} | device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_mse = total_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, lv = model(x)

            loss_rec = mse_loss(x_hat, x)
            loss_kl  = kl_loss(mu, lv)
            loss     = loss_rec + args.beta * loss_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_mse  += loss_rec.item()
            total_kl   += loss_kl.item()

        n = len(train_loader)
        print(f"Ep {epoch:04d} | loss={total_loss/n:.4f} "
              f"mse={total_mse/n:.4f} kl={total_kl/n:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # E0 has no split heads — wrap in adapter for compute_metrics
            metrics = _eval_e0(model, val_loader, device)
            save_metrics(metrics, str(out_dir), epoch)
            print(f"  val: mse={metrics['rec_mse']:.4f}  kl/dim={metrics['kl_per_dim']:.4f}")

            # Recon grid
            val_batch = next(iter(val_loader))
            save_recon_grid(
                _AdapterModel(model), val_batch, device,
                str(out_dir / f'recon_ep{epoch:04d}.png')
            )

    torch.save(model.state_dict(), str(out_dir / 'model_final.pt'))
    print(f"Saved to {out_dir}")


# ---------------------------------------------------------------------------
# Adapter: wrap MLPVAE so trainer_base.compute_metrics works
# (E0 has no disease heads — expose dummy zero mu_d1, mu_d2)
# ---------------------------------------------------------------------------

class _AdapterModel(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae     = vae
        self.decoder = _E0Decoder(vae)

    def forward(self, x):
        x_hat, mu, lv = self.vae(x)
        B = x.shape[0]
        zeros4 = torch.zeros(B, 4, device=x.device)
        zeros2 = torch.zeros(B, 2, device=x.device)
        enc = dict(
            mu_c=mu[:, :4] if mu.shape[1] >= 4 else zeros4,
            lv_c=lv[:, :4] if lv.shape[1] >= 4 else zeros4,
            mu_d1=zeros2, lv_d1=zeros2,
            mu_d2=zeros2, lv_d2=zeros2,
        )
        return x_hat, enc

    def train(self, mode=True):
        self.vae.train(mode)
        return self

    def eval(self):
        self.vae.eval()
        return self


class _E0Decoder:
    """Stub so trainer_base._composition_metrics doesn't crash."""
    def __call__(self, *args):
        # Return zeros — E0 has no disease heads for composition
        return args[0] * 0


def _eval_e0(model, loader, device):
    """Minimal eval for E0 — no split heads."""
    import numpy as np
    model.eval()
    total_mse = total_kl = n = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, lv = model(x)
            total_mse += torch.nn.functional.mse_loss(x_hat, x, reduction='sum').item()
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
            total_kl += kl.sum().item()
            n += x.shape[0]
    n_dims = model.encoder.mu_head.out_features
    return {
        'rec_mse':       total_mse / n,
        'kl_per_dim':    total_kl / (n * n_dims),
        'auc_d1_d1':     0.5, 'auc_d2_d2': 0.5,
        'auc_d1_d2':     0.5, 'auc_d2_d1': 0.5,
        'auc_c_d1':      0.5, 'auc_c_d2':  0.5,
        'spec_ratio_d1': 1.0, 'spec_ratio_d2': 1.0,
        'comp_blob_delta':   0.0,
        'comp_stripe_delta': 0.0,
        'comp_score':        0.0,
    }


if __name__ == '__main__':
    main()
