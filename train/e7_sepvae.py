"""
E7 — SepVAE (Ours)
==================
Full loss stack on A1 CNN:
  MSE + β·(KL_c + KL_d1 + KL_d2)
  + λ_null  · nulling (normal samples)
  + λ_orth  · orthogonality (d1 ⊥ d2)
  + λ_cont  · contrastive margin (R5a)
  + λ_adv   · cross-head adversarial (R5b)
  + λ_clf   · disease classifier BCE per head (R11)

Run (SynDis, all defaults):
    python train/e7_sepvae.py

Run (SpyMNIST):
    python train/e7_sepvae.py --dataset spymnist

Run (no R11, no R5):
    python train/e7_sepvae.py --no_clf --no_contrastive --no_cross_adv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.SynDis import SynDisDataset, BLOB_ROWS, BLOB_COLS, STRIPE_ROWS, STRIPE_COLS
from models.cnn_vae import CNNSepVAE
from losses.base import kl_loss, mse_loss
from losses.nulling import nulling_loss, orthogonality_loss
from losses.contrastive import margin_contrastive_loss
from losses.cross_adversarial import CrossHeadDiscriminator, cross_adv_discriminator_loss, cross_adv_encoder_loss
from losses.disease_clf_loss import DiseaseClassifier, clf_bce_loss
from train.trainer_base import compute_metrics, save_recon_grid, save_metrics


def make_datasets(name: str):
    if name == 'syndis':
        train_ds = SynDisDataset(n_normal=1500, n_disease1=750, n_disease2=750,
                                 n_both=0,   global_seed=42)
        val_ds   = SynDisDataset(n_normal=300,  n_disease1=150, n_disease2=150,
                                 n_both=150, global_seed=99)
        region_blob   = (BLOB_ROWS,   BLOB_COLS)
        region_stripe = (STRIPE_ROWS, STRIPE_COLS)
        return train_ds, val_ds, region_blob, region_stripe
    raise ValueError(f"Unknown dataset: {name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',       default='syndis')
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr_vae',        type=float, default=1e-3)
    p.add_argument('--lr_disc',       type=float, default=1e-4)
    p.add_argument('--lr_clf',        type=float, default=1e-4)
    p.add_argument('--beta',          type=float, default=1.0)
    p.add_argument('--weight_null',   type=float, default=0.05)
    p.add_argument('--weight_orth',   type=float, default=0.05)
    p.add_argument('--weight_cont',   type=float, default=0.1)
    p.add_argument('--cont_margin',   type=float, default=0.5)
    p.add_argument('--weight_adv',    type=float, default=0.05)
    p.add_argument('--weight_clf',    type=float, default=0.1)
    p.add_argument('--z_common',      type=int,   default=4)
    p.add_argument('--z_disease',     type=int,   default=2)
    p.add_argument('--base_ch',       type=int,   default=32)
    p.add_argument('--kl_warmup',     type=int,   default=10)
    # Ablation flags
    p.add_argument('--no_clf',        action='store_true')
    p.add_argument('--no_contrastive',action='store_true')
    p.add_argument('--no_cross_adv',  action='store_true')
    p.add_argument('--no_nulling',    action='store_true')
    p.add_argument('--use_label_attn',action='store_true')
    p.add_argument('--exp_name',      default='e7_sepvae')
    p.add_argument('--out_dir',       default='results')
    p.add_argument('--eval_every',    type=int, default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, region_blob, region_stripe = make_datasets(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = CNNSepVAE(
        in_ch=1, base_ch=args.base_ch,
        z_common=args.z_common, z_disease=args.z_disease,
        use_label_attn=args.use_label_attn,
    ).to(device)

    # Optional auxiliary modules
    clf_d1 = clf_d2 = None
    disc_d1_to_d2 = disc_d2_to_d1 = None

    if not args.no_clf:
        clf_d1 = DiseaseClassifier(args.z_disease, hidden=16).to(device)
        clf_d2 = DiseaseClassifier(args.z_disease, hidden=16).to(device)

    if not args.no_cross_adv:
        disc_d1_to_d2 = CrossHeadDiscriminator(args.z_disease, hidden_dim=64).to(device)
        disc_d2_to_d1 = CrossHeadDiscriminator(args.z_disease, hidden_dim=64).to(device)

    # Optimisers
    vae_params = list(model.parameters())
    if clf_d1: vae_params += list(clf_d1.parameters()) + list(clf_d2.parameters())
    opt_vae = optim.Adam(vae_params, lr=args.lr_vae)

    opt_disc = None
    if disc_d1_to_d2:
        disc_params = list(disc_d1_to_d2.parameters()) + list(disc_d2_to_d1.parameters())
        opt_disc = optim.Adam(disc_params, lr=args.lr_disc)

    print(f"[E7] SepVAE | device={device} | null={not args.no_nulling} "
          f"cont={not args.no_contrastive} adv={not args.no_cross_adv} "
          f"clf={not args.no_clf} attn={args.use_label_attn}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        kl_weight = min(1.0, epoch / max(args.kl_warmup, 1))
        stats = dict(loss=0, mse=0, kl=0, null=0, orth=0, cont=0, adv=0, clf=0)
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B    = x.shape[0]

            has_d1   = ((y == 1) | (y == 3)).float()
            has_d2   = ((y == 2) | (y == 3)).float()
            is_norm  = (y == 0).float()

            # ── Discriminator step ──────────────────────────────────────────
            if opt_disc is not None:
                with torch.no_grad():
                    x_hat, enc = model(x)
                mu_d1 = enc['mu_d1'].mean((-2,-1)) if enc['mu_d1'].dim() > 2 else enc['mu_d1']
                mu_d2 = enc['mu_d2'].mean((-2,-1)) if enc['mu_d2'].dim() > 2 else enc['mu_d2']

                l_disc = (cross_adv_discriminator_loss(disc_d1_to_d2, mu_d1, mu_d2) +
                          cross_adv_discriminator_loss(disc_d2_to_d1, mu_d2, mu_d1))
                opt_disc.zero_grad()
                l_disc.backward()
                opt_disc.step()

            # ── VAE step ────────────────────────────────────────────────────
            x_hat, enc = model(x)

            # GAP disease latents for auxiliary losses
            mu_d1g = enc['mu_d1'].mean((-2,-1)) if enc['mu_d1'].dim() > 2 else enc['mu_d1']
            mu_d2g = enc['mu_d2'].mean((-2,-1)) if enc['mu_d2'].dim() > 2 else enc['mu_d2']

            l_mse  = mse_loss(x_hat, x)
            l_kl   = (kl_loss(enc['mu_c'],  enc['lv_c'])  +
                      kl_loss(enc['mu_d1'], enc['lv_d1']) +
                      kl_loss(enc['mu_d2'], enc['lv_d2']))
            loss   = l_mse + args.beta * kl_weight * l_kl

            l_null = l_orth = l_cont = l_adv = l_clf = torch.zeros(1, device=device)[0]

            if not args.no_nulling:
                l_null = (nulling_loss(enc['mu_d1'], is_norm) +
                          nulling_loss(enc['mu_d2'], is_norm))
                loss   = loss + args.weight_null * l_null

            l_orth = orthogonality_loss(enc['mu_d1'], enc['mu_d2'])
            loss   = loss + args.weight_orth * l_orth

            if not args.no_contrastive:
                l_cont = margin_contrastive_loss(mu_d1g, mu_d2g, args.cont_margin)
                loss   = loss + args.weight_cont * l_cont

            if not args.no_cross_adv and disc_d1_to_d2 is not None:
                l_adv = (cross_adv_encoder_loss(disc_d1_to_d2, mu_d1g, mu_d2g) +
                         cross_adv_encoder_loss(disc_d2_to_d1, mu_d2g, mu_d1g))
                loss  = loss + args.weight_adv * l_adv

            if not args.no_clf and clf_d1 is not None:
                l_clf = (clf_bce_loss(clf_d1, enc['mu_d1'], has_d1) +
                         clf_bce_loss(clf_d2, enc['mu_d2'], has_d2))
                loss  = loss + args.weight_clf * l_clf

            opt_vae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_vae.step()

            stats['loss'] += loss.item()
            stats['mse']  += l_mse.item()
            stats['kl']   += l_kl.item()
            stats['null'] += l_null.item()
            stats['orth'] += l_orth.item()
            stats['cont'] += l_cont.item()
            stats['adv']  += l_adv.item()
            stats['clf']  += l_clf.item()
            n_batches     += 1

        nb = n_batches
        print(f"Ep {epoch:04d} | loss={stats['loss']/nb:.4f} mse={stats['mse']/nb:.4f} "
              f"kl={stats['kl']/nb:.4f} null={stats['null']/nb:.4f} "
              f"orth={stats['orth']/nb:.4f} cont={stats['cont']/nb:.4f} "
              f"adv={stats['adv']/nb:.4f} clf={stats['clf']/nb:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = compute_metrics(model, val_loader, device, region_blob, region_stripe)
            save_metrics(metrics, str(out_dir), epoch)
            print(f"  spec_d1={metrics['spec_ratio_d1']:.3f}  "
                  f"spec_d2={metrics['spec_ratio_d2']:.3f}  "
                  f"comp={metrics['comp_score']:.4f}")

            val_batch = next(iter(val_loader))
            save_recon_grid(model, val_batch, device,
                            str(out_dir / f'recon_ep{epoch:04d}.png'))

    torch.save(model.state_dict(), str(out_dir / 'model_final.pt'))
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
