"""
Training script for MiniSepVAE — 128×128 fast ablation harness.

Purpose: rapid loss-term ablation. A single run completes in ~5-10 minutes
on any modern GPU, allowing dozens of configurations to be explored before
committing to a full-scale 512×512 run.

Key simplifications vs train_sep_vae.py:
    - No pretrained backbone / CheSS weight loading
    - No batch_stats injection
    - No FPN, no partial unfreezing
    - spatial_attention_loss called manually with latent_hw=16
      (sepvae_loss() always receives weight_spatial_attn=0.0)
    - Reuses all eval helpers from run/train_sep_vae.py

Usage (synthetic dataset — fastest, ~5 min/run):
    python -m run.train_mini_sepvae \\
        --exp_name mini_s1_recon \\
        --batch_size 64 --epochs 30

Usage (real VinBigData at 128px):
    python -m run.train_mini_sepvae \\
        --exp_name mini_s6_vinbig \\
        --use_vinbig --img_size 128 \\
        --batch_size 32 --epochs 50 \\
        --use_label_attention --use_contrastive --use_cross_adv --use_disease_clf
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import torch
from torch.utils.data import DataLoader

# ---- Local imports ----
from models.mini_sep_vae import MiniSepVAE
from losses.sep_vae_losses import (
    MIDiscriminator, CrossHeadDiscriminator, DiseaseClassifier,
    SepVAELossConfig, sepvae_loss,
    spatial_attention_loss,       # called manually with latent_hw=16
)

# Reuse eval helpers from the full training script
from run.train_sep_vae import (
    _collect_latents_from_triplet_loader,
    _pairwise_auc_binary,
    _linear_probe_auc,
    make_sepvae_recon_grid,
    make_attention_map_grid,
    make_bbox_comparison_grid,
    make_heatmap_comparison_grid,
    ensure_dir,
)


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MiniSepVAE ablation trainer")

    # ---- Experiment ----
    p.add_argument('--exp_name', default='mini_sepvae', type=str)
    p.add_argument('--out_dir', default='runs_mini_sepvae', type=str)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='mini-sepvae-ablation', type=str)

    # ---- Dataset ----
    p.add_argument('--use_vinbig', action='store_true',
                   help='Use real VinBigData at 128px instead of SyntheticCXR')
    p.add_argument('--data_dir', default='/data/vinbigdata', type=str)
    p.add_argument('--csv_path', default='/data/vinbigdata/train.csv', type=str)
    p.add_argument('--img_size', default=128, type=int,
                   help='Image size (both H and W). Only used with --use_vinbig.')
    p.add_argument('--n_synthetic', default=2000, type=int,
                   help='Number of synthetic training samples per epoch.')
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--num_workers', default=4, type=int)

    # ---- Training ----
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--lr_vae', default=3e-4, type=float)
    p.add_argument('--lr_disc', default=1e-4, type=float)
    p.add_argument('--lr_cross_disc', default=1e-4, type=float)
    p.add_argument('--lr_disease_clf', default=1e-4, type=float)
    p.add_argument('--lr_decay_epochs', default=0, type=int,
                   help='Epoch to start cosine LR decay (0 = no decay)')
    p.add_argument('--kl_warmup_epochs', default=5, type=int)
    p.add_argument('--sample_every', default=5, type=int,
                   help='Log reconstructions/attention maps every N epochs')

    # ---- Model ----
    p.add_argument('--z_channels_common', default=4, type=int)
    p.add_argument('--z_channels_disease', default=2, type=int)
    p.add_argument('--head_hidden_channels', default=64, type=int)
    p.add_argument('--use_label_attention', action='store_true',
                   help='Use DiseaseAttentionHead (R7)')
    p.add_argument('--attn_query_dim', default=64, type=int)

    # ---- Loss weights ----
    p.add_argument('--weight_rec', default=1.0, type=float)
    p.add_argument('--weight_kl_common', default=1e-4, type=float)
    p.add_argument('--weight_kl_disease', default=1e-4, type=float)
    p.add_argument('--weight_null', default=0.0, type=float)
    p.add_argument('--sigma_inactive', default=0.05, type=float)
    p.add_argument('--weight_orthogonality', default=0.0, type=float)
    p.add_argument('--weight_mi', default=0.0, type=float)
    p.add_argument('--weight_contrastive', default=0.0, type=float)
    p.add_argument('--contrastive_margin', default=0.5, type=float)
    p.add_argument('--weight_cross_adv', default=0.0, type=float)
    p.add_argument('--cross_disc_hidden_dim', default=128, type=int)
    p.add_argument('--weight_disease_clf', default=0.0, type=float)
    p.add_argument('--weight_spatial_attn', default=0.0, type=float,
                   help='Bbox spatial attention supervision (R12). Requires --use_label_attention.')

    # ---- Convenience flags (map to weights) ----
    p.add_argument('--use_contrastive', action='store_true')
    p.add_argument('--use_cross_adv', action='store_true')
    p.add_argument('--use_disease_clf', action='store_true')

    return p.parse_args()


# ============================================================================
# LR schedule helper
# ============================================================================

def make_lr_schedule(base_lr: float, total_epochs: int, decay_start: int):
    if decay_start <= 0 or decay_start >= total_epochs:
        return optax.constant_schedule(base_lr)
    n_const  = decay_start
    n_cosine = total_epochs - decay_start
    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(base_lr),
            optax.cosine_decay_schedule(base_lr, decay_steps=n_cosine,
                                        alpha=0.1),
        ],
        boundaries=[n_const],
    )


# ============================================================================
# Train step
# ============================================================================

def make_train_step(cfg: SepVAELossConfig, weight_spatial_attn: float,
                    has_mi: bool, has_cross: bool, has_clf: bool):
    """Return a JIT-compiled train_step closure."""

    def train_step(
        vae_state, mi_state, cross_c2e_state, cross_e2c_state,
        clf_cardio_state, clf_effusion_state,
        batch, key, kl_anneal,
    ):
        # ---- VAE + auxiliary forward ----
        def vae_loss_fn(vae_params):
            total, (logs, mi_disc_loss, patch_disc_loss, cross_disc_loss) = sepvae_loss(
                model=vae_state.model,
                mi_disc=mi_state.model if has_mi else None,
                params=vae_params,
                mi_params=mi_state.params if has_mi else {},
                batch=batch,
                key=key,
                cfg=cfg,
                batch_stats=None,
                kl_anneal=kl_anneal,
                patch_disc=None,
                patch_disc_params=None,
                backbone_apply_fn=None,
                backbone_variables=None,
                cross_disc_c2e=cross_c2e_state.model if has_cross else None,
                cross_disc_c2e_params=cross_c2e_state.params if has_cross else None,
                cross_disc_e2c=cross_e2c_state.model if has_cross else None,
                cross_disc_e2c_params=cross_e2c_state.params if has_cross else None,
                clf_cardio=clf_cardio_state.model if has_clf else None,
                clf_cardio_params=clf_cardio_state.params if has_clf else None,
                clf_effusion=clf_effusion_state.model if has_clf else None,
                clf_effusion_params=clf_effusion_state.params if has_clf else None,
            )
            # ---- spatial attention term (must be handled here, latent_hw=16) ----
            l_spatial = jnp.float32(0.0)
            if weight_spatial_attn > 0.0:
                # We need the latents_dict from the forward pass — re-run encode
                variables = {'params': vae_params}
                x = jnp.concatenate([
                    batch['x_norm'], batch['x_disease1'], batch['x_disease2']
                ], axis=0)
                _labels = batch['disease_labels']
                x_rec, latents_dict, _ = vae_state.model.apply(
                    variables, x, _labels, key=key
                )
                if ('attn_maps' in latents_dict
                        and 'bbox_disease1' in batch
                        and 'bbox_disease2' in batch):
                    l_spatial = spatial_attention_loss(
                        latents_dict['attn_maps'],
                        batch['bbox_disease1'],
                        batch['bbox_disease2'],
                        latent_hw=16,
                    )
            total = total + weight_spatial_attn * l_spatial
            logs['loss/spatial_attn'] = l_spatial
            return total, (logs, mi_disc_loss, patch_disc_loss, cross_disc_loss)

        (vae_loss, (logs, mi_disc_loss, _, cross_disc_loss)), grads = jax.value_and_grad(
            vae_loss_fn, has_aux=True
        )(vae_state.params)
        vae_state = vae_state.apply_gradients(grads=grads)

        # ---- MI discriminator step ----
        if has_mi:
            def mi_loss_fn(mi_params):
                variables = {'params': vae_state.params}
                x = jnp.concatenate([
                    batch['x_norm'], batch['x_disease1'], batch['x_disease2']
                ], axis=0)
                labels = batch['disease_labels']
                _, latents_dict, _ = vae_state.model.apply(
                    variables, x, labels, key=key
                )
                from losses.sep_vae_losses import mi_discriminator_loss, MIDiscriminator
                disc_fn = lambda zc, zd, train: mi_state.model.apply(
                    {'params': mi_params}, zc, zd, train=train
                )
                disc_loss, _ = mi_discriminator_loss(disc_fn, latents_dict, labels, key)
                return disc_loss

            mi_disc_loss_val, mi_grads = jax.value_and_grad(mi_loss_fn)(mi_state.params)
            mi_state = mi_state.apply_gradients(grads=mi_grads)
        else:
            mi_disc_loss_val = jnp.float32(0.0)

        # ---- Cross-head discriminator step ----
        if has_cross:
            def cross_loss_fn(c2e_params, e2c_params):
                variables = {'params': vae_state.params}
                x = jnp.concatenate([
                    batch['x_norm'], batch['x_disease1'], batch['x_disease2']
                ], axis=0)
                labels = batch['disease_labels']
                _, latents_dict, _ = vae_state.model.apply(
                    variables, x, labels, key=key
                )
                from losses.sep_vae_losses import cross_head_disc_loss
                disc_c2e_fn = lambda z: cross_c2e_state.model.apply({'params': c2e_params}, z)
                disc_e2c_fn = lambda z: cross_e2c_state.model.apply({'params': e2c_params}, z)
                disc_loss, _ = cross_head_disc_loss(disc_c2e_fn, disc_e2c_fn, latents_dict, labels)
                return disc_loss

            cross_disc_val, (c2e_grads, e2c_grads) = jax.value_and_grad(
                cross_loss_fn, argnums=(0, 1)
            )(cross_c2e_state.params, cross_e2c_state.params)
            cross_c2e_state = cross_c2e_state.apply_gradients(grads=c2e_grads)
            cross_e2c_state = cross_e2c_state.apply_gradients(grads=e2c_grads)

        # ---- Disease classifier step ----
        if has_clf:
            def clf_loss_fn(clf_c_params, clf_e_params):
                variables = {'params': vae_state.params}
                x = jnp.concatenate([
                    batch['x_norm'], batch['x_disease1'], batch['x_disease2']
                ], axis=0)
                labels = batch['disease_labels']
                _, latents_dict, _ = vae_state.model.apply(
                    variables, x, labels, key=key
                )
                from losses.sep_vae_losses import disease_discriminability_loss
                clf_c_fn = clf_cardio_state.model.apply
                clf_e_fn = clf_effusion_state.model.apply
                loss, _, _ = disease_discriminability_loss(
                    clf_c_fn, clf_e_fn, clf_c_params, clf_e_params, latents_dict, labels
                )
                return loss

            _, (clf_c_grads, clf_e_grads) = jax.value_and_grad(
                clf_loss_fn, argnums=(0, 1)
            )(clf_cardio_state.params, clf_effusion_state.params)
            clf_cardio_state  = clf_cardio_state.apply_gradients(grads=clf_c_grads)
            clf_effusion_state = clf_effusion_state.apply_gradients(grads=clf_e_grads)

        return (
            vae_state, mi_state, cross_c2e_state, cross_e2c_state,
            clf_cardio_state, clf_effusion_state,
            vae_loss, logs,
        )

    return jax.jit(train_step)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ---- Convenience flag resolution ----
    if args.use_contrastive and args.weight_contrastive == 0.0:
        args.weight_contrastive = 0.1
    if args.use_cross_adv and args.weight_cross_adv == 0.0:
        args.weight_cross_adv = 0.05
    if args.use_disease_clf and args.weight_disease_clf == 0.0:
        args.weight_disease_clf = 0.1

    has_mi    = args.weight_mi    > 0.0
    has_cross = args.weight_cross_adv > 0.0
    has_clf   = args.weight_disease_clf > 0.0

    # ---- W&B ----
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name,
                   config=vars(args))

    # ---- Output dir ----
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(args.out_dir) / f"{args.exp_name}-{ts}"
    ensure_dir(str(run_dir))

    # ---- Dataset ----
    if args.use_vinbig:
        from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
        train_ds = VinBigDataTripletDataset(
            csv_path=args.csv_path,
            dicom_dir=args.data_dir,
            img_size=args.img_size,
            exclude_cross_disease_overlap=True,
        )
    else:
        from datasets.SyntheticCXR import SyntheticCXRTripletDataset, jax_collate_fn
        train_ds = SyntheticCXRTripletDataset(
            n_samples=args.n_synthetic,
            img_size=128,
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=jax_collate_fn,
        drop_last=True,
    )
    eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=jax_collate_fn,
        drop_last=False,
    )

    # ---- Model ----
    model = MiniSepVAE(
        z_channels_common=args.z_channels_common,
        z_channels_disease=args.z_channels_disease,
        sigma_inactive=args.sigma_inactive,
        use_label_attention=args.use_label_attention,
        attn_query_dim=args.attn_query_dim,
        head_hidden_channels=args.head_hidden_channels,
    )

    key = jax.random.PRNGKey(args.seed)
    dummy_x = jnp.zeros((2, 128, 128, 1))
    dummy_labels = jnp.array([0, 1])
    key, init_key = jax.random.split(key)
    variables = model.init(init_key, dummy_x, dummy_labels, key=init_key)
    vae_params = variables['params']
    print(f"MiniSepVAE params: {sum(p.size for p in jax.tree_util.tree_leaves(vae_params)):,}")

    # ---- Loss config ----
    cfg = SepVAELossConfig(
        weight_rec=args.weight_rec,
        weight_kl_common=args.weight_kl_common,
        weight_kl_disease=args.weight_kl_disease,
        weight_null=args.weight_null,
        weight_orthogonality=args.weight_orthogonality,
        weight_mi=args.weight_mi if has_mi else 0.0,
        weight_contrastive=args.weight_contrastive,
        contrastive_margin=args.contrastive_margin,
        weight_cross_adv=args.weight_cross_adv if has_cross else 0.0,
        weight_disease_clf=args.weight_disease_clf if has_clf else 0.0,
        weight_spatial_attn=0.0,   # always 0 here — handled manually with latent_hw=16
        sigma_inactive=args.sigma_inactive,
        disc_start_epoch=0,
    )

    # ---- Optimizers ----
    lr_sched = make_lr_schedule(args.lr_vae, args.epochs, args.lr_decay_epochs)
    vae_tx = optax.adam(lr_sched)

    class SimpleState:
        """Minimal TrainState-like object for auxiliary nets.

        Registered as a JAX pytree so it can be passed through jax.jit:
          - leaves:    (params, opt_state)  — traced JAX arrays
          - auxiliary: (model, tx)          — static Python objects, trigger
                                              recompilation only if they change
        """
        def __init__(self, model_obj, params, tx):
            self.model      = model_obj
            self.params     = params
            self._tx        = tx
            self._opt_state = tx.init(params)

        def apply_gradients(self, grads):
            updates, new_opt = self._tx.update(grads, self._opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)
            # Return a new instance rather than mutating self, so JAX's pytree
            # unflattening gets the updated leaves.
            new_state = object.__new__(SimpleState)
            new_state.model      = self.model
            new_state.params     = new_params
            new_state._tx        = self._tx
            new_state._opt_state = new_opt
            return new_state

    def _simple_state_flatten(s):
        leaves = (s.params, s._opt_state)
        aux    = (s.model, s._tx)
        return leaves, aux

    def _simple_state_unflatten(aux, leaves):
        model, tx = aux
        params, opt_state = leaves
        s = object.__new__(SimpleState)
        s.model      = model
        s.params     = params
        s._tx        = tx
        s._opt_state = opt_state
        return s

    jax.tree_util.register_pytree_node(
        SimpleState,
        _simple_state_flatten,
        _simple_state_unflatten,
    )

    # VAE TrainState
    from flax.training.train_state import TrainState as FlaxTrainState
    from flax import struct as flax_struct

    class VaeState(FlaxTrainState):
        # model must be pytree_node=False so JAX JIT treats it as static metadata,
        # not a traceable leaf (otherwise vae_state.model becomes a DynamicJaxprTracer).
        model: MiniSepVAE = flax_struct.field(pytree_node=False)

    vae_state = VaeState.create(
        apply_fn=model.apply,
        params=vae_params,
        tx=vae_tx,
        model=model,
    )

    # MI discriminator
    key, k = jax.random.split(key)
    mi_disc = MIDiscriminator(hidden_dim=128)
    dummy_zc = jnp.zeros((2, args.z_channels_common))
    dummy_zd = jnp.zeros((2, args.z_channels_disease))
    mi_params = mi_disc.init(k, dummy_zc, dummy_zd)['params'] if has_mi else {}
    mi_state  = SimpleState(mi_disc, mi_params, optax.adam(args.lr_disc)) if has_mi else SimpleState(None, {}, optax.adam(1e-4))

    # Cross-head discriminators
    key, k1, k2 = jax.random.split(key, 3)
    disc_c2e = CrossHeadDiscriminator(hidden_dim=args.cross_disc_hidden_dim)
    disc_e2c = CrossHeadDiscriminator(hidden_dim=args.cross_disc_hidden_dim)
    dummy_zd_flat = jnp.zeros((2, args.z_channels_disease))
    c2e_params = disc_c2e.init(k1, dummy_zd_flat)['params'] if has_cross else {}
    e2c_params = disc_e2c.init(k2, dummy_zd_flat)['params'] if has_cross else {}
    cross_c2e_state = SimpleState(disc_c2e, c2e_params, optax.adam(args.lr_cross_disc))
    cross_e2c_state = SimpleState(disc_e2c, e2c_params, optax.adam(args.lr_cross_disc))

    # Disease classifiers
    key, k1, k2 = jax.random.split(key, 3)
    clf_cardio   = DiseaseClassifier(hidden_dim=16)
    clf_effusion = DiseaseClassifier(hidden_dim=16)
    clf_c_params = clf_cardio.init(k1, dummy_zd_flat)['params'] if has_clf else {}
    clf_e_params = clf_effusion.init(k2, dummy_zd_flat)['params'] if has_clf else {}
    clf_cardio_state   = SimpleState(clf_cardio,   clf_c_params, optax.adam(args.lr_disease_clf))
    clf_effusion_state = SimpleState(clf_effusion, clf_e_params, optax.adam(args.lr_disease_clf))

    # ---- Build train step ----
    train_step = make_train_step(
        cfg=cfg,
        weight_spatial_attn=args.weight_spatial_attn,
        has_mi=has_mi,
        has_cross=has_cross,
        has_clf=has_clf,
    )

    # ---- Training loop ----
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        kl_anneal = jnp.float32(min(1.0, epoch / max(args.kl_warmup_epochs, 1)))
        epoch_losses = {}

        for batch in train_loader:
            key, step_key = jax.random.split(key)
            (
                vae_state, mi_state, cross_c2e_state, cross_e2c_state,
                clf_cardio_state, clf_effusion_state,
                vae_loss, logs,
            ) = train_step(
                vae_state, mi_state, cross_c2e_state, cross_e2c_state,
                clf_cardio_state, clf_effusion_state,
                batch, step_key, kl_anneal,
            )

            for k_log, v_log in logs.items():
                epoch_losses.setdefault(k_log, []).append(float(v_log))
            global_step += 1

        # ---- Epoch summary ----
        summary = {k: np.mean(vs) for k, vs in epoch_losses.items()}
        summary['epoch'] = epoch
        summary['kl_anneal'] = float(kl_anneal)

        # ---- Disentanglement metrics ----
        lc, lcard, leff, lab = _collect_latents_from_triplet_loader(
            model, {'params': vae_state.params}, eval_loader, max_samples=400
        )
        auc_c = _linear_probe_auc(lcard, (lab == 2).astype(int))
        auc_e = _linear_probe_auc(leff,  (lab == 1).astype(int))
        # Cross-head score: probe cardio from effusion head & vice versa
        xh_c2e = _linear_probe_auc(leff,  (lab == 2).astype(int))
        xh_e2c = _linear_probe_auc(lcard, (lab == 1).astype(int))
        cross_head_score = 0.5 * (xh_c2e + xh_e2c)

        summary['probe_auc/cardiomegaly'] = auc_c if not np.isnan(auc_c) else 0.0
        summary['probe_auc/effusion']     = auc_e if not np.isnan(auc_e) else 0.0
        summary['cross_head_score']       = cross_head_score if not np.isnan(cross_head_score) else 1.0

        # ---- Gate status ----
        g1 = summary['probe_auc/cardiomegaly'] > 0.75 and summary['probe_auc/effusion'] > 0.75
        g2 = summary['cross_head_score'] < 0.65
        summary['gate/G1_met'] = float(g1)
        summary['gate/G2_met'] = float(g2)

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"loss={summary.get('loss/total', 0.0):.4f}  "
            f"rec={summary.get('loss/reconstruction', 0.0):.4f}  "
            f"probe_c={summary['probe_auc/cardiomegaly']:.3f}  "
            f"probe_e={summary['probe_auc/effusion']:.3f}  "
            f"xhead={summary['cross_head_score']:.3f}  "
            f"G1={'✓' if g1 else '✗'}  G2={'✓' if g2 else '✗'}"
        )

        if use_wandb:
            import wandb
            wandb.log(summary, step=epoch)

        # ---- Visual samples ----
        if epoch % args.sample_every == 0:
            try:
                sample_batch = next(iter(eval_loader))
                x_sample = jnp.concatenate([
                    sample_batch['x_norm'],
                    sample_batch['x_disease1'],
                    sample_batch['x_disease2'],
                ], axis=0)
                labels_sample = sample_batch['disease_labels']

                x_rec_sample, latents_s, _ = model.apply(
                    {'params': vae_state.params},
                    x_sample, labels_sample, key=key
                )
                grid = make_sepvae_recon_grid(x_sample, x_rec_sample, labels_sample)
                grid_path = str(run_dir / f"recon_ep{epoch:03d}.png")
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.imsave(grid_path, grid, cmap='gray')

                wandb_log = {'viz/recon': wandb.Image(grid_path)} if use_wandb else {}

                if args.use_label_attention and 'attn_maps' in latents_s:
                    attn_grid = make_attention_map_grid(
                        x_sample, latents_s['attn_maps'], labels_sample
                    )
                    attn_path = str(run_dir / f"attn_ep{epoch:03d}.png")
                    plt.imsave(attn_path, attn_grid)
                    if use_wandb:
                        wandb_log['viz/attn_overlay'] = wandb.Image(attn_path)

                    # Bbox comparison and heatmap comparison grids
                    # (only when GT bboxes are available, i.e. VinBigData)
                    has_bboxes = (
                        'bbox_disease1' in sample_batch
                        and 'bbox_disease2' in sample_batch
                    )
                    if has_bboxes:
                        bbox_grid = make_bbox_comparison_grid(
                            x_sample, x_rec_sample,
                            latents_s['attn_maps'],
                            sample_batch['bbox_disease1'],
                            sample_batch['bbox_disease2'],
                            labels_sample,
                        )
                        bbox_path = str(run_dir / f"bbox_compare_ep{epoch:03d}.png")
                        bbox_grid.save(bbox_path)

                        heat_grid = make_heatmap_comparison_grid(
                            x_sample, x_rec_sample,
                            latents_s['attn_maps'],
                            sample_batch['bbox_disease1'],
                            sample_batch['bbox_disease2'],
                            labels_sample,
                        )
                        heat_path = str(run_dir / f"heatmap_compare_ep{epoch:03d}.png")
                        heat_grid.save(heat_path)

                        if use_wandb:
                            wandb_log['viz/bbox_gt_vs_pred']    = wandb.Image(bbox_path)
                            wandb_log['viz/heatmap_gt_vs_pred'] = wandb.Image(heat_path)

                if use_wandb and wandb_log:
                    import wandb
                    wandb.log(wandb_log, step=epoch)
            except Exception as e:
                print(f"  [viz skip: {e}]")

    print("\nTraining complete.")
    print(f"Run saved to: {run_dir}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
