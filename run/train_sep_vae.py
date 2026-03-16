"""
Binary SepVAE training script — V1 (CheSS backbone) and V2 (ResNet-50 scratch).

z = [z_common(16ch), z_cardio(16ch)]  @ 16×16 spatial (V2) / 64×64 (V1)

Architecture (V2):
  Encoder: ResNet-50 from scratch + CBAM + self-attn at layer3 bottleneck
           + two learnable Layer4 branches (bg → z_common, tg → z_cardio)
           + optional BboxCrossAttnHead (D1+)
  Nulling: hard-zero z_cardio for Normal images before decoding
  Decoder: progressive bilinear upsampling 16→256 with SE-gated ResBlocks

Loss stack (two separate optimizers):
  VAE optimizer:
    L_rec (MSE) + KL_common + KL_cardio + κ·L_mi + λ·L_bbox + γ·L_perceptual
  Discriminator optimizer:
    L_disc = BCE(D(z_c, z_ca), joint=1) + BCE(D(z_c, z_ca[perm]), marginal=0)
"""

import os
# Must be set before JAX/XLA initialises — suppresses C++ WARNING-level logs
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',    '3')
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('GLOG_minloglevel',         '3')

import argparse
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, msgpack_restore, from_state_dict
import torch
from torch.utils.data import DataLoader

from datasets.VinBigData import VinBigDataPairDataset, jax_pair_collate_fn
from losses.sep_vae_losses import (
    SepVAELossConfig, sepvae_loss,
    FactorDiscriminator, factor_disc_loss,
)

try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


def parse_args():
    p = argparse.ArgumentParser("Binary SepVAE trainer (Normal vs. Cardiomegaly)")

    # Data
    p.add_argument("--dicom_dir",  type=str, default="/datasets/mmolefe/vinbigdata/train")
    p.add_argument("--csv_path",   type=str, default="/datasets/mmolefe/vinbigdata/train.csv")
    p.add_argument("--use_cache",  action="store_true",
                   help="Load pre-cached .npy files instead of raw DICOMs.")
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--exclude_cross_disease_overlap", action="store_true")

    # Model version
    p.add_argument("--model_version",      type=str, default="v2", choices=["v1", "v2"],
                   help="v1=CheSS backbone; v2=ResNet-50 from scratch + CBAM + SE decoder")
    p.add_argument("--use_bbox_cross_attn", action="store_true",
                   help="Enable BboxCrossAttnHead (D1+). D0 uses learned query only.")
    p.add_argument("--attn_heads",          type=int, default=4,
                   help="Number of self-attention heads in ResNet-50 bottleneck (V2).")
    p.add_argument("--perceptual_only",     action="store_true",
                   help="Use CheSS as frozen perceptual loss extractor only (D4). "
                        "Does NOT inject CheSS weights into the encoder.")

    # Model dims
    p.add_argument("--z_channels_common",  type=int, default=16)
    p.add_argument("--z_channels_disease", type=int, default=16)
    p.add_argument("--attn_query_dim",     type=int, default=256)

    # CheSS weights (V1 encoder backbone, or V2 perceptual loss with --perceptual_only)
    p.add_argument("--chess_checkpoint", type=str,
                   default="/datasets/mmolefe/chess/pretrained_weights.pth.tar")
    p.add_argument("--chess_converted",  type=str, default=None)

    # Loss weights
    p.add_argument("--weight_rec",           type=float, default=1.0)
    p.add_argument("--weight_kl_common",     type=float, default=1e-4)
    p.add_argument("--weight_kl_disease",    type=float, default=5e-5)
    p.add_argument("--weight_mi_factor",     type=float, default=1.0)
    p.add_argument("--weight_bbox_attn",     type=float, default=0.0)
    p.add_argument("--weight_perceptual",    type=float, default=0.0)
    p.add_argument("--sigma_inactive",       type=float, default=0.1)
    p.add_argument("--kl_warmup_epochs",     type=int,   default=0)

    # Optimizers
    p.add_argument("--lr_vae",       type=float, default=1e-4)
    p.add_argument("--lr_backbone",  type=float, default=1e-5,
                   help="V1 only: CheSS trunk LR (lower than lr_vae).")
    p.add_argument("--lr_disc",      type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    # Training
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--seed",         type=int, default=0)

    # Logging & checkpoints
    p.add_argument("--output_root",          type=str, default="runs_sepvae")
    p.add_argument("--exp_name",             type=str, default="sepvae")
    p.add_argument("--log_every",            type=int, default=100)
    p.add_argument("--save_every",           type=int, default=1)
    p.add_argument("--sample_every",         type=int, default=5)
    p.add_argument("--n_samples_per_class",  type=int, default=4)
    p.add_argument("--manifold_every",       type=int, default=-1,
                   help="Manifold plot cadence (-1=match sample_every, 0=disable)")
    p.add_argument("--manifold_max_samples", type=int, default=600)
    p.add_argument("--manifold_method",      type=str, default="pca",
                   choices=["pca", "tsne", "both"])

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Resume
    p.add_argument("--resume", type=str, default=None)

    # W&B
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb_project",  type=str, default="baseline-sepvae")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_id",   type=str, default=None)

    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Visualisation helpers
# ============================================================================

def make_recon_grid(x_input, x_rec, labels, n_per_class=4):
    """Two-row grid: originals (top) | reconstructions (bottom), grouped by class."""
    from torchvision.utils import make_grid
    from PIL import Image

    x_01      = (np.array(x_input) + 1.0) / 2.0
    x_rec_np  = np.clip(np.array(x_rec), 0.0, 1.0)
    labels_np = np.array(labels)

    originals, reconstructions = [], []
    for cls_id in [0, 1]:
        idxs = np.where(labels_np == cls_id)[0][:n_per_class]
        for idx in idxs:
            originals.append(x_01[idx])
            reconstructions.append(x_rec_np[idx])

    all_imgs = originals + reconstructions
    imgs_np  = np.transpose(np.stack(all_imgs, axis=0), (0, 3, 1, 2))
    grid     = make_grid(torch.tensor(imgs_np).clamp(0, 1), nrow=len(originals), padding=2)
    return Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def make_attention_grid(x_input, attn_maps, labels, n_per_class=4,
                        bboxes_cardio=None):
    """
    Two-row panel: Normal | Cardiomegaly.
    Each sample shown as: CXR | cardio attention map (with GT bbox overlay).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    x_01      = (np.array(x_input) + 1.0) / 2.0
    attn_ca   = np.array(attn_maps['cardiomegaly'])
    labels_np = np.array(labels)
    B_full    = x_01.shape[0]
    B         = B_full // 2

    def _draw_bbox(ax, bbox_norm, img_h, img_w, color):
        if bbox_norm is None:
            return
        x0n, y0n, x1n, y1n = bbox_norm
        if (x1n - x0n) < 1e-4:
            return
        rect = mpatches.Rectangle(
            (x0n * img_w, y0n * img_h),
            (x1n - x0n) * img_w, (y1n - y0n) * img_h,
            linewidth=1.2, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)

    n_rows    = 2
    n_subcols = 2
    n_cols    = n_per_class * n_subcols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_per_class * n_subcols * 1.8, n_rows * 2.2))
    if n_rows == 1: axes = axes[None, :]
    if n_cols == 1: axes = axes[:, None]

    class_names = ['Normal', 'Cardiomegaly']

    for row, cls_id in enumerate([0, 1]):
        idxs = np.where(labels_np == cls_id)[0][:n_per_class]

        for col_pos, idx in enumerate(idxs):
            img = x_01[idx, :, :, 0]
            H, W = img.shape
            base = col_pos * n_subcols
            within_cls = idx - B if cls_id == 1 else None

            ax = axes[row, base]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col_pos == 0:
                ax.set_ylabel(class_names[cls_id], fontsize=8)
            if row == 0 and col_pos == 0:
                ax.set_title('CXR', fontsize=7)

            ax = axes[row, base + 1]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.imshow(attn_ca[idx], cmap='hot', alpha=0.5,
                      extent=(0, W, H, 0), interpolation='bilinear')
            if cls_id == 1 and within_cls is not None and bboxes_cardio is not None:
                if 0 <= within_cls < len(bboxes_cardio):
                    _draw_bbox(ax, bboxes_cardio[within_cls], H, W, color='lime')
            ax.axis('off')
            if row == 0 and col_pos == 0:
                ax.set_title('Cardio attn', fontsize=7)

        for col_pos in range(len(idxs), n_per_class):
            for sub in range(n_subcols):
                axes[row, col_pos * n_subcols + sub].axis('off')

    plt.tight_layout(pad=0.3)
    buf = __import__('io').BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def save_latent_manifold_plot(model, vae_params, vae_batch_stats, loader,
                               save_path, max_samples=600, method="pca",
                               use_bbox_cross_attn=False):
    """PCA/t-SNE scatter of common + cardio heads with silhouette scores."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    lc, lcard, lbls = [], [], []

    for batch_torch in loader:
        if len(lbls) >= max_samples:
            break
        for img_key, label_val in [('x_norm', 0), ('x_disease1', 1)]:
            x  = jnp.array(batch_torch[img_key].permute(0, 2, 3, 1).numpy())
            # For V2 cross-attn: pass None bbox → BboxCrossAttnHead uses fallback query
            ld = model.apply(variables, x, method=model.encode)
            remaining = max_samples - len(lbls)
            if remaining <= 0:
                break
            z_c  = np.array(jnp.mean(ld['common'][0],       axis=(1, 2)))[:remaining]
            z_ca = np.array(jnp.mean(ld['cardiomegaly'][0], axis=(1, 2)))[:remaining]
            lc.extend(z_c); lcard.extend(z_ca)
            lbls.extend([label_val] * len(z_c))

    lc    = np.array(lc)
    lcard = np.array(lcard)
    lbls  = np.array(lbls)

    if len(lbls) < 2:
        return {}

    feature_sets = {
        'all_heads':    np.concatenate([lc, lcard], axis=1),
        'disease_only': lcard,
    }
    set_titles = {
        'all_heads':    'All heads (common + cardio)',
        'disease_only': 'Cardio head only',
    }
    methods = ['pca', 'tsne'] if method == 'both' else [method]
    metrics = {}
    colors  = ['blue', 'green']
    names   = ['Normal', 'Cardiomegaly']

    fig, axes = plt.subplots(len(feature_sets), len(methods),
                             figsize=(7 * len(methods), 5 * len(feature_sets)),
                             squeeze=False)

    for row_idx, (set_name, feats) in enumerate(feature_sets.items()):
        for col_idx, m in enumerate(methods):
            ax = axes[row_idx][col_idx]
            reducer = (TSNE(n_components=2, random_state=42,
                            perplexity=min(30, max(5, len(feats) // 4)))
                       if m == 'tsne' else PCA(n_components=2, random_state=42))
            z2 = reducer.fit_transform(feats)
            for d in [0, 1]:
                mask = lbls == d
                ax.scatter(z2[mask, 0], z2[mask, 1],
                           c=colors[d], label=names[d], alpha=0.6, s=20)
            try:
                sil = float(silhouette_score(z2, lbls))
            except ValueError:
                sil = float('nan')
            metrics[f'silhouette_{set_name}_{m}'] = sil
            sil_str = f"{sil:.3f}" if np.isfinite(sil) else "N/A"
            ax.set_title(f"{set_titles[set_name]} ({m.upper()}) — sil={sil_str}")
            ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.manifold_every < 0:
        args.manifold_every = args.sample_every

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    IS_V2 = (args.model_version == 'v2')

    print("=" * 60)
    print(f"BINARY SEPVAE  model={args.model_version}  "
          f"z=[z_common({args.z_channels_common}), z_cardio({args.z_channels_disease})]")
    print("Obj 1 — orthogonality: KL + FactorVAE MI + bbox attn")
    print("Obj 2 — crisp recon:   MSE" +
          (" + CheSS perceptual" if args.weight_perceptual > 0 else ""))
    print("=" * 60)

    # ── Output dirs ───────────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_slug    = f"{args.exp_name}-{timestamp}"
    output_dir  = Path(args.output_root) / exp_slug
    ckpt_dir    = ensure_dir(output_dir / "checkpoints")
    samples_dir = ensure_dir(output_dir / "samples")
    manifold_dir = ensure_dir(output_dir / "manifold")
    diag_dir    = ensure_dir(output_dir / "diagnostics")
    print(f"Output: {output_dir}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb and _WANDB:
        wandb_kwargs = dict(project=args.wandb_project, entity=args.wandb_entity,
                            config=vars(args))
        if args.wandb_run_id:
            wandb_kwargs['id']     = args.wandb_run_id
            wandb_kwargs['resume'] = 'must'
        else:
            wandb_kwargs['name'] = exp_slug
        wandb.init(**wandb_kwargs)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VinBigDataPairDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        exclude_cross_disease_overlap=getattr(args, 'exclude_cross_disease_overlap', False),
        use_cache=args.use_cache,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=jax_pair_collate_fn,
                        drop_last=True)
    print(f"Dataset: {len(dataset)} pairs, {len(loader)} steps/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    if IS_V2:
        from models.sep_vae_v2 import SepVAEV2
        sepvae = SepVAEV2(
            z_channels_common=args.z_channels_common,
            z_channels_disease=args.z_channels_disease,
            query_dim=args.attn_query_dim,
            attn_heads=args.attn_heads,
            use_bbox_cross_attn=args.use_bbox_cross_attn,
        )
        vae_batch_stats = {}   # GroupNorm — no batch_stats
        dummy_x      = jnp.ones((1, args.img_size, args.img_size, 1))
        dummy_labels = jnp.array([0])
        rng, init_rng = jax.random.split(rng)
        vae_vars = sepvae.init(init_rng, dummy_x, dummy_labels, key=init_rng)
        vae_params = jax.tree_util.tree_map(jnp.array, vae_vars['params'])
        n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print(f"SepVAEV2 parameters: {n_vae_params:,}")
        print(f"  bbox cross-attn: {args.use_bbox_cross_attn}  "
              f"attn_heads: {args.attn_heads}  img_size: {args.img_size}")

    else:
        # ── V1: CheSS backbone ────────────────────────────────────────────────
        from models.sep_vae_jax import SepVAE
        from models.resnet_jax import ResNet50CheSS
        from utils.weight_converter import convert_chess_resnet50, load_converted_weights

        sepvae = SepVAE(
            z_channels_common=args.z_channels_common,
            z_channels_disease=args.z_channels_disease,
            attn_query_dim=args.attn_query_dim,
        )

        if args.chess_converted and os.path.exists(args.chess_converted):
            chess_params, chess_batch_stats = load_converted_weights(args.chess_converted)
        else:
            chess_params, chess_batch_stats = convert_chess_resnet50(
                args.chess_checkpoint, verbose=True
            )
            from utils.weight_converter import save_converted_weights
            converted_path = output_dir / "chess_jax_params.npy"
            save_converted_weights((chess_params, chess_batch_stats), str(converted_path))
            print(f"Saved converted weights: {converted_path}")

        chess_trunk_params = {k: v for k, v in chess_params.items()
                              if not k.startswith('layer4')}
        chess_trunk_stats  = {k: v for k, v in chess_batch_stats.items()
                              if not k.startswith('layer4')}
        chess_l4_params    = {k: v for k, v in chess_params.items()
                              if k.startswith('layer4')}
        chess_l4_stats     = {k: v for k, v in chess_batch_stats.items()
                              if k.startswith('layer4')}

        dummy_x      = jnp.ones((1, args.img_size, args.img_size, 1))
        dummy_labels = jnp.array([0])
        rng, init_rng = jax.random.split(rng)
        vae_vars = sepvae.init(
            {'params': init_rng, 'dropout': init_rng},
            dummy_x, dummy_labels, key=init_rng, train=True,
        )
        vae_params      = vae_vars['params']
        vae_batch_stats = vae_vars.get('batch_stats', {})

        vae_params = dict(vae_params)
        vae_params['encoder'] = dict(vae_params['encoder'])
        vae_params['encoder']['backbone']  = chess_trunk_params
        vae_params['encoder']['bg_branch'] = chess_l4_params
        vae_params['encoder']['tg_branch'] = chess_l4_params

        if vae_batch_stats:
            vae_batch_stats = dict(vae_batch_stats)
            vae_batch_stats['encoder'] = dict(vae_batch_stats.get('encoder', {}))
            vae_batch_stats['encoder']['backbone']  = chess_trunk_stats
            vae_batch_stats['encoder']['bg_branch'] = chess_l4_stats
            vae_batch_stats['encoder']['tg_branch'] = chess_l4_stats
        else:
            vae_batch_stats = {
                'encoder': {
                    'backbone':  chess_trunk_stats,
                    'bg_branch': chess_l4_stats,
                    'tg_branch': chess_l4_stats,
                }
            }

        vae_params      = jax.tree_util.tree_map(jnp.array, vae_params)
        vae_batch_stats = jax.tree_util.tree_map(jnp.array, vae_batch_stats)
        n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print(f"SepVAE (V1) parameters: {n_vae_params:,}")

    # ── Perceptual backbone (frozen CheSS, only loaded when needed) ───────────
    backbone_for_percep  = None
    backbone_vars_percep = None
    if args.weight_perceptual > 0.0:
        if IS_V2 and not args.perceptual_only:
            raise ValueError(
                "V2 + weight_perceptual > 0 requires --perceptual_only "
                "(CheSS is not used as V2 encoder backbone)"
            )
        from models.resnet_jax import ResNet50CheSS
        from utils.weight_converter import convert_chess_resnet50, load_converted_weights
        if args.chess_converted and os.path.exists(args.chess_converted):
            _cp, _cs = load_converted_weights(args.chess_converted)
        else:
            _cp, _cs = convert_chess_resnet50(args.chess_checkpoint, verbose=False)
        backbone_for_percep  = ResNet50CheSS()
        backbone_vars_percep = jax.tree_util.tree_map(
            jnp.array, {'params': _cp, 'batch_stats': _cs}
        )
        print(f"Perceptual loss: enabled (weight={args.weight_perceptual})")
    else:
        print("Perceptual loss: disabled")

    if args.weight_bbox_attn > 0.0:
        print(f"Bbox attention supervision: enabled (weight={args.weight_bbox_attn})")

    # ── FactorVAE discriminator ────────────────────────────────────────────────
    disc_input_dim = args.z_channels_common + args.z_channels_disease
    discriminator  = FactorDiscriminator(hidden_dim=64)
    rng, disc_rng  = jax.random.split(rng)
    disc_vars      = discriminator.init(disc_rng, jnp.ones((1, disc_input_dim)))
    disc_params    = disc_vars['params']
    n_disc_params  = sum(p.size for p in jax.tree_util.tree_leaves(disc_params))
    print(f"FactorDiscriminator parameters: {n_disc_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if IS_V2:
        # Single AdamW — all V2 params trained at the same LR (no frozen CheSS trunk)
        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=args.lr_vae, weight_decay=args.weight_decay),
        )
        print(f"VAE optimizer:  AdamW lr={args.lr_vae}  wd={args.weight_decay}")
    else:
        # V1: backbone (layers 1-3) gets a lower LR to preserve CheSS features
        from flax import traverse_util

        def _vae_label_fn(params):
            flat = traverse_util.flatten_dict(params)
            labels = {
                k: 'backbone' if k[0] == 'encoder' and len(k) > 1 and k[1] == 'backbone'
                   else 'vae'
                for k in flat
            }
            return traverse_util.unflatten_dict(labels)

        tx_vae = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.multi_transform(
                transforms={
                    'backbone': optax.adamw(learning_rate=args.lr_backbone,
                                            weight_decay=args.weight_decay),
                    'vae':      optax.adamw(learning_rate=args.lr_vae,
                                            weight_decay=args.weight_decay),
                },
                param_labels=_vae_label_fn,
            ),
        )
        print(f"VAE optimizer:  AdamW backbone_lr={args.lr_backbone}  "
              f"vae_lr={args.lr_vae}  wd={args.weight_decay}")

    tx_disc    = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(learning_rate=args.lr_disc),
    )
    vae_state  = TrainState.create(apply_fn=None, params=vae_params,  tx=tx_vae)
    disc_state = TrainState.create(apply_fn=None, params=disc_params, tx=tx_disc)
    ema_params = jax.tree_util.tree_map(jnp.array, vae_params)
    print(f"Disc optimizer: Adam  (lr={args.lr_disc})")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    if args.resume:
        with open(args.resume, 'rb') as f:
            ckpt = msgpack_restore(f.read())
        restored_vae_params = jax.tree_util.tree_map(jnp.array, ckpt['vae_params'])
        restored_vae_opt    = from_state_dict(vae_state.opt_state, ckpt['vae_opt_state'])
        vae_state = vae_state.replace(params=restored_vae_params,
                                      opt_state=restored_vae_opt,
                                      step=int(ckpt['global_step']))
        if 'disc_params' in ckpt:
            restored_disc_params = jax.tree_util.tree_map(jnp.array, ckpt['disc_params'])
            restored_disc_opt    = from_state_dict(disc_state.opt_state, ckpt['disc_opt_state'])
            disc_state = disc_state.replace(params=restored_disc_params,
                                            opt_state=restored_disc_opt)
        if vae_batch_stats and 'vae_batch_stats' in ckpt:
            vae_batch_stats = jax.tree_util.tree_map(jnp.array, ckpt['vae_batch_stats'])
        if 'ema_params' in ckpt:
            ema_params = jax.tree_util.tree_map(jnp.array, ckpt['ema_params'])
        rng         = jnp.array(ckpt['rng'])
        start_epoch = int(ckpt['epoch']) + 1
        global_step = int(ckpt['global_step'])
        print(f"Resumed from epoch {int(ckpt['epoch'])}, step {global_step}")

    # ── Loss config ───────────────────────────────────────────────────────────
    loss_cfg = SepVAELossConfig(
        weight_rec=args.weight_rec,
        weight_perceptual=args.weight_perceptual,
        weight_kl_common=args.weight_kl_common,
        weight_kl_disease=args.weight_kl_disease,
        weight_mi_factor=args.weight_mi_factor,
        weight_bbox_attn=args.weight_bbox_attn,
        sigma_inactive=args.sigma_inactive,
    )
    print(f"\nLoss weights: rec={loss_cfg.weight_rec}  "
          f"percep={loss_cfg.weight_perceptual}  "
          f"kl_c={loss_cfg.weight_kl_common}  "
          f"kl_d={loss_cfg.weight_kl_disease}  "
          f"mi_factor={loss_cfg.weight_mi_factor}  "
          f"bbox={loss_cfg.weight_bbox_attn}")

    # ── JIT'd steps ───────────────────────────────────────────────────────────
    _backbone_apply_fn = backbone_for_percep.apply if backbone_for_percep else None
    _backbone_vars     = backbone_vars_percep
    _batch_stats_arg   = vae_batch_stats if vae_batch_stats else None

    @jax.jit
    def update_ema(ema_p, params, decay):
        return jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, ema_p, params
        )

    @jax.jit
    def get_pooled_latents(vae_params_arg, x):
        """Encoder-only forward pass → spatially pooled z_c and z_ca."""
        variables = {'params': vae_params_arg}
        if _batch_stats_arg is not None:
            variables['batch_stats'] = _batch_stats_arg
        # bbox=None → BboxCrossAttnHead uses fallback learned query (safe for warm-up)
        ld = sepvae.apply(variables, x, method=sepvae.encode)
        z_c  = jnp.mean(ld['common'][0],       axis=(1, 2))
        z_ca = jnp.mean(ld['cardiomegaly'][0], axis=(1, 2))
        return z_c, z_ca

    @jax.jit
    def disc_step(disc_state_arg, z_c, z_ca, key):
        """Update discriminator to distinguish joint vs. permuted-marginal."""
        def disc_loss_fn(d_params):
            return factor_disc_loss(d_params, discriminator, z_c, z_ca, key)
        (d_loss, d_acc), grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(
            disc_state_arg.params
        )
        return disc_state_arg.apply_gradients(grads=grads), d_loss, d_acc

    @jax.jit
    def vae_step(vae_state_arg, batch, disc_params_frozen, key, kl_anneal):
        """Update VAE with all losses including FactorVAE MI (disc frozen).
        bbox_full and has_bbox are pre-assembled in the batch dict by the train loop."""
        bbox_arg     = batch.get('bbox_full')     # (2B, 4) or None
        has_bbox_arg = batch.get('has_bbox')      # (2B,)  or None

        def loss_fn(params):
            total_loss, logs, z_c, z_ca = sepvae_loss(
                sepvae, params, batch, key, loss_cfg,
                batch_stats=_batch_stats_arg,
                kl_anneal=kl_anneal,
                disc_params=disc_params_frozen,
                discriminator=discriminator,
                backbone_apply_fn=_backbone_apply_fn,
                backbone_variables=_backbone_vars,
                bbox=bbox_arg,
                has_bbox=has_bbox_arg,
            )
            return total_loss, (logs, z_c, z_ca)
        (loss, (logs, z_c, z_ca)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            vae_state_arg.params
        )
        return vae_state_arg.apply_gradients(grads=grads), logs, z_c, z_ca

    @jax.jit
    def reconstruct_and_encode(vae_params_arg, x, labels, key, bbox_full, has_bbox):
        """Full forward pass for visualisation."""
        variables = {'params': vae_params_arg}
        if _batch_stats_arg is not None:
            variables['batch_stats'] = _batch_stats_arg
        x_rec, latents_dict, _, _ = sepvae.apply(
            variables, x, labels, key=key, train=False,
            bbox=bbox_full, has_bbox=has_bbox,
        )
        return x_rec, latents_dict['attn_maps']

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING" + (f" (resuming from epoch {start_epoch})" if start_epoch > 1 else ""))
    print("=" * 60)

    # Warm up stale latents before the loop starts.
    _init_x = jnp.zeros((args.batch_size * 2, args.img_size, args.img_size, 1))
    z_c_stale, z_ca_stale = get_pooled_latents(vae_state.params, _init_x)

    for epoch in range(start_epoch, args.epochs + 1):
        kl_anneal = jnp.float32(
            min(1.0, epoch / args.kl_warmup_epochs) if args.kl_warmup_epochs > 0 else 1.0
        )
        anneal_str = (f"  [KL anneal={float(kl_anneal):.3f}]"
                      if args.kl_warmup_epochs > 0 else "")
        print(f"\nEpoch {epoch}/{args.epochs}{anneal_str}")

        epoch_logs = []

        for batch_torch in loader:
            batch = {
                'x_norm':         jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()),
                'x_disease1':     jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()),
                'disease_labels': jnp.array(batch_torch['disease_labels'].numpy()),
                'bbox_disease1':  jnp.array(batch_torch['bbox_disease1'].numpy()),
            }

            # ── Assemble (2B, 4) bbox tensor for V2 cross-attention ──────────
            # Normal images get a zero bbox (will have has_bbox=0 → fallback query)
            B_step     = batch['x_norm'].shape[0]
            bbox_cardio = batch['bbox_disease1']                          # (B, 4)
            bbox_full   = jnp.concatenate(
                [jnp.zeros_like(bbox_cardio), bbox_cardio], axis=0
            )                                                              # (2B, 4)
            has_bbox    = (
                (bbox_full[:, 2] - bbox_full[:, 0]) > 1e-4
            ).astype(jnp.float32)                                          # (2B,)
            batch['bbox_full'] = bbox_full
            batch['has_bbox']  = has_bbox

            rng, key_disc, key_vae = jax.random.split(rng, 3)

            # Step 1: discriminator update (one step stale — standard FactorVAE)
            disc_state, disc_loss, disc_acc = disc_step(
                disc_state, z_c_stale, z_ca_stale, key_disc
            )

            # Step 2: VAE update (disc frozen); returns fresh latents for next disc step
            disc_params_frozen = jax.lax.stop_gradient(disc_state.params)
            vae_state, logs, z_c_stale, z_ca_stale = vae_step(
                vae_state, batch, disc_params_frozen, key_vae, kl_anneal
            )

            if args.ema_decay > 0:
                ema_params = update_ema(ema_params, vae_state.params, args.ema_decay)

            logs = dict(logs) | {
                'loss/disc':        disc_loss,
                'metrics/disc_acc': disc_acc,
            }
            epoch_logs.append(logs)

            if global_step % args.log_every == 0:
                print(f"  Step {global_step}: "
                      f"loss={float(logs['loss/total']):.4f}  "
                      f"rec={float(logs['loss/reconstruction']):.4f}  "
                      f"kl={float(logs['loss/kl_total']):.4f}  "
                      f"mi={float(logs['loss/mi_factor']):.4f}  "
                      f"disc={float(logs['loss/disc']):.4f}  "
                      f"D_acc={float(logs['metrics/disc_acc']):.2f}  "
                      f"bbox={float(logs['loss/bbox_attn']):.4f}")
                if args.wandb and _WANDB:
                    wandb.log({k: float(v) for k, v in logs.items()} | {'epoch': epoch},
                              step=global_step)
            global_step += 1

        # Epoch summary
        avg = {k: float(np.mean([float(log[k]) for log in epoch_logs]))
               for k in epoch_logs[0]}
        print(f"  Summary: loss={avg['loss/total']:.4f}  "
              f"rec={avg['loss/reconstruction']:.4f}  "
              f"kl={avg['loss/kl_total']:.4f}  "
              f"mi={avg['loss/mi_factor']:.4f}  "
              f"disc={avg['loss/disc']:.4f}  "
              f"D_acc={avg['metrics/disc_acc']:.2f}  "
              f"bbox={avg['loss/bbox_attn']:.4f}")

        # Checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pkl"
            ckpt_data = {
                'epoch': epoch, 'global_step': global_step,
                'vae_params':      vae_state.params,
                'ema_params':      ema_params,
                'vae_batch_stats': vae_batch_stats,
                'vae_opt_state':   vae_state.opt_state,
                'disc_params':     disc_state.params,
                'disc_opt_state':  disc_state.opt_state,
                'rng': rng, 'args': vars(args),
            }
            with open(ckpt_path, 'wb') as f:
                f.write(to_bytes(ckpt_data))
            print(f"  Saved: {ckpt_path}")

        # Visualisations
        if args.sample_every > 0 and epoch % args.sample_every == 0:
            rng, vis_key = jax.random.split(rng)
            vis_params = ema_params if args.ema_decay > 0 else vae_state.params

            x_full   = jnp.concatenate([batch['x_norm'], batch['x_disease1']], axis=0)
            labels_v = batch['disease_labels']

            x_rec, attn_maps = reconstruct_and_encode(
                vis_params, x_full, labels_v, vis_key,
                batch['bbox_full'], batch['has_bbox'],
            )

            grid_path = samples_dir / f"recon_epoch{epoch:04d}.png"
            make_recon_grid(x_full, x_rec, labels_v,
                            n_per_class=args.n_samples_per_class).save(str(grid_path))
            print(f"  Saved recon grid: {grid_path}")

            attn_path = diag_dir / f"attn_maps_epoch{epoch:04d}.png"
            make_attention_grid(
                np.array(x_full),
                {k: np.array(v) for k, v in attn_maps.items()},
                np.array(labels_v),
                n_per_class=args.n_samples_per_class,
                bboxes_cardio=np.array(batch['bbox_disease1']),
            ).save(str(attn_path))
            print(f"  Saved attention maps: {attn_path}")

            if args.wandb and _WANDB:
                wandb.log({
                    "samples/recon_grid":    wandb.Image(str(grid_path)),
                    "diagnostics/attn_maps": wandb.Image(str(attn_path)),
                }, step=global_step)

        # Manifold
        if args.manifold_every > 0 and epoch % args.manifold_every == 0:
            manifold_path   = manifold_dir / f"manifold_epoch{epoch:04d}.png"
            manifold_params = ema_params if args.ema_decay > 0 else vae_state.params
            metrics = save_latent_manifold_plot(
                sepvae, manifold_params, vae_batch_stats, loader,
                manifold_path, max_samples=args.manifold_max_samples,
                method=args.manifold_method,
                use_bbox_cross_attn=args.use_bbox_cross_attn,
            )
            if metrics:
                print("  Manifold: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in metrics.items() if np.isfinite(v)
                ))
            if args.wandb and _WANDB:
                payload = {"samples/latent_manifold": wandb.Image(str(manifold_path))}
                payload.update({f"manifold/{k}": float(v)
                                 for k, v in metrics.items() if np.isfinite(v)})
                wandb.log(payload, step=global_step)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    final_path = ckpt_dir / "checkpoint_final.pkl"
    with open(final_path, 'wb') as f:
        f.write(to_bytes({
            'epoch': args.epochs, 'global_step': global_step,
            'vae_params':      vae_state.params,
            'ema_params':      ema_params,
            'vae_batch_stats': vae_batch_stats,
            'vae_opt_state':   vae_state.opt_state,
            'disc_params':     disc_state.params,
            'disc_opt_state':  disc_state.opt_state,
            'rng': rng, 'args': vars(args),
        }))
    print(f"Final checkpoint: {final_path}")

    if args.wandb and _WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
