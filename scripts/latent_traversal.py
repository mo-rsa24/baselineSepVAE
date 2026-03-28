"""
Latent traversal — SepVAE V2.

Fix z_common (and encoder skip features) for one Cardiomegaly image.
Scale z_disease from alpha=0 (anatomy-only) to alpha_max in equal steps.
Decode each scaled z_disease and lay out as a single row.

This directly tests whether z_disease encodes a continuous cardiac-size dial:
  alpha=0.0  →  Normal anatomy (enlarged heart removed)
  alpha=0.5  →  half the cardiomegaly signal
  alpha=1.0  →  original Cardiomegaly reconstruction
  alpha=1.5  →  amplified disease signal
  alpha=2.0  →  maximally amplified

Usage:
  python scripts/latent_traversal.py \\
      --checkpoint runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_epoch0120.pkl \\
      --csv_path   /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \\
      --dicom_dir  /datasets/mmolefe/vinbigdata/cache_npy \\
      --output_dir runs_sepvae/d3_gan_fix-20260325-143813/diagnostics/traversal_ep120 \\
      --n_alphas   9 \\
      --alpha_max  2.0 \\
      --n_images   4
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flax.serialization import msgpack_restore

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from datasets.VinBigData import VinBigDataPairDataset, jax_pair_collate_fn
from models.sep_vae_v2 import SepVAEV2


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading  (same as eval_counterfactual.py)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_params(target, source):
    if not isinstance(target, dict):
        return jnp.array(source)
    result = {}
    for k, v in target.items():
        if k in source:
            result[k] = (
                _merge_params(v, source[k])
                if isinstance(v, dict) and isinstance(source[k], dict)
                else jnp.array(source[k])
            )
        else:
            result[k] = v
    return result


def load_model(ckpt_path, img_size=256, z_common=16, z_disease=16,
               attn_query_dim=256, attn_heads=4, bbox_query_mix=0.7,
               decoder_res_blocks=3):
    model = SepVAEV2(
        z_channels_common=z_common,
        z_channels_disease=z_disease,
        query_dim=attn_query_dim,
        attn_heads=attn_heads,
        use_bbox_cross_attn=True,
        bbox_query_mix=bbox_query_mix,
        decoder_res_blocks=decoder_res_blocks,
    )
    dummy_x      = jnp.ones((1, img_size, img_size, 1))
    dummy_labels = jnp.array([0])
    init_rng     = jax.random.PRNGKey(0)
    fresh_vars   = model.init(init_rng, dummy_x, dummy_labels, key=init_rng)
    fresh_params = jax.tree_util.tree_map(jnp.array, fresh_vars['params'])

    with open(ckpt_path, 'rb') as f:
        ckpt = msgpack_restore(f.read())

    ckpt_params = jax.tree_util.tree_map(
        jnp.array, ckpt.get('ema_params', ckpt['vae_params']))
    params = _merge_params(fresh_params, ckpt_params)
    epoch  = int(ckpt.get('epoch', 0))
    print(f"Loaded checkpoint: epoch {epoch}  ({ckpt_path})")
    return model, params, epoch


# ─────────────────────────────────────────────────────────────────────────────
# Encode / decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_one(model, params, x, bbox, has_bbox):
    """Encode a batch, return (mu_c, mu_d, skip_feats)."""
    ld = model.apply(
        {'params': params}, x,
        bbox=bbox, has_bbox=has_bbox,
        method=model.encode,
    )
    mu_c  = ld['common'][0]         # (B, 16, 16, z_c)
    mu_d  = ld['cardiomegaly'][0]   # (B, 16, 16, z_d)
    skip  = ld.get('skip_feats')    # dict or None
    return mu_c, mu_d, skip


def decode_one(model, params, z_c, z_d, skip_feats):
    """Concatenate latents and decode, passing skip connections."""
    z = jnp.concatenate([z_c, z_d], axis=-1)
    return model.apply({'params': params}, z, skip_feats, method=model.decode)


# ─────────────────────────────────────────────────────────────────────────────
# Traversal grid
# ─────────────────────────────────────────────────────────────────────────────

def run_traversal(model, params, x_cardio, bbox_ca, has_bbox_ca,
                  alphas, output_dir, epoch, img_idx=0):
    """
    For one Cardiomegaly image (img_idx), decode at each alpha and save a row.

    Returns the path to the saved PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mu_c, mu_d, skip = encode_one(model, params, x_cardio, bbox_ca, has_bbox_ca)

    # Take a single image
    z_c    = mu_c[img_idx:img_idx+1]    # (1, 16, 16, z_c)
    z_d    = mu_d[img_idx:img_idx+1]    # (1, 16, 16, z_d)
    sf     = ({k: v[img_idx:img_idx+1] for k, v in skip.items()}
              if skip is not None else None)

    # Decode at each alpha
    frames = []
    for alpha in alphas:
        out = decode_one(model, params, z_c, alpha * z_d, sf)
        frames.append(np.clip(np.array(out)[0, :, :, 0], 0, 1))

    # Original image (input, in [-1,1]) → [0,1]
    orig = np.clip((np.array(x_cardio)[img_idx, :, :, 0] + 1.0) / 2.0, 0, 1)

    # ── Build the figure ────────────────────────────────────────────────────
    n_cols    = len(alphas) + 1   # +1 for original
    cell_size = 2.2               # inches
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * cell_size, cell_size + 0.9))

    # Original
    axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original\n(input)', fontsize=8)
    axes[0].axis('off')

    # Traversal frames
    for ci, (alpha, frame) in enumerate(zip(alphas, frames)):
        ax = axes[ci + 1]
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        label = f'α = {alpha:.2f}'
        if abs(alpha) < 1e-6:
            label += '\n(anatomy-only)'
        elif abs(alpha - 1.0) < 1e-6:
            label += '\n(original z_d)'
        ax.set_title(label, fontsize=8)
        ax.axis('off')

    fig.suptitle(
        f'z_disease traversal — Cardiomegaly image #{img_idx}\n'
        f'z_common + skip fixed; z_disease scaled by α\n'
        f'epoch {epoch}',
        fontsize=9, y=1.01,
    )
    plt.tight_layout()

    out_path = Path(output_dir) / f'latent_traversal_ep{epoch:04d}_img{img_idx}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Traversal saved → {out_path}")

    # ── Diff strip: |frame - anatomy_only| for each alpha ─────────────────
    anat_frame = frames[0]   # alpha=0 is anatomy-only (first alpha should be 0)
    raw_diffs = [np.abs(f - anat_frame) for f in frames[1:]]
    if raw_diffs:
        diff_scale = max(float(np.percentile(np.concatenate([d.ravel() for d in raw_diffs]), 99)), 1e-4)
        print(f"  Diff strip scale (p99): {diff_scale:.4f}  "
              f"max={max(d.max() for d in raw_diffs):.4f}")

        fig2, axes2 = plt.subplots(1, len(raw_diffs),
                                   figsize=(len(raw_diffs) * cell_size, cell_size + 0.9))
        if len(raw_diffs) == 1:
            axes2 = [axes2]
        for ci, (alpha, diff) in enumerate(zip(alphas[1:], raw_diffs)):
            axes2[ci].imshow(np.clip(diff / diff_scale, 0, 1), cmap='hot', vmin=0, vmax=1)
            axes2[ci].set_title(f'α={alpha:.2f}\n|Δ|', fontsize=8)
            axes2[ci].axis('off')
        fig2.suptitle(
            f'|Δ| = |decode(α·z_d) − anatomy_only|  (scale={diff_scale:.4f})\n'
            f'epoch {epoch}',
            fontsize=9, y=1.01,
        )
        plt.tight_layout()
        diff_path = Path(output_dir) / f'traversal_diff_ep{epoch:04d}_img{img_idx}.png'
        plt.savefig(str(diff_path), dpi=160, bbox_inches='tight')
        plt.close(fig2)
        print(f"Diff strip saved  → {diff_path}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("z_disease latent traversal")
    p.add_argument('--checkpoint',        required=True)
    p.add_argument('--csv_path',          required=True)
    p.add_argument('--dicom_dir',         required=True)
    p.add_argument('--output_dir',        required=True)
    p.add_argument('--n_images',          type=int,   default=1,
                   help='Number of Cardiomegaly images to traverse.')
    p.add_argument('--n_alphas',          type=int,   default=9,
                   help='Number of alpha steps including 0 and alpha_max.')
    p.add_argument('--alpha_min',         type=float, default=0.0)
    p.add_argument('--alpha_max',         type=float, default=2.0)
    p.add_argument('--img_size',          type=int,   default=256)
    p.add_argument('--z_common',          type=int,   default=16)
    p.add_argument('--z_disease',         type=int,   default=16)
    p.add_argument('--attn_query_dim',    type=int,   default=256)
    p.add_argument('--attn_heads',        type=int,   default=4)
    p.add_argument('--bbox_query_mix',    type=float, default=0.7)
    p.add_argument('--decoder_res_blocks',type=int,   default=3)
    p.add_argument('--seed',              type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, params, epoch = load_model(
        args.checkpoint,
        img_size=args.img_size,
        z_common=args.z_common,
        z_disease=args.z_disease,
        attn_query_dim=args.attn_query_dim,
        attn_heads=args.attn_heads,
        bbox_query_mix=args.bbox_query_mix,
        decoder_res_blocks=args.decoder_res_blocks,
    )

    alphas = list(np.linspace(args.alpha_min, args.alpha_max, args.n_alphas))

    dataset = VinBigDataPairDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        use_cache=True,
        deterministic_pairs=True,
        pair_seed=args.seed,
    )

    rng     = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=args.n_images, replace=False)).tolist()
    subset  = Subset(dataset, indices)
    loader  = DataLoader(
        subset,
        batch_size=args.n_images,
        shuffle=False,
        collate_fn=jax_pair_collate_fn,
        num_workers=0,
        drop_last=False,
    )

    batch    = next(iter(loader))
    x_cardio = jnp.array(batch['x_disease1'].permute(0, 2, 3, 1).numpy())
    bbox_ca  = jnp.array(batch['bbox_disease1'].numpy())
    has_bbox = ((bbox_ca[:, 2] - bbox_ca[:, 0]) > 1e-4).astype(jnp.float32)

    print(f"Traversal alphas: {[f'{a:.2f}' for a in alphas]}")

    for img_idx in range(min(args.n_images, x_cardio.shape[0])):
        run_traversal(
            model, params,
            x_cardio, bbox_ca, has_bbox,
            alphas,
            output_dir=args.output_dir,
            epoch=epoch,
            img_idx=img_idx,
        )

    print("Done.")


if __name__ == '__main__':
    main()
