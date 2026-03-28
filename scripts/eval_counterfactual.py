"""
Counterfactual reconstruction diagnostic — VAE-only, no LDM required.

Implements the "Acid Test" from idea2.md Phase 3:

  Test A — anatomy-only (zero z_cardio):
    Encode Cardiomegaly images → zero z_cardio → decode.
    Success: output looks Normal (cardiac silhouette shrinks to anatomy baseline).
    Failure: heart remains enlarged → z_common is carrying cardiac shape.

  Test B — disease injection (inject mean z_cardio into Normal):
    Compute mean posterior mean of z_cardio across a Cardiomegaly population.
    Encode Normal images → replace z_cardio with the Cardio population mean → decode.
    Success: output develops an enlarged cardiac silhouette without anatomy changes.
    Failure: lung fields shift, ribs move → z_cardio is entangled with anatomy.

Output:
  counterfactual_grid.png  — rows: [Normal | Cardio], cols: [Original | Anatomy-only | Injected]
  counterfactual_stats.json — outside-bbox MSE ratios and norm statistics

Usage:
  python scripts/eval_counterfactual.py \\
      --checkpoint runs_sepvae/d5_recon-XXXXXXXX/checkpoints/checkpoint_epoch0060.pkl \\
      --csv_path   /workspace/vinbigdata/cache_npy/train_filtered.csv \\
      --dicom_dir  /workspace/vinbigdata/cache_npy \\
      --output_dir runs_sepvae/d5_recon-XXXXXXXX/diagnostics/counterfactual_ep060 \\
      --n_samples  8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flax.serialization import msgpack_restore

# Make repo root importable when called from any working directory
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from datasets.VinBigData import VinBigDataPairDataset, jax_pair_collate_fn
from models.sep_vae_v2 import SepVAEV2


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (mirrors train_sep_vae.py _merge_params logic)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_params(target, source):
    """Recursively merge checkpoint params into freshly-initialised target."""
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


def load_model_from_checkpoint(ckpt_path: str, img_size: int = 256,
                                z_common: int = 16, z_disease: int = 16,
                                attn_query_dim: int = 256, attn_heads: int = 4,
                                bbox_query_mix: float = 0.7,
                                decoder_res_blocks: int = 3):
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

    ckpt_params = jax.tree_util.tree_map(jnp.array, ckpt.get('ema_params', ckpt['vae_params']))
    params      = _merge_params(fresh_params, ckpt_params)

    epoch = int(ckpt.get('epoch', 0))
    print(f"Loaded checkpoint: epoch {epoch}  ({ckpt_path})")
    return model, params, epoch


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_batch(model, params, x, bbox=None, has_bbox=None):
    """Return latents_dict (posterior means and logvars) for a batch."""
    variables = {'params': params}
    kwargs = {}
    if bbox is not None:
        kwargs['bbox']     = bbox
        kwargs['has_bbox'] = has_bbox
    return model.apply(variables, x, method=model.encode, **kwargs)


def decode_z(model, params, z_common_map, z_disease_map, skip_feats=None):
    """Run decoder with the given spatial latent maps.

    skip_feats: dict with 'layer3' (B,16,16,1024) and 'layer2' (B,32,32,512)
    from the encoder forward pass.  Must be passed for sharp reconstructions —
    the decoder was trained with these skip connections every step.
    """
    z_concat = jnp.concatenate([z_common_map, z_disease_map], axis=-1)
    return model.apply({'params': params}, z_concat, skip_feats, method=model.decode)


def _concat_skip_feats(list_of_skip_feats):
    """Concatenate a list of skip_feats dicts along the batch axis."""
    if not list_of_skip_feats or list_of_skip_feats[0] is None:
        return None
    keys = list_of_skip_feats[0].keys()
    return {k: np.concatenate([sf[k] for sf in list_of_skip_feats], axis=0)
            for k in keys}


def _slice_skip(skip_feats, idx):
    """Slice skip_feats dict to index range idx (slice or int array)."""
    if skip_feats is None:
        return None
    return {k: jnp.array(v[idx]) for k, v in skip_feats.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Grid visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8(img_01):
    return (np.clip(img_01, 0, 1) * 255).astype(np.uint8)


def save_counterfactual_grid(
    x_normal, x_cardio,
    recon_normal, recon_cardio,
    anatomy_normal, anatomy_cardio,
    injected_normal, injected_cardio,
    bboxes_cardio,
    save_path: Path,
    n: int = 8,
):
    """
    One row per X-ray.  Columns show what changes as we modify the disease latent.

    Normal images (top section) — 5 columns:
      Original | Reconstruction | Anatomy-only (z_d=0) | +Disease injected | |Δ| diff map

    Cardiomegaly images (bottom section) — 5 columns:
      Original (+bbox) | Reconstruction | Anatomy-only (z_d=0) | Own disease re-injected | |Δ| diff map

    The |Δ| column is abs(disease_version - anatomy_only), hot colormap, showing
    exactly which pixels changed when the disease latent was active.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize as MplNorm
    from matplotlib.cm import ScalarMappable

    def _prep(arr, is_input=False):
        """(B, H, W, 1) → list of (H, W) float32 in [0,1].

        Dataset images are in [-1, 1] (x * 2 - 1 in _preprocess_image).
        Decoder output (sigmoid) is already in [0, 1].
        Matching make_recon_grid in train_sep_vae.py:
            inputs:  (x + 1) / 2   — same as training W&B logger
            outputs: clip(x, 0, 1) — sigmoid output, already [0,1]
        """
        a = np.array(arr)
        if is_input:
            a = (a + 1.0) / 2.0   # [-1,1] → [0,1]
        return [np.clip(a[i, :, :, 0], 0, 1) for i in range(len(a))]

    B = min(n, x_normal.shape[0])
    bboxes_np = np.array(bboxes_cardio)

    # Raw dataset images are in [-1,1]; decoder outputs are in [0,1]
    orig_n    = _prep(x_normal[:B],       is_input=True)
    orig_c    = _prep(x_cardio[:B],       is_input=True)
    recon_n   = _prep(recon_normal[:B])
    recon_c   = _prep(recon_cardio[:B])
    anat_n    = _prep(anatomy_normal[:B])
    anat_c    = _prep(anatomy_cardio[:B])
    inj_n     = _prep(injected_normal[:B])
    inj_c     = _prep(injected_cardio[:B])

    # Diff maps: abs(disease version - anatomy-only).
    # Scale adaptively to the 99th percentile of the pooled differences so weak
    # disease signal is still visible.  Print the actual scale so the user knows
    # whether the differences are meaningful or noise-level.
    raw_diff_n = [np.abs(inj_n[i] - anat_n[i]) for i in range(B)]
    raw_diff_c = [np.abs(inj_c[i] - anat_c[i]) for i in range(B)]
    all_raw = np.concatenate([d.ravel() for d in raw_diff_n + raw_diff_c])
    diff_scale = max(float(np.percentile(all_raw, 99)), 1e-4)
    print(f"  Diff map scale (p99): {diff_scale:.4f}  "
          f"max_n={max(d.max() for d in raw_diff_n):.4f}  "
          f"max_c={max(d.max() for d in raw_diff_c):.4f}")
    diff_n = [np.clip(raw_diff_n[i] / diff_scale, 0, 1) for i in range(B)]
    diff_c = [np.clip(raw_diff_c[i] / diff_scale, 0, 1) for i in range(B)]

    N_COLS = 5
    col_labels = [
        'Original',
        'Reconstruction\n(z_c + z_d → decode)',
        'Anatomy-only\n(z_disease = 0)',
        'Disease active\n(Normal: +mean z_cardio\nCardio: own z_disease)',
        f'|Δ| diff\n(disease − anatomy)\nscale={diff_scale:.3f}',
    ]

    # Total rows: B Normal + 1 spacer + B Cardio
    n_rows = 2 * B + 1
    cell_h = 2.0   # inches per row
    cell_w = 2.0   # inches per column
    label_w = 1.2  # left label column width

    fig = plt.figure(figsize=(N_COLS * cell_w + label_w, n_rows * cell_h))

    # GridSpec: rows = all image rows + spacer; cols = label + 5 image cols
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        n_rows, N_COLS + 1,
        figure=fig,
        hspace=0.04,
        wspace=0.03,
        width_ratios=[label_w / cell_w] + [1.0] * N_COLS,
    )

    # Section header font
    header_kw  = dict(fontsize=10, fontweight='bold', va='center', ha='right')
    row_lbl_kw = dict(fontsize=8,  va='center', ha='right', color='#333333')

    def _draw_section_label(row, text, color):
        ax = fig.add_subplot(gs[row, 0])
        ax.text(0.92, 0.5, text, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='center', ha='right', color=color)
        ax.axis('off')

    def _img_ax(row, col):
        return fig.add_subplot(gs[row, col + 1])  # +1 for label column

    # ── Column titles (above first row) ─────────────────────────────────────────
    for ci, label in enumerate(col_labels):
        ax = _img_ax(0, ci)
        ax.set_title(label, fontsize=8, pad=4, linespacing=1.3)

    # ── Normal rows ─────────────────────────────────────────────────────────────
    for i in range(B):
        row = i
        imgs = [orig_n[i], recon_n[i], anat_n[i], inj_n[i], diff_n[i]]
        cmaps = ['gray', 'gray', 'gray', 'gray', 'hot']

        for ci, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = _img_ax(row, ci)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
            ax.axis('off')

        # Row label
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.text(0.92, 0.5, f'Normal #{i+1}', transform=ax_lbl.transAxes,
                    **row_lbl_kw)
        ax_lbl.axis('off')

    # ── Spacer row ───────────────────────────────────────────────────────────────
    spacer_row = B
    for ci in range(N_COLS + 1):
        ax = fig.add_subplot(gs[spacer_row, ci])
        ax.set_facecolor('#dddddd')
        ax.axis('off')
        if ci == 0:
            ax.text(0.5, 0.5, '─── Cardiomegaly ───', transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='center', ha='center',
                    color='#222222')

    # ── Cardiomegaly rows ────────────────────────────────────────────────────────
    for i in range(B):
        row = B + 1 + i
        imgs  = [orig_c[i], recon_c[i], anat_c[i], inj_c[i], diff_c[i]]
        cmaps = ['gray', 'gray', 'gray', 'gray', 'hot']

        for ci, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = _img_ax(row, ci)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')

            # GT bbox on Original only
            if ci == 0 and i < len(bboxes_np):
                bx = bboxes_np[i]
                if bx[2] - bx[0] > 1e-4:
                    H_im, W_im = img.shape
                    rect = mpatches.Rectangle(
                        (bx[0] * W_im, bx[1] * H_im),
                        (bx[2] - bx[0]) * W_im, (bx[3] - bx[1]) * H_im,
                        linewidth=1.2, edgecolor='#00ff88', facecolor='none',
                    )
                    ax.add_patch(rect)
            ax.axis('off')

        # Row label
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.text(0.92, 0.5, f'Cardio #{i+1}', transform=ax_lbl.transAxes,
                    **row_lbl_kw)
        ax_lbl.axis('off')

    # ── Mean diff maps ───────────────────────────────────────────────────────────
    # Save separately — one per class
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for tag, diffs in [('normal', diff_n), ('cardio', diff_c)]:
        mean_diff = np.mean(np.stack(diffs, axis=0), axis=0)
        fig_d, ax_d = plt.subplots(1, 1, figsize=(3.5, 3.5))
        im = ax_d.imshow(mean_diff, cmap='hot', vmin=0, vmax=1)
        ax_d.set_title(f'Mean |Δ| — {tag} (n={B})\nwhere disease latent changed the image\n'
                       f'(scale = p99 = {diff_scale:.4f})', fontsize=9)
        ax_d.axis('off')
        plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04).set_label(
            f'|Δ| / {diff_scale:.4f}  (1.0 = {diff_scale:.4f} intensity)', fontsize=8)
        diff_path = save_path.parent / f'mean_diff_map_{tag}_ep{save_path.stem.split("ep")[-1]}.png'
        plt.savefig(str(diff_path), dpi=150, bbox_inches='tight')
        plt.close(fig_d)
        print(f"Mean diff map saved → {diff_path}")

    fig.suptitle(
        'Counterfactual Acid Test — one row per image\n'
        'Columns: Original → Reconstruction → Anatomy-only → Disease active → |Δ| where image changed',
        fontsize=10, y=1.002,
    )
    plt.savefig(str(save_path), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Importable entry point (called from train_sep_vae.py during training)
# ─────────────────────────────────────────────────────────────────────────────

def run_counterfactual_eval(
    model,
    params,
    batch_stats,        # unused for V2 (GroupNorm) — kept for API consistency
    eval_loader,
    output_dir,
    epoch: int,
    global_step: int,
    n_samples: int = 8,
    n_inject_src: int = 64,
    use_wandb: bool = False,
):
    """
    Run the counterfactual acid test using an already-loaded eval_loader.

    Runs Test A (anatomy-only: z_cardio=0) and Test B (disease injection:
    inject mean z_cardio from Cardiomegaly population into Normal images).
    Saves a grid PNG and optionally logs it to W&B.

    Returns:
        Path to saved grid PNG, or None if the loader produced no samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_x_normal, all_x_cardio           = [], []
    all_mu_c_normal, all_mu_d_normal      = [], []
    all_mu_c_cardio, all_mu_d_cardio      = [], []
    all_skip_normal, all_skip_cardio      = [], []
    all_bbox_cardio                        = []
    n_need = max(n_samples, n_inject_src)
    n_collected = 0

    for batch in eval_loader:
        if n_collected >= n_need:
            break
        x_norm   = jnp.array(batch["x_norm"].permute(0, 2, 3, 1).numpy())
        x_cardio = jnp.array(batch["x_disease1"].permute(0, 2, 3, 1).numpy())
        bbox_ca  = jnp.array(batch["bbox_disease1"].numpy())

        has_bbox_ca = ((bbox_ca[:, 2] - bbox_ca[:, 0]) > 1e-4).astype(jnp.float32)
        bbox_zero   = jnp.zeros_like(bbox_ca)
        has_zero    = jnp.zeros(x_norm.shape[0], dtype=jnp.float32)

        ld_norm   = encode_batch(model, params, x_norm,   bbox=bbox_zero, has_bbox=has_zero)
        ld_cardio = encode_batch(model, params, x_cardio, bbox=bbox_ca,   has_bbox=has_bbox_ca)

        all_x_normal.append(np.array(x_norm))
        all_x_cardio.append(np.array(x_cardio))
        all_mu_c_normal.append(np.array(ld_norm["common"][0]))
        all_mu_d_normal.append(np.array(ld_norm["cardiomegaly"][0]))
        all_mu_c_cardio.append(np.array(ld_cardio["common"][0]))
        all_mu_d_cardio.append(np.array(ld_cardio["cardiomegaly"][0]))
        all_bbox_cardio.append(np.array(bbox_ca))
        sf_n = ld_norm.get('skip_feats')
        sf_c = ld_cardio.get('skip_feats')
        if sf_n is not None:
            all_skip_normal.append({k: np.array(v) for k, v in sf_n.items()})
        if sf_c is not None:
            all_skip_cardio.append({k: np.array(v) for k, v in sf_c.items()})
        n_collected += int(x_norm.shape[0])

    if n_collected == 0:
        return None

    x_normal_all  = jnp.array(np.concatenate(all_x_normal,    axis=0))
    x_cardio_all  = jnp.array(np.concatenate(all_x_cardio,    axis=0))
    mu_c_normal   = jnp.array(np.concatenate(all_mu_c_normal,  axis=0))
    mu_d_normal   = jnp.array(np.concatenate(all_mu_d_normal,  axis=0))
    mu_c_cardio   = jnp.array(np.concatenate(all_mu_c_cardio,  axis=0))
    mu_d_cardio   = jnp.array(np.concatenate(all_mu_d_cardio,  axis=0))
    bboxes_cardio = jnp.array(np.concatenate(all_bbox_cardio,   axis=0))
    skip_normal_all = _concat_skip_feats(all_skip_normal)
    skip_cardio_all = _concat_skip_feats(all_skip_cardio)

    N_vis = min(n_samples, int(x_normal_all.shape[0]))
    n_src = min(n_inject_src, int(mu_d_cardio.shape[0]))

    skip_n = _slice_skip(skip_normal_all, slice(N_vis))
    skip_c = _slice_skip(skip_cardio_all, slice(N_vis))

    mean_z_cardio     = jnp.mean(mu_d_cardio[:n_src], axis=0, keepdims=True)
    injected_z_cardio = jnp.broadcast_to(mean_z_cardio, mu_d_normal[:N_vis].shape)
    zeros_d_normal    = jnp.zeros_like(mu_d_normal[:N_vis])
    zeros_d_cardio    = jnp.zeros_like(mu_d_cardio[:N_vis])
    rotated_z_cardio  = jnp.roll(mu_d_cardio[:N_vis], shift=1, axis=0)

    recon_normal    = decode_z(model, params, mu_c_normal[:N_vis], mu_d_normal[:N_vis], skip_n)
    recon_cardio    = decode_z(model, params, mu_c_cardio[:N_vis], mu_d_cardio[:N_vis], skip_c)
    anatomy_normal  = decode_z(model, params, mu_c_normal[:N_vis], zeros_d_normal,      skip_n)
    anatomy_cardio  = decode_z(model, params, mu_c_cardio[:N_vis], zeros_d_cardio,      skip_c)
    injected_normal = decode_z(model, params, mu_c_normal[:N_vis], injected_z_cardio,   skip_n)
    injected_cardio = decode_z(model, params, mu_c_cardio[:N_vis], rotated_z_cardio,    skip_c)

    grid_path = output_dir / f"counterfactual_grid_ep{epoch:04d}.png"
    save_counterfactual_grid(
        x_normal_all[:N_vis], x_cardio_all[:N_vis],
        recon_normal, recon_cardio,
        anatomy_normal, anatomy_cardio,
        injected_normal, injected_cardio,
        bboxes_cardio[:N_vis],
        save_path=grid_path,
        n=N_vis,
    )
    print(f"  Counterfactual grid → {grid_path}")

    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.log(
                {"diagnostics/counterfactual": _wandb.Image(str(grid_path))},
                step=global_step,
            )
        except Exception:
            pass

    return grid_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Counterfactual reconstruction acid test")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to .pkl checkpoint (msgpack).")
    p.add_argument("--csv_path",       required=True)
    p.add_argument("--dicom_dir",      required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--n_samples",      type=int, default=8,
                   help="Number of Normal/Cardio pairs to visualise.")
    p.add_argument("--n_inject_src",   type=int, default=64,
                   help="Number of Cardio images used to compute mean z_cardio for injection.")
    p.add_argument("--img_size",       type=int, default=256)
    p.add_argument("--z_common",       type=int, default=16)
    p.add_argument("--z_disease",      type=int, default=16)
    p.add_argument("--attn_query_dim", type=int, default=256)
    p.add_argument("--attn_heads",     type=int, default=4)
    p.add_argument("--bbox_query_mix",      type=float, default=0.7)
    p.add_argument("--decoder_res_blocks",  type=int,   default=3)
    p.add_argument("--seed",                type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, params, epoch = load_model_from_checkpoint(
        args.checkpoint,
        img_size=args.img_size,
        z_common=args.z_common,
        z_disease=args.z_disease,
        attn_query_dim=args.attn_query_dim,
        attn_heads=args.attn_heads,
        bbox_query_mix=args.bbox_query_mix,
        decoder_res_blocks=args.decoder_res_blocks,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VinBigDataPairDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        use_cache=True,
        deterministic_pairs=True,
        pair_seed=args.seed,
    )

    n_total   = min(max(args.n_samples, args.n_inject_src), len(dataset))
    rng       = np.random.default_rng(args.seed)
    indices   = np.sort(rng.choice(len(dataset), size=n_total, replace=False)).tolist()
    subset    = Subset(dataset, indices)
    loader    = DataLoader(
        subset,
        batch_size=min(16, n_total),
        shuffle=False,
        collate_fn=jax_pair_collate_fn,
        num_workers=0,
        drop_last=False,
    )

    # ── Encode all samples ────────────────────────────────────────────────────
    all_x_normal, all_x_cardio     = [], []
    all_mu_c_normal, all_mu_d_normal = [], []
    all_mu_c_cardio, all_mu_d_cardio = [], []
    all_skip_normal, all_skip_cardio  = [], []
    all_bbox_cardio                  = []

    print("Encoding dataset subset …")
    for batch in loader:
        x_norm   = jnp.array(batch['x_norm'].permute(0, 2, 3, 1).numpy())
        x_cardio = jnp.array(batch['x_disease1'].permute(0, 2, 3, 1).numpy())
        bbox_ca  = jnp.array(batch['bbox_disease1'].numpy())

        has_bbox_ca = ((bbox_ca[:, 2] - bbox_ca[:, 0]) > 1e-4).astype(jnp.float32)
        bbox_zero   = jnp.zeros_like(bbox_ca)
        has_zero    = jnp.zeros(x_norm.shape[0], dtype=jnp.float32)

        ld_norm   = encode_batch(model, params, x_norm,   bbox=bbox_zero, has_bbox=has_zero)
        ld_cardio = encode_batch(model, params, x_cardio, bbox=bbox_ca,   has_bbox=has_bbox_ca)

        all_x_normal.append(np.array(x_norm))
        all_x_cardio.append(np.array(x_cardio))
        all_mu_c_normal.append(np.array(ld_norm['common'][0]))
        all_mu_d_normal.append(np.array(ld_norm['cardiomegaly'][0]))
        all_mu_c_cardio.append(np.array(ld_cardio['common'][0]))
        all_mu_d_cardio.append(np.array(ld_cardio['cardiomegaly'][0]))
        all_bbox_cardio.append(np.array(bbox_ca))
        sf_n = ld_norm.get('skip_feats')
        sf_c = ld_cardio.get('skip_feats')
        if sf_n is not None:
            all_skip_normal.append({k: np.array(v) for k, v in sf_n.items()})
        if sf_c is not None:
            all_skip_cardio.append({k: np.array(v) for k, v in sf_c.items()})

    x_normal_all   = jnp.array(np.concatenate(all_x_normal,    axis=0))
    x_cardio_all   = jnp.array(np.concatenate(all_x_cardio,    axis=0))
    mu_c_normal    = jnp.array(np.concatenate(all_mu_c_normal,  axis=0))
    mu_d_normal    = jnp.array(np.concatenate(all_mu_d_normal,  axis=0))
    mu_c_cardio    = jnp.array(np.concatenate(all_mu_c_cardio,  axis=0))
    mu_d_cardio    = jnp.array(np.concatenate(all_mu_d_cardio,  axis=0))
    bboxes_cardio  = jnp.array(np.concatenate(all_bbox_cardio,  axis=0))
    skip_normal_all = _concat_skip_feats(all_skip_normal)
    skip_cardio_all = _concat_skip_feats(all_skip_cardio)

    N_vis = min(args.n_samples, x_normal_all.shape[0])

    # ── Compute mean z_cardio from Cardiomegaly population (for injection) ────
    # Use the first n_inject_src samples as the source population
    n_src         = min(args.n_inject_src, mu_d_cardio.shape[0])
    mean_z_cardio = jnp.mean(mu_d_cardio[:n_src], axis=0, keepdims=True)  # (1, H_lat, W_lat, z_d)
    print(f"Mean z_cardio computed from {n_src} Cardiomegaly samples. "
          f"Norm: {float(jnp.linalg.norm(jnp.mean(mean_z_cardio, axis=(1, 2)))):.3f}")

    skip_n = _slice_skip(skip_normal_all, slice(N_vis))
    skip_c = _slice_skip(skip_cardio_all, slice(N_vis))

    # ── Test A: anatomy-only (z_cardio = 0) ───────────────────────────────────
    print("Running Test A (anatomy-only) …")
    zeros_d_normal = jnp.zeros_like(mu_d_normal[:N_vis])
    zeros_d_cardio = jnp.zeros_like(mu_d_cardio[:N_vis])

    anatomy_normal = decode_z(model, params, mu_c_normal[:N_vis], zeros_d_normal, skip_n)
    anatomy_cardio = decode_z(model, params, mu_c_cardio[:N_vis], zeros_d_cardio, skip_c)

    # ── Test B: disease injection (inject mean z_cardio into Normal) ──────────
    print("Running Test B (disease injection) …")
    injected_z_cardio = jnp.broadcast_to(mean_z_cardio, mu_d_normal[:N_vis].shape)

    injected_normal = decode_z(model, params, mu_c_normal[:N_vis], injected_z_cardio, skip_n)
    # For Cardio: inject z_cardio from a different Cardio image (rotate by 1) as a sanity check
    rotated_z_cardio = jnp.roll(mu_d_cardio[:N_vis], shift=1, axis=0)
    injected_cardio  = decode_z(model, params, mu_c_cardio[:N_vis], rotated_z_cardio, skip_c)

    # Reconstruct originals for reference
    recon_normal = decode_z(model, params, mu_c_normal[:N_vis], mu_d_normal[:N_vis], skip_n)
    recon_cardio = decode_z(model, params, mu_c_cardio[:N_vis], mu_d_cardio[:N_vis], skip_c)

    # ── Quantitative statistics ───────────────────────────────────────────────
    x_norm_01   = (x_normal_all[:N_vis] + 1.0) / 2.0
    x_cardio_01 = (x_cardio_all[:N_vis] + 1.0) / 2.0

    mse_recon_normal   = float(jnp.mean(jnp.square(x_norm_01   - recon_normal)))
    mse_recon_cardio   = float(jnp.mean(jnp.square(x_cardio_01 - recon_cardio)))
    mse_anatomy_normal = float(jnp.mean(jnp.square(x_norm_01   - anatomy_normal)))
    mse_anatomy_cardio = float(jnp.mean(jnp.square(x_cardio_01 - anatomy_cardio)))

    # Outside-bbox MSE for Cardio anatomy-only: low → z_common owns outside-bbox region
    H, W = x_cardio_01.shape[1], x_cardio_01.shape[2]
    y_lin = (jnp.arange(H, dtype=jnp.float32) + 0.5) / H
    x_lin = (jnp.arange(W, dtype=jnp.float32) + 0.5) / W
    bboxes_vis = bboxes_cardio[:N_vis]
    x0 = bboxes_vis[:, 0][:, None, None]
    y0 = bboxes_vis[:, 1][:, None, None]
    x1 = bboxes_vis[:, 2][:, None, None]
    y1 = bboxes_vis[:, 3][:, None, None]
    inside  = ((x_lin[None, None, :] >= x0) & (x_lin[None, None, :] <= x1) &
               (y_lin[None, :, None] >= y0) & (y_lin[None, :, None] <= y1)).astype(jnp.float32)
    outside = 1.0 - inside
    sq_anatomy_outside = jnp.square(x_cardio_01[:, :, :, 0] - anatomy_cardio[:, :, :, 0]) * outside
    mse_anatomy_outside_bbox = float(jnp.mean(sq_anatomy_outside))

    # z_cardio norm ratio: Cardio / Normal (should be >> 1 if disentangled)
    norm_d_normal = float(jnp.mean(jnp.linalg.norm(jnp.mean(mu_d_normal, axis=(1, 2)), axis=-1)))
    norm_d_cardio = float(jnp.mean(jnp.linalg.norm(jnp.mean(mu_d_cardio, axis=-1), axis=(1, 2)) if mu_d_cardio.ndim == 4
                                   else jnp.linalg.norm(mu_d_cardio, axis=-1)))
    norm_c_normal = float(jnp.mean(jnp.linalg.norm(jnp.mean(mu_c_normal, axis=(1, 2)), axis=-1)))
    norm_c_cardio = float(jnp.mean(jnp.linalg.norm(jnp.mean(mu_c_cardio, axis=(1, 2)), axis=-1)))

    stats = {
        'checkpoint':               args.checkpoint,
        'epoch':                    epoch,
        'n_vis':                    N_vis,
        'n_inject_src':             n_src,
        'mse_recon_normal':         mse_recon_normal,
        'mse_recon_cardio':         mse_recon_cardio,
        'mse_anatomy_normal':       mse_anatomy_normal,
        'mse_anatomy_cardio':       mse_anatomy_cardio,
        'mse_anatomy_outside_bbox': mse_anatomy_outside_bbox,
        'z_cardio_norm_normal':     norm_d_normal,
        'z_cardio_norm_cardio':     norm_d_cardio,
        'z_cardio_norm_ratio':      norm_d_cardio / max(norm_d_normal, 1e-6),
        'z_common_norm_normal':     norm_c_normal,
        'z_common_norm_cardio':     norm_c_cardio,
        'z_common_norm_ratio':      norm_c_cardio / max(norm_c_normal, 1e-6),
        'notes': {
            'mse_anatomy_outside_bbox': (
                'MSE of z_common-only recon outside heart bbox for Cardio images. '
                'Should be close to mse_recon_cardio (not much worse) if z_common '
                'owns the outside-bbox region.'
            ),
            'z_common_norm_ratio': (
                'Should be near 1.0 if z_common is disease-invariant.'
            ),
            'z_cardio_norm_ratio': (
                'Should be >> 1.0 (e.g. >= 3.0) if z_cardio is disease-specific.'
            ),
        },
    }

    stats_path = output_dir / f'counterfactual_stats_ep{epoch:04d}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved → {stats_path}")
    print(f"  z_cardio_norm_ratio : {stats['z_cardio_norm_ratio']:.3f}  (want >> 1)")
    print(f"  z_common_norm_ratio : {stats['z_common_norm_ratio']:.3f}  (want ~ 1)")
    print(f"  mse_anatomy_outside : {mse_anatomy_outside_bbox:.5f}  "
          f"vs mse_recon_cardio={mse_recon_cardio:.5f}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    grid_path = output_dir / f'counterfactual_grid_ep{epoch:04d}.png'
    save_counterfactual_grid(
        x_normal_all[:N_vis], x_cardio_all[:N_vis],
        recon_normal, recon_cardio,
        anatomy_normal, anatomy_cardio,
        injected_normal, injected_cardio,
        bboxes_cardio[:N_vis],
        save_path=grid_path,
        n=N_vis,
    )

    print("Done.")


if __name__ == '__main__':
    main()
