"""
7-panel visual scaffolding for the mask supervision research claim.

Builds a single wide composite strip that reads left-to-right as a
self-contained proof of disentanglement, without requiring the reader
to understand silhouette scores or latent geometry.

Panel layout (left → right):
  00  Real cardiomegaly CXR
  01  Same + GT bbox (green)
  02  Same + GT segmentation mask (CheXmask heart, cyan)
  03  Reconstruction  (z_c + z_d)
  04  Anatomy-only    (z_c alone, z_d = 0)
  05  Cardiac-only    (z_d alone, z_c = 0, no skip)  ← should converge toward col 02
  06  Pixel diff ×5   |col03 − col04|                 ← emergent cardiac segmentation

Column 05 rationale (heart_in_zd mode):
  z_d is encoded from cardiac pixels only (complement masking).  Decoding
  z_d alone (z_c=0, no skip connections) reveals exactly what the cardiac
  code has captured — it should converge to look like the GT mask in col 02.
  This is the direct visualisation of the LDM composition prerequisite:
  decode(z_d) spatially bounded to the cardiac silhouette.

Column 06 rationale:
  |full recon − anatomy-only| amplified ×5.  When z_d is properly
  disentangled, this diff equals the cardiac contribution and should
  match col 02.  Used as the emergent segmentation quality metric.

Exported function (called from train_sep_vae.py every manifold_every epochs):
    run_scaffolding(model, params, batch_stats, eval_loader,
                    epoch, global_step, output_dir,
                    use_wandb=False, use_bbox_cross_attn=True,
                    has_chexmask=False, heart_in_zd=False) → Path or None

Logged to W&B as "diagnostics/scaffolding".
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prep_img(arr):
    """(H, W, 1) or (H, W) JAX/numpy in [0,1] → (H, W) float32 numpy."""
    a = np.array(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[:, :, 0]
    return np.clip(a, 0.0, 1.0)


def _encode_batch(model, params, x, bbox=None, has_bbox=None, heart_mask=None):
    variables = {"params": params}
    kwargs = {}
    if bbox is not None:
        kwargs["bbox"]     = bbox
        kwargs["has_bbox"] = has_bbox
    if heart_mask is not None:
        kwargs["heart_mask"] = heart_mask
    return model.apply(variables, x, method=model.encode, **kwargs)


def _decode_z(model, params, z_common, z_disease, skip_feats=None, heart_mask=None):
    z = jnp.concatenate([z_common, z_disease], axis=-1)
    kwargs = {}
    if skip_feats is not None:
        kwargs["skip_feats"] = skip_feats
    if heart_mask is not None:
        kwargs["heart_mask"] = heart_mask
    return model.apply({"params": params}, z, method=model.decode, **kwargs)


# ── Public API ────────────────────────────────────────────────────────────────

def run_scaffolding(
    model,
    params,
    batch_stats,          # unused for V2; kept for API consistency
    eval_loader,
    epoch: int,
    global_step: int,
    output_dir,
    use_wandb: bool = False,
    use_bbox_cross_attn: bool = True,
    has_chexmask: bool = False,
    heart_in_zd: bool = False,
) -> Path | None:
    """
    Build the 7-panel scaffolding strip and save it.

    Picks the first Cardiomegaly image from the eval loader as the
    representative case.  When has_chexmask=True, selects the image with
    the largest GT heart mask area; otherwise falls back to largest bbox area.

    Returns the path to the saved PNG, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Get a representative Cardiomegaly image ────────────────────────────
    best_batch      = None
    best_score      = -1.0
    best_idx        = 0
    n_scanned       = 0

    for batch in eval_loader:
        if has_chexmask and 'heart_mask' in batch:
            # Prefer largest GT heart mask area (cardiomegaly half of batch)
            hm     = batch['heart_mask'].numpy()    # (B, S, S) — cardio half
            # heart_mask tensor is (2B, S, S): second half = cardio
            B_half = hm.shape[0] // 2
            hm_ca  = hm[B_half:]                    # (B, S, S) cardio samples
            scores = hm_ca.reshape(B_half, -1).sum(axis=1)
        else:
            bbox_ca = batch["bbox_disease1"].numpy()
            scores  = (bbox_ca[:, 2] - bbox_ca[:, 0]) * (bbox_ca[:, 3] - bbox_ca[:, 1])

        if scores.max() > best_score:
            best_score = float(scores.max())
            best_batch = batch
            best_idx   = int(np.argmax(scores))
        n_scanned += 1
        if n_scanned >= 8:
            break

    if best_batch is None:
        return None

    # ── 2. Extract single representative sample ───────────────────────────────
    x_cardio_np = best_batch["x_disease1"][best_idx].permute(1, 2, 0).numpy()  # (H,W,1)
    bbox_np     = best_batch["bbox_disease1"][best_idx].numpy()                  # (4,)

    x_cardio_jax = jnp.array(x_cardio_np)[None]  # (1,H,W,1)  in [-1,1]
    bbox_ca_jax  = jnp.array(bbox_np)[None]       # (1,4)
    has_bbox_jax = jnp.array([float((bbox_np[2] - bbox_np[0]) > 1e-4)])

    # CheXmask heart mask for this sample (cardio half of 2B tensor)
    heart_mask_sel = None
    heart_mask_jax = None
    if has_chexmask and 'heart_mask' in best_batch:
        hm_full   = best_batch['heart_mask'].numpy()   # (2B, S, S)
        B_half    = hm_full.shape[0] // 2
        heart_mask_sel = hm_full[B_half + best_idx]   # (S, S)  cardio sample
        heart_mask_jax = jnp.array(heart_mask_sel)[None]  # (1, S, S)

    # ── 3. Encode cardiomegaly image ──────────────────────────────────────────
    if use_bbox_cross_attn:
        ld_cardio = _encode_batch(
            model, params, x_cardio_jax,
            bbox=bbox_ca_jax, has_bbox=has_bbox_jax,
            heart_mask=heart_mask_jax,
        )
    else:
        ld_cardio = _encode_batch(model, params, x_cardio_jax,
                                   heart_mask=heart_mask_jax)

    mu_c       = ld_cardio["common"][0]       # (1, H_lat, W_lat, C_c)
    mu_d       = ld_cardio["cardiomegaly"][0] # (1, H_lat, W_lat, C_d)
    skip_feats = ld_cardio.get("skip_feats")

    # ── 4. Reconstruct: full, anatomy-only, and cardiac-only ─────────────────
    recon_full    = _decode_z(model, params, mu_c, mu_d,
                              skip_feats=skip_feats, heart_mask=heart_mask_jax)
    recon_anatomy = _decode_z(model, params, mu_c, jnp.zeros_like(mu_d),
                              skip_feats=skip_feats, heart_mask=heart_mask_jax)
    # heart_mask passed so the cardiac skip gate is CLOSED for anatomy-only.
    # Skip features are zeroed in the cardiac region, preventing the skip path
    # from bypassing z_d and filling in the heart.  This makes col 04 show a
    # CXR without a heart (cardiac region inpainted from z_c alone), and col 06
    # diff = |full − anatomy-only| will show the cardiac silhouette cleanly.

    # Panel 05: z_d-only decode (z_c=0, no skip connections).
    # With heart_in_zd, z_d was encoded from cardiac pixels only (complement
    # masking), so decoding z_d alone reveals exactly what the cardiac code
    # has learned.  No skip features: pure z_d signal, unaugmented by context.
    # Convergence goal: this should look like the GT mask in panel 02.
    recon_cardiac = _decode_z(model, params, jnp.zeros_like(mu_c), mu_d,
                               skip_feats=None, heart_mask=None)

    img_orig    = _prep_img((x_cardio_np + 1.0) / 2.0)
    img_recon   = _prep_img(np.array(recon_full[0]))
    img_anatomy = _prep_img(np.array(recon_anatomy[0]))
    img_cardiac = _prep_img(np.array(recon_cardiac[0]))
    img_diff    = np.clip(np.abs(img_recon - img_anatomy) * 5.0, 0.0, 1.0)

    H, W = img_orig.shape

    # ── 6. Compose the 7-panel strip ─────────────────────────────────────────
    CAPTIONS = [
        "00  Real CXR\n(cardiomegaly)",
        "01  + GT bbox",
        "02  + GT seg mask\n(CheXmask)",
        "03  Reconstruction\n(z_c + z_d)",
        "04  Anatomy-only\n(z_c, z_d = 0)",
        "05  Cardiac-only\n(z_d, z_c = 0)",
        "06  Diff ×5\n|03 − 04|",
    ]

    n_panels  = 7
    cell_px   = 3.0
    extra_px  = 0.35
    fig_w     = n_panels * cell_px
    fig_h     = cell_px + extra_px
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={"wspace": 0.05})

    def _show_cxr(ax, img_2d, cmap="gray", vmin=0, vmax=1):
        ax.imshow(img_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # 00: original
    _show_cxr(axes[0], img_orig)

    # 01: original + GT bbox
    axes[1].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    if (bbox_np[2] - bbox_np[0]) > 1e-4:
        rect = mpatches.Rectangle(
            (bbox_np[0] * W, bbox_np[1] * H),
            (bbox_np[2] - bbox_np[0]) * W,
            (bbox_np[3] - bbox_np[1]) * H,
            linewidth=1.5, edgecolor="lime", facecolor="none",
        )
        axes[1].add_patch(rect)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    for sp in axes[1].spines.values(): sp.set_visible(False)

    # 02: CXR + GT segmentation mask (CheXmask)
    axes[2].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    if heart_mask_sel is not None:
        mask_up = np.array(
            jax.image.resize(
                heart_mask_sel[..., None].astype(np.float32),
                (H, W, 1), method="nearest"
            )[:, :, 0]
        )
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[mask_up > 0.5] = [0.0, 0.9, 0.9, 0.5]  # cyan
        axes[2].imshow(rgba, extent=(0, W, H, 0))
        axes[2].contour(mask_up, levels=[0.5], colors=["cyan"],
                        linewidths=[1.2], extent=(0, W, 0, H))
    else:
        axes[2].text(0.5, 0.5, "no mask", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=8, color="gray",
                     style="italic")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for sp in axes[2].spines.values(): sp.set_visible(False)

    # 03: full reconstruction
    _show_cxr(axes[3], img_recon)

    # 04: anatomy-only
    _show_cxr(axes[4], img_anatomy)

    # 05: z_d-only reconstruction — what the cardiac code alone decodes to.
    # With heart_in_zd: z_d was encoded from cardiac pixels only, so this
    # should converge to look like the GT mask in panel 02.
    # Without heart_in_zd: shows attn-based z_d decode (less well-bounded).
    _show_cxr(axes[5], img_cardiac)

    # 06: amplified diff
    _show_cxr(axes[6], img_diff, cmap="hot", vmin=0, vmax=1)

    # Captions below each panel
    for ax, cap in zip(axes, CAPTIONS):
        ax.set_xlabel(cap, fontsize=6.5, labelpad=4, ha="center")

    mode_tag = "heart_in_zd" if heart_in_zd else "attn"
    fig.suptitle(
        f"Research claim scaffold — Epoch {epoch}  (step {global_step:,})  [{mode_tag}]\n"
        "Left→right: real CXR → bbox → GT seg → recon(z_c+z_d) → anatomy(z_c) "
        "→ cardiac(z_d) → diff×5",
        fontsize=8, y=1.02,
    )

    save_path = output_dir / f"scaffolding_ep{epoch:04d}.png"
    plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scaffolding → {save_path}")

    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.log(
                {"diagnostics/scaffolding": _wandb.Image(str(save_path))},
                step=global_step,
            )
        except Exception:
            pass

    return save_path
