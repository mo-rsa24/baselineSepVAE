"""
8-panel visual scaffolding for the research claim.

Builds a single wide composite strip that reads left-to-right as a
self-contained proof of disentanglement, without requiring the reader
to understand silhouette scores or latent geometry.

Panel layout (left → right):
  00  Real cardiomegaly CXR
  01  Same + GT bbox (green)
  02  Same + attention heatmap α-blended
  03  Reconstruction  (z_common, z_disease)
  04  Anatomy-only    (z_common, z_disease=0)
  05  Pixel diff ×5   |panel03 − panel04|  (RdBu colourmap)
  06  2D PCA scatter  (eval set; highlighted point = the selected image)
  07  Bar chart: ||z_disease|| mean — Normal vs Cardiomegaly (from eval set)

Exported function (called from train_sep_vae.py every manifold_every epochs):
    run_scaffolding(model, params, batch_stats, eval_loader,
                    epoch, global_step, output_dir,
                    use_wandb=False, use_bbox_cross_attn=True) → Path or None

Logged to W&B as "diagnostics/scaffolding".
"""

from __future__ import annotations

import io
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.decomposition import PCA


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prep_img(arr):
    """(H, W, 1) JAX/numpy in [0,1] → (H, W) float32 numpy."""
    return np.clip(np.array(arr[:, :, 0], dtype=np.float32), 0.0, 1.0)


def _encode_batch(model, params, x, bbox=None, has_bbox=None):
    variables = {"params": params}
    kwargs = {}
    if bbox is not None:
        kwargs["bbox"]     = bbox
        kwargs["has_bbox"] = has_bbox
    return model.apply(variables, x, method=model.encode, **kwargs)


def _decode_z(model, params, z_common, z_disease):
    z = jnp.concatenate([z_common, z_disease], axis=-1)
    return model.apply({"params": params}, z, method=model.decode)


def _collect_eval_latents(model, params, eval_loader, max_samples=300,
                           use_bbox_cross_attn=True):
    """
    Encode a subset of the eval set.

    Returns dicts:
        mu_common  (N, D_c)  — spatially-pooled z_common means
        mu_disease (N, D_d)  — spatially-pooled z_disease means
        labels     (N,)      — 0=Normal, 1=Cardiomegaly
    """
    all_mu_c, all_mu_d, all_labels = [], [], []
    n = 0
    for batch in eval_loader:
        if n >= max_samples:
            break
        x_norm   = jnp.array(batch["x_norm"].permute(0, 2, 3, 1).numpy())
        x_cardio = jnp.array(batch["x_disease1"].permute(0, 2, 3, 1).numpy())
        B = x_norm.shape[0]

        bbox_ca  = jnp.array(batch["bbox_disease1"].numpy())
        has_ca   = ((bbox_ca[:, 2] - bbox_ca[:, 0]) > 1e-4).astype(jnp.float32)
        bbox_z   = jnp.zeros_like(bbox_ca)
        has_z    = jnp.zeros(B, dtype=jnp.float32)

        for x, bbox, has_bbox, lbl in [
            (x_norm,   bbox_z, has_z,  0),
            (x_cardio, bbox_ca, has_ca, 1),
        ]:
            if use_bbox_cross_attn:
                ld = _encode_batch(model, params, x, bbox=bbox, has_bbox=has_bbox)
            else:
                ld = _encode_batch(model, params, x)
            mu_c = np.array(jnp.mean(ld["common"][0],       axis=(1, 2)))  # (B, C)
            mu_d = np.array(jnp.mean(ld["cardiomegaly"][0], axis=(1, 2)))  # (B, C)
            all_mu_c.append(mu_c)
            all_mu_d.append(mu_d)
            all_labels.append(np.full(B, lbl, dtype=np.int32))
        n += B

    if not all_mu_c:
        return None
    return {
        "mu_common":  np.concatenate(all_mu_c,   axis=0),
        "mu_disease": np.concatenate(all_mu_d,   axis=0),
        "labels":     np.concatenate(all_labels, axis=0),
    }


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
) -> Path | None:
    """
    Build the 8-panel scaffolding strip and save it.

    Picks the first Cardiomegaly image from the eval loader as the
    representative case (the one with the largest GT bbox area).

    Returns the path to the saved PNG, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Get a representative Cardiomegaly image ────────────────────────────
    best_batch = None
    best_bbox_area = -1.0
    best_idx = 0
    n_scanned = 0

    for batch in eval_loader:
        bbox_ca = batch["bbox_disease1"].numpy()          # (B, 4) normalised
        areas   = (bbox_ca[:, 2] - bbox_ca[:, 0]) * (bbox_ca[:, 3] - bbox_ca[:, 1])
        if areas.max() > best_bbox_area:
            best_bbox_area = float(areas.max())
            best_batch     = batch
            best_idx       = int(np.argmax(areas))
        n_scanned += 1
        if n_scanned >= 8:   # scan first 8 batches, pick the most prominent case
            break

    if best_batch is None:
        return None

    # Extract representative images (single sample)
    x_cardio_np = best_batch["x_disease1"][best_idx].permute(1, 2, 0).numpy()  # (H,W,1)
    x_norm_np   = best_batch["x_norm"][best_idx].permute(1, 2, 0).numpy()
    bbox_np     = best_batch["bbox_disease1"][best_idx].numpy()                  # (4,)

    x_cardio_jax = jnp.array(x_cardio_np)[None]  # (1,H,W,1)  in [-1,1]
    x_norm_jax   = jnp.array(x_norm_np)[None]

    bbox_ca_jax  = jnp.array(bbox_np)[None]       # (1,4)
    has_bbox_jax = jnp.array([float((bbox_np[2] - bbox_np[0]) > 1e-4)])
    bbox_z       = jnp.zeros_like(bbox_ca_jax)
    has_z        = jnp.zeros(1, dtype=jnp.float32)

    # ── 2. Encode cardiomegaly image ──────────────────────────────────────────
    if use_bbox_cross_attn:
        ld_cardio = _encode_batch(model, params, x_cardio_jax,
                                   bbox=bbox_ca_jax, has_bbox=has_bbox_jax)
    else:
        ld_cardio = _encode_batch(model, params, x_cardio_jax)

    mu_c  = ld_cardio["common"][0]       # (1, H_lat, W_lat, C_c)
    mu_d  = ld_cardio["cardiomegaly"][0] # (1, H_lat, W_lat, C_d)
    attn  = np.array(ld_cardio["attn_maps"]["cardiomegaly"][0])  # (H_lat, W_lat)

    # ── 3. Reconstruct: full and anatomy-only ────────────────────────────────
    recon_full     = _decode_z(model, params, mu_c, mu_d)          # (1,H,W,1) in [0,1]
    recon_anatomy  = _decode_z(model, params, mu_c, jnp.zeros_like(mu_d))

    img_orig     = _prep_img(((x_cardio_np + 1.0) / 2.0))     # (H,W) in [0,1]
    img_recon    = _prep_img(np.array(recon_full[0]))
    img_anatomy  = _prep_img(np.array(recon_anatomy[0]))
    img_diff     = np.clip(np.abs(img_recon - img_anatomy) * 5.0, 0.0, 1.0)

    H, W = img_orig.shape

    # ── 4. Collect eval latents for PCA scatter (panel 06) ───────────────────
    latent_data = _collect_eval_latents(
        model, params, eval_loader,
        max_samples=300,
        use_bbox_cross_attn=use_bbox_cross_attn,
    )

    # ── 5. Build PCA scatter (panel 06) ──────────────────────────────────────
    # z_disease head only — that's the axis of interest
    pca_fig, pca_ax = plt.subplots(figsize=(2.8, 2.8))
    if latent_data is not None:
        mu_d_all = latent_data["mu_disease"]
        lbl_all  = latent_data["labels"]
        pca = PCA(n_components=2)
        z2d = pca.fit_transform(mu_d_all)

        for cls_id, cls_name, color in [(0, "Normal", "#1f77b4"), (1, "Cardio", "#2ca02c")]:
            mask = lbl_all == cls_id
            pca_ax.scatter(z2d[mask, 0], z2d[mask, 1],
                           c=color, s=6, alpha=0.4, label=cls_name, rasterized=True)

        # Highlight the selected cardiomegaly image
        mu_d_sel = np.array(jnp.mean(mu_d, axis=(1, 2)))  # (1, C)
        z_sel    = pca.transform(mu_d_sel)
        pca_ax.scatter(z_sel[0, 0], z_sel[0, 1],
                       c="red", s=80, marker="*", zorder=10, label="this image")
    else:
        pca_ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=pca_ax.transAxes, fontsize=8, color="gray")

    pca_ax.set_title("z_disease PCA", fontsize=8)
    pca_ax.legend(fontsize=6, loc="best", markerscale=2)
    pca_ax.tick_params(labelsize=6)
    pca_ax.spines["top"].set_visible(False)
    pca_ax.spines["right"].set_visible(False)
    pca_buf = io.BytesIO()
    pca_fig.savefig(pca_buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(pca_fig)
    pca_buf.seek(0)
    pca_panel = np.array(Image.open(pca_buf).convert("RGB"), dtype=np.float32) / 255.0

    # ── 6. Build bar chart (panel 07) — ||z_disease|| Normal vs Cardiomegaly ──
    bar_fig, bar_ax = plt.subplots(figsize=(2.8, 2.8))
    if latent_data is not None:
        mu_d_all = latent_data["mu_disease"]
        lbl_all  = latent_data["labels"]
        norms_n  = np.linalg.norm(mu_d_all[lbl_all == 0], axis=1)
        norms_c  = np.linalg.norm(mu_d_all[lbl_all == 1], axis=1)
        means    = [norms_n.mean(), norms_c.mean()]
        sems     = [norms_n.std() / max(np.sqrt(len(norms_n)), 1),
                    norms_c.std() / max(np.sqrt(len(norms_c)), 1)]
        bar_ax.bar(["Normal", "Cardio"], means, yerr=sems,
                   color=["#1f77b4", "#2ca02c"], capsize=5, alpha=0.82,
                   error_kw={"linewidth": 1.2, "ecolor": "grey"})
        bar_ax.set_ylabel("mean ‖z_disease‖₂", fontsize=8)
    else:
        bar_ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=bar_ax.transAxes, fontsize=8, color="gray")

    bar_ax.set_title("Disease latent norm", fontsize=8)
    bar_ax.tick_params(labelsize=7)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_buf = io.BytesIO()
    bar_fig.savefig(bar_buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(bar_fig)
    bar_buf.seek(0)
    bar_panel = np.array(Image.open(bar_buf).convert("RGB"), dtype=np.float32) / 255.0

    # ── 7. Compose the 8-panel strip ─────────────────────────────────────────
    CAPTIONS = [
        "00  Real CXR\n(cardiomegaly)",
        "01  + GT bbox",
        "02  + attn overlay",
        "03  Reconstruction\n(z_c + z_d)",
        "04  Anatomy-only\n(z_d = 0)",
        "05  Diff ×5\n|03 − 04|",
        "06  z_disease PCA\n(red★ = this image)",
        "07  ‖z_disease‖₂\nvs class",
    ]

    n_panels   = 8
    cell_px    = 3.0        # inches per CXR panel
    extra_px   = 0.35       # caption height in inches
    fig_w      = n_panels * cell_px
    fig_h      = cell_px + extra_px
    fig, axes  = plt.subplots(1, n_panels,
                               figsize=(fig_w, fig_h),
                               gridspec_kw={"wspace": 0.05})

    # Panels 00-05: grayscale / diff images
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

    # 02: CXR + attention overlay
    # Upsample attn map to image size
    attn_up = np.array(
        jax.image.resize(attn[..., None], (H, W, 1), method="bilinear")[:, :, 0]
    )
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    axes[2].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    axes[2].imshow(attn_up, cmap="hot", alpha=0.55,
                   extent=(0, W, H, 0), interpolation="bilinear")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for sp in axes[2].spines.values(): sp.set_visible(False)

    # 03: full reconstruction
    _show_cxr(axes[3], img_recon)

    # 04: anatomy-only
    _show_cxr(axes[4], img_anatomy)

    # 05: amplified diff (RdBu: blue=0, white=small, red=large)
    _show_cxr(axes[5], img_diff, cmap="hot", vmin=0, vmax=1)

    # 06: PCA scatter (pre-rendered as RGB image)
    axes[6].imshow(pca_panel, aspect="auto")
    axes[6].set_xticks([]); axes[6].set_yticks([])
    for sp in axes[6].spines.values(): sp.set_visible(False)

    # 07: bar chart (pre-rendered)
    axes[7].imshow(bar_panel, aspect="auto")
    axes[7].set_xticks([]); axes[7].set_yticks([])
    for sp in axes[7].spines.values(): sp.set_visible(False)

    # Captions below each panel
    for ax, cap in zip(axes, CAPTIONS):
        ax.set_xlabel(cap, fontsize=6.5, labelpad=4, ha="center")

    fig.suptitle(
        f"Research claim scaffold — Epoch {epoch}  "
        f"(step {global_step:,})\n"
        "Left→right: real CXR → bbox → attn → recon → anatomy-only → diff → latent PCA → norm",
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
