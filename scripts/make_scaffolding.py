"""Scaffolding visuals for shared-decoder V2 and compositional V3 checkpoints."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _prep_img(arr):
    a = np.array(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[:, :, 0]
    return np.clip(a, 0.0, 1.0)


def _show_cxr(ax, img_2d, cmap="gray", vmin=0, vmax=1):
    ax.imshow(img_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _encode_batch(model, params, x, model_version, bbox=None, has_bbox=None, heart_mask=None):
    kwargs = {}
    if model_version == "v3":
        kwargs["heart_mask"] = heart_mask
    else:
        if bbox is not None:
            kwargs["bbox"] = bbox
            kwargs["has_bbox"] = has_bbox
        if heart_mask is not None:
            kwargs["heart_mask"] = heart_mask
    return model.apply({"params": params}, x, method=model.encode, **kwargs)


def _decode_v2(model, params, z_common, z_disease, skip_feats=None, heart_mask=None):
    z = jnp.concatenate([z_common, z_disease], axis=-1)
    kwargs = {}
    if skip_feats is not None:
        kwargs["skip_feats"] = skip_feats
    if heart_mask is not None:
        kwargs["heart_mask"] = heart_mask
    return model.apply({"params": params}, z, method=model.decode, **kwargs)


def _decode_v3(
    model,
    params,
    z_common,
    z_heart,
    s_ctr,
    skip_common=None,
    skip_heart=None,
):
    return model.apply(
        {"params": params},
        z_common,
        z_heart,
        s_ctr,
        skip_common=skip_common,
        skip_heart=skip_heart,
        method=model.decode,
    )


def run_scaffolding(
    model,
    params,
    batch_stats,
    eval_loader,
    epoch: int,
    global_step: int,
    output_dir,
    use_wandb: bool = False,
    model_version: str = "v2",
    use_bbox_cross_attn: bool = True,
    has_chexmask: bool = False,
    heart_in_zd: bool = False,
) -> Path | None:
    del batch_stats, heart_in_zd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_batch = None
    best_idx = 0
    best_score = -1.0
    for n_scanned, batch in enumerate(eval_loader):
        if has_chexmask and "heart_mask" in batch:
            hm = batch["heart_mask"].numpy()
            half = hm.shape[0] // 2
            scores = hm[half:].reshape(half, -1).sum(axis=1)
        else:
            bbox = batch["bbox_disease1"].numpy()
            scores = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        if scores.max() > best_score:
            best_score = float(scores.max())
            best_batch = batch
            best_idx = int(np.argmax(scores))
        if n_scanned >= 7:
            break

    if best_batch is None:
        return None

    x_cardio_np = best_batch["x_disease1"][best_idx].permute(1, 2, 0).numpy()
    bbox_np = best_batch["bbox_disease1"][best_idx].numpy()
    x_cardio_jax = jnp.array(x_cardio_np)[None]
    bbox_ca_jax = jnp.array(bbox_np)[None]
    has_bbox_jax = jnp.array([float((bbox_np[2] - bbox_np[0]) > 1e-4)])

    heart_mask_sel = None
    heart_mask_jax = None
    if has_chexmask and "heart_mask" in best_batch:
        hm_full = best_batch["heart_mask"].numpy()
        half = hm_full.shape[0] // 2
        heart_mask_sel = hm_full[half + best_idx]
        heart_mask_jax = jnp.array(heart_mask_sel)[None]

    img_orig = _prep_img((x_cardio_np + 1.0) / 2.0)
    H, W = img_orig.shape

    if model_version == "v3":
        enc = _encode_batch(
            model,
            params,
            x_cardio_jax,
            model_version="v3",
            heart_mask=heart_mask_jax,
        )
        mu_c = enc["common"][0]
        mu_h = enc["heart"][0]
        s_ctr = enc["ctr_pred"]
        x_hat, alpha_heart, aux = _decode_v3(
            model,
            params,
            mu_c,
            mu_h,
            s_ctr,
            skip_common=enc["skip_common"],
            skip_heart=enc["skip_heart"],
        )

        img_alpha = _prep_img(np.array(alpha_heart[0]))
        img_common = _prep_img(np.array(aux["x_common"][0]))
        img_heart = _prep_img(np.array(aux["x_heart"][0]))
        img_full = _prep_img(np.array(x_hat[0]))

        captions = [
            "00  Real CXR",
            "01  GT heart mask",
            "02  Pred alpha",
            "03  Common-only",
            "04  Heart-only",
            "05  Full recon",
        ]
        fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.5), gridspec_kw={"wspace": 0.05})
        _show_cxr(axes[0], img_orig)

        axes[1].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        if heart_mask_sel is not None:
            mask_up = np.array(
                jax.image.resize(heart_mask_sel[..., None].astype(np.float32), (H, W, 1), method="nearest")[:, :, 0]
            )
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[mask_up > 0.5] = [0.0, 0.9, 0.9, 0.5]
            axes[1].imshow(rgba, extent=(0, W, H, 0))
        axes[1].axis("off")

        _show_cxr(axes[2], img_alpha, cmap="plasma", vmin=0, vmax=1)
        _show_cxr(axes[3], img_common)
        _show_cxr(axes[4], img_heart)
        _show_cxr(axes[5], img_full)

        for ax, cap in zip(axes, captions):
            ax.set_xlabel(cap, fontsize=6.5, labelpad=4)

        fig.suptitle(
            f"V3 scaffold — Epoch {epoch}  (step {global_step:,})\n"
            "Compositional routing: GT mask vs predicted alpha, branch renders, full reconstruction",
            fontsize=8,
            y=1.02,
        )
    else:
        ld_cardio = _encode_batch(
            model,
            params,
            x_cardio_jax,
            model_version=model_version,
            bbox=bbox_ca_jax if use_bbox_cross_attn else None,
            has_bbox=has_bbox_jax if use_bbox_cross_attn else None,
            heart_mask=heart_mask_jax,
        )
        mu_c = ld_cardio["common"][0]
        mu_d = ld_cardio["cardiomegaly"][0]
        skip_feats = ld_cardio.get("skip_feats")

        recon_full = _decode_v2(model, params, mu_c, mu_d, skip_feats=skip_feats, heart_mask=heart_mask_jax)
        recon_anatomy = _decode_v2(
            model, params, mu_c, jnp.zeros_like(mu_d), skip_feats=skip_feats, heart_mask=heart_mask_jax
        )
        recon_cardiac = _decode_v2(model, params, jnp.zeros_like(mu_c), mu_d, skip_feats=None, heart_mask=None)

        img_recon = _prep_img(np.array(recon_full[0]))
        img_anatomy = _prep_img(np.array(recon_anatomy[0]))
        img_cardiac = _prep_img(np.array(recon_cardiac[0]))
        img_diff = np.clip(np.abs(img_recon - img_anatomy) * 5.0, 0.0, 1.0)

        captions = [
            "00  Real CXR\n(cardiomegaly)",
            "01  + GT bbox",
            "02  + GT seg mask\n(CheXmask)",
            "03  Reconstruction\n(z_c + z_d)",
            "04  Anatomy-only\n(z_c, z_d = 0)",
            "05  Cardiac-only\n(z_d, z_c = 0)",
            "06  Diff ×5\n|03 - 04|",
        ]
        fig, axes = plt.subplots(1, 7, figsize=(21.0, 3.5), gridspec_kw={"wspace": 0.05})
        _show_cxr(axes[0], img_orig)

        axes[1].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        if (bbox_np[2] - bbox_np[0]) > 1e-4:
            rect = mpatches.Rectangle(
                (bbox_np[0] * W, bbox_np[1] * H),
                (bbox_np[2] - bbox_np[0]) * W,
                (bbox_np[3] - bbox_np[1]) * H,
                linewidth=1.5, edgecolor="lime", facecolor="none",
            )
            axes[1].add_patch(rect)
        axes[1].axis("off")

        axes[2].imshow(img_orig, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        if heart_mask_sel is not None:
            mask_up = np.array(
                jax.image.resize(heart_mask_sel[..., None].astype(np.float32), (H, W, 1), method="nearest")[:, :, 0]
            )
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[mask_up > 0.5] = [0.0, 0.9, 0.9, 0.5]
            axes[2].imshow(rgba, extent=(0, W, H, 0))
        axes[2].axis("off")

        _show_cxr(axes[3], img_recon)
        _show_cxr(axes[4], img_anatomy)
        _show_cxr(axes[5], img_cardiac)
        _show_cxr(axes[6], img_diff, cmap="hot", vmin=0, vmax=1)

        for ax, cap in zip(axes, captions):
            ax.set_xlabel(cap, fontsize=6.5, labelpad=4)

        fig.suptitle(
            f"V2 scaffold — Epoch {epoch}  (step {global_step:,})\n"
            "Left-to-right: real CXR, bbox, GT mask, full recon, anatomy-only, cardiac-only, diff",
            fontsize=8,
            y=1.02,
        )

    save_path = output_dir / f"scaffolding_ep{epoch:04d}.png"
    plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scaffolding -> {save_path}")

    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.log({"diagnostics/scaffolding": _wandb.Image(str(save_path))}, step=global_step)
        except Exception:
            pass

    return save_path
