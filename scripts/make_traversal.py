"""Traversal and composition diagnostics for V2 and V3 SepVAE variants."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
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


def _encode(model, params, x, model_version, bbox=None, has_bbox=None, heart_mask=None):
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


def _decode_v2(model, params, z_c, z_d, skip_feats=None, heart_mask=None):
    z = jnp.concatenate([z_c, z_d], axis=-1)
    kwargs = {}
    if skip_feats is not None:
        kwargs["skip_feats"] = skip_feats
    if heart_mask is not None:
        kwargs["heart_mask"] = heart_mask
    return model.apply({"params": params}, z, method=model.decode, **kwargs)


def _decode_v3(model, params, z_c, z_h, s_ctr, skip_common=None, skip_heart=None):
    return model.apply(
        {"params": params},
        z_c,
        z_h,
        s_ctr,
        skip_common=skip_common,
        skip_heart=skip_heart,
        method=model.decode,
    )


def _get_best_cardio(eval_loader, has_chexmask, n_batches=8):
    best_batch = None
    best_score = -1.0
    best_idx = 0
    for i, batch in enumerate(eval_loader):
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
        if i + 1 >= n_batches:
            break
    return best_batch, best_idx


def _get_best_normal(eval_loader, has_chexmask, n_batches=8):
    best_batch = None
    best_score = -1.0
    best_idx = 0
    for i, batch in enumerate(eval_loader):
        if has_chexmask and "heart_mask" in batch:
            hm = batch["heart_mask"].numpy()
            half = hm.shape[0] // 2
            scores = hm[:half].reshape(half, -1).sum(axis=1)
        else:
            x_norm = batch["x_norm"].numpy()
            scores = x_norm.reshape(x_norm.shape[0], -1).var(axis=1)
        if scores.max() > best_score:
            best_score = float(scores.max())
            best_batch = batch
            best_idx = int(np.argmax(scores))
        if i + 1 >= n_batches:
            break
    return best_batch, best_idx


def _extract_sample(batch, idx, kind, has_chexmask):
    if kind == "cardio":
        x_np = batch["x_disease1"][idx].permute(1, 2, 0).numpy()
        bbox_np = batch["bbox_disease1"][idx].numpy()
        hm_full = batch["heart_mask"].numpy() if (has_chexmask and "heart_mask" in batch) else None
        heart_mask_np = hm_full[hm_full.shape[0] // 2 + idx] if hm_full is not None else None
    else:
        x_np = batch["x_norm"][idx].permute(1, 2, 0).numpy()
        bbox_np = np.zeros(4, dtype=np.float32)
        hm_full = batch["heart_mask"].numpy() if (has_chexmask and "heart_mask" in batch) else None
        heart_mask_np = hm_full[idx] if hm_full is not None else None

    x_jax = jnp.array(x_np)[None]
    bbox_jax = jnp.array(bbox_np)[None]
    has_bbox_jax = jnp.array([float((bbox_np[2] - bbox_np[0]) > 1e-4)])
    heart_mask_jax = jnp.array(heart_mask_np)[None] if heart_mask_np is not None else None
    return x_jax, heart_mask_jax, bbox_jax, has_bbox_jax, x_np


V2_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
V3_CTR_DELTAS = [-0.15, -0.07, 0.0, 0.07, 0.15]


def run_traversal(
    model,
    params,
    batch_stats,
    eval_loader,
    epoch: int,
    global_step: int,
    output_dir,
    use_wandb: bool = False,
    model_version: str = "v2",
    use_bbox_cross_attn: bool = False,
    has_chexmask: bool = False,
    heart_in_zd: bool = False,
) -> Path | None:
    del batch_stats, heart_in_zd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_batch, best_idx = _get_best_cardio(eval_loader, has_chexmask)
    if best_batch is None:
        return None

    x_jax, hm_jax, bbox_jax, has_bbox_jax, x_np = _extract_sample(
        best_batch, best_idx, "cardio", has_chexmask
    )

    if model_version == "v3":
        enc = _encode(model, params, x_jax, "v3", heart_mask=hm_jax)
        mu_c = enc["common"][0]
        mu_h = enc["heart"][0]
        s_base = float(np.array(enc["ctr_pred"][0]))
        panels = []
        sweep_vals = []
        for delta in V3_CTR_DELTAS:
            s_val = float(np.clip(s_base + delta, 0.01, 0.99))
            sweep_vals.append(s_val)
            x_hat, _, _ = _decode_v3(
                model,
                params,
                mu_c,
                mu_h,
                jnp.array([s_val], dtype=jnp.float32),
                skip_common=enc["skip_common"],
                skip_heart=enc["skip_heart"],
            )
            panels.append(_prep_img(np.array(x_hat[0])))
        title = "s_ctr sweep"
    else:
        enc = _encode(
            model,
            params,
            x_jax,
            model_version,
            bbox=bbox_jax if use_bbox_cross_attn else None,
            has_bbox=has_bbox_jax if use_bbox_cross_attn else None,
            heart_mask=hm_jax,
        )
        mu_c = enc["common"][0]
        mu_d = enc["cardiomegaly"][0]
        skip_feats = enc.get("skip_feats")
        panels = []
        sweep_vals = list(V2_ALPHAS)
        for alpha in V2_ALPHAS:
            out = _decode_v2(model, params, mu_c, alpha * mu_d, skip_feats=skip_feats, heart_mask=hm_jax)
            panels.append(_prep_img(np.array(out[0])))
        title = "z_d sweep"

    img_orig = _prep_img((x_np + 1.0) / 2.0)
    fig, axes = plt.subplots(1, 1 + len(panels), figsize=((1 + len(panels)) * 3.0, 3.5), gridspec_kw={"wspace": 0.05})
    _show_cxr(axes[0], img_orig)
    axes[0].set_xlabel("Input", fontsize=6.5, labelpad=4)
    for i, (value, img) in enumerate(zip(sweep_vals, panels)):
        _show_cxr(axes[i + 1], img)
        axes[i + 1].set_xlabel(f"{value:.2f}", fontsize=6.5, labelpad=4)

    fig.suptitle(
        f"{title} — Epoch {epoch}  (step {global_step:,})",
        fontsize=8,
        y=1.02,
    )

    save_path = output_dir / f"traversal_ep{epoch:04d}.png"
    plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Traversal strip -> {save_path}")

    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.log({"diagnostics/traversal": _wandb.Image(str(save_path))}, step=global_step)
        except Exception:
            pass

    return save_path


def run_composition(
    model,
    params,
    batch_stats,
    eval_loader,
    epoch: int,
    global_step: int,
    output_dir,
    use_wandb: bool = False,
    model_version: str = "v2",
    use_bbox_cross_attn: bool = False,
    has_chexmask: bool = False,
    heart_in_zd: bool = False,
) -> Path | None:
    del batch_stats, heart_in_zd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cardio_batch, cardio_idx = _get_best_cardio(eval_loader, has_chexmask)
    norm_batch, norm_idx = _get_best_normal(eval_loader, has_chexmask)
    if cardio_batch is None or norm_batch is None:
        return None

    xC_jax, hmC_jax, bboxC_jax, has_bboxC_jax, _ = _extract_sample(cardio_batch, cardio_idx, "cardio", has_chexmask)
    xN_jax, hmN_jax, bboxN_jax, has_bboxN_jax, _ = _extract_sample(norm_batch, norm_idx, "norm", has_chexmask)

    if model_version == "v3":
        ldC = _encode(model, params, xC_jax, "v3", heart_mask=hmC_jax)
        ldN = _encode(model, params, xN_jax, "v3", heart_mask=hmN_jax)

        def _dec(zc, zh, s, sc, sh):
            x_hat, _, _ = _decode_v3(model, params, zc, zh, s, skip_common=sc, skip_heart=sh)
            return _prep_img(np.array(x_hat[0]))

        img_NN = _dec(ldN["common"][0], ldN["heart"][0], ldN["ctr_pred"], ldN["skip_common"], ldN["skip_heart"])
        img_NC = _dec(ldN["common"][0], ldC["heart"][0], ldC["ctr_pred"], ldN["skip_common"], ldC["skip_heart"])
        img_CN = _dec(ldC["common"][0], ldN["heart"][0], ldN["ctr_pred"], ldC["skip_common"], ldN["skip_heart"])
        img_CC = _dec(ldC["common"][0], ldC["heart"][0], ldC["ctr_pred"], ldC["skip_common"], ldC["skip_heart"])
        titles = [
            "N common + N heart",
            "N common + C heart",
            "C common + N heart",
            "C common + C heart",
        ]
    else:
        ldC = _encode(
            model, params, xC_jax, model_version,
            bbox=bboxC_jax if use_bbox_cross_attn else None,
            has_bbox=has_bboxC_jax if use_bbox_cross_attn else None,
            heart_mask=hmC_jax,
        )
        ldN = _encode(
            model, params, xN_jax, model_version,
            bbox=bboxN_jax if use_bbox_cross_attn else None,
            has_bbox=has_bboxN_jax if use_bbox_cross_attn else None,
            heart_mask=hmN_jax,
        )

        def _dec(zc, zd, skip, hm):
            return _prep_img(np.array(_decode_v2(model, params, zc, zd, skip_feats=skip, heart_mask=hm)[0]))

        img_NN = _dec(ldN["common"][0], ldN["cardiomegaly"][0], ldN.get("skip_feats"), hmN_jax)
        img_NC = _dec(ldN["common"][0], ldC["cardiomegaly"][0], ldN.get("skip_feats"), hmN_jax)
        img_CN = _dec(ldC["common"][0], ldN["cardiomegaly"][0], ldC.get("skip_feats"), hmC_jax)
        img_CC = _dec(ldC["common"][0], ldC["cardiomegaly"][0], ldC.get("skip_feats"), hmC_jax)
        titles = [
            "N common + N cardio",
            "N common + C cardio",
            "C common + N cardio",
            "C common + C cardio",
        ]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.8), gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    for ax, img, title in zip(axes.flat, [img_NN, img_NC, img_CN, img_CC], titles):
        _show_cxr(ax, img)
        ax.set_title(title, fontsize=7, pad=3)

    fig.suptitle(
        f"Composition grid — Epoch {epoch}  (step {global_step:,})",
        fontsize=8,
        y=0.98,
    )

    save_path = output_dir / f"composition_ep{epoch:04d}.png"
    plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Composition grid -> {save_path}")

    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.log({"diagnostics/composition": _wandb.Image(str(save_path))}, step=global_step)
        except Exception:
            pass

    return save_path
