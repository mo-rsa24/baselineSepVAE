"""Structural loss stack for the binary compositional SepVAEV3."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from losses.sep_vae_losses import kl_divergence_standard, reconstruction_loss


def _resize_mask(mask: jnp.ndarray, target_hw: Tuple[int, int]) -> jnp.ndarray:
    bsz = mask.shape[0]
    return jax.image.resize(
        mask[..., None],
        (bsz, target_hw[0], target_hw[1], 1),
        method="nearest",
    )


def masked_region_mse(
    x_true: jnp.ndarray,
    x_pred: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    inside: bool,
    valid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Average per-sample MSE inside or outside a binary mask."""
    region = mask if inside else (1.0 - mask)
    sq_err = jnp.square(x_true - x_pred) * region
    per_sample = jnp.sum(sq_err, axis=(1, 2, 3)) / jnp.maximum(
        jnp.sum(region, axis=(1, 2, 3)), 1.0
    )
    if valid is None:
        valid = jnp.ones((x_true.shape[0],), dtype=jnp.float32)
    return jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def alpha_bce_loss(
    alpha_logits: jnp.ndarray,
    target_mask: jnp.ndarray,
    valid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Binary cross-entropy on the predicted alpha mask."""
    per_px = optax_sigmoid_bce(alpha_logits, target_mask)
    per_sample = jnp.mean(per_px, axis=(1, 2, 3))
    if valid is None:
        valid = jnp.ones((alpha_logits.shape[0],), dtype=jnp.float32)
    return jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def alpha_dice_loss(
    alpha_pred: jnp.ndarray,
    target_mask: jnp.ndarray,
    valid: Optional[jnp.ndarray] = None,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Soft Dice loss on the predicted alpha mask."""
    alpha_flat = alpha_pred.reshape(alpha_pred.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)
    intersection = jnp.sum(alpha_flat * target_flat, axis=1)
    denom = jnp.sum(alpha_flat, axis=1) + jnp.sum(target_flat, axis=1)
    per_sample = 1.0 - (2.0 * intersection + eps) / (denom + eps)
    if valid is None:
        valid = jnp.ones((alpha_pred.shape[0],), dtype=jnp.float32)
    return jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def ctr_regression_loss(
    ctr_pred: jnp.ndarray,
    ctr_gt: jnp.ndarray,
    valid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    per_sample = jnp.abs(ctr_pred - ctr_gt)
    if valid is None:
        valid = jnp.ones((ctr_pred.shape[0],), dtype=jnp.float32)
    return jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def optax_sigmoid_bce(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Minimal sigmoid BCE to avoid importing optax into the loss module."""
    return (
        jnp.maximum(logits, 0.0)
        - logits * labels
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


@dataclass
class SepVAEV3LossConfig:
    weight_rec: float = 1.0
    weight_kl_common: float = 1e-4
    weight_kl_heart: float = 1e-4
    weight_alpha: float = 1.0
    weight_common_out: float = 1.0
    weight_heart_in: float = 1.0
    weight_ctr: float = 1.0
    kl_free_bits: float = 0.0


def sepvae_v3_loss(
    model,
    params,
    batch: Dict,
    key: jax.random.PRNGKey,
    cfg: SepVAEV3LossConfig,
    kl_anneal: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return total loss, logs, pooled common latents, pooled heart latents, and recon."""
    x_norm = batch["x_norm"]
    x_cardio = batch["x_disease1"]
    x = jnp.concatenate([x_norm, x_cardio], axis=0)

    heart_mask = batch["heart_mask"]
    ctr = batch["ctr"]
    has_mask = batch.get("has_mask")
    if has_mask is None:
        has_mask = jnp.ones((x.shape[0],), dtype=jnp.float32)
    else:
        has_mask = has_mask.astype(jnp.float32)

    outputs = model.apply(
        {"params": params},
        x,
        heart_mask,
        key=key,
        train=True,
        sample=True,
    )

    x_01 = (x + 1.0) / 2.0
    x_hat = outputs["x_hat"]
    alpha_pred = outputs["alpha_heart"]
    alpha_logits = outputs["aux"]["alpha_logits"]
    x_common = outputs["aux"]["x_common"]
    x_heart = outputs["aux"]["x_heart"]

    mask_full = _resize_mask(heart_mask, x_01.shape[1:3])
    valid_mask = has_mask.reshape(-1)

    mu_c, logvar_c = outputs["common"]
    mu_h, logvar_h = outputs["heart"]

    l_rec = reconstruction_loss(x_01, x_hat)
    kl_common = jnp.mean(
        kl_divergence_standard(mu_c, logvar_c, free_bits=cfg.kl_free_bits)
    )
    kl_heart = jnp.mean(
        kl_divergence_standard(mu_h, logvar_h, free_bits=cfg.kl_free_bits)
    )
    l_alpha_bce = alpha_bce_loss(alpha_logits, mask_full, valid=valid_mask)
    l_alpha_dice = alpha_dice_loss(alpha_pred, mask_full, valid=valid_mask)
    l_alpha = l_alpha_bce + l_alpha_dice
    l_common_out = masked_region_mse(
        x_01, x_common, mask_full, inside=False, valid=valid_mask
    )
    l_heart_in = masked_region_mse(
        x_01, x_heart, mask_full, inside=True, valid=valid_mask
    )
    l_ctr = ctr_regression_loss(outputs["ctr_pred"], ctr, valid=valid_mask)

    anneal = kl_anneal if kl_anneal is not None else jnp.float32(1.0)
    l_kl = anneal * (
        cfg.weight_kl_common * kl_common + cfg.weight_kl_heart * kl_heart
    )

    total_loss = (
        cfg.weight_rec * l_rec
        + l_kl
        + cfg.weight_alpha * l_alpha
        + cfg.weight_common_out * l_common_out
        + cfg.weight_heart_in * l_heart_in
        + cfg.weight_ctr * l_ctr
    )

    z_common_pooled = jnp.mean(outputs["z_common_sample"], axis=(1, 2))
    z_heart_pooled = jnp.mean(outputs["z_heart_sample"], axis=(1, 2))
    z_common_norm = jnp.linalg.norm(z_common_pooled, axis=-1)
    z_heart_norm = jnp.linalg.norm(z_heart_pooled, axis=-1)

    labels = batch.get("disease_labels")
    if labels is None:
        labels = jnp.concatenate([
            jnp.zeros((x_norm.shape[0],), dtype=jnp.int32),
            jnp.ones((x_cardio.shape[0],), dtype=jnp.int32),
        ])
    labels = labels.reshape(-1)
    mask_normal = (labels == 0).astype(jnp.float32)
    mask_cardio = (labels == 1).astype(jnp.float32)

    z_heart_norm_normal = jnp.sum(z_heart_norm * mask_normal) / jnp.maximum(jnp.sum(mask_normal), 1.0)
    z_heart_norm_cardio = jnp.sum(z_heart_norm * mask_cardio) / jnp.maximum(jnp.sum(mask_cardio), 1.0)

    logs = {
        "loss/total": total_loss,
        "loss/reconstruction": l_rec,
        "loss/kl_common": kl_common,
        "loss/kl_heart": kl_heart,
        "loss/kl_total": kl_common + kl_heart,
        "loss/kl_weighted": l_kl,
        "loss/kl_anneal": anneal,
        "loss/alpha_bce": l_alpha_bce,
        "loss/alpha_dice": l_alpha_dice,
        "loss/alpha_mask": l_alpha,
        "loss/common_out": l_common_out,
        "loss/heart_in": l_heart_in,
        "loss/ctr": l_ctr,
        "metrics/alpha_mean": jnp.mean(alpha_pred),
        "metrics/ctr_pred_mean": jnp.mean(outputs["ctr_pred"]),
        "metrics/z_common_norm_mean": jnp.mean(z_common_norm),
        "metrics/z_heart_norm_mean": jnp.mean(z_heart_norm),
        "metrics/z_heart_norm_normal": z_heart_norm_normal,
        "metrics/z_heart_norm_cardio": z_heart_norm_cardio,
    }
    return total_loss, logs, z_common_pooled, z_heart_pooled, x_hat
