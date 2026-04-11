"""Structural loss stack for the binary compositional SepVAEV3."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from losses.sep_vae_losses import (
    kl_divergence_standard,
    kl_divergence_conditional,
    reconstruction_loss,
    supervised_contrastive_loss,
    factor_vae_mi_loss,
    ctr_adv_confusion_loss,
)


def _resize_mask(mask: jnp.ndarray, target_hw: Tuple[int, int]) -> jnp.ndarray:
    bsz = mask.shape[0]
    return jax.image.resize(
        mask[..., None],
        (bsz, target_hw[0], target_hw[1], 1),
        method="nearest",
    )


def _make_soft_mask(mask: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Gaussian-blur a binary mask to create a soft boundary blending region.

    mask  : (B, H, W, 1), values in {0, 1}
    sigma : Gaussian standard deviation in pixels

    Returns soft_mask (B, H, W, 1) with values in [0, 1]:
      ≈ 1  deep inside the heart region
      ≈ 0  deep outside
      smooth gradient of width ~2*sigma at the boundary

    When sigma <= 0 the input mask is returned unchanged (hard boundary).
    """
    if sigma <= 0.0:
        return mask.astype(jnp.float32)

    radius = max(1, int(sigma * 3.0 + 0.5))

    # Build a separable 2-D Gaussian kernel
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k2d = jnp.outer(k1d, k1d)               # (2r+1, 2r+1)
    kernel = k2d[None, None, :, :]           # (1, 1, kH, kW) for lax.conv

    # (B, H, W, 1) → (B, 1, H, W) for NCHW conv
    m = jnp.transpose(mask.astype(jnp.float32), (0, 3, 1, 2))

    # Zero-pad by `radius` so the output matches the input spatial size
    m_padded = jnp.pad(m, ((0, 0), (0, 0), (radius, radius), (radius, radius)))
    blurred = jax.lax.conv_general_dilated(
        m_padded,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        feature_group_count=1,
    )  # (B, 1, H, W)

    # Back to (B, H, W, 1)
    return jnp.clip(jnp.transpose(blurred, (0, 2, 3, 1)), 0.0, 1.0)


def split_reconstruction_loss(
    x_true: jnp.ndarray,
    x_common: jnp.ndarray,
    x_heart: jnp.ndarray,
    soft_mask: jnp.ndarray,
    valid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Split reconstruction loss with soft boundary blending.

    Each branch is solely responsible for its own region:
      - heart branch  owns pixels where soft_mask ≈ 1  (inside heart)
      - common branch owns pixels where soft_mask ≈ 0  (outside heart)
      - boundary zone  (0 < soft_mask < 1)  blends both contributions

    L = mean_pixels[ soft_M * (x - x_heart)^2  +  (1 - soft_M) * (x - x_common)^2 ]

    This prevents the common branch from acting as a residual background
    inside the heart region, which causes the ring artefact.
    """
    err_heart  = jnp.square(x_true - x_heart)
    err_common = jnp.square(x_true - x_common)
    pixelwise  = soft_mask * err_heart + (1.0 - soft_mask) * err_common
    per_sample = jnp.mean(pixelwise, axis=(1, 2, 3))

    if valid is None:
        valid = jnp.ones((x_true.shape[0],), dtype=jnp.float32)
    return jnp.sum(per_sample * valid) / jnp.maximum(jnp.sum(valid), 1.0)


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


def alpha_overlap_metrics(
    alpha_pred: jnp.ndarray,
    target_mask: jnp.ndarray,
    valid: Optional[jnp.ndarray] = None,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Thresholded Dice and IoU for the predicted alpha mask."""
    pred_bin = (alpha_pred >= threshold).astype(jnp.float32)
    target_bin = (target_mask >= 0.5).astype(jnp.float32)

    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    target_flat = target_bin.reshape(target_bin.shape[0], -1)
    inter = jnp.sum(pred_flat * target_flat, axis=1)
    pred_sum = jnp.sum(pred_flat, axis=1)
    target_sum = jnp.sum(target_flat, axis=1)
    union = pred_sum + target_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)

    if valid is None:
        valid = jnp.ones((alpha_pred.shape[0],), dtype=jnp.float32)
    valid = valid.reshape(-1)
    dice = jnp.sum(dice * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    iou = jnp.sum(iou * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    return dice, iou


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
    # V2 separation mechanisms (default off for backward compat)
    weight_mi_factor: float = 0.0        # FactorVAE TC minimisation on (z_common, z_heart)
    weight_heart_supcon: float = 0.0     # Supervised contrastive on pooled z_heart means
    supcon_temperature: float = 0.1
    use_conditional_kl_heart: bool = False  # Switch z_heart KL to label-conditional prior
    sigma_inactive: float = 1.0          # Prior sigma for Normal images (1.0 = standard N(0,I))
    # CTR decorrelation adversary (V3 only)
    weight_ctr_adv: float = 0.0          # Confusion loss: push adv(z_heart) → 0.5
    # Split reconstruction (Fix 2 — absolute branch encoding, no ring artefact)
    split_rec_weight: float = 0.0        # 0 = soft-composite l_rec; 1 = fully split; linear warmup in between
    boundary_sigma: float = 8.0          # Gaussian sigma (pixels) for soft boundary blend region


def sepvae_v3_loss(
    model,
    params,
    batch: Dict,
    key: jax.random.PRNGKey,
    cfg: SepVAEV3LossConfig,
    kl_anneal: Optional[jnp.ndarray] = None,
    disc_params=None,
    discriminator=None,
    ctr_adv_params=None,
    ctr_adversary=None,
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

    # Labels needed early for conditional KL and SupCon
    labels = batch.get("disease_labels")
    if labels is None:
        labels = jnp.concatenate([
            jnp.zeros((x_norm.shape[0],), dtype=jnp.int32),
            jnp.ones((x_cardio.shape[0],), dtype=jnp.int32),
        ])
    labels = labels.reshape(-1)

    # ── Reconstruction ────────────────────────────────────────────────────────
    # Soft-composite baseline: x_hat = (1-alpha)*x_common + alpha*x_heart
    # This trains both branches everywhere, causing the common branch to act as
    # a residual background inside the heart → ring artefact in x_heart.
    l_rec_soft = reconstruction_loss(x_01, x_hat)

    # Split reconstruction: each branch owns its region exclusively.
    # A Gaussian-blurred mask creates a soft boundary so the transition is
    # smooth rather than a hard seam.  split_rec_weight ramps 0→1 during warmup.
    soft_mask = _make_soft_mask(mask_full, cfg.boundary_sigma)
    l_rec_split = split_reconstruction_loss(
        x_01, x_common, x_heart, soft_mask, valid=valid_mask
    )
    split_w = jnp.float32(cfg.split_rec_weight)
    l_rec = (1.0 - split_w) * l_rec_soft + split_w * l_rec_split

    kl_common = jnp.mean(
        kl_divergence_standard(mu_c, logvar_c, free_bits=cfg.kl_free_bits)
    )
    if cfg.use_conditional_kl_heart:
        kl_heart = jnp.mean(
            kl_divergence_conditional(
                mu_h, logvar_h, labels, disease_id=1,
                sigma_inactive=cfg.sigma_inactive, free_bits=cfg.kl_free_bits,
            )
        )
    else:
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

    # ── V2 separation mechanisms ───────────────────────────────────────────
    z_common_pooled = jnp.mean(outputs["z_common_sample"], axis=(1, 2))
    z_heart_pooled  = jnp.mean(outputs["z_heart_sample"],  axis=(1, 2))

    if cfg.weight_mi_factor > 0.0 and disc_params is not None and discriminator is not None:
        l_mi = factor_vae_mi_loss(disc_params, discriminator, z_common_pooled, z_heart_pooled)
    else:
        l_mi = jnp.float32(0.0)

    mu_heart_pooled = jnp.mean(mu_h, axis=(1, 2))
    if cfg.weight_heart_supcon > 0.0:
        l_supcon = supervised_contrastive_loss(
            mu_heart_pooled, labels, temperature=cfg.supcon_temperature,
        )
    else:
        l_supcon = jnp.float32(0.0)

    if cfg.weight_ctr_adv > 0.0 and ctr_adv_params is not None and ctr_adversary is not None:
        l_ctr_adv = ctr_adv_confusion_loss(ctr_adv_params, ctr_adversary, z_heart_pooled)
    else:
        l_ctr_adv = jnp.float32(0.0)

    alpha_dice_metric, alpha_iou_metric = alpha_overlap_metrics(
        alpha_pred, mask_full, valid=valid_mask
    )

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
        + cfg.weight_mi_factor * l_mi
        + cfg.weight_heart_supcon * l_supcon
        + cfg.weight_ctr_adv * l_ctr_adv
    )

    z_common_norm = jnp.linalg.norm(z_common_pooled, axis=-1)
    z_heart_norm  = jnp.linalg.norm(z_heart_pooled,  axis=-1)
    mask_normal = (labels == 0).astype(jnp.float32)
    mask_cardio = (labels == 1).astype(jnp.float32)
    z_heart_norm_normal = jnp.sum(z_heart_norm * mask_normal) / jnp.maximum(jnp.sum(mask_normal), 1.0)
    z_heart_norm_cardio = jnp.sum(z_heart_norm * mask_cardio) / jnp.maximum(jnp.sum(mask_cardio), 1.0)

    logs = {
        "loss/total": total_loss,
        "loss/reconstruction": l_rec,
        "loss/rec_soft": l_rec_soft,
        "loss/rec_split": l_rec_split,
        "curriculum/split_rec_weight": split_w,
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
        "loss/mi_factor": l_mi,
        "loss/heart_supcon": l_supcon,
        "loss/ctr_adv_confuse": l_ctr_adv,
        "metrics/alpha_mean": jnp.mean(alpha_pred),
        "metrics/alpha_dice": alpha_dice_metric,
        "metrics/alpha_iou": alpha_iou_metric,
        "metrics/ctr_pred_mean": jnp.mean(outputs["ctr_pred"]),
        "metrics/ctr_mae": l_ctr,
        "metrics/z_common_norm_mean": jnp.mean(z_common_norm),
        "metrics/z_heart_norm_mean": jnp.mean(z_heart_norm),
        "metrics/z_heart_norm_normal": z_heart_norm_normal,
        "metrics/z_heart_norm_cardio": z_heart_norm_cardio,
    }
    return total_loss, logs, z_common_pooled, z_heart_pooled, x_hat
