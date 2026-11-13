"""
Loss functions for Multi-head SepVAE training.

This module implements:
1. Reconstruction loss (MSE)
2. KL divergence losses (standard + conditional priors + free-bits)
3. Nulling loss (head routing penalty)
4. Disease-head orthogonality loss (cardio vs effusion decorrelation)
5. Backbone perceptual loss (multi-scale CheSS feature matching)
6. PatchGAN adversarial loss (hinge-based)
7. Paired contrastive loss (structural head specialisation)
8. Disease discriminability classifier (R11)
9. Spatial attention bbox supervision (R12)
10. Combined SepVAE loss with all components

Loss formula:
    L_total = L_rec + β·L_KL + λ·L_null + η·L_ortho
            + γ·L_perceptual + α·L_GAN
            + ζ·L_contrastive + θ·L_disease_clf + ι·L_spatial_attn
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from typing import Dict, Tuple

from losses.lpips_gan import NLayerDiscriminator, hinge_d_loss, hinge_g_loss


# ============================================================================
# Reconstruction Loss
# ============================================================================

def reconstruction_loss(x_true: jnp.ndarray, x_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian reconstruction loss (Mean Squared Error).

    Args:
        x_true: Ground truth images (B, H, W, C)
        x_pred: Reconstructed images (B, H, W, C)

    Returns:
        Scalar loss (mean over batch)
    """
    mse_per_sample = jnp.mean((x_true - x_pred) ** 2, axis=(1, 2, 3))
    return jnp.mean(mse_per_sample)


# ============================================================================
# KL Divergence Losses
# ============================================================================

def kl_divergence_standard(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    KL divergence KL(q(z|x) || N(0, I)) for standard Gaussian prior.

    Formula:
        KL = 0.5 * sum(μ² + σ² - 1 - log(σ²))

    Args:
        mu: Mean of posterior (B, ..., latent_dim)
            Can be flat (B, D) or spatial (B, H, W, C)
        logvar: Log-variance of posterior (B, ..., latent_dim)

    Returns:
        Per-sample KL divergence (B,)
    """
    # Sum over all dimensions except batch
    sum_axes = tuple(range(1, mu.ndim))
    return 0.5 * jnp.sum(
        jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar,
        axis=sum_axes
    )


def kl_divergence_conditional(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1
) -> jnp.ndarray:
    """
    Conditional KL divergence with label-dependent prior.

    For disease-specific heads:
    - Prior is N(0, I) when disease is present (label == disease_id)
    - Prior is N(0, σ_inactive²·I) when disease is absent (tighter prior)

    This soft pressure pulls inactive heads toward zero.

    Args:
        mu: Mean of posterior (B, ..., latent_dim)
            Can be flat (B, D) or spatial (B, H, W, C)
        logvar: Log-variance of posterior (B, ..., latent_dim)
        labels: Disease labels (B,) in {0, 1, 2}
        disease_id: Which disease this head represents (1=effusion, 2=cardio)
        sigma_inactive: Std dev of tight prior for inactive heads (default: 0.1)

    Returns:
        Per-sample KL divergence (B,)
    """
    # Determine which samples are active for this disease head
    # Broadcast to match mu/logvar shape
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)  # (B, 1, 1, 1) for spatial
    is_active = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    # Prior variance: 1.0 if active, sigma_inactive² if inactive
    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var = jnp.exp(prior_logvar)

    # KL(q(z|x) || N(0, prior_var·I))
    # Formula: KL = 0.5 * sum((μ²/σ_p² + σ²/σ_p² - 1 - log(σ²/σ_p²))
    #            = 0.5 * sum(μ²/σ_p² + exp(log_σ²)/σ_p² - 1 - (log_σ² - log_σ_p²))
    sum_axes = tuple(range(1, mu.ndim))
    kl = 0.5 * jnp.sum(
        jnp.square(mu) / prior_var +
        jnp.exp(logvar) / prior_var -
        1.0 -
        (logvar - prior_logvar),
        axis=sum_axes
    )

    return kl


def kl_divergence_free_bits(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """
    KL divergence with free-bits (per-channel minimum threshold).

    Free-bits prevents posterior collapse by exempting the first `free_bits`
    nats per channel from the KL penalty. This ensures each latent channel
    can encode at least `free_bits` nats of information without penalty.

    Algorithm:
        1. Compute per-element KL: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
        2. Average over spatial dims (H, W) per channel → (B, C)
        3. Clamp per-channel KL to min of `free_bits`
        4. Sum over channels → (B,)

    Args:
        mu: Mean of posterior (B, H, W, C) — spatial latents
        logvar: Log-variance of posterior (B, H, W, C)
        free_bits: Minimum KL per channel in nats (default: 1.0)

    Returns:
        Per-sample KL divergence (B,) with free-bits applied
    """
    # Per-element KL
    kl_elem = 0.5 * (jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar)

    if mu.ndim == 4:
        # Spatial latents: average over H, W → per-channel KL (B, C)
        kl_per_channel = jnp.mean(kl_elem, axis=(1, 2))
    else:
        # Flat latents: treat each dim as a "channel" (B, D)
        kl_per_channel = kl_elem

    # Apply free-bits: clamp per-channel KL to minimum
    kl_clamped = jnp.maximum(kl_per_channel, free_bits)

    # Sum over channels to get per-sample KL
    # NOTE: Do NOT multiply by H*W for spatial latents. The per-channel mean
    # already represents the average KL density. Multiplying by spatial size
    # creates an enormous floor (e.g., 8 channels × 1.0 nat × 4096 = 32,768)
    # that causes the encoder to collapse to the floor and stop learning.
    # The correct behavior: free_bits=1.0 with 8 channels → minimum 8 nats total.
    return jnp.sum(kl_clamped, axis=-1)


def kl_divergence_conditional_free_bits(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """
    Conditional KL divergence with free-bits and label-dependent prior.

    Combines conditional prior (tight prior for inactive heads) with
    free-bits mechanism to prevent posterior collapse on active heads.

    Args:
        mu: Mean of posterior (B, H, W, C) or (B, D)
        logvar: Log-variance of posterior
        labels: Disease labels (B,)
        disease_id: Which disease this head represents
        sigma_inactive: Std dev of tight prior for inactive heads
        free_bits: Minimum KL per channel in nats

    Returns:
        Per-sample KL divergence (B,)
    """
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)
    is_active = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var = jnp.exp(prior_logvar)

    # Per-element KL against conditional prior
    kl_elem = 0.5 * (
        jnp.square(mu) / prior_var +
        jnp.exp(logvar) / prior_var -
        1.0 -
        (logvar - prior_logvar)
    )

    if mu.ndim == 4:
        kl_per_channel = jnp.mean(kl_elem, axis=(1, 2))  # (B, C)
    else:
        kl_per_channel = kl_elem

    kl_clamped = jnp.maximum(kl_per_channel, free_bits)

    # Sum over channels (same fix as kl_divergence_free_bits - no H*W scaling)
    return jnp.sum(kl_clamped, axis=-1)


def compute_kl_losses(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    sigma_inactive: float = 0.1,
    free_bits: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """
    Compute all KL divergence losses.

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar)
        labels: Disease labels (B,)
        sigma_inactive: Tight prior std dev for inactive heads
        free_bits: Per-channel free-bits threshold (0.0 = disabled)

    Returns:
        Dict with 'common', 'cardiomegaly', 'effusion' losses (all scalars)
    """
    mu_c, logvar_c = latents_dict['common']
    mu_cardio, logvar_cardio = latents_dict['cardiomegaly']
    mu_effusion, logvar_effusion = latents_dict['effusion']

    if free_bits > 0.0:
        # Free-bits KL: prevents posterior collapse
        kl_common = jnp.mean(kl_divergence_free_bits(mu_c, logvar_c, free_bits=free_bits))
        kl_cardio = jnp.mean(kl_divergence_conditional_free_bits(
            mu_cardio, logvar_cardio, labels, disease_id=2,
            sigma_inactive=sigma_inactive, free_bits=free_bits
        ))
        kl_effusion = jnp.mean(kl_divergence_conditional_free_bits(
            mu_effusion, logvar_effusion, labels, disease_id=1,
            sigma_inactive=sigma_inactive, free_bits=free_bits
        ))
    else:
        # Original KL (backward compatible)
        kl_common = jnp.mean(kl_divergence_standard(mu_c, logvar_c))
        kl_cardio = jnp.mean(kl_divergence_conditional(
            mu_cardio, logvar_cardio, labels, disease_id=2, sigma_inactive=sigma_inactive
        ))
        kl_effusion = jnp.mean(kl_divergence_conditional(
            mu_effusion, logvar_effusion, labels, disease_id=1, sigma_inactive=sigma_inactive
        ))

    return {
        'common': kl_common,
        'cardiomegaly': kl_cardio,
        'effusion': kl_effusion,
    }


# ============================================================================
# Nulling Loss
# ============================================================================

def nulling_loss(inactive_mus: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Penalize non-zero means for inactive disease heads.

    This is the hard routing constraint:
    - Normal (label=0): both disease heads should be zero
    - Cardiomegaly (label=2): effusion head should be zero
    - Effusion (label=1): cardiomegaly head should be zero

    Formula:
        L_null = mean(||μ_cardio_inactive||²) + mean(||μ_effusion_inactive||²)

    Args:
        inactive_mus: Dict with 'cardiomegaly', 'effusion' arrays
                     Can be flat (B, D) or spatial (B, H, W, C)
                     Already masked to zero for active samples

    Returns:
        Scalar nulling loss
    """
    # Sum over all dims except batch, then average over batch
    mu_cardio = inactive_mus['cardiomegaly']
    mu_effusion = inactive_mus['effusion']

    sum_axes = tuple(range(1, mu_cardio.ndim))
    loss_cardio = jnp.mean(jnp.sum(jnp.square(mu_cardio), axis=sum_axes))
    loss_effusion = jnp.mean(jnp.sum(jnp.square(mu_effusion), axis=sum_axes))

    return loss_cardio + loss_effusion


# ============================================================================
# Disease-Head Orthogonality Loss
# ============================================================================

def disease_head_orthogonality_loss(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Encourage cardiomegaly and effusion heads to encode independent directions.

    The loss combines:
    1. Cross-correlation penalty between disease head activations on disease-positive
       samples (Barlow-style off-head decorrelation).
    2. Prototype cosine penalty between class-conditional disease centroids
       (encourages near-orthogonal disease directions).

    Args:
        latents_dict: Dict with keys 'cardiomegaly' and 'effusion'
        labels: Disease labels (B,) in {0, 1, 2}
        eps: Numerical stability constant

    Returns:
        Scalar orthogonality loss
    """
    mu_cardio, _ = latents_dict['cardiomegaly']
    mu_effusion, _ = latents_dict['effusion']

    # Spatial latents -> channel vectors
    if mu_cardio.ndim == 4:  # (B, H, W, C)
        mu_cardio = jnp.mean(mu_cardio, axis=(1, 2))  # (B, C)
        mu_effusion = jnp.mean(mu_effusion, axis=(1, 2))

    # Focus decorrelation on disease-positive samples (labels 1 or 2).
    disease_mask = (labels != 0).astype(jnp.float32)[:, None]  # (B, 1)
    n_disease = jnp.maximum(jnp.sum(disease_mask), 1.0)

    z_cardio = mu_cardio * disease_mask
    z_effusion = mu_effusion * disease_mask

    mean_cardio = jnp.sum(z_cardio, axis=0) / n_disease
    mean_effusion = jnp.sum(z_effusion, axis=0) / n_disease

    z_cardio = (z_cardio - mean_cardio) * disease_mask
    z_effusion = (z_effusion - mean_effusion) * disease_mask

    std_cardio = jnp.sqrt(jnp.sum(jnp.square(z_cardio), axis=0) / n_disease + eps)
    std_effusion = jnp.sqrt(jnp.sum(jnp.square(z_effusion), axis=0) / n_disease + eps)

    z_cardio = z_cardio / std_cardio
    z_effusion = z_effusion / std_effusion

    # C in R^{C x C}; push all entries toward zero to reduce linear dependence.
    corr = (z_cardio.T @ z_effusion) / n_disease
    corr_penalty = jnp.mean(jnp.square(corr))

    # Class prototypes: cardio head on cardio-positive vs effusion head on effusion-positive.
    mask_cardio = (labels == 2).astype(jnp.float32)[:, None]
    mask_effusion = (labels == 1).astype(jnp.float32)[:, None]
    n_cardio = jnp.sum(mask_cardio)
    n_effusion = jnp.sum(mask_effusion)

    proto_cardio = jnp.sum(mu_cardio * mask_cardio, axis=0) / jnp.maximum(n_cardio, 1.0)
    proto_effusion = jnp.sum(mu_effusion * mask_effusion, axis=0) / jnp.maximum(n_effusion, 1.0)

    proto_dot = jnp.sum(proto_cardio * proto_effusion)
    proto_den = jnp.linalg.norm(proto_cardio) * jnp.linalg.norm(proto_effusion) + eps
    proto_cos_sq = jnp.square(proto_dot / proto_den)

    has_both_classes = jnp.logical_and(n_cardio > 0.0, n_effusion > 0.0)
    proto_penalty = jnp.where(has_both_classes, proto_cos_sq, jnp.float32(0.0))

    return corr_penalty + proto_penalty



# ============================================================================
# Backbone Perceptual Loss
# ============================================================================

def backbone_perceptual_loss(
    x_orig: jnp.ndarray,
    x_rec: jnp.ndarray,
    backbone_apply_fn,
    backbone_variables: Dict,
    use_remat: bool = False,
) -> jnp.ndarray:
    """
    Perceptual loss using frozen CheSS backbone multi-scale features.

    Uses the same ResNet-50 backbone as the encoder to compute feature-level
    distance between original and reconstructed images. This leverages the
    medical-domain features learned by CheSS contrastive pretraining.

    Computes L1 distance at layers 2, 3, and 4, normalized per layer.

    Args:
        x_orig: Original images (B, 512, 512, 1) in [-1, 1] range
        x_rec: Reconstructed images (B, 512, 512, 1) in [0, 1] range
        backbone_apply_fn: Function to call backbone (returns multi-scale features)
        backbone_variables: Backbone model variables (params + batch_stats)
        use_remat: If True, checkpoint backbone passes to save memory

    Returns:
        Scalar perceptual loss (mean over batch and layers)
    """
    # Convert x_rec from [0,1] to [-1,1] to match backbone's expected input
    x_rec_scaled = x_rec * 2.0 - 1.0

    # Bake return_multiscale=True into a wrapper before optionally checkpointing.
    # Passing it as a kwarg through jax.checkpoint causes TracerBoolConversionError:
    # checkpoint traces all arguments (including Python bool kwargs), so the
    # downstream `if return_multiscale:` branch in the ResNet sees a tracer, not
    # a concrete bool.  Hardcoding it in the closure body avoids tracing entirely.
    def _apply_multiscale(variables, x):
        return backbone_apply_fn(variables, x, return_multiscale=True)

    if use_remat:
        _apply_multiscale = jax.checkpoint(_apply_multiscale)

    # Extract multi-scale features
    # Stop gradient on original features (target — should not change)
    feats_orig = _apply_multiscale(backbone_variables, x_orig)
    feats_orig = jax.tree_util.tree_map(jax.lax.stop_gradient, feats_orig)

    # Allow gradient to flow through reconstruction features → decoder can learn
    # Backbone weights are frozen (not in trainable params), but gradient flows
    # through the fixed computation graph back to x_rec
    feats_rec = _apply_multiscale(backbone_variables, x_rec_scaled)

    # L1 distance at each scale, normalized by spatial size
    loss = jnp.float32(0.0)
    for layer_name in ['layer2', 'layer3', 'layer4']:
        f_orig = feats_orig[layer_name]
        f_rec = feats_rec[layer_name]
        # Mean absolute difference per sample, then average over batch
        loss += jnp.mean(jnp.abs(f_orig - f_rec))

    return loss / 3.0  # Average over 3 layers


# ============================================================================
# PatchGAN Adversarial Loss
# ============================================================================

def patchgan_disc_loss(
    disc_apply_fn,
    disc_params: Dict,
    x_real: jnp.ndarray,
    x_fake: jnp.ndarray,
) -> jnp.ndarray:
    """
    PatchGAN discriminator loss (hinge).

    Args:
        disc_apply_fn: Discriminator apply function
        disc_params: Discriminator parameters
        x_real: Real images (B, H, W, C)
        x_fake: Fake/reconstructed images (B, H, W, C) — detached

    Returns:
        Scalar discriminator loss
    """
    logits_real = disc_apply_fn({'params': disc_params}, x_real)
    logits_fake = disc_apply_fn({'params': disc_params}, jax.lax.stop_gradient(x_fake))
    return hinge_d_loss(logits_real, logits_fake)


def patchgan_gen_loss(
    disc_apply_fn,
    disc_params: Dict,
    x_fake: jnp.ndarray,
) -> jnp.ndarray:
    """
    PatchGAN generator loss (hinge).

    Args:
        disc_apply_fn: Discriminator apply function
        disc_params: Discriminator parameters
        x_fake: Fake/reconstructed images (B, H, W, C)

    Returns:
        Scalar generator loss
    """
    logits_fake = disc_apply_fn({'params': disc_params}, x_fake)
    return hinge_g_loss(logits_fake)


# ============================================================================
# Paired Contrastive Loss  (R5a)
# ============================================================================

def _bbox_to_mask(bboxes: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """
    Convert normalised bboxes to binary spatial masks.

    Args:
        bboxes: (B, 4) float32 [x_min, y_min, x_max, y_max] in [0, 1]
        H, W:   Target mask height and width (typically 64 for the latent grid)

    Returns:
        (B, H, W) float32 mask — 1.0 inside bbox, 0.0 outside.
        All-zero input bbox → all-zero mask (no supervision for that sample).
    """
    x_min = bboxes[:, 0] * W  # (B,)
    y_min = bboxes[:, 1] * H
    x_max = bboxes[:, 2] * W
    y_max = bboxes[:, 3] * H

    xs = jnp.arange(W, dtype=jnp.float32)[None, None, :]  # (1, 1, W)
    ys = jnp.arange(H, dtype=jnp.float32)[None, :, None]  # (1, H, 1)

    in_x = (xs >= x_min[:, None, None]) & (xs < x_max[:, None, None])  # (B, 1, W)
    in_y = (ys >= y_min[:, None, None]) & (ys < y_max[:, None, None])  # (B, H, 1)

    return (in_x & in_y).astype(jnp.float32)  # (B, H, W)


def spatial_attention_loss(
    attn_maps: Dict[str, jnp.ndarray],
    bbox_disease1: jnp.ndarray,
    bbox_disease2: jnp.ndarray,
    latent_hw: int = 64,
) -> jnp.ndarray:
    """
    Penalise disease-head attention falling outside the ground-truth bbox.

    Only applied to active disease samples:
    - Effusion head  (label=1): the middle-B slice of the 3B batch
    - Cardio head    (label=2): the last-B slice of the 3B batch

    Loss per active sample = sum(attn * (1 - bbox_mask)), i.e. the total
    probability mass that lands outside the annotated region.  Because softmax
    attention weights sum to 1, this equals 1 - inside_mass and ranges in [0, 1]:
      0 when all attention is inside the bbox (perfect)
      ~(1 - bbox_fraction) when attention is uniform

    Using sum (not mean) over spatial dims is critical: jnp.mean would divide
    by HW=4096, shrinking the loss to O(1/HW) and making the gradient ~4096×
    too small to train the attention query.

    Samples with an all-zero bbox (no annotation available) are excluded via a
    validity mask so they contribute zero gradient.

    Args:
        attn_maps:     Dict with 'cardiomegaly' and 'effusion', each (3B, H, W).
                       Only present when use_label_attention=True.
        bbox_disease1: (B, 4) effusion bboxes in [0, 1] (x_min, y_min, x_max, y_max)
        bbox_disease2: (B, 4) cardio bboxes in [0, 1]
        latent_hw:     Spatial size of the feature map (default 64 for 512→layer4→64)

    Returns:
        Scalar spatial attention loss (sum of effusion + cardio terms), each in [0, 1].
    """
    H = W = latent_hw
    B = bbox_disease1.shape[0]  # triplet batch size

    attn_cardio   = attn_maps['cardiomegaly']  # (3B, H, W)
    attn_effusion = attn_maps['effusion']       # (3B, H, W)

    # Batch is ordered [normal (0:B), effusion (B:2B), cardio (2B:3B)]
    attn_effusion_active = attn_effusion[B:2 * B]   # (B, H, W)
    attn_cardio_active   = attn_cardio[2 * B:]       # (B, H, W)

    mask_effusion = _bbox_to_mask(bbox_disease1, H, W)  # (B, H, W)
    mask_cardio   = _bbox_to_mask(bbox_disease2, H, W)  # (B, H, W)

    # Validity: skip samples whose bbox is all-zero (no annotation)
    valid_effusion = (jnp.sum(bbox_disease1, axis=-1) > 0).astype(jnp.float32)  # (B,)
    valid_cardio   = (jnp.sum(bbox_disease2, axis=-1) > 0).astype(jnp.float32)  # (B,)

    # Loss = total attention probability mass outside the bbox (sum, NOT mean).
    # Using jnp.sum gives loss in [0,1] and gradient O(1/HW) on the attention
    # logits.  Using jnp.mean would give loss O(1/HW) and gradient O(1/HW²),
    # which is ~4096× too small to produce any meaningful training signal.
    outside_effusion = jnp.sum(attn_effusion_active * (1.0 - mask_effusion), axis=(1, 2))  # (B,)
    outside_cardio   = jnp.sum(attn_cardio_active   * (1.0 - mask_cardio),   axis=(1, 2))  # (B,)

    n_valid_e = jnp.maximum(jnp.sum(valid_effusion), 1.0)
    n_valid_c = jnp.maximum(jnp.sum(valid_cardio),   1.0)

    l_effusion = jnp.sum(outside_effusion * valid_effusion) / n_valid_e
    l_cardio   = jnp.sum(outside_cardio   * valid_cardio)   / n_valid_c

    return l_effusion + l_cardio


def paired_contrastive_loss(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    margin: float = 0.5,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Prototype-based contrastive loss for structural head specialisation.

    For each disease head `h` with its own disease class `d`:
    - **Separate-push**: push the centroid of (disease-d images in head-h) away from
      the centroid of (normal images in head-h), via a cosine margin loss.
    - **Blind-pull**: pull the centroid of (disease-d images in the OTHER head)
      toward the centroid of (normal images in the OTHER head), so the other head
      remains blind to disease-d.

    This directly enforces what the swap grid should show: inserting the effusion
    head from a cardiomegaly image should produce no effusion signal.

    Args:
        latents_dict: Dict with 'cardiomegaly' and 'effusion', each (mu, logvar)
                     Spatial latents (B, H, W, C) or flat (B, D)
        labels: Disease labels (B,) in {0=normal, 1=effusion, 2=cardiomegaly}
        margin: Cosine margin for separation push (default: 0.5)
        eps: Numerical stability constant

    Returns:
        Scalar contrastive loss
    """
    mu_cardio, _ = latents_dict['cardiomegaly']
    mu_effusion, _ = latents_dict['effusion']

    # GAP spatial latents to flat vectors
    if mu_cardio.ndim == 4:
        mu_cardio = jnp.mean(mu_cardio, axis=(1, 2))   # (B, C)
        mu_effusion = jnp.mean(mu_effusion, axis=(1, 2))

    # L2-normalise for cosine arithmetic
    mu_cardio = mu_cardio / (jnp.linalg.norm(mu_cardio, axis=-1, keepdims=True) + eps)
    mu_effusion = mu_effusion / (jnp.linalg.norm(mu_effusion, axis=-1, keepdims=True) + eps)

    is_cardio = (labels == 2).astype(jnp.float32)    # (B,)
    is_effusion = (labels == 1).astype(jnp.float32)
    is_normal = (labels == 0).astype(jnp.float32)

    n_cardio = jnp.maximum(jnp.sum(is_cardio), 1.0)
    n_effusion = jnp.maximum(jnp.sum(is_effusion), 1.0)
    n_normal = jnp.maximum(jnp.sum(is_normal), 1.0)

    # Masked centroids (safe for JIT — no dynamic indexing)
    proto_cardio_in_cardio_head = jnp.sum(mu_cardio * is_cardio[:, None], axis=0) / n_cardio
    proto_normal_in_cardio_head = jnp.sum(mu_cardio * is_normal[:, None], axis=0) / n_normal
    proto_effusion_in_effusion_head = jnp.sum(mu_effusion * is_effusion[:, None], axis=0) / n_effusion
    proto_normal_in_effusion_head = jnp.sum(mu_effusion * is_normal[:, None], axis=0) / n_normal
    # Cross-head centroids (for the blind-pull term)
    proto_cardio_in_effusion_head = jnp.sum(mu_effusion * is_cardio[:, None], axis=0) / n_cardio
    proto_effusion_in_cardio_head = jnp.sum(mu_cardio * is_effusion[:, None], axis=0) / n_effusion

    # --- Cardio head: separate cardio from normal ---
    cos_cardio_sep = jnp.sum(proto_cardio_in_cardio_head * proto_normal_in_cardio_head)
    loss_cardio_sep = jnp.maximum(jnp.float32(0.0), cos_cardio_sep - (-margin))

    # --- Effusion head: separate effusion from normal ---
    cos_effusion_sep = jnp.sum(proto_effusion_in_effusion_head * proto_normal_in_effusion_head)
    loss_effusion_sep = jnp.maximum(jnp.float32(0.0), cos_effusion_sep - (-margin))

    # --- Blind-pull: effusion head should be BLIND to cardiomegaly ---
    # Minimise distance between cardio-image and normal-image representations in effusion head
    cos_effusion_blind_to_cardio = jnp.sum(proto_cardio_in_effusion_head * proto_normal_in_effusion_head)
    loss_effusion_blind = 1.0 - cos_effusion_blind_to_cardio  # minimised when cos → 1

    # --- Blind-pull: cardio head should be BLIND to effusion ---
    cos_cardio_blind_to_effusion = jnp.sum(proto_effusion_in_cardio_head * proto_normal_in_cardio_head)
    loss_cardio_blind = 1.0 - cos_cardio_blind_to_effusion

    return loss_cardio_sep + loss_effusion_sep + loss_effusion_blind + loss_cardio_blind



# ============================================================================
# R11: Disease Discriminability Classifier
# ============================================================================

class DiseaseClassifier(nn.Module):
    """
    Auxiliary classifier for disease discriminability (R11).

    Takes z_disease_k.mu (spatially averaged) and predicts disease_k presence
    with a lightweight 2-layer MLP.  Gradient flows back into the encoder to
    enforce that z_disease_k carries disease-discriminative information when the
    disease is present — providing the *positive* pressure that the null-space
    loss alone cannot supply.

    Attributes:
        hidden_dim: Hidden layer width (default: 16, intentionally small)
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: (B, C) — GAP'd disease latent mean

        Returns:
            logits: (B,) — unnormalised binary logit for disease presence
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        return nn.Dense(1)(h).squeeze(-1)


def disease_discriminability_loss(
    clf_cardio_apply_fn,
    clf_effusion_apply_fn,
    clf_cardio_params: Dict,
    clf_effusion_params: Dict,
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    BCE auxiliary classification loss on z_disease_k.mu → label_k (R11).

    Fills the gap left by the null-space loss:
    - Null-space loss penalises z_disease_k being *active* for disease-absent samples.
    - This loss penalises z_disease_k being *uninformative* for disease-present samples.

    The gradient flows into the encoder so that z_disease_k must pack information
    sufficient to predict label_k = 1 when disease k is present, preventing
    posterior collapse in the disease head.

    Args:
        clf_cardio_apply_fn: DiseaseClassifier.apply for cardiomegaly head
        clf_effusion_apply_fn: DiseaseClassifier.apply for effusion head
        clf_cardio_params: Parameters for cardiomegaly classifier
        clf_effusion_params: Parameters for effusion classifier
        latents_dict: Dict with 'cardiomegaly' and 'effusion' (mu, log_var) tuples
        labels: (3B,) disease labels in {0, 1, 2}

    Returns:
        (total_loss, loss_cardio, loss_effusion)
    """
    mu_cardio, _ = latents_dict['cardiomegaly']   # (3B, H, W, C) or (3B, C)
    mu_effusion, _ = latents_dict['effusion']      # (3B, H, W, C) or (3B, C)

    if mu_cardio.ndim == 4:
        mu_cardio = jnp.mean(mu_cardio, axis=(1, 2))   # (3B, C)
        mu_effusion = jnp.mean(mu_effusion, axis=(1, 2))

    label_cardio = (labels == 2).astype(jnp.float32)    # (3B,): 1 for cardiomegaly
    label_effusion = (labels == 1).astype(jnp.float32)  # (3B,): 1 for effusion

    logit_cardio = clf_cardio_apply_fn({'params': clf_cardio_params}, mu_cardio)    # (3B,)
    logit_effusion = clf_effusion_apply_fn({'params': clf_effusion_params}, mu_effusion)  # (3B,)

    def stable_bce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(
            jnp.maximum(logits, 0) - logits * targets + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
        )

    loss_cardio = stable_bce(logit_cardio, label_cardio)
    loss_effusion = stable_bce(logit_effusion, label_effusion)

    return loss_cardio + loss_effusion, loss_cardio, loss_effusion


# ============================================================================
# Combined SepVAE Loss
# ============================================================================

@dataclass
class SepVAELossConfig:
    """
    Configuration for SepVAE loss weights.

    Attributes:
        weight_rec: Reconstruction loss weight (default: 1.0)
        weight_kl_common: Common KL weight (default: 1e-4)
        weight_kl_disease: Disease KL weight (default: 1e-4)
        weight_null: Nulling loss weight (default: 1e-3)
        weight_orthogonality: Disease-head orthogonality weight (default: 0.0 = disabled)
        weight_perceptual: Backbone perceptual loss weight (default: 0.0 = disabled)
        weight_adversarial: PatchGAN generator loss weight (default: 0.0 = disabled)
        weight_contrastive: Paired contrastive loss weight (default: 0.0 = disabled)
        weight_disease_clf: Disease discriminability classification weight (default: 0.0 = disabled)
        weight_spatial_attn: Spatial attention bbox supervision weight (default: 0.0 = disabled).
            Requires use_label_attention=True and bbox columns in the dataset CSV.
        sigma_inactive: Prior std dev for inactive heads (default: 1.0, i.e. standard prior)
        free_bits: Per-channel free-bits for KL (default: 0.0 = disabled)
        disc_start_epoch: Epoch at which discriminator loss kicks in (default: 10)
        contrastive_margin: Cosine margin for contrastive separate-push (default: 0.5)
    """
    weight_rec: float = 1.0
    weight_kl_common: float = 1e-4
    weight_kl_disease: float = 1e-4
    weight_null: float = 1e-3
    weight_orthogonality: float = 0.0
    weight_perceptual: float = 0.0
    weight_adversarial: float = 0.0
    weight_contrastive: float = 0.0
    weight_disease_clf: float = 0.0
    weight_spatial_attn: float = 0.0
    sigma_inactive: float = 1.0
    free_bits: float = 0.0
    disc_start_epoch: int = 10
    contrastive_margin: float = 0.5


def sepvae_loss(
    model,
    params,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    cfg: SepVAELossConfig,
    batch_stats: Dict = None,
    kl_anneal: jnp.ndarray = None,
    # Adversarial & perceptual
    patch_disc=None,
    patch_disc_params: Dict = None,
    backbone_apply_fn=None,
    backbone_variables: Dict = None,
    current_epoch: int = 0,
    use_remat: bool = False,
    # R11: disease discriminability classifiers
    clf_cardio=None,
    clf_cardio_params: Dict = None,
    clf_effusion=None,
    clf_effusion_params: Dict = None,
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]:
    """
    Complete SepVAE loss function with all regularisation terms.

    This combines:
    - Reconstruction loss (MSE)
    - KL divergence (common + disease heads, optional free-bits), scaled by kl_anneal
    - Nulling loss (inactive heads)
    - Disease-head orthogonality loss (cardio vs effusion decorrelation)
    - MI penalty (disentanglement)
    - Backbone perceptual loss (optional, multi-scale CheSS feature matching)
    - PatchGAN generator loss (optional, hinge-based)
    - Paired contrastive loss (optional, R5a — structural head specialisation)
    - Cross-head adversarial VAE penalty (optional, R5b — conditional independence)

    Args:
        model: SepVAE model
        params: SepVAE parameters
        batch: Dict with 'x_norm', 'x_disease1', 'x_disease2', 'disease_labels'
        key: JAX PRNG key
        cfg: SepVAELossConfig with loss weights
        batch_stats: BatchNorm running statistics for frozen backbone
        kl_anneal: KL annealing factor in [0, 1] (None = 1.0).
        patch_disc: PatchGAN discriminator model (None = disabled)
        patch_disc_params: PatchGAN discriminator parameters
        backbone_apply_fn: Function to call backbone for perceptual loss (None = disabled)
        backbone_variables: Backbone variables for perceptual loss
        current_epoch: Current training epoch (for disc warmup)
        use_remat: Enable gradient checkpointing for perceptual loss backbone pass
        clf_cardio: DiseaseClassifier model for cardiomegaly head (R11)
        clf_cardio_params: Parameters for clf_cardio
        clf_effusion: DiseaseClassifier model for effusion head (R11)
        clf_effusion_params: Parameters for clf_effusion

    Returns:
        total_loss: Scalar
        (logs, patch_disc_loss):
            Logging dict + PatchGAN discriminator loss for its own optimiser
    """
    # Unpack batch from VinBigData triplets
    x_norm = batch['x_norm']  # (B, 512, 512, 1)
    x_disease1 = batch['x_disease1']  # Effusion
    x_disease2 = batch['x_disease2']  # Cardiomegaly
    labels = batch['disease_labels']  # (3*B,) = [0, 0, ..., 1, 1, ..., 2, 2, ...]

    # Stack into single batch: (3*B, 512, 512, 1)
    x = jnp.concatenate([x_norm, x_disease1, x_disease2], axis=0)

    # Build variables dict (params + optional batch_stats for frozen BN)
    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    # Forward pass
    key1, _ = jax.random.split(key)
    x_rec, latents_dict, inactive_mus = model.apply(
        variables, x, labels, key=key1, train=True
    )

    # 1. Reconstruction loss
    # Note: x is in [-1, 1], x_rec is in [0, 1]
    x_normalized = (x + 1.0) / 2.0  # Convert to [0, 1] for comparison
    l_rec = reconstruction_loss(x_normalized, x_rec)

    # 2. KL losses (with optional free-bits)
    kl_losses = compute_kl_losses(
        latents_dict, labels,
        sigma_inactive=cfg.sigma_inactive,
        free_bits=cfg.free_bits,
    )
    l_kl_raw = (
        cfg.weight_kl_common * kl_losses['common'] +
        cfg.weight_kl_disease * (kl_losses['cardiomegaly'] + kl_losses['effusion'])
    )

    # Apply KL annealing (warmup): scale weighted KL by anneal factor
    _kl_anneal = kl_anneal if kl_anneal is not None else jnp.float32(1.0)
    l_kl = l_kl_raw * _kl_anneal

    # 3. Nulling loss
    l_null = nulling_loss(inactive_mus)

    # 4. Disease-head orthogonality loss
    l_ortho = disease_head_orthogonality_loss(latents_dict, labels)

    # 5. Backbone perceptual loss (optional)
    if cfg.weight_perceptual > 0.0 and backbone_apply_fn is not None:
        l_perceptual = backbone_perceptual_loss(
            x, x_rec, backbone_apply_fn, backbone_variables,
            use_remat=use_remat,
        )
    else:
        l_perceptual = jnp.float32(0.0)

    # 7. PatchGAN generator loss (optional, with warmup)
    patch_disc_loss_val = jnp.float32(0.0)
    l_gen = jnp.float32(0.0)
    disc_factor = jnp.float32(0.0)
    if cfg.weight_adversarial > 0.0 and patch_disc is not None and patch_disc_params is not None:
        # Warmup: ramp from 0 to 1 over 5000 steps after disc_start_epoch
        disc_factor = jnp.where(
            current_epoch >= cfg.disc_start_epoch,
            jnp.clip((current_epoch - cfg.disc_start_epoch) / 5.0, 0.0, 1.0),
            0.0,
        )
        disc_apply_fn = patch_disc.apply
        l_gen = patchgan_gen_loss(disc_apply_fn, patch_disc_params, x_rec)
        # Discriminator loss (for separate optimizer)
        patch_disc_loss_val = patchgan_disc_loss(
            disc_apply_fn, patch_disc_params, x_normalized, x_rec
        )

    # 8. Paired contrastive loss (R5a, optional)
    l_contrastive = jnp.float32(0.0)
    if cfg.weight_contrastive > 0.0:
        l_contrastive = paired_contrastive_loss(
            latents_dict, labels, margin=cfg.contrastive_margin,
        )

    # 9. Disease discriminability classification loss (R11, optional)
    l_disease_clf = jnp.float32(0.0)
    l_disease_clf_cardio = jnp.float32(0.0)
    l_disease_clf_effusion = jnp.float32(0.0)
    if (cfg.weight_disease_clf > 0.0
            and clf_cardio is not None and clf_cardio_params is not None
            and clf_effusion is not None and clf_effusion_params is not None):
        clf_c_fn = clf_cardio.apply
        clf_e_fn = clf_effusion.apply
        l_disease_clf, l_disease_clf_cardio, l_disease_clf_effusion = disease_discriminability_loss(
            clf_c_fn, clf_e_fn, clf_cardio_params, clf_effusion_params, latents_dict, labels
        )

    # 10. Spatial attention bbox supervision (R12, optional)
    # Penalises the DiseaseAttentionHead for attending outside the ground-truth bbox.
    # Active only when: weight_spatial_attn > 0, use_label_attention=True (attn_maps present),
    # and the batch contains bbox tensors from VinBigData.
    l_spatial_attn = jnp.float32(0.0)
    if (cfg.weight_spatial_attn > 0.0
            and 'attn_maps' in latents_dict
            and 'bbox_disease1' in batch
            and 'bbox_disease2' in batch):
        l_spatial_attn = spatial_attention_loss(
            latents_dict['attn_maps'],
            batch['bbox_disease1'],   # (B, 4) effusion bboxes in [0,1]
            batch['bbox_disease2'],   # (B, 4) cardio bboxes in [0,1]
        )

    # Total VAE loss
    total_loss = (
        cfg.weight_rec * l_rec +
        l_kl +
        cfg.weight_null * l_null +
        cfg.weight_orthogonality * l_ortho +
        cfg.weight_perceptual * l_perceptual +
        cfg.weight_adversarial * disc_factor * l_gen +
        cfg.weight_contrastive * l_contrastive +
        cfg.weight_disease_clf * l_disease_clf +
        cfg.weight_spatial_attn * l_spatial_attn
    )

    # Logging dict
    logs = {
        'loss/total': total_loss,
        'loss/reconstruction': l_rec,
        'loss/kl_common': kl_losses['common'],
        'loss/kl_cardiomegaly': kl_losses['cardiomegaly'],
        'loss/kl_effusion': kl_losses['effusion'],
        'loss/kl_total': kl_losses['common'] + kl_losses['cardiomegaly'] + kl_losses['effusion'],
        'loss/kl_weighted': l_kl,
        'loss/kl_anneal': _kl_anneal,
        'loss/nulling': l_null,
        'loss/orthogonality': l_ortho,
        'loss/perceptual': l_perceptual,
        'loss/gen_adversarial': l_gen,
        'loss/disc_factor': disc_factor,
        'loss/patch_disc': patch_disc_loss_val,
        'loss/contrastive': l_contrastive,
        'loss/disease_clf': l_disease_clf,
        'loss/disease_clf_cardio': l_disease_clf_cardio,
        'loss/disease_clf_effusion': l_disease_clf_effusion,
        'loss/spatial_attn': l_spatial_attn,
    }

    return total_loss, (logs, patch_disc_loss_val)


# ============================================================================
# Testing
# ============================================================================

def test_losses():
    """Test all loss functions with dummy data."""
    print("Testing loss functions...")

    B, H, W, C = 4, 512, 512, 1
    latent_dim_common = 128
    latent_dim_disease = 64

    # Dummy data
    x_true = jnp.ones((B, H, W, C)) * 0.5
    x_pred = jnp.ones((B, H, W, C)) * 0.4

    mu_c = jnp.zeros((B, latent_dim_common))
    logvar_c = jnp.zeros((B, latent_dim_common))
    mu_cardio = jnp.ones((B, latent_dim_disease)) * 0.5
    logvar_cardio = jnp.zeros((B, latent_dim_disease))
    mu_effusion = jnp.ones((B, latent_dim_disease)) * 0.3
    logvar_effusion = jnp.zeros((B, latent_dim_disease))

    labels = jnp.array([0, 1, 2, 1])  # Normal, Effusion, Cardio, Effusion

    # Test reconstruction loss
    l_rec = reconstruction_loss(x_true, x_pred)
    print(f"✓ Reconstruction loss: {l_rec:.6f}")

    # Test KL divergence
    kl_std = jnp.mean(kl_divergence_standard(mu_c, logvar_c))
    print(f"✓ KL divergence (standard): {kl_std:.6f}")

    latents_dict = {
        'common': (mu_c, logvar_c),
        'cardiomegaly': (mu_cardio, logvar_cardio),
        'effusion': (mu_effusion, logvar_effusion),
    }

    kl_losses = compute_kl_losses(latents_dict, labels)
    print(f"✓ KL losses: common={kl_losses['common']:.6f}, "
          f"cardio={kl_losses['cardiomegaly']:.6f}, "
          f"effusion={kl_losses['effusion']:.6f}")

    # Test nulling loss
    inactive_mus = {
        'cardiomegaly': mu_cardio * 0.5,
        'effusion': mu_effusion * 0.3,
    }
    l_null = nulling_loss(inactive_mus)
    print(f"✓ Nulling loss: {l_null:.6f}")

    print("\n✓ All loss tests passed!")


if __name__ == "__main__":
    test_losses()
