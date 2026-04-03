"""
Loss functions for Binary SepVAE (Normal vs. Cardiomegaly).

Five targeted losses — each maps directly to a research objective:

  Objective 1 — orthogonal latent separation:
    • KL_common    (standard prior on z_common)
    • KL_cardio    (conditional tight prior on z_cardio when inactive)
    • L_mi_factor  (FactorVAE-style MI: push q(z_c,z_ca) → q(z_c)×q(z_ca))
    • L_bbox_attn  (force cardio attention head to the cardiac silhouette)

  Objective 2 — crisp VAE reconstructions:
    • L_rec        (MSE — hard-zero nulling makes this a clean signal)
    • L_perceptual (L1 in frozen CheSS feature space, layers 1–3 — zero extra params)
    • L_gan        (hinge generator loss vs frozen PatchGAN — sharpens textures)
    • L_tv         (anisotropic total variation — suppresses stripe/grid artifacts)

  Total:
    L_vae = L_rec + β_c·KL_c + β_d·KL_ca + κ·L_mi + γ·L_perceptual
          + λ·L_bbox + α·L_gan + τ·L_tv

  Discriminators (trained alternately, frozen during L_vae):
    L_factor_disc = BCE(D(z_c, z_ca), 1) + BCE(D(z_c, z_ca[perm]), 0)
    L_patch_disc  = hinge(D_patch(x_real), D_patch(x_rec_stale))
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from typing import Dict, Tuple, Optional


# ============================================================================
# Reconstruction Loss
# ============================================================================

def reconstruction_loss(x_true: jnp.ndarray, x_pred: jnp.ndarray) -> jnp.ndarray:
    """MSE reconstruction loss (mean over batch and pixels)."""
    return jnp.mean(jnp.mean((x_true - x_pred) ** 2, axis=(1, 2, 3)))


def masked_anatomy_reconstruction_loss(
    x_true_01: jnp.ndarray,
    x_rec_anatomy: jnp.ndarray,
    bboxes_cardio: jnp.ndarray,
    B: int,
) -> jnp.ndarray:
    """
    MSE loss on pixels OUTSIDE the heart bbox, for Cardiomegaly images only.

    Forces z_common to reconstruct the non-cardiac region faithfully without
    help from z_cardio.  If z_common can minimise this loss, it proves the
    shared head has not absorbed cardiac-shape information — the complementary
    half of what bbox_attention_loss guarantees for z_cardio.

    Batch ordering: [Normal(B), Cardiomegaly(B)] = 2B total.

    Args:
        x_true_01:    (2B, H, W, 1) ground truth in [0, 1]
        x_rec_anatomy:(2B, H, W, 1) decoder output with z_cardio=0
        bboxes_cardio:(B, 4)        [x0, y0, x1, y1] normalised [0,1]
        B:            per-class batch size

    Returns:
        scalar: mean outside-bbox MSE over valid Cardiomegaly images
    """
    H, W = x_true_01.shape[1], x_true_01.shape[2]

    y_lin = (jnp.arange(H, dtype=jnp.float32) + 0.5) / H   # (H,)
    x_lin = (jnp.arange(W, dtype=jnp.float32) + 0.5) / W   # (W,)

    x0 = bboxes_cardio[:, 0][:, None, None]   # (B, 1, 1)
    y0 = bboxes_cardio[:, 1][:, None, None]
    x1 = bboxes_cardio[:, 2][:, None, None]
    y1 = bboxes_cardio[:, 3][:, None, None]

    inside = (
        (x_lin[None, None, :] >= x0) & (x_lin[None, None, :] <= x1) &
        (y_lin[None, :, None] >= y0) & (y_lin[None, :, None] <= y1)
    ).astype(jnp.float32)   # (B, H, W)
    outside = 1.0 - inside  # (B, H, W)

    # Only Cardiomegaly images (indices B:2B) carry meaningful bboxes
    x_true_cardio = x_true_01[B:, :, :, 0]      # (B, H, W)
    x_rec_cardio  = x_rec_anatomy[B:, :, :, 0]  # (B, H, W)

    sq_err = jnp.square(x_true_cardio - x_rec_cardio)  # (B, H, W)

    # Skip images with missing/invalid bboxes
    has_bbox = ((bboxes_cardio[:, 2] - bboxes_cardio[:, 0]) > 1e-4).astype(jnp.float32)
    n_valid  = jnp.maximum(jnp.sum(has_bbox), 1.0)

    outside_px_count = jnp.maximum(jnp.sum(outside, axis=(1, 2)), 1.0)  # (B,)
    outside_mse      = jnp.sum(sq_err * outside, axis=(1, 2)) / outside_px_count  # (B,)
    return jnp.sum(outside_mse * has_bbox) / n_valid


def total_variation_loss(x: jnp.ndarray) -> jnp.ndarray:
    """
    Anisotropic total variation loss — penalises abrupt intensity changes
    between neighbouring pixels in both spatial directions.

    Directly suppresses the horizontal stripe artifacts that emerge when a
    strided perceptual backbone (CheSS / ResNet-50) injects its stride-aliased
    gradients into the decoder.

    Args:
        x: (B, H, W, C) in [0, 1]
    Returns:
        scalar mean TV over batch.
    """
    diff_h = x[:, 1:, :, :] - x[:, :-1, :, :]   # row-to-row differences
    diff_w = x[:, :, 1:, :] - x[:, :, :-1, :]   # col-to-col differences
    return jnp.mean(jnp.abs(diff_h)) + jnp.mean(jnp.abs(diff_w))


# ============================================================================
# KL Divergence Losses
# ============================================================================

def kl_divergence_standard(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    free_bits: float = 0.0,
) -> jnp.ndarray:
    """KL(q(z|x) || N(0,I)) — per-sample, summed over latent dims.

    free_bits > 0: clamp per-dimension KL to at least this value before
    summing.  This floors inactive dims (preventing collapse) while also
    capping the gradient of runaway dims when they are already above the
    floor — together bounding the step-to-step KL variance that causes the
    33 → 11,600 spikes observed without this guard.
    """
    kl_per_dim = 0.5 * (jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar)
    free_bits_arr = jnp.asarray(free_bits, dtype=kl_per_dim.dtype)
    kl_per_dim = jnp.where(
        free_bits_arr > 0.0,
        jnp.maximum(kl_per_dim, free_bits_arr),
        kl_per_dim,
    )
    sum_axes = tuple(range(1, mu.ndim))
    return jnp.sum(kl_per_dim, axis=sum_axes)


def kl_divergence_conditional(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1,
    free_bits: float = 0.0,
) -> jnp.ndarray:
    """
    Conditional KL with label-dependent prior:
    - Active (disease present):  prior = N(0, I)
    - Inactive (disease absent): prior = N(0, sigma_inactive²·I)  — tight

    For inactive samples (Normal images, label≠disease_id) this reduces to:
        0.5 * (μ²/σ_p² + σ_q²/σ_p² - 1 - log(σ_q²/σ_p²))
    which strongly penalises μ away from zero, complementing hard-zero nulling.

    free_bits > 0: per-dim KL floor applied after the prior-weighted computation.
    For the tight inactive prior (σ_p=0.1), most dims naturally exceed any
    reasonable free_bits threshold — the floor mainly guards the active prior
    (σ_p=1) where dims can collapse to near-zero KL.
    """
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)
    is_active  = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var    = jnp.exp(prior_logvar)

    kl_per_dim = 0.5 * (
        jnp.square(mu) / prior_var + jnp.exp(logvar) / prior_var
        - 1.0 - (logvar - prior_logvar)
    )
    free_bits_arr = jnp.asarray(free_bits, dtype=kl_per_dim.dtype)
    kl_per_dim = jnp.where(
        free_bits_arr > 0.0,
        jnp.maximum(kl_per_dim, free_bits_arr),
        kl_per_dim,
    )
    sum_axes = tuple(range(1, mu.ndim))
    return jnp.sum(kl_per_dim, axis=sum_axes)


def compute_kl_losses(
    latents_dict: Dict,
    labels: jnp.ndarray,
    sigma_inactive: float = 0.1,
    free_bits: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """Batch-mean KL for common and cardiomegaly heads (binary)."""
    mu_c,  logvar_c  = latents_dict['common']
    mu_ca, logvar_ca = latents_dict['cardiomegaly']
    return {
        'common':       jnp.mean(kl_divergence_standard(mu_c, logvar_c, free_bits=free_bits)),
        'cardiomegaly': jnp.mean(kl_divergence_conditional(
            mu_ca, logvar_ca, labels, disease_id=1,
            sigma_inactive=sigma_inactive, free_bits=free_bits)),
    }


# ============================================================================
# FactorVAE-style Mutual Information Loss  (Objective 1 — feature space)
# ============================================================================

class FactorDiscriminator(nn.Module):
    """
    Small MLP that distinguishes joint q(z_c, z_ca) from product-of-marginals
    q(z_c) × q(z_ca) via density ratio estimation.

    Input:  concat(pool(z_common), pool(z_cardio))  — dimension z_c_ch + z_ca_ch
    Output: logit (pre-sigmoid); positive = joint, negative = marginal product.

    Architecture: (z_c+z_ca) → 64 → 64 → 1  with LeakyReLU.
    With 16+16=32 input dims this has ~4 K parameters — cheap and stable.
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim, name='fc1')(z)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Dense(self.hidden_dim, name='fc2')(h)
        h = nn.leaky_relu(h, negative_slope=0.2)
        return nn.Dense(1, name='fc_out')(h)   # (B, 1) logits


def factor_disc_loss(
    disc_params: Dict,
    discriminator: nn.Module,
    z_c_pooled: jnp.ndarray,
    z_ca_pooled: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Discriminator step: distinguish joint from permuted-marginal samples.

    Real (joint):   [z_c_i, z_ca_i]       from the same image  → label 1
    Fake (product): [z_c_i, z_ca_{π(i)}]  permuted along batch → label 0

    Loss = softplus(-logit_real) + softplus(logit_fake)
           (numerically stable BCE without intermediate sigmoid)

    Returns: (disc_loss scalar, disc_accuracy scalar)
    """
    B = z_c_pooled.shape[0]
    perm = jax.random.permutation(key, B)

    z_joint = jnp.concatenate([z_c_pooled, z_ca_pooled],        axis=-1)  # (B, D)
    z_perm  = jnp.concatenate([z_c_pooled, z_ca_pooled[perm]],  axis=-1)  # (B, D)

    logits_real = discriminator.apply({'params': disc_params}, z_joint)  # (B, 1)
    logits_fake = discriminator.apply({'params': disc_params}, z_perm)   # (B, 1)

    disc_loss = jnp.mean(
        jax.nn.softplus(-logits_real) + jax.nn.softplus(logits_fake)
    )
    disc_acc = (
        jnp.mean((logits_real > 0).astype(jnp.float32)) * 0.5 +
        jnp.mean((logits_fake <= 0).astype(jnp.float32)) * 0.5
    )
    return disc_loss, disc_acc


def factor_vae_mi_loss(
    disc_params: Dict,
    discriminator: nn.Module,
    z_c_pooled: jnp.ndarray,
    z_ca_pooled: jnp.ndarray,
) -> jnp.ndarray:
    """
    VAE encoder MI loss: minimise TC(z_c; z_ca) via the density ratio.

    TC ≈ E_q[log D(z_c, z_ca) − log(1−D(z_c, z_ca))] = E_q[logit(D)]

    Minimising E_q[logit] pushes the discriminator to classify the joint
    distribution as the product of marginals → reduces mutual information.
    The discriminator params are treated as constants (stop_gradient applied
    at the call site in train_sep_vae.py).
    """
    z_joint = jnp.concatenate([z_c_pooled, z_ca_pooled], axis=-1)
    logits  = discriminator.apply({'params': disc_params}, z_joint)   # (B, 1)
    return jnp.mean(logits)


def supervised_contrastive_loss(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """
    Supervised contrastive loss on pooled latent embeddings.

    Uses class labels to pull same-class disease latents together and push
    different-class latents apart. Inputs are expected to be pooled posterior
    means rather than sampled latents to avoid stochastic target noise.
    """
    embeddings = embeddings / jnp.clip(
        jnp.linalg.norm(embeddings, axis=-1, keepdims=True), a_min=1e-6
    )
    logits = (embeddings @ embeddings.T) / temperature
    logits = logits - jnp.max(logits, axis=1, keepdims=True)

    labels = labels.reshape(-1)
    positive_mask = (labels[:, None] == labels[None, :]).astype(jnp.float32)
    identity = jnp.eye(labels.shape[0], dtype=jnp.float32)
    positive_mask = positive_mask - identity
    valid_mask = 1.0 - identity

    exp_logits = jnp.exp(logits) * valid_mask
    log_prob = logits - jnp.log(jnp.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

    positives_per_anchor = jnp.sum(positive_mask, axis=1)
    mean_log_prob_pos = jnp.sum(positive_mask * log_prob, axis=1) / jnp.maximum(
        positives_per_anchor, 1.0
    )
    valid_anchors = positives_per_anchor > 0
    return -jnp.mean(jnp.where(valid_anchors, mean_log_prob_pos, 0.0))


# ============================================================================
# Bbox Attention Supervision  (Objective 1 — spatial)
# ============================================================================

def bbox_attention_loss(
    attn_maps: Dict[str, jnp.ndarray],
    bboxes_cardio: jnp.ndarray,
    B: int,
) -> jnp.ndarray:
    """
    Penalise attention mass outside the ground-truth cardiomegaly bbox.

    Batch is ordered [Normal(B), Cardio(B)] = 2B total.
    Cardio supervision:  attn_maps['cardiomegaly'][B:2B] vs bboxes_cardio

    Args:
        attn_maps:     dict with 'cardiomegaly' key (2B, H, W), softmax probs
        bboxes_cardio: (B, 4) [x0, y0, x1, y1] normalised [0,1]
        B:             per-class batch size

    Returns:
        scalar loss (fraction of attention mass outside bbox, averaged over valid samples)
    """
    attn_ca = attn_maps['cardiomegaly']   # (2B, H, W)
    H, W    = attn_ca.shape[1], attn_ca.shape[2]

    y_lin = (jnp.arange(H, dtype=jnp.float32) + 0.5) / H   # (H,)
    x_lin = (jnp.arange(W, dtype=jnp.float32) + 0.5) / W   # (W,)

    x0 = bboxes_cardio[:, 0][:, None, None]   # (B, 1, 1)
    y0 = bboxes_cardio[:, 1][:, None, None]
    x1 = bboxes_cardio[:, 2][:, None, None]
    y1 = bboxes_cardio[:, 3][:, None, None]

    inside = (
        (x_lin[None, None, :] >= x0) & (x_lin[None, None, :] <= x1) &
        (y_lin[None, :, None] >= y0) & (y_lin[None, :, None] <= y1)
    ).astype(jnp.float32)                                         # (B, H, W)
    outside = 1.0 - inside

    # Only supervise cardio samples (indices B:2B)
    attn_cardio_slice = attn_ca[B:]                              # (B, H, W)
    frac_outside = jnp.sum(attn_cardio_slice * outside, axis=(1, 2))  # (B,)

    # Skip samples with missing/invalid bboxes (x1 - x0 < 1e-4)
    has_bbox = ((bboxes_cardio[:, 2] - bboxes_cardio[:, 0]) > 1e-4).astype(jnp.float32)
    n_valid  = jnp.maximum(jnp.sum(has_bbox), 1.0)

    return jnp.sum(frac_outside * has_bbox) / n_valid


# ============================================================================
# CTR Regression Loss  (Objective 1 — scalar cardiac-size anchor on z_cardio)
# ============================================================================

def ctr_regression_loss(
    ctr_pred: jnp.ndarray,
    ctr_gt:   jnp.ndarray,
    has_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    L1 regression loss: z_cardio must predict ground-truth CTR.

    Forces z_cardio to encode cardiac size as a continuous scalar, not just
    texture within the attention region.  Only supervised where CheXmask is
    available (has_mask=1.0).

    Args:
        ctr_pred: (2B,) model predictions from GAP(z_disease) → Dense(1)
        ctr_gt:   (2B,) ground-truth CTR from CheXmask (0.0 where unavailable)
        has_mask: (2B,) float32 — 1.0 where CheXmask quality mask exists

    Returns:
        scalar: mean L1 error over supervised samples (0.0 if none supervised)
    """
    n_valid = jnp.maximum(jnp.sum(has_mask), 1.0)
    l1 = jnp.abs(ctr_pred - ctr_gt)
    return jnp.sum(l1 * has_mask) / n_valid


# ============================================================================
# Perceptual Loss  (Objective 2 — crisp reconstructions)
# ============================================================================

def backbone_perceptual_loss(
    x_orig: jnp.ndarray,
    x_rec: jnp.ndarray,
    backbone_apply_fn,
    backbone_variables: Dict,
) -> jnp.ndarray:
    """
    L1 distance in frozen CheSS feature space, layers 1–2 only.

    Uses the same backbone already loaded for the encoder — zero extra
    parameters. Layer3 (16×16 for 256px input) is excluded: its stride-16
    gradients produce 16px-period stripes that have been visible since D2.
    Layers 1–2 provide mid-frequency texture guidance without significant aliasing.

    Args:
        x_orig: (B, H, W, 1) in [-1, 1]
        x_rec:  (B, H, W, 1) in [ 0, 1]  (decoder output)
    """
    x_rec_scaled = x_rec * 2.0 - 1.0   # align to backbone expected [-1, 1]

    feats_orig = backbone_apply_fn(backbone_variables, x_orig, return_multiscale=True)
    feats_orig = jax.tree_util.tree_map(jax.lax.stop_gradient, feats_orig)

    feats_rec  = backbone_apply_fn(backbone_variables, x_rec_scaled, return_multiscale=True)

    # Use layers 1–2 only (drop layer3 + layer4).
    # For 256px input the CheSS backbone produces:
    #   layer1: 64×64  (total stride  4) → 4px-period gradients  → negligible
    #   layer2: 32×32  (total stride  8) → 8px-period gradients  → faint
    #   layer3: 16×16  (total stride 16) → 16px-period gradients → strong stripes ✗
    #   layer4:  8×8   (total stride 32) → 32px-period gradients → dominant stripes ✗
    # Layer3 was the primary source of the grid/banding artifacts observed from
    # D2 onward: matching 16×16 feature maps at weight=0.15 injects 16px-period
    # stripe gradients that TV at 0.005 cannot suppress (300× power imbalance).
    # Layer1+2 still provide low/mid-frequency texture guidance without aliasing.
    preferred_layers = ['layer1', 'layer2']
    if all(layer in feats_orig and layer in feats_rec for layer in preferred_layers):
        layers = preferred_layers
    else:
        layers = [
            layer for layer in ('layer2', 'layer3', 'layer4')
            if layer in feats_orig and layer in feats_rec
        ]

    if not layers:
        raise KeyError(
            "Perceptual backbone returned no compatible multiscale layers: "
            f"orig={sorted(feats_orig.keys())}, rec={sorted(feats_rec.keys())}"
        )

    loss = jnp.float32(0.0)
    for layer in layers:
        loss += jnp.mean(jnp.abs(feats_orig[layer] - feats_rec[layer]))

    return loss / jnp.float32(len(layers))


# ============================================================================
# Combined Loss Config
# ============================================================================

@dataclass
class SepVAELossConfig:
    """
    Loss weights — each maps to a research objective.

    Objective 1 (orthogonal separation):
        weight_kl_common, weight_kl_disease, weight_mi_factor, weight_bbox_attn

    Objective 2 (crisp reconstructions):
        weight_rec, weight_perceptual, weight_gan, weight_tv
    """
    # Objective 2
    weight_rec:          float = 1.0
    weight_perceptual:   float = 0.05
    weight_gan:          float = 0.0   # PatchGAN hinge generator loss (D5)
    weight_tv:           float = 0.0   # Total variation — suppresses stripe artifacts (D5)
    weight_masked_rec:   float = 0.0   # Masked anatomy recon — outside-bbox MSE with z_cardio=0

    # Objective 1
    weight_kl_common:    float = 1e-4
    weight_kl_disease:   float = 5e-5
    weight_mi_factor:    float = 1.0    # κ — FactorVAE MI weight for encoder
    weight_bbox_attn:    float = 0.2
    weight_ctr_reg:      float = 0.0   # CTR regression on z_cardio (D5+ mask supervision)
    weight_cardio_supcon: float = 0.05
    supcon_temperature:   float = 0.1

    sigma_inactive:      float = 0.1
    kl_free_bits:        float = 0.0   # per-dim KL floor — 0 = disabled, 0.5 = recommended


# ============================================================================
# Combined VAE Loss
# ============================================================================

def sepvae_loss(
    model,
    params,
    batch: Dict,
    key: jax.random.PRNGKey,
    cfg: SepVAELossConfig,
    batch_stats: Dict = None,
    kl_anneal: jnp.ndarray = None,
    disc_params: Optional[Dict] = None,
    discriminator: Optional[nn.Module] = None,
    backbone_apply_fn=None,
    backbone_variables: Dict = None,
    bbox: Optional[jnp.ndarray] = None,
    has_bbox: Optional[jnp.ndarray] = None,
    has_bbox_query: Optional[jnp.ndarray] = None,
    patch_disc_params: Optional[Dict] = None,
    patch_discriminator: Optional[nn.Module] = None,
    heart_mask: Optional[jnp.ndarray] = None,
    ctr: Optional[jnp.ndarray] = None,
    has_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Binary SepVAE VAE loss (both discriminators frozen via stop_gradient at call site).

    Args:
        model:               SepVAE model (V1 or V2)
        params:              VAE parameters
        batch:               dict with x_norm, x_disease1, disease_labels, bbox_disease1
        key:                 JAX PRNG key
        cfg:                 loss weights
        batch_stats:         backbone BatchNorm stats (None for V2 GroupNorm models)
        kl_anneal:           KL warmup factor in [0,1]  (None = 1.0)
        disc_params:         FactorDiscriminator params (frozen, stop_gradient applied outside)
        discriminator:       FactorDiscriminator module (static — not a JAX array)
        backbone_apply_fn:   backbone.apply for perceptual loss (None = disabled)
        backbone_variables:  backbone variables dict for perceptual loss
        bbox:                (2B, 4) bbox tensor for V2 cross-attention encoder (None = V1/D0)
        has_bbox:            (2B,) float32 mask — 1.0 where bbox is valid (None = V1/D0)
        has_bbox_query:      (2B,) float32 mask used only for the disease-head query path
        patch_disc_params:   PatchGAN discriminator params (frozen, stop_gradient outside)
        patch_discriminator: NLayerDiscriminator module (static — not a JAX array)
        heart_mask:          (2B, S, S) float32 binary heart masks from CheXmask (D5+)
        ctr:                 (2B,) float32 ground-truth CTR values from CheXmask (D5+)
        has_mask:            (2B,) float32 — 1.0 where CheXmask quality mask exists (D5+)

    Returns:
        (total_loss, logs, z_c_pooled, z_ca_pooled, x_rec)
    """
    x_norm     = batch['x_norm']        # (B, H, W, 1) in [-1, 1]
    x_disease1 = batch['x_disease1']    # (B, H, W, 1) cardiomegaly
    labels     = batch['disease_labels']  # (2B,) — [0,...,0, 1,...,1]
    B = x_norm.shape[0]

    x = jnp.concatenate([x_norm, x_disease1], axis=0)   # (2B, H, W, 1)
    query_mask = has_bbox_query if has_bbox_query is not None else has_bbox

    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    key1, _ = jax.random.split(key)
    x_rec, latents_dict, z_c_pooled, z_ca_pooled, ctr_pred = model.apply(
        variables, x, labels, key=key1, train=True,
        bbox=bbox, has_bbox=query_mask,
        heart_mask=heart_mask,
    )

    mu_ca_pooled = jnp.mean(latents_dict['cardiomegaly'][0], axis=(1, 2))

    # ── 1. Reconstruction (MSE) ──────────────────────────────────────────────
    x_01  = (x + 1.0) / 2.0            # [-1,1] → [0,1]
    l_rec = reconstruction_loss(x_01, x_rec)

    # ── 2. KL losses ─────────────────────────────────────────────────────────
    kl       = compute_kl_losses(latents_dict, labels, sigma_inactive=cfg.sigma_inactive,
                                 free_bits=cfg.kl_free_bits)
    l_kl_raw = (
        cfg.weight_kl_common  * kl['common']
        + cfg.weight_kl_disease * kl['cardiomegaly']
    )
    _anneal = kl_anneal if kl_anneal is not None else jnp.float32(1.0)
    l_kl    = l_kl_raw * _anneal

    # ── 3. FactorVAE MI (encoder pushes joint toward product-of-marginals) ───
    if cfg.weight_mi_factor > 0.0 and disc_params is not None and discriminator is not None:
        l_mi = factor_vae_mi_loss(disc_params, discriminator, z_c_pooled, z_ca_pooled)
    else:
        l_mi = jnp.float32(0.0)

    # ── 3b. Cardio supervised contrastive loss on pooled posterior means ────
    if cfg.weight_cardio_supcon > 0.0:
        l_cardio_supcon = supervised_contrastive_loss(
            mu_ca_pooled, labels, temperature=cfg.supcon_temperature
        )
    else:
        l_cardio_supcon = jnp.float32(0.0)

    # ── 4. Bbox attention supervision (cardio only) ───────────────────────────
    if cfg.weight_bbox_attn > 0.0:
        l_bbox = bbox_attention_loss(
            latents_dict['attn_maps'],
            bboxes_cardio=batch['bbox_disease1'],   # (B, 4) cardio bboxes
            B=B,
        )
    else:
        l_bbox = jnp.float32(0.0)

    # ── 5. Perceptual loss (optional) ─────────────────────────────────────────
    if cfg.weight_perceptual > 0.0 and backbone_apply_fn is not None:
        l_perceptual = backbone_perceptual_loss(
            x, x_rec, backbone_apply_fn, backbone_variables
        )
    else:
        l_perceptual = jnp.float32(0.0)

    # ── 6. PatchGAN generator loss (optional, D5+) ───────────────────────────
    # Decoder is trained to fool the frozen image-space discriminator.
    # Hinge generator loss: -mean(D(x_rec)) pushes all patch logits positive.
    #
    # D6: Applied to full batch (Normal + Cardiomegaly, indices 0..2B). PatchGAN uses
    # local patch receptive fields, so the adversarial gradient sharpens local texture
    # (ribs, vessel walls, cardiac border edges) without pushing Normal anatomy toward
    # the global Cardiomegaly silhouette. Normal images are equally blurry and equally
    # need adversarial sharpening pressure.
    if cfg.weight_gan > 0.0 and patch_disc_params is not None and patch_discriminator is not None:
        fake_logits = patch_discriminator.apply(
            {'params': patch_disc_params}, x_rec, train=False
        )
        l_gan = -jnp.mean(fake_logits)
    else:
        l_gan = jnp.float32(0.0)

    # ── 7. Total variation loss (optional, D5+) ───────────────────────────────
    # Suppresses horizontal stripe artifacts from strided perceptual gradients.
    if cfg.weight_tv > 0.0:
        l_tv = total_variation_loss(x_rec)
    else:
        l_tv = jnp.float32(0.0)

    # ── 8. Masked anatomy reconstruction loss (optional) ─────────────────────
    # Decoder-only pass with z_cardio zeroed → forces z_common to reconstruct
    # everything outside the heart bbox without disease information.
    # Uses posterior means (no sampling) for a stable, low-variance signal.
    if cfg.weight_masked_rec > 0.0 and bbox is not None:
        mu_c = latents_dict['common'][0]           # (2B, H_lat, W_lat, z_c_ch)
        mu_d = latents_dict['cardiomegaly'][0]     # (2B, H_lat, W_lat, z_d_ch)
        z_anatomy = jnp.concatenate(
            [mu_c, jnp.zeros_like(mu_d)], axis=-1
        )   # (2B, H_lat, W_lat, z_c_ch + z_d_ch)
        x_rec_anatomy = model.apply({'params': params}, z_anatomy, method=model.decode)
        l_masked_rec = masked_anatomy_reconstruction_loss(
            x_01, x_rec_anatomy, batch['bbox_disease1'], B
        )
    else:
        l_masked_rec = jnp.float32(0.0)

    # ── 9. CTR regression loss (D5+ mask supervision) ────────────────────────
    # Forces z_cardio to encode cardiac size as a continuous scalar dial.
    # Only active when CheXmask masks are provided (weight_ctr_reg > 0 + has_mask).
    if cfg.weight_ctr_reg > 0.0 and ctr is not None and has_mask is not None:
        l_ctr = ctr_regression_loss(ctr_pred, ctr, has_mask)
    else:
        l_ctr = jnp.float32(0.0)

    # ── Total ─────────────────────────────────────────────────────────────────
    total_loss = (
        cfg.weight_rec          * l_rec
        + l_kl
        + cfg.weight_mi_factor  * l_mi
        + cfg.weight_cardio_supcon * l_cardio_supcon
        + cfg.weight_bbox_attn  * l_bbox
        + cfg.weight_perceptual * l_perceptual
        + cfg.weight_gan        * l_gan
        + cfg.weight_tv         * l_tv
        + cfg.weight_masked_rec * l_masked_rec
        + cfg.weight_ctr_reg    * l_ctr
    )

    inactive_mask = (labels == 0).astype(jnp.float32)
    active_mask   = (labels == 1).astype(jnp.float32)

    # z_cardio norms — should differ strongly between classes
    cardio_norm           = jnp.linalg.norm(mu_ca_pooled, axis=-1)
    inactive_norm_cardio  = jnp.sum(cardio_norm * inactive_mask) / jnp.maximum(jnp.sum(inactive_mask), 1.0)
    active_norm_cardio    = jnp.sum(cardio_norm * active_mask)   / jnp.maximum(jnp.sum(active_mask),   1.0)
    active_inactive_ratio = active_norm_cardio / jnp.maximum(inactive_norm_cardio, 1e-6)

    # z_common norms — should be class-invariant (ratio near 1.0)
    common_norm           = jnp.linalg.norm(z_c_pooled, axis=-1)
    inactive_norm_common  = jnp.sum(common_norm * inactive_mask) / jnp.maximum(jnp.sum(inactive_mask), 1.0)
    active_norm_common    = jnp.sum(common_norm * active_mask)   / jnp.maximum(jnp.sum(active_mask),   1.0)
    common_norm_ratio     = active_norm_common / jnp.maximum(inactive_norm_common, 1e-6)

    logs = {
        'loss/total':           total_loss,
        'loss/reconstruction':  l_rec,
        'loss/kl_common':       kl['common'],
        'loss/kl_cardiomegaly': kl['cardiomegaly'],
        'loss/kl_total':        kl['common'] + kl['cardiomegaly'],
        'loss/kl_weighted':     l_kl,
        'loss/kl_anneal':       _anneal,
        'loss/kl_free_bits':    jnp.float32(cfg.kl_free_bits),
        'loss/mi_factor':       l_mi,
        'loss/cardio_supcon':   l_cardio_supcon,
        'loss/bbox_attn':       l_bbox,
        'loss/perceptual':      l_perceptual,
        'loss/gan_g':           l_gan,
        'loss/tv':              l_tv,
        'loss/masked_rec':      l_masked_rec,
        'loss/ctr_reg':         l_ctr,
        'metrics/ctr_pred_mean': jnp.mean(ctr_pred),
        'metrics/z_cardio_norm_inactive': inactive_norm_cardio,
        'metrics/z_cardio_norm_active':   active_norm_cardio,
        'metrics/z_cardio_norm_ratio':    active_inactive_ratio,
        # Gap 3 — z_common invariance: should stay near 1.0 if shared head is disease-agnostic
        'metrics/z_common_norm_inactive': inactive_norm_common,
        'metrics/z_common_norm_active':   active_norm_common,
        'metrics/z_common_norm_ratio':    common_norm_ratio,
    }
    return total_loss, logs, z_c_pooled, z_ca_pooled, x_rec
