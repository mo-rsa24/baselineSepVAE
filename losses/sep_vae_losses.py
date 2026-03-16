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
    • L_perceptual (L1 in frozen CheSS feature space — zero extra params)

  Total:
    L_vae = L_rec + β_c·KL_c + β_d·KL_ca + κ·L_mi + γ·L_perceptual + λ·L_bbox

  Discriminator (trained alternately, frozen during L_vae):
    L_disc = BCE(D(z_c, z_ca), 1) + BCE(D(z_c, z_ca[perm]), 0)
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


# ============================================================================
# KL Divergence Losses
# ============================================================================

def kl_divergence_standard(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """KL(q(z|x) || N(0,I)) — per-sample, summed over latent dims."""
    sum_axes = tuple(range(1, mu.ndim))
    return 0.5 * jnp.sum(jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar, axis=sum_axes)


def kl_divergence_conditional(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1,
) -> jnp.ndarray:
    """
    Conditional KL with label-dependent prior:
    - Active (disease present):  prior = N(0, I)
    - Inactive (disease absent): prior = N(0, sigma_inactive²·I)  — tight

    For inactive samples (Normal images, label≠disease_id) this reduces to:
        0.5 * (μ²/σ_p² + σ_q²/σ_p² - 1 - log(σ_q²/σ_p²))
    which strongly penalises μ away from zero, complementing hard-zero nulling.
    """
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)
    is_active  = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var    = jnp.exp(prior_logvar)

    sum_axes = tuple(range(1, mu.ndim))
    kl = 0.5 * jnp.sum(
        jnp.square(mu) / prior_var + jnp.exp(logvar) / prior_var
        - 1.0 - (logvar - prior_logvar),
        axis=sum_axes,
    )
    return kl


def compute_kl_losses(
    latents_dict: Dict,
    labels: jnp.ndarray,
    sigma_inactive: float = 0.1,
) -> Dict[str, jnp.ndarray]:
    """Batch-mean KL for common and cardiomegaly heads (binary)."""
    mu_c,  logvar_c  = latents_dict['common']
    mu_ca, logvar_ca = latents_dict['cardiomegaly']
    return {
        'common':       jnp.mean(kl_divergence_standard(mu_c, logvar_c)),
        'cardiomegaly': jnp.mean(kl_divergence_conditional(
            mu_ca, logvar_ca, labels, disease_id=1, sigma_inactive=sigma_inactive)),
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
# Perceptual Loss  (Objective 2 — crisp reconstructions)
# ============================================================================

def backbone_perceptual_loss(
    x_orig: jnp.ndarray,
    x_rec: jnp.ndarray,
    backbone_apply_fn,
    backbone_variables: Dict,
) -> jnp.ndarray:
    """
    L1 distance in frozen CheSS feature space at layers 2, 3, and 4.

    Uses the same backbone already loaded for the encoder — zero extra
    parameters. Encourages the decoder to reproduce diagnostically
    relevant CXR structure that MSE alone tends to smooth away.

    Args:
        x_orig: (B, H, W, 1) in [-1, 1]
        x_rec:  (B, H, W, 1) in [ 0, 1]  (decoder output)
    """
    x_rec_scaled = x_rec * 2.0 - 1.0   # align to backbone expected [-1, 1]

    feats_orig = backbone_apply_fn(backbone_variables, x_orig, return_multiscale=True)
    feats_orig = jax.tree_util.tree_map(jax.lax.stop_gradient, feats_orig)

    feats_rec  = backbone_apply_fn(backbone_variables, x_rec_scaled, return_multiscale=True)

    loss = jnp.float32(0.0)
    for layer in ['layer2', 'layer3', 'layer4']:
        loss += jnp.mean(jnp.abs(feats_orig[layer] - feats_rec[layer]))

    return loss / 3.0


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
        weight_rec, weight_perceptual
    """
    # Objective 2
    weight_rec:          float = 1.0
    weight_perceptual:   float = 0.05

    # Objective 1
    weight_kl_common:    float = 1e-4
    weight_kl_disease:   float = 5e-5
    weight_mi_factor:    float = 1.0    # κ — FactorVAE MI weight for encoder
    weight_bbox_attn:    float = 0.2

    sigma_inactive:      float = 0.1


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
) -> Tuple[jnp.ndarray, Dict]:
    """
    Binary SepVAE VAE loss (discriminator is frozen via stop_gradient at call site).

    Args:
        model:              SepVAE model
        params:             VAE parameters
        batch:              dict with x_norm, x_disease1, disease_labels, bbox_disease1
        key:                JAX PRNG key
        cfg:                loss weights
        batch_stats:        backbone BatchNorm stats
        kl_anneal:          KL warmup factor in [0,1]  (None = 1.0)
        disc_params:        FactorDiscriminator params (frozen, stop_gradient applied outside)
        discriminator:      FactorDiscriminator module (static — not a JAX array)
        backbone_apply_fn:  backbone.apply for perceptual loss (None = disabled)
        backbone_variables: backbone variables dict for perceptual loss

    Returns:
        (total_loss, logs)
    """
    x_norm     = batch['x_norm']        # (B, H, W, 1) in [-1, 1]
    x_disease1 = batch['x_disease1']    # (B, H, W, 1) cardiomegaly
    labels     = batch['disease_labels']  # (2B,) — [0,...,0, 1,...,1]
    B = x_norm.shape[0]

    x = jnp.concatenate([x_norm, x_disease1], axis=0)   # (2B, H, W, 1)

    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    key1, _ = jax.random.split(key)
    x_rec, latents_dict, z_c_pooled, z_ca_pooled = model.apply(
        variables, x, labels, key=key1, train=True
    )

    # ── 1. Reconstruction (MSE) ──────────────────────────────────────────────
    x_01  = (x + 1.0) / 2.0            # [-1,1] → [0,1]
    l_rec = reconstruction_loss(x_01, x_rec)

    # ── 2. KL losses ─────────────────────────────────────────────────────────
    kl       = compute_kl_losses(latents_dict, labels, sigma_inactive=cfg.sigma_inactive)
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

    # ── Total ─────────────────────────────────────────────────────────────────
    total_loss = (
        cfg.weight_rec         * l_rec
        + l_kl
        + cfg.weight_mi_factor * l_mi
        + cfg.weight_bbox_attn * l_bbox
        + cfg.weight_perceptual * l_perceptual
    )

    logs = {
        'loss/total':           total_loss,
        'loss/reconstruction':  l_rec,
        'loss/kl_common':       kl['common'],
        'loss/kl_cardiomegaly': kl['cardiomegaly'],
        'loss/kl_total':        kl['common'] + kl['cardiomegaly'],
        'loss/kl_weighted':     l_kl,
        'loss/kl_anneal':       _anneal,
        'loss/mi_factor':       l_mi,
        'loss/bbox_attn':       l_bbox,
        'loss/perceptual':      l_perceptual,
    }
    return total_loss, logs
