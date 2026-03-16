"""
SepVAE V2 — ResNet-50 from scratch, CBAM, self-attention at bottleneck, bbox cross-attention.

Key differences from V1 (CheSS-based):
  • No pretrained backbone — trained from scratch, no frozen layers, no KL blowup
  • GroupNorm replaces BatchNorm throughout (stable for small batches, generative-friendly)
  • CBAM in every BottleneckBlockGN — channel recalibration + spatial attention at every scale
      Channel attention: suppresses lung-parenchyma channels, amplifies cardiac-contour channels
      Spatial attention: progressively focuses spatial features on the heart boundary (layers 1→4)
  • Self-attention at layer3 bottleneck (16×16 for 256px input) — global heart-to-lung ratio
  • Two disease head modes, selected by use_bbox_cross_attn flag:
      D0: DiseaseAttnHeadV2  — learned prototype query (no bbox needed)
      D1: BboxCrossAttnHead  — bbox Gaussian prior drives cross-attention from epoch 1

Bbox note (256×256):
  Annotations in train_filtered.csv are normalised [0,1] → resolution-independent.
  Gaussian σ = bbox_width/4 (quarter-width) so the prior concentrates inside the bbox.
  Training script must assemble the full (2B, 4) bbox tensor:
      bbox_full = concat([zeros(B, 4), bbox_cardio], axis=0)
      has_bbox  = (bbox_full[:, 2] - bbox_full[:, 0] > 1e-4).astype(float32)

Architecture (256×256 input):
    Input (256×256×1)
        → ResNet50Scratch layers 1–3     → h_shared (16×16×1024)
        → SelfAttention2D at bottleneck  → h_attn   (16×16×1024)
              ↓                                  ↓
        bg_branch Layer4 (learnable)    tg_branch Layer4 (learnable)
        → h_bg (8×8×2048)               → h_tg (8×8×2048)
              ↓                                  ↓  [upsample both to 16×16]
        ConvHeadGN                      DiseaseAttnHeadV2 | BboxCrossAttnHead
        z_common (16ch)                 z_disease (16ch)
              ↓                                  ↓
              └──── hard-zero z_disease for Normal ────┘
                              ↓
                     concat (16×16×32)
                              ↓
                    SepVAEDecoder (16→32→64→128→256)
                              ↓
                    Output (256×256×1)

Decoder (SepVAEDecoderV2):
    Each ResBlockSE = ResBlock (GN → swish → conv → GN → swish → conv)
                    + SqueezeExcitation gate (avg pool → FC → relu → FC → sigmoid)
    SE is applied to the main conv branch before the residual add.
    When z_disease is non-zero, SE learns to amplify cardiac-silhouette channels
    and suppress bone-density / lung-texture channels in overlapping regions,
    making the z_disease → cardiac feature mapping explicit and disentangled.

Loss compatibility: latents_dict uses the same keys as V1 ('common', 'cardiomegaly',
'attn_maps'/'cardiomegaly') so sep_vae_losses.py requires no changes for D0/D1.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Sequence, Optional

from models.ae_kl import ResBlock, SelfAttention2D
from models.sep_vae_jax import SmoothUp


# =============================================================================
# CBAM — Convolutional Block Attention Module
# =============================================================================

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.

    Asks "which feature channels are relevant to the heart?" at every block.
    Both the global average-pool path and global max-pool path pass through a
    shared MLP (same weights), then are summed before sigmoid gating.

    This suppresses lung-parenchyma channels (texture, air-space) and amplifies
    cardiac-boundary channels throughout the feature hierarchy.

    reduction_ratio=16 → hidden dim = C/16:
        C=256  → 16 hidden  (layer1 output)
        C=512  → 32 hidden  (layer2 output)
        C=1024 → 64 hidden  (layer3 output)
        C=2048 → 128 hidden (layer4 output)
    """
    reduction_ratio: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        hidden = max(C // self.reduction_ratio, 1)

        # Global spatial pooling: (B, 1, 1, C) — capture global channel statistics
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True).reshape(B, C)
        max_pool = jnp.max(x,  axis=(1, 2), keepdims=True).reshape(B, C)

        # Shared MLP: in Flax @compact, assigning a module once and calling it
        # twice applies the SAME weights to both inputs — this IS weight sharing.
        fc1 = nn.Dense(hidden, use_bias=False, name='ca_fc1')
        fc2 = nn.Dense(C,      use_bias=False, name='ca_fc2')

        avg_out = fc2(nn.relu(fc1(avg_pool)))   # (B, C)
        max_out = fc2(nn.relu(fc1(max_pool)))   # (B, C)  ← same fc1/fc2 weights

        scale = jax.nn.sigmoid(avg_out + max_out).reshape(B, 1, 1, C)
        return x * scale


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.

    Asks "where in the image matters?" at every block.
    Computes channel-wise average and max descriptors, concatenates them,
    then applies a large-kernel conv (7×7) to produce a spatial attention map.

    This progressively focuses spatial features on cardiac contours across
    layers 1→4, so by the time features reach BboxCrossAttnHead, the spatial
    field-of-view is already biased toward the heart boundary.
    """
    kernel_size: int = 7

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Channel-wise descriptors: (B, H, W, 1)
        avg_desc = jnp.mean(x, axis=-1, keepdims=True)
        max_desc = jnp.max(x,  axis=-1, keepdims=True)

        combined = jnp.concatenate([avg_desc, max_desc], axis=-1)   # (B, H, W, 2)

        scale = nn.Conv(
            features=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            use_bias=False,
            name='sa_conv',
        )(combined)   # (B, H, W, 1)

        return x * jax.nn.sigmoid(scale)


# =============================================================================
# ResNet-50 building blocks (GroupNorm + CBAM, trainable from scratch)
# =============================================================================

class BottleneckBlockGN(nn.Module):
    """
    ResNet-50 bottleneck with GroupNorm + CBAM.

    Pipeline: 1×1 reduce → 3×3 conv → 1×1 expand
              → ChannelAttention → SpatialAttention
              → add residual → relu

    CBAM is applied to the main branch output before the residual add.
    This is the standard integration (CBAM paper, Fig. 3):
        F_refined = SpatialAttn(ChannelAttn(F_main))
        output    = relu(F_refined + shortcut)

    GroupNorm(32) is valid for all standard ResNet-50 output channels:
        filters × 4 ∈ {256, 512, 1024, 2048} — all divisible by 32.
    """
    filters: int
    stride: int = 1
    use_projection: bool = False

    @nn.compact
    def __call__(self, x):
        residual = x

        h = nn.Conv(self.filters, (1, 1), strides=(1, 1), use_bias=False, name='conv1')(x)
        h = nn.GroupNorm(num_groups=32, name='gn1')(h)
        h = nn.relu(h)

        h = nn.Conv(self.filters, (3, 3), strides=(self.stride, self.stride),
                    padding='SAME', use_bias=False, name='conv2')(h)
        h = nn.GroupNorm(num_groups=32, name='gn2')(h)
        h = nn.relu(h)

        h = nn.Conv(self.filters * 4, (1, 1), strides=(1, 1), use_bias=False, name='conv3')(h)
        h = nn.GroupNorm(num_groups=32, name='gn3')(h)

        # CBAM: channel recalibration then spatial focus — applied before residual add
        h = ChannelAttention(reduction_ratio=16, name='cbam_channel')(h)
        h = SpatialAttention(kernel_size=7,      name='cbam_spatial')(h)

        if self.use_projection or self.stride != 1:
            residual = nn.Conv(self.filters * 4, (1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False, name='proj_conv')(x)
            residual = nn.GroupNorm(num_groups=32, name='proj_gn')(residual)

        return nn.relu(h + residual)


class Layer4BranchGN(nn.Module):
    """
    Learnable layer4 branch — trained from scratch with GroupNorm.
    Diverges from the shared trunk after layer3.

    Input:  (B, 16, 16, 1024)   — layer3 output (256×256 input)
    Output: (B,  8,  8, 2048)
    """
    @nn.compact
    def __call__(self, x):
        for i in range(3):
            x = BottleneckBlockGN(
                filters=512,
                stride=(2 if i == 0 else 1),
                use_projection=(i == 0),
                name=f'block{i}',
            )(x)
        return x   # (B, 8, 8, 2048)


# =============================================================================
# ResNet-50 from scratch
# =============================================================================

class ResNet50Scratch(nn.Module):
    """
    ResNet-50 trained from scratch, GroupNorm throughout.

    Designed for 256×256 grayscale CXR (1-channel input).
    Self-attention is inserted at the end of layer3 to capture long-range
    dependencies (e.g. cardiac-to-lung-field ratio for cardiomegaly).

    Spatial dimensions for 256×256 input:
        stem + pool : (B, 64, 64,   64)
        layer1      : (B, 64, 64,  256)
        layer2      : (B, 32, 32,  512)
        layer3      : (B, 16, 16, 1024)  ← self-attention here (256 tokens, cheap)
        layer4      : (B,  8,  8, 2048)  ← only if stop_at_layer3=False

    Returns (B, 16, 16, 1024) by default (stop_at_layer3=True).
    """
    attn_heads: int = 4

    @nn.compact
    def __call__(self, x):
        # Stem: 7×7, stride 2  →  (B, 128, 128, 64) for 256px
        h = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME',
                    use_bias=False, name='stem_conv')(x)
        h = nn.GroupNorm(num_groups=32, name='stem_gn')(h)
        h = nn.relu(h)

        # Max pool: 3×3, stride 2  →  (B, 64, 64, 64)
        h = nn.max_pool(h, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Layer 1: 3 blocks, filters=64, out=256  →  (B, 64, 64, 256)
        for i in range(3):
            h = BottleneckBlockGN(filters=64, stride=1,
                                  use_projection=(i == 0),
                                  name=f'layer1_b{i}')(h)

        # Layer 2: 4 blocks, filters=128, out=512, stride-2 at b0  →  (B, 32, 32, 512)
        for i in range(4):
            h = BottleneckBlockGN(filters=128, stride=(2 if i == 0 else 1),
                                  use_projection=(i == 0),
                                  name=f'layer2_b{i}')(h)

        # Layer 3: 6 blocks, filters=256, out=1024, stride-2 at b0  →  (B, 16, 16, 1024)
        for i in range(6):
            h = BottleneckBlockGN(filters=256, stride=(2 if i == 0 else 1),
                                  use_projection=(i == 0),
                                  name=f'layer3_b{i}')(h)

        # Self-attention at bottleneck — 256 tokens for 256px input, O(256²) = negligible
        # Captures long-range structure: cardiac silhouette vs surrounding lung fields
        h = SelfAttention2D(num_heads=self.attn_heads, name='bottleneck_attn')(h)

        return h   # (B, 16, 16, 1024) — Layer4 handled by separate Layer4BranchGN modules


# =============================================================================
# Encoder heads
# =============================================================================

class ConvHeadGN(nn.Module):
    """Projects spatial features to (μ, log σ²). GroupNorm variant of V1 ConvHead."""
    out_channels: int
    hidden_channels: int = 256

    @nn.compact
    def __call__(self, h):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', name='c1')(h)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_channels // 2, (3, 3), padding='SAME', name='c2')(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels * 2, (3, 3), padding='SAME', name='c3')(x)
        mu, logvar = jnp.split(x, 2, axis=-1)
        return mu, logvar


class DiseaseAttnHeadV2(nn.Module):
    """
    Disease head with a single learned prototype query (D0 mode).

    The query learns to focus on the anatomical region relevant to the disease.
    Near-uniform at init — the model discovers where to attend via gradient.
    Structurally identical to V1 DiseaseAttentionHead, GroupNorm variant.

    Use this for D0 (smoke test, no bbox input required).
    """
    out_channels: int
    hidden_channels: int = 256
    query_dim: int = 256

    @nn.compact
    def __call__(self, h: jnp.ndarray):
        B, H, W, C = h.shape
        HW = H * W

        # Learned disease prototype — small init → near-uniform attention at start
        q = self.param('disease_query', nn.initializers.normal(0.02), (self.query_dim,))

        h_flat = h.reshape(B, HW, C)
        K = nn.Dense(self.query_dim, use_bias=False, name='key_proj')(h_flat)   # (B, HW, D)

        scale = jnp.sqrt(jnp.array(self.query_dim, dtype=jnp.float32))
        attn_logits  = jnp.einsum('bkd,d->bk', K, q) / scale   # (B, HW)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)      # (B, HW)
        attn_map     = attn_weights.reshape(B, H, W)             # (B, H, W)

        h_attended = h * (attn_map[:, :, :, None] * HW)          # (B, H, W, C)

        mu, logvar = ConvHeadGN(
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            name='conv_head',
        )(h_attended)

        return mu, logvar, attn_map


class BboxCrossAttnHead(nn.Module):
    """
    Disease head with bbox-guided cross-attention (D1+ mode).

    For disease images, the ground-truth bbox [x0, y0, x1, y1] defines a
    Gaussian spatial prior centered on the annotated region. This prior is
    used to form a weighted aggregate of the encoder key vectors, giving the
    model a direct spatial query from epoch 1 — bypassing the slow convergence
    of a purely learned query.

    For Normal images (has_bbox=0.0): falls back to a learned query identical
    to DiseaseAttnHeadV2. KL tight prior + hard-zero nulling still drive
    z_disease → 0 for Normal images independent of the attention path.

    Mechanism:
        1. Project branch features → K, V  (B, HW, query_dim)
        2. Compute Gaussian heatmap from bbox center + sigma  (B, H, W)
        3. Weighted aggregate of K under heatmap  →  Q_bbox  (B, 1, query_dim)
        4. Interpolate Q_bbox / Q_learned by has_bbox mask
        5. Cross-attention: Q × K^T → softmax → attn_map  (B, H, W)
        6. Gate features: h * (attn_map * HW) → ConvHeadGN → (mu, logvar)

    Args:
        h:        (B, H, W, C)  encoder branch features
        bbox:     (B, 4)        [x0, y0, x1, y1] normalised [0,1]; zeros for Normal
        has_bbox: (B,)          float mask: 1.0 = valid bbox, 0.0 = Normal / missing
    """
    out_channels: int
    hidden_channels: int = 256
    query_dim: int = 256

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        bbox: Optional[jnp.ndarray] = None,
        has_bbox: Optional[jnp.ndarray] = None,
    ):
        B, H, W, C = h.shape
        HW = H * W

        h_flat = h.reshape(B, HW, C)
        K = nn.Dense(self.query_dim, use_bias=False, name='key_proj')(h_flat)   # (B, HW, D)
        V = nn.Dense(self.query_dim, use_bias=False, name='val_proj')(h_flat)   # (B, HW, D)

        # Learned fallback query — used for Normal images or when bbox is absent
        q_learned = self.param('fallback_query', nn.initializers.normal(0.02), (self.query_dim,))
        Q_learned  = jnp.tile(q_learned[None, :], (B, 1))[:, None, :]   # (B, 1, D)

        if bbox is not None and has_bbox is not None:
            # ── Gaussian spatial prior from bbox ──────────────────────────────
            y_coords = (jnp.arange(H, dtype=jnp.float32) + 0.5) / H   # (H,)
            x_coords = (jnp.arange(W, dtype=jnp.float32) + 0.5) / W   # (W,)
            xx, yy   = jnp.meshgrid(x_coords, y_coords)                # (H, W)

            cx = (bbox[:, 0] + bbox[:, 2]) * 0.5                          # (B,) centre x
            cy = (bbox[:, 1] + bbox[:, 3]) * 0.5                          # (B,) centre y
            # σ = bbox_width / 4 (quarter-width): Gaussian drops to ~14% at the bbox
            # boundary, concentrating the prior inside the heart rather than spreading
            # beyond it. Using 0.5 (half-width) would wash out into surrounding lungs,
            # especially with VinBigData's union bboxes which are already oversized.
            sx = jnp.maximum((bbox[:, 2] - bbox[:, 0]) * 0.25, 0.05)     # σ_x
            sy = jnp.maximum((bbox[:, 3] - bbox[:, 1]) * 0.25, 0.05)     # σ_y

            gauss = jnp.exp(
                -0.5 * (
                    (xx[None] - cx[:, None, None]) ** 2 / sx[:, None, None] ** 2
                  + (yy[None] - cy[:, None, None]) ** 2 / sy[:, None, None] ** 2
                )
            )   # (B, H, W)

            gauss_flat = gauss.reshape(B, HW, 1)
            gauss_norm = gauss_flat / (gauss_flat.sum(axis=1, keepdims=True) + 1e-6)

            # Bbox-weighted aggregate of K  →  cross-attention query
            Q_bbox = jnp.sum(K * gauss_norm, axis=1, keepdims=True)   # (B, 1, D)

            # Blend: disease images use Q_bbox, Normal images use Q_learned
            w = has_bbox[:, None, None]                                 # (B, 1, 1)
            Q = Q_bbox * w + Q_learned * (1.0 - w)                     # (B, 1, D)
        else:
            Q = Q_learned   # (B, 1, D)

        # ── Cross-attention: Q × K^T → attention map over spatial positions ──
        scale        = jnp.sqrt(jnp.array(self.query_dim, dtype=jnp.float32))
        attn_logits  = jnp.einsum('bqd,bkd->bqk', Q, K) / scale   # (B, 1, HW)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)         # (B, 1, HW)
        attn_map     = attn_weights[:, 0, :].reshape(B, H, W)       # (B, H, W)

        # Gate spatial features (near-identity at uniform attention)
        h_attended = h * (attn_map[:, :, :, None] * HW)             # (B, H, W, C)

        mu, logvar = ConvHeadGN(
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            name='conv_head',
        )(h_attended)

        return mu, logvar, attn_map


# =============================================================================
# Encoder
# =============================================================================

class SepVAEEncoderV2(nn.Module):
    """
    Binary encoder — ResNet-50 from scratch + self-attention + configurable disease head.

    use_bbox_cross_attn=False  →  D0: learned query, no bbox input needed
    use_bbox_cross_attn=True   →  D1+: bbox Gaussian prior drives disease head

    The trunk (layers 1–3 + bottleneck attention) is shared; two learnable
    Layer4 branches then diverge to produce z_common and z_disease separately.
    Both branch outputs are upsampled to 16×16 for the latent space.
    """
    z_channels_common:   int  = 16
    z_channels_disease:  int  = 16
    query_dim:           int  = 256
    attn_heads:          int  = 4
    use_bbox_cross_attn: bool = False   # False = D0, True = D1+

    def setup(self):
        # nn.remat wraps modules for gradient checkpointing — avoids the tracer
        # leak that occurs with jax.checkpoint(lambda: self.module(x)) closures.
        self.backbone    = nn.remat(ResNet50Scratch)(attn_heads=self.attn_heads)
        self.bg_branch   = nn.remat(Layer4BranchGN)()
        self.tg_branch   = nn.remat(Layer4BranchGN)()
        self.head_common = ConvHeadGN(
            out_channels=self.z_channels_common,
            name='head_common',
        )
        if self.use_bbox_cross_attn:
            self.head_disease = BboxCrossAttnHead(
                out_channels=self.z_channels_disease,
                query_dim=self.query_dim,
                name='head_disease',
            )
        else:
            self.head_disease = DiseaseAttnHeadV2(
                out_channels=self.z_channels_disease,
                query_dim=self.query_dim,
                name='head_disease',
            )

    def __call__(
        self,
        x,
        train: bool = True,
        bbox: Optional[jnp.ndarray] = None,
        has_bbox: Optional[jnp.ndarray] = None,
    ):
        B = x.shape[0]

        # Shared trunk (nn.remat on submodule handles gradient checkpointing)
        h_shared = self.backbone(x)   # (B, 16, 16, 1024)

        # Learnable layer4 branches
        h_bg = self.bg_branch(h_shared)   # (B, 8, 8, 2048)
        h_tg = self.tg_branch(h_shared)   # (B, 8, 8, 2048)

        # Upsample both branches to 16×16 (latent spatial resolution)
        h_bg = jax.image.resize(h_bg, (B, 16, 16, 2048), method='bilinear')
        h_tg = jax.image.resize(h_tg, (B, 16, 16, 2048), method='bilinear')

        # Common head
        mu_c, logvar_c = self.head_common(h_bg)

        # Disease head
        if self.use_bbox_cross_attn:
            mu_d, logvar_d, attn_map = self.head_disease(
                h_tg, bbox=bbox, has_bbox=has_bbox
            )
        else:
            mu_d, logvar_d, attn_map = self.head_disease(h_tg)

        # Keys match V1 exactly — sep_vae_losses.py requires no changes
        return {
            'common':       (mu_c,  logvar_c),
            'cardiomegaly': (mu_d,  logvar_d),
            'attn_maps': {
                'cardiomegaly': attn_map,   # (B, H, W) — for bbox loss
            },
        }


# =============================================================================
# Hard-zero head nulling
# =============================================================================

def apply_head_nulling_v2(
    latents_dict: Dict,
    labels: jnp.ndarray,
    key: jax.random.PRNGKey,
    disease_label_id: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Hard-zero head nulling — identical semantics to V1 apply_head_nulling.

    label == disease_label_id : z_disease_decode = z_disease  (sampled)
    label != disease_label_id : z_disease_decode = 0          (hard zero)

    Returns:
        z_concat:    (B, 16, 16, z_c + z_d)  decoder input
        z_c_pooled:  (B, z_c_ch)             for MI discriminator
        z_d_pooled:  (B, z_d_ch)             for MI discriminator
    """
    mu_c,  logvar_c = latents_dict['common']
    mu_d,  logvar_d = latents_dict['cardiomegaly']

    key_c, key_d = jax.random.split(key)
    z_c = mu_c + jnp.exp(0.5 * logvar_c) * jax.random.normal(key_c, mu_c.shape)
    z_d = mu_d + jnp.exp(0.5 * logvar_d) * jax.random.normal(key_d, mu_d.shape)

    mask = jnp.where(labels == disease_label_id, 1.0, 0.0)[:, None, None, None]
    z_d_decode = z_d * mask

    z_concat    = jnp.concatenate([z_c, z_d_decode], axis=-1)   # (B, 16, 16, z_c+z_d)
    z_c_pooled  = jnp.mean(z_c, axis=(1, 2))
    z_d_pooled  = jnp.mean(z_d, axis=(1, 2))

    return z_concat, z_c_pooled, z_d_pooled


# =============================================================================
# SE-gated decoder
# =============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation gate for the decoder.

    Decoder SE uses global average pool only (not max pool).
    Rationale: the encoder's CBAM uses avg+max for edge detection (discriminative).
    The decoder is generative — it needs the mean activation level of each channel
    to decide "how much cardiac boundary feature to render", not peak detection.

    When z_disease is non-zero (cardiomegaly), SE learns:
      • Amplify channels that encode the cardiac silhouette / myocardial border
      • Suppress channels encoding lung vascularity, rib cortex, vertebral edges
    This makes the z → channel activation mapping explicit at every decoder level,
    reducing entanglement between heart size and surrounding anatomy in O2.

    reduction_ratio=16 is consistent with encoder CBAM channel attention.
    All decoder ch_mults (64, 128, 256, 512) give hidden dims (4, 8, 16, 32) — viable.
    """
    reduction_ratio: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        hidden = max(C // self.reduction_ratio, 1)

        # Squeeze: global average pool → (B, C)
        z = jnp.mean(x, axis=(1, 2))

        # Excitation: two FC layers with relu gating
        z = nn.Dense(hidden, use_bias=False, name='se_fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(C, use_bias=False, name='se_fc2')(z)

        scale = jax.nn.sigmoid(z).reshape(B, 1, 1, C)
        return x * scale


class ResBlockSE(nn.Module):
    """
    ResBlock with Squeeze-and-Excitation gate.

    Pipeline:
        GN → swish → conv3×3 → GN → swish → conv3×3
        → SqueezeExcitation                           ← SE applied to main branch
        → + residual shortcut

    SE is placed after the second conv and before the residual add.
    This follows the standard SE-ResNet integration: the attention gate refines
    the main branch features before they are added to the identity path.
    """
    ch: int
    dropout: float = 0.0
    se_reduction: int = 16

    @nn.compact
    def __call__(self, x, train: bool = True):
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3), padding='SAME')(h)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        if self.dropout > 0 and train:
            h = nn.Dropout(self.dropout)(h, deterministic=False)
        h = nn.Conv(self.ch, (3, 3), padding='SAME')(h)

        # SE gate — recalibrate channels before residual add
        h = SqueezeExcitation(reduction_ratio=self.se_reduction, name='se')(h)

        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1))(x)
        return x + h


class SepVAEDecoderV2(nn.Module):
    """
    SE-gated decoder: z(16×16×32) → 256×256×1 via 4 bilinear upsamples.

    Replaces the plain ResBlock decoder with ResBlockSE at every level.
    The SE gate at each level gives the decoder a learned "volume knob" per
    channel conditioned on the current feature statistics — allowing it to
    amplify cardiac-boundary channels and suppress bone/lung channels when
    z_disease drives reconstruction of an enlarged heart.

    Channel schedule (high-res → low-res order, indices 0–4):
        (64, 128, 256, 512, 512)
    Upsamples (SmoothUp = bilinear resize + 2× conv, no checkerboard):
        16 → 32 → 64 → 128 → 256
    """
    ch_mults:       Sequence[int] = (64, 128, 256, 512, 512)
    num_res_blocks: int           = 2
    dropout:        float         = 0.0
    z_channels:     int           = 32
    se_reduction:   int           = 16

    @nn.compact
    def __call__(self, z, train: bool = True):
        h = nn.Conv(self.ch_mults[-1], (3, 3), padding='SAME', name='z_proj')(z)

        for i in reversed(range(len(self.ch_mults))):
            ch = self.ch_mults[i]
            for _ in range(self.num_res_blocks):
                h = ResBlockSE(
                    ch=ch,
                    dropout=self.dropout,
                    se_reduction=self.se_reduction,
                )(h, train=train)
            if i > 0:
                h = SmoothUp(ch=self.ch_mults[i - 1])(h)

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='conv_out')(h)
        return nn.sigmoid(h)


# =============================================================================
# Top-level model
# =============================================================================

class SepVAEV2(nn.Module):
    """
    SepVAE V2 — from-scratch backbone, self-attention, configurable disease head.

    Training flags:
        use_bbox_cross_attn=False   →  D0 smoke test (no bbox arg needed)
        use_bbox_cross_attn=True    →  D1+ (pass bbox and has_bbox in forward call)

    Decoder: SepVAEDecoderV2 — SE-gated ResBlocks, 16×16 → 256×256.
        ch_mults = (64, 128, 256, 512, 512)
        4 SmoothUp calls: 16 → 32 → 64 → 128 → 256

    Loss functions: fully compatible with sep_vae_losses.py (same dict keys as V1).
    """
    z_channels_common:   int  = 16
    z_channels_disease:  int  = 16
    query_dim:           int  = 256
    attn_heads:          int  = 4
    use_bbox_cross_attn: bool = False

    def setup(self):
        self.encoder = SepVAEEncoderV2(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            query_dim=self.query_dim,
            attn_heads=self.attn_heads,
            use_bbox_cross_attn=self.use_bbox_cross_attn,
        )
        # SE-gated decoder: 16×16 → 256×256, 4 SmoothUp calls
        self.decoder = SepVAEDecoderV2(
            ch_mults=(64, 128, 256, 512, 512),
            num_res_blocks=2,
            z_channels=self.z_channels_common + self.z_channels_disease,
        )

    def __call__(
        self,
        x,
        labels,
        *,
        key,
        train: bool = True,
        bbox: Optional[jnp.ndarray] = None,
        has_bbox: Optional[jnp.ndarray] = None,
    ):
        latents_dict = self.encoder(
            x, train=train, bbox=bbox, has_bbox=has_bbox
        )
        key_sample, _ = jax.random.split(key)
        z_concat, z_c_pooled, z_d_pooled = apply_head_nulling_v2(
            latents_dict, labels, key_sample, disease_label_id=1,
        )
        x_rec = self.decoder(z_concat, train=train)
        return x_rec, latents_dict, z_c_pooled, z_d_pooled

    def encode(self, x, bbox=None, has_bbox=None):
        return self.encoder(x, train=False, bbox=bbox, has_bbox=has_bbox)

    def decode(self, z):
        return self.decoder(z, train=False)
