"""
Multi-head SepVAE with frozen CheSS ResNet-50 backbone.

This module implements a Separated Variational Autoencoder (SepVAE) with:
- Frozen CheSS ResNet-50 backbone for feature extraction
- Optional Feature Pyramid Network (FPN) for multi-scale features
- Three parallel Conv heads for disentangled latent spaces:
  * Common head (64×64×4): Shared anatomy features
  * Cardiomegaly head (64×64×2): Disease-specific features
  * Pleural Effusion head (64×64×2): Disease-specific features
- Head nulling logic to prevent information leakage
- Shared decoder for reconstruction
- Optional partial backbone unfreezing (layer3/layer4)

Architecture (with FPN enabled):
    Input (512×512×1) → ResNet50 (frozen/partial)
                         ↓ multi-scale features
                    FPN fusion (64×64×512)
                         ↓
      ┌──────────────────┼──────────────────┐
      ↓                  ↓                  ↓
 Common Head       Cardio Head       Effusion Head
  (64×64×4)         (64×64×2)          (64×64×2)
      ↓                  ↓                  ↓
      └──────────────────┬──────────────────┘
                         ↓
                  Head Nulling
                         ↓
                Concat latents (64×64×8)
                         ↓
                     Decoder
                         ↓
              Output (512×512×1)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Sequence

from models.resnet_jax import ResNet50CheSS
from models.ae_kl import ResBlock


class SmoothUp(nn.Module):
    """
    Anti-aliased upsampling block for SepVAE decoder.

    Designed to prevent checkerboard artifacts that occur with nearest-neighbor
    upsampling. Supports three methods:

    1. 'bilinear': Smooth interpolation followed by conv (recommended default)
    2. 'nearest': Legacy method - prone to checkerboard artifacts
    3. 'subpixel': Learned upsampling via pixel-shuffle (best quality, more compute)

    The bilinear method matches what FPN uses internally for feature fusion,
    ensuring consistent upsampling behavior throughout the architecture.

    Attributes:
        ch: Output channels
        dtype: Compute dtype (default: float32)
        method: Upsampling method - 'bilinear', 'nearest', or 'subpixel'
    """
    ch: int
    dtype: jnp.dtype = jnp.float32
    method: str = 'bilinear'

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        if self.method == 'subpixel':
            # Sub-pixel convolution (pixel shuffle): fully learnable upsampling
            # More expensive but produces the smoothest results
            # Conv to 4x output channels, then pixel-shuffle to 2x spatial
            h = nn.Conv(self.ch * 4, (3, 3), padding="SAME", dtype=self.dtype, name='subpixel_conv')(x)
            # Reshape: (B, H, W, C*4) -> (B, H, W, 2, 2, C) -> (B, H*2, W*2, C)
            h = h.reshape(B, H, W, 2, 2, self.ch)
            h = jnp.transpose(h, (0, 1, 3, 2, 4, 5))  # (B, H, 2, W, 2, C)
            h = h.reshape(B, H * 2, W * 2, self.ch)
            # Smoothing conv to reduce any residual artifacts
            h = nn.Conv(self.ch, (3, 3), padding="SAME", dtype=self.dtype, name='smooth')(h)
        elif self.method == 'bilinear':
            # Bilinear interpolation + conv (recommended)
            # This matches the FPN upsampling strategy
            h = jax.image.resize(x, (B, H * 2, W * 2, C), method='bilinear')
            h = nn.Conv(self.ch, (3, 3), padding="SAME", dtype=self.dtype, name='conv_up')(h)
            # Additional smoothing to eliminate any aliasing
            h = nn.Conv(self.ch, (3, 3), padding="SAME", dtype=self.dtype, name='smooth')(h)
        else:  # 'nearest' - legacy, not recommended
            h = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
            h = nn.Conv(self.ch, (3, 3), padding="SAME", dtype=self.dtype, name='conv_up')(h)

        return h


class FPN(nn.Module):
    """
    Feature Pyramid Network for fusing multi-scale backbone features.

    Combines layer2 (64×64×512), layer3 (32×32×1024), and layer4 (16×16×2048)
    into a single feature map at 64×64 resolution via top-down pathway with
    lateral connections.

    This addresses the spatial resolution bottleneck where bilinear upsampling
    of layer4 alone (16×16→64×64) loses fine-grained spatial information.

    Attributes:
        out_channels: Number of output channels (default: 512)
        dtype: Compute dtype for mixed precision (default: float32)
    """
    out_channels: int = 512
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, features: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Fuse multi-scale features via FPN.

        Args:
            features: Dict with keys:
                'layer2': (B, 64, 64, 512)
                'layer3': (B, 32, 32, 1024)
                'layer4': (B, 16, 16, 2048)

        Returns:
            Fused features (B, 64, 64, out_channels)
        """
        # Lateral connections: 1×1 conv to project each scale to common channels
        lat2 = nn.Conv(self.out_channels, (1, 1), dtype=self.dtype, name='lateral_layer2')(features['layer2'])
        lat3 = nn.Conv(self.out_channels, (1, 1), dtype=self.dtype, name='lateral_layer3')(features['layer3'])
        lat4 = nn.Conv(self.out_channels, (1, 1), dtype=self.dtype, name='lateral_layer4')(features['layer4'])

        # Top-down pathway: upsample deeper features and add to shallower
        # Layer4 (16×16) → upsample to 32×32 → add to layer3
        B = lat4.shape[0]
        lat4_up = jax.image.resize(lat4, (B, 32, 32, self.out_channels), method='bilinear')
        p3 = lat3 + lat4_up

        # P3 (32×32) → upsample to 64×64 → add to layer2
        p3_up = jax.image.resize(p3, (B, 64, 64, self.out_channels), method='bilinear')
        p2 = lat2 + p3_up

        # Smooth with 3×3 conv to reduce aliasing from upsampling
        fused = nn.Conv(self.out_channels, (3, 3), padding='SAME', dtype=self.dtype, name='smooth')(p2)

        return fused  # (B, 64, 64, out_channels)


class ConvHead(nn.Module):
    """
    Convolutional head that projects spatial features to latent μ and log_σ.

    Architecture:
        h(64×64×C_in) → Conv(3×3, hidden_ch) → GroupNorm → ReLU → Conv(3×3, out_ch*2)
        Split output into μ and log_σ

    Attributes:
        out_channels: Output channels (latent channels)
        hidden_channels: Hidden layer channels (default: 512)
        dtype: Compute dtype for mixed precision (default: float32)
    """
    out_channels: int
    hidden_channels: int = 512
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h):
        """
        Args:
            h: Spatial features (B, 64, 64, in_channels)

        Returns:
            mu: Mean of latent distribution (B, 64, 64, out_channels) in float32
            logvar: Log-variance of latent distribution (B, 64, 64, out_channels) in float32
        """
        # Hidden conv layer
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, name='conv1')(h)
        x = nn.GroupNorm(num_groups=32, dtype=self.dtype)(x)
        x = nn.relu(x)

        # Output layer (2x out_channels for mu and logvar)
        x = nn.Conv(self.out_channels * 2, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, name='conv2')(x)

        # Split into mu and logvar — cast to float32 for numerical stability in KL/sampling
        x = x.astype(jnp.float32)
        mu, logvar = jnp.split(x, 2, axis=-1)

        return mu, logvar


class DiseaseAttentionHead(nn.Module):
    """
    Spatial cross-attention head using a learned disease prototype query.

    A learned disease prototype vector q attends over spatial backbone features,
    producing an explicit (B, H, W) attention map that highlights the spatial
    regions most relevant to a specific disease.  The attended features are then
    passed to a ConvHead to produce μ and log_σ for the variational posterior.

    Mechanism:
        1. Project backbone features h → keys K: (B, HW, query_dim)
        2. Dot-product attention with the learned query q: (query_dim,)
           → softmax attention weights over all HW spatial positions: (B, HW)
        3. Reshape weights to spatial map A: (B, H, W)
        4. Re-scale A by HW so uniform attention → weight 1.0 per position;
           gate features: h_attended = h * A[:,:,:,None]
        5. ConvHead(h_attended) → (mu, logvar)

    At initialisation the query is near-zero, so attention is near-uniform and
    h_attended ≈ h — identical behaviour to the baseline ConvHead. The model
    then gradually learns to concentrate attention on disease-relevant regions.

    Attributes:
        out_channels:     Latent channels for mu/logvar output.
        hidden_channels:  ConvHead hidden width (default 512).
        query_dim:        Dimensionality of the disease prototype vector (default 256).
        dtype:            Compute dtype (default float32).
    """
    out_channels: int
    hidden_channels: int = 512
    query_dim: int = 256
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray):
        """
        Args:
            h: Spatial backbone features (B, H, W, C)

        Returns:
            mu:       Posterior mean     (B, H, W, out_channels) in float32
            logvar:   Posterior log-var  (B, H, W, out_channels) in float32
            attn_map: Spatial attention  (B, H, W) — rescaled so uniform ≈ 1.0.
                      Visualise as a heatmap overlaid on the input image.
        """
        B, H, W, C = h.shape
        HW = H * W

        # Learned disease prototype query — one vector per head instance.
        # Small normal init avoids large initial attention asymmetry.
        q = self.param('disease_query', nn.initializers.normal(0.02), (self.query_dim,))
        # Shape: (query_dim,) — will broadcast against (B, HW, query_dim) via einsum

        # Project spatial features to key vectors
        h_flat = h.reshape(B, HW, C).astype(self.dtype)          # (B, HW, C)
        K = nn.Dense(self.query_dim, use_bias=False,
                     dtype=self.dtype, name='key_proj')(h_flat)   # (B, HW, query_dim)

        # Scaled dot-product: (B, HW, query_dim) · (query_dim,) → (B, HW)
        scale = jnp.sqrt(jnp.array(self.query_dim, dtype=self.dtype))
        attn_logits = jnp.einsum('bkd,d->bk', K, q.astype(self.dtype)) / scale  # (B, HW)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)                      # (B, HW)

        # Reshape to spatial attention map (float32 for clean visualisation)
        attn_map = attn_weights.reshape(B, H, W).astype(jnp.float32)  # (B, H, W)

        # Rescale: softmax weights sum to 1, so mean = 1/HW.
        # Multiply by HW → at uniform attention each position gets weight ≈ 1.0,
        # so h_attended ≈ h (no change at init).  Concentrated regions get > 1.
        h_attended = h * (attn_map[:, :, :, None].astype(self.dtype) * HW)  # (B, H, W, C)

        # ConvHead on attention-gated features → mu, logvar
        mu, logvar = ConvHead(
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            dtype=self.dtype,
            name='conv_head',
        )(h_attended)

        return mu, logvar, attn_map


class SepVAEEncoder(nn.Module):
    """
    Three-head encoder with frozen ResNet-50 backbone for spatial latents.

    Supports two modes:
    - Legacy (use_fpn=False): Single-scale features from layer4 (bilinear 16→64)
    - FPN (use_fpn=True): Multi-scale features fused via Feature Pyramid Network

    Supports partial unfreezing:
    - frozen_backbone=True, unfreeze_from=None: Fully frozen (default)
    - frozen_backbone=True, unfreeze_from='layer3': Unfreeze layer3+layer4
    - frozen_backbone=True, unfreeze_from='layer4': Unfreeze layer4 only
    - frozen_backbone=False: Nothing frozen

    Attributes:
        z_channels_common: Common latent channels (default: 4)
        z_channels_disease: Disease-specific latent channels (default: 2)
        frozen_backbone: Whether to freeze backbone gradients (default: True)
        use_fpn: Whether to use Feature Pyramid Network (default: False)
        fpn_channels: FPN output channels (default: 512)
        unfreeze_from: Layer name to unfreeze from ('layer3' or 'layer4', default: None)
        dtype: Compute dtype for mixed precision (default: float32).
               Backbone always runs in float32; dtype is applied to FPN and ConvHeads.
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True
    use_fpn: bool = False
    fpn_channels: int = 512
    unfreeze_from: str = None
    use_label_attention: bool = False
    attn_query_dim: int = 256
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """Initialize backbone, optional FPN, and three Conv heads (or attention heads)."""
        self.backbone = ResNet50CheSS()  # Always float32 (frozen weights + BatchNorm)
        if self.use_fpn:
            self.fpn = FPN(out_channels=self.fpn_channels, dtype=self.dtype)
        self.head_common = ConvHead(out_channels=self.z_channels_common, dtype=self.dtype, name='head_common')
        if self.use_label_attention:
            self.head_cardiomegaly = DiseaseAttentionHead(
                out_channels=self.z_channels_disease,
                query_dim=self.attn_query_dim,
                dtype=self.dtype,
                name='head_cardiomegaly',
            )
            self.head_effusion = DiseaseAttentionHead(
                out_channels=self.z_channels_disease,
                query_dim=self.attn_query_dim,
                dtype=self.dtype,
                name='head_effusion',
            )
        else:
            self.head_cardiomegaly = ConvHead(out_channels=self.z_channels_disease, dtype=self.dtype, name='head_cardiomegaly')
            self.head_effusion = ConvHead(out_channels=self.z_channels_disease, dtype=self.dtype, name='head_effusion')

    def _apply_selective_freeze(self, features: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Apply stop_gradient selectively based on unfreeze_from setting.

        When unfreeze_from='layer3': freeze layer2, unfreeze layer3+layer4
        When unfreeze_from='layer4': freeze layer2+layer3, unfreeze layer4
        When unfreeze_from=None: freeze everything
        """
        if not self.frozen_backbone or self.unfreeze_from is None:
            if self.frozen_backbone:
                # Fully frozen
                return jax.tree_util.tree_map(jax.lax.stop_gradient, features)
            else:
                # Nothing frozen
                return features

        frozen = {}
        if self.unfreeze_from == 'layer3':
            frozen['layer2'] = jax.lax.stop_gradient(features['layer2'])
            frozen['layer3'] = features['layer3']  # Trainable
            frozen['layer4'] = features['layer4']  # Trainable
        elif self.unfreeze_from == 'layer4':
            frozen['layer2'] = jax.lax.stop_gradient(features['layer2'])
            frozen['layer3'] = jax.lax.stop_gradient(features['layer3'])
            frozen['layer4'] = features['layer4']  # Trainable
        else:
            frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, features)
        return frozen

    def __call__(self, x, train: bool = True):
        """
        Forward pass through encoder.

        Args:
            x: Input images (B, 512, 512, 1)
            train: Training mode (unused, for API compatibility)

        Returns:
            Dict with keys 'common', 'cardiomegaly', 'effusion'
            Each value is a tuple (mu, logvar) with shape (B, 64, 64, channels) in float32
        """
        if self.use_fpn:
            # Multi-scale feature extraction (backbone always float32)
            features = self.backbone(x, return_multiscale=True)
            features = self._apply_selective_freeze(features)
            # Cast backbone features to compute dtype at the boundary
            features = jax.tree_util.tree_map(lambda f: f.astype(self.dtype), features)
            h = self.fpn(features)  # (B, 64, 64, fpn_channels) in compute dtype
        else:
            # Legacy: single-scale features from layer4 upsampled to 64×64.
            # When unfreeze_from is set, skip stop_gradient so that layer4 (the
            # only available scale here) remains trainable.  Without this, the
            # non-FPN path silently ignores unfreeze_from and fully freezes the
            # backbone regardless of the flag.
            h = self.backbone(x, return_spatial=True)
            if self.frozen_backbone and self.unfreeze_from is None:
                h = jax.lax.stop_gradient(h)
            h = h.astype(self.dtype)

        # Common head — always a ConvHead
        mu_c, logvar_c = self.head_common(h)

        # Disease heads — ConvHead or DiseaseAttentionHead depending on flag
        if self.use_label_attention:
            mu_cardio, logvar_cardio, attn_cardio = self.head_cardiomegaly(h)
            mu_effusion, logvar_effusion, attn_effusion = self.head_effusion(h)
            return {
                'common': (mu_c, logvar_c),
                'cardiomegaly': (mu_cardio, logvar_cardio),
                'effusion': (mu_effusion, logvar_effusion),
                'attn_maps': {
                    'cardiomegaly': attn_cardio,   # (B, H, W) — diagnostic output
                    'effusion':     attn_effusion,  # (B, H, W)
                },
            }
        else:
            mu_cardio, logvar_cardio = self.head_cardiomegaly(h)
            mu_effusion, logvar_effusion = self.head_effusion(h)
            return {
                'common': (mu_c, logvar_c),
                'cardiomegaly': (mu_cardio, logvar_cardio),
                'effusion': (mu_effusion, logvar_effusion),
            }


def apply_head_nulling(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    key: jax.random.PRNGKey,
    sigma_inactive: float = 0.1,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Apply disease-specific soft head nulling based on labels (spatial version).

    Soft nulling replaces inactive disease channels with samples from the
    prior N(0, sigma_inactive²) instead of hard zeros. This:
    - Produces a continuous latent distribution (no delta at zero)
    - Is consistent with the conditional KL prior
    - Handles label noise gracefully
    - Makes downstream LDM training viable (smooth marginal q(z))

    Nulling logic:
    - label=0 (Normal): Both disease heads → sample from prior
    - label=1 (Pleural Effusion): Cardiomegaly head → sample from prior
    - label=2 (Cardiomegaly): Pleural Effusion head → sample from prior

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar) with shape (B, 64, 64, channels)
        labels: Disease labels (B,) with values in {0, 1, 2}
        key: JAX PRNG key for sampling
        sigma_inactive: Std dev for inactive channel prior (default: 0.1)

    Returns:
        z_concat: Concatenated spatial latents (B, 64, 64, 8) with soft nulling
        inactive_mus: Dict of masked (inactive) means for nulling loss
    """
    B = labels.shape[0]

    # Sample from common head (always active)
    mu_c, logvar_c = latents_dict['common']  # (B, 64, 64, 4)
    key_c, key = jax.random.split(key)
    eps_c = jax.random.normal(key_c, mu_c.shape)
    z_c = mu_c + jnp.exp(0.5 * logvar_c) * eps_c

    # Sample from disease heads
    mu_cardio, logvar_cardio = latents_dict['cardiomegaly']  # (B, 64, 64, 2)
    mu_effusion, logvar_effusion = latents_dict['effusion']  # (B, 64, 64, 2)

    key_cardio, key_effusion, key_noise_c, key_noise_e = jax.random.split(key, 4)
    eps_cardio = jax.random.normal(key_cardio, mu_cardio.shape)
    eps_effusion = jax.random.normal(key_effusion, mu_effusion.shape)

    z_cardio = mu_cardio + jnp.exp(0.5 * logvar_cardio) * eps_cardio
    z_effusion = mu_effusion + jnp.exp(0.5 * logvar_effusion) * eps_effusion

    # Create nulling masks for spatial tensors
    # Cardiomegaly head is active only when label=2
    mask_cardio = jnp.where(labels == 2, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Effusion head is active only when label=1
    mask_effusion = jnp.where(labels == 1, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Soft nulling: inactive channels get samples from prior N(0, sigma_inactive²)
    # instead of hard zeros. This maintains a continuous latent distribution.
    noise_cardio = sigma_inactive * jax.random.normal(key_noise_c, z_cardio.shape)
    noise_effusion = sigma_inactive * jax.random.normal(key_noise_e, z_effusion.shape)

    z_cardio_masked = z_cardio * mask_cardio + noise_cardio * (1.0 - mask_cardio)
    z_effusion_masked = z_effusion * mask_effusion + noise_effusion * (1.0 - mask_effusion)

    # Concatenate along channel dimension: [common(4), cardio(2), effusion(2)] = 8 channels
    z_concat = jnp.concatenate([z_c, z_cardio_masked, z_effusion_masked], axis=-1)  # (B, 64, 64, 8)

    # Track inactive means for nulling loss (still penalize encoder mu toward 0)
    inactive_mus = {
        'cardiomegaly': mu_cardio * (1.0 - mask_cardio),
        'effusion': mu_effusion * (1.0 - mask_effusion),
    }

    return z_concat, inactive_mus


class SepVAEDecoder(nn.Module):
    """
    Shared decoder that reconstructs 512×512 images from spatial latents.

    Architecture:
        z(64×64×8) → Process → Upsample (3×) → 128×128 → 256×256 → 512×512
                   → Conv1×1 → Sigmoid → Output (512×512×1)

    Each upsampling stage uses:
    - ResBlocks (2 per stage)
    - Anti-aliased upsampling (bilinear or subpixel)
    - GroupNorm + Swish activation

    IMPORTANT: Channel schedule is HIGH→LOW (more capacity at lower resolutions):
        64×64: 256ch → 128×128: 128ch → 256×256: 64ch → 512×512: 32ch

    This ensures the decoder has enough capacity to process the rich spatial
    latent information before expanding to full resolution.

    Attributes:
        ch_mults: Channel multipliers indexed LOW→HIGH resolution (default: 64, 128, 256, 512)
                  The decoder iterates in reverse: 512ch@64×64 → 64ch@512×512
        num_res_blocks: ResBlocks per stage (default: 2)
        dropout: Dropout rate (default: 0.0)
        z_channels: Total latent channels (4 + 2 + 2 = 8)
        dtype: Compute dtype for mixed precision (default: float32)
        use_remat: Enable gradient checkpointing on ResBlocks (default: False)
        upsample_method: Upsampling interpolation method (default: 'bilinear')
            - 'bilinear': Smooth interpolation (recommended, matches FPN)
            - 'subpixel': Learned pixel-shuffle (best quality, more compute)
            - 'nearest': Legacy nearest-neighbor (causes checkerboard artifacts)
    """
    ch_mults: Sequence[int] = (64, 128, 256, 512)  # LOW→HIGH res: 512×512(64ch) ... 64×64(512ch)
    num_res_blocks: int = 2
    dropout: float = 0.0
    z_channels: int = 12  # 4 (common) + 4 (cardio) + 4 (effusion)
    dtype: jnp.dtype = jnp.float32
    use_remat: bool = False
    upsample_method: str = 'bilinear'  # 'bilinear', 'subpixel', or 'nearest'

    @nn.compact
    def __call__(self, z, train: bool = True):
        """
        Decode spatial latents to reconstructed images.

        Args:
            z: Spatial latents (B, 64, 64, 8)
            train: Training mode for dropout

        Returns:
            Reconstructed images (B, 512, 512, 1) in [0, 1] range
        """
        # Select ResBlock class — optionally wrapped with gradient checkpointing
        block_cls = nn.remat(ResBlock) if self.use_remat else ResBlock

        # Cast input to compute dtype
        h = z.astype(self.dtype)

        # Initial processing: project from z_channels to highest channel count
        # Start with ch_mults[-1] = 256 channels at 64×64 for maximum capacity
        init_ch = self.ch_mults[-1]
        h = nn.Conv(init_ch, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, name='z_proj')(h)

        # Progressive upsampling: 64×64 → 128×128 → 256×256 → 512×512
        # Iterate in reverse through ch_mults (high channels → low channels)
        # Stage i uses ch_mults[i] channels, then upsamples to ch_mults[i-1]
        for i in reversed(range(len(self.ch_mults))):
            ch = self.ch_mults[i]

            # ResBlocks at current resolution
            for j in range(self.num_res_blocks):
                h = block_cls(ch=ch, dropout=self.dropout, dtype=self.dtype)(h, train=train)

            # Upsample to next (higher) resolution, except at final stage
            if i > 0:
                target_ch = self.ch_mults[i - 1]
                h = SmoothUp(ch=target_ch, dtype=self.dtype, method=self.upsample_method)(h)
            # i=3: 256ch @ 64×64   → SmoothUp(128) → 128×128
            # i=2: 128ch @ 128×128 → SmoothUp(64)  → 256×256
            # i=1: 64ch  @ 256×256 → SmoothUp(32)  → 512×512
            # i=0: 32ch  @ 512×512 → no upsample (final resolution)

        # Final layers: GroupNorm → Swish → Conv → Sigmoid
        # Cast back to float32 for numerically stable sigmoid and loss computation
        h = h.astype(jnp.float32)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='conv_out')(h)
        h = nn.sigmoid(h)  # Output in [0, 1]

        return h


class SepVAE(nn.Module):
    """
    Complete SepVAE model with frozen backbone, three heads, and shared decoder.

    This is the main model that combines:
    - SepVAEEncoder: ResNet-50 backbone + optional FPN + 3 Conv heads → 64×64×8
    - Head nulling logic
    - SepVAEDecoder: Progressive upsampling decoder with anti-aliasing

    Spatial latents: 64×64×8 for LDM training
    - Common: 4 channels
    - Cardiomegaly: 2 channels
    - Pleural Effusion: 2 channels

    Attributes:
        z_channels_common: Common latent channels (default: 4)
        z_channels_disease: Disease-specific latent channels (default: 2)
        frozen_backbone: Whether to freeze backbone (default: True)
        use_fpn: Whether to use Feature Pyramid Network (default: False)
        fpn_channels: FPN output channels (default: 512)
        unfreeze_from: Layer to unfreeze from ('layer3', 'layer4', or None)
        sigma_inactive: Std dev for soft nulling prior on inactive channels (default: 0.1)
        dtype: Compute dtype for mixed precision (default: float32)
        use_remat: Enable gradient checkpointing (default: False)
        upsample_method: Decoder upsampling method (default: 'bilinear')
            - 'bilinear': Smooth interpolation (recommended, prevents checkerboard)
            - 'subpixel': Learned pixel-shuffle (best quality, more compute)
            - 'nearest': Legacy (causes checkerboard artifacts - not recommended)
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True
    use_fpn: bool = False
    fpn_channels: int = 512
    unfreeze_from: str = None
    sigma_inactive: float = 0.1
    use_label_attention: bool = False
    attn_query_dim: int = 256
    dtype: jnp.dtype = jnp.float32
    use_remat: bool = False
    upsample_method: str = 'bilinear'

    def setup(self):
        """Initialize encoder and decoder."""
        self.encoder = SepVAEEncoder(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            frozen_backbone=self.frozen_backbone,
            use_fpn=self.use_fpn,
            fpn_channels=self.fpn_channels,
            unfreeze_from=self.unfreeze_from,
            use_label_attention=self.use_label_attention,
            attn_query_dim=self.attn_query_dim,
            dtype=self.dtype,
        )
        self.decoder = SepVAEDecoder(
            z_channels=self.z_channels_common + 2 * self.z_channels_disease,
            dtype=self.dtype,
            use_remat=self.use_remat,
            upsample_method=self.upsample_method,
        )

    def __call__(self, x, labels, *, key, train: bool = True):
        """
        Full forward pass: encode → null → decode.

        Args:
            x: Input images (B, 512, 512, 1) in [-1, 1] range
            labels: Disease labels (B,) in {0: Normal, 1: Effusion, 2: Cardio}
            key: JAX PRNG key for sampling
            train: Training mode

        Returns:
            x_rec: Reconstructed images (B, 512, 512, 1) in [0, 1] range
            latents_dict: Dict with encoder outputs (mu, logvar) for each head.
                          When use_label_attention=True also contains key 'attn_maps'
                          → {'cardiomegaly': (B,H,W), 'effusion': (B,H,W)}.
            inactive_mus: Dict with masked inactive means for nulling loss
        """
        # Encode
        latents_dict = self.encoder(x, train=train)

        # Sample spatial latents with head nulling
        key_sample, key_dec = jax.random.split(key)
        z_concat, inactive_mus = apply_head_nulling(
            latents_dict, labels, key_sample, sigma_inactive=self.sigma_inactive
        )

        # Decode
        x_rec = self.decoder(z_concat, train=train)

        return x_rec, latents_dict, inactive_mus

    def encode(self, x):
        """
        Encode images to spatial latent distributions (no sampling).

        Args:
            x: Input images (B, 512, 512, 1)

        Returns:
            latents_dict: Dict with (mu, logvar) for each head
        """
        return self.encoder(x, train=False)

    def extract_backbone_features(self, x):
        """
        Extract raw spatial features from the frozen CheSS backbone.

        Args:
            x: Input images (B, 512, 512, 1)

        Returns:
            Spatial features (B, 64, 64, 2048) before ConvHeads
        """
        h = self.encoder.backbone(x, return_spatial=True)
        if self.frozen_backbone:
            h = jax.lax.stop_gradient(h)
        return h

    def extract_multiscale_features(self, x):
        """
        Extract multi-scale features from the backbone (for perceptual loss).

        Args:
            x: Input images (B, 512, 512, 1) in [-1, 1] range

        Returns:
            Dict with 'layer2', 'layer3', 'layer4' feature maps
        """
        features = self.encoder.backbone(x, return_multiscale=True)
        # Always stop gradient for perceptual loss (feature extractor only)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, features)

    def decode(self, z):
        """
        Decode spatial latents to images.

        Args:
            z: Spatial latents (B, 64, 64, 8)

        Returns:
            Reconstructed images (B, 512, 512, 1) in [0, 1]
        """
        return self.decoder(z, train=False)


# Testing utilities
def test_sepvae():
    """Test SepVAE model with dummy inputs, including mixed precision, remat, and upsampling methods."""
    print("Testing SepVAE model (Spatial Latents)...")

    configs = [
        {'use_fpn': False, 'dtype': jnp.float32, 'use_remat': False, 'upsample_method': 'bilinear'},
        {'use_fpn': True, 'dtype': jnp.float32, 'use_remat': False, 'upsample_method': 'bilinear'},
        {'use_fpn': True, 'dtype': jnp.bfloat16, 'use_remat': True, 'upsample_method': 'bilinear'},
        {'use_fpn': True, 'dtype': jnp.float32, 'use_remat': False, 'upsample_method': 'subpixel'},
    ]

    for cfg in configs:
        use_fpn = cfg['use_fpn']
        dtype = cfg['dtype']
        use_remat = cfg['use_remat']
        upsample_method = cfg['upsample_method']
        dtype_name = 'bf16' if dtype == jnp.bfloat16 else 'fp32'
        print(f"\n--- use_fpn={use_fpn}, dtype={dtype_name}, remat={use_remat}, upsample={upsample_method} ---")

        model = SepVAE(
            z_channels_common=4,
            z_channels_disease=2,
            frozen_backbone=True,
            use_fpn=use_fpn,
            unfreeze_from='layer3' if use_fpn else None,
            dtype=dtype,
            use_remat=use_remat,
            upsample_method=upsample_method,
        )

        batch_size = 2
        x = jnp.ones((batch_size, 512, 512, 1)) * 0.5
        labels = jnp.array([0, 1])

        key = jax.random.PRNGKey(0)
        key_init, key_forward = jax.random.split(key)

        print("Initializing model...")
        variables = model.init({'params': key_init, 'dropout': key_init}, x, labels, key=key_forward, train=True)

        print(f"Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables['params'])):,}")

        # Verify params are still float32 (mixed precision only affects compute, not storage)
        param_dtypes = set(p.dtype for p in jax.tree_util.tree_leaves(variables['params']))
        print(f"Parameter dtypes: {param_dtypes}")

        print("Running forward pass...")
        x_rec, latents_dict, inactive_mus = model.apply(
            variables, x, labels, key=key_forward, train=True
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {x_rec.shape}, dtype: {x_rec.dtype}")
        print(f"Output range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")
        assert x_rec.dtype == jnp.float32, f"Output should be float32, got {x_rec.dtype}"

        print("Spatial Latent dimensions (64×64×channels):")
        for head_name, (mu, logvar) in latents_dict.items():
            print(f"  {head_name}: μ{mu.shape} ({mu.dtype}), log_σ{logvar.shape} ({logvar.dtype})")
            assert mu.dtype == jnp.float32, f"mu should be float32, got {mu.dtype}"

    print("\n✓ SepVAE test passed!")


if __name__ == "__main__":
    test_sepvae()
