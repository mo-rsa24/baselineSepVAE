"""
Binary SepVAE: z = [z_common(16ch), z_cardio(16ch)] @ 64×64 spatial resolution.

Improvements over the original SepVAE paper:
  1. Frozen CheSS trunk (layers 1-3) + two learnable layer4 branches:
       bg_branch → z_common   (learns anatomy shared across all CXRs)
       tg_branch → z_cardio   (learns cardiomegaly-specific patterns)
     Both branches are initialised from CheSS layer4 weights and fine-tune
     independently, giving each head its own inductive bias.

  2. Hard-zero head nulling (paper-faithful):
       Normal images:  z_cardio_decode = 0   (decoder always sees s=0 for bg)
       Cardio images:  z_cardio_decode = z_cardio (sampled)
     Encoder still produces μ_ca for all images; KL pushes μ_ca → 0 for normal.
     Decoder learns p(x|c, s=0) for normal and p(x|c, s) for cardio separately.

  3. 16-channel disease latent (vs. 2):
     The FactorVAE MI discriminator operates on 16+16=32D pooled vectors —
     large enough to detect nonlinear dependencies that Barlow Twins misses.

Architecture:
    Input (512×512×1)
        → CheSS layers 1-3 (frozen)   →  h_shared (32×32×1024)
              ↓                              ↓
        bg_branch layer4 (learnable)   tg_branch layer4 (learnable)
        → h_bg (64×64×2048)            → h_tg (64×64×2048)
              ↓                              ↓
         ConvHead_common              DiseaseAttnHead_cardio
          z_common (16ch)               z_cardio (16ch)
              ↓                              ↓
              └───── hard-zero z_cardio for Normal ─────┘
                              ↓
                    concat (64×64×32)
                              ↓
                          Decoder
                              ↓
                    Output (512×512×1)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Sequence

from models.resnet_jax import ResNet50CheSS, ResNetLayer4Branch
from models.ae_kl import ResBlock


class SmoothUp(nn.Module):
    """Bilinear upsample + 2× conv to avoid checkerboard artifacts."""
    ch: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = jax.image.resize(x, (B, H * 2, W * 2, C), method='bilinear')
        h = nn.Conv(self.ch, (3, 3), padding="SAME", name='conv_up')(h)
        h = nn.Conv(self.ch, (3, 3), padding="SAME", name='smooth')(h)
        return h


class ConvHead(nn.Module):
    """Projects spatial features to latent μ and log_σ (3-conv for more depth)."""
    out_channels: int
    hidden_channels: int = 512

    @nn.compact
    def __call__(self, h):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', name='conv1')(h)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_channels // 2, (3, 3), padding='SAME', name='conv2')(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels * 2, (3, 3), padding='SAME', name='conv3')(x)
        mu, logvar = jnp.split(x, 2, axis=-1)
        return mu, logvar


class DiseaseAttentionHead(nn.Module):
    """
    Spatial cross-attention head with a learned disease prototype query.

    The query learns to attend to the anatomical region relevant to the disease.
    Combined with bbox supervision in the loss this forces the cardio head to
    focus on the cardiac silhouette — disjoint from the common-anatomy head.

    Mechanism:
        1. Project branch features h → keys K  (B, HW, query_dim)
        2. Dot-product with learned query q → softmax attention  (B, HW)
        3. Reshape to spatial map A  (B, H, W)          ← for bbox loss
        4. Gate features: h_attended = h * (A * HW)     ← near-identity at init
        5. ConvHead(h_attended) → (mu, logvar)
    """
    out_channels: int
    hidden_channels: int = 512
    query_dim: int = 256

    @nn.compact
    def __call__(self, h: jnp.ndarray):
        B, H, W, C = h.shape
        HW = H * W

        # Learned disease prototype — small init so attention starts near-uniform
        q = self.param('disease_query', nn.initializers.normal(0.02), (self.query_dim,))

        # Project spatial features to key vectors
        h_flat = h.reshape(B, HW, C)
        K = nn.Dense(self.query_dim, use_bias=False, name='key_proj')(h_flat)  # (B, HW, D)

        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.array(self.query_dim, dtype=jnp.float32))
        attn_logits  = jnp.einsum('bkd,d->bk', K, q) / scale   # (B, HW)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)      # (B, HW)

        # Spatial attention map (probability distribution over positions)
        attn_map = attn_weights.reshape(B, H, W)                 # (B, H, W)

        # Gate features: scale so uniform attention ≈ identity weight 1.0 per position
        h_attended = h * (attn_map[:, :, :, None] * HW)         # (B, H, W, C)

        mu, logvar = ConvHead(
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            name='conv_head',
        )(h_attended)

        return mu, logvar, attn_map


class SepVAEEncoder(nn.Module):
    """
    Binary encoder: frozen CheSS trunk + two learnable layer4 branches.

    bg_branch feeds ConvHead_common  → z_common  (captures shared anatomy)
    tg_branch feeds DiseaseAttnHead  → z_cardio  (captures cardio patterns)
    """
    z_channels_common:  int = 16
    z_channels_disease: int = 16
    attn_query_dim:     int = 256

    def setup(self):
        self.backbone      = ResNet50CheSS()
        self.bg_branch     = ResNetLayer4Branch()
        self.tg_branch     = ResNetLayer4Branch()
        self.head_common   = ConvHead(
            out_channels=self.z_channels_common,
            name='head_common',
        )
        self.head_cardio   = DiseaseAttentionHead(
            out_channels=self.z_channels_disease,
            query_dim=self.attn_query_dim,
            name='head_cardio',
        )

    def __call__(self, x, train: bool = True):
        # ── Frozen trunk (layers 1-3) ─────────────────────────────────────────
        h_shared = self.backbone(x, stop_at_layer3=True)   # (B, 32, 32, 1024)
        h_shared = jax.lax.stop_gradient(h_shared)

        # ── Learnable layer4 branches ─────────────────────────────────────────
        B = h_shared.shape[0]
        h_bg = self.bg_branch(h_shared)   # (B, 16, 16, 2048)
        h_tg = self.tg_branch(h_shared)   # (B, 16, 16, 2048)

        # Upsample to 64×64 (same spatial resolution as original design)
        h_bg = jax.image.resize(h_bg, (B, 64, 64, 2048), method='bilinear')
        h_tg = jax.image.resize(h_tg, (B, 64, 64, 2048), method='bilinear')

        # ── Heads ─────────────────────────────────────────────────────────────
        mu_c,  logvar_c          = self.head_common(h_bg)
        mu_ca, logvar_ca, attn_ca = self.head_cardio(h_tg)

        return {
            'common':       (mu_c,  logvar_c),
            'cardiomegaly': (mu_ca, logvar_ca),
            'attn_maps': {
                'cardiomegaly': attn_ca,   # (B, H, W) — for bbox loss
            },
        }


def apply_head_nulling(
    latents_dict: Dict,
    labels: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Hard-zero head nulling (paper-faithful).

    The encoder produces μ_ca for ALL images (KL pushes it toward 0 for Normal).
    Before decoding, z_cardio is hard-zeroed for Normal images:
        label=0 (Normal):      z_cardio_decode = 0     → decoder sees s=0
        label=1 (Cardiomegaly): z_cardio_decode = z_ca  → decoder sees s

    This cleanly separates decoder responsibilities:
      • p(x | c, s=0)  for background (normal)
      • p(x | c, s)    for target (cardiomegaly)

    Returns:
        z_concat:    (B, 64, 64, z_c + z_ca)  — decoder input (z_ca hard-zeroed for bg)
        z_c_pooled:  (B, z_c_ch)              — spatial-mean of z_common, for MI loss
        z_ca_pooled: (B, z_ca_ch)             — spatial-mean of z_cardio, for MI loss
    """
    mu_c,  logvar_c  = latents_dict['common']
    mu_ca, logvar_ca = latents_dict['cardiomegaly']

    key_c, key_ca = jax.random.split(key)

    # Reparameterisation
    z_c  = mu_c  + jnp.exp(0.5 * logvar_c)  * jax.random.normal(key_c,  mu_c.shape)
    z_ca = mu_ca + jnp.exp(0.5 * logvar_ca) * jax.random.normal(key_ca, mu_ca.shape)

    # Hard-zero: only cardio images (label=1) keep their z_ca for decoding
    mask_ca = jnp.where(labels == 1, 1.0, 0.0)[:, None, None, None]  # (B,1,1,1)
    z_ca_decode = z_ca * mask_ca   # zero for Normal images

    z_concat = jnp.concatenate([z_c, z_ca_decode], axis=-1)  # (B, 64, 64, z_c+z_ca)

    # Spatial-mean pooled latents for FactorVAE MI discriminator
    # Use raw z (not hard-zeroed) so MI measures independence of encoder outputs
    z_c_pooled  = jnp.mean(z_c,  axis=(1, 2))  # (B, z_c_ch)
    z_ca_pooled = jnp.mean(z_ca, axis=(1, 2))  # (B, z_ca_ch)

    return z_concat, z_c_pooled, z_ca_pooled


class SepVAEDecoder(nn.Module):
    """
    Decoder: z(64×64×32) → 512×512×1 via progressive bilinear upsampling.

    Channel schedule (high→low res):
        64×64: 512ch → 128×128: 256ch → 256×256: 128ch → 512×512: 64ch
    """
    ch_mults: Sequence[int] = (64, 128, 256, 512)
    num_res_blocks: int = 2
    dropout: float = 0.0
    z_channels: int = 32

    @nn.compact
    def __call__(self, z, train: bool = True):
        h = nn.Conv(self.ch_mults[-1], (3, 3), padding='SAME', name='z_proj')(z)

        for i in reversed(range(len(self.ch_mults))):
            ch = self.ch_mults[i]
            for _ in range(self.num_res_blocks):
                h = ResBlock(ch=ch, dropout=self.dropout)(h, train=train)
            if i > 0:
                h = SmoothUp(ch=self.ch_mults[i - 1])(h)

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='conv_out')(h)
        return nn.sigmoid(h)


class SepVAE(nn.Module):
    """
    Binary SepVAE (Normal vs. Cardiomegaly).

    Encoder: frozen CheSS trunk + two learnable layer4 branches
             + ConvHead (common) + DiseaseAttentionHead (cardio)
    Decoder: progressive bilinear upsampling from 64×64 latent to 512×512
    Latent:  z = [z_common(16ch), z_cardio(16ch)]  @ 64×64 spatial
    Nulling: hard-zero z_cardio for Normal images before decoding
    """
    z_channels_common:  int   = 16
    z_channels_disease: int   = 16
    attn_query_dim:     int   = 256

    def setup(self):
        self.encoder = SepVAEEncoder(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            attn_query_dim=self.attn_query_dim,
        )
        self.decoder = SepVAEDecoder(
            z_channels=self.z_channels_common + self.z_channels_disease,
        )

    def __call__(self, x, labels, *, key, train: bool = True):
        latents_dict = self.encoder(x, train=train)
        key_sample, _ = jax.random.split(key)
        z_concat, z_c_pooled, z_ca_pooled = apply_head_nulling(
            latents_dict, labels, key_sample
        )
        x_rec = self.decoder(z_concat, train=train)
        return x_rec, latents_dict, z_c_pooled, z_ca_pooled

    def encode(self, x):
        return self.encoder(x, train=False)

    def decode(self, z):
        return self.decoder(z, train=False)
