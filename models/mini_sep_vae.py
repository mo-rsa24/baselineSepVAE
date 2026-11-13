"""
MiniSepVAE — lightweight 128×128 version of SepVAE for fast ablation experiments.

Architecture:
    Input (128×128×1)
    → MiniCNNBackbone (3 stride-2 stages + ResBlocks)
    → Feature map (16×16×128)
    → Three heads:
        - ConvHead / DiseaseAttentionHead → z_common  (16×16×4, mu+logvar)
        - ConvHead / DiseaseAttentionHead → z_cardio  (16×16×2, mu+logvar)
        - ConvHead / DiseaseAttentionHead → z_effusion(16×16×2, mu+logvar)
    → apply_head_nulling → z_concat (16×16×8)
    → SepVAEDecoder(ch_mults=(32,64,128,128)) → Output (128×128×1)

Key differences from full SepVAE:
    - No pretrained backbone (fully trainable CNN from scratch)
    - 128×128 input  →  16×16 latent  (vs 512×512 → 64×64)
    - Smaller head hidden_channels (64 vs 512)
    - Smaller attn_query_dim (64 vs 256)
    - Decoder: ch_mults=(32,64,128,128), num_res_blocks=1
    - No CheSS weight injection, no batch_stats, no FPN

Loss functions (losses/sep_vae_losses.py) are UNCHANGED — they operate on
latent statistics and are fully resolution-agnostic.

The latents_dict contract is identical to the full SepVAE:
    {
        'common':      (mu, logvar)  shape (B, 16, 16, 4)
        'cardiomegaly':(mu, logvar)  shape (B, 16, 16, 2)
        'effusion':    (mu, logvar)  shape (B, 16, 16, 2)
        'attn_maps':   {'cardiomegaly': (B,16,16), 'effusion': (B,16,16)}  # only if use_label_attention
    }

Note on spatial_attention_loss:
    The loss function spatial_attention_loss() accepts a latent_hw parameter
    (default 64). In train_mini_sepvae.py the caller must pass latent_hw=16.
    This is handled outside sepvae_loss() by setting weight_spatial_attn=0.0
    in SepVAELossConfig and computing the term manually.
"""

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

# Reuse these unchanged from the full codebase
from models.ae_kl import ResBlock
from models.sep_vae_jax import (
    ConvHead,
    DiseaseAttentionHead,
    SepVAEDecoder,
    apply_head_nulling,
)


# ============================================================================
# MiniCNNBackbone  —  128×128 → 16×16 feature extractor
# ============================================================================

class MiniCNNBackbone(nn.Module):
    """
    Lightweight 3-stage CNN backbone for 128×128 input.

    Each stage: stride-2 Conv (downsample) → 2 × ResBlock (feature learning).
    Channel progression: 32 → 64 → 128.  All divisible by 32, compatible
    with the GroupNorm(num_groups=32) inside ResBlock.

    Output: (B, 16, 16, 128) spatial feature map.
    """
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input images (B, 128, 128, 1) in [-1, 1] range
            train: Training mode (unused at the moment, kept for API parity)

        Returns:
            Feature map (B, 16, 16, 128)
        """
        # ---- Stage 0: 128×128 → 64×64, 32ch ----
        h = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2),
                    padding='SAME', dtype=self.dtype, name='stage0_down')(x)
        h = ResBlock(ch=32, dtype=self.dtype, name='stage0_res0')(h, train=train)
        h = ResBlock(ch=32, dtype=self.dtype, name='stage0_res1')(h, train=train)

        # ---- Stage 1: 64×64 → 32×32, 64ch ----
        h = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2),
                    padding='SAME', dtype=self.dtype, name='stage1_down')(h)
        h = ResBlock(ch=64, dtype=self.dtype, name='stage1_res0')(h, train=train)
        h = ResBlock(ch=64, dtype=self.dtype, name='stage1_res1')(h, train=train)

        # ---- Stage 2: 32×32 → 16×16, 128ch ----
        h = nn.Conv(128, kernel_size=(3, 3), strides=(2, 2),
                    padding='SAME', dtype=self.dtype, name='stage2_down')(h)
        h = ResBlock(ch=128, dtype=self.dtype, name='stage2_res0')(h, train=train)
        h = ResBlock(ch=128, dtype=self.dtype, name='stage2_res1')(h, train=train)

        return h.astype(jnp.float32)  # (B, 16, 16, 128)


# ============================================================================
# MiniSepVAEEncoder  —  backbone + three heads
# ============================================================================

class MiniSepVAEEncoder(nn.Module):
    """
    Encoder for MiniSepVAE.

    Runs MiniCNNBackbone to get (B,16,16,128) features, then routes through
    three independent heads to produce the latents_dict.

    When use_label_attention=True, disease heads use DiseaseAttentionHead
    (which produces spatial attention maps logged under 'attn_maps').
    Otherwise, plain ConvHead is used for all three heads.

    Attributes:
        z_channels_common: Channels for the common head (default: 4)
        z_channels_disease: Channels per disease head (default: 2)
        use_label_attention: Use DiseaseAttentionHead for disease heads
        attn_query_dim: Query dimension for attention (default: 64)
        head_hidden_channels: Hidden channels in ConvHead / DiseaseAttentionHead (default: 64)
        dtype: Compute dtype
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    use_label_attention: bool = False
    attn_query_dim: int = 64
    head_hidden_channels: int = 64
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Dict:
        """
        Args:
            x: Input images (B, 128, 128, 1)
            train: Training mode

        Returns:
            latents_dict with keys:
                'common':       (mu, logvar)  (B, 16, 16, z_channels_common)
                'cardiomegaly': (mu, logvar)  (B, 16, 16, z_channels_disease)
                'effusion':     (mu, logvar)  (B, 16, 16, z_channels_disease)
                'attn_maps':    dict           (only when use_label_attention=True)
        """
        # Extract features
        backbone = MiniCNNBackbone(dtype=self.dtype, name='backbone')
        h = backbone(x, train=train)  # (B, 16, 16, 128)

        # ---- Common head: always a plain ConvHead ----
        # ConvHead(out_channels=C) internally projects to 2C then splits → returns
        # (mu, logvar) each of shape (B, H, W, C).  Do NOT pass C*2 here.
        head_common = ConvHead(
            out_channels=self.z_channels_common,
            hidden_channels=self.head_hidden_channels,
            dtype=self.dtype,
            name='head_common',
        )
        mu_c, logvar_c = head_common(h)  # each (B, 16, 16, z_channels_common)

        # ---- Disease heads ----
        if self.use_label_attention:
            head_cardio = DiseaseAttentionHead(
                out_channels=self.z_channels_disease,
                hidden_channels=self.head_hidden_channels,
                query_dim=self.attn_query_dim,
                dtype=self.dtype,
                name='head_cardiomegaly',
            )
            head_effusion = DiseaseAttentionHead(
                out_channels=self.z_channels_disease,
                hidden_channels=self.head_hidden_channels,
                query_dim=self.attn_query_dim,
                dtype=self.dtype,
                name='head_effusion',
            )
            # DiseaseAttentionHead returns (mu, logvar, attn_map)
            mu_cardio,   logvar_cardio,   attn_cardio   = head_cardio(h)
            mu_effusion, logvar_effusion, attn_effusion = head_effusion(h)
        else:
            head_cardio = ConvHead(
                out_channels=self.z_channels_disease,
                hidden_channels=self.head_hidden_channels,
                dtype=self.dtype,
                name='head_cardiomegaly',
            )
            head_effusion = ConvHead(
                out_channels=self.z_channels_disease,
                hidden_channels=self.head_hidden_channels,
                dtype=self.dtype,
                name='head_effusion',
            )
            # ConvHead returns (mu, logvar)
            mu_cardio,   logvar_cardio   = head_cardio(h)
            mu_effusion, logvar_effusion = head_effusion(h)

        latents_dict = {
            'common':       (mu_c,       logvar_c),
            'cardiomegaly': (mu_cardio,  logvar_cardio),
            'effusion':     (mu_effusion, logvar_effusion),
        }

        if self.use_label_attention:
            latents_dict['attn_maps'] = {
                'cardiomegaly': attn_cardio,    # (B, 16, 16)
                'effusion':     attn_effusion,  # (B, 16, 16)
            }

        return latents_dict


# ============================================================================
# MiniSepVAE  —  top-level model, same interface as SepVAE
# ============================================================================

class MiniSepVAE(nn.Module):
    """
    Complete MiniSepVAE: lightweight encoder + head nulling + lightweight decoder.

    Input/output contract is identical to SepVAE so that:
    - losses/sep_vae_losses.py  requires zero changes
    - utils/sepvae_diagnostics.py works without modification
    - run/train_sep_vae.py helper functions (linear probe, disentanglement
      metrics) can be imported directly

    Decoder uses SepVAEDecoder with:
        ch_mults=(32, 64, 128, 128)  →  16×16 → 32×32 → 64×64 → 128×128
        num_res_blocks=1             →  fast compilation

    Attributes:
        z_channels_common:   Common head channels (default: 4)
        z_channels_disease:  Disease head channels (default: 2)
        sigma_inactive:      Std dev for inactive head soft nulling (default: 0.1)
        use_label_attention: Use DiseaseAttentionHead (default: False)
        attn_query_dim:      Attention query dimension (default: 64)
        head_hidden_channels:Inner channels of ConvHead (default: 64)
        dtype:               Compute dtype
        upsample_method:     Decoder upsampling ('bilinear', 'nearest', 'subpixel')
    """
    z_channels_common:   int  = 4
    z_channels_disease:  int  = 2
    sigma_inactive:      float = 0.1
    use_label_attention: bool  = False
    attn_query_dim:      int  = 64
    head_hidden_channels: int  = 64
    dtype:               jnp.dtype = jnp.float32
    upsample_method:     str  = 'bilinear'

    def setup(self):
        self.encoder = MiniSepVAEEncoder(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            use_label_attention=self.use_label_attention,
            attn_query_dim=self.attn_query_dim,
            head_hidden_channels=self.head_hidden_channels,
            dtype=self.dtype,
        )
        # 16×16 → 32×32 → 64×64 → 128×128  (3 upsamples, ch_mults needs 4 entries)
        # All channel counts (32, 64, 128) divisible by 32 → GroupNorm(32) in ResBlock is safe
        self.decoder = SepVAEDecoder(
            ch_mults=(32, 64, 128, 128),
            num_res_blocks=1,
            z_channels=self.z_channels_common + 2 * self.z_channels_disease,  # 8
            dtype=self.dtype,
            upsample_method=self.upsample_method,
        )

    def __call__(self, x, labels, *, key, train: bool = True):
        """
        Full forward pass: encode → null → decode.

        Args:
            x:      Input images (B, 128, 128, 1) in [-1, 1] range
            labels: Disease labels (B,) in {0: Normal, 1: Effusion, 2: Cardio}
            key:    JAX PRNG key for reparameterisation sampling
            train:  Training mode

        Returns:
            x_rec:        Reconstructed images (B, 128, 128, 1) in [0, 1]
            latents_dict: Dict with (mu, logvar) per head + optional 'attn_maps'
            inactive_mus: Dict of masked inactive means for nulling loss
        """
        latents_dict = self.encoder(x, train=train)

        key_sample, _ = jax.random.split(key)
        z_concat, inactive_mus = apply_head_nulling(
            latents_dict, labels, key_sample, sigma_inactive=self.sigma_inactive
        )

        x_rec = self.decoder(z_concat, train=train)
        return x_rec, latents_dict, inactive_mus

    def encode(self, x):
        """
        Encode images to latent distributions (no sampling).

        Args:
            x: Input images (B, 128, 128, 1)

        Returns:
            latents_dict with (mu, logvar) per head
        """
        return self.encoder(x, train=False)

    def decode(self, z):
        """
        Decode spatial latents to images.

        Args:
            z: Spatial latents (B, 16, 16, 8)

        Returns:
            Reconstructed images (B, 128, 128, 1) in [0, 1]
        """
        return self.decoder(z, train=False)


# ============================================================================
# Smoke test
# ============================================================================

def test_mini_sepvae():
    """Quick smoke test — verifies shapes and forward pass."""
    import numpy as np
    print("Testing MiniSepVAE...")

    B = 4
    x = jnp.ones((B, 128, 128, 1)) * 0.1
    labels = jnp.array([0, 1, 2, 1])
    key = jax.random.PRNGKey(0)

    for use_attn in (False, True):
        tag = "with attention" if use_attn else "no attention"
        model = MiniSepVAE(use_label_attention=use_attn)
        variables = model.init(key, x, labels, key=key)

        n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables))
        print(f"  [{tag}] params: {n_params:,}")

        x_rec, latents_dict, inactive_mus = model.apply(variables, x, labels, key=key)

        assert x_rec.shape == (B, 128, 128, 1), f"Expected (B,128,128,1), got {x_rec.shape}"
        for head in ('common', 'cardiomegaly', 'effusion'):
            mu, logvar = latents_dict[head]
            expected_ch = 4 if head == 'common' else 2
            assert mu.shape == (B, 16, 16, expected_ch), \
                f"[{head}] mu shape {mu.shape} != (B,16,16,{expected_ch})"

        if use_attn:
            assert 'attn_maps' in latents_dict
            for d in ('cardiomegaly', 'effusion'):
                assert latents_dict['attn_maps'][d].shape == (B, 16, 16), \
                    f"attn_map[{d}] bad shape"

        # encode / decode round-trip
        latents2 = model.apply(variables, x, method=model.encode)
        mu_c, _ = latents2['common']
        z_dummy = jnp.zeros((B, 16, 16, 8))
        x_dec = model.apply(variables, z_dummy, method=model.decode)
        assert x_dec.shape == (B, 128, 128, 1)

        print(f"  [{tag}] ✓ all shapes OK")

    print("MiniSepVAE test passed.")


if __name__ == '__main__':
    test_mini_sepvae()
