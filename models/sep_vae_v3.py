"""
SepVAE V3 — binary compositional foreground/background VAE with separate decoders.

Latent semantics:
  z_common : non-heart anatomy / background for every sample
  z_heart  : heart foreground for every sample, including normals
  s_ctr    : deterministic scalar control predicted from the heart branch

Decoder semantics:
  f_common = D_common(z_common, skip_common)
  f_heart, alpha_logits = D_heart([z_heart, s_ctr], skip_heart)
  alpha_heart = sigmoid(alpha_logits)
  f_blend = (1 - alpha_heart) * f_common + alpha_heart * f_heart
  x_hat = Render(f_blend)

This keeps V2 intact while making the compositional routing explicit in V3.
"""

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.ae_kl import SelfAttention2D
from models.sep_vae_jax import SmoothUp
from models.sep_vae_v2 import ConvHeadGN, Layer4BranchGN, ResBlockSE, ResNet50Scratch


def _resize_mask(mask: jnp.ndarray, shape: Tuple[int, int, int, int]) -> jnp.ndarray:
    """Resize a binary heart mask to NHWC `shape` using nearest-neighbour sampling."""
    return jax.image.resize(mask[..., None], shape, method="nearest")


def _reparameterize(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    key: Optional[jax.random.PRNGKey],
    sample: bool,
) -> jnp.ndarray:
    if not sample:
        return mu
    if key is None:
        raise ValueError("Sampling latents requires a PRNG key.")
    eps = jax.random.normal(key, mu.shape)
    return mu + jnp.exp(0.5 * logvar) * eps


class CTRRegressor(nn.Module):
    """Predict a bounded CTR scalar from heart-branch features."""
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h_heart: jnp.ndarray) -> jnp.ndarray:
        pooled = jnp.mean(h_heart, axis=(1, 2))
        x = nn.Dense(self.hidden_dim, name="fc1")(pooled)
        x = nn.relu(x)
        x = nn.Dense(1, name="fc2")(x)
        return jax.nn.sigmoid(x[:, 0])


class SepVAEEncoderV3(nn.Module):
    """Shared trunk, mask-routed branches, Gaussian spatial latents, deterministic CTR."""
    z_channels_common: int = 16
    z_channels_heart: int = 16
    attn_heads: int = 4

    def setup(self):
        self.backbone = nn.remat(ResNet50Scratch)(attn_heads=self.attn_heads)
        self.common_branch = nn.remat(Layer4BranchGN)()
        self.heart_branch = nn.remat(Layer4BranchGN)()
        self.common_head = ConvHeadGN(
            out_channels=self.z_channels_common,
            name="common_head",
        )
        self.heart_head = ConvHeadGN(
            out_channels=self.z_channels_heart,
            name="heart_head",
        )
        self.ctr_head = CTRRegressor(name="ctr_head")

    def __call__(self, x: jnp.ndarray, heart_mask: jnp.ndarray, train: bool = True) -> Dict:
        if heart_mask is None:
            raise ValueError("SepVAEV3 encoder requires `heart_mask`.")

        bsz = x.shape[0]
        h_shared, h_layer2 = self.backbone(x)

        hm_16 = _resize_mask(
            heart_mask,
            (bsz, h_shared.shape[1], h_shared.shape[2], 1),
        )
        hm_32 = _resize_mask(
            heart_mask,
            (bsz, h_layer2.shape[1], h_layer2.shape[2], 1),
        )

        h_common_in = h_shared * (1.0 - hm_16)
        h_heart_in = h_shared * hm_16

        h_common = self.common_branch(h_common_in)
        h_heart = self.heart_branch(h_heart_in)

        mu_c, logvar_c = self.common_head(h_common)
        mu_h, logvar_h = self.heart_head(h_heart)
        ctr_pred = self.ctr_head(h_heart)

        return {
            "common": (mu_c, logvar_c),
            "heart": (mu_h, logvar_h),
            "ctr_pred": ctr_pred,
            "skip_common": {
                "layer3": h_shared * (1.0 - hm_16),
                "layer2": h_layer2 * (1.0 - hm_32),
            },
            "skip_heart": {
                "layer3": h_shared * hm_16,
                "layer2": h_layer2 * hm_32,
            },
        }


class SepVAEFeatureDecoderV3(nn.Module):
    """Progressive decoder that emits feature maps, with optional alpha logits."""
    ch_mults: Sequence[int] = (128, 128, 256, 512, 512)
    num_res_blocks: int = 2
    z_channels: int = 16
    feature_channels: int = 64
    predict_alpha: bool = False
    dropout: float = 0.0
    se_reduction: int = 8

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        skip_feats: Optional[Dict[str, jnp.ndarray]] = None,
        train: bool = True,
    ):
        h = nn.Conv(self.ch_mults[-1], (3, 3), padding="SAME", name="z_proj")(z)

        for i in reversed(range(len(self.ch_mults))):
            ch = self.ch_mults[i]
            if skip_feats is not None:
                if i == 4 and "layer3" in skip_feats:
                    h = nn.Conv(ch, (1, 1), use_bias=False, name="skip3_fuse")(
                        jnp.concatenate([h, skip_feats["layer3"]], axis=-1)
                    )
                elif i == 3 and "layer2" in skip_feats:
                    h = nn.Conv(ch, (1, 1), use_bias=False, name="skip2_fuse")(
                        jnp.concatenate([h, skip_feats["layer2"]], axis=-1)
                    )

            for _ in range(self.num_res_blocks):
                h = ResBlockSE(
                    ch=ch,
                    dropout=self.dropout,
                    se_reduction=self.se_reduction,
                )(h, train=train)

            if i == 3:
                h = SelfAttention2D(num_heads=4, name="dec_attn_32")(h)
            if i > 0:
                h = SmoothUp(ch=self.ch_mults[i - 1])(h)

        h = nn.GroupNorm(num_groups=32, name="out_gn")(h)
        h = nn.swish(h)
        f = nn.Conv(
            features=self.feature_channels,
            kernel_size=(3, 3),
            padding="SAME",
            name="feat_out",
        )(h)
        if not self.predict_alpha:
            return f

        alpha_logits = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            padding="SAME",
            name="alpha_out",
        )(h)
        return f, alpha_logits


class RenderHeadV3(nn.Module):
    """Shared renderer applied after branch blending."""
    feature_channels: int = 64

    @nn.compact
    def __call__(self, f: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        h = ResBlockSE(ch=self.feature_channels, se_reduction=8, name="render_res")(f, train=train)
        h = nn.GroupNorm(num_groups=32, name="render_gn")(h)
        h = nn.swish(h)
        h = nn.Conv(features=1, kernel_size=(3, 3), padding="SAME", name="render_out")(h)
        return nn.sigmoid(h)


class SepVAEV3(nn.Module):
    """Binary compositional VAE with separate common and heart decoders."""
    z_channels_common: int = 16
    z_channels_heart: int = 16
    attn_heads: int = 4
    decoder_res_blocks: int = 2
    feature_channels: int = 64
    model_version: str = "v3"

    def setup(self):
        self.encoder = SepVAEEncoderV3(
            z_channels_common=self.z_channels_common,
            z_channels_heart=self.z_channels_heart,
            attn_heads=self.attn_heads,
        )
        self.common_decoder = SepVAEFeatureDecoderV3(
            ch_mults=(128, 128, 256, 512, 512),
            num_res_blocks=self.decoder_res_blocks,
            z_channels=self.z_channels_common,
            feature_channels=self.feature_channels,
            predict_alpha=False,
            name="common_decoder",
        )
        self.heart_decoder = SepVAEFeatureDecoderV3(
            ch_mults=(128, 128, 256, 512, 512),
            num_res_blocks=self.decoder_res_blocks,
            z_channels=self.z_channels_heart + 1,
            feature_channels=self.feature_channels,
            predict_alpha=True,
            name="heart_decoder",
        )
        self.render_head = RenderHeadV3(
            feature_channels=self.feature_channels,
            name="render_head",
        )

    def __call__(
        self,
        x: jnp.ndarray,
        heart_mask: jnp.ndarray,
        *,
        key: Optional[jax.random.PRNGKey],
        train: bool = True,
        sample: bool = True,
        s_ctr_override: Optional[jnp.ndarray] = None,
    ) -> Dict:
        encoded = self.encoder(x, heart_mask, train=train)
        mu_c, logvar_c = encoded["common"]
        mu_h, logvar_h = encoded["heart"]

        if sample:
            if key is None:
                raise ValueError("SepVAEV3 sampling requires a PRNG key.")
            key_c, key_h = jax.random.split(key)
        else:
            key_c, key_h = None, None

        z_common = _reparameterize(mu_c, logvar_c, key_c, sample=sample)
        z_heart = _reparameterize(mu_h, logvar_h, key_h, sample=sample)
        s_ctr = encoded["ctr_pred"] if s_ctr_override is None else s_ctr_override

        x_hat, alpha_heart, aux = self.decode(
            z_common,
            z_heart,
            s_ctr,
            skip_common=encoded["skip_common"],
            skip_heart=encoded["skip_heart"],
            train=train,
        )

        return {
            **encoded,
            "z_common_sample": z_common,
            "z_heart_sample": z_heart,
            "s_ctr": s_ctr,
            "x_hat": x_hat,
            "alpha_heart": alpha_heart,
            "aux": aux,
        }

    def encode(self, x: jnp.ndarray, heart_mask: jnp.ndarray):
        return self.encoder(x, heart_mask, train=False)

    def decode(
        self,
        z_common: jnp.ndarray,
        z_heart: jnp.ndarray,
        s_ctr: jnp.ndarray,
        skip_common: Optional[Dict[str, jnp.ndarray]] = None,
        skip_heart: Optional[Dict[str, jnp.ndarray]] = None,
        train: bool = False,
    ):
        if s_ctr.ndim == 1:
            s_ctr = s_ctr[:, None]
        s_ctr_map = jnp.broadcast_to(
            s_ctr[:, None, None, :],
            (z_heart.shape[0], z_heart.shape[1], z_heart.shape[2], s_ctr.shape[-1]),
        )

        f_common = self.common_decoder(
            z_common,
            skip_feats=skip_common,
            train=train,
        )
        f_heart, alpha_logits = self.heart_decoder(
            jnp.concatenate([z_heart, s_ctr_map], axis=-1),
            skip_feats=skip_heart,
            train=train,
        )
        alpha_heart = jax.nn.sigmoid(alpha_logits)
        f_blend = (1.0 - alpha_heart) * f_common + alpha_heart * f_heart

        x_hat = self.render_head(f_blend, train=train)
        x_common = self.render_head(f_common, train=train)
        x_heart = self.render_head(f_heart, train=train)

        aux = {
            "alpha_logits": alpha_logits,
            "f_common": f_common,
            "f_heart": f_heart,
            "f_blend": f_blend,
            "x_common": x_common,
            "x_heart": x_heart,
        }
        return x_hat, alpha_heart, aux
