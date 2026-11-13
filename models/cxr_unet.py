# models/cxr_unet.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Sequence

class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.
    dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        W = W.astype(self.dtype)
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class DenseToMap(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features, dtype=self.dtype, param_dtype=self.param_dtype)(x)[:, None, None, :]

def _pick_gn_groups(C: int) -> int:
    g = min(32, C)
    while g > 1 and (C % g) != 0:
        g //= 2
    return max(1, g)

class ResBlock(nn.Module):
    c: int
    embed_dim: int
    scale_skip: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_ch = x.shape[-1]
        h = nn.GroupNorm(num_groups=_pick_gn_groups(in_ch), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = h + DenseToMap(self.c, dtype=self.dtype, param_dtype=self.param_dtype)(t_embed)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        if in_ch != self.c:
            x = nn.Conv(self.c, (1, 1), padding='SAME', use_bias=False, name='skip_proj',
                        dtype=self.dtype, param_dtype=self.param_dtype)(x)
        if self.scale_skip:
            x = x * (1.0 / jnp.sqrt(2.0))
        return act(h + x)


class SelfAttention2D(nn.Module):
    num_heads: int = 4
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        h = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = h.reshape((B, H*W, C))
        h = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = h.reshape((B,H,W,C))
        return x + h

class ScoreNet(nn.Module):
    z_channels: int = 3
    channels: Sequence[int] = (128, 256, 512)
    embed_dim: int = 256
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,) # Apply attention at 16x16
    num_heads: int = 4
    use_remat: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x, t):
        x = x.astype(self.dtype)
        t = t.astype(self.dtype)
        act = nn.swish
        temb = act(nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(
            GaussianFourierProjection(self.embed_dim, dtype=self.dtype)(t)
        ))
        h = nn.Conv(self.channels[0], (3, 3), padding='SAME', dtype=self.dtype, param_dtype=self.param_dtype)(x)
        skips = [h]
        res_block = ResBlock
        attention_block = SelfAttention2D
        if self.use_remat:
            res_block = nn.remat(ResBlock)
            attention_block = nn.remat(SelfAttention2D)
        for i, ch in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                h = res_block(ch, self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = attention_block(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
                skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.Conv(self.channels[i + 1], (3, 3), strides=(2, 2), padding='SAME',
                            dtype=self.dtype, param_dtype=self.param_dtype)(h)
                skips.append(h)

        # --- Bottleneck ---
        h = res_block(self.channels[-1], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
        h = attention_block(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = res_block(self.channels[-1], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)

        # --- Decoder ---
        for i in reversed(range(len(self.channels))):
            for _ in range(self.num_res_blocks + 1): # +1 for skip connections
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = res_block(self.channels[i], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads)(h)
            if i > 0:
                h = nn.ConvTranspose(self.channels[i - 1], (4, 4), strides=(2, 2), padding='SAME',
                                     dtype=self.dtype, param_dtype=self.param_dtype)(h)

        # Final projection
        h = nn.GroupNorm(num_groups=_pick_gn_groups(h.shape[-1]), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = act(h)
        out = nn.Conv(self.z_channels, (3,3), padding='SAME',
                      kernel_init=nn.initializers.zeros,
                      dtype=self.dtype, param_dtype=self.param_dtype)(h)
        return out