import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple


class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        w = self.param("W", jax.nn.initializers.normal(stddev=self.scale), (self.embed_dim // 2,))
        w = w.astype(self.dtype)
        w = jax.lax.stop_gradient(w)
        x_proj = x[:, None] * w[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class DenseToMap(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features, dtype=self.dtype, param_dtype=self.param_dtype)(x)[:, None, None, :]


def _pick_gn_groups(channels: int) -> int:
    groups = min(32, channels)
    while groups > 1 and (channels % groups) != 0:
        groups //= 2
    return max(1, groups)


class ResBlock(nn.Module):
    channels: int
    embed_dim: int
    scale_skip: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_channels = x.shape[-1]
        h = nn.GroupNorm(num_groups=_pick_gn_groups(in_channels), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = act(h)
        h = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = h + DenseToMap(self.channels, dtype=self.dtype, param_dtype=self.param_dtype)(t_embed)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.channels), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = act(h)
        h = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        if in_channels != self.channels:
            x = nn.Conv(
                self.channels,
                (1, 1),
                padding="SAME",
                use_bias=False,
                name="skip_proj",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
        if self.scale_skip:
            x = x * (1.0 / jnp.sqrt(2.0))
        return act(h + x)


class SelfAttention2D(nn.Module):
    num_heads: int = 4
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape
        h = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = h.reshape((batch, height * width, channels))
        h = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = h.reshape((batch, height, width, channels))
        return x + h


class ConditionalScoreNet(nn.Module):
    z_channels: int = 16
    cond_channels: int = 16
    channels: Sequence[int] = (128, 256, 512)
    embed_dim: int = 256
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    num_heads: int = 4
    use_remat: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t, z_common):
        if x.shape[:-1] != z_common.shape[:-1]:
            raise ValueError(f"x and z_common must share spatial dimensions, got {x.shape} and {z_common.shape}")

        x = x.astype(self.dtype)
        z_common = z_common.astype(self.dtype)
        t = t.astype(self.dtype)

        act = nn.swish
        temb = act(
            nn.Dense(
                self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype
            )(GaussianFourierProjection(self.embed_dim, dtype=self.dtype)(t))
        )

        h = jnp.concatenate([x, z_common], axis=-1)
        h = nn.Conv(self.channels[0], (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
        skips = [h]

        res_block = ResBlock
        attn_block = SelfAttention2D
        if self.use_remat:
            res_block = nn.remat(ResBlock)
            attn_block = nn.remat(SelfAttention2D)

        for i, channels in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                h = res_block(channels, self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = attn_block(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
                skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.Conv(
                    self.channels[i + 1],
                    (3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(h)
                skips.append(h)

        h = res_block(self.channels[-1], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
        h = attn_block(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = res_block(self.channels[-1], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)

        for i in reversed(range(len(self.channels))):
            for _ in range(self.num_res_blocks + 1):
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = res_block(self.channels[i], self.embed_dim, dtype=self.dtype, param_dtype=self.param_dtype)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(h)
            if i > 0:
                h = nn.ConvTranspose(
                    self.channels[i - 1],
                    (4, 4),
                    strides=(2, 2),
                    padding="SAME",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(h)

        h = nn.GroupNorm(num_groups=_pick_gn_groups(h.shape[-1]), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = act(h)
        return nn.Conv(
            self.z_channels,
            (3, 3),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(h)
