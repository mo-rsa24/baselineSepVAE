from typing import Callable
import jax
import jax.numpy as jnp
from jax import vmap
import functools

# ---------------------------
# Linear VP schedule (Song et al., ICLR 2021)
# Standard choice for SDE-based sampling.
# ---------------------------
_BETA_MIN = 0.1
_BETA_MAX = 20.0
_EPS = 1e-5  # slightly larger epsilon for stability

def beta(t: jnp.ndarray) -> jnp.ndarray:
    """
    Linear beta schedule: β(t) = β_min + t(β_max - β_min)
    Unlike cosine, this does not diverge at t=1.
    """
    return _BETA_MIN + t * (_BETA_MAX - _BETA_MIN)

def _alpha_bar_fn(t: jnp.ndarray) -> jnp.ndarray:
    """
    log ᾱ(t) = - ∫ β(s) ds from 0 to t
    Integral of linear function (a + bt) is at + 0.5bt^2.
    """
    b_diff = _BETA_MAX - _BETA_MIN
    # - (0.1t + 0.5 * 19.9 * t^2)
    log_alpha = -(_BETA_MIN * t + 0.5 * b_diff * (t ** 2))
    return jnp.exp(log_alpha)

def log_alpha_bar(t: jnp.ndarray) -> jnp.ndarray:
    """log ᾱ(t)"""
    b_diff = _BETA_MAX - _BETA_MIN
    return -(_BETA_MIN * t + 0.5 * b_diff * (t ** 2))

def _alpha_fn(t: jnp.ndarray) -> jnp.ndarray:
    """α(t) = sqrt(ᾱ(t))"""
    return jnp.sqrt(_alpha_bar_fn(t))

def marginal_prob_std(t: jnp.ndarray) -> jnp.ndarray:
    """
    σ(t) = sqrt(1 - ᾱ(t))
    """
    return jnp.sqrt(jnp.clip(1.0 - _alpha_bar_fn(t), _EPS, 1.0))

def diffusion_coeff(t: jnp.ndarray) -> jnp.ndarray:
    """
    g(t) = sqrt(β(t)) for VP-SDE.
    """
    return jnp.sqrt(beta(t))

# ---------------------------
# Utilities and Estimators
# ---------------------------

def _sum_except_batch(x):
    """Sum over all non-batch axes, keep batch axis."""
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)

@functools.partial(jax.jit, static_argnums=(2,))
def score_function_hutchinson_estimator(x, t, score_fn, params, key):
    v = jax.random.normal(key, x.shape)
    def epsilon_fn(y):
        return score_fn({'params': params}, y, t)
    _, jvp_val = jax.jvp(epsilon_fn, (x,), (v,))
    sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
    divergence = -jnp.sum(v * jvp_val, axis=(1, 2, 3)) / sigma_t.squeeze()
    return divergence, divergence

@jax.jit
def get_kappa(t, divlogs, scores):
    """
    Calculates the optimal mixing coefficient kappa for Superdiffusion.
    """
    div1, div2 = divlogs
    s1, s2 = scores
    div1 = div1[:, None, None, None]
    div2 = div2[:, None, None, None]
    g_t_squared = diffusion_coeff_fn(t)[:, None, None, None]**2
    numerator = g_t_squared * (div1 - div2) + _sum_except_batch(s1 * (s1 - s2))
    denominator = _sum_except_batch((s1 - s2)**2) + 1e-12
    kappa = numerator / denominator
    return kappa

# Vectorized (batch) versions used everywhere
marginal_prob_std_fn = vmap(marginal_prob_std)
diffusion_coeff_fn   = vmap(diffusion_coeff)
alpha_fn             = vmap(_alpha_fn)
alpha_bar_fn         = vmap(_alpha_bar_fn)

def sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)