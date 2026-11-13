# superdiff_and.py
# Faithful JAX implementation of SuperDiff (Algorithm 1) — LOGICAL AND path
# Mirrors the architecture, shapes, and loaders used in compose.py.
# Comments reference the pseudo-code blocks in Algorithm 1 and Prop. 6 (linear system for κ).

import os, sys, json, math, functools, argparse
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import tensorflow as tf
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# --- Ensure repo root on sys.path (same as compose.py) ------------------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# --- Project modules: identical imports to compose.py -------------------------
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder

# SDE utilities (your project API)
from diffusion.vp_equation import (
    marginal_prob_std_fn,      # σ(t)
    diffusion_coeff_fn,        # g(t)
    score_function_hutchinson_estimator,  # div terms (Hutchinson)
)

# -----------------------------------------------------------------------------
# Model loading (copied/adapted from compose.py to keep behavior identical)
# -----------------------------------------------------------------------------

class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict, Dict]:
    with open(config_path, 'r') as f:
        loaded_json = json.load(f)
        meta = loaded_json.get('args', loaded_json)

    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get('args', vae_loaded_json)
    z_channels = vae_meta['z_channels']

    ldm_chans   = tuple(meta['ldm_base_ch'] * int(m) for m in meta['ldm_ch_mults'].split(','))
    attn_res    = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blk = meta['ldm_num_res_blocks']

    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=num_res_blk,
        attn_resolutions=attn_res,
    )

    rng = jax.random.PRNGKey(0)
    latent_size = meta['img_size'] // 4
    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time    = jnp.ones((1,))
    variables = ldm_model.init({'params': rng, 'dropout': rng}, fake_latents, fake_time)

    tx = optax.chain(
        optax.clip_by_global_norm(meta.get('grad_clip', 1.0)),
        optax.adamw(meta.get('lr', 3e-5), weight_decay=meta.get('weight_decay', 0.01))
    )
    dummy_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply, params=variables['params'], ema_params=variables['params'], tx=tx
    )

    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    restored = from_bytes(dummy_state, blob)

    use_ema = meta.get('use_ema', False)
    params  = restored.ema_params if (use_ema and getattr(restored, 'ema_params', None) is not None) else restored.params
    return ldm_model, params, meta

# -----------------------------------------------------------------------------
# κ-solver for LOGICAL AND (Proposition 6; two-model closed form)
# -----------------------------------------------------------------------------
def _hutch_divergence(x, t, apply_fn, params, key):
    # Estimates div(u)(x,t) = ∇·u using your existing Hutchinson helper under the hood.
    dlog, _ = score_function_hutchinson_estimator(x, t, apply_fn, params, key)  # shape: (B,)
    return dlog  # this estimates ⟨∇, u⟩ evaluated at (x,t)

def _inner_batch(v, w):
    # <v, w> over spatial+channel dims, per batch item
    return jnp.sum(v * w, axis=(1, 2, 3))  # (B,)

def kappa_solver_and(x, t, score_tb, score_norm, u_tb, u_norm, key_tb, key_norm):
    """
    Solve for κ in the AND case so that *density change rates match* across models
    (Prop. 6: enforce d log q^1_{1-τ}(x_τ) = d log q^2_{1-τ}(x_τ), plus κ_1+κ_2=1).

    For SDE dx_τ = [κ1 u^1_τ(x_τ) + κ2 u^2_τ(x_τ)] dτ + g_{1-τ} dW_τ,
    the instantaneous log-density change seen through model-i’s score is
      D_i(κ) ≈ ⟨∇, κ1 u^1 + κ2 u^2⟩  +  ⟨ score_i , κ1 u^1 + κ2 u^2 ⟩
             = κ1 A_i1 + κ2 A_i2,
    where A_ij := div(u^j) + ⟨score_i, u^j⟩ .  (All terms are per-batch scalars.)

    Enforce D_1(κ) = D_2(κ) and κ1+κ2=1  ⇒ a closed form for κ1 (per-batch):
      κ1 = ΔA_2 / (ΔA_2 - ΔA_1),  κ2 = 1 - κ1,
    where ΔA_j := A_1j - A_2j.

    We clip to [0,1] to stay on the probability simplex. Shapes: (B,) or (B,1,1,1) for broadcasting.
    """
    # Divergences of each model velocity field
    div_u1 = _hutch_divergence(x, t, score_tb,   {'params': score_tb.__self__.params} if hasattr(score_tb, "__self__") else None, key_tb)
    div_u2 = _hutch_divergence(x, t, score_norm, {'params': score_norm.__self__.params} if hasattr(score_norm, "__self__") else None, key_norm)

    # Build A_ij = div(u^j) + <score_i, u^j>  (per-batch scalars)
    A11 = div_u1 + _inner_batch(score_tb,   u_tb)
    A12 = div_u1 + _inner_batch(score_tb,   u_norm)
    A21 = div_u2 + _inner_batch(score_norm, u_tb)
    A22 = div_u2 + _inner_batch(score_norm, u_norm)

    dA1 = A11 - A21  # ΔA_1
    dA2 = A12 - A22  # ΔA_2

    # κ1 = dA2 / (dA2 - dA1).  Numerically stabilize the denominator.
    denom = jnp.where(jnp.abs(dA2 - dA1) < 1e-8, jnp.sign(dA2 - dA1) * 1e-8, dA2 - dA1)
    k1 = dA2 / denom
    k1 = jnp.clip(k1, 0.0, 1.0)
    k2 = 1.0 - k1

    # Return as broadcastable tensors for composing the vector field
    k1_b = k1[:, None, None, None]
    k2_b = k2[:, None, None, None]
    return k1_b, k2_b, {'k1': k1, 'k2': k2, 'A11': A11, 'A12': A12, 'A21': A21, 'A22': A22}

# -----------------------------------------------------------------------------
# Combined score/velocity + sampler (Algorithm 1 lines)
# -----------------------------------------------------------------------------

def get_combined_score_AND(state_tb, state_norm):
    """
    Returns a jit-compiled function:
        composed_score(x, t, key) -> (score_AND, diagnostics)
    Implements Algorithm 1 (AND branch):
      - Compute per-model scores ∇_x log q^i_t(x)  (line "∇_x log q^i_t(x_τ)")
      - Solve κ via linear equations (Prop. 6)     (line "solve Linear Equations // AND")
      - u_t(x_τ) = Σ_i κ^i_τ ∇ log q^i_t(x_τ)      (line "u_t(x_τ) ← Σ κ^i ∇ log q^i")
    """

    @functools.partial(jax.jit, static_argnums=())
    def _fn(x, t, key):
        # (A) Per-model EPS -> SCORE  (use your σ(t) as in compose.py)
        eps1 = state_tb.apply_fn({'params': state_tb.params}, x, t)
        eps2 = state_norm.apply_fn({'params': state_norm.params}, x, t)
        sigma = marginal_prob_std_fn(t)[:, None, None, None]
        score1 = -eps1 / sigma
        score2 = -eps2 / sigma

        # (B) Compose model "velocities" u^i = g(t)^2 * score^i   for EM update
        #    (matches your EM form: dx = (g^2 * score) dt + g dW)
        g_t = diffusion_coeff_fn(t)[:, None, None, None]
        u1 = (g_t ** 2) * score1
        u2 = (g_t ** 2) * score2

        # (C) Solve κ for AND via Proposition 6 linear system (two-model closed form)
        key, k1key, k2key = jax.random.split(key, 3)
        k1, k2, kdiag = kappa_solver_and(
            x, t, score1, score2, u1, u2, k1key, k2key
        )

        # (D) u_t(x_τ) = Σ_i κ_i u^i  (Algorithm 1, "u_t ← Σ κ^i ∇ log q^i", here with u^i)
        u_and = k1 * u1 + k2 * u2

        # Diagnostics for plotting/verification
        diags = dict(
            score_norm_tb  = jnp.linalg.norm(score1.reshape(score1.shape[0], -1), axis=-1),
            score_norm_norm= jnp.linalg.norm(score2.reshape(score2.shape[0], -1), axis=-1),
            k1 = kdiag['k1'], k2 = kdiag['k2'],
        )
        return u_and, diags, dict(score1=score1, score2=score2, g_t=g_t)

    return _fn

def sample_and_track_AND(
    state_tb, state_norm, vae_def, vae_params, latent_scale_factor,
    batch_size: int, steps: int, seed: int, out_dir: str
):
    """
    Euler–Maruyama sampler (Algorithm 1 loop):
      x_{τ+Δτ} ← x_τ + u_t(x_τ) Δτ + g_{1-τ} dW_τ
    We keep explicit logging of κ, score norms, and save final decoded grid.
    """
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, "samples"); os.makedirs(samples_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    composed_fn = get_combined_score_AND(state_tb, state_norm)

    # Time grid and initial x ~ N(0, I) scaled by σ(1)  (Algorithm 1 init)
    time_grid = jnp.linspace(1., 1e-3, steps)
    dt = time_grid[0] - time_grid[1]

    # Latent shapes from VAE config (identical to compose.py)
    latent_h = vae_def.enc_cfg['img_h'] // 4 if 'img_h' in vae_def.enc_cfg else None
    if latent_h is None:
        # fallback: infer from AE downsample ratio in your repo ( /4 )
        # You already do: latent_size = meta['img_size'] // 4
        latent_h =  vae_def.enc_cfg.get('img_size', 256) // 4
    z_ch = vae_def.enc_cfg['z_ch']
    x_shape = (batch_size, latent_h, latent_h, z_ch)

    key, init_key = jax.random.split(key)
    x = jax.random.normal(init_key, x_shape) * marginal_prob_std_fn(jnp.ones((batch_size,)))[:, None, None, None]

    history = []
    for t in time_grid:
        key, step_key, noise_key = jax.random.split(key, 3)
        t_b = jnp.ones((batch_size,)) * t

        # u_and corresponds to (g^2 * score_AND). We keep g explicitly for the noise term.
        u_and, diags, aux = composed_fn(x, t_b, step_key)
        g_t = aux['g_t']
        noise = jax.random.normal(noise_key, x.shape)

        # Euler–Maruyama step (Algorithm 1: dx_τ ← ( ... ) dτ + g dW)
        x = x + u_and * dt + g_t * jnp.sqrt(dt) * noise

        history.append(jax.device_get(diags))

    # Decode and save a grid
    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({'params': params}, latents, method=vae_def.decode)

    decoded = vae_decode_fn(vae_params, x / latent_scale_factor)
    decoded = jnp.clip(decoded, 0., 1.)
    grid = torch.tensor(np.asarray(decoded).transpose(0, 3, 1, 2))
    save_image(grid, os.path.join(samples_dir, "final_AND_grid.png"), nrow=int(math.sqrt(batch_size)))

    # Plots
    if history:
        agg = {k: np.array([h[k] for h in history]) for k in history[0]}
        steps_axis = np.arange(len(history))
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].plot(steps_axis, agg['score_norm_tb'].mean(axis=-1), label="TB")
        axs[0].plot(steps_axis, agg['score_norm_norm'].mean(axis=-1), label="Normal")
        axs[0].set_title("Score norms"); axs[0].legend(); axs[0].set_xlabel("step")

        axs[1].plot(steps_axis, agg['k1'].mean(axis=-1), label="κ_TB")
        axs[1].plot(steps_axis, (1.0-agg['k1']).mean(axis=-1), label="κ_Normal")
        axs[1].set_ylim(0, 1); axs[1].set_title("κ traces (mean)"); axs[1].legend(); axs[1].set_xlabel("step")

        axs[2].hist(agg['k1'].flatten(), bins=50, range=(0,1))
        axs[2].set_title("κ_TB distribution")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "and_diagnostics.png"))
        plt.close()

    return os.path.join(samples_dir, "final_AND_grid.png")

# -----------------------------------------------------------------------------
# CLI glue (same UX as compose.py)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SuperDiff AND composition between two unconditional LDMs (JAX).")
    parser.add_argument("--run_tb",     type=str, required=True)
    parser.add_argument("--run_normal", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps",      type=int, default=700)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="superdiff_and_out")
    args = parser.parse_args()

    # Load LDMs
    tb_cfg  = os.path.join(args.run_tb,     "ldm_meta.json")
    tb_ckpt = os.path.join(args.run_tb,     "ckpts/last.flax")
    nm_cfg  = os.path.join(args.run_normal, "ldm_meta.json")
    nm_ckpt = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb_model,    ldm_tb_params,    tb_meta = load_ldm(tb_cfg, tb_ckpt)
    ldm_norm_model,  ldm_norm_params,  nm_meta = load_ldm(nm_cfg, nm_ckpt)

    # TrainStates (no optimizer needed at sampling)
    state_tb   = TrainState.create(apply_fn=ldm_tb_model.apply,   params=ldm_tb_params,   tx=optax.identity())
    state_norm = TrainState.create(apply_fn=ldm_norm_model.apply, params=ldm_norm_params, tx=optax.identity())

    # Shared autoencoder (identical to compose.py)
    ae_config_path = tb_meta['ae_config_path']
    ae_ckpt_path   = tb_meta['ae_ckpt_path']
    vae_def, vae_params = load_autoencoder(ae_config_path, ae_ckpt_path)
    latent_scale_factor = tb_meta['latent_scale_factor']

    # Output
    out_dir = os.path.join(args.output_dir, f"AND_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # Run sampling + diagnostics
    grid_path = sample_and_track_AND(
        state_tb, state_norm, vae_def, vae_params, latent_scale_factor,
        batch_size=args.batch_size, steps=args.steps, seed=args.seed, out_dir=out_dir
    )
    print(f"✅ Finished. Saved final AND grid to: {grid_path}")

if __name__ == "__main__":
    main()
