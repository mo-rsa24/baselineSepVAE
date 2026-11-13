
import os
import sys
import json
import math
import functools
from datetime import datetime
from typing import Tuple, Dict, Any

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

# --- Ensure repo root on sys.path (mirrors compose.py) ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Project modules (same imports used in compose.py)
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder
from diffusion.vp_equation import (
    marginal_prob_std_fn,
    score_function_hutchinson_estimator,
    diffusion_coeff_fn,
    get_kappa,  # used for OR-style softmax weighting; kept for parity + ablations
)

# ------------------------------
# Utility: load a single LDM (copy of compose.py logic, but packaged)
# ------------------------------
class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        loaded_json = json.load(f)
        meta = loaded_json.get('args', loaded_json)

    # --- Load VAE metadata ---
    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get('args', vae_loaded_json)
    z_channels = vae_meta['z_channels']

    # --- Construct U-Net parameters ---
    ldm_chans = tuple(meta['ldm_base_ch'] * int(m) for m in meta['ldm_ch_mults'].split(','))
    attn_res = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blocks = meta['ldm_num_res_blocks']

    # --- Instantiate ScoreNet ---
    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_res,
    )
    rng = jax.random.PRNGKey(0)
    latent_size = meta['img_size'] // 4
    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_variables = ldm_model.init({'params': rng, 'dropout': rng}, fake_latents, fake_time)
    tx = optax.chain(
        optax.clip_by_global_norm(meta.get('grad_clip', 1.0)),
        optax.adamw(meta.get('lr', 3e-5), weight_decay=meta.get('weight_decay', 0.01))
    )
    dummy_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=ldm_variables['params'],
        ema_params=ldm_variables['params'],
        tx=tx
    )

    print(f"Loading LDM checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    restored_state = from_bytes(dummy_state, blob)
    print("✅ LDM loaded successfully.")
    use_ema = meta.get('use_ema', False)
    if use_ema and hasattr(restored_state, 'ema_params') and restored_state.ema_params is not None:
        print("INFO: Using EMA parameters for composition.")
        params_to_return = restored_state.ema_params
    else:
        print("INFO: Using standard model parameters for composition.")
        params_to_return = restored_state.params

    return ldm_model, params_to_return


# ------------------------------
# Algorithm 1 (AND): κ from linear equations (Prop. 6) – practical solver
# ------------------------------
def kappa_solver(scores, dlogs, mode: str = "and_ls", eps: float = 1e-6):
    """
    Solve for κ given per-model score fields and divergence (d/dx · score) estimates.

    scores: tuple of [B,H,W,C] arrays (∇ log q_i^t(x))
    dlogs : tuple of [B] arrays (Hutchinson divergence estimates per model)
    returns: κ with shape [B, M]
    """
    M = len(scores)
    B = scores[0].shape[0]

    if mode == "poe":
        return jnp.ones((B, M))

    if mode == "softmax_or":
        raise ValueError("softmax_or mode of kappa_solver is not called directly; use get_kappa in get_combined_score.")

    # ---- AND least-squares via proper Gram matrix per batch ----
    # Flatten spatial dims and stack into S: [B, D, M]
    flats = [s.reshape(B, -1) for s in scores]            # each [B, D]
    S = jnp.stack(flats, axis=-1)                         # [B, D, M]

    # Gram per batch: G[b] = S[b]^T S[b] -> [M, M]
    # Result shape: [B, M, M]
    G = jnp.einsum("bdm,bdn->bmn", S, S)

    # RHS b: stack dlogs -> [B, M]
    b = jnp.stack(dlogs, axis=1)

    # Regularize to avoid degeneracy and improve conditioning
    eye = jnp.eye(M)[None, :, :]                          # [1, M, M]
    G_reg = G + eps * eye

    # Solve G κ = b per batch
    def solve_one(Gb, bb):
        # jnp.linalg.solve is batched via vmap below
        return jnp.linalg.solve(Gb, bb)

    κ = jax.vmap(solve_one, in_axes=(0, 0))(G_reg, b)     # [B, M]

    # Clamp & normalize for stability (remove/adjust if theory prefers unconstrained κ)
    κ = jnp.clip(κ, 0.0, 10.0)
    κ = κ / (jnp.sum(κ, axis=1, keepdims=True) + 1e-8)
    return κ



def get_combined_score(x, t, score_fns, params_list, key, mode: str = "and_ls"):
    """
    Compute composed score field u_t(x) = Σ_i κ_i ∇ log q_i^t(x) and required diagnostics.

    This function mirrors Alg. 1 lines:
      - Compute per-model scores (∇ log q_i^t)
      - Compute κ via "solve Linear Equations" (Prop. 6)  [AND]
      - Aggregate to obtain u_t
      - Track d log q_{1-τ}(x_τ) terms for diagnostics

    Args:
        x: [B, H, W, C] latents
        t: [B] time
        score_fns: list of apply_fns (Flax) for models
        params_list: list of params for corresponding models
        key: PRNGKey
        mode: "and_ls" (AND) | "poe" | "softmax_or" (maps to Alg.1 OR)

    Returns:
        composed_score: [B, H, W, C]
        diag: dict with norms, kappas, divergences, and raw per-model scores
    """
    B = x.shape[0]
    # Obtain epsilon predictions and convert to score via σ(t)
    eps_list = [fn({'params': p}, x, t) for fn, p in zip(score_fns, params_list)]  # [B,H,W,C] each
    sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
    score_list = [(-eps) / sigma_t for eps in eps_list]

    # Hutchinson divergence estimates for each model
    key_split = jax.random.split(key, len(score_fns))
    dlog_list = [
        score_function_hutchinson_estimator(x, t, fn, p, k)[0]
        for fn, p, k in zip(score_fns, params_list, key_split)
    ]  # each shape [B]

    if mode == "softmax_or":
        # Reuse project helper (compose.py uses this for OR)
        κ_soft = get_kappa(t, tuple(dlog_list), tuple(score_list))
        # Guard range
        κ_soft = jnp.clip(κ_soft, 0., 1.)
        κ = jnp.stack([κ_soft, 1. - κ_soft], axis=1) if len(score_list) == 2 else κ_soft
    else:
        κ = kappa_solver(tuple(score_list), tuple(dlog_list), mode=mode)  # [B, M]

    # Weighted sum: u_t(x) = Σ_i κ_i s_i(x)
    # Expand κ across HWC for broadcasting: [B, 1, 1, 1, M]
    s_stack = jnp.stack(score_list, axis=-1)  # [B,H,W,C,M]
    κ_broadcast = κ[:, None, None, None, :]   # [B,1,1,1,M]
    composed_score = jnp.sum(κ_broadcast * s_stack, axis=-1)  # [B,H,W,C]

    diag = {
        "kappa": κ,  # [B,M]
        "score_norms": jnp.stack([jnp.linalg.norm(s.reshape(B, -1), axis=-1) for s in score_list], axis=-1),  # [B,M]
        "score_norm_composed": jnp.linalg.norm(composed_score.reshape(B, -1), axis=-1),  # [B]
        "dlog": jnp.stack(dlog_list, axis=-1),  # [B,M]
    }
    return composed_score, diag


# ------------------------------
# Euler–Maruyama sampler + trackers (Alg. 1 loop)
# ------------------------------
def sample_and_track(state_list, ae_cfg, batch_size: int, steps: int, seed: int,
                     output_dir: str, mode: str = "and_ls"):
    """
    Run Algorithm 1 with Euler–Maruyama updates while logging κ, divergences, norms.

    state_list: list[TrainState] for the two unconditional LDMs
    ae_cfg: dict containing ae_config_path / ae_ckpt_path / latent_scale_factor / img_size
    mode: "and_ls" (logical AND via linear equations), "poe", "softmax_or"
    """
    # Load autoencoder (shared latent space)
    vae_def, vae_params = load_autoencoder(ae_cfg['ae_config_path'], ae_cfg['ae_ckpt_path'])
    latent_scale_factor = ae_cfg['latent_scale_factor']
    latent_size = ae_cfg['img_size'] // 4
    z_ch = vae_def.enc_cfg['z_ch']

    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({'params': params}, latents, method=vae_def.decode)

    sample_shape = (batch_size, latent_size, latent_size, z_ch)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    # Initialize latents ~ N(0, I) * σ(1)
    x = jax.random.normal(init_key, sample_shape) * marginal_prob_std_fn(jnp.ones((batch_size,)))[:, None, None, None]

    # Time grid and dt
    time_steps = jnp.linspace(1., 1e-3, steps)
    dt = time_steps[0] - time_steps[1]

    # Unpack states into apply_fns + params
    score_fns = [s.apply_fn for s in state_list]
    params_list = [s.params for s in state_list]

    # Trackers
    diags = []

    print(f"[SuperDiff-AND] Sampling {steps} steps using mode='{mode}' ...")
    for idx, t_scalar in enumerate(time_steps):
        key, step_key, noise_key = jax.random.split(key, 3)
        t = jnp.ones((batch_size,)) * t_scalar

        # u_t(x) via our composed score rule
        score, diag = get_combined_score(x, t, score_fns, params_list, step_key, mode=mode)
        diags.append(jax.device_get(diag))

        # Euler–Maruyama update: dx_t = (g_t^2 * u_t) dt + g_t dW_t
        g_t = diffusion_coeff_fn(t)[:, None, None, None]
        noise = jax.random.normal(noise_key, x.shape)
        x = x + (g_t**2) * score * dt + g_t * jnp.sqrt(jnp.abs(dt)) * noise

        if (idx + 1) % max(1, steps // 10) == 0:
            print(f"  step {idx+1}/{steps}  ⟨||u||⟩={jnp.mean(diag['score_norm_composed']):.3f}  "
                  f"⟨κ⟩={jnp.mean(diag['kappa']):.3f}")

    # Decode images from latents
    decoded = vae_decode_fn(vae_params, x / latent_scale_factor)
    decoded = jnp.clip(decoded, 0.0, 1.0)
    imgs_torch = torch.tensor(np.asarray(decoded).transpose(0, 3, 1, 2))

    # Persist artifacts
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "and_samples_grid.png")
    save_image(imgs_torch, grid_path, nrow=int(math.sqrt(batch_size)))

    # Diagnostics summary plots
    if diags:
        agg = {k: np.array([d[k] for d in diags]) for k in diags[0]}
        steps_axis = np.arange(len(diags))

        # Score norms
        plt.figure(figsize=(8, 4))
        for m in range(agg['score_norms'].shape[-1]):
            plt.plot(steps_axis, agg['score_norms'][:, :, m].mean(axis=1), label=f"model{m}")
        plt.plot(steps_axis, agg['score_norm_composed'].mean(axis=1), linestyle='--', label="composed")
        plt.title("Score norms"); plt.xlabel("step"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "diag_score_norms.png")); plt.close()

        # Kappa (mean per step)
        plt.figure(figsize=(8, 4))
        plt.plot(steps_axis, agg['kappa'].mean(axis=(1, 2)))
        plt.title("κ (mean over batch)"); plt.xlabel("step")
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "diag_kappa.png")); plt.close()

        # Divergences
        plt.figure(figsize=(8, 4))
        for m in range(agg['dlog'].shape[-1]):
            plt.plot(steps_axis, agg['dlog'][:, :, m].mean(axis=1), label=f"dlog model{m}")
        plt.title("Hutchinson divergence"); plt.xlabel("step"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "diag_divergence.png")); plt.close()

    return {"grid_path": grid_path, "output_dir": output_dir}


# ------------------------------
# Entry point mirroring compose.py but focused on AND Algorithm 1
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="SuperDiff logical AND composition between two unconditional LDMs (JAX).")
    parser.add_argument("--run_tb", type=str, required=True)
    parser.add_argument("--run_normal", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="superdiff_and_output")
    parser.add_argument("--mode", type=str, default="and_ls", choices=["and_ls", "poe", "softmax_or"],
                        help="AND via linear system (default), PoE sum, or OR-softmax for ablations.")
    args = parser.parse_args()

    # Load LDMs (same as compose.py)
    tb_cfg = os.path.join(args.run_tb, "ldm_meta.json")
    tb_ckpt = os.path.join(args.run_tb, "ckpts/last.flax")
    nm_cfg = os.path.join(args.run_normal, "ldm_meta.json")
    nm_ckpt = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb, par_tb = load_ldm(tb_cfg, tb_ckpt)
    ldm_nm, par_nm = load_ldm(nm_cfg, nm_ckpt)

    state_tb = TrainState.create(apply_fn=ldm_tb.apply, params=par_tb, tx=optax.identity())
    state_nm = TrainState.create(apply_fn=ldm_nm.apply, params=par_nm, tx=optax.identity())

    # Autoencoder config (shared latent space) – pulled from tb config for convenience
    with open(tb_cfg, 'r') as f:
        tb_loaded_json = json.load(f)
        tb_meta = tb_loaded_json.get('args', tb_loaded_json)

    ae_cfg = {
        "ae_config_path": tb_meta["ae_config_path"],
        "ae_ckpt_path": tb_meta["ae_ckpt_path"],
        "latent_scale_factor": tb_meta["latent_scale_factor"],
        "img_size": tb_meta["img_size"],
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"and_{args.mode}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    artifacts = sample_and_track(
        state_list=[state_tb, state_nm],
        ae_cfg=ae_cfg,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        output_dir=out_dir,
        mode=args.mode,
    )

    print(f"[SuperDiff-AND] Done. Grid saved to: {artifacts['grid_path']}")

if __name__ == "__main__":
    main()
