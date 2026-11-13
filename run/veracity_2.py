# veracity_2.py
import os
import sys
import json
import math
import functools
from datetime import datetime
from typing import Tuple, Dict, Any
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# --- ensure repo root on sys.path ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import optax

# --- Project modules ---
from diffusion.vp_equation import (
    marginal_prob_std_fn,
    diffusion_coeff_fn,
)
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder


class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    """Load a ScoreNet LDM; prefer EMA params if present."""
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        loaded_json = json.load(f)
        meta = loaded_json.get('args', loaded_json)

    # VAE metadata to infer z_channels
    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get('args', vae_loaded_json)
    z_channels = vae_meta['z_channels']

    # U-Net cfg
    ldm_chans = tuple(meta['ldm_base_ch'] * int(m) for m in meta['ldm_ch_mults'].split(','))
    attn_res = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blocks = meta['ldm_num_res_blocks']

    # Instantiate
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
        print("INFO: Using EMA parameters.")
        params_to_return = restored_state.ema_params
    else:
        print("INFO: Using non-EMA parameters.")
        params_to_return = restored_state.params

    return ldm_model, params_to_return


# ---------- Precompute schedule, initial latent, and fixed noise sequence (shared across λ) ----------
def precompute_schedule_and_noise(key, args, sample_shape):
    """
    Returns time_steps, dt, x_init, noise_seq.
    These remain identical for every lambda so grid (row,col) tracks the same sample across frames.
    """
    time_steps = jnp.linspace(1.0, 1e-3, args.steps)
    dt = time_steps[0] - time_steps[1]

    # Fixed initial latent
    key, k_init = jax.random.split(key)
    x_init = jax.random.normal(k_init, sample_shape)
    std_1 = marginal_prob_std_fn(jnp.ones((args.batch_size,)))[:, None, None, None]
    x_init = x_init * std_1

    # Fixed step noises
    keys = jax.random.split(key, args.steps)
    noise_seq = jax.vmap(lambda k: jax.random.normal(k, sample_shape))(keys)

    return time_steps, dt, x_init, noise_seq


# ---------- JIT-safe composed score (close over models; pass only params/x/t/λ) ----------
def make_composed_score_fn(model_tb, model_normal):
    @jax.jit
    def composed_score_fn(x, t, params_tb, params_normal, lambda_tb):
        """
        x: [B,H,W,C], t: [B]
        params_*: PyTrees
        lambda_tb: scalar or [B]
        """
        eps_tb = model_tb.apply({'params': params_tb}, x, t)
        eps_normal = model_normal.apply({'params': params_normal}, x, t)
        sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
        score_tb = -eps_tb / sigma_t
        score_normal = -eps_normal / sigma_t

        composed = (1.0 - lambda_tb) * score_normal + lambda_tb * score_tb

        # Per-sample L2 norms (no reshape with traced dims)
        def l2_per_sample(arr):
            return jnp.sqrt(jnp.sum(arr * arr, axis=(1, 2, 3)))

        diags = {
            'score_norm_tb': l2_per_sample(score_tb),
            'score_norm_normal': l2_per_sample(score_normal),
            'score_norm_composed': l2_per_sample(composed),
        }
        return composed, diags

    return composed_score_fn


def run_sampling_and_diagnostics(
    lambda_tb: float,
    args,
    output_dir: str,
    sample_shape: Tuple[int, int, int, int],
    state_tb: TrainState,
    state_normal: TrainState,
    vae_decode_fn,
    vae_params,
    latent_scale_factor: float,
    time_steps, dt, x_init, noise_seq,
    composed_score_fn
):
    """One λ run: SAME x_init + noise_seq; saves grid + diagnostics."""
    print(f"--- Running for λ_TB = {lambda_tb:.2f} ---")
    x = x_init
    diagnostics_history = []

    for k, t in enumerate(tqdm(time_steps, desc=f"Sampling (λ={lambda_tb:.2f})")):
        t_batch = jnp.ones(args.batch_size) * t
        score, diagnostics = composed_score_fn(
            x, t_batch,
            state_tb.params, state_normal.params,
            jnp.asarray(lambda_tb, dtype=x.dtype)
        )
        diagnostics_history.append(jax.device_get(diagnostics))

        g_t = diffusion_coeff_fn(t_batch)[:, None, None, None]
        noise = noise_seq[k]
        x = x + (g_t ** 2) * score * dt + g_t * jnp.sqrt(dt) * noise

    print("✅ Sampling complete.")

    # Decode and save grid
    decoded = vae_decode_fn(vae_params, x / latent_scale_factor)
    decoded = jnp.clip(decoded, 0.0, 1.0)
    samples_torch = torch.tensor(np.asarray(decoded).transpose(0, 3, 1, 2))
    grid_path = os.path.join(output_dir, f"samples/final_samples_grid_lambda_{lambda_tb:.2f}.png")
    save_image(samples_torch, grid_path, nrow=int(math.sqrt(args.batch_size)))
    print(f"Saved final sample grid to {grid_path}")

    # Diagnostics plots
    if diagnostics_history:
        print("📊 Processing diagnostics.")
        diag_agg = {k: np.array([d[k] for d in diagnostics_history]) for k in diagnostics_history[0]}
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        steps_axis = np.arange(len(diagnostics_history))
        axs[0].plot(steps_axis, diag_agg['score_norm_tb'].mean(axis=-1), label="TB")
        axs[0].plot(steps_axis, diag_agg['score_norm_normal'].mean(axis=-1), label="Normal")
        axs[0].plot(steps_axis, diag_agg['score_norm_composed'].mean(axis=-1), label="Composed", linestyle='--', alpha=0.8)
        axs[0].set_title(f"Score Norms (λ_TB = {lambda_tb:.2f})")
        axs[0].legend(); axs[0].set_xlabel("Step")

        axs[1].hist(diag_agg['score_norm_composed'][-1], bins=20, alpha=0.7, label='Composed')
        axs[1].hist(diag_agg['score_norm_normal'][-1], bins=20, alpha=0.7, label='Normal')
        axs[1].hist(diag_agg['score_norm_tb'][-1], bins=20, alpha=0.7, label='TB')
        axs[1].set_title("Distribution of Final Step Score Norms")
        axs[1].legend()

        diag_plot_path = os.path.join(output_dir, f"diagnostics_plot_lambda_{lambda_tb:.2f}.png")
        plt.tight_layout()
        plt.savefig(diag_plot_path)
        plt.close()
        print(f"Saved diagnostics plot to {diag_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Compose two pretrained LDMs and generate λ-sweep frames with consistent samples.")
    parser.add_argument("--run_tb", type=str, required=True, help="Path to the TB model run directory.")
    parser.add_argument("--run_normal", type=str, required=True, help="Path to the Normal model run directory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--steps", type=int, default=700, help="Number of reverse diffusion steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="composed_output", help="Directory to save generated samples.")

    # Sweep args
    parser.add_argument("--sweep", action="store_true", help="Run a sweep over lambda_tb values.")
    parser.add_argument("--lambda_tb", type=float, default=0.5, help="Weight for the TB model score (0.0..1.0) when not sweeping.")
    parser.add_argument("--num_sweep_steps", type=int, default=11, help="Number of λ steps (e.g., 11 for 0.0,0.1,...,1.0).")
    args = parser.parse_args()

    # Setup
    key = jax.random.PRNGKey(args.seed)
    output_dir = os.path.join(args.output_dir, f"compose_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    print(f"🚀 Starting composition run. Outputs will be saved to: {output_dir}")

    # Load models
    tb_config_path = os.path.join(args.run_tb, "ldm_meta.json")
    tb_ckpt_path = os.path.join(args.run_tb, "ckpts/last.flax")
    normal_config_path = os.path.join(args.run_normal, "ldm_meta.json")
    normal_ckpt_path = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb_model, ldm_tb_params = load_ldm(tb_config_path, tb_ckpt_path)
    ldm_normal_model, ldm_normal_params = load_ldm(normal_config_path, normal_ckpt_path)
    composed_score_fn = make_composed_score_fn(ldm_tb_model, ldm_normal_model)

    state_tb = TrainState.create(apply_fn=ldm_tb_model.apply, params=ldm_tb_params, tx=optax.identity())
    state_normal = TrainState.create(apply_fn=ldm_normal_model.apply, params=ldm_normal_params, tx=optax.identity())

    with open(tb_config_path, 'r') as f:
        tb_loaded_json = json.load(f)
        tb_meta = tb_loaded_json.get('args', tb_loaded_json)

    ae_config_path = tb_meta['ae_config_path']
    ae_ckpt_path = tb_meta['ae_ckpt_path']
    vae_def, vae_params = load_autoencoder(ae_config_path, ae_ckpt_path)
    latent_scale_factor = tb_meta['latent_scale_factor']

    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({'params': params}, latents, method=vae_def.decode)

    latent_size = tb_meta['img_size'] // 4
    z_channels = vae_def.enc_cfg['z_ch']
    sample_shape = (args.batch_size, latent_size, latent_size, z_channels)

    # ---------- Precompute schedule + fixed init + fixed noises ONCE ----------
    time_steps, dt, x_init, noise_seq = precompute_schedule_and_noise(key, args, sample_shape)

    # ---------- Run single or sweep with identical paths across λ ----------
    if args.sweep:
        print(f"🔬 Starting sweep over {args.num_sweep_steps} lambda values.")
        lambda_values = np.linspace(0.0, 1.0, args.num_sweep_steps)
        for i, lambda_val in enumerate(lambda_values):
            run_sampling_and_diagnostics(
                lambda_val, args, output_dir, sample_shape,
                state_tb, state_normal, vae_decode_fn, vae_params, latent_scale_factor,
                time_steps, dt, x_init, noise_seq,
                composed_score_fn
            )
            print(f"✅ Finished sweep step {i+1}/{args.num_sweep_steps}")
    else:
        print(f"Single run with λ_TB = {args.lambda_tb:.2f}")
        run_sampling_and_diagnostics(
            args.lambda_tb, args, output_dir, sample_shape,
            state_tb, state_normal, vae_decode_fn, vae_params, latent_scale_factor,
            time_steps, dt, x_init, noise_seq,
            composed_score_fn
        )

    print("🎉 All tasks complete.")


if __name__ == "__main__":
    main()
