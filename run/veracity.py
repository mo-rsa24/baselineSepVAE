import os
import sys
import json
import math
import functools
from datetime import datetime
from typing import Tuple, Dict
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Any
from flax.training.train_state import TrainState

# --- This import path assumes the script is run from the repository root ---
from diffusion.vp_equation import marginal_prob_std_fn, score_function_hutchinson_estimator, diffusion_coeff_fn, \
    get_kappa


class TrainStateWithEMA(TrainState):
    ema_params: Any = None
# --- ensure repo root on sys.path ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import optax

# --- Use the project's actual modules, following ldm.py's structure ---

from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder  # Directly import the trusted loader


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

# ----------------------------------------------------------------------------
# MODIFICATION: Abstracted the sampling and diagnostics logic
# This makes it reusable for the experimental sweep.
# ----------------------------------------------------------------------------
def run_sampling_and_diagnostics(
    key,
    lambda_tb: float,
    args,
    output_dir: str,
    sample_shape: Tuple,
    state_tb: TrainState,
    state_normal: TrainState,
    vae_decode_fn,
    vae_params,
    latent_scale_factor: float
):
    """Runs the full sampling and plotting pipeline for a given lambda_tb."""
    print(f"--- Running for λ_TB = {lambda_tb:.2f} ---")

    # ----------------------------------------------------------------------------
    # MODIFICATION: `composed_score_fn` now accepts `lambda_tb`
    # This function now uses a direct weighted sum (PoE-style) for simplicity
    # and to directly test the effect of the lambda weighting.
    # ----------------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
    def composed_score_fn(x, t, key, batch_size, score_fn_tb, score_fn_normal, lambda_tb_static: float):
        eps_tb = score_fn_tb({'params': state_tb.params}, x, t)
        eps_normal = score_fn_normal({'params': state_normal.params}, x, t)
        sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
        score_tb = -eps_tb / sigma_t
        score_normal = -eps_normal / sigma_t

        # --- Direct PoE-style weighting ---
        composed_score = (1.0 - lambda_tb_static) * score_normal + lambda_tb_static * score_tb

        diagnostics = {
            'score_norm_tb': jnp.linalg.norm(score_tb.reshape(batch_size, -1), axis=-1),
            'score_norm_normal': jnp.linalg.norm(score_normal.reshape(batch_size, -1), axis=-1),
            'score_norm_composed': jnp.linalg.norm(composed_score.reshape(batch_size, -1), axis=-1),
        }
        return composed_score, diagnostics

    # --- Run Sampling ---
    print(f"Starting Euler-Maruyama sampler for {args.steps} steps...")
    key, sample_key = jax.random.split(key)
    time_steps = jnp.linspace(1., 1e-3, args.steps)
    dt = time_steps[0] - time_steps[1]
    x = jax.random.normal(sample_key, sample_shape) * marginal_prob_std_fn(jnp.ones((args.batch_size,)))[:, None, None, None]
    diagnostics_history = []
    for t in tqdm(time_steps, desc=f"Sampling (λ={lambda_tb:.2f})"):
        key, step_key, noise_key = jax.random.split(key, 3)
        t_batch = jnp.ones(args.batch_size) * t
        score, diagnostics = composed_score_fn(
            x,
            t_batch,
            step_key,
            args.batch_size,
            state_tb.apply_fn,
            state_normal.apply_fn,
            lambda_tb # Pass as a static argument
        )
        diagnostics_history.append(jax.device_get(diagnostics))
        g_t = diffusion_coeff_fn(t_batch)[:, None, None, None]
        noise = jax.random.normal(noise_key, x.shape)
        x = x + (g_t**2 * score * dt) + g_t * jnp.sqrt(dt) * noise
    print("✅ Sampling complete.")

    # --- Decode and Save ---
    decoded_samples = vae_decode_fn(vae_params, x / latent_scale_factor)
    decoded_samples = jnp.clip(decoded_samples, 0.0, 1.0)
    samples_torch = torch.tensor(np.asarray(decoded_samples).transpose(0, 3, 1, 2))
    # MODIFICATION: Filename now includes lambda value
    grid_path = os.path.join(output_dir, f"samples/final_samples_grid_lambda_{lambda_tb:.2f}.png")
    save_image(samples_torch, grid_path, nrow=int(math.sqrt(args.batch_size)))
    print(f"Saved final sample grid to {grid_path}")

    # --- Diagnostics ---
    if diagnostics_history:
        print("📊 Processing and saving diagnostics...")
        diag_agg = {k: np.array([d[k] for d in diagnostics_history]) for k in diagnostics_history[0]}
        fig, axs = plt.subplots(1, 2, figsize=(16, 5)) # Simplified plot
        steps_axis = np.arange(len(diagnostics_history))
        axs[0].plot(steps_axis, diag_agg['score_norm_tb'].mean(axis=-1), label="TB")
        axs[0].plot(steps_axis, diag_agg['score_norm_normal'].mean(axis=-1), label="Normal")
        axs[0].plot(steps_axis, diag_agg['score_norm_composed'].mean(axis=-1), label="Composed", linestyle='--', alpha=0.8)
        axs[0].set_title(f"Score Norms (λ_TB = {lambda_tb:.2f})")
        axs[0].legend(); axs[0].set_xlabel("Step")

        # Plot final score norm distribution
        axs[1].hist(diag_agg['score_norm_composed'][-1], bins=20, alpha=0.7, label='Composed')
        axs[1].hist(diag_agg['score_norm_normal'][-1], bins=20, alpha=0.7, label='Normal')
        axs[1].hist(diag_agg['score_norm_tb'][-1], bins=20, alpha=0.7, label='TB')
        axs[1].set_title("Distribution of Final Step Score Norms")
        axs[1].legend()

        # MODIFICATION: Filename now includes lambda value
        diag_plot_path = os.path.join(output_dir, f"diagnostics_plot_lambda_{lambda_tb:.2f}.png")
        plt.tight_layout()
        plt.savefig(diag_plot_path)
        plt.close()
        print(f"Saved diagnostics plot to {diag_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Compose two pretrained LDM models using Superdiffusion.")
    parser.add_argument("--run_tb", type=str, required=True, help="Path to the TB model run directory.")
    parser.add_argument("--run_normal", type=str, required=True, help="Path to the Normal model run directory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--steps", type=int, default=700, help="Number of reverse diffusion steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="composed_output", help="Directory to save generated samples.")

    # ----------------------------------------------------------------------------
    # NEW FEATURE: Add arguments for lambda sweep
    # ----------------------------------------------------------------------------
    parser.add_argument("--sweep", action="store_true", help="Run a sweep over lambda_tb values.")
    parser.add_argument("--lambda_tb", type=float, default=0.5, help="Weight for the TB model score (0.0 to 1.0). Only used if --sweep is not set.")
    parser.add_argument("--num_sweep_steps", type=int, default=11, help="Number of steps in the lambda sweep (e.g., 11 for 0.0, 0.1, ..., 1.0).")
    args = parser.parse_args()

    # --- Setup ---
    key = jax.random.PRNGKey(args.seed)
    output_dir = os.path.join(args.output_dir, f"compose_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    print(f"🚀 Starting composition run. Outputs will be saved to: {output_dir}")

    # --- Load Models ---
    tb_config_path = os.path.join(args.run_tb, "ldm_meta.json")
    tb_ckpt_path = os.path.join(args.run_tb, "ckpts/last.flax")
    normal_config_path = os.path.join(args.run_normal, "ldm_meta.json")
    normal_ckpt_path = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb_model, ldm_tb_params = load_ldm(tb_config_path, tb_ckpt_path)
    ldm_normal_model, ldm_normal_params = load_ldm(normal_config_path, normal_ckpt_path)

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

    # ----------------------------------------------------------------------------
    # MODIFICATION: Main logic now handles either a single run or a sweep
    # ----------------------------------------------------------------------------
    if args.sweep:
        print(f"🔬 Starting sweep over {args.num_sweep_steps} lambda values...")
        lambda_values = np.linspace(0.0, 1.0, args.num_sweep_steps)
        for i, lambda_val in enumerate(lambda_values):
            # Use a different key for each run in the sweep for variety
            key, run_key = jax.random.split(key)
            run_sampling_and_diagnostics(
                run_key, lambda_val, args, output_dir, sample_shape,
                state_tb, state_normal, vae_decode_fn, vae_params, latent_scale_factor
            )
            print(f"✅ Finished sweep step {i+1}/{args.num_sweep_steps}")
    else:
        print(f" одиночный запуск with a single lambda_tb = {args.lambda_tb}")
        run_sampling_and_diagnostics(
            key, args.lambda_tb, args, output_dir, sample_shape,
            state_tb, state_normal, vae_decode_fn, vae_params, latent_scale_factor
        )

    print("🎉 All tasks complete.")


if __name__ == "__main__":
    main()