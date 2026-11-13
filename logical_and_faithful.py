import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import flax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.core.frozen_dict import freeze
from torchvision.utils import save_image, make_grid
import torch

# Component Imports
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn


def load_ldm_checkpoint(run_dir, ckpt_name):
    """
    Loads the Latent Diffusion Model checkpoint and configuration.
    """
    meta_path = os.path.join(run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)
    with open(meta_path, 'r') as f:
        config = json.load(f)
    with open(config['ae_config_path'], 'r') as f:
        ae_cfg = json.load(f)

    ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in (
        config['ldm_ch_mults'].split(',') if isinstance(config['ldm_ch_mults'], str) else config['ldm_ch_mults']))

    model = ScoreNet(z_channels=ae_cfg['z_channels'], channels=ldm_chans, num_res_blocks=config['ldm_num_res_blocks'],
                     attn_resolutions=tuple(int(r) for r in str(config['ldm_attn_res']).split(',')), dtype=jnp.float32)

    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        raw_state = flax.serialization.msgpack_restore(f.read())
    return model, freeze(raw_state.get('ema_params') or raw_state.get('params')), config, config['img_size'] // (
            2 ** (len(ae_cfg['ch_mults'].split(',')) - 1)), ae_cfg['z_channels']


def get_ito_kappa_and_updates(s1, s2, g_t, dt, noise, x, f_drift=None):
    """
    Calculates the mixing weight (kappa) and density updates using the Itô Density Estimator.
    """
    # Vectorized operations over batch and spatial dims
    diff_s = s1 - s2

    # Calculate norms squared sum(|s|^2) preserving batch dimension (axis 0)
    # x shape is (B, H, W, C), sum over (1, 2, 3)
    s1_sq = jnp.sum(s1 ** 2, axis=(1, 2, 3))
    s2_sq = jnp.sum(s2 ** 2, axis=(1, 2, 3))
    diff_sq = jnp.sum(diff_s ** 2, axis=(1, 2, 3)) + 1e-12

    # Project noise onto the difference vector: <noise, s1-s2>
    noise_proj = jnp.sum(noise * diff_s, axis=(1, 2, 3))

    # 1. Calculate Kappa (Mixing Weight) for Logical AND
    # Reshape g_t to match batch dimension if necessary, though usually scalar in this loop
    g_val = g_t.reshape(-1) if g_t.ndim > 0 else g_t

    # Formula: 0.5 - <noise, diff> / (g * sqrt(dt) * |diff|^2)
    # Note: dt is negative in reverse loop, so we use abs(dt) for the scaling term
    kappa = 0.5 - (noise_proj / (g_val * jnp.sqrt(jnp.abs(dt)) * diff_sq))

    # Clip kappa for stability [0, 1] as per paper recommendation
    kappa = jnp.clip(kappa, 0.0, 1.0)

    # Expand kappa for broadcasting: (B,) -> (B, 1, 1, 1)
    kappa_broad = kappa[:, None, None, None]

    # 2. Compute Density Updates (Log Likelihood tracking) for Plotting
    # Itô correction terms: - 0.5 * g^2 * |s|^2
    g2 = (g_val ** 2)
    ito_term_1 = -0.5 * g2 * s1_sq
    ito_term_2 = -0.5 * g2 * s2_sq

    if f_drift is not None:
        # If tracking absolute likelihoods, add <f, s>
        ito_term_1 += jnp.sum(f_drift * s1, axis=(1, 2, 3))
        ito_term_2 += jnp.sum(f_drift * s2, axis=(1, 2, 3))

    return kappa_broad, ito_term_1, ito_term_2


def run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, num_steps=1000, mode="average",
                            z_std=1.0):
    rng, step_rng = jax.random.split(rng)
    # Ensure correct batch size handling (here hardcoded to 1, but safe for logic)
    shape = (1, sz, sz, ch)
    x = jax.random.normal(step_rng, shape)

    # Standard VP-SDE Time configuration
    t_vals = jnp.linspace(1.0, 1e-5, num_steps)
    dt = -1.0 / num_steps  # Negative time step

    def get_score(model, params, x_t, t_scal):
        t_vec = jnp.ones((x_t.shape[0],)) * t_scal
        return model.apply({'params': params}, x_t, t_vec)

    print(f"Sampling with mode: {mode}...")

    # Log-Likelihood Tracking Lists
    ll_trace_1 = []
    ll_trace_2 = []

    # Initialize cumulative log-likelihoods
    cum_ll_1 = jnp.zeros((shape[0],))
    cum_ll_2 = jnp.zeros((shape[0],))

    for i in tqdm(range(num_steps)):
        t = t_vals[i]

        # 1. Calculate Scores
        s1 = get_score(m1, p1, x, t)
        s2 = get_score(m2, p2, x, t)

        # 2. Sample Noise for the step (shared for kappa calc and diffusion step)
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        g_t_batch = diffusion_coeff_fn(jnp.expand_dims(t, axis=0))
        g_t = g_t_batch[0]

        # 3. Determine Drift / Superposition
        if mode == "average":
            s_final = 0.5 * (s1 + s2)
            # No density tracking updates for average baseline in this context

        elif mode == "superdiff":
            # Use the faithful Itô function
            kappa, update1, update2 = get_ito_kappa_and_updates(s1, s2, g_t, dt, z, x)

            s_final = kappa * s1 + (1.0 - kappa) * s2

            # Track Log-Likelihoods
            # Update: d log q = <dx, s> + ito_term * dt
            # dx part: <f - g^2 s_final, s> dt + <g dz, s>
            # This is complex to track exactly step-by-step without full Ito expansion.
            # Simplified tracking based on the Itô Estimator term (drift part) primarily:
            cum_ll_1 += update1 * jnp.abs(dt)
            cum_ll_2 += update2 * jnp.abs(dt)

            ll_trace_1.append(float(cum_ll_1[0]))
            ll_trace_2.append(float(cum_ll_2[0]))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 4. Euler-Maruyama Step (Reverse SDE)
        # Reverse Drift: f(x) - g(t)^2 * s(x,t)
        # VP-SDE f(x) = -0.5 * beta(t) * x
        beta_t = g_t ** 2
        f_x = -0.5 * beta_t * x
        drift = f_x - (beta_t * s_final)

        diffusion = g_t * jnp.sqrt(jnp.abs(dt)) * z
        x = x + drift * dt + diffusion

    # Decode final image
    x_scaled = x / z_std
    out = ae_model.apply({'params': ae_params}, x_scaled, method=ae_model.decode)

    # Return traces if available
    traces = (ll_trace_1, ll_trace_2) if mode == "superdiff" else ([], [])
    return out, traces


def plot_trajectories(traces, out_path):
    plt.figure(figsize=(10, 5))
    has_data = False
    for name, (ln, lt) in traces.items():
        if len(ln) > 0:
            plt.plot(ln, label=f"{name} - Model 1 LL", linestyle='--')
            plt.plot(lt, label=f"{name} - Model 2 LL")
            has_data = True

    if has_data:
        plt.title("Log-Likelihood Equilibrium Tracking (Itô Estimator)")
        plt.xlabel("Diffusion Step")
        plt.ylabel("Cumulative Log-Density Change (Approx)")
        plt.legend()
        plt.savefig(out_path)
        print(f"Trajectory plot saved to {out_path}")
    else:
        print("No trajectory data to plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True)
    parser.add_argument("--run_dir_2", type=str, required=True)
    args = parser.parse_args()

    m1, p1, cfg1, sz, ch = load_ldm_checkpoint(args.run_dir_1, "last.flax")
    m2, p2, _, _, _ = load_ldm_checkpoint(args.run_dir_2, "last.flax")
    ae_model, ae_params = load_autoencoder_(cfg1['ae_config_path'], cfg1['ae_ckpt_path'])

    z_std = 1.0 / cfg1.get('latent_scale_factor', 1.0)
    rng = jax.random.PRNGKey(42)

    # 1. Generate Average Samples
    print("Generating Average Samples...")
    img_avg, trace_avg = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 1000, mode="average",
                                                 z_std=z_std)

    # 2. Generate SUPERDIFF Samples
    print("Generating SUPERDIFF (AND) Samples...")
    img_sd, trace_sd = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 1000, mode="superdiff",
                                               z_std=z_std)

    # Combine into a single grid
    comparison = jnp.concatenate([img_avg, img_sd], axis=0)
    grid = make_grid(torch.from_numpy(np.asarray(jnp.transpose(comparison, (0, 3, 1, 2)))), nrow=2)
    save_image(grid, "cxr_comparison_averaging_vs_superdiff_faithful.png")

    plot_trajectories({"SuperDiff": trace_sd}, "ll_trajectories_faithful.png")


if __name__ == "__main__":
    main()