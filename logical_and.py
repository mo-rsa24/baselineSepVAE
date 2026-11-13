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

# Component Imports from provided files
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn


def load_ldm_checkpoint(run_dir, ckpt_name):
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


def get_hutchinson_div(model, params, x, t, v):
    def score_fn(y): return model.apply({'params': params}, y, t)

    _, jvp_val = jax.jvp(score_fn, (x,), (v,))
    return jnp.sum(v * jvp_val, axis=(1, 2, 3))


def run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, batch_size, n_steps=500, z_std=1.0,
                            mode="superdiff"):
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(init_rng, (batch_size, sz, sz, ch)) * marginal_prob_std_fn(jnp.array([1.0]))[0]

    # Trackers for trajectory plotting
    ll_n_trace, ll_t_trace = [], []
    ll_n, ll_t = jnp.zeros(batch_size), jnp.zeros(batch_size)

    timesteps = jnp.linspace(1.0, 1e-5, n_steps + 1)
    for i in tqdm(range(n_steps), desc=f"Mode: {mode}"):
        t_now, dt = timesteps[i], timesteps[i + 1] - timesteps[i]
        vec_t = jnp.ones(batch_size) * t_now
        std_t, g_t = marginal_prob_std_fn(vec_t)[:, None, None, None], diffusion_coeff_fn(vec_t)[:, None, None, None]

        rng, h_rng, n_rng = jax.random.split(rng, 3)
        v, noise = jax.random.normal(h_rng, x.shape), jax.random.normal(n_rng, x.shape)

        s1 = -m1.apply({'params': p1}, x, vec_t) / std_t
        s2 = -m2.apply({'params': p2}, x, vec_t) / std_t
        div1 = get_hutchinson_div(m1, p1, x, vec_t, v) / std_t.squeeze()
        div2 = get_hutchinson_div(m2, p2, x, vec_t, v) / std_t.squeeze()

        if mode == "average":
            s_final = 0.5 * (s1 + s2)  # Simple heuristic averaging
        else:  # SuperDiff AND (Faithful to reference)
            dx_ind = (-0.5 * (g_t ** 2) * s2 * jnp.abs(dt)) + (g_t * jnp.sqrt(jnp.abs(dt)) * noise)
            diff_s, sum_s = s1 - s2, s1 + s2
            num = (0.5 * (g_t ** 2).squeeze() * jnp.abs(dt) * (
                        jnp.sum(diff_s * sum_s, axis=(1, 2, 3)) + (div1 - div2))) - jnp.sum(dx_ind * diff_s,
                                                                                            axis=(1, 2, 3))
            den = (g_t ** 2).squeeze() * jnp.abs(dt) * jnp.sum(diff_s ** 2, axis=(1, 2, 3)) + 1e-12
            s_final = s2 + jnp.clip(num / den, 0.0, 1.0)[:, None, None, None] * diff_s

        dx = (-0.5 * (g_t ** 2) * s_final * jnp.abs(dt)) + (g_t * jnp.sqrt(jnp.abs(dt)) * noise)
        x += dx

        # Itô Density Tracking
        ll_n += (-0.5 * (g_t ** 2).squeeze() * jnp.abs(dt) * (jnp.sum(s1 ** 2, axis=(1, 2, 3)) - div1) - jnp.sum(
            dx * s1, axis=(1, 2, 3)))
        ll_t += (-0.5 * (g_t ** 2).squeeze() * jnp.abs(dt) * (jnp.sum(s2 ** 2, axis=(1, 2, 3)) - div2) - jnp.sum(
            dx * s2, axis=(1, 2, 3)))
        ll_n_trace.append(jnp.mean(ll_n))
        ll_t_trace.append(jnp.mean(ll_t))

    x_hat = ae_model.apply({'params': ae_params}, x * z_std, method=ae_model.decode, train=False)
    return jnp.clip(x_hat, 0, 1), (ll_n_trace, ll_t_trace)


def plot_trajectories(traces, out_path):
    plt.figure(figsize=(10, 5))
    for name, (ln, lt) in traces.items():
        plt.plot(ln, label=f"{name} - Normal LL", linestyle='--')
        plt.plot(lt, label=f"{name} - TB LL")
    plt.title("Log-Likelihood Equilibrium Tracking")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Cumulative Log-Density")
    plt.legend()
    plt.savefig(out_path)
    print(f"Trajectory plot saved to {out_path}")


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
    img_avg, trace_avg = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 8, mode="average",
                                                 z_std=z_std)
    # 2. Generate SuperDiff AND Samples
    img_sd, trace_sd = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 8, mode="superdiff",
                                               z_std=z_std)

    # Combine into a single grid
    comparison = jnp.concatenate([img_avg, img_sd], axis=0)
    grid = make_grid(torch.from_numpy(np.asarray(jnp.transpose(comparison, (0, 3, 1, 2)))), nrow=8)
    save_image(grid, "cxr_comparison_averaging_vs_superdiff.png")

    plot_trajectories({"Averaging": trace_avg, "SuperDiff": trace_sd}, "ll_trajectories.png")


if __name__ == "__main__":
    main()