import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from typing import Any
import flax
from flax.training.train_state import TrainState
from flax.serialization import from_bytes

# Import your modules
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
# Import necessary components from vp_equation for faithful implementation
from diffusion.vp_equation import (
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    beta,
    score_function_hutchinson_estimator,
    get_kappa
)


# Define the State class (must match training)
class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def load_ldm_state(run_dir, ckpt_name, seed=0, dummy_batch_size=1):
    """
    Helper to load an LDM model and its parameters from a run directory.
    """
    meta_path = os.path.join(run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find config at {meta_path}")

    with open(meta_path, 'r') as f:
        config = json.load(f)

    # --- Initialize LDM Architecture ---
    if isinstance(config['ldm_ch_mults'], str):
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'].split(','))
    else:
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'])

    attn_res = tuple(int(r) for r in str(config['ldm_attn_res']).split(','))

    # Determine latent dimensions from AE config
    with open(config['ae_config_path'], 'r') as f:
        ae_cfg_json = json.load(f)
    if isinstance(ae_cfg_json['ch_mults'], str):
        n_down = len(ae_cfg_json['ch_mults'].split(',')) - 1
    else:
        n_down = len(ae_cfg_json['ch_mults']) - 1

    latent_size = config['img_size'] // (2 ** n_down)
    z_channels = ae_cfg_json['z_channels']

    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=config['ldm_num_res_blocks'],
        attn_resolutions=attn_res,
        use_remat=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    # --- Initialize State ---
    rng = jax.random.PRNGKey(seed)
    fake_latent = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    variables = ldm_model.init(rng, fake_latent, fake_time)

    # Dummy optimizer for state creation
    import optax
    tx = optax.adamw(1e-4)

    # --- Load Checkpoint ---
    print(f"Loading LDM Checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    raw_state = flax.serialization.msgpack_restore(blob)

    loaded_params = raw_state.get('params')
    loaded_ema_params = raw_state.get('ema_params')

    if loaded_params is None:
        raise ValueError("Checkpoint does not contain 'params'!")

    from flax.core.frozen_dict import freeze
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=freeze(loaded_params),
        ema_params=freeze(loaded_ema_params) if loaded_ema_params else None,
        tx=tx
    )

    # Select params
    use_ema = config.get('use_ema', False)
    params = ldm_state.ema_params if (use_ema and ldm_state.ema_params) else ldm_state.params
    print(f"Loaded {config.get('run_name', 'Unknown')} (EMA={use_ema})")

    return ldm_model, params, config, latent_size, z_channels


def superdiff_and_sampler(
        rng,
        model_1, params_1,
        model_2, params_2,
        ae_model, ae_params,
        latent_size,
        batch_size,
        z_channels,
        z_std_correction,
        num_steps=1000,
):
    """
    SUPERDIFF Logical AND Sampler (Prop. 6 / Algorithm 1).
    Implements Itô density estimation and precise density control.
    """
    # Setup time schedule (T -> eps)
    t0 = 1.0
    t1 = 1e-5
    # Standard linear time spacing
    timesteps = jnp.linspace(t0, t1, num_steps)
    dt = (t1 - t0) / num_steps  # Negative step size

    # Initial Noise
    rng, step_rng = jax.random.split(rng)
    # Start at x_T ~ N(0, I) * sigma(T)
    latents = jax.random.normal(step_rng, (batch_size, latent_size, latent_size, z_channels)) * \
              marginal_prob_std_fn(jnp.array([t0]))[0]

    # Initialize log-densities for the experts
    init_log_q1 = jnp.zeros((batch_size,))
    init_log_q2 = jnp.zeros((batch_size,))

    # Helper: Convert model output (epsilon) to score (nabla log p)
    # score = -epsilon / sigma
    def get_score_fn(model, params):
        def score_fn(model_params, x, t):
            # t is broadcasted inside this wrapper to match batch size if needed by apply
            t_batch = jnp.ones((x.shape[0],)) * t if t.ndim == 0 else t
            sigma = marginal_prob_std_fn(t_batch)
            sigma = sigma.reshape(-1, 1, 1, 1)
            eps_pred = model.apply(model_params, x, t_batch)
            return -eps_pred / sigma

        return score_fn

    score_fn_1 = get_score_fn(model_1, params_1)
    score_fn_2 = get_score_fn(model_2, params_2)

    def step_fn(carry, i):
        latents, log_q1, log_q2, rng = carry

        # Current time t and next time t_next
        t = timesteps[i]
        # dt is constant, but conceptually t_next = t + dt (where dt < 0)

        # Broadcast t for calculations
        t_batch = jnp.full((latents.shape[0],), t)

        # 1. Compute Scores and Divergences (Hutchinson)
        rng, hutch_rng1, hutch_rng2 = jax.random.split(rng, 3)

        # Expert 1
        div1, _ = score_function_hutchinson_estimator(
            latents, t, score_fn_1, params_1, hutch_rng1
        )
        # Note: score_function_hutchinson_estimator re-evaluates the score.
        # For efficiency one might reuse, but for correctness we call it.
        # We need the actual score vector for the SDE as well.
        s1 = score_fn_1({'params': params_1}, latents, t)

        # Expert 2
        div2, _ = score_function_hutchinson_estimator(
            latents, t, score_fn_2, params_2, hutch_rng2
        )
        s2 = score_fn_2({'params': params_2}, latents, t)

        # 2. Solve for Kappa (Prop 6)
        # Solve A * kappa = b to equalize density increments d(log q1) = d(log q2)
        # get_kappa implements the closed form solution for 2 experts.
        kappa = get_kappa(t_batch, (div1, div2), (s1, s2))

        # Broadcast Kappa [B] -> [B, 1, 1, 1]
        k_b = kappa[:, None, None, None]

        # 3. Form Composite Score
        u = k_b * s1 + (1.0 - k_b) * s2

        # 4. Reverse SDE Step (VP-SDE)
        # Form: dx = [ f(x,t) - g(t)^2 u(x,t) ] dt + g(t) dW
        # Forward Drift f(x) = -0.5 * beta(t) * x
        # Diffusion g(t) = sqrt(beta(t))

        beta_t = beta(t_batch).reshape(-1, 1, 1, 1)
        diffusion_t = diffusion_coeff_fn(t_batch).reshape(-1, 1, 1, 1)  # g(t)

        f_x = -0.5 * beta_t * latents
        g2_u = (diffusion_t ** 2) * u

        reverse_drift = f_x - g2_u

        # Sample noise dW (scaled by sqrt(abs(dt)))
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, latents.shape)

        # Update x (Euler-Maruyama)
        # dt is negative, so we use abs(dt) for diffusion term scaling
        dx = reverse_drift * dt + diffusion_t * noise * jnp.sqrt(jnp.abs(dt))
        latents_next = latents + dx

        # 5. Update Log-Densities (Itô Density Estimator)
        # Eq. 13: d log p = [ div(g^2 s - f) - 0.5 * s^T g^2 s ] dt + s^T g dW
        # Note: The estimator tracks the density of the *process*.
        # Here we update log_q1 using s1 and log_q2 using s2 to track what
        # the densities WOULD be if we followed that model, allowing us to steer.

        def update_log_q(current_log_q, s_model, div_model):
            # Term 1: div(g^2 s - f) = g^2 * div(s) - div(f)
            # div(f) for VP SDE = div(-0.5 beta x) = -0.5 * beta * d (dimension)
            d = latents.shape[1] * latents.shape[2] * latents.shape[3]
            div_f = -0.5 * beta_t.squeeze() * d

            term_drift_part1 = (diffusion_t.squeeze() ** 2) * div_model - div_f

            # Term 2: -0.5 * s^T g^2 s
            s_norm_sq = jnp.sum(s_model ** 2, axis=(1, 2, 3))
            term_drift_part2 = -0.5 * (diffusion_t.squeeze() ** 2) * s_norm_sq

            # Stochastic Term: s^T g dW
            # dW approx = noise * sqrt(dt) (conceptually, but we used noise already)
            # In the SDE step: g dW ~ diffusion_t * noise * sqrt(dt)
            # So term is s dot (diffusion_t * noise * sqrt(dt))
            dw_term = jnp.sum(s_model * (diffusion_t * noise * jnp.sqrt(jnp.abs(dt))), axis=(1, 2, 3))

            # Combine terms (note: dt is negative in loop, but formula is for forward/reverse increment)
            # We align with the integration direction (dt).
            # The density accumulates along the path.
            d_log_q = (term_drift_part1 + term_drift_part2) * dt + dw_term
            return current_log_q + d_log_q

        log_q1_next = update_log_q(log_q1, s1, div1)
        log_q2_next = update_log_q(log_q2, s2, div2)

        return (latents_next, log_q1_next, log_q2_next, rng), kappa

    init_carry = (latents, init_log_q1, init_log_q2, rng)
    (final_latents, final_log1, final_log2, _), kappas_history = jax.lax.scan(step_fn, init_carry,
                                                                              jnp.arange(num_steps))

    scaled_latents = final_latents * z_std_correction
    decoded = ae_model.apply({'params': ae_params}, scaled_latents, method=ae_model.decode)

    return decoded, final_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True, help="Path to Normal Model exp")
    parser.add_argument("--run_dir_2", type=str, required=True, help="Path to TB Model exp")
    parser.add_argument("--ckpt_name", type=str, default="last.flax")
    parser.add_argument("--output_name", type=str, default="superdiff_and_mix_3.png")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=700)
    # Lift removed as it is not part of the strict Prop 6 AND derivation
    args = parser.parse_args()

    # --- 1. Load Autoencoder ---
    meta_path = os.path.join(args.run_dir_1, "ldm_meta.json")
    with open(meta_path, 'r') as f:
        config_1 = json.load(f)

    print("Loading Autoencoder...")
    ae_model, ae_params = load_autoencoder_(config_1['ae_config_path'], config_1['ae_ckpt_path'])

    # --- 2. Load Both LDMs ---
    print("Loading Model 1 (Normal)...")
    ldm1, params1, cfg1, lsize, zch = load_ldm_state(args.run_dir_1, args.ckpt_name, args.seed)

    print("Loading Model 2 (TB)...")
    ldm2, params2, cfg2, _, _ = load_ldm_state(args.run_dir_2, args.ckpt_name, args.seed)

    assert cfg1['img_size'] == cfg2['img_size'], "Image sizes must match"

    # --- 3. Sampling ---
    print(f"Running SuperDiff AND Sampler ({args.steps} steps)...")

    scale_factor = 0.99937266
    z_std_correction = 1.0 / scale_factor

    rng = jax.random.PRNGKey(args.seed)

    images, final_z = superdiff_and_sampler(
        rng=rng,
        model_1=ldm1, params_1=params1,
        model_2=ldm2, params_2=params2,
        ae_model=ae_model, ae_params=ae_params,
        latent_size=lsize,
        batch_size=args.batch_size,
        z_channels=zch,
        z_std_correction=z_std_correction,
        num_steps=args.steps,
    )

    # --- 4. Save Results ---
    from torchvision.utils import save_image
    import torch

    images_np = np.array(images)

    if images_np.shape[-1] == 1 or images_np.shape[-1] == 3:
        images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2)
    else:
        images_torch = torch.from_numpy(images_np)

    out_path = os.path.join(args.run_dir_1, "final_samples", args.output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(images_torch, out_path)
    print(f"Saved mixed samples to: {out_path}")


if __name__ == "__main__":
    main()