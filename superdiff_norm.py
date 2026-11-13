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

# Import your modules (adjust paths if necessary based on your folder structure)
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn


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
        model_1, params_1,  # Normal Model
        model_2, params_2,  # TB Model
        ae_model, ae_params,
        marginal_prob_std_fn,
        latent_size,
        batch_size,
        z_channels,
        z_std_correction,
        num_steps=1000,
        lift=0.0
):
    """
    Stochastic SuperDiff Sampler (Logical AND) - JAX Implementation.
    Corrected to fix vmap/shape errors.
    """
    t0 = 1.0
    t1 = 1e-5
    timesteps = jnp.linspace(t0, t1, num_steps)
    sigmas = marginal_prob_std_fn(timesteps)

    # Initial Noise
    rng, step_rng = jax.random.split(rng)
    # Scale initial noise by sigma[0] (max noise)
    latents = jax.random.normal(step_rng, (batch_size, latent_size, latent_size, z_channels)) * sigmas[0]

    # Helper for model inference
    def get_vel(model, params, x, t):
        # Broadcast t to [Batch_Size]
        t_batch = jnp.full((x.shape[0],), t)
        return model.apply({'params': params}, x, t_batch)

    # Loop Step Function
    def step_fn(carry, i):
        latents, rng = carry

        # Current timestep and sigma
        t = timesteps[i]
        sigma = sigmas[i]

        # Look ahead for dsigma
        # For the last step, use the previous delta or 0
        sigma_next = jax.lax.select(
            i < num_steps - 1,
            sigmas[i + 1],
            sigmas[i] - (sigmas[i - 1] - sigmas[i])
        )
        dsigma = sigma_next - sigma

        # 1. Calculate Velocities (Model Outputs)
        vel_obj = get_vel(model_1, params_1, latents, t)
        vel_bg = get_vel(model_2, params_2, latents, t)

        # 2. Sample Noise for the SDE step
        rng, noise_rng = jax.random.split(rng)
        noise = jnp.sqrt(2 * jnp.abs(dsigma) * sigma) * jax.random.normal(noise_rng, latents.shape)

        # 3. SuperDiff Logic (Logical AND)
        # Independent step reference (using Model 2 as background/reference)
        dx_ind = 2 * dsigma * vel_bg + noise

        # Numerator: Energy Difference Term
        # (vel_bg^2 - vel_obj^2)
        diff_sq = (vel_bg - vel_obj) * (vel_bg + vel_obj)
        term1 = (jnp.abs(dsigma) * diff_sq).sum(axis=(1, 2, 3))

        # Interaction term
        term2 = (dx_ind * (vel_obj - vel_bg)).sum(axis=(1, 2, 3))

        # Lift term (optional bias)
        term3 = sigma * lift / num_steps

        numerator = term1 - term2 + term3

        # Denominator
        denom = 2 * dsigma * ((vel_obj - vel_bg) ** 2).sum(axis=(1, 2, 3))

        # Calculate Kappa
        kappa = numerator / (denom + 1e-8)

        # Broadcast Kappa [B] -> [B, 1, 1, 1]
        kappa_reshaped = kappa[:, None, None, None]

        # 4. Construct Composite Vector Field
        # vf = vel_bg + kappa * (vel_obj - vel_bg)
        vf_raw = vel_bg + kappa_reshaped * (vel_obj - vel_bg)

        norm_bg = jnp.sqrt(jnp.sum(vel_bg ** 2, axis=(1, 2, 3), keepdims=True))
        norm_vf = jnp.sqrt(jnp.sum(vf_raw ** 2, axis=(1, 2, 3), keepdims=True))

        safe_norm_vf = jnp.maximum(norm_vf, 1e-6)
        vf = vf_raw * (norm_bg / safe_norm_vf)

        # 5. Update Latents
        dx = 2 * dsigma * vf + noise
        latents_next = latents + dx

        return (latents_next, rng), kappa

    init_carry = (latents, rng)
    (final_latents, _), kappas_history = jax.lax.scan(step_fn, init_carry, jnp.arange(num_steps))
    scaled_latents = final_latents * z_std_correction
    decoded = ae_model.apply({'params': ae_params}, scaled_latents, method=ae_model.decode)

    return decoded, final_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True, help="Path to Normal Model exp")
    parser.add_argument("--run_dir_2", type=str, required=True, help="Path to TB Model exp")
    parser.add_argument("--ckpt_name", type=str, default="last.flax")
    parser.add_argument("--output_name", type=str, default="superdiff_and_mix.png")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--lift", type=float, default=0.0, help="Bias towards Model 1 (positive) or Model 2 (negative)")
    args = parser.parse_args()

    # --- 1. Load Autoencoder ---
    # We assume both models use the same VAE config, so we load from run_dir_1
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

    # Consistency check
    assert cfg1['img_size'] == cfg2['img_size'], "Image sizes must match"

    # --- 3. Sampling ---
    print(f"Running SuperDiff AND Sampler ({args.steps} steps)...")

    # Scale correction
    scale_factor = cfg1.get('latent_scale_factor', 1.0)
    z_std_correction = 1.0 / scale_factor

    rng = jax.random.PRNGKey(args.seed)

    images, final_z = superdiff_and_sampler(
        rng=rng,
        model_1=ldm1, params_1=params1,
        model_2=ldm2, params_2=params2,
        ae_model=ae_model, ae_params=ae_params,
        marginal_prob_std_fn=marginal_prob_std_fn,
        latent_size=lsize,
        batch_size=args.batch_size,
        z_channels=zch,
        z_std_correction=z_std_correction,
        num_steps=args.steps,
        lift=args.lift
    )

    # --- 4. Save Results ---
    from torchvision.utils import save_image
    import torch

    # Convert JAX array to Torch tensor for torchvision saving
    # JAX outputs usually [B, H, W, C] or [B, C, H, W] depending on VAE.
    # VAE decode usually returns [B, H, W, C] in Flax if not transposed.
    # But PyTorch expects [B, C, H, W].

    images_np = np.array(images)

    # Check shape to ensure correct permute
    if images_np.shape[-1] == 1 or images_np.shape[-1] == 3:
        # [B, H, W, C] -> [B, C, H, W]
        images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2)
    else:
        images_torch = torch.from_numpy(images_np)

    # Normalize if necessary (assuming VAE output is 0-1 or -1 to 1)
    # Usually Diffusers VAE outputs are raw, but user code might vary.
    # Just clamping for safety.
    images_torch = torch.clamp(images_torch, 0, 1)

    out_path = os.path.join(args.run_dir_1, "final_samples", args.output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(images_torch, out_path)
    print(f"Saved mixed samples to: {out_path}")


if __name__ == "__main__":
    main()