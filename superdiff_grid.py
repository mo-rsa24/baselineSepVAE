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
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Import your modules
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


def create_labelled_grid(images_np, rows, cols, lift_values, output_path):
    """Creates a grid image with Lift labels on top."""
    h, w = images_np.shape[1], images_np.shape[2]

    # Grid Dimensions
    header_height = 40
    grid_w = cols * w
    grid_h = rows * h + header_height

    # Create white canvas
    img = Image.new('L', (grid_w, grid_h), color=255)
    draw = ImageDraw.Draw(img)

    # Try to load font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # 1. Draw Column Headers (Lift Values)
    for j, lift in enumerate(lift_values):
        label = f"Lift={lift}"

        # Calculate text position (centered)
        try:
            # Pillow >= 10.0
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w = right - left
        except AttributeError:
            # Older Pillow
            text_w = draw.textlength(label, font=font)

        x_pos = j * w + (w - text_w) // 2
        y_pos = (header_height - 20) // 2

        draw.text((x_pos, y_pos), label, fill=0, font=font)

    # 2. Paste Images
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            im_arr = images_np[idx]
            pil_img = Image.fromarray(im_arr, mode='L')

            # Y-offset includes the header height
            y_offset = header_height + i * h
            x_offset = j * w

            img.paste(pil_img, (x_offset, y_offset))

    img.save(output_path)


def superdiff_and_sampler(
        latents,
        model_1, params_1,  # Normal Model
        model_2, params_2,  # TB Model
        ae_model, ae_params,
        marginal_prob_std_fn,
        z_std_correction,
        num_steps,
        lift_batch  # Expecting batch of lift values (B,)
):
    """
    Stochastic SuperDiff Sampler (Logical AND) - Sweep Version.
    """
    t0 = 1.0
    t1 = 1e-5
    # Generate N+1 steps to have N intervals
    timesteps = jnp.linspace(t0, t1, num_steps + 1)
    # Pre-calculate sigmas (vectorized)
    sigmas = marginal_prob_std_fn(timesteps)

    # Helper for model inference
    def get_vel(model, params, x, t):
        # Broadcast t to [Batch_Size]
        t_batch = jnp.full((x.shape[0],), t)
        return model.apply({'params': params}, x, t_batch)

    # Loop
    for i in tqdm(range(num_steps), desc="SuperDiff Sweep"):
        t_current = timesteps[i]

        # Current sigma and next sigma
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dsigma = sigma_next - sigma

        # 1. Calculate Velocities (Model Outputs)
        vel_obj = get_vel(model_1, params_1, latents, t_current)
        vel_bg = get_vel(model_2, params_2, latents, t_current)

        # 2. Sample Noise for the SDE step
        # Create a fresh PRNG key for noise based on step index
        noise_rng = jax.random.PRNGKey(i)
        noise = jnp.sqrt(2 * jnp.abs(dsigma) * sigma) * jax.random.normal(noise_rng, latents.shape)

        # 3. SuperDiff Logic (Logical AND)
        # Independent step reference (using Model 2 as background/reference)
        dx_ind = 2 * dsigma * vel_bg + noise

        # Numerator: Energy Difference Term
        # (vel_bg^2 - vel_obj^2) = (vel_bg - vel_obj)(vel_bg + vel_obj)
        diff_sq = (vel_bg - vel_obj) * (vel_bg + vel_obj)
        term1 = (jnp.abs(dsigma) * diff_sq).sum(axis=(1, 2, 3))

        # Interaction term
        term2 = (dx_ind * (vel_obj - vel_bg)).sum(axis=(1, 2, 3))

        # Lift term (Vectorized by batch)
        # lift_batch is (B,), sigma is scalar. Result is (B,)
        term3 = sigma * lift_batch / num_steps

        numerator = term1 - term2 + term3

        # Denominator
        denom = 2 * dsigma * ((vel_obj - vel_bg) ** 2).sum(axis=(1, 2, 3))

        # Calculate Kappa
        kappa = numerator / (denom + 1e-8)

        # Broadcast Kappa [B] -> [B, 1, 1, 1]
        kappa_reshaped = kappa[:, None, None, None]

        # 4. Construct Composite Vector Field
        vf = vel_bg + kappa_reshaped * (vel_obj - vel_bg)

        # 5. Update Latents
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

    # --- Decode ---
    print("Decoding images...")
    scaled_latents = latents * z_std_correction
    # Use train=False to ensure deterministic behavior (no dropout)
    decoded = ae_model.apply({'params': ae_params}, scaled_latents, method=ae_model.decode, train=False)

    # Clip and convert to uint8 [0, 255]
    decoded = jnp.clip(decoded, 0.0, 1.0)
    decoded = (decoded * 255).astype(jnp.uint8)

    # Squeeze channel dim if grayscale: (B, H, W, 1) -> (B, H, W)
    if decoded.shape[-1] == 1:
        decoded = decoded.squeeze(-1)

    return decoded, latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True, help="Path to Normal Model exp")
    parser.add_argument("--run_dir_2", type=str, required=True, help="Path to TB Model exp")
    parser.add_argument("--ckpt_name", type=str, default="last.flax")
    parser.add_argument("--output_path", type=str, default="superdiff_sweep_grid.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    # --- Sweep Configuration ---
    LIFT_VALUES = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
    NUM_ROWS = 4
    NUM_COLS = len(LIFT_VALUES)
    TOTAL_BATCH = NUM_ROWS * NUM_COLS

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

    # --- 3. Construct Batch ---
    print(f"Generating {NUM_ROWS} fixed samples, sweeping {NUM_COLS} lift values each...")
    rng = jax.random.PRNGKey(args.seed)
    row_latents = []

    # Get initial noise sigma
    # Note: We just need sigma[0]. We can use the scalar fn directly or get it from array.
    t0 = 1.0
    sigma_max = marginal_prob_std_fn(jnp.array([t0]))[0]

    for i in range(NUM_ROWS):
        rng, seed_rng = jax.random.split(rng)
        # Generate ONE sample (NHWC)
        z0 = jax.random.normal(seed_rng, (1, lsize, lsize, zch)) * sigma_max
        # Tile it NUM_COLS times
        z0_row = jnp.tile(z0, (NUM_COLS, 1, 1, 1))
        row_latents.append(z0_row)

    # Concatenate all rows -> (B, H, W, C)
    latents = jnp.concatenate(row_latents, axis=0)

    # Construct Lift Vector
    lifts_jnp = jnp.array(LIFT_VALUES)
    lift_batch = jnp.tile(lifts_jnp, NUM_ROWS)  # Shape (Total Batch,)

    print(f"Total Batch Shape: {latents.shape}")

    # --- 4. Run Sampler ---
    scale_factor = cfg1.get('latent_scale_factor', 1.0)
    z_std_correction = 1.0 / scale_factor

    images_np, _ = superdiff_and_sampler(
        latents=latents,
        model_1=ldm1, params_1=params1,
        model_2=ldm2, params_2=params2,
        ae_model=ae_model, ae_params=ae_params,
        marginal_prob_std_fn=marginal_prob_std_fn,
        z_std_correction=z_std_correction,
        num_steps=args.steps,
        lift_batch=lift_batch
    )

    # --- 5. Create Labeled Grid ---
    # images_np is already uint8 numpy array [B, H, W]
    print(f"Creating grid with {NUM_ROWS} rows and {NUM_COLS} columns...")
    create_labelled_grid(np.array(images_np), NUM_ROWS, NUM_COLS, LIFT_VALUES, args.output_path)
    print(f"Saved sweep to {args.output_path}")


if __name__ == "__main__":
    main()