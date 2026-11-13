import os
import optax
import flax
import argparse
import json
import jax
import jax.numpy as jnp
from diffusion.vp_equation import marginal_prob_std
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from flax.training.train_state import TrainState
from typing import Any
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder_
from tqdm import tqdm


class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def parse_args():
    parser = argparse.ArgumentParser(description="SuperDiff Lift Sweep (JAX)")
    parser.add_argument("--run_dir_normal", type=str, required=True, help="Path to Normal LDM run directory")
    parser.add_argument("--run_dir_tb", type=str, required=True, help="Path to TB LDM run directory")
    parser.add_argument("--output_path", type=str, default="superdiff_sweep.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_scale_factor", type=float, default=0.99937266)
    parser.add_argument("--steps", type=int, default=500)
    return parser.parse_args()


def load_ldm_state(run_dir, ckpt_name="last.flax"):
    """Helper to load an LDM model and its parameters from a run directory."""
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
    tx = optax.adamw(1e-4)

    # --- Load Checkpoint ---
    print(f"Loading LDM Checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
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

    use_ema = config.get('use_ema', False)
    params = ldm_state.ema_params if (use_ema and ldm_state.ema_params) else ldm_state.params
    print(f"Loaded {config.get('run_name', 'Unknown')} (EMA={use_ema})")

    return ldm_model, params, config, latent_size, z_channels


def get_score(model, params, x, t):
    return model.apply({'params': params}, x, t)


def stochastic_super_diff_and_uncond(
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        lift_batch  # Now expects a batch of lift values
):
    """
    Implements Stochastic SuperDiff AND logic with per-sample lift support.
    """
    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)

    # Initialize logs
    kappa_log = []

    for i in tqdm(range(num_inference_steps), desc="SuperDiff Sweep"):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]

        # Broadcast scalar time to batch
        t_batch = jnp.full((latents.shape[0],), t_current)

        sigma = marginal_prob_std(t_current)
        sigma_next = marginal_prob_std(t_next)
        dsigma = sigma_next - sigma

        vel_normal = get_score(model_normal, params_normal, latents, t_batch)
        vel_tb = get_score(model_tb, params_tb, latents, t_batch)

        # Independent step
        noise = jax.random.normal(jax.random.PRNGKey(i), latents.shape) * jnp.sqrt(2 * jnp.abs(dsigma) * sigma)
        dx_ind = 2 * dsigma * vel_normal + noise

        # Kappa Terms
        term1 = (jnp.abs(dsigma) * (vel_normal - vel_tb) * (vel_normal + vel_tb)).sum(axis=(1, 2, 3))
        term2 = (dx_ind * (vel_tb - vel_normal)).sum(axis=(1, 2, 3))

        # Lift is now a vector (B,), so term3 is (B,)
        term3 = sigma * lift_batch / num_inference_steps

        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * ((vel_tb - vel_normal) ** 2).sum(axis=(1, 2, 3))

        kappa = numerator / (denominator + 1e-8)
        kappa_log.append(kappa)

        kappa_b = kappa[:, None, None, None]

        # Composite Vector Field
        vf = vel_normal + kappa_b * (vel_tb - vel_normal)

        # Update Latents
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

    return latents, jnp.array(kappa_log)


def decode_image(vae, vae_params, latents, latent_scale_factor=1.0):
    """Decodes latents to images using the VAE (Fixed scaling)."""
    latents = 1 / latent_scale_factor * latents
    imgs = vae.apply({'params': vae_params}, latents, method=vae.decode, train=False)
    # Clip to [0,1] as per sampling.py
    imgs = jnp.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255).astype(jnp.uint8)
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)
    return imgs


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


def main():
    args = parse_args()
    meta_path = os.path.join(args.run_dir_normal, "ldm_meta.json")
    with open(meta_path, 'r') as f:
        config_1 = json.load(f)

    # --- Configuration ---
    LIFT_VALUES = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]  # Sorted for logical interpolation
    NUM_ROWS = 4
    NUM_COLS = len(LIFT_VALUES)
    TOTAL_BATCH = NUM_ROWS * NUM_COLS

    print("Loading Autoencoder...")
    ae_model, ae_params = load_autoencoder_(config_1['ae_config_path'], config_1['ae_ckpt_path'])

    print("Loading Models...")
    model_normal, params_normal, cfg1, lsize, zch = load_ldm_state(args.run_dir_normal)
    model_tb, params_tb, cfg2, _, _ = load_ldm_state(args.run_dir_tb)

    # --- Construct Batch ---
    # We want 4 unique seeds (Rows), each repeated 6 times (Cols for Lift)
    rng = jax.random.PRNGKey(args.seed)
    row_latents = []

    print(f"Generating {NUM_ROWS} fixed samples, sweeping {NUM_COLS} lift values each...")

    for i in range(NUM_ROWS):
        rng, seed_rng = jax.random.split(rng)
        # Generate ONE sample (NHWC)
        z0 = jax.random.normal(seed_rng, (1, lsize, lsize, zch))
        # Tile it NUM_COLS times
        z0_row = jnp.tile(z0, (NUM_COLS, 1, 1, 1))
        row_latents.append(z0_row)

    # Concatenate all rows -> (24, 64, 64, 4)
    latents = jnp.concatenate(row_latents, axis=0)

    # Construct Lift Vector -> [-1, -0.5, ..., -1, -0.5, ...]
    lifts_jnp = jnp.array(LIFT_VALUES)
    lift_batch = jnp.tile(lifts_jnp, NUM_ROWS)  # Shape (24,)

    print(f"Total Batch Shape: {latents.shape}")
    print(f"Lift Batch Shape: {lift_batch.shape}")

    # --- Run Sweep ---
    final_latents, _ = stochastic_super_diff_and_uncond(
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps=args.steps,
        lift_batch=lift_batch  # Pass vector here
    )

    # --- Decode ---
    print("Decoding images...")
    images = decode_image(ae_model, ae_params, final_latents, latent_scale_factor=args.latent_scale_factor)
    images_np = np.array(images)

    # --- Save Labelled Grid ---
    print(f"Creating grid with {NUM_ROWS} rows and {NUM_COLS} columns...")
    create_labelled_grid(images_np, NUM_ROWS, NUM_COLS, LIFT_VALUES, args.output_path)
    print(f"Saved sweep to {args.output_path}")


if __name__ == "__main__":
    main()