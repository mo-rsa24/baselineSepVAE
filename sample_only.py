import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from run.ldm import load_autoencoder_
from torchvision.utils import save_image
import flax
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import tensorflow as tf
from typing import Any

# Import your modules
from models.cxr_unet import ScoreNet
from models.ae_kl import AutoencoderKL
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn
from diffusion.sampling import DDPM_ancestral_sampler


# Define the State class (must match training)
class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the experiment folder containing ldm_meta.json")
    parser.add_argument("--ckpt_name", type=str, default="last.flax", help="Name of checkpoint in ckpts/ folder")
    parser.add_argument("--output_name", type=str, default="resample_fixed.png")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=700)
    args = parser.parse_args()

    # Paths
    meta_path = os.path.join(args.run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(args.run_dir, "ckpts", args.ckpt_name)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find config at {meta_path}")

    with open(meta_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded config for experiment: {config.get('run_name', 'Unknown')}")

    # --- 1. Load Autoencoder ---
    ae_model, ae_params = load_autoencoder_(config['ae_config_path'], config['ae_ckpt_path'])
    # Replicate for alignment with how LDM expects it (though for sampling we just need the params)
    # We will just pass ae_params directly to sampler.

    # --- 2. Initialize LDM ---
    # Calculate latent size
    # Assuming config structure matches run/ldm.py
    if isinstance(config['ldm_ch_mults'], str):
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'].split(','))
    else:
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'])

    attn_res = tuple(int(r) for r in str(config['ldm_attn_res']).split(','))

    # Determine latent spatial size based on AE config
    # We open AE config again to check downsampling
    with open(config['ae_config_path'], 'r') as f:
        ae_cfg_json = json.load(f)

    if isinstance(ae_cfg_json['ch_mults'], str):
        n_down = len(ae_cfg_json['ch_mults'].split(',')) - 1
    else:
        n_down = len(ae_cfg_json['ch_mults']) - 1

    latent_size = config['img_size'] // (2 ** n_down)
    z_channels = ae_cfg_json['z_channels']

    print(f"LDM Config: {latent_size}x{latent_size} input, {z_channels} channels")

    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=config['ldm_num_res_blocks'],
        attn_resolutions=attn_res,
        use_remat=False,  # Remat not needed for inference
        dtype=jnp.float32,  # Use float32 for safety in inference
        param_dtype=jnp.float32,
    )

    # Initialize params
    rng = jax.random.PRNGKey(args.seed)
    fake_latent = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    variables = ldm_model.init(rng, fake_latent, fake_time)

    # Create TrainState structure to load checkpoint
    import optax
    tx = optax.adamw(1e-4)  # Dummy optimizer
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=variables['params'],
        ema_params=variables['params'],  # Placeholder
        tx=tx
    )

    # --- 3. Load LDM Checkpoint ---
    # --- 3. Load LDM Checkpoint (Params Only) ---
    print(f"Loading LDM Checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    # deserialize to a raw dictionary, bypassing strict structure checks
    raw_state = flax.serialization.msgpack_restore(blob)

    # Extract params and ema_params safely
    loaded_params = raw_state.get('params')
    loaded_ema_params = raw_state.get('ema_params')

    if loaded_params is None:
        raise ValueError("Checkpoint does not contain 'params'!")

    # Re-create the state with the LOADED parameters
    # We use the dummy tx, but since we set params manually, the optimizer state
    # will be initialized to a clean 'zero' state (which is fine, we aren't training).
    from flax.core.frozen_dict import freeze
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=freeze(loaded_params),
        ema_params=freeze(loaded_ema_params) if loaded_ema_params else None,
        tx=tx
    )
    print("LDM Loaded successfully (Optimizer state ignored).")

    # Select params (EMA or regular)
    use_ema = config.get('use_ema', False)
    if use_ema and ldm_state.ema_params is not None:
        print("Using EMA parameters.")
        params = ldm_state.ema_params
    else:
        print("Using standard parameters.")
        params = ldm_state.params

    # --- 4. Sample ---
    print(f"Sampling {args.batch_size} images...")

    # Fix for latent scale: Invert the scale factor
    # If training was z * scale, decoding needs z / scale.
    # The sampler takes z_std and multiplies the final output by it.
    # So we pass 1/scale.
    scale_factor = config.get('latent_scale_factor', 1.0)
    z_std_correction = 1.0 / scale_factor

    rng, sample_rng = jax.random.split(rng)

    samples_grid, final_latents, _ = DDPM_ancestral_sampler(
        rng=sample_rng,
        ldm_model=ldm_model,
        ldm_params=params,
        ae_model=ae_model,
        ae_params=ae_params,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        alpha_bar_fn=alpha_bar_fn,
        latent_size=latent_size,
        batch_size=args.batch_size,
        z_channels=z_channels,
        z_std=z_std_correction
    )

    # Stats
    lat_np = np.asarray(final_latents)
    print(
        f"Latent Stats - Mean: {lat_np.mean():.4f}, Std: {lat_np.std():.4f}, Min: {lat_np.min():.4f}, Max: {lat_np.max():.4f}")

    # Save
    out_path = os.path.join(args.run_dir, "final_samples", args.output_name)
    save_image(samples_grid, out_path)
    print(f"Saved sample to: {out_path}")


if __name__ == "__main__":
    main()