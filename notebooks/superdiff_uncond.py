import os
import optax
import flax
import argparse
import json
import jax
import jax.numpy as jnp
from diffusion.vp_equation import marginal_prob_std, marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn, score_function_hutchinson_estimator
from PIL import Image
import numpy as np
from flax.training.train_state import TrainState
from typing import Any
from models.cxr_unet import ScoreNet
from tqdm import tqdm
from run.ldm import load_autoencoder_


class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def parse_args():
    parser = argparse.ArgumentParser(description="SuperDiff AND with two unconditional LDMs (JAX)")
    parser.add_argument("--run_dir_normal", type=str, required=True, help="Path to Normal LDM run directory")
    parser.add_argument("--run_dir_tb", type=str, required=True, help="Path to TB LDM run directory")
    parser.add_argument("--output_path", type=str, default="superdiff_result.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_scale_factor", type=float, default=0.99937266)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lift", type=float, default=0.0, help="Lift parameter for SuperDiff stability")
    parser.add_argument("--score", type=bool, default=False, help="Parameter that determines if we should divide by -sigma")
    return parser.parse_args()


def load_ldm_state(run_dir, ckpt_name="last.flax"):
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

    # Select params
    use_ema = config.get('use_ema', False)
    params = ldm_state.ema_params if (use_ema and ldm_state.ema_params) else ldm_state.params
    print(f"Loaded {config.get('run_name', 'Unknown')} (EMA={use_ema})")

    return ldm_model, params, config, latent_size, z_channels

def get_score(model, params, x, t):
    """
    Computes the score (epsilon) for a given latent x and time t.
    Note: ScoreNet in run/ldm.py is trained to predict epsilon.
    t is expected in [0, 1].
    """
    return model.apply({'params': params}, x, t)


def stochastic_super_diff_and_uncond(
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        lift=0.0,
        score = False
):
    """
    Implements Stochastic SuperDiff AND logic for two unconditional score fields.
    Mapping:
      - Normal Model -> Background/Reference (vel_bg)
      - TB Model     -> Object/Constraint (vel_obj)
    """

    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)

    # Initialize logs
    kappa_log = []

    for i in tqdm(range(num_inference_steps), desc="SuperDiff AND"):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = jnp.full((latents.shape[0],), t_current)
        sigma = marginal_prob_std(t_current)
        sigma_next = marginal_prob_std(t_next)
        dsigma = sigma_next - sigma

        vel_normal = get_score(model_normal, params_normal, latents, t_batch)
        vel_tb = get_score(model_tb, params_tb, latents, t_batch)
        if score:
            vel_normal = - vel_normal / sigma
            vel_tb = - vel_tb / sigma

        # Independent step (baseline trajectory)
        noise = jax.random.normal(jax.random.PRNGKey(i), latents.shape) * jnp.sqrt(2 * jnp.abs(dsigma) * sigma)
        dx_ind = 2 * dsigma * vel_normal + noise

        term1 = (jnp.abs(dsigma) * (vel_normal - vel_tb) * (vel_normal + vel_tb)).sum(axis=(1, 2, 3))
        term2 = (dx_ind * (vel_tb - vel_normal)).sum(axis=(1, 2, 3))
        term3 = sigma * lift / num_inference_steps
        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * ((vel_tb - vel_normal) ** 2).sum(axis=(1, 2, 3))

        kappa = numerator / (denominator + 1e-8)
        kappa_log.append(kappa)
        kappa_b = kappa[:, None, None, None]

        # Composite Vector Field
        vf = vel_normal + kappa_b * (vel_tb - vel_normal)

        # 4. Update Latents (Euler-Maruyama step)
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

    return latents, jnp.array(kappa_log)

def ddpm_ancestral_superdiff_and_uncond(
    rng,
    latents,
    model_normal, params_normal,
    model_tb, params_tb,
    num_inference_steps,
    lift=0.0,
    kappa_clip=2.0,
):
    """
    Hybrid SUPERDIFF-AND:
      - models output epsilon (ε̂)
      - convert to score: s = -ε̂ / σ(t)
      - compute kappa using divergence estimates (Hutchinson) + inner-product term
      - mix scores, convert back to ε̂_mix
      - take a DDPM ancestral posterior step using ε̂_mix (sharp, stable)

    Inputs/outputs match your code structure. :contentReference[oaicite:1]{index=1}
    """
    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)
    kappa_log = []

    x = latents

    def _sum_except_batch(xx):
        # sum over (H,W,C) axes -> keep batch dimension
        axes = tuple(range(1, xx.ndim))
        return jnp.sum(xx, axis=axes, keepdims=True)

    for i in tqdm(range(num_inference_steps), desc="SuperDiff AND (DDPM ancestral)"):
        t_now  = timesteps[i]
        t_next = timesteps[i + 1]

        B = x.shape[0]
        vec_t      = jnp.full((B,), t_now)
        vec_t_next = jnp.full((B,), t_next)

        # ---- (1) ε predictions from both experts ----
        eps_normal = model_normal.apply({'params': params_normal}, x, vec_t)
        eps_tb     = model_tb.apply({'params': params_tb}, x, vec_t)

        # ---- (2) Convert ε -> score: s = -ε/σ ----
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]  # σ(t)
        sN = -eps_normal / std_now
        sT = -eps_tb     / std_now

        # ---- (3) Divergence estimates div(s) via Hutchinson ----
        # estimator expects a function returning ε̂; it internally converts to div(score)
        rng, k1, k2 = jax.random.split(rng, 3)

        divN, _ = score_function_hutchinson_estimator(
            x, vec_t,
            score_fn=model_normal.apply,
            params=params_normal,
            key=jax.random.fold_in(k1, i),
        )
        divT, _ = score_function_hutchinson_estimator(
            x, vec_t,
            score_fn=model_tb.apply,
            params=params_tb,
            key=jax.random.fold_in(k2, i),
        )

        divN_b = divN[:, None, None, None]
        divT_b = divT[:, None, None, None]

        # ---- (4) Compute κ (SuperDiff-inspired, score-space) ----
        # Structure matches your vp_equation-style expression: g(t)^2 (divN-divT) + <sN, sN-sT> / ||sN-sT||^2
        g2 = diffusion_coeff_fn(vec_t)[:, None, None, None] ** 2

        numerator   = g2 * (divN_b - divT_b) + _sum_except_batch(sN * (sN - sT))
        if lift != 0.0:
            numerator = numerator + (std_now * (lift / num_inference_steps))

        denominator = _sum_except_batch((sN - sT) ** 2) + 1e-12
        kappa = numerator / denominator

        # stability clamp (recommended for TB ∧ Normal)
        if kappa_clip is not None:
            kappa = jnp.clip(kappa, -kappa_clip, kappa_clip)

        kappa_log.append(kappa.squeeze())

        # ---- (5) Mix scores, convert back to ε̂_mix ----
        s_mix = sN + kappa * (sT - sN)
        eps_mix = -std_now * s_mix

        # ---- (6) DDPM ancestral posterior step using ε̂_mix ----
        alpha_bar_now  = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

        # x0 estimate (same form as your DDPM sampler)
        pred_x0 = (x - std_now * eps_mix) / sqrt_alpha_bar_now

        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

        rng, kn = jax.random.split(rng)
        noise = jax.random.normal(jax.random.fold_in(kn, i), x.shape)

        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_mix) + (sigma * noise)

    return x, jnp.array(kappa_log)




def decode_image(vae, vae_params, latents, latent_scale_factor = 1.0):
    """Decodes latents to images using the VAE."""
    latents = 1 / latent_scale_factor * latents
    imgs = vae.apply({'params': vae_params}, latents, method=vae.decode, train=False)
    imgs = jnp.clip(imgs, 0.0, 1.0)

    # 4. Quantize to uint8
    imgs = (imgs * 255).astype(jnp.uint8)

    # Squeeze channel dim if grayscale: (B, H, W, 1) -> (B, H, W)
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)
    return imgs


def save_image_grid(images_np, output_path):
    """Saves images as a square grid using PIL (No Torch dependency)."""
    batch_size, h, w = images_np.shape[0], images_np.shape[1], images_np.shape[2]

    # Calculate grid dimensions (e.g., 4 -> 2x2)
    n_cols = int(np.ceil(np.sqrt(batch_size)))
    n_rows = int(np.ceil(batch_size / n_cols))

    # Create canvas
    grid_img = Image.new('L', (w * n_cols, h * n_rows))

    for i in range(batch_size):
        row = i // n_cols
        col = i % n_cols
        img = Image.fromarray(images_np[i], mode='L')
        grid_img.paste(img, (col * w, row * h))

    grid_img.save(output_path)

def main():
    args = parse_args()
    meta_path = os.path.join(args.run_dir_normal, "ldm_meta.json")
    with open(meta_path, 'r') as f:
        config_1 = json.load(f)

    print("Loading Autoencoder...")
    ae_model, ae_params = load_autoencoder_(config_1['ae_config_path'], config_1['ae_ckpt_path'])
    # --- 2. Load Both LDMs ---
    print("Loading Model 1 (Normal)...")
    model_normal, params_normal, cfg1, lsize, zch = load_ldm_state(args.run_dir_normal)
    latent_scale_factor = args.latent_scale_factor
    print("Loading Model 2 (TB)...")
    model_tb, params_tb, cfg2, _, _ = load_ldm_state(args.run_dir_tb)

    # 3. Initialize Latents
    rng = jax.random.PRNGKey(args.seed)
    latent_shape = (args.batch_size, lsize, lsize, zch)
    latents = jax.random.normal(rng, latent_shape)

    # 4. Run Stochastic SuperDiff
    print("Running SuperDiff Composition (Hybrid DDPM)...")
    if not args.score:
        final_latents, kappas = ddpm_ancestral_superdiff_and_uncond(
            rng,
            latents,
            model_normal, params_normal,
            model_tb, params_tb,
            num_inference_steps=args.steps,
            lift=args.lift,
            kappa_clip=2.0,   # tune: 1.0–3.0
        )
    else:
        print("Running SuperDiff Composition...")
        final_latents, kappas = stochastic_super_diff_and_uncond(
            latents,
            model_normal, params_normal,
            model_tb, params_tb,
            num_inference_steps=args.steps,
            lift=args.lift,
            score=args.score
        )


    # 5. Decode and Save
    print("Decoding images...")
    images = decode_image(ae_model, ae_params, final_latents, latent_scale_factor=latent_scale_factor)

    # Convert to PIL and Save
    images_np = np.array(images)  # [B, H, W, C]
    save_image_grid(images_np, args.output_path)
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    main()