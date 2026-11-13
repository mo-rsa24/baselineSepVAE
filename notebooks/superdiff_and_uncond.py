import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
from torchvision.utils import save_image

from diffusion.vp_equation import diffusion_coeff_fn, marginal_prob_std_fn, score_function_hutchinson_estimator
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder


class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    """Load a pretrained ScoreNet LDM checkpoint (EMA if available)."""
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, "r") as f:
        loaded_json = json.load(f)
        meta = loaded_json.get("args", loaded_json)

    vae_config_path = meta["ae_config_path"]
    with open(vae_config_path, "r") as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get("args", vae_loaded_json)
    z_channels = vae_meta["z_channels"]

    ldm_chans = tuple(meta["ldm_base_ch"] * int(m) for m in meta["ldm_ch_mults"].split(","))
    attn_res = tuple(int(r) for r in meta["ldm_attn_res"].split(",") if r)
    num_res_blocks = meta["ldm_num_res_blocks"]

    model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_res,
        use_remat=meta.get("use_remat", False),
    )

    rng = jax.random.PRNGKey(0)
    latent_size = meta["img_size"] // 4
    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    variables = model.init({"params": rng, "dropout": rng}, fake_latents, fake_time)
    tx = optax.chain(
        optax.clip_by_global_norm(meta.get("grad_clip", 1.0)),
        optax.adamw(meta.get("lr", 3e-5), weight_decay=meta.get("weight_decay", 0.01)),
    )
    dummy_state = TrainStateWithEMA.create(
        apply_fn=model.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=tx,
    )

    print(f"Loading LDM checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    restored_state = from_bytes(dummy_state, blob)
    use_ema = meta.get("use_ema", False)
    if use_ema and getattr(restored_state, "ema_params", None) is not None:
        print("INFO: Using EMA parameters for sampling.")
        params = restored_state.ema_params
    else:
        print("INFO: Using standard model parameters for sampling.")
        params = restored_state.params

    return model, params


def kappa_solver(scores, dlogs, eps: float = 1e-6):
    """Solve for κ for two unconditional score fields (AND via linear equations)."""
    num_models = len(scores)
    batch_size = scores[0].shape[0]

    flats = [s.reshape(batch_size, -1) for s in scores]
    stacked = jnp.stack(flats, axis=-1)
    gram = jnp.einsum("bdm,bdn->bmn", stacked, stacked)
    rhs = jnp.stack(dlogs, axis=1)
    eye = jnp.eye(num_models)[None, :, :]
    gram_reg = gram + eps * eye

    def solve_one(g, b):
        return jnp.linalg.solve(g, b)

    kappa = jax.vmap(solve_one, in_axes=(0, 0))(gram_reg, rhs)
    kappa = jnp.clip(kappa, 0.0, 10.0)
    kappa = kappa / (jnp.sum(kappa, axis=1, keepdims=True) + 1e-8)
    return kappa


def get_composed_score(latents, t, model_fns, params_list, key):
    """Compose two unconditional score fields using SuperDiff AND κ (no CFG)."""
    eps_list = [fn({"params": p}, latents, t) for fn, p in zip(model_fns, params_list)]
    sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
    score_list = [(-eps) / sigma_t for eps in eps_list]

    key_split = jax.random.split(key, len(model_fns))
    dlog_list = [
        score_function_hutchinson_estimator(latents, t, fn, p, k)[0]
        for fn, p, k in zip(model_fns, params_list, key_split)
    ]

    kappa = kappa_solver(tuple(score_list), tuple(dlog_list))
    score_stack = jnp.stack(score_list, axis=-1)
    kappa_broadcast = kappa[:, None, None, None, :]
    composed_score = jnp.sum(kappa_broadcast * score_stack, axis=-1)

    return composed_score, {
        "kappa": kappa,
        "score_norms": jnp.stack([jnp.linalg.norm(s.reshape(latents.shape[0], -1), axis=-1) for s in score_list], axis=-1),
    }


def sample_superdiff_and(
    model_1,
    params_1,
    model_2,
    params_2,
    ae_cfg,
    batch_size: int,
    steps: int,
    seed: int,
    output_dir: str,
):
    vae_def, vae_params = load_autoencoder(ae_cfg["ae_config_path"], ae_cfg["ae_ckpt_path"])
    latent_scale_factor = ae_cfg["latent_scale_factor"]
    latent_size = ae_cfg["img_size"] // 4
    z_ch = vae_def.enc_cfg["z_ch"]

    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({"params": params}, latents, method=vae_def.decode)

    sample_shape = (batch_size, latent_size, latent_size, z_ch)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    latents = jax.random.normal(init_key, sample_shape) * marginal_prob_std_fn(jnp.ones((batch_size,)))[:, None, None, None]
    time_steps = jnp.linspace(1.0, 1e-3, steps)
    dt = time_steps[0] - time_steps[1]

    model_fns = [model_1.apply, model_2.apply]
    params_list = [params_1, params_2]

    kappa_history = []
    for idx, t_scalar in enumerate(time_steps):
        key, step_key, noise_key = jax.random.split(key, 3)
        t = jnp.ones((batch_size,)) * t_scalar
        score, diag = get_composed_score(latents, t, model_fns, params_list, step_key)
        kappa_history.append(jax.device_get(diag["kappa"]))

        g_t = diffusion_coeff_fn(t)[:, None, None, None]
        noise = jax.random.normal(noise_key, latents.shape)
        latents = latents + (g_t ** 2) * score * dt + g_t * jnp.sqrt(jnp.abs(dt)) * noise

        if (idx + 1) % max(1, steps // 10) == 0:
            mean_kappa = jnp.mean(diag["kappa"], axis=0)
            print(f"Step {idx + 1}/{steps} | mean κ = {mean_kappa}")

    decoded = vae_decode_fn(vae_params, latents / latent_scale_factor)
    decoded = jnp.clip(decoded, 0.0, 1.0)
    imgs_torch = torch.tensor(np.asarray(decoded).transpose(0, 3, 1, 2))

    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "and_uncond_samples.png")
    save_image(imgs_torch, grid_path, nrow=int(math.sqrt(batch_size)))

    return {"grid_path": grid_path, "kappa": kappa_history}


def main():
    parser = argparse.ArgumentParser(
        description="SuperDiff AND sampling with two unconditional LDMs (no CFG)."
    )
    parser.add_argument("--run_dir1", type=str, required=True, help="Checkpoint dir for model 1 (e.g., TB).")
    parser.add_argument("--run_dir2", type=str, required=True, help="Checkpoint dir for model 2 (e.g., normal).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="superdiff_and_uncond_output")
    args = parser.parse_args()

    cfg_1 = os.path.join(args.run_dir1, "ldm_meta.json")
    ckpt_1 = os.path.join(args.run_dir1, "ckpts", "last.flax")
    cfg_2 = os.path.join(args.run_dir2, "ldm_meta.json")
    ckpt_2 = os.path.join(args.run_dir2, "ckpts", "last.flax")

    model_1, params_1 = load_ldm(cfg_1, ckpt_1)
    model_2, params_2 = load_ldm(cfg_2, ckpt_2)

    with open(cfg_1, "r") as f:
        meta_1 = json.load(f)
        meta_1 = meta_1.get("args", meta_1)

    ae_cfg = {
        "ae_config_path": meta_1["ae_config_path"],
        "ae_ckpt_path": meta_1["ae_ckpt_path"],
        "latent_scale_factor": meta_1["latent_scale_factor"],
        "img_size": meta_1["img_size"],
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"and_uncond_{stamp}")

    artifacts = sample_superdiff_and(
        model_1=model_1,
        params_1=params_1,
        model_2=model_2,
        params_2=params_2,
        ae_cfg=ae_cfg,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        output_dir=output_dir,
    )

    print(f"Saved samples to {artifacts['grid_path']}")


if __name__ == "__main__":
    main()