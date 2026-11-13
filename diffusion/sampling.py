import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import torch
from torchvision.utils import make_grid


def _sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)


def _broadcast_time(t_scalar, x):
    """Make t broadcast like x: (N,1[,1,1...])"""
    N = x.shape[0]
    extra_ones = (1,) * (x.ndim - 1)
    return jnp.ones((N,) + extra_ones, dtype=x.dtype) * t_scalar


def DDPM_ancestral_sampler(
        rng, ldm_model, ldm_params, ae_model, ae_params,
        marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn,
        latent_size, batch_size, z_channels, vae_z_channels=None, z_std=1.0,
        n_steps=500, eps=1e-5, return_trajectory=False
):
    rngs = jax.random.split(rng, batch_size)
    single_sample_shape = (latent_size, latent_size, z_channels)

    init_x = jax.vmap(lambda key: jax.random.normal(key, single_sample_shape))(rngs)
    init_x = init_x * marginal_prob_std_fn(jnp.array([1.0]))[0]

    timesteps = jnp.linspace(1.0, eps, n_steps + 1)
    x = init_x

    trajectory = [] if return_trajectory else None

    for i in tqdm(range(n_steps), desc="DDPM Sampling"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        vec_t = jnp.ones(batch_size) * t_now
        vec_t_next = jnp.ones(batch_size) * t_next

        eps_theta = ldm_model.apply({'params': ldm_params}, x, vec_t)

        alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

        pred_x0 = (x - std_now * eps_theta) / sqrt_alpha_bar_now
        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

        noise = jax.random.normal(jax.random.fold_in(rng, i), x.shape)
        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_theta) + (sigma * noise)
        if return_trajectory:
            trajectory.append(pred_x0)
    final_latent = x

    grid = None
    if ae_model is not None:
        target_channels = vae_z_channels if vae_z_channels is not None else z_channels

        if x.shape[-1] < target_channels:
            B, H, W, C = x.shape
            padded_x = jnp.zeros((B, H, W, target_channels))
            x = padded_x.at[..., :C].set(x)

        z_for_decode = x * z_std
        x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)
        x_hat = jnp.clip(x_hat, 0., 1.)
        x_hat = jnp.transpose(x_hat, (0, 3, 1, 2))
        x_hat_t = torch.from_numpy(np.asarray(x_hat))
        grid = make_grid(x_hat_t, nrow=int(jnp.sqrt(batch_size)))
    traj_ret = jnp.array(trajectory) if return_trajectory else None
    return grid, final_latent, traj_ret


def Euler_Maruyama_sampler(
        rng, ldm_model, ldm_params, ae_model, ae_params,
        marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn,
        latent_size, batch_size, z_channels, vae_z_channels=None, z_std=1.0,
        n_steps=500, eps=1e-5, return_trajectory=False
):
    """
    Generate samples using the Euler-Maruyama solver for the reverse SDE.
    Assumes a VP-SDE (Variance Preserving) structure compatible with the
    provided marginal_prob_std_fn.
    """
    rngs = jax.random.split(rng, batch_size)
    single_sample_shape = (latent_size, latent_size, z_channels)

    # 1. Initialization (Same as DDPM)
    init_x = jax.vmap(lambda key: jax.random.normal(key, single_sample_shape))(rngs)
    init_x = init_x * marginal_prob_std_fn(jnp.array([1.0]))[0]

    timesteps = jnp.linspace(1.0, eps, n_steps + 1)
    x = init_x

    trajectory = [] if return_trajectory else None
    # 2. Sampling Loop
    for i in tqdm(range(n_steps), desc="Euler-Maruyama Sampling"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_now - t_next  # Step size (positive)

        # Broadcast time to batch
        vec_t = jnp.ones(batch_size) * t_now

        # Get Model Prediction (Noise eps)
        eps_theta = ldm_model.apply({'params': ldm_params}, x, vec_t)

        # Get SDE coefficients
        # g(t) for the diffusion term
        g_t = diffusion_coeff_fn(vec_t)[:, None, None, None]
        # sigma(t) for converting noise prediction to score
        std_t = marginal_prob_std_fn(vec_t)[:, None, None, None]

        # Calculate Score: score = -epsilon / std
        score = -eps_theta / std_t

        # VP-SDE Drift term f(x,t) = -0.5 * beta(t) * x
        # Note: beta(t) = g(t)^2
        beta_t = g_t ** 2
        drift_f = -0.5 * beta_t * x

        # Reverse SDE Drift: f_rev = f(x,t) - g(t)^2 * score
        # f_rev = -0.5 * beta * x - beta * score
        rev_drift = drift_f - (beta_t * score)

        # Diffusion term noise
        noise = jax.random.normal(jax.random.fold_in(rng, i), x.shape)

        # Euler-Maruyama Step
        # x_{t-1} = x_t + rev_drift * dt + g(t) * sqrt(dt) * z
        x_mean = x + (rev_drift * dt)
        x = x_mean + (g_t * jnp.sqrt(dt) * noise)
        if return_trajectory:
            alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
            pred_x0 = (x - std_t * eps_theta) / jnp.sqrt(alpha_bar_now + 1e-5)
            trajectory.append(pred_x0)
    final_latent = x

    # 3. Decoding / Reconstruction (Identical to Reference)
    grid = None
    if ae_model is not None:
        target_channels = vae_z_channels if vae_z_channels is not None else z_channels

        # Handle channel mismatch if LDM latent dim differs from VAE latent dim
        if x.shape[-1] < target_channels:
            B, H, W, C = x.shape
            padded_x = jnp.zeros((B, H, W, target_channels))
            x = padded_x.at[..., :C].set(x)

        # Decode
        z_for_decode = x * z_std
        x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)

        # Process image for Grid
        x_hat = jnp.clip(x_hat, 0., 1.)
        x_hat = jnp.transpose(x_hat, (0, 3, 1, 2))  # NHWC -> NCHW
        x_hat_t = torch.from_numpy(np.asarray(x_hat))
        grid = make_grid(x_hat_t, nrow=int(jnp.sqrt(batch_size)))
    traj_ret = jnp.array(trajectory) if return_trajectory else None
    return grid, final_latent, traj_ret