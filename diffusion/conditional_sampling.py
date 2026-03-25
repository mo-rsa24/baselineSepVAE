import jax
import jax.numpy as jnp
from tqdm import tqdm

from diffusion.vp_equation import alpha_bar_fn, marginal_prob_std_fn


def _ddpm_update(model, params, x, t_now, t_next, z_common, rng):
    batch_size = x.shape[0]
    vec_t = jnp.ones(batch_size, dtype=x.dtype) * t_now
    vec_t_next = jnp.ones(batch_size, dtype=x.dtype) * t_next

    eps_theta = model.apply({"params": params}, x, vec_t, z_common)

    alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
    alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
    std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]
    sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

    pred_x0 = (x - std_now * eps_theta) / sqrt_alpha_bar_now
    ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
    sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
    sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
    dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

    noise = jax.random.normal(rng, x.shape, dtype=x.dtype)
    x_next = jnp.sqrt(alpha_bar_next) * pred_x0 + dir_xt_coeff * eps_theta + sigma * noise
    return x_next, pred_x0


def sample_z_cardio(
    rng,
    ldm_model,
    ldm_params,
    z_common_fixed,
    n_steps: int = 250,
    eps: float = 1e-5,
    return_trajectory: bool = False,
):
    batch_size, height, width, z_channels = z_common_fixed.shape
    x = jax.random.normal(rng, (batch_size, height, width, z_channels), dtype=z_common_fixed.dtype)
    x = x * marginal_prob_std_fn(jnp.array([1.0], dtype=z_common_fixed.dtype))[0]

    timesteps = jnp.linspace(1.0, eps, n_steps + 1, dtype=z_common_fixed.dtype)
    trajectory = [] if return_trajectory else None

    for i in tqdm(range(n_steps), desc="Conditional DDPM Sampling"):
        step_rng = jax.random.fold_in(rng, i)
        x, pred_x0 = _ddpm_update(
            ldm_model, ldm_params, x, timesteps[i], timesteps[i + 1], z_common_fixed, step_rng
        )
        if return_trajectory:
            trajectory.append(pred_x0)

    final = x
    traj_ret = jnp.stack(trajectory, axis=0) if return_trajectory else None
    return final, traj_ret


def refine_z_cardio(
    rng,
    ldm_model,
    ldm_params,
    z_common,
    z_cardio,
    t_start: float = 0.35,
    n_steps: int = 150,
    eps: float = 1e-5,
    return_trajectory: bool = False,
):
    if not (eps < t_start <= 1.0):
        raise ValueError(f"t_start must be in (eps, 1], got {t_start}")

    sigma = marginal_prob_std_fn(jnp.array([t_start], dtype=z_cardio.dtype))[0]
    alpha_bar = alpha_bar_fn(jnp.array([t_start], dtype=z_cardio.dtype))[0]
    alpha = jnp.sqrt(alpha_bar + 1e-5)
    init_noise = jax.random.normal(rng, z_cardio.shape, dtype=z_cardio.dtype)
    x = alpha * z_cardio + sigma * init_noise

    timesteps = jnp.linspace(t_start, eps, n_steps + 1, dtype=z_cardio.dtype)
    trajectory = [] if return_trajectory else None

    for i in tqdm(range(n_steps), desc="Conditional DDPM Refinement"):
        step_rng = jax.random.fold_in(rng, i)
        x, pred_x0 = _ddpm_update(
            ldm_model, ldm_params, x, timesteps[i], timesteps[i + 1], z_common, step_rng
        )
        if return_trajectory:
            trajectory.append(pred_x0)

    final = x
    traj_ret = jnp.stack(trajectory, axis=0) if return_trajectory else None
    return final, traj_ret
