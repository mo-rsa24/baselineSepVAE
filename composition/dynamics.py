from operator import itemgetter
import jax
import jax.numpy as jnp
from composition.utils import decode_image, save_image_grid
from diffusion.vp_equation import marginal_prob_std, score_function_hutchinson_estimator
import numpy as np
from typing import Tuple
from diffusion.sampling import Euler_Maruyama_sampler, DDPM_ancestral_sampler
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn
from tqdm import tqdm


def get_score(model, params, x, t):
    """
    Computes the score (epsilon) for a given latent x and time t.
    Note: ScoreNet in run/ldm.py is trained to predict epsilon.
    t is expected in [0, 1].
    """
    return model.apply({'params': params}, x, t)


def batch_dot(a, b):
    """
    Computes the dot product <a, b> per batch item.
    inputs: (B, H, W, C)
    output: (B,)
    """
    # 1. Multiply element-wise
    prod = a * b
    # 2. Flatten all dims except batch (B, -1)
    prod_flat = prod.reshape(prod.shape[0], -1)
    # 3. Sum over the flattened dimension to get a scalar per batch
    return prod_flat.sum(axis=-1)

def generate_ldm_samples(
    rng,
    model_normal, params_normal,
    model_tb, params_tb,
    ae_model, ae_params,
    latent_size, z_channels,
    batch_size: int = 4,
    sampler: str = "Ancestral",
    n_steps=200,
    latent_scale_factor=0.99937266,
    output_path: str ="x_ray_samples.png"
):
    """
        Generates independent samples from both models for baseline comparison and PCA visualization.
        Updated to accept explicit model/param arguments instead of extraction containers.
        """
    print("Sampling Images For Both Models (Independent Baseline)...")

    # 1. Select the Sampler
    if sampler == 'Euler':
        sampler_fn = Euler_Maruyama_sampler
    elif sampler == 'Ancestral' or sampler == 'Faithful' or sampler == 'PoE':
        # Faithful and PoE usually default to Ancestral for their independent baselines
        sampler_fn = DDPM_ancestral_sampler
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    sample_kwargs = {
        "rng": rng,
        "ae_model": ae_model,
        "ae_params": ae_params,
        "marginal_prob_std_fn": marginal_prob_std_fn,
        "diffusion_coeff_fn": diffusion_coeff_fn,
        "alpha_bar_fn": alpha_bar_fn,
        "latent_size": latent_size,
        "batch_size": batch_size,
        "z_channels": z_channels,
        "z_std": 1.0 / latent_scale_factor,
        "n_steps": n_steps,
    }
    # 3. Generate Samples
    print(f"Generating samples for Model 1 (Normal) using {sampler}...")
    _, latents_normal = sampler_fn(ldm_model=model_normal, ldm_params=params_normal, **sample_kwargs)

    print(f"Generating samples for Model 2 (TB) using {sampler}...")
    _, latents_tb = sampler_fn(ldm_model=model_tb, ldm_params=params_tb, **sample_kwargs)

    # 4. Decode and Construct 4x2 Grid
    # Stack latents: [Normal_1...Normal_4, TB_1...TB_4]
    combined_latents = jnp.concatenate([latents_normal, latents_tb], axis=0)

    print("Decoding independent samples...")
    decoded_imgs = decode_image(ae_model, ae_params, combined_latents, latent_scale_factor=latent_scale_factor)
    decoded_imgs_np = np.array(decoded_imgs)  # Shape: (8, H, W)
    sample_out_path = output_path.replace(".png", f"_reference_{sampler}.png")
    save_image_grid(decoded_imgs_np, sample_out_path , 4, 2)
    print(f"Reference samples (Normal top, TB bottom) saved to {sample_out_path}")
    return latents_normal, latents_tb


def generate_batched_baseline(
        rng,
        model_pkg,
        num_samples,
        batch_size,
        sampler="Ancestral",
        n_steps=200,
        latent_scale_factor=0.99937266,
        return_trajectory=False
):
    """
    Generates 'num_samples' latents using a standard sampler (no composition).
    Loops over batches and returns the concatenated result.
    """
    model, params, _, lsize, zch = itemgetter("ldm_model", "params", "config", "latent_size", "z_channels")(model_pkg)

    if sampler == 'Euler':
        sampler_fn = Euler_Maruyama_sampler
    elif sampler == 'Ancestral' or sampler == 'Faithful' or sampler == 'PoE':
        # Faithful and PoE usually default to Ancestral for their independent baselines
        sampler_fn = DDPM_ancestral_sampler
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Kwargs for the standard samplers
    sample_kwargs = {
        "ae_model": None, "ae_params": None,
        "marginal_prob_std_fn": marginal_prob_std_fn,
        "diffusion_coeff_fn": diffusion_coeff_fn,
        "alpha_bar_fn": alpha_bar_fn,
        "latent_size": lsize,
        "z_channels": zch,
        "z_std": 1.0 / latent_scale_factor,
        "n_steps": n_steps,
    }

    all_latents = []
    num_batches = int(np.ceil(num_samples / batch_size))
    viz_trajectory = None
    # Loop over batches
    for i in tqdm(range(num_batches), desc="Baseline Generation"):
        # Calculate current batch size (handle remainder)
        current_bs = min(batch_size, num_samples - len(all_latents) * batch_size)
        step_rng = jax.random.fold_in(rng, i)
        capture_this_batch = (return_trajectory and i == 0)
        sample_kwargs["batch_size"] = current_bs
        sample_kwargs["return_trajectory"] = capture_this_batch
        _, batch_latents, batch_traj = sampler_fn(ldm_model=model, ldm_params=params, rng=step_rng, **sample_kwargs)
        if capture_this_batch:
            viz_trajectory = batch_traj
        all_latents.append(batch_latents)

    return jnp.concatenate(all_latents, axis=0), viz_trajectory


def run_composition_batched(
        rng,
        args,
        model_n, params_n,
        model_t, params_t,
        latent_shape
):
    """
    Orchestrates the Composition Sampling loop.
    1. Loops 'num_samples' / 'batch_size' times.
    2. Runs the specific composition sampler for each batch.
    3. Accumulates Latents, Kappas, and Logs.
    4. Returns consolidated arrays.
    """
    all_final_latents = []
    all_kappas = []
    all_log_a = []
    all_log_b = []

    lsize, zch = latent_shape
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    print(f"Starting Composition: {args.sampler}")
    print(f"Total Samples: {args.num_samples} | Batch Size: {args.batch_size} | Iterations: {num_batches}")
    viz_trajectory = None
    for i in range(num_batches):
        capture_traj = (i == 0)
        current_bs = min(args.batch_size, args.num_samples - len(all_final_latents) * args.batch_size)
        step_rng = jax.random.fold_in(rng, i)

        # Generate initial noise for this batch
        batch_latents = jax.random.normal(step_rng, (current_bs, lsize, lsize, zch))

        # Route to correct sampler
        if args.sampler == 'Faithful':
            lift = None if args.lift == 0.0 else args.lift
            res = ddpm_ancestral_superdiff_and_uncond_faithful(
                step_rng, batch_latents, model_n, params_n, model_t, params_t,
                num_inference_steps=args.steps, lift=lift, return_trajectory=capture_traj
            )
        elif args.sampler == 'Ancestral':
            res = ddpm_ancestral_superdiff_and_uncond(
                step_rng, batch_latents, model_n, params_n, model_t, params_t,
                num_inference_steps=args.steps, lift=args.lift, return_trajectory=capture_traj
            )
        elif args.sampler == 'Euler':
            res = stochastic_super_diff_and_uncond(
                batch_latents, model_n, params_n, model_t, params_t,
                num_inference_steps=args.steps, lift=args.lift, score=args.score,
                return_trajectory=capture_traj
            )
        elif args.sampler == 'PoE':
            res = ddpm_ancestral_poe_tracking(
                step_rng, batch_latents, model_n, params_n, model_t, params_t,
                num_inference_steps=args.steps, return_trajectory=capture_traj
            )
        else:
            raise ValueError(f"Unknown sampler {args.sampler}")

        l, k, la, lb, traj = res

        if capture_traj:
            viz_trajectory = traj

        all_final_latents.append(l)
        if k is not None: all_kappas.append(k)
        all_log_a.append(la)
        all_log_b.append(lb)

        print(f"Batch {i + 1}/{num_batches} finished.")

    # Consolidate Results
    final_latents = jnp.concatenate(all_final_latents, axis=0)

    # Logs are usually (Steps, Batch). We concatenate on Axis 1 to get (Steps, Total_Samples)
    full_log_a = jnp.concatenate(all_log_a, axis=1) if all_log_a else None
    full_log_b = jnp.concatenate(all_log_b, axis=1) if all_log_b else None
    full_kappas = jnp.concatenate(all_kappas, axis=1) if all_kappas else None

    return final_latents, full_kappas, full_log_a, full_log_b, viz_trajectory

def ddpm_ancestral_superdiff_and_uncond(
    rng,
    latents,
    model_normal, params_normal,
    model_tb, params_tb,
    num_inference_steps,
    lift=0.0,
    kappa_clip=2.0,
    return_trajectory=False
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

    log_diff_history = []  # To track |log q_A - log q_B|

    # Initialize Relative Log Densities (start at 0)
    log_q_normal = jnp.zeros((latents.shape[0],))
    log_q_tb = jnp.zeros((latents.shape[0],))

    log_q_normal_hist = []
    log_q_tb_hist = []

    dt = 1.0 / num_inference_steps

    trajectory = [] if return_trajectory else None

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
        if lift is not None:
            numerator = numerator + (std_now * (lift[:, None, None, None] / num_inference_steps))

        denominator = _sum_except_batch((sN - sT) ** 2) + 1e-12
        kappa = numerator / denominator

        # stability clamp (recommended for TB ∧ Normal)
        if kappa_clip is not None:
            kappa = jnp.clip(kappa, -kappa_clip, kappa_clip)

        kappa_log.append(kappa.squeeze())

        # ---- (5) Mix scores, convert back to ε̂_mix ----
        s_mix = sN + kappa * (sT - sN)

        S_NN = batch_dot(sN, sN)
        S_TT = batch_dot(sT, sT)
        # Dot products for update
        dot_mix_N = batch_dot(s_mix, sN)
        dot_mix_T = batch_dot(s_mix, sT)

        d_log_N = (dot_mix_N - 0.5 * g2.squeeze() * S_NN) * dt
        d_log_T = (dot_mix_T - 0.5 * g2.squeeze() * S_TT) * dt

        log_q_normal += d_log_N
        log_q_tb += d_log_T

        log_q_normal_hist.append(log_q_normal)
        log_q_tb_hist.append(log_q_tb)
        log_diff_history.append(log_q_normal - log_q_tb)

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
        if return_trajectory:
            trajectory.append(pred_x0)
    return x, jnp.array(kappa_log), jnp.array(log_q_normal_hist), jnp.array(log_q_tb_hist), trajectory

def stochastic_super_diff_and_uncond(
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        lift=0.0,
        score = False,
        return_trajectory=False
):
    """
    Implements Stochastic SuperDiff AND logic for two unconditional score fields.
    Mapping:
      - Normal Model -> Background/Reference (vel_bg)
      - TB Model     -> Object/Constraint (vel_obj)
    """

    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)
    trajectory = [] if return_trajectory else None
    # Initialize logs
    kappa_log = []

    log_diff_history = []  # To track |log q_A - log q_B|

    # Initialize Relative Log Densities (start at 0)
    log_q_normal = jnp.zeros((latents.shape[0],))
    log_q_tb = jnp.zeros((latents.shape[0],))

    log_q_normal_hist = []
    log_q_tb_hist = []

    dt = 1.0 / num_inference_steps

    for i in tqdm(range(num_inference_steps), desc="SuperDiff AND"):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = jnp.full((latents.shape[0],), t_current)
        sigma = marginal_prob_std(t_current)
        sigma_next = marginal_prob_std(t_next)
        dsigma = sigma_next - sigma

        # Get g(t) for the log-density estimator (Shadow Logger requirement)
        g_t = diffusion_coeff_fn(t_batch)  # Shape (B,)
        g2 = g_t ** 2

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

        def simple_batch_dot(a, b):
            return (a * b).reshape(a.shape[0], -1).sum(axis=-1)

            # Compute dot products just for logging

        S_NN = simple_batch_dot(vel_normal, vel_normal)
        S_TT = simple_batch_dot(vel_tb, vel_tb)
        dot_mix_N = simple_batch_dot(vf, vel_normal)
        dot_mix_T = simple_batch_dot(vf, vel_tb)

        d_log_N = (dot_mix_N - 0.5 * g2 * S_NN) * dt
        d_log_T = (dot_mix_T - 0.5 * g2 * S_TT) * dt

        log_q_normal += d_log_N
        log_q_tb += d_log_T

        log_q_normal_hist.append(log_q_normal)
        log_q_tb_hist.append(log_q_tb)
        log_diff_history.append(log_q_normal - log_q_tb)
        if return_trajectory:
            trajectory.append(latents)
    return latents, jnp.array(kappa_log), jnp.array(log_q_normal_hist), jnp.array(log_q_tb_hist), trajectory


def prepare_latents(args, lsize, zch,  lift_values: Tuple[float] = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0), num_rows: int = 4):
    """Prepares initial latents for either a Sweep or a Single Run."""
    if args.sweep:
        latents, lift_batch = get_sweep_configuration(
            lsize, z_channels=zch, lift_values=lift_values,
            num_rows=num_rows, seed=args.seed
        )
        return latents, lift_batch
    else:
        print(f"Mode: Single Run (Batch={args.batch_size})")
        rng = jax.random.PRNGKey(args.seed)
        latents = jax.random.normal(rng, (args.batch_size, lsize, lsize, zch))
        if args.sampler == 'Euler':
            lift_batch = args.lift  # Scalar or broadcast if needed by sampler
        else:
            lift_batch = None
        return latents, lift_batch

def get_sweep_configuration(latent_size, z_channels: int =4, lift_values: Tuple[float] = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0), num_rows: int = 4, seed: int =0):
    rng = jax.random.PRNGKey(seed)
    row_latents = []
    t0 = 1.0
    sigma_max = marginal_prob_std_fn(jnp.array([t0]))[0]
    for i in range(num_rows):
        rng, seed_rng = jax.random.split(rng)
        # Generate ONE sample (NHWC)
        z0 = jax.random.normal(seed_rng, (1, latent_size, latent_size, z_channels)) * sigma_max
        # Tile it NUM_COLS times
        z0_row = jnp.tile(z0, (len(lift_values), 1, 1, 1))
        row_latents.append(z0_row)
    latents = jnp.concatenate(row_latents, axis=0)
    lifts_jnp = jnp.array(lift_values)
    lift_batch = jnp.tile(lifts_jnp, num_rows)  # Shape (Total Batch,)
    print(f"Total Batch Shape: {latents.shape}")
    return latents, lift_batch

def ddpm_ancestral_superdiff_and_uncond_faithful(
        rng,
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        lift=0.0,
        kappa_clip=2.0,
        return_trajectory=False
):
    """
    Faithful implementation of SUPERDIFF Algorithm 1 (AND logic) for JAX/Flax.

    Instead of Hutchinson's estimator, this uses the Itô Density Estimator logic
    (Prop 6 & Thm 1 from the paper) to solve for kappa analytically using only
    dot products of the scores.

    Inputs/outputs match ddpm_ancestral_superdiff_and_uncond.
    """
    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)
    kappa_log = []
    log_diff_history = []  # To track |log q_A - log q_B|
    # Initialize Relative Log Densities (start at 0)
    log_q_normal = jnp.zeros((latents.shape[0],))
    log_q_tb = jnp.zeros((latents.shape[0],))

    log_q_normal_hist = []
    log_q_tb_hist = []

    x = latents
    dt = 1.0 / num_inference_steps

    trajectory = [] if return_trajectory else None
    # Helper: Sum over (H, W, C) to get shape (Batch, 1, 1, 1)
    def _sum_except_batch(xx):
        axes = tuple(range(1, xx.ndim))
        return jnp.sum(xx, axis=axes, keepdims=True)

    for i in tqdm(range(num_inference_steps), desc="SuperDiff Faithful (DDPM)"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        B = x.shape[0]
        vec_t = jnp.full((B,), t_now)
        vec_t_next = jnp.full((B,), t_next)

        # ---- (1) ε predictions from both experts ----
        eps_normal = model_normal.apply({'params': params_normal}, x, vec_t)
        eps_tb = model_tb.apply({'params': params_tb}, x, vec_t)

        # ---- (2) Convert ε -> score: s = -ε / σ(t) ----
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]  # σ(t)
        sN = -eps_normal / std_now
        sT = -eps_tb / std_now

        # ---- (3) Compute Dot Products (Interaction Matrix) ----
        # S_ij = <s_i, s_j>
        S_NN = batch_dot(sN, sN)[:, None, None, None]
        S_TT = batch_dot(sT, sT)[:, None, None, None]
        S_NT = batch_dot(sN, sT)[:, None, None, None]

        # ---- (4) Solve for Kappa (Analytic Solution for AND) ----
        # We want d(log q_N) = d(log q_T).
        # Derivation from Ito Density Estimator (Thm 1):
        # kappa * ||sT - sN||^2 = <sN, sN - sT> + 0.5 * g(t)^2 * (||sT||^2 - ||sN||^2)

        g_t = diffusion_coeff_fn(vec_t)[:, None, None, None]  # g(t)
        g2 = g_t ** 2

        # Term A: Drift matching <sN, sN - sT>
        term_drift = S_NN - S_NT

        # Term B: Ito Correction 0.5 * g^2 * (||sT||^2 - ||sN||^2)
        # This term replaces the Divergence (Hutchinson) estimator!
        term_ito = 0.5 * g2 * (S_TT - S_NN)

        # Term C: Lift (Bias)
        # lift is scaled by sigma or 1/steps effectively
        term_lift = 0.0
        if lift is not None:
            # Matches the scaling in your original function
            term_lift = std_now * (lift[:, None, None, None] / num_inference_steps)

        numerator = term_drift + term_ito + term_lift

        # Denominator: ||sT - sN||^2 = S_TT + S_NN - 2*S_NT
        denominator = (S_TT + S_NN - 2 * S_NT) + 1e-12

        kappa = numerator / denominator

        # Stability clamp
        if kappa_clip is not None:
            kappa = jnp.clip(kappa, -kappa_clip, kappa_clip)

        kappa_log.append(kappa.squeeze())

        # ---- (5) Mix scores ----
        # Note: The derivation assumes mixing vector u = sN + kappa * (sT - sN)
        s_mix = sN + kappa * (sT - sN)

        # Dot products for update
        dot_mix_N = batch_dot(s_mix, sN)  # (B,)
        dot_mix_T = batch_dot(s_mix, sT)  # (B,)

        d_log_N = (dot_mix_N - 0.5 * g2.squeeze() * S_NN) * dt
        d_log_T = (dot_mix_T - 0.5 * g2.squeeze() * S_TT) * dt

        log_q_normal += d_log_N
        log_q_tb += d_log_T

        log_q_normal_hist.append(log_q_normal)
        log_q_tb_hist.append(log_q_tb)
        log_diff_history.append(log_q_normal - log_q_tb)


        # ---- (6) DDPM Ancestral Posterior Step ----
        # (Exact copy of the sampler logic from your existing function)
        # Convert back to epsilon for the DDPM sampler
        eps_mix = -std_now * s_mix
        alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

        # x0 estimate
        pred_x0 = (x - std_now * eps_mix) / sqrt_alpha_bar_now

        # Posterior variance calculation
        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

        rng, kn = jax.random.split(rng)
        noise = jax.random.normal(jax.random.fold_in(kn, i), x.shape)

        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_mix) + (sigma * noise)
        if return_trajectory:
            trajectory.append(pred_x0)  # Store pred_x0 for consistency with baseline samplers

    return x, jnp.array(kappa_log), jnp.array(log_q_normal_hist), jnp.array(log_q_tb_hist), trajectory


def ddpm_ancestral_poe_tracking(
        rng,
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        weights=(1.0, 1.0),  # (w_normal, w_tb) - Set to (1.0, 1.0) for strict PoE, (0.5, 0.5) for averaging
        return_trajectory=False
):
    """
    Product-of-Experts (PoE) composition with DDPM Ancestral Sampling.

    Logic:
      s_mix = w_A * s_A + w_B * s_B

    Includes tracking of log-density trajectories using the Itô estimator
    to compare with SuperDiff.
    """
    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)
    trajectory = [] if return_trajectory else None
    # Initialize Histories
    log_q_normal_hist = []
    log_q_tb_hist = []

    # Initialize Relative Log Densities (start at 0)
    log_q_normal = jnp.zeros((latents.shape[0],))
    log_q_tb = jnp.zeros((latents.shape[0],))

    x = latents
    dt = 1.0 / num_inference_steps
    w_n, w_tb = weights

    # Helper: Sum over (H, W, C)
    def _sum_except_batch(xx):
        axes = tuple(range(1, xx.ndim))
        return jnp.sum(xx, axis=axes, keepdims=True)

    for i in tqdm(range(num_inference_steps), desc="PoE (DDPM Ancestral)"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        B = x.shape[0]
        vec_t = jnp.full((B,), t_now)
        vec_t_next = jnp.full((B,), t_next)

        # ---- (1) Get Scores ----
        eps_normal = model_normal.apply({'params': params_normal}, x, vec_t)
        eps_tb = model_tb.apply({'params': params_tb}, x, vec_t)

        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]
        sN = -eps_normal / std_now
        sT = -eps_tb / std_now

        # ---- (2) PoE Mix ----
        # Standard PoE is just the sum.
        # Note: Summing scores usually results in higher magnitude gradients
        # which can "sharpen" or "burn" the image.
        s_mix = (w_n * sN) + (w_tb * sT)

        # Convert back to epsilon for sampler
        eps_mix = -std_now * s_mix

        # ---- (3) Track Log Trajectories (Itô Estimator) ----
        # We track how the individual models "feel" about this PoE path
        g_t = diffusion_coeff_fn(vec_t)[:, None, None, None]
        g2 = g_t ** 2

        # Calculate dot products
        S_NN = batch_dot(sN, sN)
        S_TT = batch_dot(sT, sT)
        dot_mix_N = batch_dot(s_mix, sN)
        dot_mix_T = batch_dot(s_mix, sT)

        # Update logs: d(log q) ≈ <s_mix, s_i> - 0.5 * g^2 * ||s_i||^2
        d_log_N = (dot_mix_N - 0.5 * g2.squeeze() * S_NN) * dt
        d_log_T = (dot_mix_T - 0.5 * g2.squeeze() * S_TT) * dt

        log_q_normal += d_log_N
        log_q_tb += d_log_T

        log_q_normal_hist.append(log_q_normal)
        log_q_tb_hist.append(log_q_tb)

        # ---- (4) DDPM Step ----
        alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

        pred_x0 = (x - std_now * eps_mix) / sqrt_alpha_bar_now

        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

        rng, kn = jax.random.split(rng)
        noise = jax.random.normal(jax.random.fold_in(kn, i), x.shape)

        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_mix) + (sigma * noise)
        if return_trajectory:
            trajectory.append(pred_x0)
    return x, None, jnp.array(log_q_normal_hist), jnp.array(log_q_tb_hist), trajectory

def run_sampler(sampler_name, latents, model_n, params_n, model_t, params_t, steps, lift, score_mode=False):
    """Routes to the correct sampler function."""

    if sampler_name == 'Ancestral':  # faithful
        print("Running Ancestral SuperDiff (DDPM)...")
        return ddpm_ancestral_superdiff_and_uncond(
            jax.random.PRNGKey(0), latents, model_n, params_n, model_t, params_t,
            num_inference_steps=steps, lift=lift, kappa_clip=2.0
        )
    elif sampler_name == 'Faithful':
        print("Running Faithful SuperDiff (DDPM)...")
        return ddpm_ancestral_superdiff_and_uncond_faithful(
            jax.random.PRNGKey(0), latents, model_n, params_n, model_t, params_t,
            num_inference_steps=steps, lift=lift, kappa_clip=2.0
        )
    elif sampler_name == 'Euler':
        print("Running Euler SuperDiff (Tracking)...")
        return stochastic_super_diff_and_uncond(
            latents, model_n, params_n, model_t, params_t,
            num_inference_steps=steps, lift=lift, score=score_mode
        )

    elif sampler_name == 'PoE':
        print("Running Product-of-Experts (PoE)...")
        # Lift is unused in PoE, but we pass it for interface consistency if needed
        return ddpm_ancestral_poe_tracking(
            jax.random.PRNGKey(0), latents, model_n, params_n, model_t, params_t,
            num_inference_steps=steps
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
