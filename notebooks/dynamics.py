from typing import List

import torch
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from notebooks.utils import get_text_embedding


@torch.no_grad
def get_vel(unet, t, sigma, latents, embeddings, eps=None, get_div=False, device=torch.device("cuda"), dtype=torch.float16):
    t = t.to(device, dtype=torch.float16)

    def v(_x, _e):
        _x = _x.to(device=device, dtype=dtype)
        _e = _e.to(device=device, dtype=dtype)

        denom = torch.sqrt(sigma * sigma + 1.0)  # stays fp16 now
        x_in = _x / denom

        with torch.autocast("cuda", dtype=dtype):
            return unet(x_in, t, encoder_hidden_states=_e).sample
    # v = lambda _x, _e: unet(_x / ((sigma**2 + 1) ** 0.5), t, encoder_hidden_states=_e).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    if get_div:
        with torch.enable_grad():
            with sdpa_kernel(SDPBackend.MATH):
                vel, div = torch.func.jvp(v, (latent_input, embeds), (eps, torch.zeros_like(embeds)))
                div = -(eps*div).sum((1,2,3))
    else:
        with torch.no_grad():
            vel = v(latent_input, embeds)
            div = None

    return vel, div

def get_latents(scheduler, z_channels: int =4, device = torch.device("cuda"), dtype = torch.float16,  num_inference_steps: int = 500, batch_size: int = 6, latent_width: int = 64, latent_height: int = 64, seed: int = None):
    # Use provided seed, or respect global seed if not provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None  # Use global random state (set by torch.manual_seed)

    latents = torch.randn(
        (batch_size, z_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype
    )
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    return latents


def stochastic_super_diff_and(
        latents,
        obj_prompt: List[str],
        bg_prompt: List[str],
        scheduler: EulerDiscreteScheduler,
        unet,
        tokenizer,
        text_encoder,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
        batch_size: int = 4,
        device=torch.device("cuda"),
        dtype=torch.float16,
        lift: float = 0.0
):
    obj_embeddings = get_text_embedding(obj_prompt * batch_size, tokenizer, text_encoder, device)
    bg_embeddings = get_text_embedding(bg_prompt * batch_size, tokenizer, text_encoder, device)
    uncond_embeddings = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        # Pass unet and dtype
        vel_obj, _ = get_vel(unet, t, sigma, latents, [obj_embeddings], device=device, dtype=dtype)
        vel_bg, _ = get_vel(unet, t, sigma, latents, [bg_embeddings], device=device, dtype=dtype)
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeddings], device=device, dtype=dtype)

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

        # SuperDiff Logic
        dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise

        # Terms for Kappa
        term1 = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
        term2 = (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        term3 = sigma * lift / num_inference_steps

        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

        # Update Kappa with stability epsilon
        kappa[i + 1] = numerator / (denominator + 1e-8)

        # Composite Vector Field
        vf = vel_uncond + guidance_scale * (
                    (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))

        dx = 2 * dsigma * vf + noise
        latents += dx

        ll_obj[i + 1] = ll_obj[i] + (-torch.abs(dsigma) / sigma * (vel_obj) ** 2 - (dx * (vel_obj / sigma))).sum(
            (1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (-torch.abs(dsigma) / sigma * (vel_bg) ** 2 - (dx * (vel_bg / sigma))).sum((1, 2, 3))
    return latents, kappa, ll_obj, ll_bg