import matplotlib.pyplot as plt
from PIL import Image
import torch
from IPython.display import display
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
import os

dtype = torch.float16
device = torch.device("cuda")

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    use_safetensors=True,
    torch_dtype=torch.float16
)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    use_safetensors=True,
    torch_dtype=torch.float16
)

torch_device = torch.device('cuda')
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

from diffusers import EulerDiscreteScheduler

scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

import torch


def _l2_norm_sq(x, eps=1e-8):
    """Calculates the squared L2 norm for a batch of PyTorch tensors."""
    return (x ** 2).sum(dim=(1, 2, 3)) + eps


@torch.no_grad()
def sampler_superdiff_sd(
        unet,
        scheduler,
        latents,
        obj_embeddings,
        bg_embeddings,
        uncond_embeddings,
        guidance_scale=7.5,
        num_inference_steps=50,
        do_diagnostics=True
):
    scheduler.set_timesteps(num_inference_steps)
    kappa_list = []

    # Main sampling loop
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # Get sigma for the current and next timestep
        sigma_t = scheduler.sigmas[i]
        sigma_t_prev = scheduler.sigmas[i + 1] if i < len(scheduler.timesteps) - 1 else 0
        latent_model_input = torch.cat([latents] * 3)
        # Scale the model input per scheduler's requirement
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Concatenate embeddings for a single model pass
        prompt_embeddings = torch.cat([uncond_embeddings, obj_embeddings, bg_embeddings])

        # Predict the noise for all three conditions
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeddings).sample
        eps_uncond, eps_obj_raw, eps_bg_raw = noise_pred.chunk(3)
        eps_obj = eps_uncond + guidance_scale * (eps_obj_raw - eps_uncond)
        eps_bg = eps_uncond + guidance_scale * (eps_bg_raw - eps_uncond)
        sA = -eps_obj / sigma_t  # Object score
        sB = -eps_bg / sigma_t  # Background score
        dsigma = sigma_t - sigma_t_prev  # This is a positive value, like `dt` in the original code
        drift_ind = eps_bg * dsigma
        noise_variance = (sigma_t ** 2 - sigma_t_prev ** 2)
        # Ensure variance is non-negative for the sqrt
        noise = torch.randn_like(latents) * torch.sqrt(torch.clamp(noise_variance, min=0.0))
        dx_ind = drift_ind + noise
        s_diff = sA - sB
        s_sum = sA + sB
        num_term1 = (dsigma * (sB - sA) * s_sum).sum(dim=(1, 2, 3))
        num_term2 = (dx_ind * s_diff).sum(dim=(1, 2, 3))
        numerator = num_term1 - num_term2
        composition_guidance_scale = 1.0
        denominator = 2 * dsigma * composition_guidance_scale * _l2_norm_sq(s_diff)
        kappa = torch.where(denominator.abs() > 1e-6, numerator / denominator, 0.5)

        if do_diagnostics:
            kappa_list.append(kappa.mean().item())
        s_combined = sB + kappa.reshape(-1, 1, 1, 1) * s_diff
        eps_combined = -s_combined * sigma_t
        pred_original_sample = latents + (sigma_t ** 2) * s_combined
        latents = scheduler.step(eps_combined, t, latents).prev_sample  # for some schedulers, eps is `model_output`
    return latents, kappa_list


import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@torch.no_grad()
def sampler_dynamic_weights(
        unet,
        scheduler,
        latents,
        embeddings_list,
        guidance_scale=7.5,
        num_inference_steps=50,
        logq_temp=1e6
):
    """
    Translates the logic of get_joint_stoch_vf from dynamics.py to PyTorch.
    Uses a dynamic, weighted-average of model scores for composition.
    """
    latents = latents.to(torch.float16)
    batch_size = latents.shape[0]

    uncond_embeddings = embeddings_list[0].to(torch.float16)
    prompt_embeddings_list = [emb.to(torch.float16) for emb in embeddings_list[1:]]
    num_prompts = len(prompt_embeddings_list)

    logq = torch.zeros(batch_size, num_prompts, device=latents.device)

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        t = t.to(torch.float16)
        latent_model_input = torch.cat([latents] * (1 + num_prompts))
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        all_embeddings = torch.cat([uncond_embeddings] + prompt_embeddings_list)

        # This call will now succeed
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=all_embeddings).sample

        eps_uncond, eps_prompts = noise_pred.chunk(1 + num_prompts)[0], noise_pred.chunk(1 + num_prompts)[1:]
        eps_prompts = torch.stack(eps_prompts)

        eps_guided = eps_uncond + guidance_scale * (eps_prompts - eps_uncond)

        weights = F.softmax(logq * logq_temp, dim=-1)
        weights_reshaped = weights.T.reshape(num_prompts, batch_size, 1, 1, 1)
        balanced_eps = (weights_reshaped * eps_guided).sum(dim=0)

        prev_sample = scheduler.step(balanced_eps, t, latents).prev_sample
        latents = prev_sample.to(torch.float16)

        error = torch.mean((eps_guided - balanced_eps) ** 2, dim=(2, 3, 4))
        logq = -error.T

    return latents

@torch.no_grad()
def get_image(latents, nrow, ncol, save_path=None, verbose=False):
    """
    Decodes latents into a PIL image, optionally saves and displays it.

    Args:
        latents (torch.Tensor): The latent tensor from the diffusion model.
        nrow (int): Number of rows in the image grid.
        ncol (int): Number of columns in the image grid.
        save_path (str, optional): Path to save the image. Defaults to None.
        verbose (bool, optional): If True, displays the image. Defaults to False.

    Returns:
        PIL.Image.Image: The generated image.
    """
    # 1. Decode latents into image tensor
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    # 2. Post-process the tensor to [0, 255] range
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)  # [B, 3, H, W]

    # 3. Rearrange to [B, H, W, C] for easier grid creation
    image = image.permute(0, 2, 3, 1)

    # 4. Build the grid manually
    rows = []
    i = 0
    for row_i in range(nrow):
        row_imgs = []
        for col_i in range(ncol):
            if i < len(image):
                row_imgs.append(image[i])
                i += 1
        if row_imgs:
            rows.append(torch.hstack(row_imgs))
    image_grid = torch.vstack(rows)

    # 5. Convert to PIL Image
    final_image = Image.fromarray(image_grid.cpu().numpy())

    # 6. Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_image.save(save_path)
        print(f"🖼️ Image saved to: {save_path}")

    # 7. Optionally display
    if verbose:
        display(final_image)

    return final_image


@torch.no_grad
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(torch_device))[0]

from torch.nn.attention import SDPBackend, sdpa_kernel

@torch.no_grad
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    sigma = sigma.to(torch.float16)
    t = t.to(torch.float16)
    v = lambda _x, _e: unet(_x / ((sigma**2 + 1) ** 0.5), t, encoder_hidden_states=_e).sample
    embeds = torch.cat(embeddings)
    latent_input = latents

    latent_input = latent_input.to(torch.float16)
    embeds = embeds.to(torch.float16)
    if get_div:
        if eps is not None:
            eps = eps.to(torch.float16)
        with sdpa_kernel(SDPBackend.MATH):
            vel, div = torch.func.jvp(v, (latent_input, embeds), (eps, torch.zeros_like(embeds)))
            div = -(eps*div).sum((1,2,3))
    else:
        vel = v(latent_input, embeds)
        div = torch.zeros([len(embeds)], device=torch_device)
    return vel, div

generator = torch.cuda.manual_seed(1)  # Seed generator to create the initial latent noise

# obj_prompt = ["A Cat"]
# bg_prompt = ["A Dog"]
obj_prompt = ["a sailboat"]
bg_prompt = ["cloudy blue sky"]
uncond_prompt = [""]
batch_size = 2

# 2. Get your text embeddings
obj_embeddings = get_text_embedding(obj_prompt * batch_size)
bg_embeddings = get_text_embedding(bg_prompt * batch_size)
uncond_embeddings = get_text_embedding(uncond_prompt * batch_size)

height, width = 512, 512
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
latents = latents * scheduler.init_noise_sigma

# 4. Call the new dynamic sampler!
# The embeddings list must contain unconditional first.
embeddings = [uncond_embeddings, obj_embeddings, bg_embeddings]

composed_latents = sampler_dynamic_weights(
    unet,
    scheduler,
    latents,
    embeddings,
    guidance_scale=7.5,
    num_inference_steps=100
)

# 5. Decode and save the final image
get_image(
    composed_latents,
    nrow=1,
    ncol=batch_size,
    save_path="outputs/cat_and_dog_dynamic.png",
    verbose=True
)

print("✅ Composition using dynamic weights is complete!")
