import matplotlib.pyplot as plt
from PIL import Image
import torch
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
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True,
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    use_safetensors=True,
    torch_dtype=torch.float16
)

torch_device = torch.device('cuda')
model_dtype = torch.float16
vae.to(torch_device, dtype=model_dtype)
text_encoder.to(torch_device, dtype=model_dtype)
unet.to(torch_device, dtype=model_dtype)

from diffusers import EulerDiscreteScheduler

scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

@torch.no_grad
def get_image(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i*nrow + col_i
            row.append(image[i])
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

@torch.no_grad
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(torch_device))[0]

from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.nn.attention import SDPBackend, sdpa_kernel


@torch.no_grad
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    embeds = torch.cat(embeddings)

    if get_div:
        original_dtype = unet.dtype
        unet.to(torch.float32)
        v_float32 = lambda _x, _e: unet(
            _x / ((sigma.float() ** 2 + 1) ** 0.5),
            t.float(),
            encoder_hidden_states=_e
        ).sample
        with sdpa_kernel(SDPBackend.MATH):
            vel_float32, div_float32 = torch.func.jvp(
                v_float32,
                (latents.float(), embeds.float()),
                (eps.float(), torch.zeros_like(embeds).float())
            )
        vel = vel_float32.to(original_dtype)
        div = -(eps * div_float32.to(original_dtype)).sum((1, 2, 3))
        unet.to(original_dtype)
    else:
        v = lambda _x, _e: unet(_x / ((sigma ** 2 + 1) ** 0.5), t, encoder_hidden_states=_e).sample
        vel = v(latents, embeds)
        div = torch.zeros([len(embeds)], device=torch_device, dtype=latents.dtype)

    return vel, div


obj_prompt = ["a sailboat"]
bg_prompt = ["cloudy blue sky"]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 100  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.cuda.manual_seed(1)  # Seed generator to create the initial latent noise
# batch_size = len(obj_prompt)
batch_size = 6

obj_embeddings = get_text_embedding(obj_prompt * batch_size).to(torch.float16)
bg_embeddings = get_text_embedding(bg_prompt * batch_size).to(torch.float16)
uncond_embeddings = get_text_embedding([""] * batch_size).to(torch.float16)
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
latents = latents * scheduler.init_noise_sigma

latents = latents.to(torch.float16)

lift = 0.0

scheduler.set_timesteps(num_inference_steps)
ll_obj = torch.zeros((num_inference_steps + 1, batch_size), device=torch_device)
ll_bg = torch.zeros((num_inference_steps + 1, batch_size), device=torch_device)
kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=torch_device)
for i, t in enumerate(scheduler.timesteps):
    t = t.to(torch.float16)
    dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
    sigma = scheduler.sigmas[i].to(torch.float16)
    eps = torch.randint_like(latents, 2, dtype=latents.dtype) * 2 - 1
    eps = eps.to(torch.float16)
    vel_obj, dlog_obj = get_vel(t, sigma, latents, [obj_embeddings], eps, True)
    vel_bg, dlog_bg = get_vel(t, sigma, latents, [bg_embeddings], eps, True)
    vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings], eps, False)

    kappa[i + 1] = sigma * (dlog_obj - dlog_bg) + ((vel_obj - vel_bg) * (vel_obj + vel_bg)).sum(
        (1, 2, 3)) + lift / dsigma * sigma / num_inference_steps
    kappa[i + 1] += -((vel_obj - vel_bg) * (vel_uncond + guidance_scale * (vel_bg - vel_uncond))).sum((1, 2, 3))
    kappa[i + 1] /= guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

    vf = vel_uncond + guidance_scale * ((vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
    latents += (dsigma * vf).to(latents.dtype)
    ll_obj[i + 1] = ll_obj[i] + dsigma * (dlog_obj - ((-vel_obj / sigma) * (vel_obj - vf)).sum((1, 2, 3)))
    ll_bg[i + 1] = ll_bg[i] + dsigma * (dlog_bg - ((-vel_bg / sigma) * (vel_bg - vf)).sum((1, 2, 3)))

get_image(latents, 1, 6)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot((ll_obj-ll_bg).cpu().numpy(), c='royalblue')
plt.ylabel('logp_obj - logp_bg')
plt.xlabel('num iterations')
plt.grid()
plt.subplot(122)
plt.plot(kappa.cpu().numpy(), c='royalblue')
plt.ylabel('kappa')
plt.xlabel('num iterations')
plt.grid()