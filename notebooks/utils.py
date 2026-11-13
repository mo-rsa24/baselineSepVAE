import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
import matplotlib.pyplot as plt

def get_sd_models(model_id: str = "Manojb/stable-diffusion-2-1-base", dtype=torch.float16, device=torch.device("cuda")):
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True).to(device)
    return {"vae": vae, "tokenizer": tokenizer, "text_encoder": text_encoder, "unet": unet}


@torch.no_grad()
def get_image(vae, latents, nrow, ncol):
    # Ensure latents are same dtype as VAE
    latents = latents.to(dtype=vae.dtype)

    # Ensure latents have batch dimension
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # Don't squeeze! Keep batch dimension for permute
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)

    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i * nrow + col_i
            if i < len(image):
                row.append(image[i])
            else:
                row.append(torch.zeros_like(image[0]))
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

# et_vel(unet, t, sigma, latents, embeddings, eps=None, get_div=False, device=torch.device("cuda"), dtype=torch.float16)
@torch.no_grad()
def get_text_embedding(prompt, tokenizer, text_encoder, device=torch.device("cuda")):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(device))[0]

def plot_trajectories(ll_obj, ll_bg, kappa):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot((ll_obj - ll_bg).cpu().numpy(), c='royalblue')
    plt.ylabel('logp_obj - logp_bg')
    plt.xlabel('num iterations')
    plt.grid()
    plt.subplot(122)
    plt.plot(kappa.cpu().numpy(), c='royalblue')
    plt.ylabel('kappa')
    plt.xlabel('num iterations')
    plt.grid()