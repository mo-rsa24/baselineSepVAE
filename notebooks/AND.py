import os
import torch
from operator import itemgetter
from diffusers import EulerDiscreteScheduler

from dynamics import get_latents, stochastic_super_diff_and
from notebooks.utils import get_sd_models, get_image, plot_trajectories

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Enable FP16
dtype = torch.float16
device = torch.device("cuda")

height, width = 512, 512
latent_height, latent_width = 64, 64
batch_size = 4
steps = 500
lift = -10

# 2. Load Models
models = get_sd_models(dtype=dtype, device=device)
vae, tokenizer, text_encoder, unet = itemgetter(
    "vae", "tokenizer", "text_encoder", "unet"
)(models)

scheduler = EulerDiscreteScheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")

obj_prompt = ["A Dog On The Left"]
bg_prompt = ["A Cat On The Right"]

# 3. Get Latents
latents = get_latents(
    scheduler,
    device=device,
    dtype=dtype,
    num_inference_steps=steps,
    batch_size=batch_size,
    latent_height=latent_height,
    latent_width=latent_width
)

latents, kappa, ll_obj, ll_bg = stochastic_super_diff_and(
    latents,
    obj_prompt,
    bg_prompt,
    scheduler,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    num_inference_steps=steps,
    batch_size=batch_size,
    lift=lift,
    device=device,
    dtype=dtype
)

# 5. Decode
img = get_image(vae, latents, nrow=2, ncol=2)
img.show()
img.save("superdiff_result_256.png")

plot_trajectories(ll_obj, ll_bg, kappa)