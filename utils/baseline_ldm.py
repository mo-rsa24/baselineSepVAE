import json
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.serialization import msgpack_restore
from torchvision.utils import make_grid

from models.conditional_cxr_unet import ConditionalScoreNet
from models.sep_vae_v2 import SepVAEV2


def load_sepvae_v2_checkpoint(ckpt_path: str, use_ema: bool = True):
    ckpt_path = Path(ckpt_path)
    with ckpt_path.open("rb") as handle:
        ckpt = msgpack_restore(handle.read())

    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("model_version", "v2") != "v2":
        raise ValueError(f"Checkpoint is not a V2 SepVAE run: {ckpt_path}")

    model = SepVAEV2(
        z_channels_common=int(ckpt_args.get("z_channels_common", 16)),
        z_channels_disease=int(ckpt_args.get("z_channels_disease", 16)),
        query_dim=int(ckpt_args.get("attn_query_dim", 256)),
        attn_heads=int(ckpt_args.get("attn_heads", 4)),
        use_bbox_cross_attn=bool(ckpt_args.get("use_bbox_cross_attn", False)),
    )

    params_key = "ema_params" if use_ema and ckpt.get("ema_params") is not None else "vae_params"
    variables = {"params": jax.tree_util.tree_map(jnp.array, ckpt[params_key])}
    batch_stats = ckpt.get("vae_batch_stats")
    if batch_stats:
        variables["batch_stats"] = jax.tree_util.tree_map(jnp.array, batch_stats)

    return model, variables, ckpt, ckpt_args


def decode_sepvae_latents(sepvae_model, sepvae_variables, z_common, z_cardio):
    z_concat = jnp.concatenate([z_common, z_cardio], axis=-1)
    return sepvae_model.apply(sepvae_variables, z_concat, method=sepvae_model.decode)


def save_image_grid(images_np: np.ndarray, path: str, nrow: int = None):
    images_np = np.clip(images_np, 0.0, 1.0).astype(np.float32)
    if images_np.ndim != 4:
        raise ValueError(f"Expected images_np with shape (B, H, W, C), got {images_np.shape}")

    chw = np.transpose(images_np, (0, 3, 1, 2))
    tensor = torch.from_numpy(chw)
    if nrow is None:
        nrow = max(1, int(np.sqrt(images_np.shape[0])))
    grid = make_grid(tensor, nrow=nrow, padding=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if grid_np.shape[-1] == 1:
        grid_np = grid_np[..., 0]
    Image.fromarray(grid_np).save(str(path))


def load_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_latent_meta(root_dir: str) -> Dict:
    return load_json_if_exists(Path(root_dir) / "latent_meta.json")


def build_conditional_ldm_from_args(args: Dict) -> ConditionalScoreNet:
    channels = tuple(int(args["ldm_base_ch"]) * int(mult) for mult in str(args["ldm_ch_mults"]).split(","))
    attn_resolutions = tuple(int(res) for res in str(args["ldm_attn_res"]).split(","))
    return ConditionalScoreNet(
        z_channels=int(args["z_channels"]),
        cond_channels=int(args["cond_channels"]),
        channels=channels,
        num_res_blocks=int(args["ldm_num_res_blocks"]),
        attn_resolutions=attn_resolutions,
        use_remat=bool(args.get("use_remat", False)),
        dtype=jnp.bfloat16 if bool(args.get("use_bfloat16", False)) else jnp.float32,
        param_dtype=jnp.float32,
    )


def load_baseline_ldm_checkpoint(ckpt_path: str, use_ema: bool = True) -> Tuple[ConditionalScoreNet, Dict, Dict]:
    ckpt_path = Path(ckpt_path)
    with ckpt_path.open("rb") as handle:
        ckpt = msgpack_restore(handle.read())

    args = ckpt.get("args", {})
    model = build_conditional_ldm_from_args(args)
    params_key = "ema_params" if use_ema and ckpt.get("ema_params") is not None else "ldm_params"
    params = jax.tree_util.tree_map(jnp.array, ckpt[params_key])
    return model, params, ckpt
