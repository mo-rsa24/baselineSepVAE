# run/ldm.py
import time
import argparse
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.serialization import from_bytes, to_bytes
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision.utils import save_image
import torch
from typing import Any
from datasets.ChestXRay import ChestXrayDataset
from datasets.Latents import PreencodedLatentDataset
from diffusion.sampling import DDPM_ancestral_sampler
from diffusion.vp_equation import alpha_fn, marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn
from models.ae_kl import AutoencoderKL
from models.cxr_unet import ScoreNet

# W&B is optional
try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ── Pretty + parseable step blocks ─────────────────────────────────────────────
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    _RICH = True
    _console = Console(log_time=False, log_path=False)
except Exception:
    _RICH = False
    _console = None

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def open_block(kind: str, step: int = None, epoch: int = None, note: str = ""):
    tag = f"{kind.upper()} | step={step} epoch={epoch} time={_now()}"
    # machine-parsable sentinels (easy to grep):
    print(f"\n<<<{kind.upper()}_BEGIN step={step} epoch={epoch}>>>", flush=True)
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
        if note:
            _console.print(note, style="dim")
    else:
        print("=" * 100)
        print(tag)
        if note: print(note)
        print("-" * 100)
    # keep stdout unbuffered on clusters: export PYTHONUNBUFFERED=1

def close_block(kind: str, step: int = None):
    tag = f"END {kind.upper()} | step={step} time={_now()}"
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
    else:
        print("-" * 100)
    print(f"<<<{kind.upper()}_END step={step}>>>", flush=True)

def pretty_table(title: str, metrics: dict):
    if _RICH:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        for k, v in metrics.items():
            table.add_row(k, f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v))
        _console.print(table)
    else:
        keys = list(metrics.keys())
        w = max(len(k) for k in keys) if keys else 10
        print(f"{title}")
        for k in keys:
            v = metrics[k]
            v = f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v)
            print(f"{k:<{w}} : {v}")

def log_sample_diversity(samples_np: np.ndarray, step: int, epoch: int):
    """Calculates and logs pairwise MSE to check for sample collapse."""
    if samples_np.ndim != 4:
        raise ValueError(f"Expected samples_np to be 4D (B, H, W, C), but got {samples_np.shape}")
    if samples_np.shape[0] <= 1:
        print("[diversity] Batch size is 1, skipping diversity check.")
        return

    batch_size = samples_np.shape[0]
    # Flatten each image into a vector
    flattened_samples = samples_np.reshape(batch_size, -1)

    # Calculate pairwise Mean Squared Error (MSE)
    # Using broadcasting for an efficient calculation: (a - b)^2 = a^2 - 2ab + b^2
    sum_sq = np.sum(flattened_samples**2, axis=1, keepdims=True)
    dot_prod = flattened_samples @ flattened_samples.T
    # mse[i, j] = mean((img[i] - img[j])^2)
    mse_matrix = (sum_sq + sum_sq.T - 2 * dot_prod) / flattened_samples.shape[1]

    # We only need the upper triangle of the matrix (excluding the diagonal)
    # as the matrix is symmetric and mse(i, i) is 0.
    indices = np.triu_indices(batch_size, k=1)
    pairwise_mse_vals = mse_matrix[indices]

    metrics = {
        "pairwise_mse_mean": float(np.mean(pairwise_mse_vals)),
        "pairwise_mse_std": float(np.std(pairwise_mse_vals)),
        "pairwise_mse_min": float(np.min(pairwise_mse_vals)),
        "pairwise_mse_max": float(np.max(pairwise_mse_vals)),
    }

    open_block("diversity", step=step, epoch=epoch, note="Pairwise MSE between generated samples in the batch")
    pretty_table("sample_diversity/metrics", metrics)
    close_block("diversity", step=step)
    if _WANDB and wandb.run is not None:
        wandb.log({"sample_diversity/metrics": metrics, "epoch": epoch, "step": step})

class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def parse_args():
    p = argparse.ArgumentParser("JAX Latent Diffusion Model (CXR) Trainer")
    # --- Data & Debugging ---
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB", "PNEUMONIA"], default="TB")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int, default=1,
                   help="Optional: keep a class index only (e.g., 1 for disease, 0 for normal)")
    p.add_argument("--overfit_one", action="store_true", help="Repeat a single sample to overfit.")
    # p.add_argument(
    #     "--overfit_one",
    #     type=bool,
    #     default=False,
    #     help="Overfit on a single batch of data."
    # )
    p.add_argument("--overfit_k", type=int, default=0, help="If >0, train on a fixed tiny subset of size K.")
    p.add_argument("--repeat_len", type=int, default=500,
                   help="Virtual length for the repeated one-sample dataset.")

    # --- Pretrained Autoencoder ---
    p.add_argument("--ae_ckpt_path", default=None,
                   help="Path to the last.flax of the pretrained autoencoder. "
                        "Optional when --preencoded_latents_dir is set.")
    p.add_argument("--ae_config_path", default=None,
                   help="Path to the run_meta.json of the AE run. "
                        "Optional when --preencoded_latents_dir is set.")
    p.add_argument("--latent_scale_factor", type=float, default=1.0, help="From stable-diffusion v1.")
    p.add_argument(
        "--preencoded_latents_dir",
        type=str,
        default=None,
        help="Path to directory containing pre-encoded latent .npy files and manifest.jsonl.",
    )
    p.add_argument(
        "--preencoded_manifest",
        type=str,
        default="manifest.jsonl",
        help="Manifest filename inside preencoded_latents_dir (jsonl).",
    )

    # --- LDM UNet Architecture ---
    p.add_argument("--ldm_ch_mults", type=str, default="1,2,4", help="Channel multipliers for UNet, relative to base_ch.")
    p.add_argument("--ldm_base_ch", type=int, default=128)
    p.add_argument("--ldm_num_res_blocks", type=int, default=2)
    p.add_argument("--ldm_attn_res", type=str, default="16", help="Resolutions for attention blocks, e.g., '16,8'")
    p.add_argument("--ldm_z_channels", type=int, default=None,
                   help="Explicitly set LDM channels (e.g., 1) to train on a bottleneck subset of the VAE latent.")
    p.add_argument("--latent_size", type=int, default=None,
                   help="Override latent spatial size (e.g. 64 for SepVAE). "
                        "If not set, inferred from AE config and --img_size.")
    p.add_argument("--vae_z_channels", type=int, default=None,
                   help="Override VAE z_channels. Required when --ae_config_path is not provided.")
    # --- Optimizer ---
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # --- Logging & Checkpoints ---
    p.add_argument("--output_root", default="runs_ldm")
    p.add_argument("--exp_name", default="cxr_ldm")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_dir", default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--sample_batch_size", type=int, default=16)
    p.add_argument("--use_bfloat16", action="store_true", help="Enable bfloat16 compute to reduce activation memory.")
    p.add_argument("--use_remat", action="store_true", help="Enable rematerialization for memory savings.")
    # 2. Add EMA command-line arguments
    p.add_argument("--use_ema", action="store_true", help="Enable EMA for model parameters.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="Decay rate for EMA.")
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    p.add_argument("--wandb_project", default="cxr-ldm")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--git_hash", default=None)
    p.add_argument("--git_branch", default=None)
    p.add_argument("--git_parent", default=None)
    return p.parse_args()


def load_autoencoder(config_path, ckpt_path):
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # Parse ch_mults relative to base_ch if needed
    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        base_ch = ae_args.get('base_ch', 64)
        ae_ch_mults = tuple(base_ch * m for m in ch_mult_factors)
    else:
        ae_ch_mults = tuple(ae_args['ch_mults'])

    attn_res = tuple(int(r) for r in ae_args.get('attn_res', '16').split(',') if r)

    enc_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args['z_channels'],
                   double_z=True, attn_resolutions=attn_res, in_ch=1)
    dec_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1,
                   attn_resolutions=attn_res)

    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=ae_args['embed_dim'])

    # Init variables
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    # Optimizer scaffold to load checkpoint
    def get_ae_tx(lr, grad_clip, weight_decay):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    tx = get_ae_tx(lr=ae_args.get('lr', 1e-4), grad_clip=ae_args.get('grad_clip', 1.0),
                   weight_decay=ae_args.get('weight_decay', 1e-4))

    gen_params = {'ae': ae_variables['params']}
    from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig
    loss_cfg = LPIPSGANConfig(disc_num_layers=ae_args.get('disc_layers', 3))
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg)
    loss_params_dummy = loss_mod.init({'params': rng}, x_in=fake_img, x_rec=fake_img, posterior=None, step=jnp.array(0))['params']
    disc_params_dummy = {'loss': loss_params_dummy}

    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    print("Autoencoder loaded successfully.")
    return ae_model, restored_gen_state.params['ae']

def load_autoencoder_(config_path, ckpt_path):
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # Parse ch_mults relative to base_ch if needed
    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        base_ch = ae_args.get('base_ch', 64)
        ae_ch_mults = tuple(base_ch * m for m in ch_mult_factors)
    else:
        ae_ch_mults = tuple(ae_args['ch_mults'])

    attn_res = tuple(int(r) for r in ae_args.get('attn_res', '16').split(',') if r)

    enc_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args['z_channels'],
                   double_z=True, attn_resolutions=attn_res, in_ch=1)
    dec_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1,
                   attn_resolutions=attn_res)

    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=ae_args['embed_dim'])

    # Init variables
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    # Optimizer scaffold to load checkpoint
    def get_ae_tx(ae_args, lr):
        base_tx = optax.chain(
            optax.clip_by_global_norm(ae_args.get("grad_clip", 1.0)) if ae_args.get("grad_clip", 1.0) > 0 else optax.identity(),
            optax.adamw(
                lr,
                weight_decay=ae_args.get("weight_decay", 1e-4),
                b1=ae_args.get("adam_beta1", 0.9),
                b2=ae_args.get("adam_beta2", 0.999),
                eps=ae_args.get("adam_eps", 1e-8),
            ),
        )
        m = ae_args.get("max_consecutive_nan_updates", 0)
        if m and m > 0:
            return optax.apply_if_finite(base_tx, max_consecutive_errors=m)
        return base_tx


    tx = get_ae_tx(ae_args, lr=ae_args.get("lr", 1e-4))
    gen_params = {'ae': ae_variables['params']}
    from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig
    loss_cfg = LPIPSGANConfig(disc_num_layers=ae_args.get('disc_layers', 3))
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg)
    loss_params_dummy = loss_mod.init({'params': rng}, x_in=fake_img, x_rec=fake_img, posterior=None, step=jnp.array(0))['params']
    disc_params_dummy = {'loss': loss_params_dummy}

    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    print("Autoencoder loaded successfully.")
    return ae_model, restored_gen_state.params['ae']


def main():
    args = parse_args()
    print(f"[config] latent_scale_factor = {args.latent_scale_factor}")
    if args.use_ema:
        print(f"✅ EMA is enabled with a decay rate of {args.ema_decay}")
    else:
        print("❌ EMA is disabled for this run.")
    if args.use_bfloat16:
        print("✅ bfloat16 compute enabled for LDM activations.")
    if args.use_remat:
        print("✅ Rematerialization enabled for LDM blocks.")
    rng = jax.random.PRNGKey(args.seed)

    # --- Setup Directories ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, args.run_name or f"{args.exp_name}-{ts}")
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")
    with open(os.path.join(run_dir, "ldm_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[{datetime.now()}] 🔍 Starting Dataset Initialization...", flush=True)  # ADD THIS
    # --- Setup Dataset ---
    base_ds = ChestXrayDataset(root_dir=args.data_root, task=args.task, split=args.split, img_size=args.img_size,
                               class_filter=args.class_filter)
    if args.preencoded_latents_dir:
        base_ds = PreencodedLatentDataset(
            args.preencoded_latents_dir,
            manifest_name=args.preencoded_manifest,
        )
    else:
        base_ds = ChestXrayDataset(
            root_dir=args.data_root,
            task=args.task,
            split=args.split,
            img_size=args.img_size,
            class_filter=args.class_filter,
        )
    print(f"[{datetime.now()}] ✅ Dataset Initialized. Size: {len(base_ds)}", flush=True)  # ADD THIS
    batch_size = args.batch_per_device * jax.local_device_count()
    if args.overfit_one:
        ds = Subset(base_ds, [0])
        ds = torch.utils.data.ConcatDataset([ds] * args.repeat_len)
    elif args.overfit_k > 0:
        ds = Subset(base_ds, list(range(min(args.overfit_k, len(base_ds)))))
    else:
        ds = base_ds

    if args.overfit_one:
        print("INFO: Overfitting on one sample. Disabling data loader workers and shuffle.")
        loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    else:
        loader_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    loader = DataLoader(ds, **loader_kwargs)

    # --- Load Pretrained Autoencoder (optional when using pre-encoded latents) ---
    if args.ae_ckpt_path and args.ae_config_path:
        ae_model, ae_params = load_autoencoder_(args.ae_config_path, args.ae_ckpt_path)
        ae_params = jax.device_put_replicated(ae_params, jax.local_devices())
    else:
        if not args.preencoded_latents_dir:
            raise ValueError(
                "--ae_ckpt_path and --ae_config_path are required unless "
                "--preencoded_latents_dir is set."
            )
        print("[info] No AE checkpoint provided — pre-encoded latents mode. "
              "Sampling visualisations will be skipped.")
        ae_model = None
        ae_params = jax.device_put_replicated(jnp.zeros((1,)), jax.local_devices())

    # --- Setup LDM UNet ---
    if args.ae_config_path:
        with open(args.ae_config_path, 'r') as f:
            ae_args = json.load(f)
        vae_z_channels = ae_args['z_channels']
        if isinstance(ae_args['ch_mults'], str):
            num_downsamples = len(ae_args['ch_mults'].split(',')) - 1
        else:
            num_downsamples = len(ae_args['ch_mults']) - 1
        downsample_factor = 2 ** num_downsamples
        _latent_size_from_config = args.img_size // downsample_factor
        print("--- Shape & Channel Verification ---")
        print(f"AE z_channels: {vae_z_channels}")
        print(f"AE ch_mults: {ae_args['ch_mults']}")
        print(f"Downsample factor: {downsample_factor}")
        print(f"Latent spatial size (from config): {_latent_size_from_config}x{_latent_size_from_config}")
    else:
        if args.vae_z_channels is None:
            raise ValueError(
                "--vae_z_channels must be set when --ae_config_path is not provided."
            )
        vae_z_channels = args.vae_z_channels
        _latent_size_from_config = None

    latent_size = args.latent_size if args.latent_size is not None else _latent_size_from_config
    if latent_size is None:
        raise ValueError(
            "Cannot determine latent_size. Provide --latent_size or --ae_config_path."
        )

    ldm_z_channels = args.ldm_z_channels if args.ldm_z_channels is not None else vae_z_channels

    if ldm_z_channels < vae_z_channels:
        print(f"⚠️  Training with CHANNEL BOTTLENECK: VAE({vae_z_channels}) -> LDM({ldm_z_channels})")

    print(f"latent_size={latent_size}x{latent_size}  ldm_z_channels={ldm_z_channels}")

    ldm_chans = tuple(args.ldm_base_ch * int(m) for m in args.ldm_ch_mults.split(','))
    attn_res = tuple(int(r) for r in args.ldm_attn_res.split(','))
    compute_dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32

    ldm_model = ScoreNet(
        z_channels=ldm_z_channels,
        channels=ldm_chans,
        num_res_blocks=args.ldm_num_res_blocks,
        attn_resolutions=attn_res,
        use_remat=args.use_remat,
        dtype=compute_dtype,
        param_dtype=jnp.float32,
    )
    rng, init_rng = jax.random.split(rng)
    fake_latent = jnp.ones((1, latent_size, latent_size, ldm_z_channels))
    fake_time = jnp.ones((1,))
    ldm_params = ldm_model.init(init_rng, fake_latent, fake_time)['params']

    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr, weight_decay=args.weight_decay))
    ema_params = ldm_params if args.use_ema else None
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply, params=ldm_params, ema_params=ema_params, tx=tx
    )

    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        try:
            with tf.io.gfile.GFile(ckpt_latest, "rb") as f:
                blob = f.read()
            ldm_state = from_bytes(ldm_state, blob)
            print(f"[info] Resumed LDM from {ckpt_latest}")
        except Exception as e:
            print(f"[warning] Failed to resume: {e}. Starting from scratch.")

    ldm_state = jax.device_put_replicated(ldm_state, jax.local_devices())

    use_wandb = bool(args.wandb and _WANDB)
    if use_wandb:
        wandb_config = {**vars(args), "vae_z_channels": vae_z_channels, "ldm_z_channels": ldm_z_channels}
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name or args.run_name or f"{args.exp_name}-{ts}",
            config=wandb_config,
            tags=[t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else None
        )

    precomputed_z0 = None
    if args.overfit_one:
        if args.preencoded_latents_dir:
            z0 = jnp.asarray(base_ds[0][0].numpy())[None, ...]
        else:
            one_loader = DataLoader(Subset(base_ds, [0]), batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            (x0, _), = list(one_loader)
            x0 = jnp.asarray(x0.numpy()).transpose(0, 2, 3, 1)
            x0 = (x0 + 1.0) / 2.0
            unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))
            posterior0 = ae_model.apply({'params': unrep_ae_params}, x0, method=ae_model.encode, train=False)
            rng_z0 = jax.random.PRNGKey(42)
            z0 = posterior0.sample(rng_z0)

        global_bs = args.batch_per_device * jax.local_device_count()
        z0_tiled = jnp.tile(z0, (global_bs, 1, 1, 1))
        precomputed_z0 = z0_tiled.reshape((jax.local_device_count(), -1) + z0.shape[1:])

    # --- Define Training Step ---
    def train_step(rng, ldm_state, ae_params, z_batch, precomputed_z0):
        """One pmap-ed training step."""
        # z_batch: precomputed latents or latents encoded from images.
        rng, rng_diff = jax.random.split(rng, 2)

        def loss_fn(ldm_params):
            if precomputed_z0 is not None:
                z = precomputed_z0 * args.latent_scale_factor
            elif args.preencoded_latents_dir:
                z = z_batch
            else:
                posterior = ae_model.apply({'params': ae_params}, z_batch, method=ae_model.encode, train=False)
                z = posterior.sample(rng) * args.latent_scale_factor
            if z.shape[-1] > ldm_z_channels:
                z = z[..., :ldm_z_channels]
            z = z.astype(compute_dtype)
            rng_t, rng_noise = jax.random.split(rng_diff, 2)
            t = jax.random.uniform(rng_t, (z.shape[0],), minval=1e-5, maxval=1.0)
            noise = jax.random.normal(rng_noise, z.shape).astype(compute_dtype)

            # VP forward perturbation
            sigma = marginal_prob_std_fn(t).astype(compute_dtype)  # σ(t)  [B]
            alpha = alpha_fn(t).astype(compute_dtype)  # α(t)  [B]
            sigma_b = sigma[:, None, None, None]
            alpha_b = alpha[:, None, None, None]
            x_t = alpha_b * z + sigma_b * noise  # x_t = α z + σ ε

            # Predict ε and compute simple ε-MSE
            eps_hat = ldm_model.apply({'params': ldm_params}, x_t, t)  # [B,H,W,C]
            loss = jnp.mean((eps_hat.astype(jnp.float32) - noise.astype(jnp.float32)) ** 2)
            def _cos(a, b, eps=1e-8):
                num = jnp.sum(a * b, axis=tuple(range(1, a.ndim)))
                den = jnp.linalg.norm(a.reshape(a.shape[0], -1), axis=-1) * \
                      jnp.linalg.norm(b.reshape(b.shape[0], -1), axis=-1) + eps
                return num / den

            aux = dict(
                t_mean=jnp.mean(t),
                sigma_mean=jnp.mean(sigma),
                alpha_mean=jnp.mean(alpha),
                cos_eps=jnp.mean(_cos(eps_hat, noise)),
                z_mean=jnp.mean(z), z_std=jnp.std(z),
                z_min=jnp.min(z), z_max=jnp.max(z),
                xt_mean=jnp.mean(x_t), xt_std=jnp.std(x_t),
                eps_hat_mean=jnp.mean(eps_hat), eps_hat_std=jnp.std(eps_hat),
            )
            return loss, aux
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ldm_state.params)
        grad_norm = optax.global_norm(grads)
        aux = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
        aux = {**aux, "grad_norm": jax.lax.pmean(grad_norm, axis_name='device')}
        loss = jax.lax.pmean(loss, axis_name='device')
        new_ldm_state = ldm_state.apply_gradients(grads=grads)
        if args.use_ema:
            new_ema_params = optax.incremental_update(new_ldm_state.params, ldm_state.ema_params, args.ema_decay)
            new_ldm_state = new_ldm_state.replace(ema_params=new_ema_params)

        return new_ldm_state, loss, aux

    pmapped_train_step = jax.pmap(train_step, axis_name='device')


    # --- Training Loop ---
    global_step = int(ldm_state.step[0])
    for ep in range(args.epochs):
        progress_bar = tqdm(loader, desc=f"Epoch {ep + 1}/{args.epochs}", leave=False)
        for batch in progress_bar:
            x, _ = batch
            if args.preencoded_latents_dir:
                z = jnp.asarray(x.numpy())
            else:
                z = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
                z = (z + 1.0) / 2.0
            z_sharded = z.reshape((jax.local_device_count(), -1) + z.shape[1:])
            rng, step_rng = jax.random.split(rng)
            rng_sharded = jax.random.split(step_rng, jax.local_device_count())
            ldm_state, loss, aux = pmapped_train_step(rng_sharded, ldm_state, ae_params, z_sharded, precomputed_z0)

            if global_step % args.log_every == 0:
                loss_val = float(np.asarray(loss[0]))
                aux_host = jax.tree_map(lambda x: float(np.asarray(x[0])), aux)  # take device 0
                metrics = {
                    "step": global_step,
                    "loss": loss_val,
                    "grad_norm": aux_host["grad_norm"],
                    "t.mean": aux_host["t_mean"],
                    "sigma.mean": aux_host["sigma_mean"],
                    "alpha.mean": aux_host["alpha_mean"],
                    "cos_eps.mean": aux_host["cos_eps"],
                    "z.mean": aux_host["z_mean"], "z.std": aux_host["z_std"],
                    "x_t.mean": aux_host["xt_mean"], "x_t.std": aux_host["xt_std"],
                    "eps_hat.mean": aux_host["eps_hat_mean"], "eps_hat.std": aux_host["eps_hat_std"],
                }
                open_block("train", step=global_step, epoch=ep + 1, note="per-device means (pmean)")
                pretty_table("train/metrics", metrics)
                close_block("train", step=global_step)
                if use_wandb:
                    wandb.log({"train/loss": loss_val, "train/step": global_step})
            global_step += 1

        if ae_model is None and (ep + 1) % args.sample_every == 0:
            print(f"[info] Skipping sampling at epoch {ep + 1} (no AE loaded, pre-encoded latents mode).")

        if ae_model is not None and (ep + 1) % args.sample_every == 0:
            print(f"Sampling at epoch {ep + 1}...")
            print(f"[sampling] using z_std = {1.0 / args.latent_scale_factor}")
            rng, sample_rng = jax.random.split(rng)
            unrep_ldm_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.params))
            unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))
            if args.use_ema and ldm_state.ema_params is not None:
                print("[info] Using EMA parameters for sampling.")
                sampling_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.ema_params))
            else:
                sampling_params = unrep_ldm_params


            if args.use_ema:
                open_block("ema", step=global_step, epoch=ep + 1, note="Validate EMA Impact")
                flat_params = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(unrep_ldm_params)])
                flat_ema_params = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(sampling_params)])
                cosine_sim = jnp.dot(flat_params, flat_ema_params) / (
                            jnp.linalg.norm(flat_params) * jnp.linalg.norm(flat_ema_params))
                print(f"[info] Cosine similarity between base and EMA weights: {cosine_sim:.6f}")
                close_block("ema", step=global_step)

                if use_wandb:
                    wandb.log({"train/cosine_similarity": float(cosine_sim), "epoch": ep + 1})

            open_block("sample", step=global_step, epoch=ep + 1, note="Euler-Maruyama SDE Sampler")
            sample_rng = jax.random.fold_in(rng, ep + 1)
            samples_grid, final_latent, _ = DDPM_ancestral_sampler(
                rng=sample_rng,
                ldm_model=ldm_model,
                ldm_params=sampling_params,
                ae_model=ae_model,
                ae_params=unrep_ae_params,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
                alpha_bar_fn=alpha_bar_fn,
                latent_size=latent_size,
                batch_size=args.sample_batch_size,
                z_channels=ldm_z_channels,
                vae_z_channels=vae_z_channels,
                z_std=1.0 / args.latent_scale_factor
            )
            final_latent_np = np.asarray(final_latent)
            log_sample_diversity(final_latent_np, step=global_step, epoch=ep + 1)
            stats = {
                "mean": np.mean(final_latent_np),
                "std": np.std(final_latent_np),
                "min": np.min(final_latent_np),
                "max": np.max(final_latent_np),
            }
            pretty_table("final_latent_stats", stats)
            out_path = os.path.join(samples_dir, f"sample_ep{ep + 1:04d}.png")
            save_image(samples_grid, out_path)
            close_block("sample", step=global_step)
            if use_wandb:
                wandb.log({"samples": wandb.Image(out_path), "epoch": ep + 1})

        # Save checkpoint
        unrep_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state))
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(to_bytes(unrep_state))

    if use_wandb:
        wandb.finish()
    print(f"Training complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
