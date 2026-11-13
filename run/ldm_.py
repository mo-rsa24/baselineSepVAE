# run/ldm.py
import argparse
import os
import json
import math
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.serialization import from_bytes, to_bytes
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import torch

# Local imports
from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL
from models.cxr_unet import ScoreNet
from diffusion.equations import marginal_prob_std_fn, diffusion_coeff_fn
from diffusion.sampling import Euler_Maruyama_sampler

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
    p.add_argument("--overfit_k", type=int, default=0, help="If >0, train on a fixed tiny subset of size K.")
    p.add_argument("--repeat_len", type=int, default=500,
                   help="Virtual length for the repeated one-sample dataset.")

    # --- Pretrained Autoencoder ---
    p.add_argument("--ae_ckpt_path", required=True, help="Path to the last.flax of the pretrained autoencoder.")
    p.add_argument("--ae_config_path", required=True, help="Path to the run_meta.json of the AE run.")
    p.add_argument("--latent_scale_factor", type=float, default=1.0, help="From stable-diffusion v1.")

    # --- LDM UNet Architecture ---
    p.add_argument("--ldm_ch_mults", type=str, default="1,2,4", help="Channel multipliers for UNet, relative to base_ch.")
    p.add_argument("--ldm_base_ch", type=int, default=128)
    p.add_argument("--ldm_num_res_blocks", type=int, default=2)
    p.add_argument("--ldm_attn_res", type=str, default="16", help="Resolutions for attention blocks, e.g., '16,8'")

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

    # --- W&B ---
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="cxr-ldm")
    p.add_argument("--wandb_tags", default="")
    return p.parse_args()


def load_autoencoder(config_path, ckpt_path):
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # Correctly parse ch_mults by incorporating base_ch
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

    # Robust loading logic
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    # Recreate the optimizer structure from AE training to load the checkpoint
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
    loss_params_dummy = \
    loss_mod.init({'params': rng}, x_in=fake_img, x_rec=fake_img, posterior=None, step=jnp.array(0))['params']
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
    rng = jax.random.PRNGKey(args.seed)

    # --- Setup Directories ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, args.run_name or f"{args.exp_name}-{ts}")
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")
    with open(os.path.join(run_dir, "ldm_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Setup Dataset ---
    base_ds = ChestXrayDataset(root_dir=args.data_root, task=args.task, split=args.split, img_size=args.img_size,
                               class_filter=args.class_filter
                               )
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
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,  # Not needed for a single repeating item
            "num_workers": 0,  # CRITICAL: Avoids worker overhead
            "drop_last": True,
            "pin_memory": True
        }
    else:
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 8,
            "drop_last": True,
            "pin_memory": True
        }
    loader = DataLoader(ds, **loader_kwargs)

    # --- Load Pretrained Autoencoder ---
    ae_model, ae_params = load_autoencoder(args.ae_config_path, args.ae_ckpt_path)
    ae_params = jax.device_put_replicated(ae_params, jax.local_devices())

    # --- Setup LDM UNet ---
    with open(args.ae_config_path, 'r') as f: ae_args = json.load(f)
    z_channels = ae_args['z_channels']
    # --- ADD THIS BLOCK FOR DIAGNOSTICS ---
    print("--- Shape & Channel Verification ---")
    if isinstance(ae_args['ch_mults'], str):
        num_downsamples = len(ae_args['ch_mults'].split(',')) - 1
    else:
        num_downsamples = len(ae_args['ch_mults']) - 1
    downsample_factor = 2 ** num_downsamples
    latent_size = args.img_size // downsample_factor

    print(f"AE `z_channels`: {z_channels}")
    print(f"AE `ch_mults`: {ae_args['ch_mults']}")
    print(f"Calculated downsample factor: 2^{num_downsamples} = {downsample_factor}")
    print(f"Expected latent spatial size: {latent_size}x{latent_size}")
    # --- END DIAGNOSTIC BLOCK ---
    ldm_chans = tuple(args.ldm_base_ch * int(m) for m in args.ldm_ch_mults.split(','))
    attn_res = tuple(int(r) for r in args.ldm_attn_res.split(','))
    ldm_model = ScoreNet(z_channels=z_channels, channels=ldm_chans, num_res_blocks=args.ldm_num_res_blocks, attn_resolutions=attn_res)
    rng, init_rng = jax.random.split(rng)
    fake_latent = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_params = ldm_model.init(init_rng, fake_latent, fake_time)['params']

    # --- Setup TrainState ---
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr, weight_decay=args.weight_decay))
    ldm_state = TrainState.create(apply_fn=ldm_model.apply, params=ldm_params, tx=tx)
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        print(f"[info] Resuming LDM from {ckpt_latest}")
        with tf.io.gfile.GFile(ckpt_latest, "rb") as f: blob = f.read()
        ldm_state = from_bytes(ldm_state, blob)
    ldm_state = jax.device_put_replicated(ldm_state, jax.local_devices())

    # --- Setup W&B ---
    use_wandb = bool(args.wandb and _WANDB)
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name or f"{args.exp_name}-{ts}", config=args, tags=args.wandb_tags.split(','))

    # --- Define Training Step ---
    def train_step(rng, ldm_state, ae_params, x_batch):
        def loss_fn(ldm_params):
            rng_ae, rng_diff = jax.random.split(rng)
            jax.debug.print("x_batch_stats | min: {mn}, max: {mx}, mean: {m}",
                            mn=jnp.min(x_batch), mx=jnp.max(x_batch), m=jnp.mean(x_batch))
            posterior = ae_model.apply({'params': ae_params}, x_batch, method=ae_model.encode, train=False)
            z = posterior.sample(rng_ae) * args.latent_scale_factor
            # z = posterior.mode() * args.latent_scale_factor
            jax.debug.print("z_stats | mean: {m}, std: {d}, min: {mn}, max: {mx}",
                            m=jnp.mean(z), d=jnp.std(z), mn=jnp.min(z), mx=jnp.max(z))
            rng_t, rng_noise = jax.random.split(rng_diff) # Use the second key here
            t = jax.random.uniform(rng_t, (z.shape[0],), minval=1e-5, maxval=1.0)
            noise = jax.random.normal(rng_noise, z.shape)
            std = marginal_prob_std_fn(t)
            perturbed_z = z + noise * std[:, None, None, None]
            predicted_noise = ldm_model.apply({'params': ldm_params}, perturbed_z, t)
            jax.debug.print(
                "noise_stats | target_mean: {tm}, target_std: {ts} | pred_mean: {pm}, pred_std: {ps}",
                tm=jnp.mean(noise), ts=jnp.std(noise), pm=jnp.mean(predicted_noise), ps=jnp.std(predicted_noise))
            return jnp.mean((predicted_noise - noise) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(ldm_state.params)
        grad_norm = optax.global_norm(grads)
        jax.debug.print("train_step | loss: {l}, grad_norm: {g}", l=loss, g=grad_norm)
        grads = jax.lax.pmean(grads, axis_name='device')
        loss = jax.lax.pmean(loss, axis_name='device')
        new_ldm_state = ldm_state.apply_gradients(grads=grads)
        return new_ldm_state, loss

    pmapped_train_step = jax.pmap(train_step, axis_name='device')

    # --- Training Loop ---
    global_step = int(ldm_state.step[0])
    for ep in range(args.epochs):
        progress_bar = tqdm(loader, desc=f"Epoch {ep + 1}/{args.epochs}", leave=False)
        for batch in progress_bar:
            x, _ = batch
            x = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
            x = (x + 1.0) / 2.0
            x_sharded = x.reshape((jax.local_device_count(), -1) + x.shape[1:])
            rng, step_rng = jax.random.split(rng)
            rng_sharded = jax.random.split(step_rng, jax.local_device_count())
            ldm_state, loss = pmapped_train_step(rng_sharded, ldm_state, ae_params, x_sharded)
            if global_step % args.log_every == 0:
                loss_val = np.asarray(loss[0])
                progress_bar.set_postfix(loss=f"{loss_val:.4f}")
                unrep_ae_params_dbg = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))
                posterior_dbg = ae_model.apply({'params': unrep_ae_params_dbg}, x_sharded[0], method=ae_model.encode,
                                               train=False)
                z_dbg = posterior_dbg.sample(jax.random.PRNGKey(global_step)) * args.latent_scale_factor
                z_mean = float(jnp.mean(z_dbg))
                z_std = float(jnp.std(z_dbg))
                z_min = float(jnp.min(z_dbg))
                z_max = float(jnp.max(z_dbg))
                progress_bar.set_postfix_str(
                    f"loss={loss_val:.4f} | zμ={z_mean:.3f} zσ={z_std:.3f} [{z_min:.3f},{z_max:.3f}]")
                if use_wandb: wandb.log({"train/loss": loss_val, "train/step": global_step})
            global_step += 1

        # --- Sampling & Checkpointing ---
        if (ep + 1) % args.sample_every == 0:
            print(f"Sampling at epoch {ep + 1}...")
            rng, sample_rng = jax.random.split(rng)

            # Get a single replica of the model parameters for inference
            unrep_ldm_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.params))
            unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))

            # Call the sampler with the correct arguments
            samples_grid = Euler_Maruyama_sampler(
                rng=sample_rng,
                ldm_model=ldm_model,
                ldm_params=unrep_ldm_params,
                ae_model=ae_model,
                ae_params=unrep_ae_params,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
                latent_size=latent_size,
                batch_size=args.sample_batch_size,
                z_channels=z_channels,
                z_std=(1.0 / args.latent_scale_factor)
            )

            # Save the returned image grid
            out_path = os.path.join(samples_dir, f"sample_ep{ep + 1:04d}.png")
            save_image(samples_grid, out_path)

            # Log to WandB if enabled
            if use_wandb:
                wandb.log({"samples": wandb.Image(out_path), "epoch": ep + 1})

        # Save checkpoint
        unrep_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state))
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(to_bytes(unrep_state))

    if use_wandb: wandb.finish()
    print(f"Training complete. Artifacts saved to: {run_dir}")

if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    main()