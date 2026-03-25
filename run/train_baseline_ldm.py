import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import msgpack_restore, to_bytes
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.paired_latents import PairedLatentDataset
from diffusion.conditional_sampling import sample_z_cardio
from diffusion.vp_equation import alpha_fn, marginal_prob_std_fn
from utils.baseline_ldm import (
    build_conditional_ldm_from_args,
    decode_sepvae_latents,
    load_latent_meta,
    load_sepvae_v2_checkpoint,
    save_image_grid,
)

try:
    import wandb

    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def parse_args():
    parser = argparse.ArgumentParser("Conditional baselineLDM trainer")
    parser.add_argument("--preencoded_latents_dir", required=True)
    parser.add_argument("--preencoded_manifest", default="manifest.jsonl")
    parser.add_argument("--sepvae_ckpt", required=True)
    parser.add_argument("--output_root", default="runs_baseline_ldm")
    parser.add_argument("--exp_name", default="baseline_ldm_cardio")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ldm_base_ch", type=int, default=128)
    parser.add_argument("--ldm_ch_mults", type=str, default="1,2,4")
    parser.add_argument("--ldm_num_res_blocks", type=int, default=2)
    parser.add_argument("--ldm_attn_res", type=str, default="16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=250)
    parser.add_argument("--overfit_k", type=int, default=0)
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument("--use_remat", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="baseline-sepvae-ldm")
    parser.add_argument("--wandb_entity", default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine_similarity(pred, target, eps=1e-8):
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    numerator = jnp.sum(pred_flat * target_flat, axis=-1)
    denominator = (
        jnp.linalg.norm(pred_flat, axis=-1) * jnp.linalg.norm(target_flat, axis=-1) + eps
    )
    return numerator / denominator


def main():
    args = parse_args()
    dataset = PairedLatentDataset(args.preencoded_latents_dir, args.preencoded_manifest)
    if args.overfit_k > 0:
        dataset = Subset(dataset, list(range(min(args.overfit_k, len(dataset)))))

    sample_cardio, sample_common, _ = dataset[0]
    latent_size = int(sample_cardio.shape[0])
    z_channels = int(sample_cardio.shape[-1])
    cond_channels = int(sample_common.shape[-1])

    args.z_channels = z_channels
    args.cond_channels = cond_channels
    args.latent_size = latent_size

    if args.resume:
        ckpt_path = Path(args.resume)
        run_dir = ckpt_path.parent.parent
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.output_root) / f"{args.exp_name}-{ts}"
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    sample_dir = ensure_dir(run_dir / "samples")

    (run_dir / "run_meta.json").write_text(json.dumps({**vars(args), "latent_meta": load_latent_meta(args.preencoded_latents_dir)}, indent=2))

    sepvae_model, sepvae_variables, _, _ = load_sepvae_v2_checkpoint(args.sepvae_ckpt, use_ema=True)

    model = build_conditional_ldm_from_args(vars(args))
    compute_dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.ones((1, latent_size, latent_size, z_channels), dtype=compute_dtype)
    init_cond = jnp.ones((1, latent_size, latent_size, cond_channels), dtype=compute_dtype)
    init_t = jnp.ones((1,), dtype=compute_dtype)
    params = model.init(init_rng, init_x, init_t, init_cond)["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(args.lr, weight_decay=args.weight_decay),
    )
    state = TrainStateWithEMA.create(
        apply_fn=model.apply,
        params=params,
        ema_params=params if args.use_ema else None,
        tx=tx,
    )

    start_epoch = 1
    global_step = 0
    if args.resume:
        with open(args.resume, "rb") as handle:
            ckpt = msgpack_restore(handle.read())
        state = state.replace(
            params=jax.tree_util.tree_map(jnp.array, ckpt["ldm_params"]),
            opt_state=jax.tree_util.tree_map(jnp.array, ckpt["opt_state"]),
            ema_params=jax.tree_util.tree_map(jnp.array, ckpt.get("ema_params")) if ckpt.get("ema_params") is not None else None,
        )
        rng = jnp.array(ckpt["rng"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])

    def train_step(train_state, z_cardio_batch, z_common_batch, step_rng):
        z_cardio_batch = z_cardio_batch.astype(compute_dtype)
        z_common_batch = z_common_batch.astype(compute_dtype)
        rng_t, rng_noise = jax.random.split(step_rng)
        t = jax.random.uniform(rng_t, (z_cardio_batch.shape[0],), minval=1e-5, maxval=1.0)
        noise = jax.random.normal(rng_noise, z_cardio_batch.shape, dtype=compute_dtype)
        alpha = alpha_fn(t).astype(compute_dtype)[:, None, None, None]
        sigma = marginal_prob_std_fn(t).astype(compute_dtype)[:, None, None, None]
        x_t = alpha * z_cardio_batch + sigma * noise

        def loss_fn(current_params):
            eps_hat = model.apply({"params": current_params}, x_t, t, z_common_batch)
            loss = jnp.mean((eps_hat.astype(jnp.float32) - noise.astype(jnp.float32)) ** 2)
            aux = {
                "cos_eps": jnp.mean(_cosine_similarity(eps_hat.astype(jnp.float32), noise.astype(jnp.float32))),
                "z_cardio_mean": jnp.mean(z_cardio_batch.astype(jnp.float32)),
                "z_cardio_std": jnp.std(z_cardio_batch.astype(jnp.float32)),
                "z_common_mean": jnp.mean(z_common_batch.astype(jnp.float32)),
                "z_common_std": jnp.std(z_common_batch.astype(jnp.float32)),
                "xt_mean": jnp.mean(x_t.astype(jnp.float32)),
                "xt_std": jnp.std(x_t.astype(jnp.float32)),
            }
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        grad_norm = optax.global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)
        if args.use_ema:
            ema_params = optax.incremental_update(train_state.params, train_state.ema_params, args.ema_decay)
            train_state = train_state.replace(ema_params=ema_params)
        aux["grad_norm"] = grad_norm
        return train_state, loss, aux

    train_step = jax.jit(train_step)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    sample_loader = DataLoader(
        dataset,
        batch_size=min(args.sample_batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sample_batch = next(iter(sample_loader))
    _sample_z_cardio_batch = jnp.asarray(sample_batch[0].numpy())
    sample_z_common = jnp.asarray(sample_batch[1].numpy())

    use_wandb = bool(args.wandb and _WANDB)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_dir.name,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress, start=1):
            z_cardio = jnp.asarray(batch[0].numpy())
            z_common = jnp.asarray(batch[1].numpy())
            rng, step_rng = jax.random.split(rng)
            state, loss, aux = train_step(state, z_cardio, z_common, step_rng)
            global_step += 1
            loss_value = float(loss)
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}", cos=f"{float(aux['cos_eps']):.3f}")

            if global_step % args.log_every == 0:
                payload = {
                    "train/loss": loss_value,
                    "train/cos_eps": float(aux["cos_eps"]),
                    "train/grad_norm": float(aux["grad_norm"]),
                    "train/z_cardio_mean": float(aux["z_cardio_mean"]),
                    "train/z_cardio_std": float(aux["z_cardio_std"]),
                    "train/z_common_mean": float(aux["z_common_mean"]),
                    "train/z_common_std": float(aux["z_common_std"]),
                    "train/xt_mean": float(aux["xt_mean"]),
                    "train/xt_std": float(aux["xt_std"]),
                    "epoch": epoch,
                    "step": global_step,
                }
                if use_wandb:
                    wandb.log(payload, step=global_step)
                else:
                    print(payload)

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch}: mean_loss={mean_epoch_loss:.6f}")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            sample_params = state.ema_params if args.use_ema else state.params
            rng, sample_rng = jax.random.split(rng)
            sampled_cardio, _ = sample_z_cardio(
                sample_rng,
                model,
                sample_params,
                sample_z_common.astype(compute_dtype),
                n_steps=args.sample_steps,
            )
            decoded = decode_sepvae_latents(
                sepvae_model,
                sepvae_variables,
                sample_z_common.astype(jnp.float32),
                sampled_cardio.astype(jnp.float32),
            )
            sample_path = sample_dir / f"samples_epoch{epoch:04d}.png"
            save_image_grid(np.asarray(decoded), sample_path)
            if use_wandb:
                wandb.log({"samples/conditional_decode": wandb.Image(str(sample_path)), "epoch": epoch}, step=global_step)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pkl"
            with ckpt_path.open("wb") as handle:
                handle.write(
                    to_bytes(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "ldm_params": state.params,
                            "ema_params": state.ema_params,
                            "opt_state": state.opt_state,
                            "rng": rng,
                            "args": vars(args),
                        }
                    )
                )

    final_path = ckpt_dir / "checkpoint_final.pkl"
    with final_path.open("wb") as handle:
        handle.write(
            to_bytes(
                {
                    "epoch": args.epochs,
                    "global_step": global_step,
                    "ldm_params": state.params,
                    "ema_params": state.ema_params,
                    "opt_state": state.opt_state,
                    "rng": rng,
                    "args": vars(args),
                }
            )
        )
    print(f"Final checkpoint: {final_path}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
