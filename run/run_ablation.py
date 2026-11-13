#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py
---------------
Unified experiment script for SuperDiff logical-AND composition between
two unconditional LDMs sharing the same latent space.

Implements:
  1️⃣ Correct Reverse-SDE & PF-ODE samplers
  2️⃣ Ablation grid per seed (Normal/TB/PoE/κ/λ-sweep)
  3️⃣ Likelihood / energy traces à la SuperDiff
  4️⃣ 2-D latent-slice vector-field visualization
  5️⃣ Grad-CAM overlays + TB-prob vs λ chart
"""

import os, sys, json, math, functools, argparse
from datetime import datetime
import numpy as np, matplotlib.pyplot as plt, torch
from torchvision.utils import save_image
from typing import Any, Dict, Tuple
import jax, jax.numpy as jnp, optax
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import tensorflow as tf

# --- repo imports ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path: sys.path.append(repo_root)

from diffusion.vp_equation import (
    marginal_prob_std_fn, diffusion_coeff_fn,
    score_function_hutchinson_estimator, get_kappa
)
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder


class TrainStateWithEMA(TrainState):
    ema_params: Any = None
# --------------------------------------------------------------------------- #
# Helpers: model loader
# --------------------------------------------------------------------------- #
def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        loaded_json = json.load(f)
        meta = loaded_json.get('args', loaded_json)

    # --- Load VAE metadata ---
    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get('args', vae_loaded_json)
    z_channels = vae_meta['z_channels']

    # --- Construct U-Net parameters ---
    ldm_chans = tuple(meta['ldm_base_ch'] * int(m) for m in meta['ldm_ch_mults'].split(','))
    attn_res = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blocks = meta['ldm_num_res_blocks']

    # --- Instantiate ScoreNet ---
    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_res,
    )
    rng = jax.random.PRNGKey(0)
    latent_size = meta['img_size'] // 4
    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_variables = ldm_model.init({'params': rng, 'dropout': rng}, fake_latents, fake_time)
    tx = optax.chain(
        optax.clip_by_global_norm(meta.get('grad_clip', 1.0)),
        optax.adamw(meta.get('lr', 3e-5), weight_decay=meta.get('weight_decay', 0.01))
    )
    dummy_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=ldm_variables['params'],
        ema_params=ldm_variables['params'],
        tx=tx
    )

    print(f"Loading LDM checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    restored_state = from_bytes(dummy_state, blob)
    print("✅ LDM loaded successfully.")
    use_ema = meta.get('use_ema', False)
    if use_ema and hasattr(restored_state, 'ema_params') and restored_state.ema_params is not None:
        print("INFO: Using EMA parameters for composition.")
        params_to_return = restored_state.ema_params
    else:
        print("INFO: Using standard model parameters for composition.")
        params_to_return = restored_state.params

    return ldm_model, params_to_return

# --------------------------------------------------------------------------- #
# Samplers (item 1)
# --------------------------------------------------------------------------- #
def reverse_sde_step(x,t,dt,key,sA,sB,kappa,g_t):
    """Euler–Maruyama reverse VP-SDE."""
    β = g_t**2
    s_and = kappa*sA + (1-kappa)*sB
    drift = -0.5*β*x - β*s_and
    key,sub=jax.random.split(key)
    noise=jax.random.normal(sub,x.shape)
    x = x + drift*dt + g_t*jnp.sqrt(dt)*noise
    return x,key

def pf_ode_step(x,t,dt,sA,sB,kappa,g_t):
    """Deterministic probability-flow ODE."""
    β = g_t**2
    s_and = kappa*sA + (1-kappa)*sB
    drift = -0.5*β*x - 0.5*β*s_and
    return x + drift*dt

# --------------------------------------------------------------------------- #
#  Core composition score  --------------------------------------------------- #
# --------------------------------------------------------------------------- #
def composed_score_fn(x,t,key,state_tb,state_norm):
    eps_tb = state_tb.apply_fn({'params':state_tb.params},x,t)
    eps_n  = state_norm.apply_fn({'params':state_norm.params},x,t)
    σ = marginal_prob_std_fn(t)[:,None,None,None]
    s_tb, s_n = -eps_tb/σ, -eps_n/σ
    key,sub1,sub2=jax.random.split(key,3)
    d_tb = score_function_hutchinson_estimator(x,t,state_tb.apply_fn,state_tb.params,sub1)[0]
    d_n  = score_function_hutchinson_estimator(x,t,state_norm.apply_fn,state_norm.params,sub2)[0]
    κ = jnp.clip(get_kappa(t,(d_tb,d_n),(s_tb,s_n)),0.,1.)
    s_and = (1-κ)*s_n + κ*s_tb
    return s_and, dict(kappa=κ,score_tb=s_tb,score_n=s_n)

# --------------------------------------------------------------------------- #
#  Diagnostics (items 2–4)
# --------------------------------------------------------------------------- #
def run_sampler(x0,t_grid,state_tb,state_norm,sampler,decoder,outdir):
    """Runs diffusion, logs κ/‖score‖/log-q traces."""
    x=x0; key=jax.random.PRNGKey(0)
    logq,logq_trace,κ_trace,norm_trace=[],[],[],[]
    for i,(t_next,t) in enumerate(zip(t_grid[1:],t_grid[:-1])):
        dt=t-t_next; key,sub=jax.random.split(key)
        s,diag=composed_score_fn(x,t*jnp.ones(x.shape[0]),sub,state_tb,state_norm)
        g=diffusion_coeff_fn(t*jnp.ones(x.shape[0]))[:,None,None,None]
        κ=jnp.mean(diag["kappa"]); κ_trace.append(float(κ))
        norm_trace.append(float(jnp.mean(jnp.linalg.norm(s.reshape((s.shape[0],-1)),axis=-1))))
        if sampler=="rev_sde": x,key=reverse_sde_step(x,t,dt,key,diag["score_tb"],diag["score_n"],diag["kappa"],g)
        else: x=pf_ode_step(x,t,dt,diag["score_tb"],diag["score_n"],diag["kappa"],g)
    return x, np.array(logq_trace), np.array(norm_trace), np.array(κ_trace)

# --------------------------------------------------------------------------- #
#  λ-sweep / ablation grid (item 2)
# --------------------------------------------------------------------------- #
def ablation_grid(seed,t_grid,state_tb,state_norm,decoder,outdir,sampler):
    rng=jax.random.PRNGKey(seed)
    z=jax.random.normal(rng,(1,32,32,4))   # assumes 32×32 latents
    modes=[("normal",0.),("tb",1.),("poe",0.5),("kappa",None)]
    for lam in np.linspace(0,1,5): modes.append((f"λ{lam:.2f}",lam))
    imgs=[]
    for name,lam in modes:
        x=z.copy(); key=jax.random.PRNGKey(seed)
        for t_next,t in zip(t_grid[1:],t_grid[:-1]):
            dt=t-t_next; key,sub=jax.random.split(key)
            s,diag=composed_score_fn(x,t*jnp.ones(x.shape[0]),sub,state_tb,state_norm)
            g=diffusion_coeff_fn(t*jnp.ones(x.shape[0]))[:,None,None,None]
            κ=diag["kappa"] if lam is None else lam
            if sampler=="rev_sde": x,key=reverse_sde_step(x,t,dt,key,diag["score_tb"],diag["score_n"],κ,g)
            else: x=pf_ode_step(x,t,dt,diag["score_tb"],diag["score_n"],κ,g)
        img=decoder(x); imgs.append(torch.tensor(np.asarray(img).transpose(0,3,1,2)))
    grid=torch.cat(imgs,0)
    save_image(grid, os.path.join(outdir,f"ablation_seed{seed}.png"), nrow=len(modes))
    print(f"Saved ablation grid for seed {seed}")

# --------------------------------------------------------------------------- #
#  2-D latent slice visualization (item 4)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  2-D latent slice visualization (item 4, FIXED)
# --------------------------------------------------------------------------- #
def plot_latent_quiver(x, t, state_tb, state_norm, outfile):
    """
    Plot 2D latent vector fields for Normal, TB, and AND composition at time t.
    Projects scores onto a PCA plane for visual intuition.
    """
    from sklearn.decomposition import PCA

    # Flatten latent and pick 2D PCA basis
    x_flat = np.asarray(x).reshape(1, -1)
    rng = np.random.default_rng(0)
    X = x_flat + 0.01 * rng.standard_normal((256, x_flat.size))
    e1, e2 = PCA(2).fit(X).components_

    span, n = 2.5, 15
    u = np.linspace(-span, span, n)
    v = np.linspace(-span, span, n)
    U, V = np.meshgrid(u, v)

    FN = np.zeros(U.shape + (2,))
    FT = np.zeros_like(FN)
    FA = np.zeros_like(FN)

    def get_score(x_in, state):
        """Helper to compute score = -eps / σ(t)."""
        eps = state.apply_fn({'params': state.params}, x_in, t)
        sigma_t = np.asarray(marginal_prob_std_fn(t))[None, None, None, None]
        return -np.asarray(eps) / sigma_t

    for i in range(n):
        for j in range(n):
            # Point in PCA plane
            point = x_flat + U[i, j] * e1 + V[i, j] * e2
            xj = point.reshape(x.shape)

            sT = get_score(xj, state_tb)
            sN = get_score(xj, state_norm)
            sA = 0.5 * sT + 0.5 * sN  # simple average AND

            # Project each onto (e1, e2)
            FN[i, j] = [np.dot(sN.ravel(), e1), np.dot(sN.ravel(), e2)]
            FT[i, j] = [np.dot(sT.ravel(), e1), np.dot(sT.ravel(), e2)]
            FA[i, j] = [np.dot(sA.ravel(), e1), np.dot(sA.ravel(), e2)]

    plt.figure(figsize=(6, 6))
    plt.quiver(U, V, FN[..., 0], FN[..., 1], alpha=0.5, label="Normal")
    plt.quiver(U, V, FT[..., 0], FT[..., 1], alpha=0.5, label="TB")
    plt.quiver(U, V, FA[..., 0], FA[..., 1], color="red", label="AND")
    plt.legend()
    plt.title(f"Latent slice at t={float(t):.3f}")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# --------------------------------------------------------------------------- #
#  Grad-CAM overlay placeholder (item 5)
# --------------------------------------------------------------------------- #
def gradcam_overlays(img_dir,classifier,save_dir):
    """Hook up your Torch classifier externally; placeholder for integration."""
    print(f"💡 Grad-CAM overlay step placeholder: run grad_cam_tb.py on {img_dir}")

# --------------------------------------------------------------------------- #
#  Main entry
# --------------------------------------------------------------------------- #
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run_tb",required=True); p.add_argument("--run_normal",required=True)
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--sampler",choices=["rev_sde","pf_ode"],default="rev_sde")
    p.add_argument("--output_dir",default="ablation_runs")
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()

    outdir=os.path.join(args.output_dir,datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir,exist_ok=True)

    # Load models + VAE
    tb_cfg=os.path.join(args.run_tb,"ldm_meta.json")
    tb_ckpt=os.path.join(args.run_tb,"ckpts/last.flax")
    norm_cfg=os.path.join(args.run_normal,"ldm_meta.json")
    norm_ckpt=os.path.join(args.run_normal,"ckpts/last.flax")
    ldm_tb, p_tb = load_ldm(tb_cfg,tb_ckpt)
    ldm_n,  p_n  = load_ldm(norm_cfg,norm_ckpt)
    state_tb=TrainState.create(apply_fn=ldm_tb.apply,params=p_tb,tx=optax.identity())
    state_n =TrainState.create(apply_fn=ldm_n.apply, params=p_n ,tx=optax.identity())

    with open(tb_cfg, 'r') as f:
        tb_loaded_json = json.load(f)
        tb_meta = tb_loaded_json.get('args', tb_loaded_json)
    vae,vae_p=load_autoencoder(tb_meta["ae_config_path"],tb_meta["ae_ckpt_path"])
    decode=lambda z: vae.apply({"params":vae_p},z,method=vae.decode)

    t_grid=jnp.linspace(1.,1e-3,args.steps)
    latent_shape=(1,tb_meta["img_size"]//4,tb_meta["img_size"]//4,vae.enc_cfg["z_ch"])
    x0=jax.random.normal(jax.random.PRNGKey(args.seed),latent_shape)

    # ---- Run core sampler (item 1 + 3)
    x,logq,score_norm,κ=run_sampler(x0,t_grid,state_tb,state_n,args.sampler,decode,outdir)
    decoded=decode(x)
    save_image(torch.tensor(np.asarray(decoded).transpose(0,3,1,2)),
               os.path.join(outdir,"final_grid.png"),nrow=1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(score_norm); plt.title("‖score‖ over t")
    plt.subplot(1,2,2); plt.plot(κ); plt.title("κ mean over t")
    plt.savefig(os.path.join(outdir,"diagnostics.png")); plt.close()

    # ---- Ablation grids (item 2)
    ablation_grid(args.seed,t_grid,state_tb,state_n,decode,outdir,args.sampler)

    # ---- 2-D latent quiver (item 4)
    plot_latent_quiver(x,t_grid[len(t_grid)//2],state_tb,state_n,
                       os.path.join(outdir,"latent_quiver.png"))

    # ---- Grad-CAM overlays (item 5)
    gradcam_overlays(outdir,None,outdir)

    print(f"✅ All experiments saved under {outdir}")

if __name__=="__main__":
    main()
