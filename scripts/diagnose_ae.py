import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from tqdm import tqdm
# === ADD: imports near the top (keep matplotlib optional) ===
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path to allow importing local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL

# === ADD: helpers (place above main()) ===
from pathlib import Path

# ---------- ADD near top of diagnose_ae.py ----------
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

def _safe_write_plotly(fig, out_html: Path, out_png: Path = None, scale=2):
    fig.write_html(out_html)
    if out_png is not None:
        try:
            fig.write_image(out_png, scale=scale)
        except Exception:
            pass

def _qq_data_vs_std_normal(x: np.ndarray, n_points: int = 2000):
    """Return (theoretical quantiles, empirical quantiles) for QQ plot vs N(0,1)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    if x.size > n_points:
        idx = np.random.default_rng(0).choice(x.size, size=n_points, replace=False)
        x = x[idx]
    x_sorted = np.sort(x)
    probs = (np.arange(1, x_sorted.size + 1) - 0.5) / x_sorted.size
    # Theoretical quantiles for N(0,1)
    from math import sqrt, log, pi
    # Use scipy-free approximation of Φ^{-1} (Acklam):
    def inv_norm_cdf(p):
        # constants for rational approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02
        a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02
        a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02
        b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01
        b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00;  c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03;  d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00;  d4 = 3.754408661907416e+00
        plow = 0.02425; phigh = 1 - plow
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                   ((((d1*q + d2)*q + d3)*q + d4)*q + 1)
        elif p <= phigh:
            q = p - 0.5
            r = q*q
            return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
                   (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1)
        else:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                    ((((d1*q + d2)*q + d3)*q + d4)*q + 1)
    theo = np.array([inv_norm_cdf(p) for p in probs], dtype=np.float64)
    return theo, x_sorted

def _latent_stats_and_visuals(full_latents: np.ndarray, out_dir: Path, run_name: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    N,H,W,C = full_latents.shape
    flat = full_latents.reshape(N*H*W, C)           # (M, C)
    overall = flat.reshape(-1)                      # (M*C,)

    # ---------- numbers ----------
    def _stats(x):
        x = x[np.isfinite(x)]
        mu = x.mean(); sd = x.std()
        mn = x.min(); mx = x.max()
        # skewness, kurtosis (excess):
        if x.size > 0:
            m3 = ((x - mu)**3).mean()
            m4 = ((x - mu)**4).mean()
            skew = m3 / (sd**3 + 1e-12)
            kurt = m4 / (sd**4 + 1e-12) - 3.0
        else:
            skew, kurt = np.nan, np.nan
        return dict(mean=mu, std=sd, min=mn, max=mx, skew=skew, kurtosis=kurt)

    overall_stats = _stats(overall)
    ch_stats = [ _stats(flat[:,c]) for c in range(C) ]

    # Recommended scale factors
    global_scale = 1.0 / max(overall_stats["std"], 1e-12)
    per_ch_scale = [ 1.0 / max(s["std"], 1e-12) for s in ch_stats ]

    # Save CSV summary
    import csv
    csv_path = out_dir / "latent_stats_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", run_name])
        w.writerow([])
        w.writerow(["overall_mean", overall_stats["mean"]])
        w.writerow(["overall_std",  overall_stats["std"]])
        w.writerow(["overall_min",  overall_stats["min"]])
        w.writerow(["overall_max",  overall_stats["max"]])
        w.writerow(["overall_skew", overall_stats["skew"]])
        w.writerow(["overall_kurtosis", overall_stats["kurtosis"]])
        w.writerow(["global_scale_factor", global_scale])
        w.writerow([])
        w.writerow(["channel","mean","std","min","max","skew","kurtosis","per_channel_scale_factor"])
        for c,s in enumerate(ch_stats):
            w.writerow([f"ch{c}", s["mean"], s["std"], s["min"], s["max"], s["skew"], s["kurtosis"], per_ch_scale[c]])

    print("—— Latent Space (overall) ——")
    for k,v in overall_stats.items():
        print(f"{k:>8}: {v: .6f}")
    print(f"global scale factor (1/std): {global_scale:.6f}")
    print("—— Per-channel ——")
    for c,s in enumerate(ch_stats):
        print(f"[ch{c}] μ={s['mean']:.6f} σ={s['std']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}  skew={s['skew']:.3f}  kurt={s['kurtosis']:.3f}  scale=1/σ={per_ch_scale[c]:.6f}")

    # ---------- histograms with legends ----------
    title_suffix = f" — {run_name}" if run_name else ""
    if _HAS_PLOTLY:
        # overall
        fig = px.histogram(overall, nbins=150, title=f"Latent values (overall){title_suffix}")
        fig.update_layout(xaxis_title="z", yaxis_title="count")
        fig.add_annotation(text=f"μ={overall_stats['mean']:.3f} • σ={overall_stats['std']:.3f} • "
                                f"min={overall_stats['min']:.2f} • max={overall_stats['max']:.2f} • "
                                f"skew={overall_stats['skew']:.2f} • kurt={overall_stats['kurtosis']:.2f}",
                           xref="paper", yref="paper", x=0.01, y=1.12, showarrow=False)
        _safe_write_plotly(fig, out_dir/"latent_overall_hist.html", out_dir/"latent_overall_hist.png")

        # per-channel overlay
        fig = go.Figure()
        for c in range(C):
            s = ch_stats[c]
            fig.add_trace(go.Histogram(x=flat[:,c], nbinsx=120, name=f"ch{c} | μ={s['mean']:.3f}, σ={s['std']:.3f}, "
                                                                     f"min={s['min']:.2f}, max={s['max']:.2f}",
                                       opacity=0.6))
        fig.update_layout(title=f"Latent per-channel histograms{title_suffix}", barmode="overlay",
                          xaxis_title="z", yaxis_title="count")
        _safe_write_plotly(fig, out_dir/"latent_channel_hists.html", out_dir/"latent_channel_hists.png")

        # per-channel mean/std bars
        chans = [f"ch{c}" for c in range(C)]
        fig = go.Figure([
            go.Bar(name="mean", x=chans, y=[s["mean"] for s in ch_stats]),
            go.Bar(name="std",  x=chans, y=[s["std"]  for s in ch_stats])
        ])
        fig.update_layout(title=f"Per-channel mean / std{title_suffix}", barmode="group", yaxis_title="value")
        _safe_write_plotly(fig, out_dir/"latent_channel_mean_std.html", out_dir/"latent_channel_mean_std.png")

    else:
        # Matplotlib fallback: overall
        plt.figure(figsize=(6,4))
        plt.hist(overall, bins=150, alpha=0.9)
        plt.title(f"Latent values (overall){title_suffix}\n"
                  f"μ={overall_stats['mean']:.3f} • σ={overall_stats['std']:.3f} • "
                  f"min={overall_stats['min']:.2f} • max={overall_stats['max']:.2f} • "
                  f"skew={overall_stats['skew']:.2f} • kurt={overall_stats['kurtosis']:.2f}")
        plt.xlabel("z"); plt.ylabel("count"); plt.tight_layout()
        plt.savefig(out_dir/"latent_overall_hist.png", dpi=180); plt.close()

        # per-channel grid
        cols = min(3, C); rows = int(np.ceil(C/cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 3.8*rows), squeeze=False)
        for c in range(C):
            r,k = divmod(c, cols)
            s = ch_stats[c]
            axs[r][k].hist(flat[:,c], bins=120, alpha=0.9)
            axs[r][k].set_title(f"ch{c} | μ={s['mean']:.3f}, σ={s['std']:.3f}, "
                                f"min={s['min']:.2f}, max={s['max']:.2f}")
            axs[r][k].set_xlabel("z"); axs[r][k].set_ylabel("count")
        plt.tight_layout(); plt.savefig(out_dir/"latent_channel_hists.png", dpi=180); plt.close()

        # bars
        chans = np.arange(C)
        means = [s["mean"] for s in ch_stats]; stds = [s["std"] for s in ch_stats]
        plt.figure(figsize=(max(6, 1.2*C), 4))
        w=0.4
        plt.bar(chans - w/2, means, width=w, label="mean")
        plt.bar(chans + w/2, stds,  width=w, label="std")
        plt.xticks(chans, [f"ch{c}" for c in range(C)]); plt.legend()
        plt.title(f"Per-channel mean / std{title_suffix}"); plt.tight_layout()
        plt.savefig(out_dir/"latent_channel_mean_std.png", dpi=180); plt.close()

    # ---------- QQ plots vs N(0,1) ----------
    if _HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=C, subplot_titles=[f"ch{c}" for c in range(C)])
        for c in range(C):
            theo, emp = _qq_data_vs_std_normal(flat[:,c])
            fig.add_trace(go.Scatter(x=theo, y=emp, mode="markers", name=f"ch{c}", marker=dict(size=3)), row=1, col=c+1)
            # y=x line
            lo = float(np.nanmin([theo.min() if theo.size else 0, emp.min() if emp.size else 0])) - 1
            hi = float(np.nanmax([theo.max() if theo.size else 0, emp.max() if emp.size else 0])) + 1
            fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=c+1)
        fig.update_layout(title=f"QQ vs 𝒩(0,1){title_suffix}")
        _safe_write_plotly(fig, out_dir/"latent_channel_qq.html", out_dir/"latent_channel_qq.png")
    else:
        cols = min(3, C); rows = int(np.ceil(C/cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
        for c in range(C):
            theo, emp = _qq_data_vs_std_normal(flat[:,c])
            r,k = divmod(c, cols)
            axs[r][k].plot(theo, emp, ".", ms=2)
            lims = [min(theo.min() if theo.size else 0, emp.min() if emp.size else 0),
                    max(theo.max() if theo.size else 0, emp.max() if emp.size else 0)]
            axs[r][k].plot(lims, lims, "k--", lw=1)
            axs[r][k].set_title(f"ch{c}")
            axs[r][k].set_xlabel("theoretical N(0,1)"); axs[r][k].set_ylabel("empirical")
        fig.suptitle(f"QQ vs 𝒩(0,1){title_suffix}")
        plt.tight_layout(); plt.savefig(out_dir/"latent_channel_qq.png", dpi=180); plt.close()

    # ---------- correlation heatmap & PCA ----------
    # Corr over channels
    if C >= 2:
        corr = np.corrcoef(flat.T)
        if _HAS_PLOTLY:
            fig = px.imshow(corr, text_auto=True, title=f"Channel correlation{title_suffix}", x=[f"ch{c}" for c in range(C)], y=[f"ch{c}" for c in range(C)])
            _safe_write_plotly(fig, out_dir/"latent_channel_corr.html", out_dir/"latent_channel_corr.png")
        else:
            plt.figure(figsize=(3+C*0.6, 3+C*0.6))
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(); plt.xticks(range(C), [f"ch{c}" for c in range(C)]); plt.yticks(range(C), [f"ch{c}" for c in range(C)])
            plt.title(f"Channel correlation{title_suffix}")
            plt.tight_layout(); plt.savefig(out_dir/"latent_channel_corr.png", dpi=180); plt.close()

    # PCA variance explained (over concatenated pixels)
    # Center channels
    Z = flat - flat.mean(axis=0, keepdims=True)
    # SVD on CxC covariance if you want channel-PCA, but here do feature PCA across channels*spatial can be huge.
    # We'll do PCA only along channel dimension by using channel covariance:
    cov_ch = np.cov(flat, rowvar=False)   # (C, C)
    evals, _ = np.linalg.eigh(cov_ch)
    evals = np.sort(evals)[::-1]
    var_expl = evals / (evals.sum() + 1e-12)
    if _HAS_PLOTLY:
        fig = px.bar(x=[f"PC{i+1}" for i in range(C)], y=var_expl, title=f"PCA (channel) variance explained{title_suffix}", labels={"x":"PC", "y":"fraction"})
        _safe_write_plotly(fig, out_dir/"latent_channel_pca_var.html", out_dir/"latent_channel_pca_var.png")
    else:
        plt.figure(figsize=(max(5, 1.0*C), 3.6))
        plt.bar(np.arange(C), var_expl); plt.xticks(np.arange(C), [f"PC{i+1}" for i in range(C)])
        plt.ylabel("fraction"); plt.title(f"PCA (channel) variance explained{title_suffix}")
        plt.tight_layout(); plt.savefig(out_dir/"latent_channel_pca_var.png", dpi=180); plt.close()

    return dict(overall=overall_stats, per_channel=ch_stats,
                global_scale=global_scale, per_channel_scale=per_ch_scale)
# ---------- END of block ----------


def _save_histograms_and_bars(full_latents: np.ndarray, out_dir: Path):
    """
    Saves:
      - latent_overall_hist.(png/html)
      - latent_channel_hists.(png/html)
      - latent_channel_mean_std.(png/html)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    N,H,W,C = full_latents.shape
    flat = full_latents.reshape(N*H*W, C)

    # ---- Overall histogram ----
    overall = flat.reshape(-1)
    if _HAS_PLOTLY:
        fig = px.histogram(overall, nbins=120, title="Latent values (overall)")
        fig.update_layout(xaxis_title="z", yaxis_title="count", bargap=0.02)
        fig.write_html(out_dir / "latent_overall_hist.html")
        try: fig.write_image(out_dir / "latent_overall_hist.png", scale=2)
        except Exception: pass
    else:
        plt.figure(figsize=(5,4))
        plt.hist(overall, bins=120)
        plt.title("Latent values (overall)")
        plt.xlabel("z"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(out_dir / "latent_overall_hist.png", dpi=160); plt.close()

    # ---- Per-channel histograms ----
    if _HAS_PLOTLY:
        fig = make_subplots = go.Figure()
        for c in range(C):
            fig.add_trace(go.Histogram(x=flat[:, c], nbinsx=100, name=f"ch{c}", opacity=0.6))
        fig.update_layout(title="Latent per-channel histograms", barmode="overlay",
                          xaxis_title="z", yaxis_title="count")
        fig.write_html(out_dir / "latent_channel_hists.html")
        try: fig.write_image(out_dir / "latent_channel_hists.png", scale=2)
        except Exception: pass
    else:
        cols = min(C, 4)
        rows = int(np.ceil(C/cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
        for c in range(C):
            r, k = divmod(c, cols)
            axs[r][k].hist(flat[:, c], bins=100)
            axs[r][k].set_title(f"channel {c}")
        plt.tight_layout(); plt.savefig(out_dir / "latent_channel_hists.png", dpi=160); plt.close()

    # ---- Per-channel mean/std bar chart ----
    ch_mean = flat.mean(axis=0)
    ch_std  = flat.std(axis=0)
    channels = [f"ch{c}" for c in range(C)]
    if _HAS_PLOTLY:
        fig = go.Figure([
            go.Bar(name="mean", x=channels, y=ch_mean),
            go.Bar(name="std",  x=channels, y=ch_std),
        ])
        fig.update_layout(title="Latent per-channel mean/std", barmode="group",
                          yaxis_title="value")
        fig.write_html(out_dir / "latent_channel_mean_std.html")
        try: fig.write_image(out_dir / "latent_channel_mean_std.png", scale=2)
        except Exception: pass
    else:
        x = np.arange(C)
        w = 0.35
        plt.figure(figsize=(max(5, C*0.9), 4))
        plt.bar(x - w/2, ch_mean, width=w, label="mean")
        plt.bar(x + w/2, ch_std,  width=w, label="std")
        plt.xticks(x, channels); plt.title("Latent per-channel mean/std")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "latent_channel_mean_std.png", dpi=160); plt.close()


def _decode_scale_sweep(ae_model, ae_params, decode_fn, z_batch_jnp: jnp.ndarray,
                        out_dir: Path, scales=(0.5, 1.0, 1.387, 2.0)):
    """
    Saves: decode_scale_sweep.png (grid: each row = a scale)
    """
    import torch
    from torchvision.utils import make_grid
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for s in scales:
        x_rec = decode_fn(ae_params, z_batch_jnp * s)
        x_t   = torch.from_numpy(np.asarray(x_rec).transpose(0,3,1,2)).clamp(0,1)
        imgs.append(x_t)

    grid = make_grid(torch.cat(imgs, dim=0),
                     nrow=z_batch_jnp.shape[0], padding=2, normalize=False)
    from PIL import Image
    Image.fromarray((grid.permute(1,2,0).numpy()*255).astype(np.uint8)) \
        .save(out_dir / "decode_scale_sweep.png")


def load_autoencoder(config_path, ckpt_path):
    """Loads a trained autoencoder model and its parameters."""
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # --- FIX: The ch_mults from run_meta.json is a string, needs parsing ---
    # The training script saves "64,128,256" but the model expects a tuple of ints.
    # The original script was missing this parsing logic, which I've taken from autoencoder.py
    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        # The training script might save base multipliers or the final values.
        # Let's check if a base_ch exists.
        base_ch = ae_args.get('base_ch', None)
        if base_ch:
            ch_mults = tuple(base_ch * m for m in ch_mult_factors)
        else:  # Assumes the values in ch_mults are the final channel counts
            ch_mults = ch_mult_factors
    else:
        ch_mults = tuple(ae_args['ch_mults'])

    # The training script now uses attn_resolutions, let's add it for compatibility
    attn_res_str = ae_args.get('attn_res', "")
    attn_res = tuple(int(r.strip()) for r in attn_res_str.split(',') if r.strip())

    enc_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args.get('z_channels', 4),
                   double_z=True, attn_resolutions=attn_res, in_ch=1)  # added in_ch
    dec_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1, attn_resolutions=attn_res)

    embed_dim = ae_args.get('embed_dim', ae_args.get('z_channels', 4))
    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=embed_dim)

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    # Create dummy states to restore into
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    from flax.training.train_state import TrainState
    import optax

    def get_tx(lr, grad_clip, weight_decay):
        """Replicates the optimizer creation from the training script."""
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    tx = get_tx(
        lr=ae_args.get('lr', 1e-4),
        grad_clip=ae_args.get('grad_clip', 1.0),
        weight_decay=ae_args.get('weight_decay', 1e-4)
    )

    # Recreate the structure from training (gen_state, disc_state)
    gen_params = {'ae': ae_variables['params']}
    disc_params_dummy = {}  # Dummy structure, params are not needed for inference

    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    from flax.serialization import from_bytes
    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    ae_params = restored_gen_state.params['ae']

    print("Autoencoder loaded successfully.")
    return ae_model, ae_params, ae_args


# --- REMOVED Global encode/decode functions that caused the JIT error ---
# ae_full_tb_b4_20250924/20250924-041825
def check_1_reconstruction(encode_fn, decode_fn, ae_params, loader, output_dir, n_images=8):
    """CHECK 1: Visual Reconstruction Test."""
    print("\n--- Running Check 1: Visual Reconstruction Test ---")
    batch, _ = next(iter(loader))
    x_orig = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)  # to (N, H, W, C)

    # The training script normalizes to [0, 1]. Let's match that.
    x_orig_norm = (x_orig + 1.0) * 0.5

    # Use only a subset for the grid
    x_orig_subset = x_orig_norm[:n_images]

    posterior = encode_fn(ae_params, x_orig_subset)
    z = posterior.mode()  # Use mode for deterministic reconstruction
    x_rec = decode_fn(ae_params, z)

    # Convert back to torch for grid saving
    x_orig_torch = torch.from_numpy(np.asarray(x_orig_subset).transpose(0, 3, 1, 2))
    x_rec_torch = torch.from_numpy(np.asarray(x_rec).transpose(0, 3, 1, 2))

    # Interleave original and reconstructed images for comparison
    comparison_grid = torch.cat([x_orig_torch, x_rec_torch], dim=0)
    # Correctly interleave images: [orig1, rec1, orig2, rec2, ...]
    interleaved = torch.empty((2 * n_images, 1, x_orig_subset.shape[1], x_orig_subset.shape[2]),
                              dtype=x_orig_torch.dtype)
    interleaved[0::2] = x_orig_torch
    interleaved[1::2] = x_rec_torch

    grid_img = make_grid(interleaved, nrow=2, padding=2, normalize=True)
    grid_img_np = (grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    output_path = Path(output_dir) / "reconstruction_comparison.png"
    Image.fromarray(grid_img_np).save(output_path)
    print(f"✅ Reconstruction grid saved to: {output_path}")
    print("   (Left column: Original, Right column: Reconstructed)")


def check_2_latent_stats(encode_fn, ae_params, loader, n_samples=1024):
    """CHECK 2: Latent Statistics Analysis."""
    print("\n--- Running Check 2: Latent Statistics Analysis ---")
    all_latents = []

    # Limit number of samples for efficiency
    dataset = loader.dataset
    if len(dataset) > n_samples:
        dataset = Subset(dataset, list(range(n_samples)))

    stat_loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=False, num_workers=4)

    print(f"Encoding {len(dataset)} samples to calculate latent statistics...")
    for batch, _ in tqdm(stat_loader):
        x = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)  # to (N, H, W, C)
        x_norm = (x + 1.0) * 0.5  # [-1, 1] -> [0, 1] as in training

        posterior = encode_fn(ae_params, x_norm)
        z = posterior.mode()  # Use mode (mean) for stable statistics
        all_latents.append(np.asarray(z))

    full_latents = np.concatenate(all_latents, axis=0)

    mean = np.mean(full_latents)
    std = np.std(full_latents)
    min_val = np.min(full_latents)
    max_val = np.max(full_latents)

    print("\n--- Latent Space Statistics ---")
    print(f"  Mean:     {mean:.6f}")
    print(f"  Std Dev:  {std:.6f}")
    print(f"  Min:      {min_val:.6f}")
    print(f"  Max:      {max_val:.6f}")
    print("---------------------------------")

    # Provide feedback
    if abs(mean) < 0.1 and 0.85 < std < 1.15:
        print("✅ PASSED: Latent distribution appears well-normalized (mean≈0, std≈1).")
    else:
        print("⚠️ FAILED: Latent distribution is NOT well-normalized. The LDM may struggle.")
        print("   Consider adjusting the KL_WEIGHT during AE training.")
    return full_latents

def main():
    parser = argparse.ArgumentParser(description="Run diagnostic checks on a trained AutoencoderKL model.")
    parser.add_argument("run_dir", type=str,
                        help="Path to the AE run directory (containing run_meta.json and ckpts/last.flax).")
    parser.add_argument("--data_root", type=str, default="../datasets/cleaned", help="Root directory of the dataset.")
    args = parser.parse_args()

    # Disable GPU usage by TensorFlow, which is often used for data loading.
    tf.config.experimental.set_visible_devices([], "GPU")

    run_path = Path(args.run_dir)
    config_path = run_path / "run_meta.json"
    ckpt_path = run_path / "ckpts" / "last.flax"

    if not all([config_path.exists(), ckpt_path.exists()]):
        print(f"Error: Could not find 'run_meta.json' or 'ckpts/last.flax' in {run_path}")
        sys.exit(1)

    ae_model, ae_params, ae_args = load_autoencoder(config_path, ckpt_path)

    # --- FIX: Define JIT-compiled functions here, closing over the model instance ---
    @jax.jit
    def encode_jitted(params, x):
        return ae_model.apply({'params': params}, x, method=ae_model.encode, train=False)

    @jax.jit
    def decode_jitted(params, z):
        return ae_model.apply({'params': params}, z, method=ae_model.decode, train=False)

    # Setup dataset loader
    ds = ChestXrayDataset(
        root_dir=args.data_root, task=ae_args['task'], split="train",
        img_size=ae_args['img_size'], class_filter=ae_args.get('class_filter')
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    # Run Checks
    output_dir = run_path / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    # Pass the jitted functions to the check routines
    check_1_reconstruction(encode_jitted, decode_jitted, ae_params, loader, output_dir)
    full_latents = check_2_latent_stats(encode_jitted, ae_params, loader)

    print("\n--- Check 3: LDM Overfitting Test (Guidance) ---")
    print("To perform this check, you must now train the LDM on these verified latents.")
    print("1. Update 'ldm_tb_diagnostic_train.sh' with the path to this verified AE run.")
    print(
        "2. IMPORTANT: If your latent space is now normalized (std≈1), you may not need a 'latent_scale_factor' > 1.0 for the LDM.")
    print("   You can start by removing it or setting it to 1.0 in your LDM launch script.")
    print(
        "3. Run the LDM diagnostic script. If the loss decreases and samples are non-black, your pipeline is working.")

    _save_histograms_and_bars(full_latents, output_dir)

    # NEW: scale sweep on a small fixed latent batch (first 8 samples)
    z_batch = jnp.asarray(full_latents[:8])
    _decode_scale_sweep(ae_model, ae_params, decode_jitted, z_batch, output_dir,
                        scales=(0.5, 1.0, 1.387, 2.0))
    print(f"📁 Saved plots to: {output_dir}")
    stats = _latent_stats_and_visuals(full_latents, output_dir, run_name="ae_full_tb_b4_20250924")
    # Optional: print recommended scale
    print("Recommended global latent_scale_factor (1/std):", stats["global_scale"])
    print("Per-channel 1/std (diagnostic):", stats["per_channel_scale"])

    print("✅ Diagnostics complete.")



if __name__ == "__main__":
    main()