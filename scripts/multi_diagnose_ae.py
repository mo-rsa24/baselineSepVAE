import argparse
import json
import os
import sys
from pathlib import Path
import re  # <-- FIX: Import the regular expression module

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Add project root to path to allow importing local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL


def find_latest_run_dir(parent_dir: Path):
    """Finds the subdirectory with the latest timestamp."""
    # --- FIX: Use re.match() for regex matching on the directory name string ---
    subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and re.match(r'^\d{8}-\d{6}$', d.name)]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)


def load_autoencoder(config_path, ckpt_path):
    """Loads a trained autoencoder model and its parameters."""
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        base_ch = ae_args.get('base_ch', None)
        if base_ch:
            ch_mults = tuple(base_ch * m for m in ch_mult_factors)
        else:
            ch_mults = ch_mult_factors
    else:
        ch_mults = tuple(ae_args['ch_mults'])

    attn_res_str = ae_args.get('attn_res', "")
    attn_res = tuple(int(r.strip()) for r in attn_res_str.split(',') if r.strip())

    enc_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args.get('z_channels', 4),
                   double_z=True, attn_resolutions=attn_res, in_ch=1)
    dec_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1, attn_resolutions=attn_res)

    embed_dim = ae_args.get('embed_dim', ae_args.get('z_channels', 4))
    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=embed_dim)

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    from flax.training.train_state import TrainState
    import optax

    def get_tx(lr, grad_clip, weight_decay):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    tx = get_tx(
        lr=ae_args.get('lr', 1e-4),
        grad_clip=ae_args.get('grad_clip', 1.0),
        weight_decay=ae_args.get('weight_decay', 1e-4)
    )

    gen_params = {'ae': ae_variables['params']}
    disc_params_dummy = {}
    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    from flax.serialization import from_bytes
    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    ae_params = restored_gen_state.params['ae']

    print("Autoencoder loaded successfully.")
    return ae_model, ae_params, ae_args


def run_diagnostics_for_single_run(run_dir: Path, data_root: str):
    """Runs all diagnostic checks for a single specified run directory."""
    print(f"\n{'=' * 20}\nDiagnosing run: {run_dir.parent.name}/{run_dir.name}\n{'=' * 20}")

    config_path = run_dir / "run_meta.json"
    ckpt_path = run_dir / "ckpts" / "last.flax"

    if not all([config_path.exists(), ckpt_path.exists()]):
        print(f"  [SKIP] Could not find 'run_meta.json' or 'ckpts/last.flax' in {run_dir}")
        return None

    ae_model, ae_params, ae_args = load_autoencoder(config_path, ckpt_path)

    @jax.jit
    def encode_jitted(params, x):
        return ae_model.apply({'params': params}, x, method=ae_model.encode, train=False)

    @jax.jit
    def decode_jitted(params, z):
        return ae_model.apply({'params': params}, z, method=ae_model.decode, train=False)

    ds = ChestXrayDataset(
        root_dir=data_root, task=ae_args['task'], split="train",
        img_size=ae_args['img_size'], class_filter=ae_args.get('class_filter')
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    # Check 1: Reconstruction
    print("\n--- Running Check 1: Visual Reconstruction Test ---")
    batch, _ = next(iter(loader))
    x_orig_subset = jnp.asarray(batch[:8].numpy()).transpose(0, 2, 3, 1)
    x_orig_norm = (x_orig_subset + 1.0) * 0.5
    posterior = encode_jitted(ae_params, x_orig_norm)
    x_rec = decode_jitted(ae_params, posterior.mode())

    comparison_grid = torch.cat([
        torch.from_numpy(np.asarray(x_orig_norm).transpose(0, 3, 1, 2)),
        torch.from_numpy(np.asarray(x_rec).transpose(0, 3, 1, 2))
    ], dim=0)
    grid_img = make_grid(comparison_grid, nrow=8, padding=2)
    Image.fromarray((grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        output_dir / "reconstruction_comparison.png")
    print(f"✅ Reconstruction grid saved to: {output_dir / 'reconstruction_comparison.png'}")

    # Check 2: Latent Stats
    print("\n--- Running Check 2: Latent Statistics Analysis ---")
    all_latents = []
    stat_loader = DataLoader(Subset(ds, list(range(min(1024, len(ds))))), batch_size=32, shuffle=False)
    for batch, _ in tqdm(stat_loader, desc="Encoding for stats"):
        x = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)
        x_norm = (x + 1.0) * 0.5
        posterior = encode_jitted(ae_params, x_norm)
        all_latents.append(np.asarray(posterior.mode()))

    full_latents = np.concatenate(all_latents, axis=0)

    # Generate and save individual plots
    if _HAS_PLOTLY:
        generate_plots(full_latents, output_dir, run_dir.parent.name)

    return {"name": run_dir.parent.name, "latents": full_latents}


def generate_plots(latents, out_dir, run_name):
    """Generates a suite of plotly graphs for a given set of latents."""
    if not _HAS_PLOTLY: return

    mean = np.mean(latents)
    std = np.std(latents)

    fig = px.histogram(latents.flatten(), nbins=150, title=f"Overall Latent Distribution - {run_name}")
    fig.update_layout(showlegend=False)
    fig.add_annotation(text=f"μ={mean:.3f}, σ={std:.3f}", xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False)
    fig.write_html(out_dir / f"hist_overall_{run_name}.html")

    # Per-channel plots
    C = latents.shape[-1]
    ch_means = [np.mean(latents[..., c]) for c in range(C)]
    ch_stds = [np.std(latents[..., c]) for c in range(C)]

    fig = go.Figure(data=[
        go.Bar(name='Mean', x=[f'ch{i}' for i in range(C)], y=ch_means),
        go.Bar(name='Std Dev', x=[f'ch{i}' for i in range(C)], y=ch_stds)
    ])
    fig.update_layout(barmode='group', title=f"Per-Channel Stats - {run_name}")
    fig.write_html(out_dir / f"bar_stats_{run_name}.html")


def create_comparison_report(results, output_dir: Path):
    """Creates plots comparing statistics across multiple runs."""
    if not _HAS_PLOTLY or not results:
        print("\nSkipping comparison report (Plotly not installed or no results).")
        return

    print(f"\n{'=' * 20}\nCreating Comparison Report\n{'=' * 20}")

    # Overall Std Dev Comparison
    fig = go.Figure()
    for res in results:
        fig.add_trace(go.Bar(
            x=[res['name']],
            y=[np.std(res['latents'])],
            name=res['name']
        ))
    fig.update_layout(title="Comparison of Overall Latent Standard Deviation")
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Ideal (1.0)")
    fig.write_html(output_dir / "comparison_overall_std.html")

    # Per-channel Std Dev Comparison
    fig = go.Figure()
    for res in results:
        stds = [np.std(res['latents'][..., c]) for c in range(res['latents'].shape[-1])]
        fig.add_trace(go.Bar(name=res['name'], x=[f'ch{i}' for i in range(len(stds))], y=stds))
    fig.update_layout(barmode='group', title="Comparison of Per-Channel Standard Deviation")
    fig.write_html(output_dir / "comparison_channel_std.html")

    print(f"✅ Comparison reports saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run and compare diagnostic checks on multiple AutoencoderKL runs.")
    parser.add_argument("run_dirs", nargs='+', type=str,
                        help="One or more parent run directories (e.g., 'runs/ae_tb_full_kl_1e-5_z2').")
    parser.add_argument("--data_root", type=str, default="../datasets/cleaned", help="Root directory of the dataset.")
    parser.add_argument("--output_dir", type=str, default="runs/comparison_diagnostics",
                        help="Directory to save comparison reports.")
    args = parser.parse_args()

    tf.config.experimental.set_visible_devices([], "GPU")

    all_results = []

    for parent_dir_str in args.run_dirs:
        parent_dir = Path(parent_dir_str)
        if not parent_dir.is_dir():
            print(f"[WARN] Directory not found, skipping: {parent_dir}")
            continue

        latest_run_dir = find_latest_run_dir(parent_dir)
        if not latest_run_dir:
            print(f"[WARN] No valid timestamped run directories found in {parent_dir}, skipping.")
            continue

        result = run_diagnostics_for_single_run(latest_run_dir, args.data_root)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        comparison_output_dir = Path(args.output_dir)
        comparison_output_dir.mkdir(exist_ok=True)
        create_comparison_report(all_results, comparison_output_dir)

    print("\n✅ All diagnostics complete.")


if __name__ == "__main__":
    main()

