"""
Evaluation and sampling script for Multi-head SepVAE.

Supports:
- Reconstruction grids from test set
- Linear probe evaluation (disease separability)
- Latent manifold visualization (all heads + per-head)
- GradCAM spatial attention maps
- W&B logging (attach to existing run or create new)

Usage:
    python -m run.eval_sep_vae --checkpoint path/to/ckpt.pkl --all
    python -m run.eval_sep_vae --checkpoint path/to/ckpt.pkl --reconstruct --manifold
    python -m run.eval_sep_vae --checkpoint path/to/ckpt.pkl --all --wandb --wandb_run_id abc123
"""

import argparse
import os
from pathlib import Path

# Prevent JAX from pre-allocating all GPU memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from flax.serialization import msgpack_restore

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
from models.sep_vae_jax import SepVAE

# Optional W&B
try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


DISEASE_NAMES = ['Normal', 'Effusion', 'Cardiomegaly']
DISEASE_COLORS = ['blue', 'red', 'green']


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("SepVAE Evaluation & Sampling")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to SepVAE checkpoint (.pkl file)")
    p.add_argument("--output_dir", type=str, default="eval_sepvae",
                   help="Output directory for evaluation results")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")

    # Dataset (same defaults as train_sep_vae.py)
    p.add_argument("--dicom_dir", type=str, default="/datasets/mmolefe/vinbigdata/train",
                   help="Path to DICOM files")
    p.add_argument("--csv_path", type=str, default="/datasets/mmolefe/vinbigdata/train.csv",
                   help="Path to train.csv")
    p.add_argument("--img_size", type=int, default=512,
                   help="Image size (default: 512)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Batch size")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers")

    # Model architecture (must match training config)
    p.add_argument("--z_channels_common", type=int, default=4,
                   help="Common latent channels (default: 4)")
    p.add_argument("--z_channels_disease", type=int, default=2,
                   help="Disease-specific latent channels (default: 2)")
    p.add_argument("--use_fpn", action="store_true", default=None,
                   help="Enable FPN (auto-detected from checkpoint if not specified)")
    p.add_argument("--fpn_channels", type=int, default=None,
                   help="FPN output channels (auto-detected from checkpoint if not specified)")
    p.add_argument("--unfreeze_from", type=str, default=None,
                   help="Backbone unfreezing layer (auto-detected from checkpoint if not specified)")

    # W&B integration
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Attach to existing W&B run ID (resume='allow')")
    p.add_argument("--wandb_project", type=str, default="sepvae-chess",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (username/team)")

    # Evaluation modes
    p.add_argument("--reconstruct", action="store_true",
                   help="Generate reconstruction grids from test set")
    p.add_argument("--linear_probe", action="store_true",
                   help="Run linear probe evaluation")
    p.add_argument("--manifold", action="store_true",
                   help="Generate latent manifold visualizations")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate GradCAM visualizations")
    p.add_argument("--separability", action="store_true",
                   help="Quantitative disease separability analysis (Fisher ratio, "
                        "inter/intra-class distances, head comparison)")
    p.add_argument("--all", action="store_true",
                   help="Run all evaluations")

    # Manifold options
    p.add_argument("--manifold_method", type=str, default="both",
                   choices=["tsne", "pca", "both"],
                   help="Dimensionality reduction method for manifold")
    p.add_argument("--manifold_max_samples", type=int, default=1000,
                   help="Max samples for manifold visualization")

    # Reconstruction options
    p.add_argument("--n_samples_per_class", type=int, default=2,
                   help="Number of samples per class in reconstruction grid")

    return p.parse_args()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_checkpoint(ckpt_path, z_channels_common=4, z_channels_disease=2,
                    use_fpn=None, fpn_channels=None, unfreeze_from=None):
    """
    Load SepVAE checkpoint using msgpack_restore (no template needed).

    This avoids the OOM issue of the template-based approach which had to
    allocate the full model + optimizer states on GPU just for deserialization.
    msgpack_restore deserializes directly to numpy arrays.

    Architecture parameters (use_fpn, fpn_channels, unfreeze_from) are
    auto-detected from the checkpoint's saved training args when not
    explicitly provided.

    Args:
        ckpt_path: Path to .pkl checkpoint
        z_channels_common: Common latent channels
        z_channels_disease: Disease-specific latent channels
        use_fpn: Override FPN setting (None = auto-detect from checkpoint)
        fpn_channels: Override FPN channels (None = auto-detect)
        unfreeze_from: Override unfreezing (None = auto-detect)

    Returns:
        model: SepVAE model instance (uninitialized, used for .apply())
        vae_params: Trained parameters (as JAX arrays)
        vae_batch_stats: BatchNorm statistics (as JAX arrays)
        ckpt_meta: Dict with epoch, global_step, training args
    """
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        raw = msgpack_restore(f.read())

    # Only convert the parameters we need to JAX arrays (skip optimizer states)
    vae_params = jax.tree_util.tree_map(jnp.array, raw['vae_params'])
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, raw['vae_batch_stats'])

    ckpt_meta = {
        'epoch': int(raw['epoch']),
        'global_step': int(raw['global_step']),
        'args': raw.get('args', {}),
    }

    # Auto-detect architecture from checkpoint args
    ckpt_args = ckpt_meta.get('args', {})
    _use_fpn = use_fpn if use_fpn is not None else bool(ckpt_args.get('use_fpn', False))
    _fpn_channels = fpn_channels if fpn_channels is not None else int(ckpt_args.get('fpn_channels', 512))
    _unfreeze_from = unfreeze_from if unfreeze_from is not None else ckpt_args.get('unfreeze_from', None)
    _sigma_inactive = float(ckpt_args.get('sigma_inactive', 0.1))

    # Auto-detect mixed precision dtype from checkpoint
    _half_prec = ckpt_args.get('half_precision', 'none')
    if _half_prec == 'bf16':
        _dtype = jnp.bfloat16
    elif _half_prec == 'fp16':
        _dtype = jnp.float16
    else:
        _dtype = jnp.float32

    # Build model instance (no GPU allocation - just the module definition)
    # use_remat=False for eval (no gradient checkpointing needed)
    model = SepVAE(
        z_channels_common=z_channels_common,
        z_channels_disease=z_channels_disease,
        frozen_backbone=True,
        use_fpn=_use_fpn,
        fpn_channels=_fpn_channels,
        unfreeze_from=_unfreeze_from,
        sigma_inactive=_sigma_inactive,
        dtype=_dtype,
        use_remat=False,
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    print(f"  Loaded {param_count:,} parameters from epoch {ckpt_meta['epoch']}")
    if _use_fpn:
        print(f"  Architecture: FPN enabled ({_fpn_channels} channels)")
    if _unfreeze_from:
        print(f"  Architecture: Partial unfreezing from {_unfreeze_from}")

    return model, vae_params, vae_batch_stats, ckpt_meta


# ============================================================================
# Data Loading
# ============================================================================

def build_data_loaders(dicom_dir, csv_path, img_size, batch_size,
                       num_workers, test_fraction=0.2, seed=0):
    """
    Build train and test DataLoaders from VinBigData.

    Uses torch.utils.data.random_split for train/test splitting.
    """
    dataset = VinBigDataTripletDataset(
        dicom_dir=dicom_dir,
        csv_path=csv_path,
        img_size=img_size
    )

    total = len(dataset)
    test_size = int(total * test_fraction)
    train_size = total - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=jax_collate_fn, drop_last=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=jax_collate_fn, drop_last=True
    )

    print(f"  Total triplets: {total}")
    print(f"  Train: {train_size}, Test: {test_size}")
    print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    return train_loader, test_loader


# ============================================================================
# Reconstruction Grid
# ============================================================================

def make_sepvae_recon_grid(x_input, x_rec, labels, n_per_class=4):
    """
    Create a reconstruction grid organized by disease type.

    Layout:
        Row 0 (Original):       Normal ... | Effusion ... | Cardiomegaly ...
        Row 1 (Reconstruction): Normal ... | Effusion ... | Cardiomegaly ...

    Args:
        x_input: Original images (3B, H, W, 1) in [-1, 1]
        x_rec: Reconstructed images (3B, H, W, 1) in [0, 1]
        labels: Disease labels (3B,)
        n_per_class: Number of samples to show per class

    Returns:
        PIL.Image: The grid image
    """
    from torchvision.utils import make_grid
    from PIL import Image

    x_input_01 = (np.array(x_input) + 1.0) / 2.0
    x_rec_np = np.clip(np.array(x_rec), 0.0, 1.0)
    labels_np = np.array(labels)

    originals = []
    reconstructions = []

    for cls_id in [0, 1, 2]:
        mask = labels_np == cls_id
        idxs = np.where(mask)[0][:n_per_class]
        for idx in idxs:
            originals.append(x_input_01[idx])
            reconstructions.append(x_rec_np[idx])

    all_imgs = originals + reconstructions
    nrow = len(originals)

    imgs_np = np.stack(all_imgs, axis=0)
    imgs_np = np.transpose(imgs_np, (0, 3, 1, 2))  # NHWC -> NCHW
    imgs_torch = torch.tensor(imgs_np).clamp(0, 1)

    grid = make_grid(imgs_torch, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_img = Image.fromarray((grid_np * 255).astype(np.uint8))

    return grid_img


def evaluate_reconstruction(model, vae_params, vae_batch_stats,
                            test_loader, output_dir, n_per_class=4,
                            use_wandb=False):
    """
    Generate reconstruction grids from the test set.

    Takes the first batch from the test loader, runs a forward pass,
    and creates a grid showing originals vs reconstructions grouped
    by disease type.
    """
    print("\nGenerating reconstruction grids...")
    recon_dir = ensure_dir(os.path.join(output_dir, "reconstructions"))

    batch_torch = next(iter(test_loader))
    batch = {
        'x_norm': jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()),
        'x_disease1': jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()),
        'x_disease2': jnp.array(batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy()),
        'disease_labels': jnp.array(batch_torch['disease_labels'].numpy()),
    }

    x = jnp.concatenate([batch['x_norm'], batch['x_disease1'], batch['x_disease2']], axis=0)
    labels = batch['disease_labels']

    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    rng = jax.random.PRNGKey(42)
    x_rec, _, _ = model.apply(variables, x, labels, key=rng, train=False)

    grid_img = make_sepvae_recon_grid(x, x_rec, labels, n_per_class=n_per_class)

    grid_path = os.path.join(recon_dir, "test_reconstruction_grid.png")
    grid_img.save(grid_path)
    print(f"  Saved reconstruction grid: {grid_path}")

    if use_wandb and wandb:
        wandb.log({"eval/reconstruction_grid": wandb.Image(grid_path)})

    return grid_path


# ============================================================================
# Linear Probe
# ============================================================================

def _extract_latents(model, vae_params, vae_batch_stats, loader, max_batches=None):
    """Extract latent representations from all three image types in each batch."""
    latents_common = []
    latents_cardio = []
    latents_effusion = []
    labels_list = []

    variables = {'params': vae_params}
    if vae_batch_stats:
        variables['batch_stats'] = vae_batch_stats

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        # Encode all three image types
        for img_key, label_val in [('x_norm', 0), ('x_disease1', 1), ('x_disease2', 2)]:
            x = jnp.array(batch[img_key].permute(0, 2, 3, 1).numpy())
            B = x.shape[0]

            latents_dict = model.apply(variables, x, method=model.encode)

            mu_c, _ = latents_dict['common']
            mu_cardio, _ = latents_dict['cardiomegaly']
            mu_effusion, _ = latents_dict['effusion']

            # Global average pool spatial dims
            latents_common.extend(np.array(jnp.mean(mu_c, axis=(1, 2))))
            latents_cardio.extend(np.array(jnp.mean(mu_cardio, axis=(1, 2))))
            latents_effusion.extend(np.array(jnp.mean(mu_effusion, axis=(1, 2))))
            labels_list.extend([label_val] * B)

    return {
        'common': np.array(latents_common),
        'cardio': np.array(latents_cardio),
        'effusion': np.array(latents_effusion),
        'labels': np.array(labels_list),
    }


def _save_confusion_matrix(y_true, y_pred, save_dir):
    """Generate and save confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISEASE_NAMES, yticklabels=DISEASE_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Linear Probe Confusion Matrix (Disease Heads)')

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {cm_path}")
    return cm_path


def evaluate_linear_probe(model, vae_params, vae_batch_stats,
                          train_loader, test_loader, output_dir,
                          use_wandb=False):
    """
    Run linear probe evaluation.

    Trains logistic regression classifiers on frozen latent representations
    to evaluate whether disease information is properly separated across heads.

    Expected results:
    - common_only: LOW accuracy (disease-agnostic)
    - disease_only: HIGH accuracy (disease-specific)
    - all_heads: Baseline accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    print("\nRunning linear probe evaluation...")
    probe_dir = ensure_dir(os.path.join(output_dir, "linear_probe"))

    # Extract latents
    print("  Extracting train latents...")
    train_data = _extract_latents(model, vae_params, vae_batch_stats, train_loader)
    print(f"    {len(train_data['labels'])} samples, "
          f"common={train_data['common'].shape[1]}D, "
          f"cardio={train_data['cardio'].shape[1]}D, "
          f"effusion={train_data['effusion'].shape[1]}D")

    print("  Extracting test latents...")
    test_data = _extract_latents(model, vae_params, vae_batch_stats, test_loader)
    print(f"    {len(test_data['labels'])} samples")

    # Feature sets to evaluate
    feature_sets = {
        'common_only': (train_data['common'], test_data['common']),
        'disease_only': (
            np.concatenate([train_data['cardio'], train_data['effusion']], axis=1),
            np.concatenate([test_data['cardio'], test_data['effusion']], axis=1)
        ),
        'all_heads': (
            np.concatenate([train_data['common'], train_data['cardio'],
                            train_data['effusion']], axis=1),
            np.concatenate([test_data['common'], test_data['cardio'],
                            test_data['effusion']], axis=1)
        ),
    }

    results = {}

    for name, (X_train, X_test) in feature_sets.items():
        print(f"\n  {name} (dim={X_train.shape[1]}):")

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, train_data['labels'])

        train_acc = clf.score(X_train, train_data['labels'])
        test_acc = clf.score(X_test, test_data['labels'])

        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}

        print(f"    Train accuracy: {train_acc:.3f}")
        print(f"    Test accuracy:  {test_acc:.3f}")

        y_pred = clf.predict(X_test)
        report = classification_report(
            test_data['labels'], y_pred,
            target_names=DISEASE_NAMES, output_dict=False
        )
        print(f"\n    Classification Report:\n{report}")

        # Save confusion matrix for disease_only (most relevant)
        if name == 'disease_only':
            cm_path = _save_confusion_matrix(test_data['labels'], y_pred, probe_dir)

    # Summary
    print("\n  " + "-" * 40)
    print(f"  Common only (should be LOW):   {results['common_only']['test_acc']:.3f}")
    print(f"  Disease only (should be HIGH):  {results['disease_only']['test_acc']:.3f}")
    print(f"  All heads (baseline):           {results['all_heads']['test_acc']:.3f}")

    if results['common_only']['test_acc'] < 0.5:
        print("\n  Common head is disease-agnostic (good)")
    else:
        print("\n  WARNING: Common head contains disease info (check nulling)")

    if results['disease_only']['test_acc'] > 0.8:
        print("  Disease heads are discriminative (good)")
    else:
        print("  WARNING: Disease heads not discriminative (check training)")

    if use_wandb and wandb:
        for head_name, metrics in results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"eval/probe/{head_name}/{metric_name}": value})
        cm_path = os.path.join(probe_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            wandb.log({"eval/confusion_matrix": wandb.Image(cm_path)})

    return results


# ============================================================================
# Manifold Visualization
# ============================================================================

def _visualize_head_specific_manifolds(latents_common, latents_cardio,
                                       latents_effusion, labels_all,
                                       save_dir, method='tsne',
                                       use_wandb=False):
    """
    Generate separate manifold plots for each latent head.

    Creates a 2x2 figure:
    - common_only: Expected mixed/weak disease separation
    - disease_only: Expected strong disease separation
    - cardio_head: Cardiomegaly head alone
    - effusion_head: Effusion head alone
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    feature_sets = {
        'common_only': latents_common,
        'disease_only': np.concatenate([latents_cardio, latents_effusion], axis=1),
        'cardio_head': latents_cardio,
        'effusion_head': latents_effusion,
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    for idx, (name, features) in enumerate(feature_sets.items()):
        if method == 'tsne':
            # Adjust perplexity for small datasets
            perp = min(30, max(5, len(features) // 4))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        else:
            reducer = PCA(n_components=2, random_state=42)

        z_2d = reducer.fit_transform(features)

        ax = axes_flat[idx]
        for disease_id in range(3):
            mask = labels_all == disease_id
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=DISEASE_COLORS[disease_id],
                       label=DISEASE_NAMES[disease_id],
                       alpha=0.6, s=20)

        try:
            sil = silhouette_score(z_2d, labels_all)
            sil_str = f"{sil:.3f}"
        except ValueError:
            sil_str = "N/A"

        ax.set_title(f'{name} ({method.upper()})\nSilhouette: {sil_str}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Head-Specific Latent Manifold ({method.upper()})', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"manifold_heads_comparison_{method}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved head-specific manifold: {save_path}")

    if use_wandb and wandb:
        wandb.log({f"eval/manifold_heads_{method}": wandb.Image(save_path)})


def _visualize_all_heads_manifold(latents_common, latents_cardio,
                                  latents_effusion, labels_all,
                                  save_dir, method='tsne',
                                  use_wandb=False):
    """Visualize all heads concatenated in a single manifold plot."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    z_all = np.concatenate([latents_common, latents_cardio, latents_effusion], axis=1)
    print(f"  All-heads latent dim: {z_all.shape[1]}")

    if method == 'tsne':
        perp = min(30, max(5, len(z_all) // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
    else:
        reducer = PCA(n_components=2, random_state=42)

    z_2d = reducer.fit_transform(z_all)

    fig, ax = plt.subplots(figsize=(10, 8))

    for disease_id in range(3):
        mask = labels_all == disease_id
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=DISEASE_COLORS[disease_id],
                   label=DISEASE_NAMES[disease_id],
                   alpha=0.6, s=20)

    try:
        sil = silhouette_score(z_2d, labels_all)
        print(f"  Silhouette score: {sil:.3f}")
    except ValueError:
        sil = 0.0

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(f"Latent Space Manifold - All Heads ({method.upper()})\n"
                 f"Silhouette: {sil:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"manifold_all_{method}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved all-heads manifold: {save_path}")

    if use_wandb and wandb:
        wandb.log({f"eval/manifold_all_{method}": wandb.Image(save_path)})


def evaluate_manifold(model, vae_params, vae_batch_stats,
                      data_loader, output_dir, method='both',
                      max_samples=1000, use_wandb=False):
    """
    Generate manifold visualizations including head-specific plots.

    Produces:
    1. All-heads concatenated (overview)
    2. Per-head comparison (common vs disease vs individual heads)
    """
    print("\nGenerating manifold visualizations...")
    manifold_dir = ensure_dir(os.path.join(output_dir, "manifold"))

    # Extract latents from all image types
    print("  Extracting latents...")
    max_batches = max(1, max_samples // (3 * data_loader.batch_size))
    data = _extract_latents(model, vae_params, vae_batch_stats,
                            data_loader, max_batches=max_batches)

    print(f"  Extracted {len(data['labels'])} samples")

    methods = ['tsne', 'pca'] if method == 'both' else [method]

    for m in methods:
        print(f"\n  Running {m.upper()}...")

        _visualize_all_heads_manifold(
            data['common'], data['cardio'], data['effusion'],
            data['labels'], manifold_dir, method=m,
            use_wandb=use_wandb
        )

        _visualize_head_specific_manifolds(
            data['common'], data['cardio'], data['effusion'],
            data['labels'], manifold_dir, method=m,
            use_wandb=use_wandb
        )


# ============================================================================
# Quantitative Separability Analysis
# ============================================================================

def _fisher_discriminant_ratio(features, labels):
    """
    Compute Fisher's discriminant ratio (FDR) for multi-class data.

    FDR = trace(S_b) / trace(S_w)

    where S_b is the between-class scatter matrix and S_w is the within-class
    scatter matrix. Higher values mean better class separation.

    This is the classic criterion from Fisher's LDA -- it measures how spread
    apart the class means are relative to how spread each class is internally.
    """
    classes = np.unique(labels)
    n_features = features.shape[1]
    overall_mean = features.mean(axis=0)

    S_b = np.zeros((n_features, n_features))
    S_w = np.zeros((n_features, n_features))

    for c in classes:
        mask = labels == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        class_features = features[mask]
        class_mean = class_features.mean(axis=0)

        # Between-class scatter
        diff = (class_mean - overall_mean).reshape(-1, 1)
        S_b += n_c * (diff @ diff.T)

        # Within-class scatter
        centered = class_features - class_mean
        S_w += centered.T @ centered

    trace_sw = np.trace(S_w)
    if trace_sw < 1e-10:
        return float('inf')

    return float(np.trace(S_b) / trace_sw)


def _inter_intra_class_distances(features, labels):
    """
    Compute inter-class and intra-class distance statistics.

    Returns:
        inter_mean: Mean distance between class centroids
        intra_mean: Mean within-class distance to centroid
        ratio: inter_mean / intra_mean (higher = better separation)
    """
    classes = np.unique(labels)
    centroids = {}
    intra_dists = []

    for c in classes:
        mask = labels == c
        class_features = features[mask]
        centroid = class_features.mean(axis=0)
        centroids[c] = centroid

        # Intra-class: mean distance of samples to their centroid
        dists = np.linalg.norm(class_features - centroid, axis=1)
        intra_dists.append(dists.mean())

    # Inter-class: distances between all pairs of centroids
    inter_dists = []
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_dists.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))

    inter_mean = np.mean(inter_dists) if inter_dists else 0.0
    intra_mean = np.mean(intra_dists) if intra_dists else 1e-10
    ratio = inter_mean / max(intra_mean, 1e-10)

    return float(inter_mean), float(intra_mean), float(ratio)


def evaluate_separability(model, vae_params, vae_batch_stats,
                          data_loader, output_dir, max_samples=1000,
                          use_wandb=False):
    """
    Quantitative disease separability analysis.

    Computes per-head metrics to answer: does the SepVAE actually route
    disease information into the disease heads and keep it out of common?

    Metrics computed for each head subset:
    1. Fisher Discriminant Ratio (FDR): trace(S_b)/trace(S_w)
       - Higher = better class separation
    2. Inter/Intra class distance ratio
       - Higher = tighter clusters that are farther apart
    3. Silhouette score (in original feature space, not reduced)
       - Range [-1, 1], higher = better

    Produces a comparison bar chart and prints a summary table.
    """
    from sklearn.metrics import silhouette_score

    print("\nRunning quantitative separability analysis...")
    sep_dir = ensure_dir(os.path.join(output_dir, "separability"))

    # Extract latents
    max_batches = max(1, max_samples // (3 * data_loader.batch_size))
    data = _extract_latents(model, vae_params, vae_batch_stats,
                            data_loader, max_batches=max_batches)
    labels = data['labels']
    n_samples = len(labels)
    print(f"  Extracted {n_samples} samples")

    # Define feature subsets to compare
    feature_subsets = {
        'Common Head\n(should be LOW)': data['common'],
        'Disease Heads\n(should be HIGH)': np.concatenate(
            [data['cardio'], data['effusion']], axis=1),
        'Cardio Head': data['cardio'],
        'Effusion Head': data['effusion'],
        'All Heads\n(baseline)': np.concatenate(
            [data['common'], data['cardio'], data['effusion']], axis=1),
    }

    # Compute metrics for each subset
    metrics = {}
    print(f"\n  {'Subset':<28s} {'FDR':>8s} {'Inter/Intra':>12s} {'Silhouette':>11s} {'Dim':>4s}")
    print("  " + "-" * 67)

    for name, features in feature_subsets.items():
        short_name = name.replace('\n', ' ')

        fdr = _fisher_discriminant_ratio(features, labels)
        inter, intra, ratio = _inter_intra_class_distances(features, labels)

        try:
            sil = silhouette_score(features, labels)
        except ValueError:
            sil = 0.0

        metrics[name] = {
            'fisher_ratio': fdr,
            'inter_class_dist': inter,
            'intra_class_dist': intra,
            'distance_ratio': ratio,
            'silhouette': sil,
            'dim': features.shape[1],
        }

        print(f"  {short_name:<28s} {fdr:>8.3f} {ratio:>12.3f} {sil:>11.3f} {features.shape[1]:>4d}")

    # Key comparison: disease vs common
    disease_key = 'Disease Heads\n(should be HIGH)'
    common_key = 'Common Head\n(should be LOW)'

    disease_fdr = metrics[disease_key]['fisher_ratio']
    common_fdr = metrics[common_key]['fisher_ratio']
    disease_sil = metrics[disease_key]['silhouette']
    common_sil = metrics[common_key]['silhouette']

    print(f"\n  KEY RESULT: Separability Gap")
    print(f"    FDR gap (disease - common):       {disease_fdr - common_fdr:+.3f}")
    print(f"    Silhouette gap (disease - common): {disease_sil - common_sil:+.3f}")

    if disease_fdr > common_fdr * 2:
        print("    Disease heads have >2x the separability of common (good)")
    elif disease_fdr > common_fdr:
        print("    Disease heads are more separable than common (moderate)")
    else:
        print("    WARNING: Common head is as separable as disease heads")

    # Generate comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    short_names = [n.replace('\n', ' ') for n in feature_subsets.keys()]
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#A8E6CF', '#95A5A6']

    # FDR bars
    fdr_vals = [metrics[n]['fisher_ratio'] for n in feature_subsets.keys()]
    bars = axes[0].bar(range(len(short_names)), fdr_vals, color=colors)
    axes[0].set_xticks(range(len(short_names)))
    axes[0].set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('Fisher Discriminant Ratio')
    axes[0].set_title('Class Separability (FDR)\nHigher = Better Separation')
    for bar, val in zip(bars, fdr_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Distance ratio bars
    ratio_vals = [metrics[n]['distance_ratio'] for n in feature_subsets.keys()]
    bars = axes[1].bar(range(len(short_names)), ratio_vals, color=colors)
    axes[1].set_xticks(range(len(short_names)))
    axes[1].set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('Inter / Intra Class Distance')
    axes[1].set_title('Cluster Tightness Ratio\nHigher = Tighter & More Separated')
    for bar, val in zip(bars, ratio_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Silhouette bars
    sil_vals = [metrics[n]['silhouette'] for n in feature_subsets.keys()]
    bars = axes[2].bar(range(len(short_names)), sil_vals, color=colors)
    axes[2].set_xticks(range(len(short_names)))
    axes[2].set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Silhouette Score')
    axes[2].set_title('Silhouette Score [-1, 1]\nHigher = Better Clustering')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, sil_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Disease Separability: Common vs Disease Heads', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(sep_dir, "separability_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved separability comparison: {save_path}")

    if use_wandb and wandb:
        wandb.log({"eval/separability_chart": wandb.Image(save_path)})
        # Log key scalar metrics
        wandb.log({
            "eval/separability/fdr_disease": disease_fdr,
            "eval/separability/fdr_common": common_fdr,
            "eval/separability/fdr_gap": disease_fdr - common_fdr,
            "eval/separability/silhouette_disease": disease_sil,
            "eval/separability/silhouette_common": common_sil,
            "eval/separability/silhouette_gap": disease_sil - common_sil,
        })

    return metrics


# ============================================================================
# GradCAM
# ============================================================================

def evaluate_gradcam(model, vae_params, vae_batch_stats,
                     test_loader, output_dir, use_wandb=False):
    """
    Generate GradCAM visualizations.

    Takes one sample per disease class and shows spatial attention
    for each latent head (common, effusion, cardiomegaly).
    """
    from utils.sepvae_analysis import gradcam_sepvae

    print("\nGenerating GradCAM visualizations...")
    gradcam_dir = ensure_dir(os.path.join(output_dir, "gradcam"))

    # Get test batch and select one sample per class
    batch_torch = next(iter(test_loader))
    x_norm = jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy())
    x_dis1 = jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy())
    x_dis2 = jnp.array(batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy())

    test_images = jnp.concatenate([x_norm[:1], x_dis1[:1], x_dis2[:1]], axis=0)
    test_labels = [0, 1, 2]

    head_names = ['common', 'effusion', 'cardiomegaly']
    n_samples = 3
    n_heads = len(head_names)

    fig, axes = plt.subplots(n_samples, n_heads + 1, figsize=(4 * (n_heads + 1), 4 * n_samples))

    for i in range(n_samples):
        img = test_images[i]
        label = test_labels[i]
        img_np = np.array(img)

        # Normalize to [0,1] for display
        img_display = (img_np[:, :, 0] + 1.0) / 2.0

        # Original image
        axes[i, 0].imshow(img_display, cmap='gray')
        axes[i, 0].set_title(f"{DISEASE_NAMES[label]}", fontsize=12)
        axes[i, 0].axis('off')

        # GradCAM for each head
        for j, head in enumerate(head_names):
            heatmap = gradcam_sepvae(
                model, vae_params, img[None], label,
                target_head=head, batch_stats=vae_batch_stats
            )

            axes[i, j + 1].imshow(img_display, cmap='gray')
            axes[i, j + 1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[i, j + 1].set_title(f"{head} head", fontsize=12)
            axes[i, j + 1].axis('off')

    plt.suptitle("GradCAM: Spatial Attention per Head", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(gradcam_dir, "gradcam_grid.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved GradCAM grid: {save_path}")

    if use_wandb and wandb:
        wandb.log({"eval/gradcam": wandb.Image(save_path)})


# ============================================================================
# Summary
# ============================================================================

def print_summary(results_summary, output_dir):
    """Print consolidated evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    if 'linear_probe' in results_summary:
        probe = results_summary['linear_probe']
        print("\nLinear Probe Results:")
        for head_name, metrics in probe.items():
            print(f"  {head_name}: "
                  f"train={metrics['train_acc']:.3f}, "
                  f"test={metrics['test_acc']:.3f}")

    if 'separability' in results_summary:
        sep = results_summary['separability']
        print("\nSeparability Metrics:")
        for name, m in sep.items():
            short = name.replace('\n', ' ')
            print(f"  {short:<30s}  FDR={m['fisher_ratio']:.3f}  "
                  f"Dist={m['distance_ratio']:.3f}  "
                  f"Sil={m['silhouette']:.3f}")

    print("\nGenerated files:")
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), output_dir)
            print(f"  {rel}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Resolve --all flag
    if args.all:
        args.reconstruct = True
        args.linear_probe = True
        args.manifold = True
        args.gradcam = True
        args.separability = True

    if not any([args.reconstruct, args.linear_probe, args.manifold,
                args.gradcam, args.separability]):
        print("ERROR: No evaluation mode selected. Use --all or specify modes.")
        print("Available: --reconstruct, --linear_probe, --manifold, --gradcam, --all")
        return

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("SEPVAE EVALUATION & SAMPLING")
    print("=" * 60)

    # Setup output directory
    output_dir = ensure_dir(args.output_dir)

    # Initialize W&B
    use_wandb = args.wandb and _WANDB
    if use_wandb:
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        if args.wandb_run_id:
            init_kwargs['id'] = args.wandb_run_id
            init_kwargs['resume'] = 'allow'
        else:
            init_kwargs['name'] = f"eval-{Path(args.checkpoint).stem}"
        wandb.init(**init_kwargs)
        print("W&B initialized" + (f" (resumed run {args.wandb_run_id})" if args.wandb_run_id else ""))
    elif args.wandb and not _WANDB:
        print("WARNING: W&B requested but not installed, skipping")

    # Load checkpoint
    print("\n" + "=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)
    model, vae_params, vae_batch_stats, ckpt_meta = load_checkpoint(
        args.checkpoint,
        z_channels_common=args.z_channels_common,
        z_channels_disease=args.z_channels_disease,
        use_fpn=args.use_fpn,
        fpn_channels=args.fpn_channels,
        unfreeze_from=args.unfreeze_from,
    )

    if ckpt_meta.get('args'):
        ckpt_args = ckpt_meta['args']
        print(f"  Training config: z_common={ckpt_args.get('z_channels_common')}, "
              f"z_disease={ckpt_args.get('z_channels_disease')}, "
              f"epochs={ckpt_args.get('epochs')}")

    # Build data loaders
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    train_loader, test_loader = build_data_loaders(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Run evaluations
    results_summary = {}

    if args.reconstruct:
        print("\n" + "=" * 60)
        print("RECONSTRUCTION EVALUATION")
        print("=" * 60)
        evaluate_reconstruction(
            model, vae_params, vae_batch_stats,
            test_loader, output_dir,
            n_per_class=args.n_samples_per_class,
            use_wandb=use_wandb
        )

    if args.linear_probe:
        print("\n" + "=" * 60)
        print("LINEAR PROBE EVALUATION")
        print("=" * 60)
        probe_results = evaluate_linear_probe(
            model, vae_params, vae_batch_stats,
            train_loader, test_loader, output_dir,
            use_wandb=use_wandb
        )
        results_summary['linear_probe'] = probe_results

    if args.manifold:
        print("\n" + "=" * 60)
        print("MANIFOLD VISUALIZATION")
        print("=" * 60)
        evaluate_manifold(
            model, vae_params, vae_batch_stats,
            test_loader, output_dir,
            method=args.manifold_method,
            max_samples=args.manifold_max_samples,
            use_wandb=use_wandb
        )

    if args.separability:
        print("\n" + "=" * 60)
        print("SEPARABILITY ANALYSIS")
        print("=" * 60)
        sep_results = evaluate_separability(
            model, vae_params, vae_batch_stats,
            test_loader, output_dir,
            max_samples=args.manifold_max_samples,
            use_wandb=use_wandb
        )
        results_summary['separability'] = sep_results

    if args.gradcam:
        print("\n" + "=" * 60)
        print("GRADCAM VISUALIZATION")
        print("=" * 60)
        evaluate_gradcam(
            model, vae_params, vae_batch_stats,
            test_loader, output_dir,
            use_wandb=use_wandb
        )

    # Print summary
    print_summary(results_summary, output_dir)

    if use_wandb:
        wandb.finish()

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
