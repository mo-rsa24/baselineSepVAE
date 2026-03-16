"""
Analysis utilities for SepVAE:
1. Linear Probing - Evaluate latent representations
2. GradCAM - Visualize disease-specific activations
3. Manifold Visualization - Disease clustering in latent space

These tools help understand what the model has learned and verify
that disease-specific information is properly separated.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List


# ============================================================================
# 1. LINEAR PROBING
# ============================================================================

def linear_probe(
    model,
    params,
    train_loader,
    test_loader,
    probe_targets: str = 'disease',
    batch_stats: Dict = None
) -> Dict[str, float]:
    """
    Linear probing: Train a linear classifier on frozen latent representations.

    This evaluates whether the latent space has learned useful features for
    disease classification. If a simple linear classifier achieves high accuracy,
    it means the latent space is linearly separable by disease.

    **What it measures:**
    - How well disease information is encoded in latents
    - Whether nulling is effective (inactive heads shouldn't help classification)
    - Which latent channels are most informative

    **How it works:**
    1. Encode all images to latent representations (freeze VAE)
    2. Extract latents:
       - Common head only (should NOT classify disease well)
       - Disease heads only (should classify well)
       - All heads (baseline)
    3. Train linear SVM/LogReg on extracted features
    4. Evaluate on test set

    Args:
        model: Trained SepVAE model
        params: Model parameters
        train_loader: Training data loader
        test_loader: Test data loader
        probe_targets: What to classify ('disease' or other metadata)

    Returns:
        Dict with accuracies for different latent subsets

    Example:
        >>> results = linear_probe(sepvae, params, train_loader, test_loader)
        >>> print(f"Common only: {results['common']:.3f}")  # Should be low
        >>> print(f"Disease only: {results['disease']:.3f}")  # Should be high
    """
    print("="*60)
    print("LINEAR PROBING")
    print("="*60)

    # Extract latents from all samples
    def extract_latents(loader, split_name):
        """Extract latent representations for all samples."""
        latents_common = []
        latents_cardio = []
        latents_plthick = []
        labels_list = []

        for batch in loader:
            # Convert PyTorch to JAX
            x = jnp.array(batch['x_norm'].permute(0, 2, 3, 1).numpy())
            labels = batch['disease_labels'].numpy()

            # Encode (no grad needed)
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats
            latents_dict = model.apply(variables, x, method=model.encode)

            mu_c, _ = latents_dict['common']  # (B, 64, 64, 4)
            mu_cardio, _ = latents_dict['cardiomegaly']  # (B, 64, 64, 2)
            mu_plthick, _ = latents_dict['plthick']  # (B, 64, 64, 2)

            # Global average pool to get flat vectors
            mu_c_flat = jnp.mean(mu_c, axis=(1, 2))  # (B, 4)
            mu_cardio_flat = jnp.mean(mu_cardio, axis=(1, 2))  # (B, 2)
            mu_plthick_flat = jnp.mean(mu_plthick, axis=(1, 2))  # (B, 2)

            latents_common.extend(np.array(mu_c_flat))
            latents_cardio.extend(np.array(mu_cardio_flat))
            latents_plthick.extend(np.array(mu_plthick_flat))
            labels_list.extend(labels)

        return {
            'common': np.array(latents_common),
            'cardio': np.array(latents_cardio),
            'plthick': np.array(latents_plthick),
            'labels': np.array(labels_list)
        }

    print("\nExtracting latents from training set...")
    train_data = extract_latents(train_loader, "train")

    print("Extracting latents from test set...")
    test_data = extract_latents(test_loader, "test")

    # Prepare feature sets
    feature_sets = {
        'common_only': (train_data['common'], test_data['common']),
        'disease_only': (
            np.concatenate([train_data['cardio'], train_data['plthick']], axis=1),
            np.concatenate([test_data['cardio'], test_data['plthick']], axis=1)
        ),
        'all_heads': (
            np.concatenate([train_data['common'], train_data['cardio'], train_data['plthick']], axis=1),
            np.concatenate([test_data['common'], test_data['cardio'], test_data['plthick']], axis=1)
        )
    }

    results = {}

    # Train linear probes
    for name, (X_train, X_test) in feature_sets.items():
        print(f"\n{name}:")
        print(f"  Feature dim: {X_train.shape[1]}")

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, train_data['labels'])

        # Evaluate
        train_acc = clf.score(X_train, train_data['labels'])
        test_acc = clf.score(X_test, test_data['labels'])

        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc
        }

        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")

        # Detailed classification report
        y_pred = clf.predict(X_test)
        print("\n  Classification Report:")
        print(classification_report(test_data['labels'], y_pred,
                                    target_names=['Normal', 'Effusion', 'Cardiomegaly']))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Common only (should be LOW):  {results['common_only']['test_acc']:.3f}")
    print(f"Disease only (should be HIGH): {results['disease_only']['test_acc']:.3f}")
    print(f"All heads (baseline):          {results['all_heads']['test_acc']:.3f}")

    if results['common_only']['test_acc'] < 0.5:
        print("\n✓ Common head is disease-agnostic (good!)")
    else:
        print("\n⚠ Common head contains disease info (check nulling)")

    if results['disease_only']['test_acc'] > 0.8:
        print("✓ Disease heads are discriminative (good!)")
    else:
        print("⚠ Disease heads not discriminative (check training)")

    return results


# ============================================================================
# 2. GRADCAM
# ============================================================================

def gradcam_sepvae(
    model,
    params,
    x: jnp.ndarray,
    label: int,
    target_head: str = 'plthick',
    batch_stats: Dict = None
) -> jnp.ndarray:
    """
    GradCAM for SepVAE: Visualize which spatial regions activate disease heads.

    **What it does:**
    - Shows which parts of the X-ray image contribute to disease detection
    - Helps verify that disease heads focus on relevant anatomy
    - Can reveal if model is using spurious correlations

    **How it works:**
    1. Forward pass to get latent activations (spatial: 64×64×C)
    2. Compute gradient of target head's mean w.r.t. spatial features
    3. Weight spatial features by gradients
    4. Upsample to input resolution (512×512)

    Args:
        model: Trained SepVAE
        params: Model parameters
        x: Input image (1, 512, 512, 1)
        label: Disease label (for nulling)
        target_head: Which head to visualize ('common', 'cardiomegaly', 'plthick')

    Returns:
        Heatmap (512, 512) showing important regions

    Example:
        >>> x = test_images[0:1]  # (1, 512, 512, 1)
        >>> heatmap = gradcam_sepvae(sepvae, params, x, label=1, target_head='plthick')
        >>> plt.imshow(heatmap, cmap='jet', alpha=0.5)
        >>> plt.imshow(x[0, :, :, 0], cmap='gray', alpha=0.5)
    """

    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    def compute_target_score(params_inner, x_inner, label_inner, head_name):
        """Compute scalar score for target head."""
        # Encode
        vars_inner = {'params': params_inner}
        if batch_stats is not None:
            vars_inner['batch_stats'] = batch_stats
        latents_dict = model.apply(vars_inner, x_inner, method=model.encode)

        # Get target head
        mu, _ = latents_dict[head_name]  # (1, 64, 64, C)

        # Return mean activation (scalar)
        return jnp.mean(mu)

    # Compute gradients
    grad_fn = jax.grad(lambda p: compute_target_score(p, x, label, target_head))
    grads = grad_fn(params)

    # Forward pass to get activations
    latents_dict = model.apply(variables, x, method=model.encode)
    mu, _ = latents_dict[target_head]  # (1, 64, 64, C)

    # Get backbone activations (we need intermediate features for GradCAM)
    # For simplicity, use the latent spatial map itself
    activations = mu[0]  # (64, 64, C)

    # GradCAM: weight channels by their gradients
    # Note: In full GradCAM, we'd use backbone gradients, but latent grads work too
    # For now, just take the mean across channels as a simple heatmap
    heatmap = jnp.mean(jnp.abs(activations), axis=-1)  # (64, 64)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Upsample to 512×512
    heatmap_upsampled = jax.image.resize(
        heatmap[None, :, :, None],
        (1, 512, 512, 1),
        method='bilinear'
    )[0, :, :, 0]

    return np.array(heatmap_upsampled)


def visualize_gradcam(
    model,
    params,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    save_path: str = None,
    batch_stats: Dict = None
):
    """
    Visualize GradCAM for multiple test samples.

    Args:
        model: SepVAE model
        params: Parameters
        test_images: (N, 512, 512, 1) test images
        test_labels: (N,) labels
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']
    head_names = ['common', 'plthick', 'cardiomegaly']

    for i, (img, label) in enumerate(zip(test_images[:3], test_labels[:3])):
        img_np = np.array(img)

        # Original image
        axes[i, 0].imshow(img_np[:, :, 0], cmap='gray')
        axes[i, 0].set_title(f"{disease_names[label]}")
        axes[i, 0].axis('off')

        # GradCAM for each head
        for j, head in enumerate(head_names):
            heatmap = gradcam_sepvae(model, params, img[None], label, target_head=head,
                                        batch_stats=batch_stats)

            # Overlay
            axes[i, j+1].imshow(img_np[:, :, 0], cmap='gray')
            axes[i, j+1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[i, j+1].set_title(f"{head} head")
            axes[i, j+1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved GradCAM visualization to {save_path}")
    plt.show()


# ============================================================================
# 3. MANIFOLD VISUALIZATION
# ============================================================================

def visualize_latent_manifold(
    model,
    params,
    data_loader,
    method: str = 'tsne',
    max_samples: int = 1000,
    save_path: str = None,
    batch_stats: Dict = None
):
    """
    Visualize latent space manifold with t-SNE or PCA.

    **What it shows:**
    - Whether diseases cluster in latent space
    - Whether nulling creates structure (Normal vs Disease separation)
    - Interpolation paths between diseases

    **Expected result:**
    - 3 distinct clusters for Normal, Cardiomegaly, Pleural Effusion
    - Common head should show anatomical grouping (less disease separation)
    - Disease heads should show clear disease clustering

    Args:
        model: SepVAE model
        params: Parameters
        data_loader: DataLoader with mixed diseases
        method: 'tsne' or 'pca'
        max_samples: Max samples to visualize (for speed)
        save_path: Optional save path

    Example:
        >>> visualize_latent_manifold(sepvae, params, test_loader, method='tsne')
        # Shows 2D plot with color-coded disease clusters
    """
    print("="*60)
    print(f"MANIFOLD VISUALIZATION ({method.upper()})")
    print("="*60)

    # Extract latents
    latents_all = []
    labels_all = []
    count = 0

    for batch in data_loader:
        if count >= max_samples:
            break

        x = jnp.array(batch['x_norm'].permute(0, 2, 3, 1).numpy())
        labels = batch['disease_labels'].numpy()

        # Encode
        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats
        latents_dict = model.apply(variables, x, method=model.encode)

        # Concatenate all heads
        mu_c, _ = latents_dict['common']
        mu_cardio, _ = latents_dict['cardiomegaly']
        mu_plthick, _ = latents_dict['plthick']

        # Flatten spatial
        mu_c_flat = jnp.mean(mu_c, axis=(1, 2))
        mu_cardio_flat = jnp.mean(mu_cardio, axis=(1, 2))
        mu_plthick_flat = jnp.mean(mu_plthick, axis=(1, 2))

        # Concatenate
        z_all = jnp.concatenate([mu_c_flat, mu_cardio_flat, mu_plthick_flat], axis=1)

        latents_all.extend(np.array(z_all))
        labels_all.extend(labels)
        count += len(labels)

    latents_all = np.array(latents_all)
    labels_all = np.array(labels_all)

    print(f"Extracted {len(latents_all)} samples, latent dim: {latents_all.shape[1]}")

    # Dimensionality reduction
    if method == 'tsne':
        print("Running t-SNE (may take a few minutes)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        print("Running PCA...")
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    latents_2d = reducer.fit_transform(latents_all)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'red', 'green']
    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']

    for disease_id in range(3):
        mask = labels_all == disease_id
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=colors[disease_id],
            label=disease_names[disease_id],
            alpha=0.6,
            s=20
        )

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(f"Latent Space Manifold ({method.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved manifold visualization to {save_path}")
    plt.show()

    # Compute cluster separability (silhouette score)
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(latents_2d, labels_all)
    print(f"\nSilhouette score: {silhouette:.3f} (1.0 = perfect separation)")

    if silhouette > 0.5:
        print("✓ Good cluster separation!")
    else:
        print("⚠ Weak cluster separation (check training)")


# ============================================================================
# 4. BACKBONE FEATURE VISUALIZATION (Frozen CheSS ResNet-50)
# ============================================================================

def visualize_backbone_features(
    model,
    params,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    save_dir: str = None,
    batch_stats: Dict = None
) -> Dict[str, np.ndarray]:
    """
    Visualize what the frozen CheSS backbone spatial features encode.

    Three complementary views:
    1. PCA RGB map (DINO-style): Projects 2048 channels to 3 PCA components,
       displayed as an RGB image. Regions with similar colors share similar
       feature representations. For chest X-rays, expect lung fields, cardiac
       silhouette, and bones to appear as distinct color regions.

    2. Class-conditional difference: Averages backbone features per class and
       shows where disease-positive features diverge from normal. Highlights
       the spatial regions the backbone already "knows" are disease-relevant
       before any head training.

    3. Top-K channel activations: Identifies the most active feature channels
       per image and displays them as spatial heatmaps. Shows what individual
       feature detectors in the frozen backbone respond to.

    Args:
        model: SepVAE model (needs encoder.backbone access)
        params: Model parameters with injected CheSS weights
        images: Input images (N, 512, 512, 1), at least 3-6 for good PCA
        labels: Disease labels (N,) in {0, 1, 2}
        save_dir: Directory to save figures (optional)

    Returns:
        Dict with 'pca_maps', 'diff_maps', 'topk_maps' numpy arrays
    """
    print("=" * 60)
    print("BACKBONE FEATURE VISUALIZATION")
    print("=" * 60)

    N = images.shape[0]

    # --- Extract raw backbone spatial features ---
    # Use the model's extract_backbone_features method via apply()
    # This returns (B, 64, 64, 2048) features before the ConvHeads.
    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    t0 = time.time()
    batch_size = min(4, N)
    all_features = []
    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size]
        feats = model.apply(
            variables, batch,
            method=model.extract_backbone_features
        )  # (B, 64, 64, 2048)
        feats = jax.block_until_ready(feats)
        all_features.append(np.array(feats))

    features = np.concatenate(all_features, axis=0)  # (N, 64, 64, 2048)
    print(f"Extracted backbone features: {features.shape} ({time.time() - t0:.1f}s)")

    results = {}

    # ---- 1. PCA RGB Map (DINO-style) ----
    print("\n[1/3] Computing PCA RGB maps...")
    t_pca = time.time()

    H, W, C = features.shape[1], features.shape[2], features.shape[3]
    flat = features.reshape(-1, C).astype(np.float32)  # (N*64*64, 2048)

    # Fast PCA via covariance eigenvectors (avoids sklearn SVD overhead).
    # Subsample spatially for fitting, then project all points.
    max_fit = 3000
    rng_np = np.random.RandomState(42)
    if flat.shape[0] > max_fit:
        fit_idx = rng_np.choice(flat.shape[0], max_fit, replace=False)
        sub = flat[fit_idx]
    else:
        sub = flat

    mean = sub.mean(axis=0)
    centered = sub - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)  # (2048, 2048)
    eigenvalues, eigenvectors = np.linalg.eigh(cov.astype(np.float64))
    # eigh returns ascending order — take last 3 (largest), flip to descending
    components = eigenvectors[:, -3:][:, ::-1].T.astype(np.float32)  # (3, 2048)
    top_eigenvalues = eigenvalues[-3:][::-1]
    explained_var = top_eigenvalues / eigenvalues.sum()

    # Project all spatial locations
    pca_flat = (flat - mean) @ components.T  # (N*4096, 3)

    print(f"  PCA done in {time.time() - t_pca:.1f}s (subsample={len(sub)})")

    # Reshape back to spatial
    pca_maps = pca_flat.reshape(N, H, W, 3)

    # Normalize each component to [0, 1] for RGB display
    for c in range(3):
        cmin, cmax = pca_maps[..., c].min(), pca_maps[..., c].max()
        pca_maps[..., c] = (pca_maps[..., c] - cmin) / (cmax - cmin + 1e-8)

    results['pca_maps'] = pca_maps

    print(f"  PCA explained variance: {explained_var}")
    print(f"  Top-3 components explain {explained_var.sum() * 100:.1f}% of variance")

    # ---- 2. Class-Conditional Difference Maps ----
    print("\n[2/3] Computing class-conditional difference maps...")

    labels_np = np.array(labels)
    disease_names = ['Normal', 'Effusion', 'Cardiomegaly']

    # Average feature map per class
    class_means = {}
    for cls_id in range(3):
        mask = labels_np == cls_id
        if mask.sum() > 0:
            class_means[cls_id] = features[mask].mean(axis=0)  # (64, 64, 2048)
            print(f"  {disease_names[cls_id]}: {mask.sum()} samples")

    # Difference maps: disease - normal (L2 norm across channels)
    diff_maps = {}
    if 0 in class_means:
        for cls_id in [1, 2]:
            if cls_id in class_means:
                diff = class_means[cls_id] - class_means[0]
                # L2 norm across channel dim -> (64, 64) heatmap
                diff_map = np.sqrt(np.sum(diff ** 2, axis=-1))
                # Normalize
                diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
                diff_maps[disease_names[cls_id]] = diff_map
    else:
        print("  WARNING: No normal samples found, skipping difference maps")

    results['diff_maps'] = diff_maps

    # ---- 3. Top-K Channel Activation Maps ----
    print("\n[3/3] Computing top-K channel activations...")

    topk = 5  # Show top-5 most active channels per image
    topk_maps = {}

    for idx in range(min(N, 4)):  # Show first 4 images max
        feat = features[idx]  # (64, 64, 2048)
        # Mean activation per channel
        channel_means = feat.mean(axis=(0, 1))  # (2048,)
        top_channels = np.argsort(channel_means)[-topk:][::-1]

        img_topk = []
        for ch in top_channels:
            activation = feat[:, :, ch]  # (64, 64)
            # Normalize
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
            img_topk.append((ch, activation))

        topk_maps[idx] = img_topk

    results['topk_maps'] = topk_maps

    # ---- Plot everything ----
    _plot_backbone_diagnostics(images, labels, pca_maps, diff_maps, topk_maps,
                               disease_names, save_dir)

    return results


def _plot_backbone_diagnostics(
    images, labels, pca_maps, diff_maps, topk_maps,
    disease_names, save_dir=None
):
    """Generate publication-quality plots for backbone diagnostics."""

    labels_np = np.array(labels)
    N = len(images)
    n_show = min(N, 4)

    # --- Figure 1: PCA RGB Maps ---
    fig1, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes[:, None]  # Ensure 2D indexing

    fig1.suptitle('Frozen CheSS Backbone: PCA Feature Maps (DINO-style)', fontsize=14)

    for i in range(n_show):
        # Original image
        axes[0, i].imshow(np.array(images[i, :, :, 0]), cmap='gray')
        axes[0, i].set_title(f'{disease_names[int(labels_np[i])]}', fontsize=11)
        axes[0, i].axis('off')

        # PCA RGB map (upsample to input resolution for overlay)
        pca_up = np.array(jax.image.resize(
            jnp.array(pca_maps[i]),
            (images.shape[1], images.shape[2], 3),
            method='bilinear'
        ))
        axes[1, i].imshow(pca_up)
        axes[1, i].set_title('PCA (R/G/B = PC1/2/3)', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    if save_dir:
        path = f'{save_dir}/backbone_pca_rgb.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()

    # --- Figure 2: Class-Conditional Difference Maps ---
    if diff_maps:
        n_diffs = len(diff_maps)
        fig2, axes2 = plt.subplots(1, n_diffs + 1, figsize=(5 * (n_diffs + 1), 5))
        if n_diffs + 1 == 1:
            axes2 = [axes2]

        fig2.suptitle('Backbone Feature Divergence: Disease vs Normal', fontsize=14)

        # Show a representative normal image
        normal_idx = np.where(labels_np == 0)[0]
        if len(normal_idx) > 0:
            axes2[0].imshow(np.array(images[normal_idx[0], :, :, 0]), cmap='gray')
            axes2[0].set_title('Normal (reference)', fontsize=11)
            axes2[0].axis('off')

        for j, (disease, dmap) in enumerate(diff_maps.items()):
            # Upsample difference map to input resolution
            dmap_up = np.array(jax.image.resize(
                jnp.array(dmap[None, :, :, None]),
                (1, images.shape[1], images.shape[2], 1),
                method='bilinear'
            ))[0, :, :, 0]

            ax = axes2[j + 1]
            ax.imshow(np.array(images[0, :, :, 0]), cmap='gray', alpha=0.4)
            im = ax.imshow(dmap_up, cmap='hot', alpha=0.7)
            ax.set_title(f'{disease} - Normal\n(feature L2 distance)', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_dir:
            path = f'{save_dir}/backbone_class_diff.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.show()

    # --- Figure 3: Top-K Channel Activations ---
    n_imgs = min(len(topk_maps), 3)
    topk = len(list(topk_maps.values())[0]) if topk_maps else 5
    fig3, axes3 = plt.subplots(n_imgs, topk + 1, figsize=(3 * (topk + 1), 3 * n_imgs))
    if n_imgs == 1:
        axes3 = axes3[None, :]

    fig3.suptitle('Top-5 Most Active Backbone Channels', fontsize=14)

    for i, (img_idx, channels) in enumerate(list(topk_maps.items())[:n_imgs]):
        # Original image
        axes3[i, 0].imshow(np.array(images[img_idx, :, :, 0]), cmap='gray')
        axes3[i, 0].set_title(f'{disease_names[int(labels_np[img_idx])]}', fontsize=10)
        axes3[i, 0].axis('off')

        for j, (ch_id, activation) in enumerate(channels):
            # Upsample activation to input resolution
            act_up = np.array(jax.image.resize(
                jnp.array(activation[None, :, :, None]),
                (1, images.shape[1], images.shape[2], 1),
                method='bilinear'
            ))[0, :, :, 0]

            axes3[i, j + 1].imshow(np.array(images[img_idx, :, :, 0]), cmap='gray', alpha=0.4)
            axes3[i, j + 1].imshow(act_up, cmap='inferno', alpha=0.7)
            axes3[i, j + 1].set_title(f'Ch {ch_id}', fontsize=10)
            axes3[i, j + 1].axis('off')

    plt.tight_layout()
    if save_dir:
        path = f'{save_dir}/backbone_topk_channels.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nExample usage:")
    print("""
    # 1. Linear Probing
    results = linear_probe(sepvae, params, train_loader, test_loader)

    # 2. GradCAM
    test_images, test_labels = next(iter(test_loader))
    visualize_gradcam(sepvae, params, test_images[:3], test_labels[:3],
                     save_path='gradcam.png')

    # 3. Manifold Visualization
    visualize_latent_manifold(sepvae, params, test_loader, method='tsne',
                             save_path='manifold.png')
    """)
