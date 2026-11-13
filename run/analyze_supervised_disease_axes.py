"""
Supervised 2D disease-axis analysis for SepVAE checkpoints.

This script builds explicit, interpretable latent axes for:
- Pleural Effusion
- Cardiomegaly

Pipeline:
1. Extract pooled disease features [z_cardiomegaly, z_effusion] from a checkpoint.
2. Fit two one-vs-rest linear models (binary logistic regression):
   - effusion vs non-effusion
   - cardiomegaly vs non-cardiomegaly
3. Use model weights as axis directions, then orthogonalize the cardiomegaly
   axis against the effusion axis with Gram-Schmidt.
4. Project samples onto the 2D axis system.
5. Report held-out AUC and leakage ratios for each axis.
6. Optionally compute additional embeddings (t-SNE / UMAP / LDA / MDS).
7. Save plots and metrics JSON.

Usage:
    python -m run.analyze_supervised_disease_axes \
        --checkpoint runs_sepvae/exp/checkpoints/checkpoint_epoch0050.pkl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from flax.serialization import msgpack_restore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
from models.sep_vae_jax import SepVAE


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def load_checkpoint(ckpt_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load a SepVAE checkpoint.

    Returns:
        vae_params: model parameters
        vae_batch_stats: BatchNorm statistics (possibly empty dict)
        ckpt_args: checkpoint args dict
    """
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = msgpack_restore(f.read())

    vae_params = jax.tree_util.tree_map(jnp.array, ckpt["vae_params"])
    raw_bs = ckpt.get("vae_batch_stats", {})
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, raw_bs) if raw_bs else {}
    ckpt_args = ckpt.get("args", {})

    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")
    return vae_params, vae_batch_stats, ckpt_args


def create_model_from_args(ckpt_args: Dict, override_upsample: str = None) -> SepVAE:
    """
    Create SepVAE model from checkpoint metadata.
    """
    upsample_method = override_upsample or ckpt_args.get("upsample_method", "nearest")
    return SepVAE(
        z_channels_common=ckpt_args.get("z_channels_common", 4),
        z_channels_disease=ckpt_args.get("z_channels_disease", 2),
        frozen_backbone=ckpt_args.get("frozen_backbone", True),
        use_fpn=ckpt_args.get("use_fpn", False),
        fpn_channels=ckpt_args.get("fpn_channels", 512),
        unfreeze_from=ckpt_args.get("unfreeze_from", None),
        sigma_inactive=ckpt_args.get("sigma_inactive", 1.0),
        upsample_method=upsample_method,
    )


def collect_pooled_disease_features(
    model: SepVAE,
    vae_params: Dict,
    vae_batch_stats: Dict,
    loader: DataLoader,
    max_samples: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pooled disease features from images.

    Feature vector per image:
        [mean(mu_cardiomegaly), mean(mu_effusion)]  -> shape (2*C_disease,)

    Returns:
        X: (N, D) feature matrix
        y: (N,) labels in {0: Normal, 1: Effusion, 2: Cardiomegaly}
    """
    variables = {"params": vae_params}
    if vae_batch_stats:
        variables["batch_stats"] = vae_batch_stats

    features_all = []
    labels_all = []

    for batch_torch in loader:
        if len(labels_all) >= max_samples:
            break

        for img_key, label_val in [("x_norm", 0), ("x_disease1", 1), ("x_disease2", 2)]:
            x = jnp.array(batch_torch[img_key].permute(0, 2, 3, 1).numpy())
            latents_dict = model.apply(variables, x, method=model.encode)

            mu_cardio, _ = latents_dict["cardiomegaly"]
            mu_effusion, _ = latents_dict["effusion"]

            z_cardio = np.array(jnp.mean(mu_cardio, axis=(1, 2)))
            z_effusion = np.array(jnp.mean(mu_effusion, axis=(1, 2)))
            x_feat = np.concatenate([z_cardio, z_effusion], axis=1)

            remaining = max_samples - len(labels_all)
            if remaining <= 0:
                break

            x_feat = x_feat[:remaining]
            features_all.extend(x_feat)
            labels_all.extend([label_val] * len(x_feat))

            if len(labels_all) >= max_samples:
                break

    X = np.asarray(features_all, dtype=np.float64)
    y = np.asarray(labels_all, dtype=np.int32)
    return X, y


def _safe_unit_vector(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    """Normalize a vector to unit length. Returns None if near-zero."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def gram_schmidt_orthogonalize(
    v_target: np.ndarray, v_anchor: np.ndarray
) -> Optional[np.ndarray]:
    """
    Orthogonalize v_target against v_anchor using Gram-Schmidt.
    Returns None if either input is degenerate.
    """
    u_anchor = _safe_unit_vector(v_anchor)
    if u_anchor is None:
        return _safe_unit_vector(v_target)

    u_target = _safe_unit_vector(v_target)
    if u_target is None:
        return None

    v_proj = v_target - np.dot(v_target, u_anchor) * u_anchor

    if np.linalg.norm(v_proj) < 1e-10:
        # Fallback: pick a basis vector least aligned with anchor, then orthogonalize.
        eye = np.eye(len(u_anchor), dtype=np.float64)
        basis_scores = np.abs(eye @ u_anchor)
        idx = int(np.argmin(basis_scores))
        candidate = eye[idx]
        v_proj = candidate - np.dot(candidate, u_anchor) * u_anchor

    return _safe_unit_vector(v_proj)


def _orient_axis_positive_for_target(
    axis: np.ndarray,
    scores: np.ndarray,
    y_binary: np.ndarray,
) -> np.ndarray:
    """
    Flip axis sign if needed so target class has larger mean score than non-target.
    """
    pos_mask = y_binary == 1
    neg_mask = y_binary == 0
    if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
        return axis

    if float(np.mean(scores[pos_mask])) < float(np.mean(scores[neg_mask])):
        return -axis
    return axis


def _safe_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC with guardrails."""
    if np.sum(y_binary == 1) == 0 or np.sum(y_binary == 0) == 0:
        return float("nan")
    try:
        return float(roc_auc_score(y_binary, scores))
    except ValueError:
        return float("nan")


def _leakage_ratios(
    scores: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    off_label: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute leakage ratios for one axis.

    cross_disease_ratio:
        mean(|score| on off-label disease) / mean(|score| on target disease)

    normal_ratio:
        mean(|score| on normal) / mean(|score| on target disease)
    """
    target_vals = np.abs(scores[labels == target_label])
    off_vals = np.abs(scores[labels == off_label])
    normal_vals = np.abs(scores[labels == 0])

    if len(target_vals) == 0:
        return {
            "cross_disease_ratio": float("nan"),
            "normal_ratio": float("nan"),
            "target_abs_mean": float("nan"),
            "off_abs_mean": float("nan"),
            "normal_abs_mean": float("nan"),
        }

    t_mean = float(np.mean(target_vals))
    o_mean = float(np.mean(off_vals)) if len(off_vals) > 0 else float("nan")
    n_mean = float(np.mean(normal_vals)) if len(normal_vals) > 0 else float("nan")

    cross_ratio = o_mean / (t_mean + eps) if np.isfinite(o_mean) else float("nan")
    normal_ratio = n_mean / (t_mean + eps) if np.isfinite(n_mean) else float("nan")

    return {
        "cross_disease_ratio": float(cross_ratio),
        "normal_ratio": float(normal_ratio),
        "target_abs_mean": t_mean,
        "off_abs_mean": o_mean,
        "normal_abs_mean": n_mean,
    }


def fit_axes_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    seed: int = 0,
    l2_inv_strength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Fit supervised disease axes, orthogonalize, project, and compute metrics.

    Returns:
        coords_all: (N, 2) projected coordinates [effusion_axis, cardiomegaly_axis]
        y_all: (N,) labels
        axis_matrix: (2, D), rows are [effusion_axis, cardiomegaly_axis_orth]
        origin: (D,) projection origin (train mean)
        metrics: evaluation metrics
    """
    if X.ndim != 2 or len(X) < 12:
        raise ValueError("Need at least 12 samples for supervised axis analysis.")
    if len(np.unique(y)) < 3:
        raise ValueError("Need all three labels {0,1,2} in the extracted samples.")

    # Stratified split to evaluate on held-out samples.
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    y_eff_train = (y_train == 1).astype(np.int32)
    y_card_train = (y_train == 2).astype(np.int32)

    eff_model = LogisticRegression(
        penalty="l2",
        C=l2_inv_strength,
        max_iter=2000,
        random_state=seed + 11,
        class_weight="balanced",
    )
    card_model = LogisticRegression(
        penalty="l2",
        C=l2_inv_strength,
        max_iter=2000,
        random_state=seed + 22,
        class_weight="balanced",
    )

    eff_model.fit(X_train, y_eff_train)
    card_model.fit(X_train, y_card_train)

    w_eff = np.asarray(eff_model.coef_[0], dtype=np.float64)
    w_card = np.asarray(card_model.coef_[0], dtype=np.float64)

    axis_eff = _safe_unit_vector(w_eff)
    axis_card = gram_schmidt_orthogonalize(w_card, w_eff)

    # Fallback to PCA if supervised axes are degenerate (disease heads not yet active).
    pca_fallback = False
    if axis_eff is None or axis_card is None:
        pca_fallback = True
        print("  WARNING: Supervised axes degenerate (near-zero logistic weights).")
        print("           Falling back to PCA on pooled disease features.")
        from sklearn.decomposition import PCA as _PCA
        _pca = _PCA(n_components=2, random_state=seed)
        _pca.fit(X_train)
        axis_eff = _pca.components_[0].astype(np.float64)
        axis_card = _pca.components_[1].astype(np.float64)

    # Use train mean as origin for stable projections.
    origin = np.mean(X_train, axis=0)

    train_eff_scores = (X_train - origin) @ axis_eff
    axis_eff = _orient_axis_positive_for_target(axis_eff, train_eff_scores, y_eff_train)

    train_card_scores = (X_train - origin) @ axis_card
    axis_card = _orient_axis_positive_for_target(axis_card, train_card_scores, y_card_train)

    coords_all = np.column_stack([(X - origin) @ axis_eff, (X - origin) @ axis_card])
    coords_test = coords_all[test_idx]

    eff_scores_test = coords_test[:, 0]
    card_scores_test = coords_test[:, 1]

    y_eff_test = (y_test == 1).astype(np.int32)
    y_card_test = (y_test == 2).astype(np.int32)

    auc_eff = _safe_auc(y_eff_test, eff_scores_test)
    auc_card = _safe_auc(y_card_test, card_scores_test)

    leak_eff = _leakage_ratios(
        scores=eff_scores_test,
        labels=y_test,
        target_label=1,  # effusion target
        off_label=2,     # cardiomegaly leakage
    )
    leak_card = _leakage_ratios(
        scores=card_scores_test,
        labels=y_test,
        target_label=2,  # cardiomegaly target
        off_label=1,     # effusion leakage
    )

    axis_dot = float(np.dot(axis_eff, axis_card))
    axis_matrix = np.stack([axis_eff, axis_card], axis=0)

    metrics = {
        "n_samples_total": int(len(X)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "auc/effusion_axis_vs_non_effusion": auc_eff,
        "auc/cardiomegaly_axis_vs_non_cardiomegaly": auc_card,
        "leakage/effusion_axis_cross_disease_ratio": leak_eff["cross_disease_ratio"],
        "leakage/effusion_axis_normal_ratio": leak_eff["normal_ratio"],
        "leakage/cardiomegaly_axis_cross_disease_ratio": leak_card["cross_disease_ratio"],
        "leakage/cardiomegaly_axis_normal_ratio": leak_card["normal_ratio"],
        "axis/orthogonality_dot": axis_dot,
        "axis/effusion_target_abs_mean": leak_eff["target_abs_mean"],
        "axis/effusion_off_abs_mean": leak_eff["off_abs_mean"],
        "axis/cardiomegaly_target_abs_mean": leak_card["target_abs_mean"],
        "axis/cardiomegaly_off_abs_mean": leak_card["off_abs_mean"],
        "axis/origin_norm": float(np.linalg.norm(origin)),
        "axis/pca_fallback": pca_fallback,
    }

    # Extra class means in projected 2D space.
    for class_id, name in [(0, "normal"), (1, "effusion"), (2, "cardiomegaly")]:
        mask = y == class_id
        if np.sum(mask) > 0:
            class_mean = np.mean(coords_all[mask], axis=0)
            metrics[f"class_mean/{name}_eff_axis"] = float(class_mean[0])
            metrics[f"class_mean/{name}_cardio_axis"] = float(class_mean[1])

    # Keep train/test coords for plotting diagnostics.
    metrics["_train_indices"] = train_idx.tolist()
    metrics["_test_indices"] = test_idx.tolist()
    return coords_all, y, axis_matrix, origin, metrics


def save_projection_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    metrics: Dict[str, float],
) -> None:
    """
    Save 2D scatter in the supervised disease-axis coordinate system.
    """
    names = {0: "Normal", 1: "Effusion", 2: "Cardiomegaly"}
    colors = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
    markers = {0: "o", 1: "^", 2: "s"}

    plt.figure(figsize=(10, 8))

    for class_id in [0, 1, 2]:
        mask = labels == class_id
        if np.sum(mask) == 0:
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=20,
            alpha=0.65,
            c=colors[class_id],
            marker=markers[class_id],
            label=f"{names[class_id]} (n={int(np.sum(mask))})",
        )

    # Class centroids.
    for class_id in [0, 1, 2]:
        mask = labels == class_id
        if np.sum(mask) == 0:
            continue
        cxy = np.mean(coords[mask], axis=0)
        plt.scatter(
            [cxy[0]],
            [cxy[1]],
            s=140,
            c=colors[class_id],
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)

    is_fallback = metrics.get("axis/pca_fallback", False)
    method_label = "PCA FALLBACK (supervised axes degenerate)" if is_fallback else "Supervised Disease Axes (Orthogonalized)"
    title = (
        f"{method_label}\n"
        f"AUC eff={metrics['auc/effusion_axis_vs_non_effusion']:.3f} | "
        f"AUC cardio={metrics['auc/cardiomegaly_axis_vs_non_cardiomegaly']:.3f} | "
        f"dot={metrics['axis/orthogonality_dot']:.4f}"
    )
    plt.title(title)
    if is_fallback:
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
    else:
        plt.xlabel("Effusion axis score")
        plt.ylabel("Cardiomegaly axis score (Gram-Schmidt orthogonalized)")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def _parse_embedding_methods(methods_csv: str):
    """Parse comma-separated embedding method names."""
    if methods_csv is None:
        return []
    allowed = {"tsne", "umap", "lda", "mds"}
    out = []
    for raw in methods_csv.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name in allowed and name not in out:
            out.append(name)
    return out


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified subsample for expensive embeddings.

    Returns:
        X_sub, y_sub, idx_sub
    """
    n = len(X)
    if max_samples <= 0 or n <= max_samples:
        idx = np.arange(n, dtype=np.int32)
        return X, y, idx

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    idx_sub, _ = next(splitter.split(X, y))
    idx_sub = np.asarray(idx_sub, dtype=np.int32)
    return X[idx_sub], y[idx_sub], idx_sub


def _compute_embedding_2d(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    tsne_perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    mds_max_iter: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Compute a 2D embedding.

    Returns:
        coords_2d or None, warning_message or None
    """
    n = len(X)
    if n < 4:
        return None, "not enough samples (need >=4)."

    if method == "tsne":
        if tsne_perplexity > 0:
            perplexity = float(tsne_perplexity)
        else:
            perplexity = float(min(30, max(5, n // 4)))
        perplexity = min(perplexity, float(max(2, n - 1)))
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(X), None

    if method == "umap":
        try:
            import umap.umap_ as umap
        except ImportError:
            return None, "umap-learn not installed; skipping UMAP."

        n_neighbors = int(min(max(2, umap_n_neighbors), max(2, n - 1)))
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            metric="euclidean",
        )
        return reducer.fit_transform(X), None

    if method == "lda":
        n_classes = len(np.unique(y))
        if n_classes < 2:
            return None, "not enough classes for LDA."
        lda = LinearDiscriminantAnalysis(n_components=2)
        coords = lda.fit_transform(X, y)
        if coords.ndim != 2:
            return None, "unexpected LDA output shape."
        if coords.shape[1] == 1:
            # With degenerate class geometry, LDA can collapse to 1D.
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
        return coords[:, :2], None

    if method == "mds":
        reducer = MDS(
            n_components=2,
            random_state=seed,
            n_init=4,
            max_iter=int(mds_max_iter),
            dissimilarity="euclidean",
        )
        return reducer.fit_transform(X), None

    return None, f"unknown method '{method}'."


def _plot_embedding_axes(ax, coords: np.ndarray, labels: np.ndarray, title: str):
    """Scatter a single embedding view on the provided axis."""
    names = {0: "Normal", 1: "Effusion", 2: "Cardiomegaly"}
    colors = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
    markers = {0: "o", 1: "^", 2: "s"}

    for class_id in [0, 1, 2]:
        mask = labels == class_id
        if np.sum(mask) == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            alpha=0.65,
            c=colors[class_id],
            marker=markers[class_id],
            label=names[class_id],
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.grid(alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def run_additional_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    methods_csv: str,
    seed: int,
    max_samples: int,
    tsne_perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    mds_max_iter: int,
) -> Dict[str, float]:
    """
    Compute and save additional 2D embeddings (t-SNE/UMAP/LDA/MDS).
    """
    methods = _parse_embedding_methods(methods_csv)
    metrics = {}
    if len(methods) == 0:
        return metrics

    results = []

    for i, method in enumerate(methods):
        method_seed = seed + 1000 + i
        use_subsample = method in {"tsne", "umap", "mds"}
        X_work, y_work, idx_work = _stratified_subsample(
            X,
            y,
            max_samples=max_samples if use_subsample else 0,
            seed=method_seed,
        )

        coords, warn_msg = _compute_embedding_2d(
            method=method,
            X=X_work,
            y=y_work,
            seed=method_seed,
            tsne_perplexity=tsne_perplexity,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            mds_max_iter=mds_max_iter,
        )

        if warn_msg is not None:
            print(f"  [{method.upper()}] {warn_msg}")
            metrics[f"embed/{method}/status"] = warn_msg
            continue

        if coords is None or coords.shape[1] < 2:
            msg = "embedding failed or returned invalid shape."
            print(f"  [{method.upper()}] {msg}")
            metrics[f"embed/{method}/status"] = msg
            continue

        # Quality summary.
        if len(np.unique(y_work)) >= 2 and len(y_work) >= 4:
            try:
                sil = float(silhouette_score(coords[:, :2], y_work))
            except ValueError:
                sil = float("nan")
        else:
            sil = float("nan")

        metrics[f"embed/{method}/silhouette"] = sil
        metrics[f"embed/{method}/n_samples"] = int(len(y_work))
        metrics[f"embed/{method}/status"] = "ok"

        # Save per-method coordinates (+labels and original indices).
        coords_path = os.path.join(output_dir, f"embedding_{method}_coords.npz")
        np.savez_compressed(
            coords_path,
            coords=np.asarray(coords[:, :2], dtype=np.float32),
            labels=np.asarray(y_work, dtype=np.int32),
            source_indices=np.asarray(idx_work, dtype=np.int32),
        )
        print(
            f"  [{method.upper()}] saved {coords_path} "
            f"(n={len(y_work)}, silhouette={sil:.3f})"
        )

        title = f"{method.upper()} (n={len(y_work)})"
        if np.isfinite(sil):
            title += f"\nSilhouette={sil:.3f}"
        results.append((method, coords[:, :2], y_work, title))

    if len(results) == 0:
        return metrics

    # Render a combined comparison figure.
    n_plots = len(results)
    n_cols = min(2, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 5.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, (_, coords, labels, title) in enumerate(results):
        _plot_embedding_axes(axes_flat[idx], coords, labels, title)

    # Hide unused panels.
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Single legend on first axis.
    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels_leg, loc="best")

    plt.tight_layout()
    panel_path = os.path.join(output_dir, "additional_embeddings_comparison.png")
    plt.savefig(panel_path, dpi=180)
    plt.close(fig)
    print(f"  Saved combined embedding panel: {panel_path}")
    metrics["embed/comparison_panel_path"] = panel_path

    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Supervised 2D Disease Axis Analysis for SepVAE")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SepVAE checkpoint (.pkl)")

    parser.add_argument("--dicom_dir", type=str, default="/datasets/mmolefe/vinbigdata/train",
                        help="Path to DICOM files")
    parser.add_argument("--csv_path", type=str, default="/datasets/mmolefe/vinbigdata/train.csv",
                        help="Path to train.csv")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size (must match checkpoint)")
    parser.add_argument("--exclude_cross_disease_overlap", action="store_true",
                        help="Exclude samples where cardiomegaly and effusion co-occur in disease pools")

    parser.add_argument("--batch_size", type=int, default=8, help="Triplet batch size for latent extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=1200, help="Maximum image samples to analyze")
    parser.add_argument("--train_frac", type=float, default=0.7,
                        help="Train fraction for fitting linear axes (rest used for evaluation)")
    parser.add_argument("--l2_inv_strength", type=float, default=1.0,
                        help="Inverse L2 regularization strength C for logistic regression")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--upsample_method", type=str, default=None,
                        choices=["bilinear", "subpixel", "nearest"],
                        help="Optional decoder upsample override for old checkpoints")

    parser.add_argument(
        "--embedding_methods",
        type=str,
        default="tsne,umap,lda,mds",
        help=(
            "Comma-separated extra embeddings to run on pooled features. "
            "Choices: tsne,umap,lda,mds. Use empty string to disable."
        ),
    )
    parser.add_argument(
        "--embedding_max_samples",
        type=int,
        default=500,
        help="Max samples for expensive methods (t-SNE/UMAP/MDS). 0 uses all samples.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=0.0,
        help="t-SNE perplexity. <=0 uses automatic value.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=20,
        help="UMAP neighborhood size.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--mds_max_iter",
        type=int,
        default=300,
        help="Maximum MDS optimization iterations.",
    )

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint_dir/supervised_axes_analysis)")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 68)
    print("SUPERVISED 2D DISEASE AXIS ANALYSIS")
    print("=" * 68)

    if not (0.5 <= args.train_frac < 1.0):
        raise ValueError("--train_frac must be in [0.5, 1.0).")

    vae_params, vae_batch_stats, ckpt_args = load_checkpoint(args.checkpoint)
    model = create_model_from_args(ckpt_args, override_upsample=args.upsample_method)

    actual_upsample = args.upsample_method or ckpt_args.get("upsample_method", "nearest")
    print("\nModel config:")
    print(f"  z_channels_common: {ckpt_args.get('z_channels_common', 4)}")
    print(f"  z_channels_disease: {ckpt_args.get('z_channels_disease', 2)}")
    print(f"  use_fpn: {ckpt_args.get('use_fpn', False)}")
    print(f"  upsample_method: {actual_upsample}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir = Path(args.checkpoint).parent.parent
        output_dir = str(ckpt_dir / "supervised_axes_analysis")
    ensure_dir(output_dir)

    print("\nLoading dataset...")
    dataset = VinBigDataTripletDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size,
        exclude_cross_disease_overlap=args.exclude_cross_disease_overlap,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=jax_collate_fn,
        drop_last=False,
    )

    print("\nExtracting pooled disease latents...")
    X, y = collect_pooled_disease_features(
        model=model,
        vae_params=vae_params,
        vae_batch_stats=vae_batch_stats,
        loader=loader,
        max_samples=args.max_samples,
    )
    print(f"  Extracted features: X={X.shape}, y={y.shape}")

    coords, labels, axis_matrix, origin, metrics = fit_axes_and_evaluate(
        X=X,
        y=y,
        train_frac=args.train_frac,
        seed=args.seed,
        l2_inv_strength=args.l2_inv_strength,
    )

    # Save plot
    plot_path = os.path.join(output_dir, "supervised_disease_axes_scatter.png")
    save_projection_plot(coords, labels, plot_path, metrics)

    # Optional extra embeddings (t-SNE / UMAP / LDA / MDS) on pooled features.
    if args.embedding_methods.strip():
        print("\nComputing additional embeddings...")
        extra_metrics = run_additional_embeddings(
            X=X,
            y=y,
            output_dir=output_dir,
            methods_csv=args.embedding_methods,
            seed=args.seed,
            max_samples=args.embedding_max_samples,
            tsne_perplexity=args.tsne_perplexity,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            mds_max_iter=args.mds_max_iter,
        )
        metrics.update(extra_metrics)
    else:
        print("\nSkipping additional embeddings (embedding_methods is empty).")

    # Save lightweight axis vectors and metrics
    metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
    axis_payload = {
        "axis_effusion": axis_matrix[0].tolist(),
        "axis_cardiomegaly_orth": axis_matrix[1].tolist(),
        "projection_origin": origin.tolist(),
        "metrics": metrics_to_save,
    }

    metrics_path = os.path.join(output_dir, "supervised_disease_axes_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(axis_payload, f, indent=2)

    print("\nResults:")
    print(f"  Plot: {plot_path}")
    print(f"  Metrics: {metrics_path}")
    panel_path = metrics_to_save.get("embed/comparison_panel_path")
    if panel_path:
        print(f"  Extra embeddings panel: {panel_path}")
    if metrics.get("axis/pca_fallback", False):
        print("\n  ** PCA FALLBACK: Supervised logistic axes were degenerate (near-zero weights).")
        print("     Disease heads are likely not yet active. Axes show PCA of pooled features.")
    print("\nKey metrics (held-out):")
    print(f"  AUC effusion axis: {metrics_to_save['auc/effusion_axis_vs_non_effusion']:.4f}")
    print(f"  AUC cardiomegaly axis: {metrics_to_save['auc/cardiomegaly_axis_vs_non_cardiomegaly']:.4f}")
    print(f"  Leakage effusion axis (cardio/target): "
          f"{metrics_to_save['leakage/effusion_axis_cross_disease_ratio']:.4f}")
    print(f"  Leakage cardiomegaly axis (eff/target): "
          f"{metrics_to_save['leakage/cardiomegaly_axis_cross_disease_ratio']:.4f}")
    print(f"  Axis orthogonality dot: {metrics_to_save['axis/orthogonality_dot']:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
