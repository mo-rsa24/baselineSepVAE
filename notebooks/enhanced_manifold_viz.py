#!/usr/bin/env python3
"""
Enhanced Manifold Visualizations for Hypothesis Testing

Additional visualization capabilities beyond basic projections:
1. Confidence Ellipses - Show cluster spread and orientation
2. Convex Hulls - Visualize cluster boundaries
3. Interpolation Paths - Linear vs manifold interpolation
4. Density Contours - Where do samples concentrate?
5. Cluster Metrics - Silhouette, Davies-Bouldin, separation ratios
6. Distance Matrices - Pairwise centroid relationships
7. Variance Explained - PCA component analysis
8. Decision Boundaries - Where would a classifier separate clusters?

Usage:
    from enhanced_manifold_viz import EnhancedManifoldVisualizer
    viz = EnhancedManifoldVisualizer(projections, condition_indices, condition_keys)
    viz.create_all_visualizations(output_dir)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ---------------------------------------------------------------------------
# Condition Styles (matching manifold_diagnostics.py)
# ---------------------------------------------------------------------------
CONDITION_STYLES = {
    "mono_dog": {
        "label": "Base: A Dog",
        "short": "Dog",
        "color": "#3498db",
        "marker": "circle",
        "is_base": True,
        "group": "base",
    },
    "mono_cat": {
        "label": "Base: A Cat",
        "short": "Cat",
        "color": "#e67e22",
        "marker": "circle",
        "is_base": True,
        "group": "base",
    },
    "mono_dog_and_cat": {
        "label": "CLIP AND: Dog+Cat",
        "short": "CLIP AND",
        "color": "#2ecc71",
        "marker": "diamond",
        "is_base": False,
        "group": "clip",
    },
    "mono_spatial": {
        "label": "CLIP AND: Spatial",
        "short": "CLIP Spatial",
        "color": "#9b59b6",
        "marker": "diamond",
        "is_base": False,
        "group": "clip",
    },
    "superdiff_semantic": {
        "label": "SuperDiff AND: Dog+Cat",
        "short": "SD AND",
        "color": "#e74c3c",
        "marker": "star",
        "is_base": False,
        "group": "superdiff",
    },
    "superdiff_spatial": {
        "label": "SuperDiff AND: Spatial",
        "short": "SD Spatial",
        "color": "#1abc9c",
        "marker": "star",
        "is_base": False,
        "group": "superdiff",
    },
}

# Marker mapping for Plotly
PLOTLY_MARKERS = {
    "circle": "circle",
    "diamond": "diamond",
    "star": "star",
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if len(x) < 3:
        return None

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def compute_convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
    """Compute convex hull vertices for a set of 2D points."""
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        # Return hull vertices in order, closing the polygon
        hull_points = points[hull.vertices]
        return np.vstack([hull_points, hull_points[0]])
    except Exception:
        return None


def compute_cluster_metrics(projections: np.ndarray,
                           condition_indices: List[int]) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    labels = np.array(condition_indices)

    # Need at least 2 clusters with at least 2 samples each
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {}

    label_counts = [np.sum(labels == l) for l in unique_labels]
    if min(label_counts) < 2:
        return {}

    metrics = {}

    # Silhouette score (-1 to 1, higher is better)
    try:
        metrics["silhouette"] = silhouette_score(projections, labels)
        metrics["silhouette_samples"] = silhouette_samples(projections, labels)
    except Exception:
        pass

    # Davies-Bouldin index (lower is better, 0 is perfect)
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(projections, labels)
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Enhanced Visualizer Class
# ---------------------------------------------------------------------------
class EnhancedManifoldVisualizer:
    """Enhanced visualization tools for latent manifold analysis."""

    def __init__(self,
                 projections: Dict[str, Dict[str, np.ndarray]],
                 condition_indices: List[int],
                 condition_keys: List[str],
                 all_latents: np.ndarray = None):
        """
        Initialize visualizer.

        Parameters
        ----------
        projections : Dict[str, Dict[str, np.ndarray]]
            Dictionary with keys like "PCA", "t-SNE", "UMAP"
            Each contains "2d" and "3d" projections
        condition_indices : List[int]
            Index of condition for each sample
        condition_keys : List[str]
            Ordered list of condition keys
        all_latents : np.ndarray, optional
            Original high-dimensional latents
        """
        self.projections = projections
        self.condition_indices = np.array(condition_indices)
        self.condition_keys = condition_keys
        self.all_latents = all_latents
        self.key_to_idx = {k: i for i, k in enumerate(condition_keys)}

    def _compute_centroids(self, proj: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute centroids for each condition."""
        centroids = {}
        for cond_idx in range(len(self.condition_keys)):
            mask = self.condition_indices == cond_idx
            if mask.sum() > 0:
                centroids[cond_idx] = proj[mask].mean(axis=0)
        return centroids

    # -----------------------------------------------------------------------
    # 1. Confidence Ellipses Visualization
    # -----------------------------------------------------------------------
    def create_confidence_ellipse_plot(self, output_path: Path, n_std: float = 2.0):
        """Create 2D projections with confidence ellipses showing cluster spread."""
        print("\n  Creating confidence ellipse visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        methods = ["PCA", "t-SNE", "UMAP"]

        for ax, method in zip(axes, methods):
            proj = self.projections[method]["2d"]

            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                points = proj[mask]

                if len(points) < 3:
                    continue

                color = style.get("color", "#999999")
                label = style.get("short", key)

                # Plot points
                ax.scatter(points[:, 0], points[:, 1],
                          c=color, s=40, alpha=0.4,
                          edgecolors='white', linewidths=0.5,
                          label=label)

                # Draw confidence ellipse
                confidence_ellipse(points[:, 0], points[:, 1], ax,
                                  n_std=n_std, edgecolor=color,
                                  linewidth=2, linestyle='--', alpha=0.8)

                # Mark centroid
                centroid = points.mean(axis=0)
                marker = 'D' if not style.get("is_base") else 'o'
                ax.scatter(centroid[0], centroid[1],
                          c=color, s=150, marker=marker,
                          edgecolors='black', linewidths=2, zorder=10)

            ax.set_xlabel(f"{method} 1", fontsize=11)
            ax.set_ylabel(f"{method} 2", fontsize=11)
            ax.set_title(f"{method} with {n_std}σ Confidence Ellipses",
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6,
                  bbox_to_anchor=(0.5, -0.02), fontsize=9)

        plt.suptitle("Cluster Spread Analysis: Confidence Ellipses",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        save_path = output_path / "confidence_ellipses.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

    # -----------------------------------------------------------------------
    # 2. Convex Hulls Visualization
    # -----------------------------------------------------------------------
    def create_convex_hull_plot(self, output_path: Path):
        """Create 2D projections with convex hulls showing cluster boundaries."""
        print("\n  Creating convex hull visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        methods = ["PCA", "t-SNE", "UMAP"]

        for ax, method in zip(axes, methods):
            proj = self.projections[method]["2d"]

            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                points = proj[mask]

                if len(points) < 3:
                    continue

                color = style.get("color", "#999999")
                label = style.get("short", key)

                # Plot points
                ax.scatter(points[:, 0], points[:, 1],
                          c=color, s=40, alpha=0.5,
                          edgecolors='white', linewidths=0.5,
                          label=label)

                # Draw convex hull
                hull_vertices = compute_convex_hull(points)
                if hull_vertices is not None:
                    ax.fill(hull_vertices[:, 0], hull_vertices[:, 1],
                           alpha=0.15, color=color)
                    ax.plot(hull_vertices[:, 0], hull_vertices[:, 1],
                           color=color, linewidth=2, alpha=0.8)

            ax.set_xlabel(f"{method} 1", fontsize=11)
            ax.set_ylabel(f"{method} 2", fontsize=11)
            ax.set_title(f"{method} with Convex Hulls", fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6,
                  bbox_to_anchor=(0.5, -0.02), fontsize=9)

        plt.suptitle("Cluster Boundaries: Convex Hulls",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        save_path = output_path / "convex_hulls.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

    # -----------------------------------------------------------------------
    # 3. Interactive Plot with Full Annotations
    # -----------------------------------------------------------------------
    def create_annotated_interactive_plot(self, output_path: Path):
        """Create fully annotated interactive Plotly visualization."""
        print("\n  Creating annotated interactive visualization...")

        methods = ["PCA", "t-SNE", "UMAP"]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"<b>{m}</b><br><sup>Latent Space Projection</sup>"
                for m in methods
            ],
            horizontal_spacing=0.08,
        )

        for col_idx, method in enumerate(methods, start=1):
            proj = self.projections[method]["2d"]
            centroids = self._compute_centroids(proj)

            # Get base concept centroids for geometric annotations
            dog_idx = self.key_to_idx.get("mono_dog")
            cat_idx = self.key_to_idx.get("mono_cat")
            dog_cent = centroids.get(dog_idx)
            cat_cent = centroids.get(cat_idx)

            # 1. Draw base axis (dotted line)
            if dog_cent is not None and cat_cent is not None:
                midpoint = (dog_cent + cat_cent) / 2

                fig.add_trace(
                    go.Scatter(
                        x=[dog_cent[0], cat_cent[0]],
                        y=[dog_cent[1], cat_cent[1]],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0.4)", width=3, dash="dot"),
                        name="Base Concept Axis" if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        legendgroup="axis",
                        hoverinfo="skip",
                    ),
                    row=1, col=col_idx
                )

                # Midpoint marker
                fig.add_trace(
                    go.Scatter(
                        x=[midpoint[0]], y=[midpoint[1]],
                        mode="markers+text",
                        marker=dict(size=16, color="gray", symbol="x",
                                   line=dict(width=2, color="black")),
                        text=["Linear<br>Midpoint"],
                        textposition="top center",
                        textfont=dict(size=9, color="gray"),
                        name="Linear Midpoint" if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        legendgroup="midpoint",
                        hovertemplate="<b>Linear Midpoint</b><br>Halfway between Dog and Cat<extra></extra>",
                    ),
                    row=1, col=col_idx
                )

            # 2. Plot each condition with proper styling
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                points = proj[mask]

                if len(points) == 0:
                    continue

                color = style.get("color", "#999999")
                label = style.get("label", key)
                short = style.get("short", key)
                marker_type = PLOTLY_MARKERS.get(style.get("marker", "circle"), "circle")
                is_base = style.get("is_base", False)
                group = style.get("group", "other")

                # Sample points
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0], y=points[:, 1],
                        mode="markers",
                        marker=dict(
                            size=10 if is_base else 8,
                            color=color,
                            opacity=0.6,
                            symbol=marker_type,
                            line=dict(width=1, color="white"),
                        ),
                        name=label if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        legendgroup=key,
                        legendgrouptitle_text=group.upper() if col_idx == 1 else None,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            f"x: %{{x:.3f}}<br>"
                            f"y: %{{y:.3f}}<br>"
                            f"<extra>Sample point</extra>"
                        ),
                    ),
                    row=1, col=col_idx
                )

                # Centroid with label
                if cond_idx in centroids:
                    cent = centroids[cond_idx]

                    # Centroid marker (larger, with border)
                    fig.add_trace(
                        go.Scatter(
                            x=[cent[0]], y=[cent[1]],
                            mode="markers+text",
                            marker=dict(
                                size=18 if is_base else 20,
                                color=color,
                                symbol=marker_type,
                                line=dict(width=3, color="black"),
                            ),
                            text=[short],
                            textposition="top center",
                            textfont=dict(size=10, color=color, family="Arial Black"),
                            showlegend=False,
                            legendgroup=key,
                            hovertemplate=(
                                f"<b>{label} CENTROID</b><br>"
                                f"x: %{{x:.3f}}<br>"
                                f"y: %{{y:.3f}}<br>"
                                f"<extra>Cluster center</extra>"
                            ),
                        ),
                        row=1, col=col_idx
                    )

                    # Draw dashed line from midpoint to composition centroids
                    if not is_base and dog_cent is not None and cat_cent is not None:
                        midpoint = (dog_cent + cat_cent) / 2
                        dist_to_mid = np.linalg.norm(cent - midpoint)

                        fig.add_trace(
                            go.Scatter(
                                x=[midpoint[0], cent[0]],
                                y=[midpoint[1], cent[1]],
                                mode="lines",
                                line=dict(color=color, width=2, dash="dash"),
                                showlegend=False,
                                hovertemplate=f"Distance to midpoint: {dist_to_mid:.3f}<extra></extra>",
                            ),
                            row=1, col=col_idx
                        )

            # Add axis labels
            fig.update_xaxes(title_text=f"{method} Component 1", row=1, col=col_idx)
            fig.update_yaxes(title_text=f"{method} Component 2", row=1, col=col_idx)

        # Layout
        fig.update_layout(
            title=dict(
                text=(
                    "<b>Latent Space Geometry: CLIP AND vs SuperDiff AND</b><br>"
                    "<sup>Interactive exploration of semantic composition in latent manifold</sup>"
                ),
                font=dict(size=16),
                x=0.5,
            ),
            width=1900,
            height=700,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=10),
                groupclick="toggleitem",
            ),
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=11),
        )

        # Add annotation explaining the plot
        fig.add_annotation(
            text=(
                "● Base concepts (Dog, Cat) define the semantic axis<br>"
                "◆ CLIP AND uses monolithic prompt composition<br>"
                "★ SuperDiff AND uses score-based composition<br>"
                "✕ Linear midpoint = simple averaging"
            ),
            xref="paper", yref="paper",
            x=0.01, y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left",
        )

        save_path = output_path / "projections_annotated_interactive.html"
        fig.write_html(str(save_path), include_plotlyjs=True, full_html=True)
        print(f"    Saved: {save_path}")

    # -----------------------------------------------------------------------
    # 4. Cluster Separation Metrics
    # -----------------------------------------------------------------------
    def create_cluster_metrics_visualization(self, output_path: Path):
        """Visualize cluster separation quality metrics."""
        print("\n  Creating cluster metrics visualization...")

        methods = ["PCA", "t-SNE", "UMAP"]
        all_metrics = {}

        for method in methods:
            proj = self.projections[method]["2d"]
            metrics = compute_cluster_metrics(proj, self.condition_indices)
            all_metrics[method] = metrics

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Silhouette scores comparison
        ax = axes[0, 0]
        sil_scores = [all_metrics[m].get("silhouette", 0) for m in methods]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(methods, sil_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel("Silhouette Score", fontsize=11)
        ax.set_title("Cluster Separation Quality\n(Higher = Better Separation)",
                    fontsize=12, fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3, axis='y')

        # Add value labels
        for bar, score in zip(bars, sil_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        # 2. Davies-Bouldin index
        ax = axes[0, 1]
        db_scores = [all_metrics[m].get("davies_bouldin", 0) for m in methods]
        bars = ax.bar(methods, db_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Davies-Bouldin Index", fontsize=11)
        ax.set_title("Cluster Compactness\n(Lower = More Compact Clusters)",
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        for bar, score in zip(bars, db_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. Per-condition silhouette (PCA)
        ax = axes[1, 0]
        pca_sil_samples = all_metrics["PCA"].get("silhouette_samples", np.array([]))
        if len(pca_sil_samples) > 0:
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                sil_vals = pca_sil_samples[mask]

                if len(sil_vals) > 0:
                    color = style.get("color", "#999999")
                    short = style.get("short", key)
                    positions = np.ones(len(sil_vals)) * cond_idx
                    ax.scatter(positions + np.random.normal(0, 0.1, len(sil_vals)),
                              sil_vals, c=color, s=30, alpha=0.6, label=short)
                    ax.scatter([cond_idx], [np.mean(sil_vals)],
                              c=color, s=150, marker='D', edgecolors='black',
                              linewidths=2, zorder=10)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_xticks(range(len(self.condition_keys)))
            ax.set_xticklabels([CONDITION_STYLES.get(k, {}).get("short", k)
                               for k in self.condition_keys], rotation=30, ha='right')
            ax.set_ylabel("Silhouette Score", fontsize=11)
            ax.set_title("Per-Sample Silhouette (PCA)\n(Higher = Better Cluster Fit)",
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

        # 4. Inter-cluster distance matrix (PCA)
        ax = axes[1, 1]
        proj = self.projections["PCA"]["2d"]
        centroids = self._compute_centroids(proj)
        n_conditions = len(self.condition_keys)

        dist_matrix = np.zeros((n_conditions, n_conditions))
        for i in range(n_conditions):
            for j in range(n_conditions):
                if i in centroids and j in centroids:
                    dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n_conditions))
        ax.set_yticks(range(n_conditions))
        short_labels = [CONDITION_STYLES.get(k, {}).get("short", k)
                       for k in self.condition_keys]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_title("Centroid Distance Matrix (PCA)", fontsize=12, fontweight='bold')

        # Add distance values
        for i in range(n_conditions):
            for j in range(n_conditions):
                text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if dist_matrix[i, j] > dist_matrix.max()/2 else 'black')

        plt.colorbar(im, ax=ax, label='Distance')

        plt.suptitle("Cluster Quality Metrics Across Projection Methods",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "cluster_metrics.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

        return all_metrics

    # -----------------------------------------------------------------------
    # 5. Density Contours
    # -----------------------------------------------------------------------
    def create_density_contour_plot(self, output_path: Path):
        """Create density contour visualization."""
        print("\n  Creating density contour visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        methods = ["PCA", "t-SNE", "UMAP"]

        # Row 1: Individual condition densities
        # Row 2: Comparison (CLIP AND vs SuperDiff AND)

        for col_idx, method in enumerate(methods):
            proj = self.projections[method]["2d"]

            # Top row: All conditions
            ax = axes[0, col_idx]

            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                points = proj[mask]

                if len(points) < 3:
                    continue

                color = style.get("color", "#999999")

                # Plot points
                ax.scatter(points[:, 0], points[:, 1],
                          c=color, s=30, alpha=0.3)

            ax.set_xlabel(f"{method} 1", fontsize=10)
            ax.set_ylabel(f"{method} 2", fontsize=10)
            ax.set_title(f"{method}: All Conditions", fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)

            # Bottom row: CLIP AND vs SuperDiff AND comparison
            ax = axes[1, col_idx]

            # Get semantic AND conditions
            clip_key = "mono_dog_and_cat"
            sd_key = "superdiff_semantic"

            for key, label in [(clip_key, "CLIP AND"), (sd_key, "SuperDiff AND")]:
                style = CONDITION_STYLES.get(key, {})
                idx = self.key_to_idx.get(key)
                if idx is None:
                    continue

                mask = self.condition_indices == idx
                points = proj[mask]

                if len(points) < 3:
                    continue

                color = style.get("color", "#999999")

                # KDE contours
                try:
                    # Create grid
                    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
                    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])

                    kde = gaussian_kde(points.T)
                    f = np.reshape(kde(positions).T, xx.shape)

                    # Plot contours
                    ax.contour(xx, yy, f, levels=5, colors=[color], alpha=0.7)
                    ax.contourf(xx, yy, f, levels=5, colors=[color], alpha=0.2)
                except Exception:
                    pass

                # Plot points
                ax.scatter(points[:, 0], points[:, 1],
                          c=color, s=50, alpha=0.7, label=label,
                          edgecolors='white', linewidths=0.5)

            ax.set_xlabel(f"{method} 1", fontsize=10)
            ax.set_ylabel(f"{method} 2", fontsize=10)
            ax.set_title(f"{method}: CLIP AND vs SuperDiff AND Density",
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)

        plt.suptitle("Density Analysis: Where Do Samples Concentrate?",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "density_contours.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

    # -----------------------------------------------------------------------
    # 6. Interpolation Path Analysis
    # -----------------------------------------------------------------------
    def create_interpolation_analysis(self, output_path: Path, n_interp: int = 10):
        """Analyze linear interpolation paths between conditions."""
        print("\n  Creating interpolation path analysis...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        methods = ["PCA", "t-SNE", "UMAP"]

        for ax, method in zip(axes, methods):
            proj = self.projections[method]["2d"]
            centroids = self._compute_centroids(proj)

            # Plot all points faded
            for cond_idx, key in enumerate(self.condition_keys):
                style = CONDITION_STYLES.get(key, {})
                mask = self.condition_indices == cond_idx
                points = proj[mask]
                color = style.get("color", "#999999")
                ax.scatter(points[:, 0], points[:, 1],
                          c=color, s=20, alpha=0.2)

            # Get key centroids
            dog_idx = self.key_to_idx.get("mono_dog")
            cat_idx = self.key_to_idx.get("mono_cat")
            clip_idx = self.key_to_idx.get("mono_dog_and_cat")
            sd_idx = self.key_to_idx.get("superdiff_semantic")

            dog_cent = centroids.get(dog_idx)
            cat_cent = centroids.get(cat_idx)
            clip_cent = centroids.get(clip_idx)
            sd_cent = centroids.get(sd_idx)

            if dog_cent is not None and cat_cent is not None:
                # Linear interpolation path
                alphas = np.linspace(0, 1, n_interp)
                interp_path = np.array([
                    (1 - a) * dog_cent + a * cat_cent for a in alphas
                ])

                # Plot interpolation path
                ax.plot(interp_path[:, 0], interp_path[:, 1],
                       'k--', linewidth=2, alpha=0.7, label="Linear Interp")
                ax.scatter(interp_path[:, 0], interp_path[:, 1],
                          c=alphas, cmap='coolwarm', s=50, zorder=5,
                          edgecolors='black', linewidths=1)

                # Mark endpoints
                ax.scatter([dog_cent[0]], [dog_cent[1]],
                          c=CONDITION_STYLES["mono_dog"]["color"],
                          s=200, marker='o', edgecolors='black',
                          linewidths=2, zorder=10, label="Dog")
                ax.scatter([cat_cent[0]], [cat_cent[1]],
                          c=CONDITION_STYLES["mono_cat"]["color"],
                          s=200, marker='o', edgecolors='black',
                          linewidths=2, zorder=10, label="Cat")

                # Mark CLIP AND and SuperDiff AND positions
                if clip_cent is not None:
                    ax.scatter([clip_cent[0]], [clip_cent[1]],
                              c=CONDITION_STYLES["mono_dog_and_cat"]["color"],
                              s=200, marker='D', edgecolors='black',
                              linewidths=2, zorder=10, label="CLIP AND")

                    # Draw arrow from nearest interpolation point
                    nearest_idx = np.argmin([np.linalg.norm(clip_cent - p)
                                            for p in interp_path])
                    ax.annotate("", xy=clip_cent, xytext=interp_path[nearest_idx],
                               arrowprops=dict(arrowstyle="->", color="green",
                                             lw=2, ls='--'))

                if sd_cent is not None:
                    ax.scatter([sd_cent[0]], [sd_cent[1]],
                              c=CONDITION_STYLES["superdiff_semantic"]["color"],
                              s=250, marker='*', edgecolors='black',
                              linewidths=2, zorder=10, label="SuperDiff AND")

                    nearest_idx = np.argmin([np.linalg.norm(sd_cent - p)
                                            for p in interp_path])
                    ax.annotate("", xy=sd_cent, xytext=interp_path[nearest_idx],
                               arrowprops=dict(arrowstyle="->", color="red",
                                             lw=2, ls='--'))

            ax.set_xlabel(f"{method} 1", fontsize=11)
            ax.set_ylabel(f"{method} 2", fontsize=11)
            ax.set_title(f"{method}: Interpolation vs Composition",
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(
            "Linear Interpolation Path Analysis\n"
            "How far do compositions deviate from simple linear blending?",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        save_path = output_path / "interpolation_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

    # -----------------------------------------------------------------------
    # 7. PCA Variance Explained
    # -----------------------------------------------------------------------
    def create_variance_explained_plot(self, output_path: Path):
        """Analyze and visualize PCA variance explained."""
        print("\n  Creating variance explained analysis...")

        if self.all_latents is None:
            print("    Warning: Original latents not provided, skipping")
            return

        # Fit PCA with more components
        n_components = min(20, len(self.all_latents) - 1, self.all_latents.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(self.all_latents)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Variance explained by each component
        ax = axes[0]
        var_exp = pca.explained_variance_ratio_ * 100
        cumsum = np.cumsum(var_exp)

        ax.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.7,
              color='steelblue', edgecolor='black', label='Individual')
        ax.plot(range(1, len(var_exp) + 1), cumsum, 'ro-',
               linewidth=2, label='Cumulative')

        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        ax.text(len(var_exp), 91, '90%', fontsize=9, color='gray')

        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Variance Explained (%)", fontsize=11)
        ax.set_title("PCA Variance Explained", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Find components needed for 90% variance
        n_90 = np.argmax(cumsum >= 90) + 1
        ax.axvline(x=n_90, color='red', linestyle=':', alpha=0.7)
        ax.text(n_90 + 0.5, max(var_exp) * 0.8,
               f'{n_90} components\nfor 90%', fontsize=9, color='red')

        # 2. Per-condition variance in PC space
        ax = axes[1]
        proj = self.projections["PCA"]["2d"]

        variances = []
        labels = []
        colors = []

        for cond_idx, key in enumerate(self.condition_keys):
            style = CONDITION_STYLES.get(key, {})
            mask = self.condition_indices == cond_idx
            points = proj[mask]

            if len(points) > 1:
                var = np.var(points, axis=0).sum()
                variances.append(var)
                labels.append(style.get("short", key))
                colors.append(style.get("color", "#999999"))

        bars = ax.bar(labels, variances, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Cluster Variance (PC1 + PC2)", fontsize=11)
        ax.set_title("Per-Condition Spread in PCA Space", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = output_path / "variance_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

        return {
            "variance_explained": var_exp,
            "cumulative": cumsum,
            "n_components_90": n_90,
        }

    # -----------------------------------------------------------------------
    # 8. Hypothesis-Specific Visualization
    # -----------------------------------------------------------------------
    def create_hypothesis_visualization(self, output_path: Path):
        """Create visualizations specifically designed to test hypotheses."""
        print("\n  Creating hypothesis-specific visualizations...")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        proj = self.projections["PCA"]["2d"]
        centroids = self._compute_centroids(proj)

        # Get key indices
        dog_idx = self.key_to_idx.get("mono_dog")
        cat_idx = self.key_to_idx.get("mono_cat")
        clip_idx = self.key_to_idx.get("mono_dog_and_cat")
        sd_idx = self.key_to_idx.get("superdiff_semantic")

        dog_cent = centroids.get(dog_idx)
        cat_cent = centroids.get(cat_idx)
        clip_cent = centroids.get(clip_idx)
        sd_cent = centroids.get(sd_idx)

        # ===== H1: Intersection Location =====
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("H1: Semantic Intersection Location\n"
                    "Do CLIP AND and SuperDiff AND agree?",
                    fontsize=11, fontweight='bold')

        if dog_cent is not None and cat_cent is not None:
            midpoint = (dog_cent + cat_cent) / 2

            # Draw base axis
            ax.plot([dog_cent[0], cat_cent[0]], [dog_cent[1], cat_cent[1]],
                   'k:', linewidth=2, alpha=0.6)

            # Plot base concepts
            ax.scatter([dog_cent[0]], [dog_cent[1]],
                      c=CONDITION_STYLES["mono_dog"]["color"],
                      s=200, marker='o', edgecolors='black', linewidths=2,
                      label="Dog", zorder=10)
            ax.scatter([cat_cent[0]], [cat_cent[1]],
                      c=CONDITION_STYLES["mono_cat"]["color"],
                      s=200, marker='o', edgecolors='black', linewidths=2,
                      label="Cat", zorder=10)

            # Midpoint
            ax.scatter([midpoint[0]], [midpoint[1]],
                      c='gray', s=150, marker='X', edgecolors='black',
                      linewidths=2, label="Linear Midpoint", zorder=10)

            # Compositions
            if clip_cent is not None:
                ax.scatter([clip_cent[0]], [clip_cent[1]],
                          c=CONDITION_STYLES["mono_dog_and_cat"]["color"],
                          s=200, marker='D', edgecolors='black', linewidths=2,
                          label="CLIP AND", zorder=10)
                dist_clip = np.linalg.norm(clip_cent - midpoint)
                ax.annotate(f'd={dist_clip:.2f}',
                           xy=clip_cent, xytext=(clip_cent[0]+0.5, clip_cent[1]+0.5),
                           fontsize=9, color=CONDITION_STYLES["mono_dog_and_cat"]["color"])

            if sd_cent is not None:
                ax.scatter([sd_cent[0]], [sd_cent[1]],
                          c=CONDITION_STYLES["superdiff_semantic"]["color"],
                          s=250, marker='*', edgecolors='black', linewidths=2,
                          label="SuperDiff AND", zorder=10)
                dist_sd = np.linalg.norm(sd_cent - midpoint)
                ax.annotate(f'd={dist_sd:.2f}',
                           xy=sd_cent, xytext=(sd_cent[0]+0.5, sd_cent[1]-0.5),
                           fontsize=9, color=CONDITION_STYLES["superdiff_semantic"]["color"])

            # Agreement distance
            if clip_cent is not None and sd_cent is not None:
                agreement_dist = np.linalg.norm(clip_cent - sd_cent)
                base_dist = np.linalg.norm(dog_cent - cat_cent)
                agreement_norm = agreement_dist / base_dist

                ax.plot([clip_cent[0], sd_cent[0]], [clip_cent[1], sd_cent[1]],
                       'purple', linewidth=2, linestyle='-.', alpha=0.8)
                mid_agree = (clip_cent + sd_cent) / 2
                ax.annotate(f'Agreement: {(1-agreement_norm)*100:.1f}%',
                           xy=mid_agree, fontsize=10, fontweight='bold',
                           color='purple', ha='center')

        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # ===== H2: Spatial vs Semantic =====
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title("H2: Spatial Constraints Effect\n"
                    "Does 'on the left' change geometry?",
                    fontsize=11, fontweight='bold')

        # Compare semantic vs spatial for both methods
        for method_type, (sem_key, spa_key) in [
            ("CLIP", ("mono_dog_and_cat", "mono_spatial")),
            ("SuperDiff", ("superdiff_semantic", "superdiff_spatial")),
        ]:
            sem_idx = self.key_to_idx.get(sem_key)
            spa_idx = self.key_to_idx.get(spa_key)

            sem_mask = self.condition_indices == sem_idx
            spa_mask = self.condition_indices == spa_idx

            sem_points = proj[sem_mask]
            spa_points = proj[spa_mask]

            sem_style = CONDITION_STYLES.get(sem_key, {})
            spa_style = CONDITION_STYLES.get(spa_key, {})

            if len(sem_points) > 0 and len(spa_points) > 0:
                # Semantic cluster
                ax.scatter(sem_points[:, 0], sem_points[:, 1],
                          c=sem_style.get("color"), s=50, alpha=0.5,
                          marker='o' if method_type == "CLIP" else '*',
                          label=f"{method_type} Semantic")

                # Spatial cluster
                ax.scatter(spa_points[:, 0], spa_points[:, 1],
                          c=spa_style.get("color"), s=50, alpha=0.5,
                          marker='s' if method_type == "CLIP" else 'p',
                          label=f"{method_type} Spatial")

                # Draw arrow between centroids
                sem_cent = sem_points.mean(axis=0)
                spa_cent = spa_points.mean(axis=0)
                shift = np.linalg.norm(spa_cent - sem_cent)

                ax.annotate("", xy=spa_cent, xytext=sem_cent,
                           arrowprops=dict(arrowstyle="->",
                                         color=sem_style.get("color"),
                                         lw=2))
                mid = (sem_cent + spa_cent) / 2
                ax.text(mid[0], mid[1] + 0.3, f'Δ={shift:.2f}',
                       fontsize=9, ha='center')

        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # ===== H3: Cluster Separation =====
        ax = fig.add_subplot(gs[1, 0])
        ax.set_title("H3: Cluster Separation\n"
                    "Which method produces more distinct clusters?",
                    fontsize=11, fontweight='bold')

        # Silhouette scores per method
        methods = ["PCA", "t-SNE", "UMAP"]
        clip_silhouettes = []
        sd_silhouettes = []

        for method in methods:
            proj_m = self.projections[method]["2d"]

            # CLIP conditions only
            clip_mask = np.isin(self.condition_indices,
                               [self.key_to_idx.get("mono_dog_and_cat"),
                                self.key_to_idx.get("mono_spatial")])
            if clip_mask.sum() > 3:
                clip_labels = self.condition_indices[clip_mask]
                try:
                    clip_sil = silhouette_score(proj_m[clip_mask], clip_labels)
                except Exception:
                    clip_sil = 0
                clip_silhouettes.append(clip_sil)
            else:
                clip_silhouettes.append(0)

            # SuperDiff conditions only
            sd_mask = np.isin(self.condition_indices,
                             [self.key_to_idx.get("superdiff_semantic"),
                              self.key_to_idx.get("superdiff_spatial")])
            if sd_mask.sum() > 3:
                sd_labels = self.condition_indices[sd_mask]
                try:
                    sd_sil = silhouette_score(proj_m[sd_mask], sd_labels)
                except Exception:
                    sd_sil = 0
                sd_silhouettes.append(sd_sil)
            else:
                sd_silhouettes.append(0)

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width/2, clip_silhouettes, width,
              label='CLIP AND', color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, sd_silhouettes, width,
              label='SuperDiff AND', color='#e74c3c', alpha=0.7, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Silhouette Score (Semantic vs Spatial)")
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # ===== Summary Statistics =====
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')

        summary_text = "HYPOTHESIS TESTING SUMMARY\n" + "=" * 40 + "\n\n"

        if dog_cent is not None and cat_cent is not None:
            base_dist = np.linalg.norm(dog_cent - cat_cent)
            midpoint = (dog_cent + cat_cent) / 2

            summary_text += f"Base Distance (Dog ↔ Cat): {base_dist:.3f}\n\n"

            if clip_cent is not None:
                d_clip = np.linalg.norm(clip_cent - midpoint)
                summary_text += f"CLIP AND → Midpoint: {d_clip:.3f} ({100*d_clip/base_dist:.1f}% of base)\n"

            if sd_cent is not None:
                d_sd = np.linalg.norm(sd_cent - midpoint)
                summary_text += f"SuperDiff AND → Midpoint: {d_sd:.3f} ({100*d_sd/base_dist:.1f}% of base)\n"

            if clip_cent is not None and sd_cent is not None:
                agreement = np.linalg.norm(clip_cent - sd_cent)
                summary_text += f"\nAgreement Distance: {agreement:.3f}\n"
                summary_text += f"Agreement Score: {100*(1-agreement/base_dist):.1f}%\n"

                # Interpretation
                summary_text += "\n" + "-" * 40 + "\n"
                norm_agree = agreement / base_dist
                if norm_agree < 0.2:
                    summary_text += "✓ STRONG AGREEMENT\n"
                    summary_text += "  Methods find similar intersection"
                elif norm_agree < 0.5:
                    summary_text += "~ PARTIAL AGREEMENT\n"
                    summary_text += "  Some divergence in composition"
                else:
                    summary_text += "✗ WEAK AGREEMENT\n"
                    summary_text += "  Methods find different intersections"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle("Hypothesis Testing: CLIP AND vs SuperDiff AND",
                    fontsize=14, fontweight='bold')

        save_path = output_path / "hypothesis_visualization.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
        plt.close()

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def create_all_visualizations(self, output_dir: str):
        """Generate all enhanced visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("CREATING ENHANCED MANIFOLD VISUALIZATIONS")
        print("=" * 80)

        # 1. Confidence ellipses
        self.create_confidence_ellipse_plot(output_path)

        # 2. Convex hulls
        self.create_convex_hull_plot(output_path)

        # 3. Annotated interactive plot
        self.create_annotated_interactive_plot(output_path)

        # 4. Cluster metrics
        metrics = self.create_cluster_metrics_visualization(output_path)

        # 5. Density contours
        self.create_density_contour_plot(output_path)

        # 6. Interpolation analysis
        self.create_interpolation_analysis(output_path)

        # 7. Variance analysis (if latents available)
        if self.all_latents is not None:
            self.create_variance_explained_plot(output_path)

        # 8. Hypothesis-specific visualization
        self.create_hypothesis_visualization(output_path)

        print("\n" + "=" * 80)
        print("ENHANCED VISUALIZATIONS COMPLETE")
        print("=" * 80)
        print(f"\nSaved to: {output_path.absolute()}")

        return metrics
