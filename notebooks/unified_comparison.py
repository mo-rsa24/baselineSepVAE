"""
Unified 6-Condition Comparison for SUPERDIFF Analysis

Combines semantic and spatial experiment suites into a single
comparison across 6 conditions:
  1. Monolithic: semantic composed prompt
  2. Monolithic: spatial composed prompt
  3. Individual: prompt A
  4. Individual: prompt B
  5. SuperDiff: semantic (A AND B)
  6. SuperDiff: spatial (A AND B)

Produces 4 visualizations:
  - sample_images_comparison.png
  - pca_tsne_projections.png
  - unified_latent_space_3d_interactive.html
  - trajectory_evolution_3d_interactive_averaged.html
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import plotly.graph_objects as go

from notebooks.utils import get_image


@dataclass
class UnifiedCondition:
    """One condition in the unified 6-condition comparison."""
    key: str
    label: str
    short_label: str
    color: str
    marker: str
    latents: list       # list of (batch_size, C, H, W) tensors, one per run
    trajectories: list  # list of LatentTrajectoryCollector, one per run


def build_unified_conditions(semantic_suite, spatial_suite, config):
    """
    Assemble the 6 UnifiedCondition entries from two experiment suites.

    Args:
        semantic_suite: CompositionExperimentSuite for semantic prompts
        spatial_suite: CompositionExperimentSuite for spatial prompts
        config: SpatialGroundingConfig with prompt text

    Returns:
        List of 6 UnifiedCondition objects
    """
    conditions = [
        UnifiedCondition(
            key="semantic_mono",
            label=f'Monolithic: "{config.semantic_composed}"',
            short_label="Mono (semantic)",
            color="#2ecc71",
            marker="s",
            latents=semantic_suite.results['monolithic']['latents'],
            trajectories=semantic_suite.results['monolithic']['trajectories'],
        ),
        UnifiedCondition(
            key="spatial_mono",
            label=f'Monolithic: "{config.spatial_composed}"',
            short_label="Mono (spatial)",
            color="#1abc9c",
            marker="D",
            latents=spatial_suite.results['monolithic']['latents'],
            trajectories=spatial_suite.results['monolithic']['trajectories'],
        ),
        UnifiedCondition(
            key="individual_a",
            label=f'Individual: "{config.semantic_a}"',
            short_label=f'Individual: "{config.semantic_a}"',
            color="#3498db",
            marker="^",
            latents=semantic_suite.results['prompt_a']['latents'],
            trajectories=semantic_suite.results['prompt_a']['trajectories'],
        ),
        UnifiedCondition(
            key="individual_b",
            label=f'Individual: "{config.semantic_b}"',
            short_label=f'Individual: "{config.semantic_b}"',
            color="#e67e22",
            marker="v",
            latents=semantic_suite.results['prompt_b']['latents'],
            trajectories=semantic_suite.results['prompt_b']['trajectories'],
        ),
        UnifiedCondition(
            key="semantic_superdiff",
            label=f'SuperDiff: "{config.semantic_a}" AND "{config.semantic_b}"',
            short_label="SD (semantic)",
            color="#e74c3c",
            marker="*",
            latents=semantic_suite.results['superdiff']['latents'],
            trajectories=semantic_suite.results['superdiff']['trajectories'],
        ),
        UnifiedCondition(
            key="spatial_superdiff",
            label=f'SuperDiff: "{config.spatial_a}" AND "{config.spatial_b}"',
            short_label="SD (spatial)",
            color="#9b59b6",
            marker="P",
            latents=spatial_suite.results['superdiff']['latents'],
            trajectories=spatial_suite.results['superdiff']['trajectories'],
        ),
    ]
    return conditions


def _collect_flat_latents(conditions: List[UnifiedCondition]):
    """
    Flatten and concatenate all latents across conditions.

    Returns:
        all_data: numpy array of shape (N_total, D)
        condition_indices: dict mapping condition key -> (start_idx, end_idx)
    """
    all_data = []
    condition_indices = {}
    offset = 0

    for cond in conditions:
        flat = torch.cat([l.cpu().flatten(1) for l in cond.latents], dim=0).numpy()
        all_data.append(flat)
        condition_indices[cond.key] = (offset, offset + len(flat))
        offset += len(flat)

    return np.vstack(all_data), condition_indices


def generate_unified_comparison(conditions, vae, output_dir, num_runs):
    """
    Generate all 4 unified comparison visualizations.

    Args:
        conditions: List of 6 UnifiedCondition objects
        vae: VAE decoder for image generation
        output_dir: Output directory path
        num_runs: Number of experiment runs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING UNIFIED 6-CONDITION COMPARISON")
    print("="*80)

    plot_sample_images_comparison(conditions, vae, output_dir, num_runs)
    plot_pca_tsne_projections(conditions, output_dir)
    plot_unified_latent_space_3d_interactive(conditions, output_dir)
    plot_trajectory_evolution_3d_averaged(conditions, output_dir)

    print("\n" + "="*80)
    print("UNIFIED COMPARISON COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {output_path / 'sample_images_comparison.png'}")
    print(f"  - {output_path / 'pca_tsne_projections.png'}")
    print(f"  - {output_path / 'unified_latent_space_3d_interactive.html'}")
    print(f"  - {output_path / 'trajectory_evolution_3d_interactive_averaged.html'}")


def plot_sample_images_comparison(conditions, vae, output_dir, num_runs):
    """
    Generate sample images comparison grid.

    Rows: 6 conditions, each labeled with condition name and prompt.
    Columns: runs (up to 8 displayed).
    """
    print("\n  Generating sample images comparison...")

    output_path = Path(output_dir)
    n_display = min(8, num_runs)
    n_conditions = len(conditions)

    fig, axes = plt.subplots(
        n_conditions, n_display,
        figsize=(2.5 * n_display, 3 * n_conditions)
    )

    # Handle single-column case
    if n_display == 1:
        axes = axes.reshape(n_conditions, 1)

    for row_idx, cond in enumerate(conditions):
        for col_idx in range(n_display):
            latent = cond.latents[col_idx][0:1]  # first sample in batch
            img = get_image(vae, latent, nrow=1, ncol=1)

            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')

            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    cond.label,
                    fontsize=8, rotation=0, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor=cond.color, alpha=0.3)
                )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f'Run {col_idx+1}', fontsize=10, fontweight='bold'
                )

    fig.suptitle(
        'Unified Comparison: All 6 Conditions Across Runs',
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout()
    plt.savefig(output_path / 'sample_images_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: sample_images_comparison.png")


def plot_pca_tsne_projections(conditions, output_dir):
    """
    PCA and t-SNE projections with all 6 conditions.

    Left panel: PCA with scatter points and centroids.
    Right panel: t-SNE with scatter points and centroids.
    """
    print("\n  Generating PCA/t-SNE projections...")

    output_path = Path(output_dir)
    all_data, condition_indices = _collect_flat_latents(conditions)

    # PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(all_data)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data) // 4))
    data_tsne = tsne.fit_transform(all_data)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    projections = [
        (axes[0], data_pca, 'PCA Projection'),
        (axes[1], data_tsne, 't-SNE Projection'),
    ]

    for ax, proj_data, title in projections:
        for cond in conditions:
            start, end = condition_indices[cond.key]
            subset = proj_data[start:end]

            # Scatter points
            ax.scatter(subset[:, 0], subset[:, 1],
                       c=cond.color, label=cond.short_label,
                       alpha=0.5, s=30, edgecolors='none')

            # Centroid
            centroid = subset.mean(axis=0)
            ax.scatter(centroid[0], centroid[1],
                       c=cond.color, marker=cond.marker,
                       s=400, edgecolors='black', linewidths=2, zorder=10)

        if 'PCA' in title:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                          fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                          fontsize=12)
        else:
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Unified 6-Condition Projections\n(Large markers = centroids)',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path / 'pca_tsne_projections.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: pca_tsne_projections.png")


def plot_unified_latent_space_3d_interactive(conditions, output_dir):
    """
    Interactive 3D PCA scatter plot with all 6 conditions using Plotly.
    """
    print("\n  Generating interactive 3D latent space...")

    output_path = Path(output_dir)
    all_data, condition_indices = _collect_flat_latents(conditions)

    # 3D PCA
    pca_3d = PCA(n_components=3)
    data_3d = pca_3d.fit_transform(all_data)

    fig = go.Figure()

    for cond in conditions:
        start, end = condition_indices[cond.key]
        subset = data_3d[start:end]

        # Sample points
        fig.add_trace(go.Scatter3d(
            x=subset[:, 0], y=subset[:, 1], z=subset[:, 2],
            mode='markers',
            name=cond.short_label,
            marker=dict(size=4, color=cond.color, opacity=0.4),
            hovertext=[cond.label] * len(subset)
        ))

        # Centroid
        centroid = subset.mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers',
            name=f'Centroid: {cond.short_label}',
            marker=dict(size=12, color=cond.color,
                        line=dict(color='black', width=2)),
            hovertext=[f'Centroid: {cond.label}']
        ))

    fig.update_layout(
        title='Unified 3D Latent Space: 6 Conditions',
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
        ),
        width=1100,
        height=850
    )

    fig.write_html(str(output_path / 'unified_latent_space_3d_interactive.html'))
    print(f"    Saved: unified_latent_space_3d_interactive.html")


def plot_trajectory_evolution_3d_averaged(conditions, output_dir):
    """
    Interactive 3D trajectory visualization averaged across runs.

    For each condition, averages the trajectory (first sample per run)
    across all runs, then projects into a shared 3D PCA space.
    """
    print("\n  Generating averaged 3D trajectory evolution...")

    output_path = Path(output_dir)

    # Compute averaged trajectory for each condition
    averaged_trajs = {}
    for cond in conditions:
        run_trajs = []
        for traj_collector in cond.trajectories:
            # trajectories: (num_steps+1, batch_size, C, H, W)
            # Extract first sample, flatten spatial dims
            single = traj_collector.trajectories[:, 0, :].flatten(1)  # (num_steps+1, D)
            run_trajs.append(single)

        stacked = torch.stack(run_trajs, dim=0)  # (num_runs, num_steps+1, D)
        averaged = stacked.float().mean(dim=0)    # (num_steps+1, D)
        averaged_trajs[cond.key] = averaged.numpy()

    # Fit PCA on all averaged trajectory points combined
    all_points = np.vstack(list(averaged_trajs.values()))
    pca_3d = PCA(n_components=3)
    pca_3d.fit(all_points)

    # Project each averaged trajectory
    projected = {}
    for key, traj in averaged_trajs.items():
        projected[key] = pca_3d.transform(traj)

    # Plotly figure
    fig = go.Figure()

    for cond in conditions:
        traj_3d = projected[cond.key]
        num_steps = len(traj_3d)

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=traj_3d[:, 0], y=traj_3d[:, 1], z=traj_3d[:, 2],
            mode='lines',
            name=cond.short_label,
            line=dict(color=cond.color, width=5),
            hovertext=[f'{cond.short_label} step {i}' for i in range(num_steps)]
        ))

        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[traj_3d[0, 0]], y=[traj_3d[0, 1]], z=[traj_3d[0, 2]],
            mode='markers',
            name=f'{cond.short_label} (start)',
            marker=dict(size=8, color='white',
                        line=dict(color=cond.color, width=3)),
            hovertext=[f'{cond.short_label} start'],
            showlegend=False
        ))

        # End marker
        fig.add_trace(go.Scatter3d(
            x=[traj_3d[-1, 0]], y=[traj_3d[-1, 1]], z=[traj_3d[-1, 2]],
            mode='markers',
            name=f'{cond.short_label} (end)',
            marker=dict(size=12, color=cond.color,
                        line=dict(color='black', width=2)),
            hovertext=[f'{cond.short_label} end'],
            showlegend=False
        ))

    fig.update_layout(
        title='Averaged 3D Trajectory Evolution: 6 Conditions<br>'
              '<sub>Trajectories averaged across all runs (first sample per run)</sub>',
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
        ),
        width=1100,
        height=850
    )

    fig.write_html(str(output_path / 'trajectory_evolution_3d_interactive_averaged.html'))
    print(f"    Saved: trajectory_evolution_3d_interactive_averaged.html")
