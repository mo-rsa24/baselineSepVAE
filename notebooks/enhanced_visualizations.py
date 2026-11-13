"""
Enhanced Visualizations for SUPERDIFF Analysis

Addresses key limitations:
1. Single unified latent space plot with all conditions
2. 3D projections alongside 2D
3. Trajectory evolution animations and interactive plots
4. Temporal dynamics with phase identification
5. Velocity field vector plots

These visualizations provide richer geometric and temporal diagnostics
to disambiguate semantic conjunction, latent intersection, and off-manifold drift.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px


def plot_unified_latent_space_2d3d(
    results_dict: Dict,
    output_dir: str,
    config
):
    """
    Create unified 2D and 3D visualizations showing ALL conditions in same space

    Args:
        results_dict: Dictionary with keys ['monolithic', 'prompt_a', 'prompt_b', 'superdiff']
        output_dir: Output directory
        config: Experiment configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating unified latent space visualizations...")

    # Collect all final latents
    latents_mono = torch.cat([l.cpu().flatten(1) for l in results_dict['monolithic']['latents']], dim=0)
    latents_a = torch.cat([l.cpu().flatten(1) for l in results_dict['prompt_a']['latents']], dim=0)
    latents_b = torch.cat([l.cpu().flatten(1) for l in results_dict['prompt_b']['latents']], dim=0)
    latents_sd = torch.cat([l.cpu().flatten(1) for l in results_dict['superdiff']['latents']], dim=0)

    # Combine all data
    all_latents = torch.cat([latents_mono, latents_a, latents_b, latents_sd], dim=0).numpy()

    # Labels
    n_mono, n_a, n_b, n_sd = len(latents_mono), len(latents_a), len(latents_b), len(latents_sd)
    labels = (['Monolithic'] * n_mono +
              ['Prompt A'] * n_a +
              ['Prompt B'] * n_b +
              ['SUPERDIFF'] * n_sd)

    colors_map = {
        'Monolithic': '#2ecc71',  # green
        'Prompt A': '#3498db',    # blue
        'Prompt B': '#e74c3c',    # red
        'SUPERDIFF': '#9b59b6'    # purple
    }
    colors = [colors_map[l] for l in labels]

    # Compute centroids
    centroid_mono = latents_mono.mean(dim=0).numpy()
    centroid_a = latents_a.mean(dim=0).numpy()
    centroid_b = latents_b.mean(dim=0).numpy()
    centroid_sd = latents_sd.mean(dim=0).numpy()
    centroid_midpoint = (centroid_a + centroid_b) / 2

    # ========================================================================
    # 2D Visualization (PCA)
    # ========================================================================
    print("  - Creating 2D PCA visualization...")

    pca_2d = PCA(n_components=2)
    latents_2d = pca_2d.fit_transform(all_latents)

    # Project centroids
    centroid_mono_2d = pca_2d.transform([centroid_mono])[0]
    centroid_a_2d = pca_2d.transform([centroid_a])[0]
    centroid_b_2d = pca_2d.transform([centroid_b])[0]
    centroid_sd_2d = pca_2d.transform([centroid_sd])[0]
    centroid_mid_2d = pca_2d.transform([centroid_midpoint])[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot samples
    for label, color in colors_map.items():
        mask = np.array(labels) == label
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                  c=color, label=label, alpha=0.4, s=40, edgecolors='none')

    # Plot centroids with large markers
    ax.scatter(centroid_mono_2d[0], centroid_mono_2d[1],
              c='#2ecc71', marker='s', s=400, edgecolors='black',
              linewidths=2.5, zorder=10, label='Centroid: Monolithic')

    ax.scatter(centroid_a_2d[0], centroid_a_2d[1],
              c='#3498db', marker='^', s=400, edgecolors='black',
              linewidths=2.5, zorder=10, label='Centroid: A')

    ax.scatter(centroid_b_2d[0], centroid_b_2d[1],
              c='#e74c3c', marker='v', s=400, edgecolors='black',
              linewidths=2.5, zorder=10, label='Centroid: B')

    ax.scatter(centroid_sd_2d[0], centroid_sd_2d[1],
              c='#9b59b6', marker='*', s=600, edgecolors='black',
              linewidths=2.5, zorder=10, label='Centroid: SUPERDIFF')

    ax.scatter(centroid_mid_2d[0], centroid_mid_2d[1],
              c='#95a5a6', marker='D', s=400, edgecolors='black',
              linewidths=2.5, zorder=10, label='Midpoint: (A+B)/2')

    # Draw lines connecting key centroids
    # Line from A to B
    ax.plot([centroid_a_2d[0], centroid_b_2d[0]],
           [centroid_a_2d[1], centroid_b_2d[1]],
           'k--', linewidth=2, alpha=0.5, label='A-B line')

    # Line from SD to midpoint
    ax.plot([centroid_sd_2d[0], centroid_mid_2d[0]],
           [centroid_sd_2d[1], centroid_mid_2d[1]],
           'purple', linewidth=2.5, linestyle=':', alpha=0.7,
           label='SD to midpoint')

    # Line from SD to monolithic
    ax.plot([centroid_sd_2d[0], centroid_mono_2d[0]],
           [centroid_sd_2d[1], centroid_mono_2d[1]],
           'green', linewidth=2.5, linestyle=':', alpha=0.7,
           label='SD to monolithic')

    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=13)
    ax.set_title('Unified Latent Space: All Conditions (2D PCA)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'unified_latent_space_2d.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Saved: unified_latent_space_2d.png")

    # ========================================================================
    # 3D Visualization (PCA)
    # ========================================================================
    print("  - Creating 3D PCA visualization...")

    pca_3d = PCA(n_components=3)
    latents_3d = pca_3d.fit_transform(all_latents)

    # Project centroids
    centroid_mono_3d = pca_3d.transform([centroid_mono])[0]
    centroid_a_3d = pca_3d.transform([centroid_a])[0]
    centroid_b_3d = pca_3d.transform([centroid_b])[0]
    centroid_sd_3d = pca_3d.transform([centroid_sd])[0]
    centroid_mid_3d = pca_3d.transform([centroid_midpoint])[0]

    # Matplotlib 3D plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    for label, color in colors_map.items():
        mask = np.array(labels) == label
        ax.scatter(latents_3d[mask, 0], latents_3d[mask, 1], latents_3d[mask, 2],
                  c=color, label=label, alpha=0.3, s=30, edgecolors='none')

    # Plot centroids
    ax.scatter(centroid_mono_3d[0], centroid_mono_3d[1], centroid_mono_3d[2],
              c='#2ecc71', marker='s', s=300, edgecolors='black',
              linewidths=2, label='Centroid: Monolithic')

    ax.scatter(centroid_a_3d[0], centroid_a_3d[1], centroid_a_3d[2],
              c='#3498db', marker='^', s=300, edgecolors='black',
              linewidths=2, label='Centroid: A')

    ax.scatter(centroid_b_3d[0], centroid_b_3d[1], centroid_b_3d[2],
              c='#e74c3c', marker='v', s=300, edgecolors='black',
              linewidths=2, label='Centroid: B')

    ax.scatter(centroid_sd_3d[0], centroid_sd_3d[1], centroid_sd_3d[2],
              c='#9b59b6', marker='*', s=500, edgecolors='black',
              linewidths=2, label='Centroid: SUPERDIFF')

    ax.scatter(centroid_mid_3d[0], centroid_mid_3d[1], centroid_mid_3d[2],
              c='#95a5a6', marker='D', s=300, edgecolors='black',
              linewidths=2, label='Midpoint: (A+B)/2')

    # Draw connecting lines
    ax.plot([centroid_a_3d[0], centroid_b_3d[0]],
           [centroid_a_3d[1], centroid_b_3d[1]],
           [centroid_a_3d[2], centroid_b_3d[2]],
           'k--', linewidth=2, alpha=0.5)

    ax.plot([centroid_sd_3d[0], centroid_mid_3d[0]],
           [centroid_sd_3d[1], centroid_mid_3d[1]],
           [centroid_sd_3d[2], centroid_mid_3d[2]],
           color='purple', linewidth=2.5, linestyle=':', alpha=0.7)

    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax.set_title('Unified Latent Space: All Conditions (3D PCA)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / 'unified_latent_space_3d.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Saved: unified_latent_space_3d.png")

    # ========================================================================
    # Interactive 3D Visualization (Plotly)
    # ========================================================================
    print("  - Creating interactive 3D visualization...")

    # Create plotly figure
    fig_plotly = go.Figure()

    # Add samples
    for label, color in colors_map.items():
        mask = np.array(labels) == label
        fig_plotly.add_trace(go.Scatter3d(
            x=latents_3d[mask, 0],
            y=latents_3d[mask, 1],
            z=latents_3d[mask, 2],
            mode='markers',
            name=label,
            marker=dict(size=4, color=color, opacity=0.4),
            hovertext=[label] * mask.sum()
        ))

    # Add centroids
    centroids_data = [
        ('Monolithic', centroid_mono_3d, '#2ecc71', 'square'),
        ('A', centroid_a_3d, '#3498db', 'diamond'),
        ('B', centroid_b_3d, '#e74c3c', 'diamond'),
        ('SUPERDIFF', centroid_sd_3d, '#9b59b6', 'cross'),
        ('(A+B)/2', centroid_mid_3d, '#95a5a6', 'circle')
    ]

    for name, centroid, color, symbol in centroids_data:
        fig_plotly.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            name=f'Centroid: {name}',
            marker=dict(size=12, color=color, symbol=symbol,
                       line=dict(color='black', width=2)),
            hovertext=[f'Centroid: {name}']
        ))

    # Add lines
    # A to B
    fig_plotly.add_trace(go.Scatter3d(
        x=[centroid_a_3d[0], centroid_b_3d[0]],
        y=[centroid_a_3d[1], centroid_b_3d[1]],
        z=[centroid_a_3d[2], centroid_b_3d[2]],
        mode='lines',
        name='A-B line',
        line=dict(color='black', width=4, dash='dash'),
        hoverinfo='skip'
    ))

    # SD to midpoint
    fig_plotly.add_trace(go.Scatter3d(
        x=[centroid_sd_3d[0], centroid_mid_3d[0]],
        y=[centroid_sd_3d[1], centroid_mid_3d[1]],
        z=[centroid_sd_3d[2], centroid_mid_3d[2]],
        mode='lines',
        name='SD to midpoint',
        line=dict(color='purple', width=5, dash='dot'),
        hoverinfo='skip'
    ))

    fig_plotly.update_layout(
        title='Interactive 3D Latent Space: All Conditions',
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
        ),
        width=1000,
        height=800
    )

    fig_plotly.write_html(str(output_path / 'unified_latent_space_3d_interactive.html'))
    print(f"    Saved: unified_latent_space_3d_interactive.html")


def plot_trajectory_evolution_2d3d(
    results_dict: Dict,
    output_dir: str,
    config,
    run_idx: int = 0,
    sample_idx: int = 0
):
    """
    Plot trajectory paths through latent space over time (2D and 3D)

    Args:
        results_dict: Results dictionary
        output_dir: Output directory
        config: Configuration
        run_idx: Which run to visualize
        sample_idx: Which sample in batch to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating trajectory evolution visualizations...")

    # Extract trajectories
    traj_mono = results_dict['monolithic']['trajectories'][run_idx]
    traj_a = results_dict['prompt_a']['trajectories'][run_idx]
    traj_b = results_dict['prompt_b']['trajectories'][run_idx]
    traj_sd = results_dict['superdiff']['trajectories'][run_idx]

    # Get trajectory data for specific sample
    # Shape: (num_steps+1, batch_size, C, H, W) -> (num_steps+1, D)
    def extract_trajectory_sample(traj, sample_idx):
        return traj.trajectories[:, sample_idx, :].flatten(1).numpy()

    traj_mono_data = extract_trajectory_sample(traj_mono, sample_idx)
    traj_a_data = extract_trajectory_sample(traj_a, sample_idx)
    traj_b_data = extract_trajectory_sample(traj_b, sample_idx)
    traj_sd_data = extract_trajectory_sample(traj_sd, sample_idx)

    # Combine all trajectory points for PCA fitting
    all_points = np.vstack([traj_mono_data, traj_a_data, traj_b_data, traj_sd_data])

    # ========================================================================
    # 2D Trajectory Visualization
    # ========================================================================
    print("  - Creating 2D trajectory plot...")

    pca_2d = PCA(n_components=2)
    pca_2d.fit(all_points)

    # Transform trajectories
    traj_mono_2d = pca_2d.transform(traj_mono_data)
    traj_a_2d = pca_2d.transform(traj_a_data)
    traj_b_2d = pca_2d.transform(traj_b_data)
    traj_sd_2d = pca_2d.transform(traj_sd_data)

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    # Plot trajectories as paths with gradient colors (time progression)
    def plot_trajectory_with_gradient(ax, traj, color, label):
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create colormap from light to dark
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("", ["white", color])
        lc = LineCollection(segments, cmap=cmap, linewidth=3)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)

        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], c='white', s=200, marker='o',
                  edgecolors=color, linewidths=3, zorder=10, label=f'{label} (start)')
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=300, marker='*',
                  edgecolors='black', linewidths=2, zorder=10, label=f'{label} (end)')

    plot_trajectory_with_gradient(ax, traj_mono_2d, '#2ecc71', 'Monolithic')
    plot_trajectory_with_gradient(ax, traj_a_2d, '#3498db', 'A')
    plot_trajectory_with_gradient(ax, traj_b_2d, '#e74c3c', 'B')
    plot_trajectory_with_gradient(ax, traj_sd_2d, '#9b59b6', 'SUPERDIFF')

    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})', fontsize=13)
    ax.set_title(f'Trajectory Evolution (2D) - Run {run_idx+1}, Sample {sample_idx+1}',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio for undistorted view
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path / f'trajectory_evolution_2d_run{run_idx}_sample{sample_idx}.png',
               dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Saved: trajectory_evolution_2d_run{run_idx}_sample{sample_idx}.png")

    # ========================================================================
    # 3D Trajectory Visualization
    # ========================================================================
    print("  - Creating 3D trajectory plot...")

    pca_3d = PCA(n_components=3)
    pca_3d.fit(all_points)

    # Transform trajectories
    traj_mono_3d = pca_3d.transform(traj_mono_data)
    traj_a_3d = pca_3d.transform(traj_a_data)
    traj_b_3d = pca_3d.transform(traj_b_data)
    traj_sd_3d = pca_3d.transform(traj_sd_data)

    # Matplotlib 3D
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    trajectories = [
        (traj_mono_3d, '#2ecc71', 'Monolithic'),
        (traj_a_3d, '#3498db', 'A'),
        (traj_b_3d, '#e74c3c', 'B'),
        (traj_sd_3d, '#9b59b6', 'SUPERDIFF')
    ]

    for traj, color, label in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               color=color, linewidth=2.5, alpha=0.7, label=label)

        # Mark start
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                  c='white', s=150, marker='o', edgecolors=color,
                  linewidths=3)

        # Mark end
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                  c=color, s=250, marker='*', edgecolors='black',
                  linewidths=2)

    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax.set_title(f'Trajectory Evolution (3D) - Run {run_idx+1}, Sample {sample_idx+1}',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / f'trajectory_evolution_3d_run{run_idx}_sample{sample_idx}.png',
               dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Saved: trajectory_evolution_3d_run{run_idx}_sample{sample_idx}.png")

    # ========================================================================
    # Interactive 3D Trajectory (Plotly)
    # ========================================================================
    print("  - Creating interactive 3D trajectory...")

    fig_plotly = go.Figure()

    for traj, color, label in trajectories:
        # Trajectory line
        fig_plotly.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines',
            name=label,
            line=dict(color=color, width=5),
            hovertext=[f'{label} - step {i}' for i in range(len(traj))]
        ))

        # Start point
        fig_plotly.add_trace(go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            name=f'{label} (start)',
            marker=dict(size=8, color='white', symbol='circle',
                       line=dict(color=color, width=3)),
            hovertext=[f'{label} start']
        ))

        # End point
        fig_plotly.add_trace(go.Scatter3d(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            z=[traj[-1, 2]],
            mode='markers',
            name=f'{label} (end)',
            marker=dict(size=12, color=color, symbol='cross',
                       line=dict(color='black', width=2)),
            hovertext=[f'{label} end']
        ))

    fig_plotly.update_layout(
        title=f'Interactive 3D Trajectories - Run {run_idx+1}, Sample {sample_idx+1}',
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
        ),
        width=1000,
        height=800
    )

    fig_plotly.write_html(str(output_path / f'trajectory_evolution_3d_interactive_run{run_idx}_sample{sample_idx}.html'))
    print(f"    Saved: trajectory_evolution_3d_interactive_run{run_idx}_sample{sample_idx}.html")


def plot_temporal_phase_diagram(
    results_dict: Dict,
    output_dir: str,
    config
):
    """
    Analyze temporal dynamics and identify phase transitions during diffusion

    Creates plots showing:
    1. When do trajectories diverge?
    2. Critical timesteps for composition
    3. Velocity magnitude evolution
    4. Phase identification (early/mid/late diffusion)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating temporal phase diagrams...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract first run trajectories for analysis
    run_idx = 0

    traj_mono = results_dict['monolithic']['trajectories'][run_idx]
    traj_a = results_dict['prompt_a']['trajectories'][run_idx]
    traj_b = results_dict['prompt_b']['trajectories'][run_idx]
    traj_sd = results_dict['superdiff']['trajectories'][run_idx]

    def flatten_trajectory(traj):
        return traj.trajectories.reshape(traj.trajectories.shape[0],
                                        traj.trajectories.shape[1], -1)

    # Plot 1: Pairwise divergence over time
    ax = axes[0, 0]

    traj_sd_flat = flatten_trajectory(traj_sd)
    traj_mono_flat = flatten_trajectory(traj_mono)
    traj_a_flat = flatten_trajectory(traj_a)
    traj_b_flat = flatten_trajectory(traj_b)

    # Compute distances (average over batch)
    dist_sd_mono = torch.norm(traj_sd_flat - traj_mono_flat, dim=2).mean(dim=1).numpy()
    dist_sd_a = torch.norm(traj_sd_flat - traj_a_flat, dim=2).mean(dim=1).numpy()
    dist_sd_b = torch.norm(traj_sd_flat - traj_b_flat, dim=2).mean(dim=1).numpy()
    dist_a_b = torch.norm(traj_a_flat - traj_b_flat, dim=2).mean(dim=1).numpy()

    steps = np.arange(len(dist_sd_mono))

    ax.plot(steps, dist_sd_mono, label='SUPERDIFF vs Monolithic', color='purple', linewidth=2.5)
    ax.plot(steps, dist_sd_a, label='SUPERDIFF vs A', color='blue', linewidth=2.5)
    ax.plot(steps, dist_sd_b, label='SUPERDIFF vs B', color='red', linewidth=2.5)
    ax.plot(steps, dist_a_b, label='A vs B', color='gray', linestyle='--', linewidth=2)

    # Mark phases
    total_steps = len(steps)
    early_end = total_steps // 3
    mid_end = 2 * total_steps // 3

    ax.axvspan(0, early_end, alpha=0.1, color='blue', label='Early diffusion')
    ax.axvspan(early_end, mid_end, alpha=0.1, color='green', label='Mid diffusion')
    ax.axvspan(mid_end, total_steps, alpha=0.1, color='red', label='Late diffusion')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Trajectory Divergence Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Rate of divergence (derivative of distances)
    ax = axes[0, 1]

    # Compute derivatives (rate of change)
    def compute_derivative(signal):
        return np.gradient(signal)

    rate_sd_mono = compute_derivative(dist_sd_mono)
    rate_sd_a = compute_derivative(dist_sd_a)
    rate_sd_b = compute_derivative(dist_sd_b)

    ax.plot(steps, rate_sd_mono, label='SUPERDIFF vs Monolithic', color='purple', linewidth=2.5)
    ax.plot(steps, rate_sd_a, label='SUPERDIFF vs A', color='blue', linewidth=2.5)
    ax.plot(steps, rate_sd_b, label='SUPERDIFF vs B', color='red', linewidth=2.5)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Mark phases
    ax.axvspan(0, early_end, alpha=0.1, color='blue')
    ax.axvspan(early_end, mid_end, alpha=0.1, color='green')
    ax.axvspan(mid_end, total_steps, alpha=0.1, color='red')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Rate of Divergence', fontsize=12)
    ax.set_title('When Do Trajectories Diverge? (Derivative)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Velocity magnitude evolution
    ax = axes[1, 0]

    vel_mono = traj_mono.velocities.reshape(traj_mono.velocities.shape[0],
                                            traj_mono.velocities.shape[1], -1)
    vel_sd = traj_sd.velocities.reshape(traj_sd.velocities.shape[0],
                                        traj_sd.velocities.shape[1], -1)

    vel_mag_mono = torch.norm(vel_mono, dim=2).mean(dim=1).numpy()
    vel_mag_sd = torch.norm(vel_sd, dim=2).mean(dim=1).numpy()

    steps_vel = np.arange(len(vel_mag_mono))

    ax.plot(steps_vel, vel_mag_mono, label='Monolithic', color='green', linewidth=2.5)
    ax.plot(steps_vel, vel_mag_sd, label='SUPERDIFF', color='purple', linewidth=2.5)

    # Mark phases
    ax.axvspan(0, early_end, alpha=0.1, color='blue')
    ax.axvspan(early_end, mid_end, alpha=0.1, color='green')
    ax.axvspan(mid_end, total_steps, alpha=0.1, color='red')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Velocity Magnitude', fontsize=12)
    ax.set_title('Velocity Field Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Critical timestep detection (peaks in divergence rate)
    ax = axes[1, 1]

    # Find peaks in divergence rate
    from scipy.signal import find_peaks

    peaks_mono, _ = find_peaks(np.abs(rate_sd_mono), prominence=rate_sd_mono.std())
    peaks_a, _ = find_peaks(np.abs(rate_sd_a), prominence=rate_sd_a.std())
    peaks_b, _ = find_peaks(np.abs(rate_sd_b), prominence=rate_sd_b.std())

    ax.plot(steps, np.abs(rate_sd_mono), label='|Rate| SD-Monolithic', color='purple', linewidth=2)
    ax.plot(steps, np.abs(rate_sd_a), label='|Rate| SD-A', color='blue', linewidth=2)
    ax.plot(steps, np.abs(rate_sd_b), label='|Rate| SD-B', color='red', linewidth=2)

    # Mark critical points
    ax.scatter(peaks_mono, np.abs(rate_sd_mono)[peaks_mono], c='purple', s=100,
              marker='x', linewidths=3, zorder=10)
    ax.scatter(peaks_a, np.abs(rate_sd_a)[peaks_a], c='blue', s=100,
              marker='x', linewidths=3, zorder=10)
    ax.scatter(peaks_b, np.abs(rate_sd_b)[peaks_b], c='red', s=100,
              marker='x', linewidths=3, zorder=10)

    # Mark phases
    ax.axvspan(0, early_end, alpha=0.1, color='blue')
    ax.axvspan(early_end, mid_end, alpha=0.1, color='green')
    ax.axvspan(mid_end, total_steps, alpha=0.1, color='red')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('Absolute Rate of Divergence', fontsize=12)
    ax.set_title('Critical Timesteps (Peaks = Phase Transitions)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'temporal_phase_diagram.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Saved: temporal_phase_diagram.png")

    # Report critical timesteps
    print(f"\n  Critical Timesteps Detected:")
    print(f"    SD-Monolithic divergence peaks: {peaks_mono[:5]}... ({len(peaks_mono)} total)")
    print(f"    SD-A divergence peaks:          {peaks_a[:5]}... ({len(peaks_a)} total)")
    print(f"    SD-B divergence peaks:          {peaks_b[:5]}... ({len(peaks_b)} total)")


# ============================================================================
# Integration Function
# ============================================================================

def generate_enhanced_visualizations(results_dict: Dict, output_dir: str, config):
    """
    Generate all enhanced visualizations

    Args:
        results_dict: Results from CompositionExperimentSuite
        output_dir: Output directory
        config: Experiment configuration
    """
    print("\n" + "="*80)
    print("GENERATING ENHANCED VISUALIZATIONS")
    print("="*80)

    # 1. Unified latent space (2D + 3D)
    plot_unified_latent_space_2d3d(results_dict, output_dir, config)

    # 2. Trajectory evolution (2D + 3D)
    # Visualize first 3 runs
    for run_idx in range(min(3, config.num_runs)):
        plot_trajectory_evolution_2d3d(results_dict, output_dir, config,
                                       run_idx=run_idx, sample_idx=0)

    # 3. Temporal phase diagram
    plot_temporal_phase_diagram(results_dict, output_dir, config)

    print("\n" + "="*80)
    print("ENHANCED VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - unified_latent_space_2d.png")
    print("  - unified_latent_space_3d.png")
    print("  - unified_latent_space_3d_interactive.html (3D, rotatable)")
    print("  - trajectory_evolution_2d_run*.png")
    print("  - trajectory_evolution_3d_run*.png")
    print("  - trajectory_evolution_3d_interactive_run*.html (3D, rotatable)")
    print("  - temporal_phase_diagram.png")
    print("\nKey findings:")
    print("  → See unified plots for relative positioning of all conditions")
    print("  → See trajectories for path evolution through latent space")
    print("  → See phase diagram for critical timesteps where divergence occurs")


if __name__ == "__main__":
    print("This module provides enhanced visualization tools.")
    print("Import and use with your experiment results.")
    print("\nExample:")
    print("""
from notebooks.enhanced_visualizations import generate_enhanced_visualizations
from notebooks.composition_experiments import CompositionExperimentSuite

suite = CompositionExperimentSuite(config)
suite.run_all_experiments()

generate_enhanced_visualizations(
    suite.results,
    suite.output_dir,
    suite.config
)
""")
