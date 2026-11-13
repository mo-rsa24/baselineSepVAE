"""
Advanced Manifold Geometry Analysis for SUPERDIFF Composition

This module provides specialized tools for investigating whether SUPERDIFF
composition creates off-manifold trajectories and how this relates to hybridization.

Key analyses:
1. Local intrinsic dimensionality estimation
2. Geodesic vs. Euclidean distance comparisons
3. Curvature estimation along trajectories
4. Manifold tangent space alignment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple
import torch.nn.functional as F


class ManifoldGeometryAnalyzer:
    """Advanced manifold geometry analysis tools"""

    def __init__(self, latent_samples: torch.Tensor):
        """
        Args:
            latent_samples: Tensor of shape (N, D) where N is number of samples
                           and D is the flattened latent dimension
        """
        self.samples = latent_samples.cpu().numpy()
        self.n_samples, self.dim = self.samples.shape

    def estimate_intrinsic_dimension(self, k: int = 20, method: str = 'mle') -> float:
        """
        Estimate the intrinsic dimensionality of the latent manifold using
        Maximum Likelihood Estimation (MLE) based on nearest neighbor distances.

        This helps determine if SUPERDIFF samples lie on a different dimensional
        subspace than individual prompts.

        Args:
            k: Number of nearest neighbors to consider (will be capped at n_samples - 1)
            method: 'mle' for maximum likelihood or 'correlation' for correlation dimension

        Returns:
            Estimated intrinsic dimension (or np.nan if too few samples)

        Reference: "Intrinsic Dimensionality Estimation" (Levina & Bickel, 2005)
        """
        # Ensure k is valid for the number of samples
        n_samples = self.samples.shape[0]
        k = min(k, n_samples - 1)

        if k < 2:
            # Cannot estimate intrinsic dimension with too few neighbors
            return np.nan

        if method == 'mle':
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.samples)
            distances, indices = nbrs.kneighbors(self.samples)

            # Remove the point itself (distance = 0)
            distances = distances[:, 1:]

            # MLE estimate: d ≈ (k-1) / Σ log(r_k / r_i)
            r_k = distances[:, -1:]  # Distance to k-th neighbor
            ratios = r_k / (distances[:, :-1] + 1e-10)
            log_ratios = np.log(ratios + 1e-10)

            dimensions = (k - 2) / (log_ratios.sum(axis=1) + 1e-10)

            # Return median to avoid outliers
            return np.median(dimensions)

        elif method == 'correlation':
            # Correlation dimension: log N(r) vs log r
            dists = pdist(self.samples)
            hist, bin_edges = np.histogram(dists, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Cumulative count
            cum_count = np.cumsum(hist)
            valid = cum_count > 0

            if valid.sum() < 2:
                return np.nan

            log_r = np.log(bin_centers[valid])
            log_N = np.log(cum_count[valid])

            # Linear fit in log-log plot gives correlation dimension
            coeffs = np.polyfit(log_r, log_N, 1)
            return coeffs[0]

        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_local_pca_alignment(self, other_samples: np.ndarray, k: int = 10) -> Tuple[float, float]:
        """
        Compare local tangent spaces by computing PCA on k-nearest neighbors
        and measuring alignment between principal directions.

        This tests whether SUPERDIFF samples lie in the same tangent space as
        individual prompt samples, or if they've moved to a different region.

        Args:
            other_samples: Comparison samples (e.g., individual prompt samples)
            k: Number of neighbors for local PCA

        Returns:
            (mean_alignment, std_alignment): Mean and std of alignment scores
        """
        from sklearn.decomposition import PCA

        nbrs_self = NearestNeighbors(n_neighbors=k).fit(self.samples)
        nbrs_other = NearestNeighbors(n_neighbors=k).fit(other_samples)

        alignments = []

        for sample in self.samples:
            # Get local neighborhood in both spaces
            _, idx_self = nbrs_self.kneighbors([sample])
            _, idx_other = nbrs_other.kneighbors([sample])

            local_self = self.samples[idx_self[0]]
            local_other = other_samples[idx_other[0]]

            # Compute local PCAs
            pca_self = PCA(n_components=min(5, k))
            pca_other = PCA(n_components=min(5, k))

            pca_self.fit(local_self)
            pca_other.fit(local_other)

            # Measure alignment of first principal component
            pc1_self = pca_self.components_[0]
            pc1_other = pca_other.components_[0]

            # Cosine similarity (absolute value because direction can flip)
            alignment = np.abs(np.dot(pc1_self, pc1_other))
            alignments.append(alignment)

        return np.mean(alignments), np.std(alignments)

    def estimate_curvature_along_path(self, trajectory: torch.Tensor) -> np.ndarray:
        """
        Estimate manifold curvature along a trajectory using the Menger curvature.

        Menger curvature for three consecutive points approximates the curvature
        of the circle passing through them. High curvature indicates sharp turns,
        which may suggest off-manifold shortcuts.

        Args:
            trajectory: Tensor of shape (T, D) representing a path through latent space

        Returns:
            Array of curvatures at each point (except first and last)

        Reference: Menger curvature for discrete curves
        """
        traj = trajectory.cpu().numpy()
        T = len(traj)

        if T < 3:
            return np.array([])

        curvatures = []

        for i in range(1, T - 1):
            p1, p2, p3 = traj[i-1], traj[i], traj[i+1]

            # Menger curvature: κ = 4·Area(triangle) / (|p1-p2|·|p2-p3|·|p3-p1|)
            # Area = 0.5·||(p2-p1) × (p3-p1)||

            a = p2 - p1
            b = p3 - p1
            c = p3 - p2

            # For high-dimensional spaces, use Gram determinant approach
            cross_norm = np.sqrt(np.dot(a, a) * np.dot(b, b) - np.dot(a, b)**2)
            area = 0.5 * cross_norm

            side1 = np.linalg.norm(a)
            side2 = np.linalg.norm(c)
            side3 = np.linalg.norm(b)

            if side1 * side2 * side3 > 1e-10:
                kappa = 4 * area / (side1 * side2 * side3)
            else:
                kappa = 0

            curvatures.append(kappa)

        return np.array(curvatures)

    def compute_geodesic_approximation(self, start: np.ndarray, end: np.ndarray,
                                       n_steps: int = 20, step_size: float = 0.1) -> Tuple[float, np.ndarray]:
        """
        Approximate geodesic distance by taking small steps along the manifold.

        The geodesic is approximated by iteratively moving toward the target
        while projecting onto the local tangent space (estimated via PCA of neighbors).

        Args:
            start: Starting point
            end: End point
            n_steps: Number of discrete steps
            step_size: Size of each step

        Returns:
            (geodesic_distance, path): Approximated geodesic length and path points
        """
        from sklearn.decomposition import PCA

        current = start.copy()
        path = [current.copy()]
        total_distance = 0.0

        # Fit PCA on all samples to get manifold structure
        # n_components must be <= min(n_samples, n_features)
        n_samples, n_features = self.samples.shape
        n_components = min(50, n_samples, n_features)

        if n_components < 2:
            # Cannot do PCA with less than 2 components, fall back to Euclidean
            return np.linalg.norm(end - start), np.array([start, end])

        pca = PCA(n_components=n_components)
        pca.fit(self.samples)

        for _ in range(n_steps):
            # Direction toward target
            direction = end - current
            direction_norm = np.linalg.norm(direction)

            if direction_norm < 1e-6:
                break

            direction = direction / direction_norm

            # Project direction onto tangent space (principal components)
            direction_proj = pca.transform([direction])[0]
            direction_proj = pca.inverse_transform([direction_proj])[0]
            direction_proj = direction_proj / (np.linalg.norm(direction_proj) + 1e-10)

            # Take step
            step = step_size * direction_norm / n_steps
            next_point = current + step * direction_proj

            total_distance += np.linalg.norm(next_point - current)
            current = next_point
            path.append(current.copy())

        return total_distance, np.array(path)


def analyze_composition_geometry(
    latents_mono: torch.Tensor,
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    latents_superdiff: torch.Tensor,
    output_dir: str
):
    """
    Comprehensive manifold geometry analysis comparing different composition methods.

    Args:
        latents_mono: Monolithic prompt samples (N, D)
        latents_a: Individual prompt A samples (N, D)
        latents_b: Individual prompt B samples (N, D)
        latents_superdiff: SUPERDIFF composition samples (N, D)
        output_dir: Directory to save results
    """
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MANIFOLD GEOMETRY ANALYSIS")
    print("="*80)

    # Initialize analyzers
    analyzer_mono = ManifoldGeometryAnalyzer(latents_mono)
    analyzer_a = ManifoldGeometryAnalyzer(latents_a)
    analyzer_b = ManifoldGeometryAnalyzer(latents_b)
    analyzer_sd = ManifoldGeometryAnalyzer(latents_superdiff)

    # 1. Intrinsic dimensionality
    print("\n1. INTRINSIC DIMENSIONALITY ESTIMATION")
    print("-" * 80)

    dim_mono_mle = analyzer_mono.estimate_intrinsic_dimension(method='mle')
    dim_a_mle = analyzer_a.estimate_intrinsic_dimension(method='mle')
    dim_b_mle = analyzer_b.estimate_intrinsic_dimension(method='mle')
    dim_sd_mle = analyzer_sd.estimate_intrinsic_dimension(method='mle')

    print(f"  MLE estimates:")
    print(f"    Monolithic:  {dim_mono_mle:8.2f}")
    print(f"    Prompt A:    {dim_a_mle:8.2f}")
    print(f"    Prompt B:    {dim_b_mle:8.2f}")
    print(f"    SUPERDIFF:   {dim_sd_mle:8.2f}")

    dim_mono_corr = analyzer_mono.estimate_intrinsic_dimension(method='correlation')
    dim_a_corr = analyzer_a.estimate_intrinsic_dimension(method='correlation')
    dim_b_corr = analyzer_b.estimate_intrinsic_dimension(method='correlation')
    dim_sd_corr = analyzer_sd.estimate_intrinsic_dimension(method='correlation')

    print(f"\n  Correlation dimension estimates:")
    print(f"    Monolithic:  {dim_mono_corr:8.2f}")
    print(f"    Prompt A:    {dim_a_corr:8.2f}")
    print(f"    Prompt B:    {dim_b_corr:8.2f}")
    print(f"    SUPERDIFF:   {dim_sd_corr:8.2f}")

    # 2. Local tangent space alignment
    print("\n2. LOCAL TANGENT SPACE ALIGNMENT")
    print("-" * 80)

    align_sd_a_mean, align_sd_a_std = analyzer_sd.compute_local_pca_alignment(
        analyzer_a.samples
    )
    align_sd_b_mean, align_sd_b_std = analyzer_sd.compute_local_pca_alignment(
        analyzer_b.samples
    )
    align_sd_mono_mean, align_sd_mono_std = analyzer_sd.compute_local_pca_alignment(
        analyzer_mono.samples
    )

    print(f"  SUPERDIFF tangent alignment with:")
    print(f"    Prompt A:     {align_sd_a_mean:.4f} ± {align_sd_a_std:.4f}")
    print(f"    Prompt B:     {align_sd_b_mean:.4f} ± {align_sd_b_std:.4f}")
    print(f"    Monolithic:   {align_sd_mono_mean:.4f} ± {align_sd_mono_std:.4f}")
    print(f"\n  (Values close to 1.0 indicate aligned tangent spaces)")

    # 3. Geodesic vs Euclidean distances
    print("\n3. GEODESIC VS EUCLIDEAN DISTANCES")
    print("-" * 80)

    # Sample a few point pairs for analysis
    n_samples = min(10, len(latents_superdiff))
    euclidean_dists = []
    geodesic_dists = []

    centroid_a = latents_a.mean(dim=0).cpu().numpy()
    centroid_b = latents_b.mean(dim=0).cpu().numpy()

    for i in range(n_samples):
        sd_sample = latents_superdiff[i].cpu().numpy()

        # Euclidean distance from SD to midpoint of A and B
        midpoint = (centroid_a + centroid_b) / 2
        euclidean_dist = np.linalg.norm(sd_sample - midpoint)

        # Approximate geodesic distance
        # Use combined A+B samples as manifold reference
        combined_samples = torch.cat([latents_a, latents_b], dim=0)
        combined_analyzer = ManifoldGeometryAnalyzer(combined_samples)

        geodesic_dist, _ = combined_analyzer.compute_geodesic_approximation(
            sd_sample, midpoint, n_steps=20
        )

        euclidean_dists.append(euclidean_dist)
        geodesic_dists.append(geodesic_dist)

    euclidean_dists = np.array(euclidean_dists)
    geodesic_dists = np.array(geodesic_dists)
    ratios = geodesic_dists / (euclidean_dists + 1e-10)

    print(f"  Distance from SUPERDIFF samples to A-B midpoint:")
    print(f"    Euclidean (mean):     {euclidean_dists.mean():.4f} ± {euclidean_dists.std():.4f}")
    print(f"    Geodesic (mean):      {geodesic_dists.mean():.4f} ± {geodesic_dists.std():.4f}")
    print(f"    Ratio (geodesic/euclidean): {ratios.mean():.4f} ± {ratios.std():.4f}")
    print(f"\n  (Ratio > 1 suggests off-manifold shortcuts)")

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Intrinsic dimensions
    ax = axes[0, 0]
    conditions = ['Monolithic', 'Prompt A', 'Prompt B', 'SUPERDIFF']
    mle_dims = [dim_mono_mle, dim_a_mle, dim_b_mle, dim_sd_mle]
    corr_dims = [dim_mono_corr, dim_a_corr, dim_b_corr, dim_sd_corr]

    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width/2, mle_dims, width, label='MLE', color='steelblue')
    ax.bar(x + width/2, corr_dims, width, label='Correlation', color='coral')

    ax.set_ylabel('Estimated Dimension')
    ax.set_title('Intrinsic Dimensionality Estimates')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Tangent space alignment
    ax = axes[0, 1]

    alignments = [align_sd_mono_mean, align_sd_a_mean, align_sd_b_mean]
    stds = [align_sd_mono_std, align_sd_a_std, align_sd_b_std]
    labels = ['SD vs Mono', 'SD vs A', 'SD vs B']

    bars = ax.bar(range(len(alignments)), alignments, color=['green', 'blue', 'orange'])
    ax.errorbar(range(len(alignments)), alignments, yerr=stds, fmt='none',
               color='black', capsize=5)

    ax.set_ylabel('Tangent Space Alignment')
    ax.set_title('Local Tangent Space Alignment with SUPERDIFF')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Geodesic vs Euclidean
    ax = axes[1, 0]

    ax.scatter(euclidean_dists, geodesic_dists, s=100, alpha=0.6, color='purple')

    # Plot diagonal (geodesic = euclidean)
    max_dist = max(euclidean_dists.max(), geodesic_dists.max())
    ax.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='Geodesic = Euclidean')

    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Geodesic Distance (approx)')
    ax.set_title('Geodesic vs Euclidean Distance to Midpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Distribution of geodesic/euclidean ratios
    ax = axes[1, 1]

    ax.hist(ratios, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Ratio = 1.0')
    ax.axvline(x=ratios.mean(), color='darkblue', linestyle='-', linewidth=2,
              label=f'Mean = {ratios.mean():.3f}')

    ax.set_xlabel('Geodesic / Euclidean Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Distance Ratios')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'manifold_geometry_analysis.png', dpi=150)
    plt.close()

    print(f"\n  Saved: manifold_geometry_analysis.png")

    # Save numerical results
    with open(output_path / 'manifold_geometry_results.txt', 'w') as f:
        f.write("MANIFOLD GEOMETRY ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("1. INTRINSIC DIMENSIONALITY (MLE)\n")
        f.write(f"   Monolithic:  {dim_mono_mle:8.2f}\n")
        f.write(f"   Prompt A:    {dim_a_mle:8.2f}\n")
        f.write(f"   Prompt B:    {dim_b_mle:8.2f}\n")
        f.write(f"   SUPERDIFF:   {dim_sd_mle:8.2f}\n\n")

        f.write("2. INTRINSIC DIMENSIONALITY (Correlation)\n")
        f.write(f"   Monolithic:  {dim_mono_corr:8.2f}\n")
        f.write(f"   Prompt A:    {dim_a_corr:8.2f}\n")
        f.write(f"   Prompt B:    {dim_b_corr:8.2f}\n")
        f.write(f"   SUPERDIFF:   {dim_sd_corr:8.2f}\n\n")

        f.write("3. TANGENT SPACE ALIGNMENT\n")
        f.write(f"   SD vs Monolithic: {align_sd_mono_mean:.4f} ± {align_sd_mono_std:.4f}\n")
        f.write(f"   SD vs A:          {align_sd_a_mean:.4f} ± {align_sd_a_std:.4f}\n")
        f.write(f"   SD vs B:          {align_sd_b_mean:.4f} ± {align_sd_b_std:.4f}\n\n")

        f.write("4. GEODESIC VS EUCLIDEAN DISTANCES\n")
        f.write(f"   Mean Euclidean:   {euclidean_dists.mean():.4f} ± {euclidean_dists.std():.4f}\n")
        f.write(f"   Mean Geodesic:    {geodesic_dists.mean():.4f} ± {geodesic_dists.std():.4f}\n")
        f.write(f"   Mean Ratio:       {ratios.mean():.4f} ± {ratios.std():.4f}\n")

    print(f"  Saved: manifold_geometry_results.txt")

    print("\n" + "="*80)
    print("MANIFOLD ANALYSIS COMPLETE")
    print("="*80)


def analyze_trajectory_curvature(trajectories_dict: dict, output_dir: str):
    """
    Analyze curvature along diffusion trajectories to detect off-manifold behavior.

    High curvature may indicate the trajectory is taking shortcuts through
    off-manifold regions.

    Args:
        trajectories_dict: Dictionary with keys for conditions and LatentTrajectoryCollector values
        output_dir: Output directory
    """
    from pathlib import Path

    output_path = Path(output_dir)

    print("\n" + "="*80)
    print("TRAJECTORY CURVATURE ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Analyze curvature for each condition
    curvatures_all = {}

    for condition, color, label in [
        ('monolithic', 'green', 'Monolithic'),
        ('prompt_a', 'blue', 'Prompt A'),
        ('prompt_b', 'orange', 'Prompt B'),
        ('superdiff', 'red', 'SUPERDIFF')
    ]:
        if condition not in trajectories_dict:
            continue

        curvatures_condition = []

        for traj in trajectories_dict[condition]:
            # Get trajectory for first sample in batch
            traj_flat = traj.trajectories[:, 0, :].flatten(1)  # (T, D)

            analyzer = ManifoldGeometryAnalyzer(traj_flat)
            curv = analyzer.estimate_curvature_along_path(traj_flat)
            curvatures_condition.append(curv)

        curvatures_all[condition] = curvatures_condition

        # Plot mean curvature over time
        ax = axes[0, 0]
        if curvatures_condition:
            # Stack and compute statistics
            curv_array = np.array([c for c in curvatures_condition if len(c) > 0])
            if len(curv_array) > 0:
                curv_mean = curv_array.mean(axis=0)
                curv_std = curv_array.std(axis=0)
                steps = np.arange(len(curv_mean))

                ax.plot(steps, curv_mean, color=color, label=label, linewidth=2)
                ax.fill_between(steps, curv_mean - curv_std, curv_mean + curv_std,
                              color=color, alpha=0.2)

    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Menger Curvature')
    ax.set_title('Mean Trajectory Curvature Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot distribution of curvatures
    ax = axes[0, 1]

    for condition, color, label in [
        ('monolithic', 'green', 'Monolithic'),
        ('superdiff', 'red', 'SUPERDIFF')
    ]:
        if condition in curvatures_all:
            all_curvs = np.concatenate([c for c in curvatures_all[condition] if len(c) > 0])
            ax.hist(np.log10(all_curvs + 1e-10), bins=50, alpha=0.5,
                   color=color, label=label, density=True)

    ax.set_xlabel('log10(Curvature)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Trajectory Curvatures')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot cumulative curvature (total bending)
    ax = axes[1, 0]

    for condition, color, label in [
        ('monolithic', 'green', 'Monolithic'),
        ('prompt_a', 'blue', 'Prompt A'),
        ('prompt_b', 'orange', 'Prompt B'),
        ('superdiff', 'red', 'SUPERDIFF')
    ]:
        if condition in curvatures_all:
            total_curvatures = [c.sum() for c in curvatures_all[condition] if len(c) > 0]
            if total_curvatures:
                ax.bar(label, np.mean(total_curvatures), color=color,
                      yerr=np.std(total_curvatures), capsize=5)

    ax.set_ylabel('Total Curvature (sum)')
    ax.set_title('Total Trajectory Bending')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot curvature at different stages
    ax = axes[1, 1]

    early_steps = slice(0, 50)
    mid_steps = slice(50, 150)
    late_steps = slice(150, None)

    for condition, color, label in [
        ('monolithic', 'green', 'Monolithic'),
        ('superdiff', 'red', 'SUPERDIFF')
    ]:
        if condition in curvatures_all:
            curvs = np.array([c for c in curvatures_all[condition] if len(c) > 0])
            if len(curvs) > 0:
                early_mean = curvs[:, early_steps].mean() if curvs.shape[1] > 50 else 0
                mid_mean = curvs[:, mid_steps].mean() if curvs.shape[1] > 150 else 0
                late_mean = curvs[:, late_steps].mean() if curvs.shape[1] > 150 else 0

                x = ['Early\n(0-50)', 'Mid\n(50-150)', 'Late\n(150+)']
                y = [early_mean, mid_mean, late_mean]

                ax.plot(x, y, marker='o', linewidth=2, markersize=10,
                       color=color, label=label)

    ax.set_ylabel('Mean Curvature')
    ax.set_title('Curvature at Different Diffusion Stages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path / 'trajectory_curvature_analysis.png', dpi=150)
    plt.close()

    print(f"  Saved: trajectory_curvature_analysis.png")

    print("\n" + "="*80)
    print("CURVATURE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    print("This module provides manifold geometry analysis tools.")
    print("Import and use the functions in your experiment scripts.")
    print("\nExample usage:")
    print("""
from manifold_geometry_analysis import analyze_composition_geometry

analyze_composition_geometry(
    latents_mono,
    latents_a,
    latents_b,
    latents_superdiff,
    output_dir='experiments/manifold_analysis'
)
""")
