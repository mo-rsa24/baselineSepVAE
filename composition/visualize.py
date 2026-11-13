import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from composition.utils import decode_image  # Ensure this import exists

# --- VISUAL STYLE CONSTANTS ---
COLORS = {'normal': '#1f77b4', 'tb': '#ff7f0e', 'comp': '#2ca02c'}  # Blue, Orange, Green
STYLES = {'normal': '--', 'tb': '--', 'comp': '-'}
ALPHAS = {'trace': 0.15, 'centroid': 1.0, 'overlay': 0.3}


def setup_plot(title, xlabel, ylabel):
    """Helper to standardize plot styling."""
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.2)


# ==============================================================================
# 1. SEMANTIC VALIDITY (Is it a real lung?)
#    Focus: Reference Manifolds, Off-Manifold Detection
# ==============================================================================
def plot_disease_difference(img_normal, img_comp, output_path):
    """
    Level 1 Diagnostic: The 'Difference Map'.
    Subtracts the Normal Baseline from the Composition to isolate the 'Disease Trace'.

    Args:
        img_normal: Numpy array (H, W) or (H, W, C) - The 'Health' Baseline
        img_comp:   Numpy array (H, W) or (H, W, C) - The 'Sick' Result
    """
    print(f"Generating Disease Difference Map -> {output_path}")

    # Normalize to 0-1 floats for math
    norm = img_normal.astype(np.float32) / 255.0
    comp = img_comp.astype(np.float32) / 255.0

    # 1. Compute Absolute Difference
    # We essentially ask: |Composition - Normal|
    diff = np.abs(comp - norm)

    # Enhance contrast (Lesions can be subtle 10-20% opacity changes)
    # We clip the top 20% of intensity to make the heatmap brighter
    vmax = np.percentile(diff, 99) if np.max(diff) > 0 else 1.0

    # 2. Plot
    plt.figure(figsize=(15, 5))

    # A. Normal
    plt.subplot(1, 3, 1)
    plt.imshow(img_normal, cmap='gray')
    plt.title("Baseline (Normal)", fontsize=12)
    plt.axis('off')

    # B. Composition
    plt.subplot(1, 3, 2)
    plt.imshow(img_comp, cmap='gray')
    plt.title("Composition (SuperDiff)", fontsize=12)
    plt.axis('off')

    # C. The Trace (Heatmap)
    plt.subplot(1, 3, 3)
    # 'inferno' or 'magma' are good for medical attention maps
    plt.imshow(diff, cmap='inferno', vmin=0, vmax=vmax)
    plt.colorbar(label="Pixel Intensity Delta")
    plt.title("Level 1: Disease Difference Map", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_x0_manifold_diagnostics(traj_n, traj_t, traj_comp, output_path_base):
    """
    Checks if Composition stays on the valid manifold defined by Normal/TB baselines.
    Outputs: Reference PCA Projection (Qualitative) & Reconstruction Error (Quantitative)
    """
    print(">> Generating Validity Diagnostics...")

    # Flatten & Standardize
    def flatten(t): return np.array(t).reshape(-1, np.prod(t[0].shape[1:]))

    flat_n, flat_t, flat_c = flatten(traj_n), flatten(traj_t), flatten(traj_comp)

    # Fit ONLY on Baselines
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(np.concatenate([flat_n, flat_t], axis=0))
    pca = PCA(n_components=10).fit(baseline_scaled)
    print(f"PCA Fitted. Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")

    # --- Plot A: Reference PCA ---
    def project(traj):
        means = np.mean(np.array(traj), axis=1).reshape(len(traj), -1)
        return pca.transform(scaler.transform(means))

    p_n, p_t, p_c = project(traj_n), project(traj_t), project(traj_comp)

    setup_plot("Validity: Reference Manifold Projection", "PC1 (Valid Lungs)", "PC2 (Valid Lungs)")
    plt.plot(p_n[:, 0], p_n[:, 1], label='Normal Baseline', c=COLORS['normal'], ls=STYLES['normal'], alpha=0.5)
    plt.plot(p_t[:, 0], p_t[:, 1], label='TB Baseline', c=COLORS['tb'], ls=STYLES['tb'], alpha=0.5)
    plt.plot(p_c[:, 0], p_c[:, 1], label='Composition', c=COLORS['comp'], lw=3)
    plt.scatter(p_c[0, 0], p_c[0, 1], c='black', label='Start')
    plt.scatter(p_c[-1, 0], p_c[-1, 1], c=COLORS['comp'], marker='*', s=200, label='End')
    plt.legend()
    plt.savefig(f"{output_path_base}_pca_x0.png")
    plt.close()

    # --- Plot B: Reconstruction Error (Off-Manifold Test) ---
    def get_error(traj):
        traj = np.array(traj)
        flat = traj.reshape(traj.shape[0], traj.shape[1], -1)
        errs = []
        for step in flat:
            scaled = scaler.transform(step)
            recon = pca.inverse_transform(pca.transform(scaled))
            errs.append(np.mean(np.linalg.norm(scaled - recon, axis=1)))
        return errs

    err_c = get_error(traj_comp)
    time = np.linspace(1, 0, len(err_c))

    setup_plot("Validity: Distance to Manifold", "Diffusion Time (t)", "Reconstruction Error (MSE)")
    plt.plot(time, err_c, c='red', lw=2)
    plt.gca().invert_xaxis()
    plt.savefig(f"{output_path_base}_reconstruction_error.png")
    plt.close()


def plot_dynamics_suite(traj_n, traj_t, traj_comp, manager, ae_model=None, ae_params=None, latent_scale=1.0):
    """
    Runs the full suite of dynamics visualizations.
    """
    print(">> Generating Temporal Dynamics Suite...")

    # A. Centroid Dynamics (Signal vs Noise) - Uses Strict Baseline PCA
    plot_dynamics_with_centroids_x0(traj_n, traj_t, traj_comp, manager.get_path("dynamics", "centroids_x0.png"))

    # B. Stepwise Displacement (Stability)
    plot_stepwise_displacement(traj_n, traj_t, traj_comp, manager.get_path("dynamics", "displacement_stability.png"))

    # C. Bifurcation Analysis (Timing)
    plot_bifurcation_metrics(traj_n, traj_t, traj_comp, manager.get_path("dynamics", "bifurcation_timing.png"))

    # D. Vector Field (Flow)
    plot_vector_field_dynamics(traj_n, traj_t, traj_comp, manager.get_path("dynamics", "vector_field.png"))

    # E. 3D Trajectory (Orbital Mechanics)
    plot_trajectory_dynamics_3d(traj_n, traj_t, traj_comp, manager.get_path("dynamics", "trajectory_3d.png"))

    # F. Visual Overlay (Grounding)
    if ae_model:
        plot_manifold_with_xray_overlays(
            traj_n, traj_t, traj_comp, ae_model, ae_params, latent_scale,
            manager.get_path("dynamics", "manifold_overlay.png")
        )

    # G. Difference Vector Dynamics (Disease-Sensitive Projection)
    plot_difference_vector_dynamics(
        traj_n, traj_t, traj_comp, manager.get_path("dynamics", "diff_vec")
    )


# ==============================================================================
# 2. TEMPORAL DYNAMICS (How does it evolve?)
#    Focus: Trajectories, Velocity, Stability, Bifurcation
# ==============================================================================

def plot_dynamics_with_centroids_x0(traj_n, traj_t, traj_comp, output_path):
    """
    Visualizes Signal (Centroid) vs Noise (Individual) using x0 predictions.
    Uses rigorous baseline-only PCA fitting.
    """
    print(f"Generating Rigorous Centroid Dynamics -> {output_path}")

    # 1. Flatten Data
    def flatten_trajectory(t):
        arr = np.array(t)  # (Steps, Batch, H, W, C)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)  # (Steps, Batch, Features)

    raw_a = flatten_trajectory(traj_n)
    raw_b = flatten_trajectory(traj_t)
    raw_sd = flatten_trajectory(traj_comp)

    # 2. Compute Centroids
    cent_a = np.mean(raw_a, axis=1)
    cent_b = np.mean(raw_b, axis=1)
    cent_sd = np.mean(raw_sd, axis=1)

    # 3. Standardize & Fit PCA (On Baselines Only)
    baseline_cents = np.concatenate([cent_a, cent_b], axis=0)
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline_cents)
    pca = PCA(n_components=2)
    pca.fit(baseline_scaled)

    # 4. Transform EVERYTHING
    def transform_set(raw_data, centroids):
        flat_cents = scaler.transform(centroids)
        proj_cents = pca.transform(flat_cents)
        # Transform First Individual Sample (for noise viz)
        flat_ind = scaler.transform(raw_data[:, 0, :])
        proj_ind = pca.transform(flat_ind)
        return proj_cents, proj_ind

    p_cent_a, p_ind_a = transform_set(raw_a, cent_a)
    p_cent_b, p_ind_b = transform_set(raw_b, cent_b)
    p_cent_sd, p_ind_sd = transform_set(raw_sd, cent_sd)

    # 5. Plot
    setup_plot("Dynamics (x0): Signal vs Noise on Valid Manifold", "PC1 (Baseline Variance)", "PC2 (Baseline Variance)")

    # Individual Paths (Faint)
    plt.plot(p_ind_a[:, 0], p_ind_a[:, 1], c=COLORS['normal'], alpha=ALPHAS['trace'], lw=1, label='Individual (Normal)')
    plt.plot(p_ind_sd[:, 0], p_ind_sd[:, 1], c=COLORS['comp'], alpha=ALPHAS['trace'], lw=1, label='Individual (Comp)')

    # Centroids (Bold)
    plt.plot(p_cent_a[:, 0], p_cent_a[:, 1], c=COLORS['normal'], lw=3, ls=STYLES['normal'], label='Normal Centroid')
    plt.plot(p_cent_b[:, 0], p_cent_b[:, 1], c=COLORS['tb'], lw=3, ls=STYLES['tb'], label='TB Centroid')
    plt.plot(p_cent_sd[:, 0], p_cent_sd[:, 1], c=COLORS['comp'], lw=4, label='Composition Centroid')

    # Annotations
    plt.scatter(p_cent_sd[0, 0], p_cent_sd[0, 1], c='black', s=50, label='Start (t=1)')
    plt.scatter(p_cent_sd[-1, 0], p_cent_sd[-1, 1], c=COLORS['comp'], marker='*', s=300, edgecolors='black',
                label='End (t=0)')

    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stepwise_displacement(traj_n, traj_t, traj_comp, output_path):
    """
    Measures how much the prediction x0 changes at each step.
    Large jumps = Instability. Smooth decay = Convergence.
    """
    print("Computing Stepwise Displacement...")

    def get_displacement_curve(traj):
        arr = np.array(traj)
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        diffs = np.diff(flat, axis=0)  # (Steps-1, Batch, Features)
        norms = np.linalg.norm(diffs, axis=2)
        return np.mean(norms, axis=1), np.std(norms, axis=1)

    mu_n, std_n = get_displacement_curve(traj_n)
    mu_t, std_t = get_displacement_curve(traj_t)
    mu_c, std_c = get_displacement_curve(traj_comp)

    steps = len(mu_c)
    time_axis = np.linspace(1, 0, steps)

    setup_plot("Visualization 4: Stepwise Displacement (|x0_t - x0_{t-1}|)", "Diffusion Time (t)", "Displacement Norm")

    plt.plot(time_axis, mu_n, label='Normal', c=COLORS['normal'], alpha=0.5)
    plt.fill_between(time_axis, mu_n - std_n, mu_n + std_n, color=COLORS['normal'], alpha=0.1)

    plt.plot(time_axis, mu_t, label='TB', c=COLORS['tb'], alpha=0.5)
    plt.plot(time_axis, mu_c, label='Composition', c=COLORS['comp'], linewidth=2)
    plt.fill_between(time_axis, mu_c - std_c, mu_c + std_c, color=COLORS['comp'], alpha=0.2)

    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_bifurcation_metrics(hist_a, hist_b, hist_sd, output_path):
    """
    Quantifies the Bifurcation Point by measuring distance to baselines.
    """
    print(f"Generating Bifurcation Metric Plot -> {output_path}")

    def get_centroid(h):
        arr = np.array(h)
        return np.mean(arr.reshape(arr.shape[0], arr.shape[1], -1), axis=1)

    c_a, c_b, c_sd = get_centroid(hist_a), get_centroid(hist_b), get_centroid(hist_sd)

    d_normal = np.linalg.norm(c_sd - c_a, axis=1)
    d_tb = np.linalg.norm(c_sd - c_b, axis=1)

    time = np.linspace(1.0, 0.0, len(d_normal))

    setup_plot("Bifurcation Analysis: When does Composition diverge?", "Diffusion Time (t)", "Euclidean Distance")
    plt.plot(time, d_normal, label='Dist(Comp, Normal)', c=COLORS['normal'], lw=2)
    plt.plot(time, d_tb, label='Dist(Comp, TB)', c=COLORS['tb'], lw=2)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_vector_field_dynamics(hist_a, hist_b, hist_sd, output_path):
    """
    Visualizes the 'Flow' of generation by projecting velocity vectors into PCA space.
    """
    print(f"Generating Vector Field Dynamics -> {output_path}")

    def flatten_hist(h):
        arr = np.array(h)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    # Use centroids for mean field flow
    raw_a = flatten_hist(hist_a).mean(axis=1)
    raw_b = flatten_hist(hist_b).mean(axis=1)
    raw_sd = flatten_hist(hist_sd).mean(axis=1)

    all_pos = np.concatenate([raw_a, raw_b, raw_sd], axis=0)
    pca = PCA(n_components=2).fit(all_pos)

    def get_velocity(pos): return np.diff(pos, axis=0)

    vel_a, vel_b, vel_sd = get_velocity(raw_a), get_velocity(raw_b), get_velocity(raw_sd)

    # Project Positions & Velocities
    pos_a_2d = pca.transform(raw_a[:-1])
    pos_b_2d = pca.transform(raw_b[:-1])
    pos_sd_2d = pca.transform(raw_sd[:-1])

    vec_a_2d = vel_a @ pca.components_.T
    vec_b_2d = vel_b @ pca.components_.T
    vec_sd_2d = vel_sd @ pca.components_.T

    plt.figure(figsize=(14, 10))
    # Faint background trajectories
    plt.plot(pos_a_2d[:, 0], pos_a_2d[:, 1], c=COLORS['normal'], alpha=0.2)
    plt.plot(pos_b_2d[:, 0], pos_b_2d[:, 1], c=COLORS['tb'], alpha=0.2)

    # Quivers (Subsampled)
    step = 5
    plt.quiver(pos_a_2d[::step, 0], pos_a_2d[::step, 1], vec_a_2d[::step, 0], vec_a_2d[::step, 1],
               color=COLORS['normal'], alpha=0.4, label='Normal Flow', scale=20, width=0.002)
    plt.quiver(pos_b_2d[::step, 0], pos_b_2d[::step, 1], vec_b_2d[::step, 0], vec_b_2d[::step, 1],
               color=COLORS['tb'], alpha=0.4, label='TB Flow', scale=20, width=0.002)
    plt.quiver(pos_sd_2d[::step, 0], pos_sd_2d[::step, 1], vec_sd_2d[::step, 0], vec_sd_2d[::step, 1],
               color=COLORS['comp'], label='Comp Flow', scale=20, width=0.005)

    plt.scatter(pos_sd_2d[0, 0], pos_sd_2d[0, 1], c='black', label='Start')
    plt.scatter(pos_sd_2d[-1, 0], pos_sd_2d[-1, 1], c=COLORS['comp'], marker='*', s=200, label='End')

    plt.title("Vector Field Dynamics: The 'Flow' of Generation")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_trajectory_dynamics_3d(hist_a, hist_b, hist_sd, output_path):
    """
    3D PCA Trajectory (Rigorous Baseline Fit).
    """
    print(f"Computing Rigorous 3D Trajectory PCA -> {output_path}")

    def get_centroid_matrix(hist):
        arr = np.array(hist)
        means = np.mean(arr, axis=1)
        return means.reshape(means.shape[0], -1)

    c_a, c_b, c_sd = get_centroid_matrix(hist_a), get_centroid_matrix(hist_b), get_centroid_matrix(hist_sd)

    # Standardize & Fit PCA (Baselines Only)
    baseline_cents = np.concatenate([c_a, c_b], axis=0)
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline_cents)
    pca = PCA(n_components=3).fit(baseline_scaled)
    print(f"3D PCA Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")

    def transform(c): return pca.transform(scaler.transform(c))

    t_a, t_b, t_sd = transform(c_a), transform(c_b), transform(c_sd)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(t_a[:, 0], t_a[:, 1], t_a[:, 2], c=COLORS['normal'], label='Normal Path', alpha=0.4, ls='--')
    ax.plot(t_b[:, 0], t_b[:, 1], t_b[:, 2], c=COLORS['tb'], label='TB Path', alpha=0.4, ls='--')
    ax.plot(t_sd[:, 0], t_sd[:, 1], t_sd[:, 2], c=COLORS['comp'], linewidth=3, label='SuperDiff Path')

    ax.scatter(t_sd[0, 0], t_sd[0, 1], t_sd[0, 2], c='black', marker='o', s=50, label='Start')
    ax.scatter(t_sd[-1, 0], t_sd[-1, 1], t_sd[-1, 2], c=COLORS['comp'], marker='*', s=200, label='End')

    ax.set_title(f"3D Trajectory Dynamics\nExplained Var: {np.sum(pca.explained_variance_ratio_):.2%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    ax.view_init(elev=20, azim=135)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_manifold_with_xray_overlays(hist_a, hist_b, hist_sd, ae_model, ae_params, latent_scale_factor, output_path,
                                     num_overlays=6, zoom=0.15):
    """
    Plots Centroid Trajectory & overlays decoded X-rays.
    """
    print(f"Generating Manifold with X-ray Overlays -> {output_path}")

    # Data Prep
    def get_flat_centroid(h):
        arr = np.array(h)
        return np.mean(arr, axis=1).reshape(arr.shape[0], -1)

    c_a, c_b, c_sd = get_flat_centroid(hist_a), get_flat_centroid(hist_b), get_flat_centroid(hist_sd)

    # PCA Projection (Fit all for global map)
    all_cents = np.concatenate([c_a, c_b, c_sd], axis=0)
    pca = PCA(n_components=2).fit(all_cents)
    p_a, p_b, p_sd = pca.transform(c_a), pca.transform(c_b), pca.transform(c_sd)

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(p_a[:, 0], p_a[:, 1], c=COLORS['normal'], alpha=0.3, lw=2, ls='--', label='Normal Path')
    ax.plot(p_b[:, 0], p_b[:, 1], c=COLORS['tb'], alpha=0.3, lw=2, ls='--', label='TB Path')
    ax.plot(p_sd[:, 0], p_sd[:, 1], c=COLORS['comp'], alpha=0.8, lw=3, label='Composition Path')

    # Overlay Logic
    total_steps = len(hist_sd)
    indices = np.linspace(0, total_steps - 1, num_overlays, dtype=int)

    for i, idx in enumerate(indices):
        # Decode Sample 0
        latent_snapshot = np.array(hist_sd)[idx, 0:1, :, :, :]
        img_jax = decode_image(ae_model, ae_params, latent_snapshot, latent_scale_factor)
        img_np = np.array(img_jax).squeeze()

        imagebox = OffsetImage(img_np, zoom=zoom, cmap='gray', norm=plt.Normalize(0, 255))
        imagebox.image.axes = ax
        xy = (p_sd[idx, 0], p_sd[idx, 1])

        ab = AnnotationBbox(imagebox, xy, xybox=(30., 30.), xycoords='data', boxcoords="offset points", pad=0.3,
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=3"))
        ax.add_artist(ab)

        diffusion_time = 1.0 - (idx / total_steps)
        ax.text(xy[0], xy[1], f"t={diffusion_time:.1f}", fontsize=9, fontweight='bold', color=COLORS['comp'])

    ax.set_title("Manifold Dynamics with X-ray Overlays")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==============================================================================
# 3. STATIC GEOMETRY (Final State Distribution)
#    Focus: PCA, t-SNE, UMAP, Density Maps
# ==============================================================================

def plot_geometry_suite(latents_n, latents_t, latents_c, manager):
    """
    Visualizes the final distribution of samples using PCA, t-SNE, UMAP and Density Maps.
    """
    print(">> Generating Static Geometry Suite...")

    # A. PCA
    plot_latent_pca(latents_n, latents_t, latents_c, manager.get_path("geometry", "pca_distribution.png"))

    # B. t-SNE
    plot_latent_tsne(latents_n, latents_t, latents_c, manager.get_path("geometry", "tsne_2d.png"))

    # C. Density Landscapes (Level Sets)
    plot_density_landscapes(latents_n, latents_t, latents_c, manager.get_path("geometry", "density_contours.png"))

    # D. UMAP 2D & 3D
    plot_latent_umap(latents_n, latents_t, latents_c, manager.get_path("geometry", "umap_2d.png"))
    plot_latent_umap_3d(latents_n, latents_t, latents_c, manager.get_path("geometry", "umap_3d.png"))

    # E. Difference Vector Static (Disease-Sensitive Projection)
    plot_difference_vector_static(latents_n, latents_t, latents_c, manager.get_path("geometry", "diff_vec"))


def plot_latent_pca(latents_a, latents_b, latents_superdiff, output_path):
    """
    Performs PCA on the flattened latents to visualize distribution intersection.
    """

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_a, flat_b, flat_sd = flatten(latents_a), flatten(latents_b), flatten(latents_superdiff)

    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)
    pca = PCA(n_components=2).fit(all_data)

    res = pca.transform(all_data)
    n_a, n_b = len(flat_a), len(flat_b)
    pca_a, pca_b, pca_sd = res[:n_a], res[n_a:n_a + n_b], res[n_a + n_b:]

    setup_plot("Latent Space PCA Analysis", f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
               f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

    sns.scatterplot(x=pca_a[:, 0], y=pca_a[:, 1], color=COLORS['normal'], label='Normal', alpha=0.6, s=60)
    sns.scatterplot(x=pca_b[:, 0], y=pca_b[:, 1], color=COLORS['tb'], label='TB', alpha=0.6, s=60)
    sns.scatterplot(x=pca_sd[:, 0], y=pca_sd[:, 1], color=COLORS['comp'], label='Composition', marker='X', s=100,
                    edgecolor='white')

    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latent_tsne(latents_a, latents_b, latents_superdiff, output_path):
    """
    Performs t-SNE on the flattened latents to visualize distribution clusters.
    t-SNE is particularly good at revealing local structure and clusters.
    """
    print("Computing t-SNE 2D...")

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_a, flat_b, flat_sd = flatten(latents_a), flatten(latents_b), flatten(latents_superdiff)

    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)

    # t-SNE with perplexity tuned for the data size
    n_samples = len(all_data)
    perplexity = min(30, n_samples // 4)  # Adaptive perplexity

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(all_data)

    n_a, n_b = len(flat_a), len(flat_b)
    tsne_a, tsne_b, tsne_sd = embedding[:n_a], embedding[n_a:n_a + n_b], embedding[n_a + n_b:]

    setup_plot("Latent Space t-SNE Analysis", "t-SNE 1", "t-SNE 2")

    sns.scatterplot(x=tsne_a[:, 0], y=tsne_a[:, 1], color=COLORS['normal'], label='Normal', alpha=0.6, s=60)
    sns.scatterplot(x=tsne_b[:, 0], y=tsne_b[:, 1], color=COLORS['tb'], label='TB', alpha=0.6, s=60)
    sns.scatterplot(x=tsne_sd[:, 0], y=tsne_sd[:, 1], color=COLORS['comp'], label='Composition', marker='X', s=100,
                    edgecolor='white')

    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_density_landscapes(latents_a, latents_b, latents_sd, output_path):
    """
    Visualizes the 'Energy Landscape' using Kernel Density Estimation (KDE) contours.
    """
    print(f"Generating Density Landscape (Level Sets) -> {output_path}")

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_a, flat_b, flat_sd = flatten(latents_a), flatten(latents_b), flatten(latents_sd)

    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)
    pca = PCA(n_components=2).fit(all_data)
    res = pca.transform(all_data)
    n_a, n_b = len(flat_a), len(flat_b)
    p_a, p_b, p_sd = res[:n_a], res[n_a:n_a + n_b], res[n_a + n_b:]

    plt.figure(figsize=(12, 10))
    sns.kdeplot(x=p_a[:, 0], y=p_a[:, 1], levels=8, color=COLORS['normal'], linewidths=1.5, alpha=0.5,
                label='Normal Density')
    sns.kdeplot(x=p_b[:, 0], y=p_b[:, 1], levels=8, color=COLORS['tb'], linewidths=1.5, alpha=0.5, label='TB Density')
    sns.kdeplot(x=p_sd[:, 0], y=p_sd[:, 1], levels=10, color=COLORS['comp'], fill=True, alpha=0.3,
                label='Composition Density')
    plt.scatter(p_sd[:, 0], p_sd[:, 1], c=COLORS['comp'], s=10, alpha=0.4, marker='.', label='Comp Samples')

    plt.title("Level Sets: Where does the Composition settle?")
    custom_lines = [Line2D([0], [0], color=COLORS['normal'], lw=2),
                    Line2D([0], [0], color=COLORS['tb'], lw=2),
                    Line2D([0], [0], color=COLORS['comp'], lw=4, alpha=0.5)]
    plt.legend(custom_lines, ['Normal Density', 'TB Density', 'Composition Mass'])
    plt.grid(True, alpha=0.2)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latent_umap(latents_a, latents_b, latents_superdiff, output_path):
    """2D UMAP Projection."""
    print("Computing UMAP 2D...")

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_a, flat_b, flat_sd = flatten(latents_a), flatten(latents_b), flatten(latents_superdiff)
    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(all_data)
    n_a, n_b = len(flat_a), len(flat_b)
    u_a, u_b, u_sd = embedding[:n_a], embedding[n_a:n_a + n_b], embedding[n_a + n_b:]

    setup_plot("Latent Manifold (UMAP)", "U1", "U2")
    sns.scatterplot(x=u_a[:, 0], y=u_a[:, 1], color=COLORS['normal'], label='Normal', alpha=0.3)
    sns.scatterplot(x=u_b[:, 0], y=u_b[:, 1], color=COLORS['tb'], label='TB', alpha=0.3)
    sns.scatterplot(x=u_sd[:, 0], y=u_sd[:, 1], color=COLORS['comp'], label='Composition', marker='X', s=60,
                    edgecolor='white')
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latent_umap_3d(latents_a, latents_b, latents_superdiff, output_path):
    """3D UMAP Projection."""
    print("Computing UMAP 3D...")

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_a, flat_b, flat_sd = flatten(latents_a), flatten(latents_b), flatten(latents_superdiff)
    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)

    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(all_data)
    n_a, n_b = len(flat_a), len(flat_b)
    u_a, u_b, u_sd = embedding[:n_a], embedding[n_a:n_a + n_b], embedding[n_a + n_b:]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u_a[:, 0], u_a[:, 1], u_a[:, 2], c=COLORS['normal'], label='Normal', alpha=0.3, s=20)
    ax.scatter(u_b[:, 0], u_b[:, 1], u_b[:, 2], c=COLORS['tb'], label='TB', alpha=0.3, s=20)
    ax.scatter(u_sd[:, 0], u_sd[:, 1], u_sd[:, 2], c=COLORS['comp'], label='Composition', marker='X', s=50,
               depthshade=False)

    ax.set_title("3D Latent Manifold (UMAP)")
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==============================================================================
# 4. DIFFERENCE VECTOR ANALYSIS (Disease-Sensitive Projections)
#    Focus: Δz = E[z_TB] - E[z_Normal], project onto disease axis + orthogonal
# ==============================================================================

def plot_difference_vector_dynamics(traj_n, traj_t, traj_comp, output_path_base):
    """
    Projects diffusion trajectories onto the disease direction
    Δz = E[z_TB] - E[z_Normal] and its orthogonal complement.

    Outputs three plots:
      - Disease axis projection over diffusion time
      - Orthogonal drift over diffusion time
      - Disease signal ratio (|disease| / total displacement)
    """
    print(f">> Generating Difference Vector Dynamics -> {output_path_base}")

    # 1. Flatten trajectories: (steps, batch, H, W, C) -> (steps, batch, features)
    def flatten_traj(t):
        arr = np.array(t)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    flat_n = flatten_traj(traj_n)
    flat_t = flatten_traj(traj_t)
    flat_c = flatten_traj(traj_comp)

    # 2. Compute disease direction from final-step centroids (cleanest estimates)
    mu_n_final = np.mean(flat_n[-1], axis=0)
    mu_t_final = np.mean(flat_t[-1], axis=0)
    delta_z = mu_t_final - mu_n_final
    delta_z_norm = np.linalg.norm(delta_z)

    if delta_z_norm < 1e-10:
        print("   WARNING: Disease direction has near-zero norm. Skipping.")
        return

    delta_z_hat = delta_z / delta_z_norm

    # 3. Per-step centroids
    cent_n = np.mean(flat_n, axis=1)  # (S, F)
    cent_t = np.mean(flat_t, axis=1)
    cent_c = np.mean(flat_c, axis=1)

    # 4. Project centroids (centered on Normal path at each step)
    def project_onto_disease(centroids):
        centered = centroids - cent_n
        disease_proj = centered @ delta_z_hat                      # (S,)
        ortho = centered - np.outer(disease_proj, delta_z_hat)     # (S, F)
        ortho_norm = np.linalg.norm(ortho, axis=1)                 # (S,)
        return disease_proj, ortho_norm

    dis_n, ort_n = project_onto_disease(cent_n)
    dis_t, ort_t = project_onto_disease(cent_t)
    dis_c, ort_c = project_onto_disease(cent_c)

    n_steps = len(dis_c)
    time_axis = np.linspace(1.0, 0.0, n_steps)

    # --- Plot A: Disease Axis Projection ---
    setup_plot(
        r"Disease Axis Projection: $\langle x_t - \mu_{Normal}, \hat{\Delta z} \rangle$",
        "Diffusion Time (t)", "Projection onto Disease Direction"
    )
    plt.plot(time_axis, dis_n, c=COLORS['normal'], ls=STYLES['normal'], lw=2, label='Normal')
    plt.plot(time_axis, dis_t, c=COLORS['tb'], ls=STYLES['tb'], lw=2, label='TB')
    plt.plot(time_axis, dis_c, c=COLORS['comp'], lw=3, label='Composition')
    plt.axhline(0, color='gray', ls=':', alpha=0.4)
    plt.axhline(dis_t[-1], color=COLORS['tb'], ls=':', alpha=0.3, label=f'TB final ({dis_t[-1]:.1f})')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(f"{output_path_base}_disease_axis.png", dpi=300)
    plt.close()

    # --- Plot B: Orthogonal Complement ---
    setup_plot(
        "Orthogonal Complement: Non-Disease Drift from Normal",
        "Diffusion Time (t)", r"$\| x_t^{\perp} \|$ (Drift orthogonal to disease)"
    )
    plt.plot(time_axis, ort_n, c=COLORS['normal'], ls=STYLES['normal'], lw=2, label='Normal')
    plt.plot(time_axis, ort_t, c=COLORS['tb'], ls=STYLES['tb'], lw=2, label='TB')
    plt.plot(time_axis, ort_c, c=COLORS['comp'], lw=3, label='Composition')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(f"{output_path_base}_ortho_drift.png", dpi=300)
    plt.close()

    # --- Plot C: Disease Signal Ratio ---
    total_c = np.sqrt(dis_c**2 + ort_c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(total_c > 1e-8, np.abs(dis_c) / total_c, 0.0)

    setup_plot(
        "Disease Signal Ratio: |disease| / total displacement",
        "Diffusion Time (t)", "Disease Fraction (1.0 = pure disease direction)"
    )
    plt.plot(time_axis, ratio, c=COLORS['comp'], lw=3, label='Composition')
    plt.axhline(1.0, color='gray', ls=':', alpha=0.3, label='Pure disease')
    plt.axhline(0.0, color='gray', ls=':', alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(f"{output_path_base}_disease_ratio.png", dpi=300)
    plt.close()

    print(f"   Disease direction norm: {delta_z_norm:.4f}")
    print(f"   Composition final disease proj: {dis_c[-1]:.4f} (TB ref: {dis_t[-1]:.4f})")
    print(f"   Composition final ortho drift: {ort_c[-1]:.4f} (TB ref: {ort_t[-1]:.4f})")
    print(f"   Final disease ratio: {ratio[-1]:.4f}")


def plot_difference_vector_static(latents_n, latents_t, latents_c, output_path_base):
    """
    Static difference vector analysis on final latent samples.
    Projects all samples onto the disease direction and orthogonal complement.

    Outputs:
      - Histogram of per-sample disease axis projections
      - 2D scatter: disease projection vs orthogonal norm
    """
    print(f">> Generating Static Difference Vector Analysis -> {output_path_base}")

    def flatten(l): return np.array(l).reshape(l.shape[0], -1)

    flat_n = flatten(latents_n)
    flat_t = flatten(latents_t)
    flat_c = flatten(latents_c)

    # Disease direction from population means
    mu_n = np.mean(flat_n, axis=0)
    mu_t = np.mean(flat_t, axis=0)
    delta_z = mu_t - mu_n
    delta_z_norm = np.linalg.norm(delta_z)

    if delta_z_norm < 1e-10:
        print("   WARNING: Disease direction has near-zero norm. Skipping.")
        return

    delta_z_hat = delta_z / delta_z_norm

    # Project each sample (centered on Normal mean)
    def project_samples(flat):
        centered = flat - mu_n
        disease = centered @ delta_z_hat                      # (N,)
        ortho = centered - np.outer(disease, delta_z_hat)     # (N, F)
        ortho_norm = np.linalg.norm(ortho, axis=1)            # (N,)
        return disease, ortho_norm

    d_n, o_n = project_samples(flat_n)
    d_t, o_t = project_samples(flat_t)
    d_c, o_c = project_samples(flat_c)

    # --- Plot A: Disease axis histogram ---
    setup_plot("Disease Axis Projection (Final Samples)",
               "Projection onto Disease Direction", "Density")
    plt.hist(d_n, bins=20, alpha=0.5, color=COLORS['normal'], label='Normal', density=True)
    plt.hist(d_t, bins=20, alpha=0.5, color=COLORS['tb'], label='TB', density=True)
    plt.hist(d_c, bins=20, alpha=0.5, color=COLORS['comp'], label='Composition', density=True)
    plt.axvline(0, color='gray', ls=':', alpha=0.5)
    plt.legend()
    plt.savefig(f"{output_path_base}_histogram.png", dpi=300)
    plt.close()

    # --- Plot B: 2D scatter (disease vs orthogonal) ---
    setup_plot("Disease vs Non-Disease Components",
               "Disease Direction Projection", "Orthogonal Norm")
    plt.scatter(d_n, o_n, c=COLORS['normal'], alpha=0.5, s=40, label='Normal')
    plt.scatter(d_t, o_t, c=COLORS['tb'], alpha=0.5, s=40, label='TB')
    plt.scatter(d_c, o_c, c=COLORS['comp'], alpha=0.7, s=60,
                marker='X', edgecolor='white', label='Composition')
    plt.axvline(0, color='gray', ls=':', alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_path_base}_scatter.png", dpi=300)
    plt.close()

    # Summary
    print(f"   Disease direction norm: {delta_z_norm:.4f}")
    print(f"   Normal  -> disease: {np.mean(d_n):.3f} +/- {np.std(d_n):.3f}")
    print(f"   TB      -> disease: {np.mean(d_t):.3f} +/- {np.std(d_t):.3f}")
    print(f"   Comp    -> disease: {np.mean(d_c):.3f} +/- {np.std(d_c):.3f}")


# ==============================================================================
# 5. LOG DIAGNOSTICS
# ==============================================================================

def plot_log_trajectories(log_q_a_hist, log_q_b_hist, steps=None, output_path="log_trajectories.png"):
    log_q_a, log_q_b = np.array(log_q_a_hist).squeeze(), np.array(log_q_b_hist).squeeze()
    if log_q_a.ndim > 1: log_q_a, log_q_b = np.mean(log_q_a, axis=1), np.mean(log_q_b, axis=1)

    t = np.arange(len(log_q_a))
    xlabel, x_vals = ("Diffusion Time (t)", np.linspace(1.0, 0.0, len(log_q_a))) if steps else ("Inference Steps", t)

    setup_plot("Log-Density Trajectories", xlabel, "Cumulative Log Likelihood (Relative)")
    plt.plot(x_vals, log_q_a, label="Log q(Normal)", linewidth=2, color=COLORS['normal'])
    plt.plot(x_vals, log_q_b, label="Log q(TB)", linewidth=2, color=COLORS['tb'], linestyle="--")
    if steps: plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_kappa_trajectory(kappa_hist, steps=None, output_path="kappa_trajectory.png"):
    kappa = np.array(kappa_hist)
    kappa_mean = np.mean(kappa, axis=tuple(range(1, kappa.ndim))) if kappa.ndim > 1 else kappa
    kappa_std = np.std(kappa, axis=tuple(range(1, kappa.ndim))) if kappa.ndim > 1 else None

    xlabel, x_vals = ("Diffusion Time (t)", np.linspace(1.0, 0.0, len(kappa_mean))) if steps else ("Inference Steps",
                                                                                                   np.arange(
                                                                                                       len(kappa_mean)))

    setup_plot(r"Mixing Trajectory ($\kappa$)", xlabel, r"$\kappa$ Value")
    plt.plot(x_vals, kappa_mean, color='purple', linewidth=2, label=r"Average $\kappa$")
    if kappa_std is not None:
        plt.fill_between(x_vals, kappa_mean - kappa_std, kappa_mean + kappa_std, color='purple', alpha=0.2)

    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.ylim(-0.5, 2.5)
    if steps: plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_logs_and_kappa(logs_a, logs_b, kappas, steps, manager):
    """Wrapper to route logs/kappa to the manager."""
    if logs_a is not None:
        plot_log_trajectories(logs_a, logs_b, steps=steps,
                              output_path=manager.get_path("dynamics", "log_likelihoods.png"))
    if kappas is not None:
        plot_kappa_trajectory(kappas, steps=steps, output_path=manager.get_path("dynamics", "kappa_mixing.png"))