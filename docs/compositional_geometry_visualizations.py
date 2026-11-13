"""
Compositional Geometry for Multi-Disease Synthesis - Interactive Visualizations

This module creates interactive Plotly visualizations demonstrating the different
latent manifold structures that can emerge from SepVAE training and their impact
on disease composition for chest X-ray synthesis.

Structures covered:
    A: Entangled Overlap (FAILS)
    B: Single-Axis Collapse (FAILS)
    C: Isolated Clusters with Gaps (FAILS for LDM)
    D: Orthogonal Factorized (IDEAL)
    E: Curved Compositional Manifold (REQUIRES RIEMANNIAN)

Usage:
    python compositional_geometry_visualizations.py

Requirements:
    pip install plotly pandas numpy scipy kaleido
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Color Palette - Medical/Scientific Theme
# =============================================================================
COLORS = {
    'normal': '#2E86AB',      # Blue - Normal/Healthy
    'effusion': '#A23B72',    # Magenta - Pleural Effusion
    'cardiomegaly': '#F18F01', # Orange - Cardiomegaly
    'composition': '#C73E1D', # Red - Composition point
    'void': '#E8E8E8',        # Light gray - Void regions
    'valid': '#28A745',       # Green - Valid regions
    'invalid': '#DC3545',     # Red - Invalid regions
    'geodesic': '#6F42C1',    # Purple - Geodesic paths
    'euclidean': '#FD7E14',   # Orange - Euclidean paths
    'background': '#FAFAFA',  # Off-white background
}

# =============================================================================
# Helper Functions
# =============================================================================

def generate_cluster(center, n_points=100, std=0.3, seed=None):
    """Generate a 2D Gaussian cluster."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_points, 2) * std + np.array(center)


def generate_3d_cluster(center, n_points=100, std=0.3, seed=None):
    """Generate a 3D Gaussian cluster."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_points, 3) * std + np.array(center)


def create_density_grid(clusters, labels, grid_size=100, sigma=0.15):
    """Create a density heatmap from cluster points."""
    x_all = np.concatenate([c[:, 0] for c in clusters])
    y_all = np.concatenate([c[:, 1] for c in clusters])

    x_range = [x_all.min() - 1, x_all.max() + 1]
    y_range = [y_all.min() - 1, y_all.max() + 1]

    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = np.zeros_like(X)
    for cluster in clusters:
        for point in cluster:
            Z += np.exp(-((X - point[0])**2 + (Y - point[1])**2) / (2 * sigma**2))

    return X, Y, Z, x_range, y_range


# =============================================================================
# Structure A: Entangled Overlap (FAILS)
# =============================================================================

def create_structure_a_visualization():
    """
    Structure A: Entangled Overlap
    All disease classes overlap completely - composition produces garbage.
    """
    np.random.seed(42)

    # Generate overlapping clusters
    normal = generate_cluster([0, 0], n_points=150, std=0.8, seed=42)
    effusion = generate_cluster([0.1, 0.1], n_points=150, std=0.8, seed=43)
    cardiomegaly = generate_cluster([-0.1, -0.1], n_points=150, std=0.8, seed=44)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Structure A: Entangled Overlap</b><br><sup>All classes overlapping - NO distinct disease directions</sup>',
            '<b>Composition Attempt</b><br><sup>Adding noise to noise = garbage</sup>'
        ),
        horizontal_spacing=0.12
    )

    # Left plot: Entangled clusters
    fig.add_trace(
        go.Scatter(
            x=normal[:, 0], y=normal[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['normal'], opacity=0.6,
                       line=dict(width=1, color='white')),
            name='Normal (●)',
            legendgroup='normal'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=effusion[:, 0], y=effusion[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['effusion'], opacity=0.6,
                       symbol='triangle-up', line=dict(width=1, color='white')),
            name='Effusion (▲)',
            legendgroup='effusion'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=cardiomegaly[:, 0], y=cardiomegaly[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['cardiomegaly'], opacity=0.6,
                       symbol='square', line=dict(width=1, color='white')),
            name='Cardiomegaly (■)',
            legendgroup='cardiomegaly'
        ),
        row=1, col=1
    )

    # Add axes
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Right plot: Failed composition
    # Show that Δ vectors are random
    normal_mean = normal.mean(axis=0)
    effusion_mean = effusion.mean(axis=0)
    cardiomegaly_mean = cardiomegaly.mean(axis=0)

    delta_eff = effusion_mean - normal_mean
    delta_card = cardiomegaly_mean - normal_mean

    # Multiple "composition" attempts showing randomness
    for i in range(5):
        n_sample = normal[np.random.randint(len(normal))]
        e_sample = effusion[np.random.randint(len(effusion))]
        c_sample = cardiomegaly[np.random.randint(len(cardiomegaly))]

        delta_e = e_sample - n_sample
        delta_c = c_sample - n_sample
        composed = n_sample + delta_e + delta_c

        fig.add_trace(
            go.Scatter(
                x=[composed[0]], y=[composed[1]],
                mode='markers',
                marker=dict(size=15, color=COLORS['invalid'], opacity=0.7,
                           symbol='x', line=dict(width=2, color='darkred')),
                name='Failed Composition' if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate='Composition attempt %d<br>Result: GARBAGE<extra></extra>' % (i+1)
            ),
            row=1, col=2
        )

    # Show delta vectors (essentially random)
    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0], normal_mean[0] + delta_eff[0]],
            y=[normal_mean[1], normal_mean[1] + delta_eff[1]],
            mode='lines+markers',
            line=dict(color=COLORS['effusion'], width=3, dash='dot'),
            marker=dict(size=[10, 0]),
            name='Δ_eff (≈ noise)',
            hovertemplate='Δ_eff magnitude: %.3f<br>Direction: RANDOM<extra></extra>' % np.linalg.norm(delta_eff)
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0], normal_mean[0] + delta_card[0]],
            y=[normal_mean[1], normal_mean[1] + delta_card[1]],
            mode='lines+markers',
            line=dict(color=COLORS['cardiomegaly'], width=3, dash='dot'),
            marker=dict(size=[10, 0]),
            name='Δ_card (≈ noise)',
            hovertemplate='Δ_card magnitude: %.3f<br>Direction: RANDOM<extra></extra>' % np.linalg.norm(delta_card)
        ),
        row=1, col=2
    )

    # Add origin marker
    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0]], y=[normal_mean[1]],
            mode='markers',
            marker=dict(size=15, color=COLORS['normal'], symbol='circle',
                       line=dict(width=2, color='black')),
            name='Origin (Normal mean)',
        ),
        row=1, col=2
    )

    # Add annotation for failure
    fig.add_annotation(
        x=0.5, y=-1.5,
        text="<b>COMPOSITION FAILS</b><br>No consistent disease directions exist",
        showarrow=False,
        font=dict(size=14, color=COLORS['invalid']),
        bgcolor='rgba(255,200,200,0.8)',
        bordercolor=COLORS['invalid'],
        borderwidth=2,
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>Structure A: Entangled Overlap - COMPOSITION FAILS</b>',
            font=dict(size=20),
            x=0.5
        ),
        height=550,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    for col in [1, 2]:
        fig.update_xaxes(title_text='PC1 (z_salient)', row=1, col=col, range=[-3, 3])
        fig.update_yaxes(title_text='PC2 (z_salient)', row=1, col=col, range=[-3, 3])

    return fig


# =============================================================================
# Structure B: Single-Axis Collapse (FAILS)
# =============================================================================

def create_structure_b_visualization():
    """
    Structure B: Single-Axis Collapse
    Both diseases encoded along same axis - composition = interpolation/cancellation.
    """
    np.random.seed(42)

    # Generate clusters along single axis
    normal = generate_cluster([0, 0], n_points=100, std=0.25, seed=42)
    normal[:, 1] *= 0.3  # Compress in y direction

    effusion = generate_cluster([2, 0], n_points=100, std=0.25, seed=43)
    effusion[:, 1] *= 0.3

    cardiomegaly = generate_cluster([4, 0], n_points=100, std=0.25, seed=44)
    cardiomegaly[:, 1] *= 0.3

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Structure B: Single-Axis Collapse</b><br><sup>Diseases lie on same axis - only 1 degree of freedom</sup>',
            '<b>Composition = Extrapolation</b><br><sup>z_eff + z_card overshoots or cancels</sup>'
        ),
        horizontal_spacing=0.12
    )

    # Left plot: Single-axis clusters
    fig.add_trace(
        go.Scatter(
            x=normal[:, 0], y=normal[:, 1],
            mode='markers',
            marker=dict(size=10, color=COLORS['normal'], opacity=0.7),
            name='Normal (●)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=effusion[:, 0], y=effusion[:, 1],
            mode='markers',
            marker=dict(size=10, color=COLORS['effusion'], opacity=0.7, symbol='triangle-up'),
            name='Effusion (▲)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=cardiomegaly[:, 0], y=cardiomegaly[:, 1],
            mode='markers',
            marker=dict(size=10, color=COLORS['cardiomegaly'], opacity=0.7, symbol='square'),
            name='Cardiomegaly (■)'
        ),
        row=1, col=1
    )

    # Add axis line showing 1D structure
    fig.add_trace(
        go.Scatter(
            x=[-1, 6], y=[0, 0],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='Single disease axis',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Right plot: Composition failure
    normal_mean = np.array([0, 0])
    effusion_mean = np.array([2, 0])
    cardiomegaly_mean = np.array([4, 0])

    delta_eff = effusion_mean - normal_mean  # [2, 0]
    delta_card = cardiomegaly_mean - normal_mean  # [4, 0]

    # Composition overshoots!
    composed = normal_mean + delta_eff + delta_card  # [6, 0]

    # Show the axis with regions
    x_line = np.linspace(-1, 7, 100)

    fig.add_trace(
        go.Scatter(
            x=x_line, y=np.zeros_like(x_line),
            mode='lines',
            line=dict(color='gray', width=3),
            name='Disease axis',
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # Mark regions
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=20, color=COLORS['normal'], symbol='circle'),
            text=['Normal'],
            textposition='bottom center',
            name='Normal'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[2], y=[0],
            mode='markers+text',
            marker=dict(size=20, color=COLORS['effusion'], symbol='triangle-up'),
            text=['Effusion'],
            textposition='bottom center',
            name='Effusion'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[4], y=[0],
            mode='markers+text',
            marker=dict(size=20, color=COLORS['cardiomegaly'], symbol='square'),
            text=['Cardiomegaly'],
            textposition='bottom center',
            name='Cardiomegaly'
        ),
        row=1, col=2
    )

    # Show composition point (extrapolation!)
    fig.add_trace(
        go.Scatter(
            x=[6], y=[0],
            mode='markers+text',
            marker=dict(size=25, color=COLORS['invalid'], symbol='x',
                       line=dict(width=3, color='darkred')),
            text=['EXTRAPOLATION!'],
            textposition='top center',
            name='Composition (FAILS)',
            hovertemplate='z_composed = z_● + Δ▲ + Δ■<br>= 0 + 2 + 4 = 6<br><b>OVERSHOOTS beyond training data!</b><extra></extra>'
        ),
        row=1, col=2
    )

    # Draw delta vectors
    fig.add_annotation(
        x=1, y=0.3, ax=0, ay=0.3,
        xref="x2", yref="y2", axref="x2", ayref="y2",
        text="Δ_eff = +2",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['effusion'],
        font=dict(color=COLORS['effusion'], size=12)
    )

    fig.add_annotation(
        x=2, y=0.5, ax=0, ay=0.5,
        xref="x2", yref="y2", axref="x2", ayref="y2",
        text="Δ_card = +4",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['cardiomegaly'],
        font=dict(color=COLORS['cardiomegaly'], size=12)
    )

    # Add failure box
    fig.add_shape(
        type="rect",
        x0=5, y0=-0.8, x1=7, y1=0.8,
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(color=COLORS['invalid'], width=2, dash='dash'),
        row=1, col=2
    )

    fig.add_annotation(
        x=6, y=-1.2,
        text="<b>OUT OF DISTRIBUTION</b><br>Decoder never trained here",
        showarrow=False,
        font=dict(size=11, color=COLORS['invalid']),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>Structure B: Single-Axis Collapse - COMPOSITION FAILS</b>',
            font=dict(size=20),
            x=0.5
        ),
        height=500,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template='plotly_white'
    )

    fig.update_xaxes(title_text='PC1 (z_salient)', row=1, col=1, range=[-2, 6])
    fig.update_yaxes(title_text='PC2 (z_salient)', row=1, col=1, range=[-2, 2])
    fig.update_xaxes(title_text='Disease Axis', row=1, col=2, range=[-1, 8])
    fig.update_yaxes(title_text='', row=1, col=2, range=[-2, 2])

    return fig


# =============================================================================
# Structure C: Isolated Clusters with Gaps (FAILS for LDM)
# =============================================================================

def create_structure_c_visualization():
    """
    Structure C: Isolated Clusters with Gaps
    Well-separated clusters but empty space between - LDM traverses void.
    """
    np.random.seed(42)

    # Generate well-separated clusters
    normal = generate_cluster([-2, -2], n_points=100, std=0.4, seed=42)
    effusion = generate_cluster([2, 2], n_points=100, std=0.4, seed=43)
    cardiomegaly = generate_cluster([2, -2], n_points=100, std=0.4, seed=44)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Structure C: Isolated Clusters with Gaps</b><br><sup>Good separation but large voids between clusters</sup>',
            '<b>Composition Lands in VOID</b><br><sup>Decoder has never seen this region</sup>'
        ),
        horizontal_spacing=0.12
    )

    # Create density heatmap for background
    X, Y, Z, x_range, y_range = create_density_grid(
        [normal, effusion, cardiomegaly],
        ['Normal', 'Effusion', 'Cardiomegaly'],
        grid_size=80, sigma=0.4
    )

    # Left plot: Clusters with density
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], 80),
            y=np.linspace(y_range[0], y_range[1], 80),
            z=Z,
            colorscale=[[0, 'white'], [0.3, '#E8F4F8'], [1, '#B8D4E3']],
            showscale=False,
            contours=dict(showlines=False),
            hoverinfo='skip',
            opacity=0.7
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=normal[:, 0], y=normal[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['normal'], opacity=0.8),
            name='Normal (●)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=effusion[:, 0], y=effusion[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['effusion'], opacity=0.8, symbol='triangle-up'),
            name='Effusion (▲)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=cardiomegaly[:, 0], y=cardiomegaly[:, 1],
            mode='markers',
            marker=dict(size=8, color=COLORS['cardiomegaly'], opacity=0.8, symbol='square'),
            name='Cardiomegaly (■)'
        ),
        row=1, col=1
    )

    # Right plot: Show void problem
    # Density with void highlighted
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], 80),
            y=np.linspace(y_range[0], y_range[1], 80),
            z=Z,
            colorscale=[[0, 'rgba(255,200,200,0.5)'], [0.2, 'white'], [1, '#B8D4E3']],
            showscale=False,
            contours=dict(showlines=True, showlabels=False),
            hoverinfo='skip',
            opacity=0.8
        ),
        row=1, col=2
    )

    # Mark cluster centers
    normal_mean = normal.mean(axis=0)
    effusion_mean = effusion.mean(axis=0)
    cardiomegaly_mean = cardiomegaly.mean(axis=0)

    # Compute composition point
    delta_eff = effusion_mean - normal_mean
    delta_card = cardiomegaly_mean - normal_mean
    composed = normal_mean + delta_eff + delta_card

    # Draw delta vectors
    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0], normal_mean[0] + delta_eff[0]],
            y=[normal_mean[1], normal_mean[1] + delta_eff[1]],
            mode='lines',
            line=dict(color=COLORS['effusion'], width=3, dash='dash'),
            name='Δ_eff',
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0] + delta_eff[0], composed[0]],
            y=[normal_mean[1] + delta_eff[1], composed[1]],
            mode='lines',
            line=dict(color=COLORS['cardiomegaly'], width=3, dash='dash'),
            name='Δ_card',
        ),
        row=1, col=2
    )

    # Mark cluster means
    for pos, color, name in [
        (normal_mean, COLORS['normal'], 'Normal'),
        (effusion_mean, COLORS['effusion'], 'Effusion'),
        (cardiomegaly_mean, COLORS['cardiomegaly'], 'Cardiomegaly')
    ]:
        fig.add_trace(
            go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers',
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                name=name,
                showlegend=False
            ),
            row=1, col=2
        )

    # Mark composition point IN THE VOID
    fig.add_trace(
        go.Scatter(
            x=[composed[0]], y=[composed[1]],
            mode='markers',
            marker=dict(size=25, color=COLORS['invalid'], symbol='x',
                       line=dict(width=3, color='darkred')),
            name='Composition (IN VOID)',
            hovertemplate='z_composed = z_● + Δ▲ + Δ■<br>Position: (%.1f, %.1f)<br><b>FALLS IN UNTRAINED VOID!</b><extra></extra>' % (composed[0], composed[1])
        ),
        row=1, col=2
    )

    # Add void annotation
    fig.add_shape(
        type="circle",
        x0=composed[0]-1, y0=composed[1]-1,
        x1=composed[0]+1, y1=composed[1]+1,
        line=dict(color=COLORS['invalid'], width=2, dash='dash'),
        fillcolor='rgba(255,0,0,0.1)',
        row=1, col=2
    )

    fig.add_annotation(
        x=composed[0], y=composed[1]+1.5,
        text="<b>VOID REGION</b><br>Zero training density<br>Decoder produces artifacts",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['invalid'],
        font=dict(size=11, color=COLORS['invalid']),
        bgcolor='rgba(255,255,255,0.9)',
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>Structure C: Isolated Clusters - COMPOSITION LANDS IN VOID</b>',
            font=dict(size=20),
            x=0.5
        ),
        height=550,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template='plotly_white'
    )

    for col in [1, 2]:
        fig.update_xaxes(title_text='PC1 (z_salient)', row=1, col=col)
        fig.update_yaxes(title_text='PC2 (z_salient)', row=1, col=col)

    return fig


# =============================================================================
# Structure D: Orthogonal Factorized (IDEAL)
# =============================================================================

def create_structure_d_visualization():
    """
    Structure D: Orthogonal Factorized - IDEAL for composition
    Diseases occupy orthogonal axes, dense manifold coverage.
    """
    np.random.seed(42)

    # Generate orthogonal structure
    # Normal at origin
    normal = generate_cluster([0, 0], n_points=120, std=0.35, seed=42)

    # Effusion along PC2 (vertical)
    effusion = generate_cluster([0, 3], n_points=120, std=0.35, seed=43)

    # Cardiomegaly along PC1 (horizontal)
    cardiomegaly = generate_cluster([3, 0], n_points=120, std=0.35, seed=44)

    # Composition zone (upper right quadrant) - this is where both diseases combine!
    composition_zone = generate_cluster([3, 3], n_points=80, std=0.4, seed=45)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Structure D: Orthogonal Factorized (IDEAL)</b><br><sup>Independent disease axes with dense coverage</sup>',
            '<b>Successful Composition</b><br><sup>z_eff + z_card lands in valid region!</sup>'
        ),
        horizontal_spacing=0.12
    )

    # Create density including composition zone
    all_points = np.vstack([normal, effusion, cardiomegaly, composition_zone])
    X, Y, Z, x_range, y_range = create_density_grid(
        [normal, effusion, cardiomegaly, composition_zone],
        ['Normal', 'Effusion', 'Cardiomegaly', 'Composition'],
        grid_size=80, sigma=0.5
    )

    # Left plot: Orthogonal structure
    fig.add_trace(
        go.Contour(
            x=np.linspace(-2, 5, 80),
            y=np.linspace(-2, 5, 80),
            z=Z,
            colorscale=[[0, 'white'], [0.2, '#E8F8E8'], [1, '#90EE90']],
            showscale=False,
            contours=dict(showlines=False),
            hoverinfo='skip',
            opacity=0.6
        ),
        row=1, col=1
    )

    # Add quadrant labels
    fig.add_annotation(x=0, y=-1.5, text="<b>Normal</b><br>(Origin)", showarrow=False,
                      font=dict(size=11, color=COLORS['normal']), row=1, col=1)
    fig.add_annotation(x=0, y=4.5, text="<b>Effusion Only</b>", showarrow=False,
                      font=dict(size=11, color=COLORS['effusion']), row=1, col=1)
    fig.add_annotation(x=4.5, y=0, text="<b>Cardiomegaly Only</b>", showarrow=False,
                      font=dict(size=11, color=COLORS['cardiomegaly']), row=1, col=1)
    fig.add_annotation(x=4.5, y=4.5, text="<b>BOTH DISEASES</b><br>★ Composition Zone", showarrow=False,
                      font=dict(size=12, color=COLORS['valid']), row=1, col=1,
                      bgcolor='rgba(200,255,200,0.8)')

    # Plot clusters
    for data, color, name, symbol in [
        (normal, COLORS['normal'], 'Normal (●)', 'circle'),
        (effusion, COLORS['effusion'], 'Effusion (▲)', 'triangle-up'),
        (cardiomegaly, COLORS['cardiomegaly'], 'Cardiomegaly (■)', 'square'),
        (composition_zone, COLORS['valid'], 'Composition Zone (★)', 'star')
    ]:
        fig.add_trace(
            go.Scatter(
                x=data[:, 0], y=data[:, 1],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.7, symbol=symbol),
                name=name
            ),
            row=1, col=1
        )

    # Add orthogonal axes
    fig.add_trace(
        go.Scatter(
            x=[-1, 5], y=[0, 0],
            mode='lines',
            line=dict(color=COLORS['cardiomegaly'], width=2, dash='dot'),
            name='Cardiomegaly axis (PC1)',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 0], y=[-1, 5],
            mode='lines',
            line=dict(color=COLORS['effusion'], width=2, dash='dot'),
            name='Effusion axis (PC2)',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Right plot: Composition success
    normal_mean = np.array([0, 0])
    effusion_mean = np.array([0, 3])
    cardiomegaly_mean = np.array([3, 0])

    delta_eff = effusion_mean - normal_mean
    delta_card = cardiomegaly_mean - normal_mean
    composed = normal_mean + delta_eff + delta_card

    # Background density
    fig.add_trace(
        go.Contour(
            x=np.linspace(-2, 5, 80),
            y=np.linspace(-2, 5, 80),
            z=Z,
            colorscale=[[0, 'white'], [0.2, '#E8F8E8'], [1, '#90EE90']],
            showscale=False,
            contours=dict(showlines=True, showlabels=False),
            hoverinfo='skip',
            opacity=0.6
        ),
        row=1, col=2
    )

    # Draw composition formula visually
    # Origin
    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0]], y=[normal_mean[1]],
            mode='markers+text',
            marker=dict(size=20, color=COLORS['normal']),
            text=['z_●'],
            textposition='bottom right',
            name='Origin',
            showlegend=False
        ),
        row=1, col=2
    )

    # Delta vectors with arrows
    # Δ_eff (vertical)
    fig.add_trace(
        go.Scatter(
            x=[0, 0], y=[0, 3],
            mode='lines',
            line=dict(color=COLORS['effusion'], width=4),
            name='Δ_eff (+PC2)',
        ),
        row=1, col=2
    )
    fig.add_annotation(
        x=0, y=1.5, ax=-50, ay=0,
        text="<b>+Δ_eff</b>",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor=COLORS['effusion'],
        font=dict(color=COLORS['effusion'], size=14),
        row=1, col=2
    )

    # Δ_card (horizontal) from the effusion endpoint
    fig.add_trace(
        go.Scatter(
            x=[0, 3], y=[3, 3],
            mode='lines',
            line=dict(color=COLORS['cardiomegaly'], width=4),
            name='Δ_card (+PC1)',
        ),
        row=1, col=2
    )
    fig.add_annotation(
        x=1.5, y=3, ax=0, ay=-40,
        text="<b>+Δ_card</b>",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor=COLORS['cardiomegaly'],
        font=dict(color=COLORS['cardiomegaly'], size=14),
        row=1, col=2
    )

    # Composition point (SUCCESS!)
    fig.add_trace(
        go.Scatter(
            x=[composed[0]], y=[composed[1]],
            mode='markers',
            marker=dict(size=30, color=COLORS['valid'], symbol='star',
                       line=dict(width=2, color='darkgreen')),
            name='Composition (SUCCESS!)',
            hovertemplate='z_composed = z_● + Δ_eff + Δ_card<br>= (0,0) + (0,3) + (3,0)<br>= (3, 3)<br><b>LANDS IN VALID REGION!</b><extra></extra>'
        ),
        row=1, col=2
    )

    # Success annotation
    fig.add_annotation(
        x=3, y=4.2,
        text="<b>COMPOSITION SUCCEEDS!</b><br>Both diseases expressed<br>Anatomy preserved",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['valid'],
        font=dict(size=12, color=COLORS['valid']),
        bgcolor='rgba(200,255,200,0.9)',
        bordercolor=COLORS['valid'],
        row=1, col=2
    )

    # Formula box
    fig.add_annotation(
        x=-0.5, y=4.8,
        text="<b>Composition Formula:</b><br>z_★ = z_● + Δ_eff + Δ_card",
        showarrow=False,
        font=dict(size=13),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1,
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>Structure D: Orthogonal Factorized - COMPOSITION SUCCEEDS!</b>',
            font=dict(size=20, color=COLORS['valid']),
            x=0.5
        ),
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        template='plotly_white'
    )

    for col in [1, 2]:
        fig.update_xaxes(title_text='PC1 (Cardiomegaly axis)', row=1, col=col, range=[-2, 5.5])
        fig.update_yaxes(title_text='PC2 (Effusion axis)', row=1, col=col, range=[-2, 5.5])

    return fig


# =============================================================================
# Structure E: Curved Manifold (Requires Riemannian)
# =============================================================================

def create_structure_e_visualization():
    """
    Structure E: Curved Compositional Manifold
    Shows both FAILURE (Euclidean) and SUCCESS (Geodesic) composition approaches.

    The key insight: On a curved manifold, the composition zone EXISTS but can only
    be reached by following the manifold's curvature (geodesic paths), not by
    straight-line (Euclidean) addition.
    """
    np.random.seed(42)

    # ==========================================================================
    # Create a horseshoe/U-shaped manifold where:
    # - Normal is at the bottom-left
    # - Effusion is at the top (apex of the curve)
    # - Cardiomegaly is at the bottom-right
    # - COMPOSITION ZONE is on the manifold, reachable via geodesic
    # ==========================================================================

    # Horseshoe manifold parameters
    R = 3  # Radius of the horseshoe

    # Generate points along the horseshoe curve
    # Normal: bottom-left arm (theta from -pi/2 to -pi/6)
    theta_normal = np.linspace(-np.pi/2, -np.pi/6, 60)
    normal_base = np.column_stack([R * np.cos(theta_normal), R * np.sin(theta_normal)])
    normal = normal_base + np.random.randn(60, 2) * 0.25

    # Effusion: top of horseshoe (theta from -pi/6 to pi/6)
    theta_effusion = np.linspace(-np.pi/6, np.pi/6, 60)
    effusion_base = np.column_stack([R * np.cos(theta_effusion), R * np.sin(theta_effusion)])
    effusion = effusion_base + np.random.randn(60, 2) * 0.25

    # Cardiomegaly: bottom-right arm (theta from pi/6 to pi/2)
    theta_card = np.linspace(np.pi/6, np.pi/2, 60)
    card_base = np.column_stack([R * np.cos(theta_card), R * np.sin(theta_card)])
    cardiomegaly = card_base + np.random.randn(60, 2) * 0.25

    # COMPOSITION ZONE: Top-right quadrant of the horseshoe
    # This is where BOTH effusion AND cardiomegaly features combine
    # It's at the junction between effusion and cardiomegaly regions
    theta_comp = np.linspace(0, np.pi/6, 40)
    # Slightly outside the main curve to represent "more disease"
    comp_base = np.column_stack([
        (R + 0.5) * np.cos(theta_comp),
        (R + 0.5) * np.sin(theta_comp)
    ])
    composition_zone = comp_base + np.random.randn(40, 2) * 0.2

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Structure E: Curved Manifold with Composition Zone</b><br><sup>Valid composition exists ON the manifold</sup>',
            '<b>Euclidean FAILS vs Geodesic SUCCEEDS</b><br><sup>Must follow manifold curvature to compose</sup>'
        ),
        horizontal_spacing=0.1
    )

    # Create density map including composition zone
    X, Y, Z, x_range, y_range = create_density_grid(
        [normal, effusion, cardiomegaly, composition_zone],
        ['Normal', 'Effusion', 'Cardiomegaly', 'Composition'],
        grid_size=120, sigma=0.4
    )

    # ==========================================================================
    # LEFT PLOT: Show the curved manifold structure with labeled regions
    # ==========================================================================

    # Density contour background
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], 120),
            y=np.linspace(y_range[0], y_range[1], 120),
            z=Z,
            colorscale=[
                [0, 'rgba(255,220,220,0.3)'],  # Void = light red
                [0.1, 'white'],
                [0.25, '#E8F4F8'],
                [0.5, '#B8D4E3'],
                [1, '#87CEEB']
            ],
            showscale=False,
            contours=dict(showlines=True, showlabels=False),
            hoverinfo='skip',
            opacity=0.85,
            name='Manifold density'
        ),
        row=1, col=1
    )

    # Draw the "spine" of the manifold to emphasize the curve
    theta_spine = np.linspace(-np.pi/2, np.pi/2, 100)
    spine_x = R * np.cos(theta_spine)
    spine_y = R * np.sin(theta_spine)

    fig.add_trace(
        go.Scatter(
            x=spine_x, y=spine_y,
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            name='Manifold spine',
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )

    # Plot clusters with distinct markers
    for data, color, name, symbol in [
        (normal, COLORS['normal'], 'Normal (●)', 'circle'),
        (effusion, COLORS['effusion'], 'Effusion (▲)', 'triangle-up'),
        (cardiomegaly, COLORS['cardiomegaly'], 'Cardiomegaly (■)', 'square'),
        (composition_zone, COLORS['valid'], 'Composition Zone (★)', 'star')
    ]:
        fig.add_trace(
            go.Scatter(
                x=data[:, 0], y=data[:, 1],
                mode='markers',
                marker=dict(size=7, color=color, opacity=0.75, symbol=symbol,
                           line=dict(width=0.5, color='white')),
                name=name
            ),
            row=1, col=1
        )

    # Compute means for composition calculations
    normal_mean = normal.mean(axis=0)
    effusion_mean = effusion.mean(axis=0)
    cardiomegaly_mean = cardiomegaly.mean(axis=0)
    composition_mean = composition_zone.mean(axis=0)

    # Mark cluster centers
    for pos, color, label in [
        (normal_mean, COLORS['normal'], 'N'),
        (effusion_mean, COLORS['effusion'], 'E'),
        (cardiomegaly_mean, COLORS['cardiomegaly'], 'C'),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=18, color=color, line=dict(width=2, color='white')),
                text=[label],
                textfont=dict(color='white', size=10),
                textposition='middle center',
                showlegend=False,
                hovertemplate=f'{label} center<extra></extra>'
            ),
            row=1, col=1
        )

    # Euclidean composition point (FAILS - lands in void)
    euclidean_composed = normal_mean + (effusion_mean - normal_mean) + (cardiomegaly_mean - normal_mean)

    fig.add_trace(
        go.Scatter(
            x=[euclidean_composed[0]], y=[euclidean_composed[1]],
            mode='markers',
            marker=dict(size=22, color=COLORS['invalid'], symbol='x',
                       line=dict(width=3, color='darkred')),
            name='Euclidean composition (VOID)',
            hovertemplate='<b>Euclidean Composition</b><br>z_N + Δ_E + Δ_C<br>= (%.2f, %.2f)<br><b>LANDS IN VOID!</b><extra></extra>' % (euclidean_composed[0], euclidean_composed[1])
        ),
        row=1, col=1
    )

    # Geodesic composition point (SUCCESS - on the manifold)
    fig.add_trace(
        go.Scatter(
            x=[composition_mean[0]], y=[composition_mean[1]],
            mode='markers',
            marker=dict(size=22, color=COLORS['valid'], symbol='star',
                       line=dict(width=2, color='darkgreen')),
            name='Geodesic composition (ON MANIFOLD)',
            hovertemplate='<b>Geodesic Composition</b><br>Following manifold curve<br>= (%.2f, %.2f)<br><b>VALID REGION!</b><extra></extra>' % (composition_mean[0], composition_mean[1])
        ),
        row=1, col=1
    )

    # Add region labels
    fig.add_annotation(x=normal_mean[0]-0.8, y=normal_mean[1]-0.5,
                      text="<b>Normal</b><br>(Origin)", showarrow=False,
                      font=dict(size=10, color=COLORS['normal']), row=1, col=1)
    fig.add_annotation(x=effusion_mean[0], y=effusion_mean[1]+0.8,
                      text="<b>Effusion</b>", showarrow=False,
                      font=dict(size=10, color=COLORS['effusion']), row=1, col=1)
    fig.add_annotation(x=cardiomegaly_mean[0]+0.8, y=cardiomegaly_mean[1]-0.5,
                      text="<b>Cardiomegaly</b>", showarrow=False,
                      font=dict(size=10, color=COLORS['cardiomegaly']), row=1, col=1)

    # ==========================================================================
    # RIGHT PLOT: Detailed comparison of Euclidean vs Geodesic paths
    # ==========================================================================

    # Same density background
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], 120),
            y=np.linspace(y_range[0], y_range[1], 120),
            z=Z,
            colorscale=[
                [0, 'rgba(255,220,220,0.3)'],
                [0.1, 'white'],
                [0.25, '#E8F4F8'],
                [0.5, '#B8D4E3'],
                [1, '#87CEEB']
            ],
            showscale=False,
            contours=dict(showlines=True, showlabels=False),
            hoverinfo='skip',
            opacity=0.85
        ),
        row=1, col=2
    )

    # Draw manifold spine
    fig.add_trace(
        go.Scatter(
            x=spine_x, y=spine_y,
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # --- EUCLIDEAN PATHS (FAIL) ---
    # Path 1: Normal → Euclidean composition point
    fig.add_trace(
        go.Scatter(
            x=[normal_mean[0], euclidean_composed[0]],
            y=[normal_mean[1], euclidean_composed[1]],
            mode='lines',
            line=dict(color=COLORS['invalid'], width=4, dash='dash'),
            name='Euclidean: N + Δ_E + Δ_C',
            hovertemplate='Straight-line composition<br><b>CROSSES VOID</b><extra></extra>'
        ),
        row=1, col=2
    )

    # Show the delta vectors
    # Δ_E vector
    fig.add_annotation(
        x=normal_mean[0] + (effusion_mean[0] - normal_mean[0])*0.5,
        y=normal_mean[1] + (effusion_mean[1] - normal_mean[1])*0.5,
        ax=normal_mean[0], ay=normal_mean[1],
        xref="x2", yref="y2", axref="x2", ayref="y2",
        text="", showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=COLORS['effusion'],
    )

    # Δ_C vector (from N, parallel shifted for visibility)
    fig.add_annotation(
        x=normal_mean[0] + (cardiomegaly_mean[0] - normal_mean[0])*0.5 + 0.3,
        y=normal_mean[1] + (cardiomegaly_mean[1] - normal_mean[1])*0.5,
        ax=normal_mean[0] + 0.3, ay=normal_mean[1],
        xref="x2", yref="y2", axref="x2", ayref="y2",
        text="", showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=COLORS['cardiomegaly'],
    )

    # Failed composition point
    fig.add_trace(
        go.Scatter(
            x=[euclidean_composed[0]], y=[euclidean_composed[1]],
            mode='markers+text',
            marker=dict(size=25, color=COLORS['invalid'], symbol='x',
                       line=dict(width=3, color='darkred')),
            text=['✗'],
            textposition='top center',
            textfont=dict(size=16, color=COLORS['invalid']),
            showlegend=False,
            hovertemplate='<b>EUCLIDEAN FAILS</b><br>Lands in void<extra></extra>'
        ),
        row=1, col=2
    )

    # --- GEODESIC PATHS (SUCCESS) ---
    # Path following the manifold curve from Normal through Effusion to Composition Zone

    # Geodesic from Normal to Effusion (along the horseshoe)
    theta_path1 = np.linspace(-np.pi/2 + 0.2, -np.pi/6 + 0.1, 15)
    geo_x1 = R * np.cos(theta_path1)
    geo_y1 = R * np.sin(theta_path1)

    # Geodesic from Effusion to Composition Zone
    theta_path2 = np.linspace(-np.pi/6 + 0.1, np.pi/12, 15)
    geo_x2 = (R + 0.25) * np.cos(theta_path2)
    geo_y2 = (R + 0.25) * np.sin(theta_path2)

    # Combined geodesic path
    geo_full_x = np.concatenate([geo_x1, geo_x2])
    geo_full_y = np.concatenate([geo_y1, geo_y2])

    fig.add_trace(
        go.Scatter(
            x=geo_full_x, y=geo_full_y,
            mode='lines+markers',
            line=dict(color=COLORS['geodesic'], width=5),
            marker=dict(size=5, color=COLORS['geodesic']),
            name='Geodesic: follows manifold',
            hovertemplate='Geodesic path<br><b>STAYS ON MANIFOLD</b><extra></extra>'
        ),
        row=1, col=2
    )

    # Second geodesic arm: from Cardiomegaly toward composition
    theta_path3 = np.linspace(np.pi/2 - 0.2, np.pi/12, 15)
    geo_x3 = (R + 0.1) * np.cos(theta_path3)
    geo_y3 = (R + 0.1) * np.sin(theta_path3)

    fig.add_trace(
        go.Scatter(
            x=geo_x3, y=geo_y3,
            mode='lines+markers',
            line=dict(color=COLORS['geodesic'], width=5, dash='dot'),
            marker=dict(size=5, color=COLORS['geodesic']),
            showlegend=False,
            hovertemplate='Geodesic path<br><b>STAYS ON MANIFOLD</b><extra></extra>'
        ),
        row=1, col=2
    )

    # Successful composition point
    fig.add_trace(
        go.Scatter(
            x=[composition_mean[0]], y=[composition_mean[1]],
            mode='markers+text',
            marker=dict(size=28, color=COLORS['valid'], symbol='star',
                       line=dict(width=2, color='darkgreen')),
            text=['✓'],
            textposition='middle center',
            textfont=dict(size=14, color='white'),
            showlegend=False,
            hovertemplate='<b>GEODESIC SUCCEEDS</b><br>Composition zone reached!<extra></extra>'
        ),
        row=1, col=2
    )

    # Mark disease centers on right plot
    for pos, color, label in [
        (normal_mean, COLORS['normal'], 'N'),
        (effusion_mean, COLORS['effusion'], 'E'),
        (cardiomegaly_mean, COLORS['cardiomegaly'], 'C'),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=16, color=color, line=dict(width=2, color='white')),
                text=[label],
                textfont=dict(color='white', size=9),
                textposition='middle center',
                showlegend=False
            ),
            row=1, col=2
        )

    # Annotations explaining the difference
    fig.add_annotation(
        x=euclidean_composed[0], y=euclidean_composed[1] - 1.0,
        text="<b>VOID</b><br>Euclidean addition<br>lands OFF manifold",
        showarrow=True, arrowhead=2,
        arrowcolor=COLORS['invalid'],
        font=dict(size=10, color=COLORS['invalid']),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=COLORS['invalid'],
        row=1, col=2
    )

    fig.add_annotation(
        x=composition_mean[0] + 1.2, y=composition_mean[1] + 0.5,
        text="<b>COMPOSITION</b><br><b>ZONE</b><br>Geodesic reaches<br>valid region!",
        showarrow=True, arrowhead=2,
        ax=40, ay=-20,
        arrowcolor=COLORS['valid'],
        font=dict(size=10, color=COLORS['valid']),
        bgcolor='rgba(220,255,220,0.9)',
        bordercolor=COLORS['valid'],
        row=1, col=2
    )

    # Formula comparison box
    fig.add_annotation(
        x=0.5, y=-0.12,
        xref='paper', yref='paper',
        text="<b>Euclidean:</b> z_★ = z_N + Δ_E + Δ_C  →  <span style='color:#DC3545'>VOID (decoder fails)</span>     |     " +
             "<b>Geodesic:</b> z_★ = γ(N→E→C)  →  <span style='color:#28A745'>ON MANIFOLD (decoder succeeds)</span>",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='gray',
        borderwidth=1
    )

    fig.update_layout(
        title=dict(
            text='<b>Structure E: Curved Manifold - GEODESIC COMPOSITION SUCCEEDS!</b>',
            font=dict(size=20),
            x=0.5
        ),
        height=600,
        width=1300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        template='plotly_white'
    )

    for col in [1, 2]:
        fig.update_xaxes(title_text='PC1 (z_salient)', row=1, col=col, range=[-5, 5])
        fig.update_yaxes(title_text='PC2 (z_salient)', row=1, col=col, range=[-4.5, 5])

    return fig


# =============================================================================
# 3D Manifold Surfaces
# =============================================================================

def create_3d_double_well_manifold():
    """
    3D visualization showing a double-well potential landscape.
    Demonstrates why Euclidean paths fail on curved manifolds.
    """
    # Create double-well surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Double well in x-direction, single well in y-direction
    # This creates two "basins" with a ridge between them
    Z = (X**2 - 1)**2 + 0.5 * Y**2

    # Invert for visualization (wells become hills of probability)
    Z_prob = np.exp(-Z)

    fig = go.Figure()

    # Surface plot
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_prob,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title='Probability<br>Density', x=1.02),
            name='Manifold surface'
        )
    )

    # Mark disease locations (on the surface)
    # Normal: left well
    normal_pos = [-1, 0, np.exp(-0)]
    # Effusion: right well
    effusion_pos = [1, 0, np.exp(-0)]
    # Cardiomegaly: front of manifold
    cardiomegaly_pos = [0, 1.5, np.exp(-(1 + 0.5*1.5**2))]

    # Add disease markers
    fig.add_trace(
        go.Scatter3d(
            x=[normal_pos[0]], y=[normal_pos[1]], z=[normal_pos[2] + 0.1],
            mode='markers+text',
            marker=dict(size=12, color=COLORS['normal'], symbol='circle'),
            text=['Normal'],
            textposition='top center',
            name='Normal'
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[effusion_pos[0]], y=[effusion_pos[1]], z=[effusion_pos[2] + 0.1],
            mode='markers+text',
            marker=dict(size=12, color=COLORS['effusion'], symbol='diamond'),
            text=['Effusion'],
            textposition='top center',
            name='Effusion'
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[cardiomegaly_pos[0]], y=[cardiomegaly_pos[1]], z=[cardiomegaly_pos[2] + 0.1],
            mode='markers+text',
            marker=dict(size=12, color=COLORS['cardiomegaly'], symbol='square'),
            text=['Cardiomegaly'],
            textposition='top center',
            name='Cardiomegaly'
        )
    )

    # Euclidean path (straight line through low-density region)
    euclidean_t = np.linspace(0, 1, 30)
    euclidean_x = normal_pos[0] + (effusion_pos[0] - normal_pos[0]) * euclidean_t
    euclidean_y = np.zeros_like(euclidean_t)
    euclidean_z = np.exp(-((euclidean_x**2 - 1)**2 + 0.5 * euclidean_y**2))

    fig.add_trace(
        go.Scatter3d(
            x=euclidean_x, y=euclidean_y, z=euclidean_z + 0.05,
            mode='lines',
            line=dict(color=COLORS['euclidean'], width=8, dash='dash'),
            name='Euclidean path (crosses barrier)'
        )
    )

    # Geodesic path (goes around the barrier)
    geodesic_t = np.linspace(0, np.pi, 30)
    geodesic_x = np.cos(geodesic_t)
    geodesic_y = 0.8 * np.sin(geodesic_t)
    geodesic_z = np.exp(-((geodesic_x**2 - 1)**2 + 0.5 * geodesic_y**2))

    fig.add_trace(
        go.Scatter3d(
            x=geodesic_x, y=geodesic_y, z=geodesic_z + 0.05,
            mode='lines',
            line=dict(color=COLORS['geodesic'], width=8),
            name='Geodesic path (follows manifold)'
        )
    )

    # Mark the barrier (low probability region)
    barrier_x = np.zeros(20)
    barrier_y = np.linspace(-1, 1, 20)
    barrier_z = np.exp(-((barrier_x**2 - 1)**2 + 0.5 * barrier_y**2))

    fig.add_trace(
        go.Scatter3d(
            x=barrier_x, y=barrier_y, z=barrier_z + 0.02,
            mode='lines',
            line=dict(color=COLORS['invalid'], width=6, dash='dot'),
            name='Energy barrier (VOID)'
        )
    )

    fig.update_layout(
        title=dict(
            text='<b>3D Double-Well Manifold: Why Euclidean Fails</b><br>' +
                 '<sup>Euclidean path crosses low-density barrier; Geodesic stays in high-density regions</sup>',
            font=dict(size=18),
            x=0.5
        ),
        scene=dict(
            xaxis_title='z₁ (Disease dimension 1)',
            yaxis_title='z₂ (Disease dimension 2)',
            zaxis_title='Probability Density',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=700,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_3d_saddle_composition():
    """
    3D Saddle/Swiss Roll manifold showing SUCCESSFUL geodesic composition.

    This visualization demonstrates that on a curved manifold:
    1. Disease clusters lie on a curved surface
    2. A valid composition zone EXISTS on the manifold
    3. Euclidean composition misses it (lands off-manifold)
    4. Geodesic composition reaches it (stays on-manifold)
    """
    # Create a saddle-like surface where diseases lie on different regions
    # The key: the composition zone is ON the surface, reachable by geodesic

    u = np.linspace(-2, 2, 80)
    v = np.linspace(-2, 2, 80)
    U, V = np.meshgrid(u, v)

    # Saddle surface: z = x² - y² (hyperbolic paraboloid)
    # Modified to create a "basin" shape with raised edges
    Z_surface = 0.3 * (U**2 - V**2) + 0.1 * np.sin(U * np.pi) * np.cos(V * np.pi)

    fig = go.Figure()

    # Main surface
    fig.add_trace(
        go.Surface(
            x=U, y=V, z=Z_surface,
            colorscale=[
                [0, '#FFE4E1'],      # Light red (low)
                [0.3, '#FFFACD'],    # Pale yellow
                [0.5, '#F0FFF0'],    # Honeydew
                [0.7, '#E0FFFF'],    # Light cyan
                [1, '#87CEEB']       # Sky blue (high)
            ],
            opacity=0.75,
            showscale=True,
            colorbar=dict(title='Surface<br>Height', x=1.02, len=0.6),
            name='Curved Manifold',
            hovertemplate='Manifold surface<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
        )
    )

    # Disease positions ON the surface
    # Normal: bottom-left region
    normal_x, normal_y = -1.5, -1.0
    normal_z = 0.3 * (normal_x**2 - normal_y**2) + 0.1 * np.sin(normal_x * np.pi) * np.cos(normal_y * np.pi)

    # Effusion: top region (high y)
    effusion_x, effusion_y = 0.0, 1.5
    effusion_z = 0.3 * (effusion_x**2 - effusion_y**2) + 0.1 * np.sin(effusion_x * np.pi) * np.cos(effusion_y * np.pi)

    # Cardiomegaly: right region (high x)
    card_x, card_y = 1.5, -0.5
    card_z = 0.3 * (card_x**2 - card_y**2) + 0.1 * np.sin(card_x * np.pi) * np.cos(card_y * np.pi)

    # COMPOSITION ZONE: upper-right area where geodesics from both diseases meet
    comp_x, comp_y = 1.2, 1.0
    comp_z = 0.3 * (comp_x**2 - comp_y**2) + 0.1 * np.sin(comp_x * np.pi) * np.cos(comp_y * np.pi)

    # Euclidean composition point (OFF the surface!)
    delta_eff = np.array([effusion_x - normal_x, effusion_y - normal_y, effusion_z - normal_z])
    delta_card = np.array([card_x - normal_x, card_y - normal_y, card_z - normal_z])
    euclidean_comp = np.array([normal_x, normal_y, normal_z]) + delta_eff + delta_card

    # Calculate where the Euclidean point SHOULD be on the surface
    euclidean_surface_z = 0.3 * (euclidean_comp[0]**2 - euclidean_comp[1]**2) + \
                          0.1 * np.sin(euclidean_comp[0] * np.pi) * np.cos(euclidean_comp[1] * np.pi)

    # Add disease cluster markers
    # Normal cluster
    np.random.seed(42)
    n_cluster = 15
    normal_scatter = np.random.randn(n_cluster, 2) * 0.2 + [normal_x, normal_y]
    normal_scatter_z = 0.3 * (normal_scatter[:, 0]**2 - normal_scatter[:, 1]**2) + \
                       0.1 * np.sin(normal_scatter[:, 0] * np.pi) * np.cos(normal_scatter[:, 1] * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=normal_scatter[:, 0], y=normal_scatter[:, 1], z=normal_scatter_z + 0.08,
            mode='markers',
            marker=dict(size=6, color=COLORS['normal'], opacity=0.8),
            name='Normal cluster',
            hovertemplate='Normal sample<extra></extra>'
        )
    )

    # Effusion cluster
    effusion_scatter = np.random.randn(n_cluster, 2) * 0.2 + [effusion_x, effusion_y]
    effusion_scatter_z = 0.3 * (effusion_scatter[:, 0]**2 - effusion_scatter[:, 1]**2) + \
                         0.1 * np.sin(effusion_scatter[:, 0] * np.pi) * np.cos(effusion_scatter[:, 1] * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=effusion_scatter[:, 0], y=effusion_scatter[:, 1], z=effusion_scatter_z + 0.08,
            mode='markers',
            marker=dict(size=6, color=COLORS['effusion'], opacity=0.8, symbol='diamond'),
            name='Effusion cluster',
            hovertemplate='Effusion sample<extra></extra>'
        )
    )

    # Cardiomegaly cluster
    card_scatter = np.random.randn(n_cluster, 2) * 0.2 + [card_x, card_y]
    card_scatter_z = 0.3 * (card_scatter[:, 0]**2 - card_scatter[:, 1]**2) + \
                     0.1 * np.sin(card_scatter[:, 0] * np.pi) * np.cos(card_scatter[:, 1] * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=card_scatter[:, 0], y=card_scatter[:, 1], z=card_scatter_z + 0.08,
            mode='markers',
            marker=dict(size=6, color=COLORS['cardiomegaly'], opacity=0.8, symbol='square'),
            name='Cardiomegaly cluster',
            hovertemplate='Cardiomegaly sample<extra></extra>'
        )
    )

    # Composition zone cluster
    comp_scatter = np.random.randn(n_cluster, 2) * 0.15 + [comp_x, comp_y]
    comp_scatter_z = 0.3 * (comp_scatter[:, 0]**2 - comp_scatter[:, 1]**2) + \
                     0.1 * np.sin(comp_scatter[:, 0] * np.pi) * np.cos(comp_scatter[:, 1] * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=comp_scatter[:, 0], y=comp_scatter[:, 1], z=comp_scatter_z + 0.08,
            mode='markers',
            marker=dict(size=6, color=COLORS['valid'], opacity=0.8, symbol='cross'),
            name='Composition zone',
            hovertemplate='Composition (Both diseases)<extra></extra>'
        )
    )

    # Mark disease centers
    fig.add_trace(
        go.Scatter3d(
            x=[normal_x], y=[normal_y], z=[normal_z + 0.15],
            mode='markers+text',
            marker=dict(size=14, color=COLORS['normal'], symbol='circle',
                       line=dict(width=2, color='white')),
            text=['Normal<br>(Origin)'],
            textposition='top center',
            textfont=dict(size=11, color=COLORS['normal']),
            name='Normal center',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[effusion_x], y=[effusion_y], z=[effusion_z + 0.15],
            mode='markers+text',
            marker=dict(size=14, color=COLORS['effusion'], symbol='diamond',
                       line=dict(width=2, color='white')),
            text=['Effusion'],
            textposition='top center',
            textfont=dict(size=11, color=COLORS['effusion']),
            name='Effusion center',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[card_x], y=[card_y], z=[card_z + 0.15],
            mode='markers+text',
            marker=dict(size=14, color=COLORS['cardiomegaly'], symbol='square',
                       line=dict(width=2, color='white')),
            text=['Cardiomegaly'],
            textposition='top center',
            textfont=dict(size=11, color=COLORS['cardiomegaly']),
            name='Cardiomegaly center',
            showlegend=False
        )
    )

    # === EUCLIDEAN COMPOSITION (FAILS) ===
    # Draw Euclidean delta vectors from Normal
    # Show that straight-line addition lands OFF the surface

    fig.add_trace(
        go.Scatter3d(
            x=[normal_x, euclidean_comp[0]],
            y=[normal_y, euclidean_comp[1]],
            z=[normal_z + 0.1, euclidean_comp[2] + 0.1],
            mode='lines',
            line=dict(color=COLORS['euclidean'], width=8, dash='dash'),
            name='Euclidean composition'
        )
    )

    # Mark the Euclidean composition point (floating ABOVE or BELOW surface)
    fig.add_trace(
        go.Scatter3d(
            x=[euclidean_comp[0]], y=[euclidean_comp[1]], z=[euclidean_comp[2] + 0.1],
            mode='markers+text',
            marker=dict(size=18, color=COLORS['invalid'], symbol='x',
                       line=dict(width=2, color='darkred')),
            text=['OFF MANIFOLD'],
            textposition='top center',
            textfont=dict(size=10, color=COLORS['invalid']),
            name='Euclidean result (FAILS)',
            hovertemplate='<b>Euclidean Composition</b><br>z_N + Δ_E + Δ_C<br>' +
                         'x: %.2f, y: %.2f, z: %.2f<br>' % (euclidean_comp[0], euclidean_comp[1], euclidean_comp[2]) +
                         'Surface z at this (x,y): %.2f<br>' % euclidean_surface_z +
                         '<b>OFF SURFACE by %.2f!</b><extra></extra>' % abs(euclidean_comp[2] - euclidean_surface_z)
        )
    )

    # Draw a vertical line showing how far off the surface it is
    fig.add_trace(
        go.Scatter3d(
            x=[euclidean_comp[0], euclidean_comp[0]],
            y=[euclidean_comp[1], euclidean_comp[1]],
            z=[euclidean_surface_z, euclidean_comp[2] + 0.1],
            mode='lines',
            line=dict(color=COLORS['invalid'], width=4, dash='dot'),
            name='Distance from manifold',
            showlegend=False
        )
    )

    # === GEODESIC COMPOSITION (SUCCEEDS) ===
    # Path from Normal → Effusion (following surface)
    t1 = np.linspace(0, 1, 20)
    geo1_x = normal_x + (effusion_x - normal_x) * t1
    geo1_y = normal_y + (effusion_y - normal_y) * t1
    geo1_z = 0.3 * (geo1_x**2 - geo1_y**2) + 0.1 * np.sin(geo1_x * np.pi) * np.cos(geo1_y * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=geo1_x, y=geo1_y, z=geo1_z + 0.1,
            mode='lines',
            line=dict(color=COLORS['geodesic'], width=7),
            name='Geodesic path 1 (N→E)',
            hovertemplate='Geodesic: Normal → Effusion<br>Stays on manifold!<extra></extra>'
        )
    )

    # Path from Effusion → Composition (following surface)
    t2 = np.linspace(0, 1, 20)
    geo2_x = effusion_x + (comp_x - effusion_x) * t2
    geo2_y = effusion_y + (comp_y - effusion_y) * t2
    geo2_z = 0.3 * (geo2_x**2 - geo2_y**2) + 0.1 * np.sin(geo2_x * np.pi) * np.cos(geo2_y * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=geo2_x, y=geo2_y, z=geo2_z + 0.1,
            mode='lines',
            line=dict(color=COLORS['geodesic'], width=7),
            name='Geodesic path 2 (E→Comp)',
            showlegend=False
        )
    )

    # Path from Cardiomegaly → Composition (following surface)
    t3 = np.linspace(0, 1, 20)
    geo3_x = card_x + (comp_x - card_x) * t3
    geo3_y = card_y + (comp_y - card_y) * t3
    geo3_z = 0.3 * (geo3_x**2 - geo3_y**2) + 0.1 * np.sin(geo3_x * np.pi) * np.cos(geo3_y * np.pi)

    fig.add_trace(
        go.Scatter3d(
            x=geo3_x, y=geo3_y, z=geo3_z + 0.1,
            mode='lines',
            line=dict(color=COLORS['geodesic'], width=7, dash='dot'),
            name='Geodesic path 3 (C→Comp)',
            showlegend=False
        )
    )

    # Mark the geodesic composition point (ON THE SURFACE)
    fig.add_trace(
        go.Scatter3d(
            x=[comp_x], y=[comp_y], z=[comp_z + 0.18],
            mode='markers+text',
            marker=dict(size=20, color=COLORS['valid'], symbol='diamond',
                       line=dict(width=2, color='darkgreen')),
            text=['ON MANIFOLD'],
            textposition='top center',
            textfont=dict(size=11, color=COLORS['valid']),
            name='Geodesic result (SUCCEEDS)',
            hovertemplate='<b>Geodesic Composition</b><br>Following manifold curvature<br>' +
                         'x: %.2f, y: %.2f, z: %.2f<br>' % (comp_x, comp_y, comp_z) +
                         '<b>ON SURFACE - Valid composition!</b><extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text='<b>3D Curved Manifold: Geodesic Composition SUCCEEDS</b><br>' +
                 '<sup style="color:#28A745">Geodesic finds composition zone ON the manifold</sup>  |  ' +
                 '<sup style="color:#DC3545">Euclidean lands OFF the manifold</sup>',
            font=dict(size=18),
            x=0.5
        ),
        scene=dict(
            xaxis_title='z₁ (Effusion dimension)',
            yaxis_title='z₂ (Cardiomegaly dimension)',
            zaxis_title='Manifold Height',
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.3)),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.2, z=0.6)
        ),
        height=750,
        width=1100,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig


def create_3d_torus_manifold():
    """
    3D Torus manifold demonstrating periodic/cyclic disease representations.
    Shows why simple addition fails on non-Euclidean geometries.
    """
    # Create torus
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    U, V = np.meshgrid(u, v)

    R = 2  # Major radius
    r = 0.8  # Minor radius

    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)

    fig = go.Figure()

    # Torus surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, '#E8F4F8'], [0.5, '#B8D4E3'], [1, '#87CEEB']],
            opacity=0.7,
            showscale=False,
            name='Torus manifold'
        )
    )

    # Disease locations on torus
    diseases = {
        'Normal': (0, 0),
        'Effusion': (np.pi/2, 0),
        'Cardiomegaly': (0, np.pi/2),
        'Composed': (np.pi/2, np.pi/2)
    }

    colors = {
        'Normal': COLORS['normal'],
        'Effusion': COLORS['effusion'],
        'Cardiomegaly': COLORS['cardiomegaly'],
        'Composed': COLORS['valid']
    }

    symbols = {
        'Normal': 'circle',
        'Effusion': 'diamond',
        'Cardiomegaly': 'square',
        'Composed': 'cross'  # Note: Scatter3d doesn't support 'star'
    }

    for name, (u_pos, v_pos) in diseases.items():
        x = (R + r * np.cos(v_pos)) * np.cos(u_pos)
        y = (R + r * np.cos(v_pos)) * np.sin(u_pos)
        z = r * np.sin(v_pos)

        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z + 0.2],
                mode='markers+text',
                marker=dict(size=15, color=colors[name], symbol=symbols[name],
                           line=dict(width=2, color='white')),
                text=[name],
                textposition='top center',
                name=name
            )
        )

    # Geodesic path on torus (from Normal to Effusion)
    path_u = np.linspace(0, np.pi/2, 30)
    path_v = np.zeros_like(path_u)
    path_x = (R + r * np.cos(path_v)) * np.cos(path_u)
    path_y = (R + r * np.cos(path_v)) * np.sin(path_u)
    path_z = r * np.sin(path_v)

    fig.add_trace(
        go.Scatter3d(
            x=path_x, y=path_y, z=path_z + 0.1,
            mode='lines',
            line=dict(color=COLORS['geodesic'], width=6),
            name='Geodesic path'
        )
    )

    # Wrong Euclidean path (straight line through torus)
    wrong_path_t = np.linspace(0, 1, 30)
    start_x = (R + r) * 1
    start_y = 0
    start_z = 0
    end_x = 0
    end_y = (R + r) * 1
    end_z = 0

    wrong_x = start_x + (end_x - start_x) * wrong_path_t
    wrong_y = start_y + (end_y - start_y) * wrong_path_t
    wrong_z = start_z + (end_z - start_z) * wrong_path_t

    fig.add_trace(
        go.Scatter3d(
            x=wrong_x, y=wrong_y, z=wrong_z,
            mode='lines',
            line=dict(color=COLORS['invalid'], width=6, dash='dash'),
            name='Euclidean path (OFF MANIFOLD!)'
        )
    )

    fig.update_layout(
        title=dict(
            text='<b>Torus Manifold: Periodic Disease Space</b><br>' +
                 '<sup>Diseases may have cyclic/periodic relationships requiring manifold-aware navigation</sup>',
            font=dict(size=18),
            x=0.5
        ),
        scene=dict(
            xaxis_title='z₁',
            yaxis_title='z₂',
            zaxis_title='z₃',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            aspectmode='data'
        ),
        height=700,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


# =============================================================================
# Comparison Dashboard
# =============================================================================

def create_structure_comparison_dashboard():
    """
    Side-by-side comparison of all five manifold structures.
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '<b>A: Entangled</b><br><sup>❌ FAILS</sup>',
            '<b>B: Single-Axis</b><br><sup>❌ FAILS</sup>',
            '<b>C: Isolated</b><br><sup>❌ FAILS (LDM)</sup>',
            '<b>D: Orthogonal</b><br><sup>IDEAL</sup>',
            '<b>E: Curved</b><br><sup>⚠️ CONDITIONAL</sup>',
            '<b>Composition Outcomes</b>'
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    np.random.seed(42)

    # Structure A: Entangled
    for i, (color, name, symbol) in enumerate([
        (COLORS['normal'], 'Normal', 'circle'),
        (COLORS['effusion'], 'Effusion', 'triangle-up'),
        (COLORS['cardiomegaly'], 'Cardiomegaly', 'square')
    ]):
        cluster = generate_cluster([0, 0], n_points=50, std=0.6, seed=42+i)
        fig.add_trace(
            go.Scatter(x=cluster[:, 0], y=cluster[:, 1], mode='markers',
                      marker=dict(size=6, color=color, symbol=symbol, opacity=0.6),
                      showlegend=False),
            row=1, col=1
        )

    # Structure B: Single-Axis
    for i, (x_offset, color, symbol) in enumerate([
        (0, COLORS['normal'], 'circle'),
        (2, COLORS['effusion'], 'triangle-up'),
        (4, COLORS['cardiomegaly'], 'square')
    ]):
        cluster = generate_cluster([x_offset, 0], n_points=50, std=0.2, seed=42+i)
        cluster[:, 1] *= 0.3
        fig.add_trace(
            go.Scatter(x=cluster[:, 0], y=cluster[:, 1], mode='markers',
                      marker=dict(size=6, color=color, symbol=symbol, opacity=0.6),
                      showlegend=False),
            row=1, col=2
        )

    # Structure C: Isolated
    for i, (center, color, symbol) in enumerate([
        ([-1.5, -1.5], COLORS['normal'], 'circle'),
        ([1.5, 1.5], COLORS['effusion'], 'triangle-up'),
        ([1.5, -1.5], COLORS['cardiomegaly'], 'square')
    ]):
        cluster = generate_cluster(center, n_points=50, std=0.3, seed=42+i)
        fig.add_trace(
            go.Scatter(x=cluster[:, 0], y=cluster[:, 1], mode='markers',
                      marker=dict(size=6, color=color, symbol=symbol, opacity=0.6),
                      showlegend=False),
            row=1, col=3
        )

    # Structure D: Orthogonal (IDEAL)
    for i, (center, color, symbol) in enumerate([
        ([0, 0], COLORS['normal'], 'circle'),
        ([0, 2], COLORS['effusion'], 'triangle-up'),
        ([2, 0], COLORS['cardiomegaly'], 'square'),
        ([2, 2], COLORS['valid'], 'star')
    ]):
        cluster = generate_cluster(center, n_points=40, std=0.25, seed=42+i)
        fig.add_trace(
            go.Scatter(x=cluster[:, 0], y=cluster[:, 1], mode='markers',
                      marker=dict(size=6, color=color, symbol=symbol, opacity=0.6),
                      showlegend=False),
            row=2, col=1
        )

    # Structure E: Curved
    t = np.linspace(0, np.pi, 100)
    curve_x = np.cos(t) * 2
    curve_y = np.sin(t) * 2 - 1
    noise = np.random.randn(100, 2) * 0.15
    curve_points = np.column_stack([curve_x, curve_y]) + noise

    fig.add_trace(
        go.Scatter(x=curve_points[:30, 0], y=curve_points[:30, 1], mode='markers',
                  marker=dict(size=6, color=COLORS['normal'], opacity=0.6),
                  showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=curve_points[30:70, 0], y=curve_points[30:70, 1], mode='markers',
                  marker=dict(size=6, color=COLORS['effusion'], symbol='triangle-up', opacity=0.6),
                  showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=curve_points[70:, 0], y=curve_points[70:, 1], mode='markers',
                  marker=dict(size=6, color=COLORS['cardiomegaly'], symbol='square', opacity=0.6),
                  showlegend=False),
        row=2, col=2
    )

    # Outcomes summary (row 2, col 3)
    outcomes = pd.DataFrame({
        'Structure': ['A: Entangled', 'B: Single-Axis', 'C: Isolated', 'D: Orthogonal', 'E: Curved'],
        'Outcome': ['FAILS', 'FAILS', 'FAILS (LDM)', 'SUCCESS', 'CONDITIONAL'],
        'Color': [COLORS['invalid'], COLORS['invalid'], COLORS['invalid'],
                 COLORS['valid'], COLORS['euclidean']]
    })

    fig.add_trace(
        go.Bar(
            x=outcomes['Structure'],
            y=[1, 1, 1, 1, 1],
            marker_color=outcomes['Color'],
            text=outcomes['Outcome'],
            textposition='inside',
            textfont=dict(color='white', size=11),
            showlegend=False
        ),
        row=2, col=3
    )

    fig.update_layout(
        title=dict(
            text='<b>Manifold Structure Comparison: Which Enables Composition?</b>',
            font=dict(size=20),
            x=0.5
        ),
        height=700,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )

    # Update axes
    for row in [1, 2]:
        for col in [1, 2, 3]:
            if not (row == 2 and col == 3):
                fig.update_xaxes(title_text='PC1', row=row, col=col, showticklabels=False)
                fig.update_yaxes(title_text='PC2', row=row, col=col, showticklabels=False)

    fig.update_xaxes(tickangle=45, row=2, col=3)
    fig.update_yaxes(showticklabels=False, row=2, col=3)

    return fig


# =============================================================================
# Interpolation Quality Visualization
# =============================================================================

def create_interpolation_quality_comparison():
    """
    Visualize interpolation quality across different manifold structures.
    Shows "smoothness" metric vs interpolation step.
    """
    np.random.seed(42)

    n_steps = 20
    t = np.linspace(0, 1, n_steps)

    # Structure D (Good): Smooth interpolation
    good_diffs = 0.1 + 0.02 * np.random.randn(n_steps - 1)
    good_diffs = np.abs(good_diffs)

    # Structure C (Bad): Spike in the middle (void crossing)
    bad_diffs = 0.1 + 0.02 * np.random.randn(n_steps - 1)
    bad_diffs = np.abs(bad_diffs)
    bad_diffs[8:12] = 0.5 + 0.1 * np.random.randn(4)  # Spike!

    # Structure E with Euclidean (Very bad)
    very_bad_diffs = 0.1 + 0.02 * np.random.randn(n_steps - 1)
    very_bad_diffs = np.abs(very_bad_diffs)
    very_bad_diffs[7:13] = 0.8 + 0.15 * np.random.randn(6)  # Big spike!

    # Structure E with Geodesic (Good again)
    geodesic_diffs = 0.12 + 0.03 * np.random.randn(n_steps - 1)
    geodesic_diffs = np.abs(geodesic_diffs)

    fig = go.Figure()

    # Good interpolation (Structure D)
    fig.add_trace(
        go.Scatter(
            x=t[:-1], y=good_diffs,
            mode='lines+markers',
            name='Structure D (Orthogonal) - SMOOTH',
            line=dict(color=COLORS['valid'], width=3),
            marker=dict(size=8)
        )
    )

    # Bad interpolation (Structure C)
    fig.add_trace(
        go.Scatter(
            x=t[:-1], y=bad_diffs,
            mode='lines+markers',
            name='Structure C (Isolated) - VOID SPIKE',
            line=dict(color=COLORS['cardiomegaly'], width=3),
            marker=dict(size=8)
        )
    )

    # Very bad (Structure E, Euclidean)
    fig.add_trace(
        go.Scatter(
            x=t[:-1], y=very_bad_diffs,
            mode='lines+markers',
            name='Structure E (Euclidean path) - SEVERE VOID',
            line=dict(color=COLORS['invalid'], width=3, dash='dash'),
            marker=dict(size=8)
        )
    )

    # Geodesic on curved
    fig.add_trace(
        go.Scatter(
            x=t[:-1], y=geodesic_diffs,
            mode='lines+markers',
            name='Structure E (Geodesic path) - SMOOTH',
            line=dict(color=COLORS['geodesic'], width=3),
            marker=dict(size=8)
        )
    )

    # Threshold line
    fig.add_hline(y=0.2, line_dash="dash", line_color="gray",
                 annotation_text="Spike threshold (2x mean)")

    # Shade void region
    fig.add_vrect(x0=0.35, x1=0.65, fillcolor="rgba(255,0,0,0.1)",
                 line_width=0, annotation_text="VOID REGION",
                 annotation_position="top")

    fig.update_layout(
        title=dict(
            text='<b>Interpolation Smoothness Test</b><br>' +
                 '<sup>Frame-to-frame difference during disease interpolation</sup>',
            font=dict(size=20),
            x=0.5
        ),
        xaxis_title='Interpolation parameter t (Effusion → Cardiomegaly)',
        yaxis_title='Frame-to-frame difference |x(t+1) - x(t)|',
        height=500,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99
        ),
        template='plotly_white'
    )

    return fig


# =============================================================================
# Orthogonality & Coverage Metrics Visualization
# =============================================================================

def create_metrics_dashboard():
    """
    Dashboard showing key metrics for compositional readiness.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Disease Axis Orthogonality</b>',
            '<b>Manifold Coverage</b>',
            '<b>Normal Anchoring</b>',
            '<b>Manifold Flatness</b>'
        ),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )

    # Orthogonality gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=0.15,
            title={'text': "cos(Δz_eff, Δz_card)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': COLORS['valid']},
                'steps': [
                    {'range': [0, 0.3], 'color': '#90EE90'},
                    {'range': [0.3, 0.6], 'color': '#FFE4B5'},
                    {'range': [0.6, 1], 'color': '#FFB6C1'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 0.3
                }
            },
            domain={'row': 0, 'column': 0}
        ),
        row=1, col=1
    )

    # Coverage gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=72,
            title={'text': "% bins occupied", 'font': {'size': 14}},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': COLORS['valid']},
                'steps': [
                    {'range': [0, 30], 'color': '#FFB6C1'},
                    {'range': [30, 60], 'color': '#FFE4B5'},
                    {'range': [60, 100], 'color': '#90EE90'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            },
            domain={'row': 0, 'column': 1}
        ),
        row=1, col=2
    )

    # Anchoring gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=0.23,
            title={'text': "‖z_normal - 0‖", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': COLORS['valid']},
                'steps': [
                    {'range': [0, 0.5], 'color': '#90EE90'},
                    {'range': [0.5, 0.8], 'color': '#FFE4B5'},
                    {'range': [0.8, 1], 'color': '#FFB6C1'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            },
            domain={'row': 1, 'column': 0}
        ),
        row=2, col=1
    )

    # Flatness gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=3.2,
            title={'text': "max/min det(G)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [1, 20]},
                'bar': {'color': COLORS['valid']},
                'steps': [
                    {'range': [1, 10], 'color': '#90EE90'},
                    {'range': [10, 15], 'color': '#FFE4B5'},
                    {'range': [15, 20], 'color': '#FFB6C1'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            },
            domain={'row': 1, 'column': 1}
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>Compositional Readiness Metrics</b><br>' +
                 '<sup>Green = Good | Yellow = Marginal | Red threshold = Fail</sup>',
            font=dict(size=20),
            x=0.5
        ),
        height=600,
        width=900,
        template='plotly_white',
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"}
    )

    return fig


# =============================================================================
# Failure Case Gallery
# =============================================================================

def create_failure_gallery():
    """
    Visual gallery of composition failure modes with simulated X-ray patterns.
    """
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=(
            'Intended:<br>Effusion + Cardiomegaly',
            'Case 1: Ghosting<br>(Entangled)',
            'Case 2: Cancellation<br>(Single-Axis)',
            'Case 3: Void Artifacts<br>(Isolated)',
            '',
            'Expected:<br>Both diseases clear',
            'Result:<br>Overlapping features',
            'Result:<br>Neither disease visible'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Create synthetic "X-ray" patterns using 2D Gaussians
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Intended composition (good)
    heart_good = 3 * np.exp(-((X-5)**2 + (Y-5)**2) / 4)  # Enlarged heart
    effusion_good = 2 * np.exp(-((X-2)**2/8 + (Y-3)**2/2))  # Fluid
    intended = heart_good + effusion_good

    # Case 1: Ghosting (features overlap incorrectly)
    ghost = heart_good + effusion_good + 1.5 * np.exp(-((X-5)**2 + (Y-6)**2) / 3)  # Ghost feature

    # Case 2: Cancellation (neither visible)
    cancelled = 0.5 * np.ones_like(X) + 0.2 * np.random.randn(*X.shape)  # Just noise

    # Case 3: Void artifacts (blurry mess)
    void_artifact = 0.8 * np.exp(-((X-5)**2 + (Y-5)**2) / 20)  # Blurred everything
    void_artifact += 0.3 * np.random.randn(*X.shape)

    # Expected good result
    expected = heart_good + effusion_good

    colorscale = 'gray_r'  # X-ray style

    # Row 1: Cases
    for col, (data, title) in enumerate([
        (intended, 'Target'),
        (ghost, 'Ghosting'),
        (cancelled, 'Cancelled'),
        (void_artifact, 'Void')
    ], 1):
        fig.add_trace(
            go.Heatmap(z=data, colorscale=colorscale, showscale=False),
            row=1, col=col
        )

    # Row 2: Results comparison
    # Empty first cell
    fig.add_trace(
        go.Heatmap(z=np.zeros_like(X), colorscale=colorscale, showscale=False, opacity=0),
        row=2, col=1
    )

    fig.add_trace(
        go.Heatmap(z=expected, colorscale=colorscale, showscale=False),
        row=2, col=2
    )

    fig.add_trace(
        go.Heatmap(z=ghost, colorscale=colorscale, showscale=False),
        row=2, col=3
    )

    fig.add_trace(
        go.Heatmap(z=cancelled, colorscale=colorscale, showscale=False),
        row=2, col=4
    )

    fig.update_layout(
        title=dict(
            text='<b>Composition Failure Gallery</b><br>' +
                 '<sup>Simulated X-ray patterns showing how different manifold structures cause artifacts</sup>',
            font=dict(size=20),
            x=0.5
        ),
        height=600,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )

    # Hide axes
    for i in range(1, 3):
        for j in range(1, 5):
            fig.update_xaxes(showticklabels=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, row=i, col=j)

    return fig


# =============================================================================
# Main execution and save functions
# =============================================================================

def save_all_visualizations(output_dir: str = None):
    """
    Generate and save all visualizations as interactive HTML files.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'visualizations'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Compositional Geometry Visualizations")
    print("=" * 60)

    visualizations = [
        ('structure_a_entangled', create_structure_a_visualization, 'Structure A: Entangled Overlap'),
        ('structure_b_single_axis', create_structure_b_visualization, 'Structure B: Single-Axis Collapse'),
        ('structure_c_isolated', create_structure_c_visualization, 'Structure C: Isolated Clusters'),
        ('structure_d_orthogonal', create_structure_d_visualization, 'Structure D: Orthogonal Factorized (IDEAL)'),
        ('structure_e_curved', create_structure_e_visualization, 'Structure E: Curved Manifold'),
        ('3d_double_well', create_3d_double_well_manifold, '3D Double-Well Manifold'),
        ('3d_saddle_composition', create_3d_saddle_composition, '3D Saddle Manifold - Geodesic Composition'),
        ('3d_torus', create_3d_torus_manifold, '3D Torus Manifold'),
        ('structure_comparison', create_structure_comparison_dashboard, 'All Structures Comparison'),
        ('interpolation_quality', create_interpolation_quality_comparison, 'Interpolation Quality'),
        ('metrics_dashboard', create_metrics_dashboard, 'Compositional Readiness Metrics'),
        ('failure_gallery', create_failure_gallery, 'Failure Case Gallery'),
    ]

    saved_files = []

    for filename, create_func, description in visualizations:
        print(f"\n  → Generating: {description}...")
        try:
            fig = create_func()
            filepath = output_dir / f'{filename}.html'
            fig.write_html(str(filepath), include_plotlyjs='cdn')
            saved_files.append(filepath)
            print(f"    Saved: {filepath}")
        except Exception as e:
            print(f"    Error: {e}")

    # Create index HTML
    index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Compositional Geometry Visualizations - SepVAE</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }
        h2 {
            color: #A23B72;
            margin-top: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }
        .card a {
            color: #2E86AB;
            text-decoration: none;
            font-size: 1.2em;
            font-weight: bold;
        }
        .card a:hover {
            text-decoration: underline;
        }
        .card p {
            color: #666;
            margin-top: 10px;
        }
        .fail { border-left: 5px solid #DC3545; }
        .success { border-left: 5px solid #28A745; }
        .conditional { border-left: 5px solid #FD7E14; }
        .info { border-left: 5px solid #2E86AB; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
    </style>
</head>
<body>
    <h1>🫁 Compositional Geometry for Multi-Disease Synthesis</h1>
    <p>Interactive visualizations demonstrating latent manifold structures and their impact on disease composition in chest X-ray synthesis.</p>

    <h2>📊 Manifold Structure Taxonomy</h2>
    <div class="grid">
        <div class="card fail">
            <a href="structure_a_entangled.html">Structure A: Entangled Overlap</a>
            <p>❌ <strong>FAILS</strong> - All disease classes overlap, no distinct directions exist.</p>
        </div>
        <div class="card fail">
            <a href="structure_b_single_axis.html">Structure B: Single-Axis Collapse</a>
            <p>❌ <strong>FAILS</strong> - Diseases encoded along same axis, composition = interpolation.</p>
        </div>
        <div class="card fail">
            <a href="structure_c_isolated.html">Structure C: Isolated Clusters</a>
            <p>❌ <strong>FAILS (LDM)</strong> - Well-separated but voids between clusters.</p>
        </div>
        <div class="card success">
            <a href="structure_d_orthogonal.html">Structure D: Orthogonal Factorized</a>
            <p>✅ <strong>IDEAL</strong> - Independent disease axes with dense coverage.</p>
        </div>
        <div class="card success">
            <a href="structure_e_curved.html">Structure E: Curved Manifold</a>
            <p>✅ <strong>SUCCEEDS with Geodesics</strong> - Composition zone exists ON manifold, reachable via geodesic paths.</p>
        </div>
    </div>

    <h2>🌐 3D Manifold Visualizations</h2>
    <div class="grid">
        <div class="card info">
            <a href="3d_double_well.html">3D Double-Well Manifold</a>
            <p>Demonstrates why Euclidean paths fail when crossing energy barriers.</p>
        </div>
        <div class="card success">
            <a href="3d_saddle_composition.html">3D Saddle - Geodesic Composition</a>
            <p>✅ <strong>KEY VISUALIZATION:</strong> Shows how geodesic paths find the composition zone on a curved surface.</p>
        </div>
        <div class="card info">
            <a href="3d_torus.html">3D Torus Manifold</a>
            <p>Shows periodic/cyclic disease relationships requiring manifold-aware navigation.</p>
        </div>
    </div>

    <h2>📈 Diagnostic Tools</h2>
    <div class="grid">
        <div class="card info">
            <a href="structure_comparison.html">All Structures Comparison</a>
            <p>Side-by-side comparison of all five manifold structures.</p>
        </div>
        <div class="card info">
            <a href="interpolation_quality.html">Interpolation Quality Test</a>
            <p>Frame-to-frame smoothness analysis across structures.</p>
        </div>
        <div class="card info">
            <a href="metrics_dashboard.html">Compositional Readiness Metrics</a>
            <p>Gauge dashboard for orthogonality, coverage, anchoring, and flatness.</p>
        </div>
        <div class="card info">
            <a href="failure_gallery.html">Failure Case Gallery</a>
            <p>Simulated X-ray patterns showing composition failure modes.</p>
        </div>
    </div>

    <hr style="margin-top: 40px;">
    <p style="color: #888; text-align: center;">
        Generated for the Multi-head SepVAE Diagnostic Manual<br>
        See <code>docs/sepvae_diagnostic_manual.md</code> Section 4
    </p>
</body>
</html>
"""

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(index_html)
    print(f"\n  → Created index: {index_path}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print(f"Open {index_path} in a browser to view all visualizations")
    print("=" * 60)

    return saved_files


def show_all_visualizations():
    """
    Generate and display all visualizations (for Jupyter notebooks).
    """
    figs = [
        create_structure_a_visualization(),
        create_structure_b_visualization(),
        create_structure_c_visualization(),
        create_structure_d_visualization(),
        create_structure_e_visualization(),
        create_3d_double_well_manifold(),
        create_3d_torus_manifold(),
        create_structure_comparison_dashboard(),
        create_interpolation_quality_comparison(),
        create_metrics_dashboard(),
        create_failure_gallery(),
    ]

    for fig in figs:
        fig.show()

    return figs


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate compositional geometry visualizations for SepVAE'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for HTML files (default: docs/visualizations/)'
    )
    parser.add_argument(
        '--single', '-s',
        type=str,
        choices=['a', 'b', 'c', 'd', 'e', '3d_well', '3d_saddle', '3d_torus', 'compare',
                 'interp', 'metrics', 'gallery'],
        help='Generate only a single visualization'
    )

    args = parser.parse_args()

    if args.single:
        mapping = {
            'a': ('structure_a', create_structure_a_visualization),
            'b': ('structure_b', create_structure_b_visualization),
            'c': ('structure_c', create_structure_c_visualization),
            'd': ('structure_d', create_structure_d_visualization),
            'e': ('structure_e', create_structure_e_visualization),
            '3d_well': ('3d_double_well', create_3d_double_well_manifold),
            '3d_saddle': ('3d_saddle_composition', create_3d_saddle_composition),
            '3d_torus': ('3d_torus', create_3d_torus_manifold),
            'compare': ('comparison', create_structure_comparison_dashboard),
            'interp': ('interpolation', create_interpolation_quality_comparison),
            'metrics': ('metrics', create_metrics_dashboard),
            'gallery': ('gallery', create_failure_gallery),
        }

        name, func = mapping[args.single]
        fig = func()

        output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'visualizations'
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / f'{name}.html'
        fig.write_html(str(filepath), include_plotlyjs='cdn')
        print(f"Saved: {filepath}")
    else:
        save_all_visualizations(args.output_dir)
