"""
VAE Introspection Toolkit
=========================

Visualization tools to understand what's happening inside a VAE:
1. Activation Map Visualization - What features do conv layers detect?
2. Latent Traversals - What does each latent dimension control?
3. Latent Ablation Studies - What information is encoded where?
4. Gradient Attribution - Which input pixels affect which latents?
5. Filter Visualization - What patterns activate each filter?
6. Reconstruction Decomposition - How do z_c and z_d contribute?

Usage:
    from vae_introspection import VAEIntrospector
    introspector = VAEIntrospector(model)
    introspector.visualize_all(sample_image, label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import OrderedDict


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

class ActivationExtractor:
    """Hook-based activation extraction from any layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_activation_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def _get_gradient_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks on specified layers."""
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                # Forward hook for activations
                hook = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)
                # Backward hook for gradients
                hook = module.register_full_backward_hook(self._get_gradient_hook(name))
                self.hooks.append(hook)

    def register_conv_hooks(self):
        """Register hooks on all Conv2d and ConvTranspose2d layers."""
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                hook = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._get_gradient_hook(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return self.gradients


# =============================================================================
# MAIN INTROSPECTOR CLASS
# =============================================================================

class VAEIntrospector:
    """
    Comprehensive VAE visualization and introspection toolkit.

    Provides methods to understand:
    - What features the encoder extracts (activation maps)
    - What each latent dimension controls (traversals)
    - How information flows through the network (gradients)
    - How z_c vs z_d contribute to reconstruction (ablation)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.extractor = ActivationExtractor(model)

    # -------------------------------------------------------------------------
    # 1. ACTIVATION MAP VISUALIZATION
    # -------------------------------------------------------------------------

    def visualize_encoder_activations(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        max_channels: int = 16,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize feature maps at each encoder conv layer.

        Shows what patterns/features each layer detects.
        Early layers: edges, textures
        Later layers: higher-level semantic features
        """
        self.model.eval()
        self.extractor.register_conv_hooks()

        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            _ = self.model(x, label)

        activations = self.extractor.get_activations()
        self.extractor.clear_hooks()

        # Filter to encoder conv layers only
        encoder_acts = {k: v for k, v in activations.items()
                       if 'encoder' in k and 'conv' in k.lower()}

        if not encoder_acts:
            # Try alternative naming
            encoder_acts = {k: v for k, v in activations.items()
                           if not ('decoder' in k or 'deconv' in k)}

        n_layers = len(encoder_acts)
        if n_layers == 0:
            print("No encoder activations found. Check layer naming.")
            return None

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_layers + 1, max_channels + 1, figure=fig)

        # Show input image
        ax_input = fig.add_subplot(gs[0, :4])
        img = x[0].cpu().squeeze().numpy()
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        ax_input.imshow(img * 0.5 + 0.5, cmap='gray')
        ax_input.set_title('Input Image', fontsize=12)
        ax_input.axis('off')

        # Show each layer's activations
        for layer_idx, (name, act) in enumerate(encoder_acts.items()):
            act = act[0].cpu().numpy()  # First sample, shape: [C, H, W]
            n_channels = min(act.shape[0], max_channels)

            # Layer info
            ax_info = fig.add_subplot(gs[layer_idx + 1, 0])
            ax_info.text(0.5, 0.5, f'{name}\n{act.shape}',
                        ha='center', va='center', fontsize=8,
                        transform=ax_info.transAxes)
            ax_info.axis('off')

            # Channel activations
            for ch in range(n_channels):
                ax = fig.add_subplot(gs[layer_idx + 1, ch + 1])
                ax.imshow(act[ch], cmap='viridis')
                ax.axis('off')
                if layer_idx == 0:
                    ax.set_title(f'Ch {ch}', fontsize=8)

        plt.suptitle('Encoder Activation Maps\n(Each row = conv layer, columns = channels)',
                    fontsize=14)
        plt.tight_layout()
        return fig

    def visualize_decoder_activations(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        max_channels: int = 16,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize feature maps at each decoder conv layer.

        Shows how reconstruction is built up from latent code.
        Early layers: coarse structure
        Later layers: fine details
        """
        self.model.eval()
        self.extractor.register_conv_hooks()

        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            x_recon, _, _, _ = self.model(x, label)

        activations = self.extractor.get_activations()
        self.extractor.clear_hooks()

        # Filter to decoder layers
        decoder_acts = {k: v for k, v in activations.items()
                       if 'decoder' in k or 'deconv' in k}

        n_layers = len(decoder_acts)
        if n_layers == 0:
            print("No decoder activations found.")
            return None

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_layers + 2, max_channels + 1, figure=fig)

        # Show reconstruction
        ax_recon = fig.add_subplot(gs[0, :4])
        img = x_recon[0].cpu().squeeze().numpy()
        ax_recon.imshow(img * 0.5 + 0.5, cmap='gray')
        ax_recon.set_title('Reconstruction', fontsize=12)
        ax_recon.axis('off')

        # Show original for comparison
        ax_orig = fig.add_subplot(gs[0, 4:8])
        img_orig = x[0].cpu().squeeze().numpy()
        ax_orig.imshow(img_orig * 0.5 + 0.5, cmap='gray')
        ax_orig.set_title('Original', fontsize=12)
        ax_orig.axis('off')

        # Show each layer's activations
        for layer_idx, (name, act) in enumerate(decoder_acts.items()):
            act = act[0].cpu().numpy()
            n_channels = min(act.shape[0], max_channels)

            ax_info = fig.add_subplot(gs[layer_idx + 1, 0])
            ax_info.text(0.5, 0.5, f'{name}\n{act.shape}',
                        ha='center', va='center', fontsize=8,
                        transform=ax_info.transAxes)
            ax_info.axis('off')

            for ch in range(n_channels):
                ax = fig.add_subplot(gs[layer_idx + 1, ch + 1])
                ax.imshow(act[ch], cmap='viridis')
                ax.axis('off')

        plt.suptitle('Decoder Activation Maps\n(Reconstruction built from coarse → fine)',
                    fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 2. LATENT TRAVERSAL VISUALIZATION
    # -------------------------------------------------------------------------

    def latent_traversal(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        latent_type: str = 'common',  # 'common' or 'disease'
        n_dims: int = 8,
        n_steps: int = 9,
        range_std: float = 3.0,
        figsize: Tuple[int, int] = (15, 12)
    ) -> plt.Figure:
        """
        Traverse each latent dimension independently to see what it controls.

        For each dimension:
        - Fix all other dimensions at their encoded values
        - Vary the target dimension from -range_std to +range_std
        - Observe what changes in the reconstruction

        This reveals the semantic meaning of each latent dimension.
        """
        self.model.eval()
        x = x.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            z_c, z_d, info = self.model.encode(x, label)

        # Select which latent to traverse
        if latent_type == 'common':
            z_base = z_c.clone()
            z_fixed = z_d.clone()
            z_dim = z_c.shape[1]
            title = 'Common Latent (z_c) Traversal - What anatomy factors are encoded?'
        else:
            z_base = z_d.clone()
            z_fixed = z_c.clone()
            z_dim = z_d.shape[1]
            title = 'Disease Latent (z_d) Traversal - What disease factors are encoded?'

        n_dims = min(n_dims, z_dim)
        traversal_range = torch.linspace(-range_std, range_std, n_steps).to(self.device)

        fig, axes = plt.subplots(n_dims, n_steps, figsize=figsize)

        for dim in range(n_dims):
            for step_idx, delta in enumerate(traversal_range):
                z_modified = z_base.clone()
                z_modified[0, dim] = delta

                with torch.no_grad():
                    if latent_type == 'common':
                        x_recon = self.model.decode(z_modified, z_fixed)
                    else:
                        x_recon = self.model.decode(z_fixed, z_modified)

                img = x_recon[0].cpu().squeeze().numpy() * 0.5 + 0.5

                ax = axes[dim, step_idx] if n_dims > 1 else axes[step_idx]
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')

                if dim == 0:
                    ax.set_title(f'{delta:.1f}σ', fontsize=9)
                if step_idx == 0:
                    ax.set_ylabel(f'Dim {dim}', fontsize=9, rotation=0,
                                 labelpad=25, va='center')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def latent_traversal_grid(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        dim1: int = 0,
        dim2: int = 1,
        latent_type: str = 'common',
        n_steps: int = 7,
        range_std: float = 2.5,
        figsize: Tuple[int, int] = (10, 10)
    ) -> plt.Figure:
        """
        2D traversal grid varying two latent dimensions simultaneously.

        Reveals interactions between dimensions and the manifold structure.
        """
        self.model.eval()
        x = x.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            z_c, z_d, _ = self.model.encode(x, label)

        if latent_type == 'common':
            z_base = z_c.clone()
            z_fixed = z_d.clone()
        else:
            z_base = z_d.clone()
            z_fixed = z_c.clone()

        traversal_range = torch.linspace(-range_std, range_std, n_steps).to(self.device)

        fig, axes = plt.subplots(n_steps, n_steps, figsize=figsize)

        for i, val1 in enumerate(traversal_range):
            for j, val2 in enumerate(traversal_range):
                z_modified = z_base.clone()
                z_modified[0, dim1] = val1
                z_modified[0, dim2] = val2

                with torch.no_grad():
                    if latent_type == 'common':
                        x_recon = self.model.decode(z_modified, z_fixed)
                    else:
                        x_recon = self.model.decode(z_fixed, z_modified)

                img = x_recon[0].cpu().squeeze().numpy() * 0.5 + 0.5
                axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')

        plt.suptitle(f'{latent_type.title()} Latent 2D Grid: Dim {dim1} (rows) × Dim {dim2} (cols)',
                    fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 3. LATENT ABLATION STUDIES
    # -------------------------------------------------------------------------

    def ablation_study(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Ablation study: What happens when we zero out different parts of z?

        Shows:
        - Full reconstruction (z_c + z_d)
        - z_c only (z_d = 0): What anatomy/common features look like
        - z_d only (z_c = 0): What disease-specific features look like
        - Individual dimension ablations
        """
        self.model.eval()
        x = x.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            z_c, z_d, _ = self.model.encode(x, label)

            # Full reconstruction
            x_full = self.model.decode(z_c, z_d)

            # z_c only (zero z_d)
            z_d_zero = torch.zeros_like(z_d)
            x_c_only = self.model.decode(z_c, z_d_zero)

            # z_d only (zero z_c)
            z_c_zero = torch.zeros_like(z_c)
            x_d_only = self.model.decode(z_c_zero, z_d)

            # Swap: use z_c from normal prior
            z_c_prior = torch.randn_like(z_c)
            x_prior_c = self.model.decode(z_c_prior, z_d)

            # Random z_d
            z_d_prior = torch.randn_like(z_d)
            x_prior_d = self.model.decode(z_c, z_d_prior)

        fig, axes = plt.subplots(2, 4, figsize=figsize)

        images = [
            (x, 'Original'),
            (x_full, 'Full Recon\n(z_c + z_d)'),
            (x_c_only, 'z_c Only\n(z_d = 0)'),
            (x_d_only, 'z_d Only\n(z_c = 0)'),
            (x_prior_c, 'Random z_c\n(keep z_d)'),
            (x_prior_d, 'Random z_d\n(keep z_c)'),
        ]

        for idx, (img_tensor, title) in enumerate(images):
            row, col = idx // 4, idx % 4
            if idx >= 4:
                row, col = 1, idx - 4

            img = img_tensor[0].cpu().squeeze().numpy() * 0.5 + 0.5
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        # Hide unused subplots
        for idx in range(len(images), 8):
            row, col = idx // 4, idx % 4
            axes[row, col].axis('off')

        # Add latent value annotations
        z_c_np = z_c[0].cpu().numpy()
        z_d_np = z_d[0].cpu().numpy()

        info_text = f'z_c stats: μ={z_c_np.mean():.2f}, σ={z_c_np.std():.2f}, |z|={np.linalg.norm(z_c_np):.2f}\n'
        info_text += f'z_d stats: μ={z_d_np.mean():.2f}, σ={z_d_np.std():.2f}, |z|={np.linalg.norm(z_d_np):.2f}'

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.suptitle('Latent Ablation Study: What information is in z_c vs z_d?', fontsize=14)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        return fig

    def dimension_importance(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        latent_type: str = 'common',
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Measure importance of each latent dimension by reconstruction error
        when that dimension is zeroed out.

        Dimensions with high importance carry more information.
        """
        self.model.eval()
        x = x.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            z_c, z_d, _ = self.model.encode(x, label)
            x_full = self.model.decode(z_c, z_d)
            base_error = F.mse_loss(x_full, x).item()

            if latent_type == 'common':
                z_target = z_c
                z_other = z_d
                decode_fn = lambda z_t: self.model.decode(z_t, z_other)
            else:
                z_target = z_d
                z_other = z_c
                decode_fn = lambda z_t: self.model.decode(z_other, z_t)

            n_dims = z_target.shape[1]
            importance = []

            for dim in range(n_dims):
                z_ablated = z_target.clone()
                z_ablated[0, dim] = 0
                x_ablated = decode_fn(z_ablated)
                error = F.mse_loss(x_ablated, x).item()
                importance.append(error - base_error)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar chart of importance
        colors = plt.cm.RdYlGn_r(np.array(importance) / (max(importance) + 1e-8))
        axes[0].bar(range(n_dims), importance, color=colors)
        axes[0].set_xlabel('Latent Dimension')
        axes[0].set_ylabel('Δ MSE (ablation effect)')
        axes[0].set_title(f'{latent_type.title()} Dimension Importance')
        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Show actual latent values
        z_vals = z_target[0].cpu().numpy()
        axes[1].bar(range(n_dims), z_vals, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Latent Dimension')
        axes[1].set_ylabel('Encoded Value')
        axes[1].set_title(f'{latent_type.title()} Latent Values')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.suptitle(f'Dimension Analysis for {latent_type.title()} Latent', fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 4. GRADIENT-BASED ATTRIBUTION
    # -------------------------------------------------------------------------

    def latent_saliency_map(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        latent_type: str = 'common',
        dim: int = 0,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Compute saliency map: which input pixels influence a specific latent dimension?

        Uses gradient of latent w.r.t. input to show pixel importance.
        """
        self.model.eval()
        x = x.to(self.device).requires_grad_(True)
        label = label.to(self.device)

        z_c, z_d, _ = self.model.encode(x, label)

        if latent_type == 'common':
            target = z_c[0, dim]
        else:
            target = z_d[0, dim]

        target.backward()

        saliency = x.grad.abs()[0].cpu().squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        img = x[0].detach().cpu().squeeze().numpy() * 0.5 + 0.5
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Saliency map
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title(f'Saliency for {latent_type}[{dim}]')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(img, cmap='gray')
        axes[2].imshow(saliency, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(f'Which pixels affect {latent_type} dimension {dim}?', fontsize=14)
        plt.tight_layout()
        return fig

    def full_saliency_analysis(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        n_dims: int = 8,
        figsize: Tuple[int, int] = (16, 8)
    ) -> plt.Figure:
        """
        Show saliency maps for multiple latent dimensions side by side.
        """
        self.model.eval()

        z_c_dim = self.model.cfg.z_dim_common
        z_d_dim = self.model.cfg.z_dim_disease

        n_dims_c = min(n_dims, z_c_dim)
        n_dims_d = min(n_dims, z_d_dim)

        fig, axes = plt.subplots(2, max(n_dims_c, n_dims_d) + 1, figsize=figsize)

        # Show input
        img = x[0].cpu().squeeze().numpy() * 0.5 + 0.5
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Input')
        axes[0, 0].axis('off')
        axes[1, 0].imshow(img, cmap='gray')
        axes[1, 0].axis('off')

        # Common latent saliencies
        for dim in range(n_dims_c):
            x_input = x.clone().to(self.device).requires_grad_(True)
            label_input = label.to(self.device)

            z_c, z_d, _ = self.model.encode(x_input, label_input)
            z_c[0, dim].backward()

            saliency = x_input.grad.abs()[0].cpu().squeeze().numpy()
            axes[0, dim + 1].imshow(saliency, cmap='hot')
            axes[0, dim + 1].set_title(f'z_c[{dim}]', fontsize=9)
            axes[0, dim + 1].axis('off')

        # Disease latent saliencies
        for dim in range(n_dims_d):
            x_input = x.clone().to(self.device).requires_grad_(True)
            label_input = label.to(self.device)

            z_c, z_d, _ = self.model.encode(x_input, label_input)
            z_d[0, dim].backward()

            saliency = x_input.grad.abs()[0].cpu().squeeze().numpy()
            axes[1, dim + 1].imshow(saliency, cmap='hot')
            axes[1, dim + 1].set_title(f'z_d[{dim}]', fontsize=9)
            axes[1, dim + 1].axis('off')

        # Hide unused
        for i in range(n_dims_c + 1, axes.shape[1]):
            axes[0, i].axis('off')
        for i in range(n_dims_d + 1, axes.shape[1]):
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel('Common (z_c)', fontsize=11)
        axes[1, 0].set_ylabel('Disease (z_d)', fontsize=11)

        plt.suptitle('Saliency Analysis: Which pixels affect each latent dimension?', fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 5. FILTER VISUALIZATION
    # -------------------------------------------------------------------------

    def visualize_conv_filters(
        self,
        layer_name: str = 'encoder.conv.0',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Visualize the learned convolutional filters (weights).

        First layer filters are directly interpretable as edge/texture detectors.
        """
        # Find the layer
        module = self.model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            print(f"Layer {layer_name} is not a Conv layer")
            return None

        weights = module.weight.detach().cpu().numpy()
        n_filters = weights.shape[0]
        n_channels = weights.shape[1]

        # For first layer (single input channel), show all filters
        if n_channels == 1:
            n_rows = int(np.ceil(np.sqrt(n_filters)))
            n_cols = int(np.ceil(n_filters / n_rows))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()

            for i in range(n_filters):
                axes[i].imshow(weights[i, 0], cmap='RdBu_r',
                              vmin=-weights.max(), vmax=weights.max())
                axes[i].axis('off')
                axes[i].set_title(f'F{i}', fontsize=8)

            for i in range(n_filters, len(axes)):
                axes[i].axis('off')
        else:
            # For deeper layers, show filter norms or first few
            fig, axes = plt.subplots(2, min(16, n_filters), figsize=figsize)

            for i in range(min(16, n_filters)):
                # Sum across input channels
                filter_vis = np.mean(np.abs(weights[i]), axis=0)
                axes[0, i].imshow(filter_vis, cmap='viridis')
                axes[0, i].axis('off')
                axes[0, i].set_title(f'F{i}', fontsize=8)

                # Show per-channel structure
                if n_channels <= 3:
                    for c in range(n_channels):
                        # Create RGB visualization
                        pass

            axes[1, 0].axis('off')

        plt.suptitle(f'Convolutional Filters: {layer_name}\nShape: {weights.shape}',
                    fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # 6. LATENT SPACE GEOMETRY
    # -------------------------------------------------------------------------

    def latent_interpolation_with_activations(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor,
        n_steps: int = 8,
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Interpolate between two samples and show how activations change.

        Reveals manifold structure and potential discontinuities.
        """
        self.model.eval()
        self.extractor.register_conv_hooks()

        x1, x2 = x1.to(self.device), x2.to(self.device)
        label1, label2 = label1.to(self.device), label2.to(self.device)

        with torch.no_grad():
            z_c1, z_d1, _ = self.model.encode(x1, label1)
            z_c2, z_d2, _ = self.model.encode(x2, label2)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, n_steps, figure=fig)

        activation_traces = []

        for i, t in enumerate(np.linspace(0, 1, n_steps)):
            z_c = (1 - t) * z_c1 + t * z_c2
            z_d = (1 - t) * z_d1 + t * z_d2

            with torch.no_grad():
                x_interp = self.model.decode(z_c, z_d)

            activations = self.extractor.get_activations()

            # Store activation statistics
            act_stats = {k: v.mean().item() for k, v in activations.items()}
            activation_traces.append(act_stats)

            # Show interpolated image
            ax = fig.add_subplot(gs[0, i])
            img = x_interp[0].cpu().squeeze().numpy() * 0.5 + 0.5
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f't={t:.2f}', fontsize=9)

        self.extractor.clear_hooks()

        # Plot activation traces
        ax_trace = fig.add_subplot(gs[1:3, :])
        for key in list(activation_traces[0].keys())[:5]:  # First 5 layers
            values = [at.get(key, 0) for at in activation_traces]
            ax_trace.plot(np.linspace(0, 1, n_steps), values, '-o', label=key[:30], markersize=4)
        ax_trace.set_xlabel('Interpolation t')
        ax_trace.set_ylabel('Mean Activation')
        ax_trace.legend(fontsize=8, loc='upper right')
        ax_trace.set_title('Activation Traces During Interpolation')
        ax_trace.grid(True, alpha=0.3)

        # Show latent distances
        ax_dist = fig.add_subplot(gs[3, :])
        z_c_dist = torch.norm(z_c2 - z_c1).item()
        z_d_dist = torch.norm(z_d2 - z_d1).item()
        ax_dist.bar(['z_c distance', 'z_d distance', 'Total'],
                   [z_c_dist, z_d_dist, z_c_dist + z_d_dist],
                   color=['blue', 'red', 'purple'])
        ax_dist.set_title('Latent Space Distances')

        plt.suptitle('Interpolation Analysis: Images and Activation Dynamics', fontsize=14)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # MASTER VISUALIZATION FUNCTION
    # -------------------------------------------------------------------------

    def visualize_all(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        save_prefix: Optional[str] = None
    ):
        """
        Run all visualizations for a single sample.

        Args:
            x: Input image [1, C, H, W]
            label: Class label [1]
            save_prefix: If provided, save figures to files
        """
        print("=" * 60)
        print("VAE INTROSPECTION REPORT")
        print("=" * 60)

        figs = {}

        # 1. Encoder activations
        print("\n[1/6] Visualizing encoder activations...")
        figs['encoder_acts'] = self.visualize_encoder_activations(x, label)
        plt.show()

        # 2. Decoder activations
        print("\n[2/6] Visualizing decoder activations...")
        figs['decoder_acts'] = self.visualize_decoder_activations(x, label)
        plt.show()

        # 3. Latent traversals
        print("\n[3/6] Latent traversals (z_c)...")
        figs['traversal_c'] = self.latent_traversal(x, label, 'common')
        plt.show()

        print("       Latent traversals (z_d)...")
        figs['traversal_d'] = self.latent_traversal(x, label, 'disease')
        plt.show()

        # 4. Ablation study
        print("\n[4/6] Ablation study...")
        figs['ablation'] = self.ablation_study(x, label)
        plt.show()

        # 5. Dimension importance
        print("\n[5/6] Dimension importance...")
        figs['importance_c'] = self.dimension_importance(x, label, 'common')
        plt.show()
        figs['importance_d'] = self.dimension_importance(x, label, 'disease')
        plt.show()

        # 6. Saliency analysis
        print("\n[6/6] Saliency analysis...")
        figs['saliency'] = self.full_saliency_analysis(x, label)
        plt.show()

        print("\n" + "=" * 60)
        print("INTROSPECTION COMPLETE")
        print("=" * 60)

        if save_prefix:
            for name, fig in figs.items():
                if fig is not None:
                    fig.savefig(f'{save_prefix}_{name}.png', dpi=150, bbox_inches='tight')
            print(f"Figures saved with prefix: {save_prefix}")

        return figs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_introspect(model, dataloader, device='cuda', n_samples=3):
    """
    Quick introspection on a few random samples.
    """
    introspector = VAEIntrospector(model, device)

    samples = []
    for x, labels in dataloader:
        for i in range(len(x)):
            if len(samples) < n_samples:
                samples.append((x[i:i+1], labels[i:i+1]))
        if len(samples) >= n_samples:
            break

    for idx, (x, label) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {idx + 1} (Label: {label.item()})")
        print('='*60)
        introspector.visualize_all(x, label)


def compare_samples_introspection(model, x1, label1, x2, label2, device='cuda'):
    """
    Compare introspection between two samples (e.g., normal vs disease).
    """
    introspector = VAEIntrospector(model, device)

    print("SAMPLE 1 ANALYSIS:")
    introspector.visualize_all(x1, label1)

    print("\nSAMPLE 2 ANALYSIS:")
    introspector.visualize_all(x2, label2)

    print("\nINTERPOLATION ANALYSIS:")
    introspector.latent_interpolation_with_activations(x1, x2, label1, label2)
    plt.show()
