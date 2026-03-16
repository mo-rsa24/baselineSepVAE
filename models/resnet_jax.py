"""
ResNet-50 JAX implementation for frozen CheSS backbone.

This module implements a ResNet-50 architecture optimized for medical imaging:
- 1-channel grayscale input (512x512)
- Frozen BatchNorm layers (use_running_average=True)
- Global average pooling output: (B, 2048)

Architecture follows standard ResNet-50 with bottleneck blocks.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable


class BottleneckBlock(nn.Module):
    """
    ResNet-50 Bottleneck block with 3 convolutional layers.

    Architecture:
        1x1 conv (reduction) → 3x3 conv → 1x1 conv (expansion)
        Each conv is followed by BatchNorm and ReLU (except the last one)
        Skip connection with potential downsampling/channel adjustment

    Attributes:
        filters: Number of filters in the first two convolutions
        stride: Stride for the 3x3 convolution (for downsampling)
        use_projection: Whether to use 1x1 conv for skip connection
    """
    filters: int
    stride: int = 1
    use_projection: bool = False

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H', W', C') where H', W' depend on stride
        """
        residual = x

        # 1x1 conv reduction
        h = nn.Conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            name='conv1'
        )(x)
        h = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
            name='bn1'
        )(h)
        h = nn.relu(h)

        # 3x3 conv (potentially with stride for downsampling)
        h = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False,
            name='conv2'
        )(h)
        h = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
            name='bn2'
        )(h)
        h = nn.relu(h)

        # 1x1 conv expansion (4x filters)
        h = nn.Conv(
            features=self.filters * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            name='conv3'
        )(h)
        h = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
            name='bn3'
        )(h)

        # Skip connection with potential projection
        if self.use_projection or self.stride != 1:
            residual = nn.Conv(
                features=self.filters * 4,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name='downsample_conv'
            )(x)
            residual = nn.BatchNorm(
                use_running_average=True,
                momentum=0.9,
                epsilon=1e-5,
                name='downsample_bn'
            )(residual)

        # Add skip connection and apply ReLU
        out = nn.relu(h + residual)
        return out


class ResNetLayer4Branch(nn.Module):
    """
    Learnable layer4 branch — initialised from CheSS layer4 weights, fine-tuned.

    Parameter names (layer4_block{i}) match weight_converter output so CheSS
    layer4 weights can be sliced directly into this module at startup.

    Input:  (B, 32, 32, 1024)  — layer3 output from the frozen trunk
    Output: (B, 16, 16, 2048)  — same spatial shape as original ResNet layer4
    """

    @nn.compact
    def __call__(self, x):
        for i in range(3):
            x = BottleneckBlock(
                filters=512,
                stride=(2 if i == 0 else 1),
                use_projection=(i == 0),
                name=f'layer4_block{i}',
            )(x)
        return x  # (B, 16, 16, 2048)


class ResNet50CheSS(nn.Module):
    """
    ResNet-50 backbone for CheSS encoder (frozen feature extractor).

    This is a modified ResNet-50 for medical imaging:
    - 1-channel input (grayscale chest X-rays)
    - Standard ResNet-50 architecture with bottleneck blocks
    - [3, 4, 6, 3] blocks in layers 1-4
    - Output channels: [256, 512, 1024, 2048]
    - Global average pooling → (B, 2048) feature vector

    The backbone is designed to be frozen during training (all BatchNorm
    layers use running statistics, no gradient flow).

    Input shape: (B, 512, 512, 1)
    Output shape: (B, 2048)

    When stop_at_layer3=True the forward pass returns (B, 32, 32, 1024) after
    layer3 and layer4 blocks are never instantiated — allowing two separate
    ResNetLayer4Branch modules to act as learnable branched layer4 paths.
    """

    @nn.compact
    def __call__(self, x, return_spatial=False, return_multiscale=False,
                 stop_at_layer3=False):
        """
        Forward pass through ResNet-50.

        Args:
            x: Input images (B, 512, 512, 1)
            return_spatial: If True, return spatial features (B, 64, 64, 2048)
                          If False, return global pooled features (B, 2048)
            return_multiscale: If True, return dict of multi-scale features:
                              {'layer2': (B, 64, 64, 512),
                               'layer3': (B, 32, 32, 1024),
                               'layer4': (B, 16, 16, 2048)}
                              Overrides return_spatial when True.

        Returns:
            If return_multiscale=True: Dict of multi-scale features
            If return_spatial=True: Spatial features (B, 64, 64, 2048)
            If return_spatial=False: Feature vectors (B, 2048)
        """
        # Initial conv: 7x7, stride 2
        # Input: (B, 512, 512, 1) → Output: (B, 256, 256, 64)
        h = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            name='conv1'
        )(x)
        h = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
            name='bn1'
        )(h)
        h = nn.relu(h)

        # Max pool: 3x3, stride 2
        # (B, 256, 256, 64) → (B, 128, 128, 64)
        h = nn.max_pool(
            h,
            window_shape=(3, 3),
            strides=(2, 2),
            padding='SAME'
        )

        # Layer 1: 3 blocks, filters=64, output_channels=256
        # (B, 128, 128, 64) → (B, 128, 128, 256)
        for i in range(3):
            h = BottleneckBlock(
                filters=64,
                stride=1,
                use_projection=(i == 0),
                name=f'layer1_block{i}'
            )(h)

        # Layer 2: 4 blocks, filters=128, output_channels=512
        # (B, 128, 128, 256) → (B, 64, 64, 512)
        for i in range(4):
            h = BottleneckBlock(
                filters=128,
                stride=(2 if i == 0 else 1),
                use_projection=(i == 0),
                name=f'layer2_block{i}'
            )(h)
        feat_layer2 = h  # (B, 64, 64, 512)

        # Layer 3: 6 blocks, filters=256, output_channels=1024
        # (B, 64, 64, 512) → (B, 32, 32, 1024)
        for i in range(6):
            h = BottleneckBlock(
                filters=256,
                stride=(2 if i == 0 else 1),
                use_projection=(i == 0),
                name=f'layer3_block{i}'
            )(h)
        feat_layer3 = h  # (B, 32, 32, 1024)

        # Early exit for branched-layer4 architecture:
        # layer4 blocks are NOT instantiated, so they won't appear in the
        # variable tree — only trunk (layers 1-3) params are needed.
        if stop_at_layer3:
            return feat_layer3  # (B, 32, 32, 1024)

        # Layer 4: 3 blocks, filters=512, output_channels=2048
        # (B, 32, 32, 1024) → (B, 16, 16, 2048)
        for i in range(3):
            h = BottleneckBlock(
                filters=512,
                stride=(2 if i == 0 else 1),
                use_projection=(i == 0),
                name=f'layer4_block{i}'
            )(h)
        feat_layer4 = h  # (B, 16, 16, 2048)

        if return_multiscale:
            return {
                'layer2': feat_layer2,  # (B, 64, 64, 512)  - native resolution
                'layer3': feat_layer3,  # (B, 32, 32, 1024) - native resolution
                'layer4': feat_layer4,  # (B, 16, 16, 2048) - native resolution
            }
        elif return_spatial:
            # Upsample layer4 to 64×64 for LDM compatibility
            h = jax.image.resize(feat_layer4,
                                (feat_layer4.shape[0], 64, 64, feat_layer4.shape[3]),
                                method='bilinear')
            return h  # (B, 64, 64, 2048)
        else:
            # Global average pooling
            # (B, 16, 16, 2048) → (B, 2048)
            h = jnp.mean(feat_layer4, axis=(1, 2))
            return h


# Utility function for testing
def test_resnet50():
    """Quick test to verify output shape."""
    model = ResNet50CheSS()
    x = jnp.ones((2, 512, 512, 1))

    # Initialize
    key = jax.random.PRNGKey(0)
    variables = model.init(key, x)

    # Forward pass
    y = model.apply(variables, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 2048), f"Expected (2, 2048), got {y.shape}"
    print("✓ ResNet-50 output shape correct!")

    return model, variables


if __name__ == "__main__":
    test_resnet50()
