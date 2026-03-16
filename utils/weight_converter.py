"""
Weight converter for CheSS ResNet-50: PyTorch → JAX/Flax

This module handles conversion of pre-trained CheSS weights from PyTorch format
to JAX/Flax format for the ResNet-50 backbone.

CheSS checkpoint structure:
- File: pretrained_weights.pth.tar (PyTorch checkpoint)
- Contains: encoder_q (query encoder), encoder_k (key encoder), queue
- We only use encoder_q weights (322 parameter keys)

Conversion details:
- Conv2d: (out_ch, in_ch, H, W) → (H, W, in_ch, out_ch)
- BatchNorm: {weight, bias, running_mean, running_var} → {scale, bias, mean, var}
"""

import os
from typing import Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze


def convert_conv_weights(torch_weight: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch Conv2d weights to JAX format.

    PyTorch: (out_channels, in_channels, kernel_H, kernel_W)
    JAX/Flax: (kernel_H, kernel_W, in_channels, out_channels)

    Args:
        torch_weight: PyTorch conv weight array

    Returns:
        JAX-compatible weight array
    """
    # Transpose: (O, I, H, W) → (H, W, I, O)
    return np.transpose(torch_weight, (2, 3, 1, 0))


def convert_bn_params(torch_bn_dict: Dict[str, np.ndarray]) -> tuple:
    """
    Convert PyTorch BatchNorm2d parameters to JAX format.

    PyTorch BatchNorm has:
        - weight (scale)       → Flax params collection
        - bias                 → Flax params collection
        - running_mean         → Flax batch_stats collection
        - running_var          → Flax batch_stats collection

    Flax nn.BatchNorm stores these in TWO separate collections:
        params:      {scale, bias}
        batch_stats: {mean, var}

    Args:
        torch_bn_dict: Dict with PyTorch BatchNorm parameters

    Returns:
        (params_dict, batch_stats_dict) tuple
    """
    params = {
        'scale': torch_bn_dict['weight'],
        'bias': torch_bn_dict['bias'],
    }
    batch_stats = {
        'mean': torch_bn_dict['running_mean'],
        'var': torch_bn_dict['running_var'],
    }
    return params, batch_stats


def convert_chess_resnet50(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Load CheSS PyTorch checkpoint and convert encoder_q to JAX/Flax format.

    This function:
    1. Loads the PyTorch checkpoint
    2. Extracts encoder_q weights (only the query encoder, not encoder_k)
    3. Converts layer-by-layer to JAX format
    4. Returns a nested dict compatible with ResNet50CheSS Flax model

    Args:
        checkpoint_path: Path to pretrained_weights.pth.tar
        verbose: Whether to print conversion progress

    Returns:
        jax_params: Nested dict of parameters for ResNet50CheSS

    Example:
        >>> jax_params = convert_chess_resnet50('/datasets/mmolefe/chess/pretrained_weights.pth.tar')
        >>> # Use with Flax model:
        >>> variables = {'params': jax_params, 'batch_stats': batch_stats}
        >>> y = model.apply(variables, x)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CheSS checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"Loading CheSS checkpoint from: {checkpoint_path}")

    # Load PyTorch checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    if verbose:
        print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
        print(f"Architecture: {ckpt.get('arch', 'unknown')}")
        print(f"Total keys in checkpoint: {len(state_dict)}")

    # Extract encoder_q weights (remove 'module.encoder_q.' prefix)
    # Following original CheSS loading: retain encoder_q but exclude the fc projection head
    encoder_q_weights = {}
    for key in list(state_dict.keys()):
        if key.startswith('module.encoder_q.') and not key.startswith('module.encoder_q.fc'):
            new_key = key[len('module.encoder_q.'):]
            encoder_q_weights[new_key] = state_dict[key].cpu().numpy()

    if verbose:
        print(f"Extracted {len(encoder_q_weights)} encoder_q parameters")

    # Convert to JAX format — separate params and batch_stats trees
    jax_params = {}
    jax_batch_stats = {}

    # ===== Initial conv + bn =====
    if verbose:
        print("\nConverting initial conv1 + bn1...")

    jax_params['conv1'] = {
        'kernel': convert_conv_weights(encoder_q_weights['conv1.weight'])
    }

    bn1_params, bn1_stats = convert_bn_params({
        'weight': encoder_q_weights['bn1.weight'],
        'bias': encoder_q_weights['bn1.bias'],
        'running_mean': encoder_q_weights['bn1.running_mean'],
        'running_var': encoder_q_weights['bn1.running_var'],
    })
    jax_params['bn1'] = bn1_params
    jax_batch_stats['bn1'] = bn1_stats

    # ===== ResNet layers 1-4 =====
    # layer1: 3 blocks, layer2: 4 blocks, layer3: 6 blocks, layer4: 3 blocks
    layer_configs = [
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]

    for layer_name, num_blocks in layer_configs:
        if verbose:
            print(f"\nConverting {layer_name} ({num_blocks} blocks)...")

        for block_idx in range(num_blocks):
            torch_block_prefix = f'{layer_name}.{block_idx}'
            jax_block_name = f'{layer_name}_block{block_idx}'

            jax_params[jax_block_name] = {}
            jax_batch_stats[jax_block_name] = {}

            # Convert 3 convs in bottleneck (conv1, conv2, conv3)
            for conv_idx in range(1, 4):
                conv_key = f'{torch_block_prefix}.conv{conv_idx}.weight'
                bn_key_prefix = f'{torch_block_prefix}.bn{conv_idx}'

                if conv_key in encoder_q_weights:
                    jax_params[jax_block_name][f'conv{conv_idx}'] = {
                        'kernel': convert_conv_weights(encoder_q_weights[conv_key])
                    }

                    bn_p, bn_s = convert_bn_params({
                        'weight': encoder_q_weights[f'{bn_key_prefix}.weight'],
                        'bias': encoder_q_weights[f'{bn_key_prefix}.bias'],
                        'running_mean': encoder_q_weights[f'{bn_key_prefix}.running_mean'],
                        'running_var': encoder_q_weights[f'{bn_key_prefix}.running_var'],
                    })
                    jax_params[jax_block_name][f'bn{conv_idx}'] = bn_p
                    jax_batch_stats[jax_block_name][f'bn{conv_idx}'] = bn_s

            # Convert downsample (skip connection) if it exists
            downsample_conv_key = f'{torch_block_prefix}.downsample.0.weight'
            if downsample_conv_key in encoder_q_weights:
                downsample_bn_prefix = f'{torch_block_prefix}.downsample.1'

                jax_params[jax_block_name]['downsample_conv'] = {
                    'kernel': convert_conv_weights(encoder_q_weights[downsample_conv_key])
                }

                ds_bn_p, ds_bn_s = convert_bn_params({
                    'weight': encoder_q_weights[f'{downsample_bn_prefix}.weight'],
                    'bias': encoder_q_weights[f'{downsample_bn_prefix}.bias'],
                    'running_mean': encoder_q_weights[f'{downsample_bn_prefix}.running_mean'],
                    'running_var': encoder_q_weights[f'{downsample_bn_prefix}.running_var'],
                })
                jax_params[jax_block_name]['downsample_bn'] = ds_bn_p
                jax_batch_stats[jax_block_name]['downsample_bn'] = ds_bn_s

    if verbose:
        print(f"\n✓ Conversion complete! Converted {len(jax_params)} top-level modules")
        print(f"  params keys: {len(jax_params)}, batch_stats keys: {len(jax_batch_stats)}")

    return jax_params, jax_batch_stats


def save_converted_weights(chess_weights: tuple, output_path: str):
    """
    Save converted JAX parameters and batch_stats to disk.

    Args:
        chess_weights: Tuple of (params, batch_stats) from convert_chess_resnet50
        output_path: Path to save .npy file
    """
    jax_params, jax_batch_stats = chess_weights
    np.save(output_path, {'params': jax_params, 'batch_stats': jax_batch_stats},
            allow_pickle=True)
    print(f"Saved converted weights to: {output_path}")


def load_converted_weights(input_path: str) -> tuple:
    """
    Load previously converted JAX parameters.

    Args:
        input_path: Path to .npy file

    Returns:
        (params, batch_stats) tuple
    """
    data = np.load(input_path, allow_pickle=True).item()
    # Handle both old format (flat dict) and new format (params + batch_stats)
    if 'params' in data and 'batch_stats' in data:
        print(f"Loaded converted weights from: {input_path}")
        return data['params'], data['batch_stats']
    else:
        # Legacy format: everything in one dict, need to split
        print(f"Loaded legacy format from: {input_path}, splitting params/batch_stats")
        return _split_legacy_params(data)


def _split_legacy_params(flat_params: Dict[str, Any]) -> tuple:
    """Split old-format params (with BN mean/var in params) into (params, batch_stats)."""
    params = {}
    batch_stats = {}
    for key, value in flat_params.items():
        if isinstance(value, dict):
            if 'mean' in value and 'var' in value and 'scale' in value:
                # This is a BN layer in old format
                params[key] = {'scale': value['scale'], 'bias': value['bias']}
                batch_stats[key] = {'mean': value['mean'], 'var': value['var']}
            elif any(isinstance(v, dict) and 'mean' in v for v in value.values()):
                # Nested block with BN layers inside
                params[key] = {}
                batch_stats[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and 'mean' in sub_value and 'scale' in sub_value:
                        params[key][sub_key] = {'scale': sub_value['scale'], 'bias': sub_value['bias']}
                        batch_stats[key][sub_key] = {'mean': sub_value['mean'], 'var': sub_value['var']}
                    else:
                        params[key][sub_key] = sub_value
            else:
                params[key] = value
        else:
            params[key] = value
    return params, batch_stats


def verify_conversion(chess_weights: tuple, checkpoint_path: str, tolerance: float = 1e-4):
    """
    Verify weight conversion by comparing PyTorch and JAX outputs on dummy input.

    This loads both PyTorch and JAX models, runs a forward pass on the same
    input, and checks that outputs match within tolerance.

    Args:
        chess_weights: Tuple of (params, batch_stats) from convert_chess_resnet50
        checkpoint_path: Path to original PyTorch checkpoint
        tolerance: Maximum allowed difference in outputs

    Returns:
        bool: True if verification passes

    Note:
        Requires PyTorch and torchvision to be installed.
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("⚠ PyTorch not available, skipping verification")
        return False

    from models.resnet_jax import ResNet50CheSS

    jax_params, jax_batch_stats = chess_weights

    print("\n" + "="*60)
    print("VERIFYING WEIGHT CONVERSION")
    print("="*60)

    # Create dummy input
    np.random.seed(42)
    x_np = np.random.randn(1, 512, 512, 1).astype(np.float32)

    # ===== JAX forward pass =====
    print("\n1. Running JAX forward pass...")
    jax_model = ResNet50CheSS()

    variables = {'params': jax_params, 'batch_stats': jax_batch_stats}
    y_jax = jax_model.apply(variables, jnp.array(x_np))
    y_jax_np = np.array(y_jax)

    print(f"   JAX output shape: {y_jax_np.shape}")
    print(f"   JAX output range: [{y_jax_np.min():.4f}, {y_jax_np.max():.4f}]")

    # ===== PyTorch forward pass =====
    print("\n2. Running PyTorch forward pass...")

    # Load PyTorch ResNet-50 with 1-channel input
    torch_model = models.resnet50(pretrained=False)
    torch_model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Extract encoder_q and remove prefix
    encoder_q_state = {}
    for key, value in state_dict.items():
        if key.startswith('module.encoder_q.'):
            new_key = key.replace('module.encoder_q.', '')
            encoder_q_state[new_key] = value

    # Load into model (strict=False because we're missing fc layer, but that's OK)
    torch_model.load_state_dict(encoder_q_state, strict=False)
    torch_model.eval()

    # Forward pass (convert NHWC → NCHW)
    x_torch = torch.from_numpy(x_np).permute(0, 3, 1, 2)  # (1, 1, 512, 512)

    with torch.no_grad():
        # Forward through all layers except fc
        h = torch_model.conv1(x_torch)
        h = torch_model.bn1(h)
        h = torch_model.relu(h)
        h = torch_model.maxpool(h)
        h = torch_model.layer1(h)
        h = torch_model.layer2(h)
        h = torch_model.layer3(h)
        h = torch_model.layer4(h)
        h = torch.mean(h, dim=(2, 3))  # Global average pooling
        y_torch = h

    y_torch_np = y_torch.cpu().numpy()

    print(f"   PyTorch output shape: {y_torch_np.shape}")
    print(f"   PyTorch output range: [{y_torch_np.min():.4f}, {y_torch_np.max():.4f}]")

    # ===== Compare outputs =====
    print("\n3. Comparing outputs...")
    diff = np.abs(y_torch_np - y_jax_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print(f"   Tolerance: {tolerance:.2e}")

    if max_diff < tolerance:
        print("\n✓ VERIFICATION PASSED!")
        print("="*60)
        return True
    else:
        print(f"\n✗ VERIFICATION FAILED! Difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
        print("="*60)
        return False


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CheSS ResNet-50 weights from PyTorch to JAX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/datasets/mmolefe/chess/pretrained_weights.pth.tar",
        help="Path to CheSS PyTorch checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/datasets/mmolefe/chess/resnet50_jax_params.npy",
        help="Path to save converted JAX parameters"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by comparing outputs"
    )

    args = parser.parse_args()

    # Convert weights
    chess_weights = convert_chess_resnet50(args.checkpoint, verbose=True)

    # Save converted weights
    save_converted_weights(chess_weights, args.output)

    # Verify if requested
    if args.verify:
        verify_conversion(chess_weights, args.checkpoint)
