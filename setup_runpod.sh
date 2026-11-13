#!/bin/bash
set -e #

echo "=========================================="
echo "Setting up superdiff-ldm environment on RunPod..."
echo "=========================================="

# CRITICAL: RunPod has disk quota limits on root filesystem
# All environments and caches must be in /workspace to avoid quota errors

# Set up workspace directories
echo "Setting up workspace directories..."
export MAMBA_ROOT_PREFIX=/workspace/micromamba
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p $MAMBA_ROOT_PREFIX $TMPDIR $PIP_CACHE_DIR

# Add to ~/.bashrc for persistence
echo "Configuring environment variables..."
grep -q "MAMBA_ROOT_PREFIX" ~/.bashrc || echo "export MAMBA_ROOT_PREFIX=/workspace/micromamba" >> ~/.bashrc
grep -q "TMPDIR=/workspace/tmp" ~/.bashrc || echo "export TMPDIR=/workspace/tmp" >> ~/.bashrc
grep -q "PIP_CACHE_DIR=/workspace/pip-cache" ~/.bashrc || echo "export PIP_CACHE_DIR=/workspace/pip-cache" >> ~/.bashrc

# Navigate to workspace
cd /workspace/superdiff-ldm

# Create environment in /workspace (NOT in home directory)
echo "Creating micromamba environment in /workspace..."
micromamba create -p /workspace/envs/superdiff python=3.11 -c conda-forge -y

# Activate environment
echo "Activating environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate /workspace/envs/superdiff

# Install CUDA dependencies
echo "Installing CUDA dependencies..."
micromamba install cuda-toolkit cudnn -c nvidia -c conda-forge -y

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install JAX with CUDA
echo "Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12]"

# Install remaining requirements
echo "Installing remaining Python packages..."
pip install -r requirements_clean.txt --upgrade

# Clean up to save space
echo "Cleaning up cache..."
micromamba clean --all -y
rm -rf $TMPDIR/*

echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

# Verify installations
echo ""
echo "JAX devices:"
python -c "import jax; print(jax.devices())"

echo ""
echo "PyTorch CUDA:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  micromamba activate superdiff"
echo ""
echo "Or add to ~/.bashrc for auto-activation:"
echo '  eval "$(micromamba shell hook --shell bash)"'
echo "  micromamba activate superdiff"
echo ""
