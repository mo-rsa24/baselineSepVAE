# RunPod Setup Instructions

This guide will help you set up the development environment on your RunPod instance using micromamba.

## ⚠️ CRITICAL: RunPod Disk Quota Issue

**RunPod has strict disk quotas on different filesystems:**
- Root filesystem (`/`): ~20GB quota - fills up quickly
- Network volume (`/workspace`): Large storage - use this for everything

**You MUST install all environments and caches in `/workspace` to avoid "Disk quota exceeded" errors.**

## Prerequisites

- RunPod instance with SSH access
- Repository already cloned in `/workspace`
- SSH key at `~/.ssh/id_ed25519`

## Step 1: Connect to RunPod Instance

```bash
ssh 4d3rlwlo6pycbb-64410b8d@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Step 2: Configure Environment Variables for /workspace

**CRITICAL FIRST STEP** - Configure all caches and environments to use /workspace:

```bash
# Set up workspace directories
export MAMBA_ROOT_PREFIX=/workspace/micromamba
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p $MAMBA_ROOT_PREFIX $TMPDIR $PIP_CACHE_DIR

# Add to ~/.bashrc for persistence
echo "export MAMBA_ROOT_PREFIX=/workspace/micromamba" >> ~/.bashrc
echo "export TMPDIR=/workspace/tmp" >> ~/.bashrc
echo "export PIP_CACHE_DIR=/workspace/pip-cache" >> ~/.bashrc

source ~/.bashrc
```

## Step 3: Install Micromamba

If micromamba is not already installed:

```bash
cd /workspace
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash
source ~/.bashrc
```

## Step 4: Navigate to the Repository

```bash
cd /workspace/superdiff-ldm
```

## Step 5: Create Micromamba Environment in /workspace

**IMPORTANT:** Use `-p` flag to create environment in `/workspace`, NOT `-n` flag:

```bash
micromamba create -p /workspace/envs/superdiff python=3.11 -c conda-forge -y
```

Activate the environment:

```bash
micromamba activate /workspace/envs/superdiff
```

## Step 6: Install CUDA Dependencies

```bash
micromamba install cuda-toolkit cudnn -c nvidia -c conda-forge -y
```

## Step 7: Install PyTorch with CUDA Support

```bash
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

For different CUDA versions, check https://pytorch.org/get-started/locally/

## Step 8: Install JAX with CUDA Support

```bash
pip install --upgrade "jax[cuda12]"
```

## Step 9: Install Remaining Requirements

```bash
pip install -r requirements_clean.txt --upgrade
```

## Step 10: Verify Installation

Verify that JAX can see the GPU:

```bash
python -c "import jax; print('JAX devices:', jax.devices())"
```

Verify PyTorch GPU support:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

## Step 11: Configure Weights & Biases (Optional)

If you plan to use wandb for experiment tracking:

```bash
wandb login
```

Enter your API key when prompted.

## Quick Start Script

Use the provided `setup_runpod.sh` script which handles all disk quota issues:

```bash
cd /workspace/superdiff-ldm
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Troubleshooting

### Disk Quota Exceeded Error

**This is the most common RunPod issue.** If you see "Disk quota exceeded" or "No space left on device":

1. Check which filesystem is full:
   ```bash
   df -h
   ```

2. Clean up existing installations in home directory:
   ```bash
   # Remove old conda/mamba environments
   rm -rf ~/micromamba
   rm -rf ~/.conda
   rm -rf ~/.cache/pip

   # Clean up temp files
   rm -rf /tmp/*
   ```

3. Ensure environment variables point to /workspace:
   ```bash
   export MAMBA_ROOT_PREFIX=/workspace/micromamba
   export TMPDIR=/workspace/tmp
   export PIP_CACHE_DIR=/workspace/pip-cache
   ```

4. Recreate environment in /workspace:
   ```bash
   micromamba create -p /workspace/envs/superdiff python=3.11 -c conda-forge -y
   ```

### CUDA Version Mismatch

Check your RunPod GPU's CUDA version:

```bash
nvidia-smi
```

Then install matching PyTorch and JAX versions.

### Import Errors

Ensure you've activated the environment:

```bash
micromamba activate /workspace/envs/superdiff
```

### JAX Not Detecting GPU

Set these environment variables:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Add to `~/.bashrc` for persistence.

## Running Your Code

Once setup is complete, activate the environment and run your scripts:

```bash
micromamba activate /workspace/envs/superdiff
cd /workspace/superdiff-ldm
python your_script.py
```

## Persistent Setup

To automatically activate the environment on login, add to `~/.bashrc`:

```bash
# Environment variables
export MAMBA_ROOT_PREFIX=/workspace/micromamba
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache

# Auto-activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate /workspace/envs/superdiff
cd /workspace/superdiff-ldm
```

## Immediate Fix for Current Disk Quota Issue

If you're already experiencing the disk quota error:

```bash
# 1. Remove old installations from home directory
rm -rf ~/micromamba ~/.conda ~/.cache/pip

# 2. Set environment variables
export MAMBA_ROOT_PREFIX=/workspace/micromamba
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p $MAMBA_ROOT_PREFIX $TMPDIR $PIP_CACHE_DIR

# 3. Remove partial installation if it exists
micromamba env remove -n superdiff -y 2>/dev/null || true

# 4. Run the setup script
cd /workspace/superdiff-ldm
./setup_runpod.sh
```
