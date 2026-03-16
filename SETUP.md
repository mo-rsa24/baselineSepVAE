# RunPod Setup Guide — baselineSepVAE
**Binary-first**: Normal (class 14) vs. Cardiomegaly (class 3).
Pleural Thickening (class 11) is added in C5 once the binary model is validated.

Binary pair length: **min(normal, cardio) → 2 300 pairs**
At batch_size=8 (4 normal + 4 cardio per batch): **~575 steps/epoch**

---

## PHASE A — On the cluster (run once before touching RunPod)

### A1 — Pre-cache DICOMs to .npy  *(161 GB → ~7.5 GB, 40× reduction)*

```bash
cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE

python scripts/cache_dicoms.py \
    --dicom_dir  /datasets/mmolefe/vinbigdata/train \
    --csv_path   /datasets/mmolefe/vinbigdata/train.csv \
    --output_dir /datasets/mmolefe/vinbigdata/cache_npy \
    --img_size   512 \
    --num_workers 16
```

Verify:
```bash
ls /datasets/mmolefe/vinbigdata/cache_npy/
# Expected: images/  train_filtered.csv  cache_manifest.json

cat /datasets/mmolefe/vinbigdata/cache_manifest.json
# Confirm: total_images ~14900+, errors 0 or near-zero

ls /datasets/mmolefe/vinbigdata/cache_npy/images/ | wc -l
# Expect ~14 900 files (all three classes: 14, 3, 11)

du -sh /datasets/mmolefe/vinbigdata/cache_npy/
# Expect ~7–8 GB
```

The output `train_filtered.csv` has columns:
`image_id, class_name, class_id, x_min_norm, y_min_norm, x_max_norm, y_max_norm`
(bbox coordinates pre-normalised to [0, 1] — dataloader reads these directly).

---

### A2 — Initialise git and push to GitHub

```bash
cd /home-mscluster/mmolefe/Playground/PhD/baselineSepVAE

git init

git add \
    datasets/VinBigData.py \
    datasets/__init__.py \
    losses/sep_vae_losses.py \
    losses/lpips_gan.py \
    losses/__init__.py \
    models/ae_kl.py \
    models/sep_vae_jax.py \
    models/resnet_jax.py \
    models/__init__.py \
    run/train_sep_vae.py \
    run/__init__.py \
    scripts/cache_dicoms.py \
    slurm_scripts/sep_vae_biggpu.slurm \
    slurm_scripts/sep_vae.slurm \
    utils/sepvae_analysis.py \
    utils/sepvae_diagnostics.py \
    utils/weight_converter.py \
    utils/__init__.py \
    requirements.txt \
    .gitignore \
    SETUP.md

# Confirm nothing large is staged (should be only .py / .md / .txt)
git status
git diff --cached --stat

git commit -m "Initial commit: baselineSepVAE — Cardiomegaly + Pleural Thickening, JAX/Flax, npy cache support"

# Create repo on GitHub first (no README, no gitignore), then:
git remote add origin https://github.com/<your-username>/baselineSepVAE.git
git branch -M main
git push -u origin main
```

---

### A3 — Transfer data to the RunPod network volume

Attach the **network volume** to a cheap CPU-only temporary pod first.
Note the SSH host/port from the pod details page.

```bash
RUNPOD_IP="<pod-public-ip>"
RUNPOD_PORT="<pod-ssh-port>"
DEST="/workspace"          # adjust if your volume mounts elsewhere

# 1. Pre-cached dataset (~7–8 GB) — the only transfer needed; raw DICOMs stay on cluster
rsync -avz --progress --no-owner --no-group \
  /datasets/mmolefe/vinbigdata/cache_npy/ \
  runpod-tcp:${DEST}/vinbigdata/cache_npy/

# 2. CheSS weights (~353 MB)
rsync -avz --progress --no-owner --no-group \
  /datasets/mmolefe/chess/pretrained_weights.pth.tar \
  runpod-tcp:${DEST}/chess/pretrained_weights.pth.tar
```

Verify on the pod:
```bash
ls ${DEST}/vinbigdata/cache_npy/
# images/  train_filtered.csv  cache_manifest.json

ls ${DEST}/vinbigdata/cache_npy/images/ | wc -l  # expect ~14 900
du -sh ${DEST}/vinbigdata/cache_npy/              # ~7–8 GB
ls ${DEST}/chess/                                  # pretrained_weights.pth.tar
```

---

## PHASE B — On the RunPod A100 pod

All commands below are run from the **pod shell**.
For speed, install Miniconda on pod-local storage (`/root/miniconda3`), which is
ephemeral and can be deleted/reinstalled at any time.
Keep data, repo, and checkpoints on `/workspace` if you want them to persist.

SSH into the pod:
```bash
ssh -p <port> root@<pod-ip>
```

### B1 — Check CUDA (determines JAX wheel)

```bash
nvidia-smi            # note Driver and CUDA versions
nvcc --version        # confirm CUDA 12.x
# A100 SXM pods ship CUDA 12.1–12.4 — the jax[cuda12] wheel covers all of these
```

### B2 — Fast Miniconda reinstall (pod-local, non-persistent)

```bash
MINICONDA_DIR="/root/miniconda3"
INSTALLER="/tmp/miniconda.sh"

# Optional hard reset: delete old install if it exists
[ -d "$MINICONDA_DIR" ] && rm -rf "$MINICONDA_DIR"

# Fresh download with retries
wget -q --show-progress --tries=5 --timeout=30 \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O "$INSTALLER"

# Install and activate
bash "$INSTALLER" -b -p "$MINICONDA_DIR"
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
conda init bash
source ~/.bashrc

# Verify
which conda
conda --version
```

### B3 — Create conda environment

```bash
# If reinstalling, remove old env name first (safe if missing)
conda env remove -n jaxstack -y 2>/dev/null || true
conda create -n jaxstack python=3.11 -y
conda activate jaxstack
```

### B4 — Install JAX with CUDA 12  *(GPU-accelerated — install FIRST)*

```bash
# Bundles cuDNN + NCCL; no separate CUDA toolkit install needed
pip install -U "jax[cuda12]"

# Verify GPU is visible
python -c "
import jax
print('devices:', jax.devices())
print('backend:', jax.default_backend())
# Must print: [CudaDevice(id=0)]  cuda
"
```

If the backend prints `cpu`, the CUDA wheel did not link. Most common cause:
CUDA driver < 525 or toolkit mismatch. Re-install with explicit index:
```bash
pip install -U "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### B5 — Install PyTorch CPU-only  *(DataLoader only — keeps VRAM for JAX)*

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### B6 — Clone code and install remaining requirements

```bash
cd /workspace
git clone https://github.com/<your-username>/baselineSepVAE.git
cd baselineSepVAE

pip install -r requirements.txt
```

### B7 — Verify the full stack

```bash
python -c "
import jax, jax.numpy as jnp
import flax, optax
import torch

x = jnp.ones((4, 512, 512, 1))
print('JAX backend   :', jax.default_backend())       # cuda
print('JAX device    :', x.devices())
print('Flax          :', flax.__version__)
print('Optax         :', optax.__version__)
print('PyTorch CUDA  :', torch.cuda.is_available())   # False (CPU wheel — expected)
print('OK')
"
```

### B8 — Configure W&B

```bash
conda activate jaxstack

# Interactive login (stores key in ~/.netrc)
wandb login    # paste API key from wandb.ai/authorize

# Verify
python -c "import wandb; print('wandb', wandb.__version__)"
```

### B9 — Set persistent environment variables

```bash
cat >> ~/.bashrc << 'EOF'

# ── JAX / XLA ───────────────────────────────────────────────────────────────
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # allocate on demand, not upfront
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90   # cap JAX at 90% of 80 GB

# ── TensorFlow (pulled in by some JAX deps; keep it quiet) ──────────────────
export TF_FORCE_GPU_ALLOW_GROWTH=true        # stop TF from stealing all VRAM
export TF_CPP_MIN_LOG_LEVEL=3               # suppress TF C++ logs
export CUDA_VISIBLE_DEVICES=0

# ── W&B ─────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="<your-key>"
export WANDB_CONSOLE=off                     # no interactive prompt in scripts

# ── Python path ─────────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/baselineSepVAE:${PYTHONPATH}"
EOF

source ~/.bashrc
```

---

## PHASE C — Experimental ladder (progressive loss activation)

Research objectives:
- **O1 — Orthogonal latent separation**: z_common ⊥ z_cardio; the cardio head responds only to cardiomegaly
- **O2 — Crisp VAE reconstructions**: visually sharp CXR output at 512×512

Architecture constants (always active in all phases):
- Hard-zero head nulling: `z_cardio_decode = 0` for Normal images (baked into `apply_head_nulling`)
- Tight KL prior for inactive heads: `sigma_inactive=0.1`
- Branched CheSS Y-network: frozen trunk (layers 1–3) + two learnable layer4 branches

Each phase adds exactly one new component. A phase **must pass its gate** before activating the next.
All phases use the npy cache (`--use_cache`) on the A100 pod.
Phases C0–C4 use the **binary** pair dataset (Normal + Cardiomegaly only).
Phase C5 adds Pleural Thickening to complete the three-class model.

---

### Summary table

| Phase | Losses active | New component | Gate to proceed |
|---|---|---|---|
| C0 | defaults | Smoke test (binary) | No OOM; KL printed; recon grid saved |
| C1 | MSE + KL only | Reconstruction baseline — MI and bbox off | KL < 5k by epoch 5; CXR visible in recon grid |
| C2 | C1 + FactorVAE MI discriminator | Latent independence (O1) | MI loss trending down; z norms diverge by class |
| C3 | C2 + bbox attention supervision | Anatomical grounding (O1) | Attn maps localise to cardiac silhouette |
| C4 | C3 + perceptual (CheSS L1) | Reconstruction sharpness (O2) | Visually sharper recons; KL/MI metrics stable |
| C5 | C4 + Pleural Thickening | Three-class extension (O1 complete) | Both disease heads independently respond to their class |

---

### C0 — Smoke test (2 epochs)

**Purpose:** Confirm no OOM, forward pass runs end-to-end, W&B connects. All loss weights at defaults.

```bash
cd /workspace/baselineSepVAE
conda activate jaxstack

python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       4 \
    --epochs           2 \
    --num_workers      4 \
    --lr_disc          1e-4 \
    --sample_every     1 \
    --manifold_every   -1 \
    --exp_name         c0_smoke \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate:** No OOM. KL and MI loss printed at step 0. Recon grid saved. W&B run appears.
**Kill if:** KL > 100 000 at step 0 → backbone issue, see Note 1.

---

### C1 — Reconstruction baseline (O2 foundation, 20 epochs)

**Losses:** MSE + KL_common + KL_disease (tight prior). MI discriminator and bbox off.
**Purpose:** Confirm the encoder/decoder can reconstruct CXR with a well-behaved posterior.
Hard-zero nulling is always active (baked in). KL warmup prevents posterior collapse.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       8 \
    --epochs           20 \
    --num_workers      8 \
    --kl_warmup_epochs 5 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     5e-5 \
    --weight_mi_factor      0.0 \
    --weight_bbox_attn      0.0 \
    --weight_perceptual     0.0 \
    --lr_disc          1e-4 \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         c1_recon_baseline \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O2 check):** KL_total < 5 000 by epoch 5; recon grid shows recognisable lung fields.
**Kill if:** KL diverges upward after warmup; or recon grid is uniform grey at epoch 10.

---

### C2 — FactorVAE MI discriminator (O1, 30 epochs)

**New component:** FactorVAE mutual-information discriminator. A small MLP (32→64→64→1)
is trained to distinguish joint samples `(z_common, z_cardio)` from permuted-marginal samples.
The VAE is penalised when the discriminator can tell them apart — pushing the two latents
toward statistical independence. This is an adversarial O1 constraint.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       8 \
    --epochs           30 \
    --num_workers      8 \
    --kl_warmup_epochs 0 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     1e-4 \
    --weight_mi_factor      1.0 \
    --weight_bbox_attn      0.0 \
    --weight_perceptual     0.0 \
    --lr_disc          1e-4 \
    --resume           /workspace/runs_sepvae/c1_recon_baseline/checkpoint_final.pkl \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         c2_mi_disc \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O1 step 1):** MI discriminator accuracy drops toward 0.5 (chance); manifold shows
z_common and z_cardio clusters separate by class. ‖z_cardio‖ larger on cardio than normal.
**Kill if:** MI loss diverges or reconstruction degrades sharply (MSE > 2× C1).

---

### C3 — Anatomical grounding via bbox attention (O1 step 2, 30 epochs)

**New component:** Bbox attention loss. The cardio `DiseaseAttentionHead` is supervised to
concentrate its learned attention map inside the ground-truth bounding box. This forces the
cardio head to attend the cardiac silhouette rather than drifting to background regions.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       8 \
    --epochs           30 \
    --num_workers      8 \
    --kl_warmup_epochs 0 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     1e-4 \
    --weight_mi_factor      1.0 \
    --weight_bbox_attn      0.2 \
    --weight_perceptual     0.0 \
    --lr_disc          1e-4 \
    --resume           /workspace/runs_sepvae/c2_mi_disc/checkpoint_final.pkl \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         c3_bbox_attn \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O1 step 2):** Attention maps (logged to W&B) show the cardio head concentrated
on the cardiac silhouette for Cardiomegaly images; diffuse/small for Normal images.
**Kill if:** Attn maps remain diffuse/uniform after 15 epochs with bbox loss active.

---

### C4 — Perceptual sharpening (O2 completion, 20 epochs)

**New component:** CheSS backbone perceptual loss (frozen CheSS feature L1). No new parameters —
the backbone is already loaded. Sharpens reconstruction texture without adding compute cost.
O2 is complete when recons are visually sharp and the O1 metrics from C3 are maintained.

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       8 \
    --epochs           20 \
    --num_workers      8 \
    --kl_warmup_epochs 0 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     1e-4 \
    --weight_mi_factor      1.0 \
    --weight_bbox_attn      0.2 \
    --weight_perceptual     0.05 \
    --lr_disc          1e-4 \
    --resume           /workspace/runs_sepvae/c3_bbox_attn/checkpoint_final.pkl \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         c4_perceptual \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O2 complete):** Recon grid visually sharper than C3 (finer rib/vessel detail);
KL, MI, and attn metrics do not regress more than 10% from C3 values.
**Kill if:** KL_common spikes > 2× C3 value (perceptual loss fighting the prior).

---

### C5 — Three-class extension: add Pleural Thickening (O1 complete)

**New component:** Switch to `VinBigDataTripletDataset` (Normal + Cardiomegaly + Pleural Thickening).
Add `pt_branch` and `head_plthick` to the encoder. Update MI pair from `(z_common, z_cardio)`
to `(z_cardio, z_plthick)` — the two disease heads must now be independent of each other.

**Code changes required before running C5:**
1. Add `pt_branch = ResNetLayer4Branch()` and `head_plthick = DiseaseAttentionHead(16ch)` to `SepVAEEncoder`
2. Update `apply_head_nulling` to hard-zero `z_cardio` for Normal/PT and `z_pt` for Normal/Cardio
3. Switch dataset: `--dataset triplet` (or update train script to use `VinBigDataTripletDataset`)
4. MI discriminator input: `jnp.concatenate([z_ca_pooled, z_pt_pooled], axis=-1)` (32D)

```bash
python run/train_sep_vae.py \
    --dicom_dir        /workspace/vinbigdata/cache_npy \
    --csv_path         /workspace/vinbigdata/cache_npy/train_filtered.csv \
    --use_cache \
    --chess_checkpoint /workspace/chess/pretrained_weights.pth.tar \
    --batch_size       8 \
    --epochs           30 \
    --num_workers      8 \
    --kl_warmup_epochs 0 \
    --weight_rec            1.0 \
    --weight_kl_common      1e-4 \
    --weight_kl_disease     1e-4 \
    --weight_mi_factor      1.0 \
    --weight_bbox_attn      0.2 \
    --weight_perceptual     0.05 \
    --lr_disc          1e-4 \
    --resume           /workspace/runs_sepvae/c4_perceptual/checkpoint_final.pkl \
    --sample_every     5 \
    --manifold_every   10 \
    --exp_name         c5_three_class \
    --output_root      /workspace/runs_sepvae \
    --wandb --wandb_project baseline-sepvae
```

**Gate (O1 complete):** Latent swap grid shows z_cardio changes cardiac silhouette without
affecting pleural region, and z_plthick changes pleural surface without affecting cardiac region.
MI discriminator accuracy ≤ 0.55 for (z_cardio, z_plthick) pair.
**Kill if:** Either disease head collapses (near-zero norms across all classes).

---

## Notes

### Note 1 — KL catastrophically high (> 100k at step 0)
The frozen CheSS backbone produces discriminative features incompatible with N(0, I).
Fix: partially unfreeze the backbone with differential LR:
- In `models/sep_vae_jax.py`, remove `jax.lax.stop_gradient(h)` from the backbone forward pass.
- Use backbone LR = 1e-5, encoder-head LR = 1e-4 (split param groups in optimizer).

### Note 2 — Scaling batch size on the A100 80 GB
| batch_size | VRAM est. (512×512) | feasible? |
|---|---|---|
| 4  | ~8 GB  | yes |
| 8  | ~15 GB | yes |
| 16 | ~28 GB | yes |
| 32 | ~54 GB | yes (borderline with perceptual loss) |

Start at 8 after smoke test passes; push higher if VRAM headroom allows.

### Note 3 — Preserving checkpoints before stopping the pod
Pod-local storage (`/workspace/runs_sepvae/`) is lost when the pod is stopped.
Before stopping, copy to the network volume:
```bash
cp /workspace/runs_sepvae/<exp>/checkpoint_final.pkl /workspace/<volume-mount>/checkpoints/
```
W&B preserves all metrics and sample images automatically.

### Note 4 — Dataset class IDs (reference)
| class_id | name | unique images |
|---|---|---|
| 14 | Normal (No finding) | 10 606 |
| 3  | Cardiomegaly        | 2 300  |
| 11 | Pleural Thickening  | 1 981  |

**C0–C4 (binary pair):** min(normal, cardio) = **2 300 pairs**.
At batch_size=8 (4 normal + 4 cardio): **~575 steps/epoch**.

**C5 (triplet):** min(normal, cardio, plthick) = **1 981 triplets**.
At batch_size=8 (batch of 3×B/3): **~742 steps/epoch**.
