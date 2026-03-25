# baselineLDM Pipeline

This repo now contains a repo-local `baselineLDM` implementation for the Stage 2 baseline described in [SEPVAE_CONDITIONAL_LDM.md](SEPVAE_CONDITIONAL_LDM.md):

\[
p_{\psi}(z_{\text{cardio}} \mid z_{\text{common}})
\]

## Scope

- The SepVAE remains frozen in Stage 2.
- The LDM trains on `z_cardio` only.
- `z_common` is passed as a spatial condition by channel concatenation.
- Decoding is done with the frozen SepVAE decoder:

\[
\hat{x} = \mathrm{Dec}_{\theta}(z_{\text{common}}, \hat{z}_{\text{cardio}})
\]

## Files

- `scripts/preencode_sepvae_v2_latents.py`
- `datasets/paired_latents.py`
- `models/conditional_cxr_unet.py`
- `diffusion/conditional_sampling.py`
- `run/train_baseline_ldm.py`
- `run/sample_baseline_ldm.py`

## Latent Export

Export paired latents from a trained V2 SepVAE checkpoint:

```bash
python scripts/preencode_sepvae_v2_latents.py \
  --sepvae_ckpt runs_sepvae/d5_recon-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl \
  --csv_path /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
  --dicom_dir /datasets/mmolefe/vinbigdata/cache_npy \
  --output_dir runs_baseline_ldm/latents/d5_cardio_train \
  --split train \
  --use_cache \
  --latent_mode mean \
  --batch_size 8 \
  --num_workers 4
```

Manifest schema:

```json
{
  "image_id": "....",
  "split": "train",
  "label": 0,
  "z_common_path": "z_common/00000000.npy",
  "z_cardio_path": "z_cardio/00000000.npy",
  "latent_format": "NHWC",
  "z_common_shape": [16, 16, 16],
  "z_cardio_shape": [16, 16, 16],
  "sepvae_ckpt": "/abs/path/checkpoint_final.pkl"
}
```

If the CSV has no `split` column, the exporter uses a deterministic hash split:

- train: 90%
- val: 5%
- test: 5%

## Train baselineLDM

```bash
python run/train_baseline_ldm.py \
  --preencoded_latents_dir runs_baseline_ldm/latents/d5_cardio_train \
  --sepvae_ckpt runs_sepvae/d5_recon-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl \
  --output_root runs_baseline_ldm \
  --exp_name baseline_ldm_cardio \
  --epochs 200 \
  --batch_size 16 \
  --sample_every 5 \
  --sample_steps 250 \
  --use_ema
```

The trainer saves:

- checkpoints in `runs_baseline_ldm/<run>/checkpoints/`
- decoded conditional samples in `runs_baseline_ldm/<run>/samples/`
- run metadata in `run_meta.json`

### Phase-Aware Launcher

The launcher can now auto-pick the latest SepVAE checkpoint from `d0` through `d5` and export latents if they are missing:

```bash
SEPVAE_PHASE=d4 \
./launchers/single_runs/ldm/train_baseline_ldm_cardio.sh
```

Supported values:

- `SEPVAE_PHASE=d0`
- `SEPVAE_PHASE=d1`
- `SEPVAE_PHASE=d2`
- `SEPVAE_PHASE=d3`
- `SEPVAE_PHASE=d4`
- `SEPVAE_PHASE=d5`
- `SEPVAE_PHASE=auto`  (default, tries `d5 -> d4 -> d3 -> d2 -> d1 -> d0`)

If the phase checkpoint exists but the latent export does not, the launcher runs `scripts/preencode_sepvae_v2_latents.py` automatically.

### 24GB VRAM Profile

The launcher defaults to a conservative 24GB VRAM profile:

```bash
BATCH_SIZE=16
NUM_WORKERS=4
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
LDM_BASE_CH=128
LDM_CH_MULTS=1,2,4
LDM_NUM_RES_BLOCKS=2
LDM_ATTN_RES=16
USE_BFLOAT16=1
USE_REMAT=1
USE_EMA=1
EMA_DECAY=0.999
SAMPLE_BATCH_SIZE=8
SAMPLE_STEPS=250
```

Recommended adjustment order if you hit OOM on a 24GB card:

1. Reduce `SAMPLE_BATCH_SIZE` from `8` to `4`
2. Reduce `BATCH_SIZE` from `16` to `12` or `8`
3. Only then reduce model width, for example `LDM_BASE_CH=96`

## Sample or Refine

Sample from the learned conditional prior using preencoded `z_common`:

```bash
python run/sample_baseline_ldm.py \
  --ldm_ckpt runs_baseline_ldm/<run>/checkpoints/checkpoint_final.pkl \
  --sepvae_ckpt runs_sepvae/d5_recon-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl \
  --preencoded_latents_dir runs_baseline_ldm/latents/d5_cardio_train \
  --output_dir runs_baseline_ldm/samples/example_sample \
  --mode sample \
  --num_records 8
```

Refine an encoded `z_cardio` instead of sampling from noise:

```bash
python run/sample_baseline_ldm.py \
  --ldm_ckpt runs_baseline_ldm/<run>/checkpoints/checkpoint_final.pkl \
  --sepvae_ckpt runs_sepvae/d5_recon-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl \
  --preencoded_latents_dir runs_baseline_ldm/latents/d5_cardio_train \
  --output_dir runs_baseline_ldm/samples/example_refine \
  --mode refine \
  --num_records 8 \
  --refine_t_start 0.35
```

Direct image IDs can also be used as the condition source:

```bash
python run/sample_baseline_ldm.py \
  --ldm_ckpt runs_baseline_ldm/<run>/checkpoints/checkpoint_final.pkl \
  --sepvae_ckpt runs_sepvae/d5_recon-YYYYMMDD-HHMMSS/checkpoints/checkpoint_final.pkl \
  --dicom_dir /datasets/mmolefe/vinbigdata/cache_npy \
  --csv_path /datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv \
  --use_cache \
  --image_ids 000a1f4a1 000c3d2f9 \
  --output_dir runs_baseline_ldm/samples/from_images \
  --mode sample
```

## Tensor Shapes

- `z_common`: `(B, 16, 16, 16)`
- `z_cardio`: `(B, 16, 16, 16)`
- LDM input: `[x_t, z_common]` concatenated along channels
- LDM output: predicted noise for `z_cardio` only
