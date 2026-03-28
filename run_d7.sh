#!/usr/bin/env bash
# D7 — direct run on mscluster107, GPU 1
# Usage: bash run_d7.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

D3_CKPT="${WORKDIR}/runs_sepvae/d3_gan_fix-20260325-143813/checkpoints/checkpoint_final.pkl"
DATA_DIR="/datasets/mmolefe/vinbigdata/cache_npy"
CSV_PATH="/datasets/mmolefe/vinbigdata/cache_npy/train_filtered.csv"
OUTPUT_ROOT="${WORKDIR}/runs_sepvae"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

python -u run/train_sep_vae.py \
  --dicom_dir              "$DATA_DIR" \
  --csv_path               "$CSV_PATH" \
  --use_cache \
  --deterministic_data \
  \
  --model_version          v2 \
  --img_size               256 \
  --z_channels_common      32 \
  --z_channels_disease     16 \
  --attn_query_dim         256 \
  --attn_heads             4 \
  --decoder_res_blocks     3 \
  --use_bbox_cross_attn \
  --bbox_query_mix         1.0 \
  --bbox_dropout_prob      0.3 \
  \
  --batch_size             4 \
  --epochs                 200 \
  --num_workers            8 \
  --eval_num_workers       0 \
  --seed                   0 \
  --kl_warmup_epochs       0 \
  \
  --lr_vae                 5e-5 \
  --lr_disc                1e-4 \
  --weight_decay           1e-4 \
  --grad_clip              1.0 \
  \
  --weight_rec             1.0 \
  --weight_kl_common       1e-4 \
  --weight_kl_disease      5e-5 \
  --kl_free_bits           0.5 \
  --weight_mi_factor       1.0 \
  --weight_bbox_attn       0.10 \
  --weight_cardio_supcon   0.05 \
  --weight_perceptual      0.0 \
  --weight_gan             0.1 \
  --weight_tv              0.005 \
  --weight_masked_rec      0.3 \
  --sigma_inactive         0.1 \
  \
  --gan_start_step         500 \
  --lr_patch_disc          1e-4 \
  --disc_r1_penalty        0.0 \
  \
  --output_root            "$OUTPUT_ROOT" \
  --exp_name               d7_skip \
  --save_every             5 \
  --sample_every           5 \
  --manifold_every         5 \
  --manifold_bbox_mode     both \
  --eval_subset_size       1024 \
  --manifold_max_samples   1024 \
  \
  --resume                 "$D3_CKPT" \
  \
  --wandb --wandb_project  baseline-sepvae \
  2>&1 | tee logs/d7_skip_$(date +%Y%m%d_%H%M%S).log
