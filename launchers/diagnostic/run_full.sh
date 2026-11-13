#!/bin/bash

export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/superdiff-ldm
# Update these paths to your actual model directories
NORMAL_RUN="runs_ldm/ldm-normal-composition-preencode-latents-ancestral-8b4bb7d-20260120-111058"
TB_RUN="runs_ldm/ldm-tb-composition-preencode-latents-ancestral-8b4bb7d-20260120-123239"

python3 compose_batch.py \
    --run_dir_normal "$NORMAL_RUN" \
    --run_dir_tb "$TB_RUN" \
    --output_path "superdiff_full_experiment.png" \
    --sampler Faithful \
    --steps 200 \
    --seed 42 \
    --lift 0.25 \
    --num_samples 500 \
    --batch_size 8 \
    --num_visual_samples 8 \
    --sample_images True