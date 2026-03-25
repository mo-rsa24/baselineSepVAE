# Chapter 08 — Phase 7: Full SepVAE Sweeps with All Fixes

**Previous chapter:** [07 LDM Proof of Concept](07_ldm_proof_of_concept.md)
**Next chapter:** [09 Composition Theory](09_composition_theory.md)

---

## 9. Phase 7 — Full SepVAE Sweeps with All Fixes

This is the current recommended training configuration combining all improvements from Phases 3–6. The goal is to satisfy the verification gates (Section 11) before proceeding to Strategy A LDM composition (Phase 8).

### Sweep matrix

| Config name | R1–R4 | R5a | R5b | R7 | GAN | Subpixel | Purpose |
|-------------|-------|-----|-----|----|-----|----------|---------|
| `baseline_fixed` | ✓ | — | — | — | — | — | Establish corrected baseline |
| `contrastive` | ✓ | ✓ | — | — | — | — | Test paired contrastive alone |
| `cross_adv` | ✓ | — | ✓ | — | — | — | Test cross-adversarial alone |
| `sharp_test` | ✓ | — | — | — | ✓ | ✓ | Test sharpness fixes alone |
| `attention` | ✓ | — | — | ✓ | — | — | Test spatial routing alone |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full combined stack |

Each run logs to W&B project `sepvae-disentangle`. After 100 epochs, compare:
- `probe_auc/cardiomegaly` and `probe_auc/effusion` (rolling 3-epoch average)
- `cross_head_score` at epoch 50 and epoch 100
- Attention heatmaps (for runs with R7)
- Reconstruction quality (visual + SSIM to be added)

---

## Supplementary: Complete Configuration Blocks for All Sweep Variants

*The following are the full, ready-to-run configurations for each entry in the sweep matrix.*

### Baseline fixed (R1–R4, R6)

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_baseline_fixed \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

### With paired contrastive loss (R5a)

```bash
# same as baseline_fixed, plus:
  --use_contrastive \
  --weight_contrastive 0.1 \
  --exp_name sepvae_contrastive
```

### With cross-head adversarial discriminator (R5b)

```bash
# same as baseline_fixed, plus:
  --use_cross_adv \
  --weight_cross_adv 0.05 \
  --lr_cross_disc 1e-4 \
  --exp_name sepvae_cross_adv
```

### Full (R5a + R5b)

```bash
# same as baseline_fixed, plus:
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --exp_name sepvae_full_recommendations
```

### With disease prototype cross-attention heads (R7)

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_label_attention \
  --use_label_attention \
  --attn_query_dim 256 \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

**What to look for in W&B:** Attention map images logged under `diagnostics/attn_maps` every `--sample_every` epochs. Early epochs should show diffuse/uniform maps (expected — query is near-zero at init). By epoch 20–30, cardiomegaly maps should begin concentrating over the central cardiac silhouette region; effusion maps over the lower lateral pleural angles. If both maps remain diffuse after epoch 50, the disease signal is too weak relative to nulling pressure — reduce `--weight_null` or `--sigma_inactive`.

### Full stack (R5a + R5b + R7)

```bash
python -m run.train_sep_vae \
  --exp_name sepvae_full \
  --use_label_attention \
  --attn_query_dim 256 \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --batch_size 10 \
  --epochs 150 \
  --free_bits 0.5 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_cross_disc 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

### SLURM launchers (preferred)

```bash
# R1+R2+R3+R4 only (baseline, no structural disentanglement)
bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh

# R1+R2+R3+R4 + R5a (paired contrastive)
bash launchers/single_runs/vae/train_sep_vae_contrastive.sh

# R1+R2+R3+R4 + R5b (cross-head adversarial)
bash launchers/single_runs/vae/train_sep_vae_cross_adv.sh

# All recommendations (R1+R2+R3+R4+R5a+R5b)
bash launchers/single_runs/vae/train_sep_vae_full.sh
```

Override options (apply to any launcher):
```bash
# Change partition
bash launchers/single_runs/vae/train_sep_vae_full.sh --partition gpuq

# Disable W&B
bash launchers/single_runs/vae/train_sep_vae_full.sh --no_wandb

# Resume from checkpoint
bash launchers/single_runs/vae/train_sep_vae_full.sh \
  --resume runs_sepvae/sepvae_full-20260312-xxxxxx/checkpoints/checkpoint_epoch0050.pkl

# Adjust contrastive weight
bash launchers/single_runs/vae/train_sep_vae_full.sh \
  --weight_contrastive 0.2 --contrastive_margin 0.3
```

---

*End of Chapter 08. Continue to [Chapter 09: Composition Theory](09_composition_theory.md).*
