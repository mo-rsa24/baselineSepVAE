# Chapter 11 — Future Roadmap and Complete Run Commands

**Previous chapter:** [10 Verification and Hypothesis](10_verification_and_hypothesis.md)
**Next chapter:** [12 File Registry](12_file_registry.md)

---

## 12. Future Roadmap: Phases 8–11

### Phase 8 — Strategy A LDM training (conditional on G1–G4 passing)

```
Train:
  LDM_common   — unconditional VP-SDE on z_common (4ch × 64×64)
  LDM_cardio   — VP-SDE on z_cardio (2ch × 64×64), conditioned on z_common
  LDM_effusion — VP-SDE on z_effusion (2ch × 64×64), conditioned on z_common
```

Conditioning on $z_{\text{common}}$: cross-attention in the ScoreNet UNet denoiser, or concatenation along the channel axis. Cross-attention is preferred (allows attending to specific spatial positions of $z_{\text{common}}$ that are relevant to disease localisation).

### Phase 9 — Composition evaluation

```bash
python run/compose_diseases.py \
  --strategy a \
  --vae_checkpoint runs_sepvae/sepvae_full/best_checkpoint \
  --ldm_common_checkpoint runs_ldm/common \
  --ldm_cardio_checkpoint runs_ldm/cardio \
  --ldm_effusion_checkpoint runs_ldm/effusion \
  --n_samples 1000 \
  --out_dir results/composed_samples/

python run/eval_fid.py \
  --generated results/composed_samples/ \
  --reference data/vinbig/comorbid_holdout/
```

### Phase 10 — Ablation baseline

Train a single jointly-conditional LDM (or VAE) with a 4-class label (normal, cardio, effusion, both). Compare FID and region SSIM against Strategy A. If jointly-conditional wins, the factorisation assumption (A1) is the bottleneck. If Strategy A wins, disentangled composition is adding value.

### Phase 11 — CFG composition variant (Addition P4)

Train one conditional LDM with CFG dropout (`p_uncond=0.1`). At inference, apply dual-condition CFG guidance. Compare against Strategy A — divergence is diagnostic of how much cross-head information the disease LDMs learn.

---

## Supplementary: Complete Run Commands and Verification Sequence

*The following are the complete run commands for all phases, including SLURM launchers, direct Python invocations, diagnostic commands, and the recommended verification sequence.*

---

### SLURM launchers (recommended for cluster)

#### Biggpu launcher — **primary recommended command** (runs both configs in one job)

Submits a single SLURM job to `mscluster106` (2× RTX 8000, 49 GB each). Both training runs
share the node; GPU 0 runs `sepvae_baseline_fixed` and GPU 1 runs `sepvae_full`.

```bash
# Both configurations in one job — GPU 0: baseline+R11, GPU 1: full+R11+R12
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh

# Disable W&B logging
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --no_wandb

# Override batch sizes (defaults: 4/4 after OOM calibration)
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh \
  --batch_size_baseline 6 --batch_size_full 6

# Override spatial attention loss weight (default: 0.05, full run only)
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --weight_spatial_attn 0.1

# Disable spatial attention supervision
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --weight_spatial_attn 0.0

# Override target node
bash launchers/single_runs/vae/train_sep_vae_biggpu.sh --node mscluster107
```

What each GPU trains:

| GPU | Config | Loss stack | Batch size |
|-----|--------|------------|-----------|
| 0   | `sepvae_baseline_fixed` | R1+R2+R3+R4+**R11** | 4 |
| 1   | `sepvae_full`           | R1+R2+R3+R4+R5a+R5b+R7+**R11**+**R12** | 4 |

> Batch sizes were tuned down from 20/8 after OOM at the first JIT step (36 GB allocation). The launcher defaults are now `4/4`; the SLURM script internal defaults are `6/6`. Pass `--batch_size_baseline` / `--batch_size_full` to override.

Monitor logs after submission:
```bash
tail -f logs/biggpu-<JOB_ID>.out                          # SLURM wrapper
tail -f <STAGING_DIR>/logs/biggpu-baseline-<JOB_ID>.log  # baseline training
tail -f <STAGING_DIR>/logs/biggpu-full-<JOB_ID>.log      # full training
```

#### Single-GPU launchers (alternative)

```bash
# R1+R2+R3+R4 only (baseline, no R11)
bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh

# R1+R2+R3+R4 + R11 (baseline with disease discriminability)
USE_DISEASE_CLF=1 bash launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh

# R1+R2+R3+R4 + R5a (paired contrastive)
bash launchers/single_runs/vae/train_sep_vae_contrastive.sh

# R1+R2+R3+R4 + R5b (cross-head adversarial)
bash launchers/single_runs/vae/train_sep_vae_cross_adv.sh

# All recommendations (R1+R2+R3+R4+R5a+R5b)
bash launchers/single_runs/vae/train_sep_vae_full.sh

# All recommendations + R11
USE_DISEASE_CLF=1 bash launchers/single_runs/vae/train_sep_vae_full.sh
```

Override options (apply to any single-GPU launcher):
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

### Direct Python invocations — SepVAE training

R11 (`--use_disease_clf`) can be added to any invocation below with:
```
--use_disease_clf --weight_disease_clf 0.1 --lr_disease_clf 1e-4
```

```bash
# Baseline fixed (R1–R4)
python run/train_sep_vae.py \
  --exp_name sepvae_baseline_fixed \
  --batch_size 10 --epochs 100 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --weight_perceptual 0.05 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 \
  --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Baseline fixed + R11 (recommended; mirrors GPU 0 of biggpu run)
python run/train_sep_vae.py \
  --exp_name sepvae_baseline_fixed \
  --batch_size 8 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --weight_perceptual 0.05 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --use_disease_clf --weight_disease_clf 0.1 --lr_disease_clf 1e-4 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Add R5a (contrastive)
python run/train_sep_vae.py \
  ... \
  --use_contrastive \
  --weight_contrastive 0.1 \
  --contrastive_margin 0.5 \
  --exp_name sepvae_contrastive

# Add R5b (cross-head adversarial)
python run/train_sep_vae.py \
  ... \
  --use_cross_adv \
  --weight_cross_adv 0.05 \
  --lr_cross_disc 1e-4 \
  --cross_disc_hidden_dim 256 \
  --exp_name sepvae_cross_adv

# Full (R5a + R5b)
python run/train_sep_vae.py \
  ... \
  --use_contrastive --weight_contrastive 0.1 \
  --use_cross_adv   --weight_cross_adv 0.05 \
  --exp_name sepvae_full

# With spatial attention (R7)
python -m run.train_sep_vae \
  --exp_name sepvae_label_attention \
  --use_label_attention \
  --attn_query_dim 256 \
  --batch_size 10 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Full stack (R5a + R5b + R7 + R11 + R12 — mirrors GPU 1 of biggpu run)
python -m run.train_sep_vae \
  --exp_name sepvae_full \
  --use_label_attention --attn_query_dim 256 \
  --use_contrastive --weight_contrastive 0.1 --contrastive_margin 0.5 \
  --use_cross_adv --weight_cross_adv 0.05 --lr_cross_disc 1e-4 --cross_disc_hidden_dim 256 \
  --use_disease_clf --weight_disease_clf 0.1 --lr_disease_clf 1e-4 \
  --weight_spatial_attn 0.05 \
  --batch_size 4 --epochs 150 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.05 \
  --weight_orthogonality 0.05 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 80 \
  --kl_warmup_epochs 10 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle

# Sharpness sweep (R1–R4 + GAN + subpixel)
python run/train_sep_vae.py \
  --exp_name sepvae_sharp_test \
  --upsample_method subpixel \
  --weight_adversarial 0.1 \
  --weight_perceptual 0.1 \
  --disc_start_epoch 10 \
  --free_bits 0.0 \
  --sigma_inactive 0.05 \
  --weight_null 0.02 \
  --weight_orthogonality 0.02 \
  --weight_mi 0.005 \
  --lr_vae 1e-4 --lr_disc 1e-4 \
  --lr_decay_epochs 60 \
  --kl_warmup_epochs 10 \
  --batch_size 10 --epochs 100 \
  --half_precision bf16 --gradient_checkpointing \
  --wandb --wandb_project sepvae-disentangle
```

---

### Pre-encoding pipeline (LDM input preparation)

```bash
# Cardiomegaly — mscluster72
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio \
  slurm_scripts/preencode_sepvae.slurm

# Effusion — mscluster76
sbatch --nodelist=mscluster76 \
  --job-name=preencode-effusion \
  --export=ALL,\
DISEASE=effusion,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_effusion \
  slurm_scripts/preencode_sepvae.slurm

# Read scale factor after completion
python -c "import json; d=json.load(open('preencoded_latents/disentangle_cardio/latent_meta.json')); print(d['latent_scale_factor'])"
```

---

### LDM training (POC unconditional sub-block)

```bash
./launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh full_train
./launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh full_train
```

---

### Diagnostic commands (post-training)

```bash
# Analyse a completed run directory
python run/analyze_supervised_disease_axes.py \
  --run_dir runs_sepvae/sepvae_full-20260312-xxxxxx \
  --checkpoint checkpoint_epoch0100.pkl

# Edit purity gate check (implements G3)
python utils/sepvae_diagnostics.py \
  --mode edit_purity \
  --checkpoint runs_sepvae/sepvae_full/best_checkpoint \
  --n_samples 200 \
  --cardiac_row_range 0.35 0.55 \
  --cardiac_col_range 0.30 0.70 \
  --effusion_row_range 0.70 1.00 \
  --effusion_col_range 0.10 0.90 \
  --out results/research_log/edit_purity_report.json

# Generate swap grid manually
python -c "
from utils.sepvae_diagnostics import plot_latent_swap_grid
# ... load model, call function
"
```

---

### Recommended Verification Sequence (4 phases)

```
Phase 1: SepVAE quality gate
  → Train sepvae_full (R1–R5 + sharp fixes)
  → Measure specificity_ratio V1, V2 on validation set
  → Gate: specificity_ratio > 2.0 on both

Phase 2: LDM training (conditional on Phase 1 pass)
  → Train LDM_common on z_common maps
  → Train LDM_cardio conditioned on z_common
  → Train LDM_effusion conditioned on z_common
  → Train LDM_joint (CFG baseline, P4)

Phase 3: Composition evaluation
  → Strategy A sequential sampling
  → CFG composition (P4)
  → Jointly conditioned baseline (P3)
  → Compute FID, region SSIM, classifier confidence on all three

Phase 4: Ablation
  → Remove z_common conditioning from disease LDMs → measure degradation
  → Remove cross-adv discriminator → measure leakage increase
  → Replace SepVAE with standard VAE → measure composition quality drop
```

Proceed to Phase 2 only if `edit_purity_report.json` shows `specificity_ratio_cardio > 2.0` and `specificity_ratio_effusion > 2.0`.

---

*End of Chapter 11. Continue to [Chapter 12: File Registry](12_file_registry.md).*
