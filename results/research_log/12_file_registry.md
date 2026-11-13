# Chapter 12 — File and Checkpoint Registry

**Previous chapter:** [11 Roadmap and Commands](11_roadmap_and_commands.md)
**Back to index:** [00 Index](00_INDEX.md)

---

## 13. File and Checkpoint Registry

### Code files

| File | Role |
|------|------|
| `run/train_sep_vae.py` | Main SepVAE training script — all R1–R12 flags |
| `losses/sep_vae_losses.py` | All loss functions: `paired_contrastive_loss`, `CrossHeadDiscriminator`, `cross_head_disc_loss`, `spatial_attention_loss` (R12), `_bbox_to_mask`, `sepvae_loss` |
| `models/sep_vae_jax.py` | SepVAE architecture: `SepVAEEncoder`, `SepVAEDecoder`, `SmoothUp`, `DiseaseAttentionHead` |
| `datasets/VinBigData.py` | Triplet dataloader; returns `bbox_disease1`/`bbox_disease2` (B×4, normalised [0,1]) for R12 spatial supervision |
| `utils/sepvae_diagnostics.py` | Diagnostics: `plot_latent_swap_grid`, `plot_per_channel_kl_heatmap`, PCA, attention maps |
| `scripts/preencode_sepvae_latents.py` | Pre-encoding pipeline for LDM training |
| `run/analyze_supervised_disease_axes.py` | Post-hoc analysis: probe classifier, cross_head_score |
| `slurm_scripts/preencode_sepvae.slurm` | SLURM script for pre-encoding |
| `slurm_scripts/sep_vae.slurm` | SLURM script for SepVAE training |
| `launchers/single_runs/vae/train_sep_vae_baseline_fixed.sh` | R1+R2+R3+R4 launcher |
| `launchers/single_runs/vae/train_sep_vae_contrastive.sh` | R1–R4 + R5a launcher |
| `launchers/single_runs/vae/train_sep_vae_cross_adv.sh` | R1–R4 + R5b launcher |
| `launchers/single_runs/vae/train_sep_vae_full.sh` | All recommendations launcher |
| `launchers/single_runs/vae/train_sep_vae_biggpu.sh` | Dual-GPU biggpu launcher — `--weight_spatial_attn` override supported |
| `launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh` | LDM (cardiomegaly sub-block) launcher |
| `launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh` | LDM (effusion sub-block) launcher |

### Analysis documents

| File | Content |
|------|---------|
| `results/research_log/00_INDEX.md` | Chapter navigation guide |
| `results/research_log/01_project_overview_and_motivation.md` | Research objective, informal hypothesis, clinical motivation |
| `results/research_log/02_architecture.md` | Encoder, decoder, routing, loss portfolio |
| `results/research_log/03_empirical_results.md` | All Phase 1 & 2 runs, full tables, group consensus |
| `results/research_log/04_diagnoses_and_patterns.md` | Root causes (free_bits, NaN, cardiomegaly collapse, leakage), cross-run patterns |
| `results/research_log/05_fixes_r1_to_r7.md` | All code changes R1–R7 with full rationale and math |
| `results/research_log/06_reconstruction_sharpness.md` | Blurriness root causes, Fix 1–4, sharpness sweep |
| `results/research_log/07_ldm_proof_of_concept.md` | POC feasibility runs, pre-encoding pipeline, LDM training |
| `results/research_log/08_full_sweeps.md` | Phase 7 sweep matrix with all configurations |
| `results/research_log/09_composition_theory.md` | Assumptions A1–A3, Strategies A/B/C/CFG with full math |
| `results/research_log/10_verification_and_hypothesis.md` | Gates G1–G4, criteria V1–V4, additions P1–P4, revised hypothesis |
| `results/research_log/11_roadmap_and_commands.md` | Phases 8–11 roadmap, complete run commands, verification sequence |
| `results/research_log/12_file_registry.md` | **This file** — code files and checkpoints |
| `results/research_log/RESEARCH_LOG.md` | Master research log (source for all above chapters) |
| `results/sepvae_disentangle_analysis.md` | Full cross-run comparison table, per-group consensus, recommendations, sharpness and research theory (detailed supplementary reference) |
| `results/research_discussion.md` | Theory-first reference: architecture, loss formulations, strategy comparison, hypothesis — structured as a standalone document for writing/presentation |

### Key checkpoints

| Checkpoint | W&B | Description | Use |
|-----------|-----|-------------|-----|
| `runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl` | `laikh8dr` | disentangle-E ep180 — best usable checkpoint | LDM pre-encoding (Phase 4 / Chapter 07) |
| *(to be added)* | — | sepvae_full ep100 | Gate evaluation (G1–G4) |
| `preencoded_latents/disentangle_cardio/` | — | Pre-encoded $z_{\text{cardio}}$ from ep180 | LDM_cardio training |
| `preencoded_latents/disentangle_effusion/` | — | Pre-encoded $z_{\text{effusion}}$ from ep180 | LDM_effusion training |

---

## Supplementary: File Purpose Index

*The following is a condensed reference for quickly locating where specific functionality lives.*

| What you want | Where to look |
|---------------|---------------|
| Main training entrypoint | `run/train_sep_vae.py` |
| Loss functions (all R1–R5) | `losses/sep_vae_losses.py` |
| Model architecture | `models/sep_vae_jax.py` |
| Attention head implementation | `models/sep_vae_jax.py` → `DiseaseAttentionHead` |
| Bbox spatial attention loss (R12) | `losses/sep_vae_losses.py` → `spatial_attention_loss`, `_bbox_to_mask` |
| Bbox loading from VinBigData CSV | `datasets/VinBigData.py` → `_build_bbox_lookup`, `_load_disease_image` |
| Subpixel upsampling | `models/sep_vae_jax.py:68–82` → `SmoothUp` |
| PatchGAN discriminator | `losses/sep_vae_losses.py` |
| Probe AUC and cross_head_score | `run/analyze_supervised_disease_axes.py` |
| Swap grid visualisation | `utils/sepvae_diagnostics.py` → `plot_latent_swap_grid` |
| KL heatmap | `utils/sepvae_diagnostics.py` → `plot_per_channel_kl_heatmap` |
| Attention map logging | `utils/sepvae_diagnostics.py` |
| Edit purity metric (to implement) | `utils/sepvae_diagnostics.py` → add `--mode edit_purity` |
| Dataset with triplets | `datasets/VinBigData.py` |
| Pre-encoding SLURM | `slurm_scripts/preencode_sepvae.slurm` |

---

*End of Chapter 12. Return to [Chapter Index](00_INDEX.md).*

---

*This documentation series is a living reference. Update `RESEARCH_LOG.md` whenever a new training run is launched or a significant finding is made, then propagate changes to the relevant chapter file. The goal is that anyone returning to this project after a gap — including the original authors — can reconstruct exactly why each decision was made and what it led to.*
