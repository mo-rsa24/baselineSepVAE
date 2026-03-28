# SepVAE Documentation System — Index

**Project:** Separable VAE for compositional chest X-ray generation
**Current milestone:** D3 (`d3_gan_fix-20260325-143813`) — clean reconstructions achieved
**Last updated:** 2026-03-25

---

## How to use this index

Each document is self-contained but cross-references the others. Use the **Reading guide** below to find the right entry point for your question.

| If you want to... | Start at |
|---|---|
| Understand the full history of the project | [01 Experiment Timeline](01_experiment_timeline.md) |
| Understand the data and preprocessing pipeline | [02 Data & Preprocessing](02_data_preprocessing.md) |
| Understand the multi-stage training strategy | [03 Training Curriculum](03_training_curriculum.md) |
| Understand the model architecture in detail | [04 Model Architecture](04_model_architecture.md) |
| Understand every loss term and what it enforces | [05 Objective Functions](05_objective_functions.md) |
| Debug a training failure or understand what went wrong before | [06 Failures & Debugging Log](06_failures_debugging.md) |
| See quantitative results across all phases | [07 Results & Evaluation](07_results_evaluation.md) |
| Understand exactly why D3 worked | [08 Current State — D3 Analysis](08_current_state_d3.md) |
| Plan or review D4–D7 stages | [09 Forward Plan](09_forward_plan.md) |
| Understand why bbox supervision failed and how CheXmask replaces it | [10 Mask Supervision Journey](10_mask_supervision_journey.md) |

---

## Documents

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | [01_experiment_timeline.md](01_experiment_timeline.md) | Chronological record: every phase, every run, every decision | Complete |
| 2 | [02_data_preprocessing.md](02_data_preprocessing.md) | VinBigData pipeline, MONOCHROME1 fix, caching | Complete |
| 3 | [03_training_curriculum.md](03_training_curriculum.md) | Multi-stage curriculum design, stage gates, rationale | Complete |
| 4 | [04_model_architecture.md](04_model_architecture.md) | SepVAE V2 full architecture with design rationale | Complete |
| 5 | [05_objective_functions.md](05_objective_functions.md) | All losses: formula, objective, evolution over stages | Complete |
| 6 | [06_failures_debugging.md](06_failures_debugging.md) | Categorised failure log: V1 and V2, symptoms → fixes | Complete |
| 7 | [07_results_evaluation.md](07_results_evaluation.md) | Metrics, quantitative results by phase, monitoring | Complete |
| 8 | [08_current_state_d3.md](08_current_state_d3.md) | Why D3 worked, what changed, analysis of success | Complete |
| 9 | [09_forward_plan.md](09_forward_plan.md) | D4–D7 plan, risks, hypotheses, monitoring | Complete |
| 10 | [10_mask_supervision_journey.md](10_mask_supervision_journey.md) | MedSAM investigation, z_disease collapse diagnosis, CheXmask validation | Complete |

---

## Related root-level documents

These pre-date the `docs/` system and are kept as-is:

| File | Content |
|------|---------|
| [SEPVAE_D3_MILESTONE.md](../SEPVAE_D3_MILESTONE.md) | Comprehensive D3 milestone snapshot (architecture, curriculum, checkpoint registry) |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Architecture overview (shorter form of Doc 4) |
| [PREPROCESSING.md](../PREPROCESSING.md) | Preprocessing reference (shorter form of Doc 2) |
| [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) | High-level project positioning |
| [PLAN_D5.md](../PLAN_D5.md) | Original D5 fix plan (implemented in D3 codebase) |
| [research_log/](../research_log/) | V1 architecture research log (Phases 1–7, CheSS backbone) |

---

## Checkpoint registry

| Phase | Run directory | Final checkpoint | Epoch |
|-------|--------------|-----------------|-------|
| D0 | `runs_sepvae/d0_smoke_v2-20260324-063142/` | `checkpoint_final.pkl` | 5 |
| D1 | `runs_sepvae/d1_recon_bbox_xattn-20260321-004241/` | `checkpoints/checkpoint_final.pkl` | 30 |
| D2 | `runs_sepvae/d2_perceptual_bbox-20260324-105108/` | `checkpoints/checkpoint_final.pkl` | 55 |
| D3 | `runs_sepvae/d3_gan_fix-20260325-143813/` | `checkpoints/checkpoint_epoch01XX.pkl` | 55→120 |

**Snapshot branch:** `snapshot/d3-gan-fix` (commit `ba6ab62`)
**To restore:** `git checkout snapshot/d3-gan-fix`
