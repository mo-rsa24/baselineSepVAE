# SepVAE Research Log — Chapter Index

**Project:** Separable VAE for compositional chest X-ray generation
**W&B project group:** `sepvae-disentangle`
**Branch:** `sepVAEIndependet`
**Source documents:** `RESEARCH_LOG.md`, `research_discussion.md`, `sepvae_disentangle_analysis.md`
**Last updated:** 2026-03-13

---

> **How this series is organised.**
> These chapters form a textbook-style progression from project motivation through to future composition experiments. Each chapter corresponds to one logical phase or topic. The backbone of every chapter is drawn verbatim from `RESEARCH_LOG.md`; supplementary detail from `research_discussion.md` and `sepvae_disentangle_analysis.md` is integrated into the appropriate chapter without duplication.
>
> **Start here if you are returning to the project:** Read the chapter whose title matches the phase you were in when you left. Use this index to navigate.

---

## Chapter Map

| # | File | Contents | Source sections |
|---|------|----------|-----------------|
| 00 | `00_INDEX.md` | **This file** — navigation guide | — |
| 01 | `01_project_overview_and_motivation.md` | Research objective, informal hypothesis, clinical motivation | RESEARCH_LOG §1; discussion §1 |
| 02 | `02_architecture.md` | Encoder, decoder, disease routing, loss portfolio, metrics | RESEARCH_LOG §2; discussion §2 |
| 03 | `03_empirical_results.md` | Phases 1 & 2 training runs, full run tables, hyperparameter matrix, group consensus | RESEARCH_LOG §3–4; analysis §1–4 |
| 04 | `04_diagnoses_and_patterns.md` | Root causes (free-bits dead zone, NaN, head collapse, leakage), cross-run patterns | RESEARCH_LOG §4.3; discussion §4–5; analysis §4–5 |
| 05 | `05_fixes_r1_to_r7.md` | All code & config changes: R1–R6 (Phase 3), R7 spatial attention (Phase 6), **R11 disease discriminability classifier** (Phase 7), with full math; R8–R10 proposed next fixes | RESEARCH_LOG §5, §8; discussion §6; analysis §6–7 |
| 06 | `06_reconstruction_sharpness.md` | Why outputs are blurry; Fix 1–4 ordered by impact; recommended sharpness sweep | RESEARCH_LOG §7; analysis §8 |
| 07 | `07_ldm_proof_of_concept.md` | Feasibility gate for diffusion over disease sub-blocks; pre-encoding pipeline; LDM training | RESEARCH_LOG §6; analysis §10 |
| 08 | `08_full_sweeps.md` | Phase 7 sweep matrix combining all fixes; what to track in W&B | RESEARCH_LOG §9; analysis §7 |
| 09 | `09_composition_theory.md` | Assumptions A1–A3; three composition strategies; mathematical double-counting problem | RESEARCH_LOG §10; discussion §7–8 |
| 10 | `10_verification_and_hypothesis.md` | Gates G1–G4; V1–V4 criteria with code; four proposed additions P1–P4; revised falsifiable hypothesis | RESEARCH_LOG §11; discussion §9–11; analysis §9.3–9.6 |
| 11 | `11_roadmap_and_commands.md` | Phases 8–11 roadmap; full run commands (SLURM launchers + Python); verification sequence | RESEARCH_LOG §12; discussion §12; analysis §9.6 |
| 12 | `12_file_registry.md` | Code file purposes, checkpoint locations, analysis documents | RESEARCH_LOG §13; discussion Appendix |

---

## Reading Guide by Task

### "I want to understand what we built and why."
→ [Chapter 01](01_project_overview_and_motivation.md) then [Chapter 02](02_architecture.md)

### "I want to see what the runs showed."
→ [Chapter 03](03_empirical_results.md) for tables and numbers
→ [Chapter 04](04_diagnoses_and_patterns.md) for root causes

### "I want to know what code changes were made and why."
→ [Chapter 05](05_fixes_r1_to_r7.md) (R1–R7, R11 implemented; R8–R10 proposed with full implementation specs)

### "I want to understand the sharpness problem."
→ [Chapter 06](06_reconstruction_sharpness.md)

### "I want to know what LDM runs were started and how to reproduce them."
→ [Chapter 07](07_ldm_proof_of_concept.md) for feasibility runs
→ [Chapter 11](11_roadmap_and_commands.md) for full run commands

### "I want to understand the composition theory and mathematical risks."
→ [Chapter 09](09_composition_theory.md)

### "I want to know what must be verified before composition experiments."
→ [Chapter 10](10_verification_and_hypothesis.md) for gates and hypothesis

### "I want to plan the next run."
→ [Chapter 08](08_full_sweeps.md) for current sweep matrix
→ [Chapter 11](11_roadmap_and_commands.md) for all run commands

### "I want to find a file or checkpoint."
→ [Chapter 12](12_file_registry.md)

---

## Phase Timeline Summary

| Phase | Chapter | Period | What happened |
|-------|---------|--------|---------------|
| Phase 1 | 03 | Feb 16–17, 2026 | Initial 200-epoch disentangle chain; peak AUC 0.774; NaN at ep200 |
| Phase 2 | 03–04 | Feb 20, 2026 | Inactivity-G and independence-I sweeps; diagnosed free_bits dead zone, cardiomegaly collapse |
| Phase 3 | 05 | Mar 2026 | Implemented R1–R6; cosine decay, rebalanced nulling, contrastive + cross-adv losses |
| Phase 4 (POC) | 07 | Mar 2026 | Launched unconditional sub-block LDMs over disentangle-E ep180 latents |
| Phase 5 | 06 | Mar 2026 | Diagnosed reconstruction blurriness; identified Fix 1–4; PatchGAN enabled |
| Phase 6 | 05 | Mar 2026 | Implemented R7 spatial attention heads (`--use_label_attention`) |
| Phase 7 | 05, 08 | Mar 2026 | Full sweep matrix; implemented R11 disease discriminability classifier (`--use_disease_clf`); biggpu dual-GPU launcher |
| Phase 8+ | 10–11 | Pending G1–G4 | Strategy A conditional LDM training; composition evaluation; ablation baseline |

---

*To update: add new phases to the timeline table above and create new chapter files as needed.*
