# Chapter 07 — LDM Proof-of-Concept: Unconditional Sub-Block LDMs

**Previous chapter:** [06 Reconstruction Sharpness](06_reconstruction_sharpness.md)
**Next chapter:** [08 Full Sweeps](08_full_sweeps.md)

---

## 6. Phase 4 — LDM Proof-of-Concept: Unconditional Sub-Block LDMs

### 6.1 Why we ran LDMs before the SepVAE was fully fixed

The full composition pipeline (Strategy A, described in [Chapter 09](09_composition_theory.md)) requires:
1. Phase 1 SepVAE quality gate to pass (`specificity_ratio > 2.0`)
2. `LDM_common` trained on $z_{\text{common}}$ with $z_{\text{disease}}$ LDMs conditioned on $z_{\text{common}}$

However, before investing engineering effort in conditional LDM architecture, there is a more basic question: **Can a diffusion model learn to generate the disease sub-block latents at all?** Specifically:
- Is the marginal distribution of $z_{\text{cardio}}$ (2 channels × 64×64 spatial) well-shaped for VP-SDE diffusion?
- Does sampling from $p(z_{\text{disease}})$ and decoding produce recognisable disease morphology?
- Are the latent scale statistics reasonable after pre-encoding?

If these marginal LDMs fail, investing in conditional Strategy A architecture is premature. These runs are a **feasibility gate**, not a detour.

### 6.2 Checkpoint selection — why disentangle-E ep180

We had three completed checkpoints:
- `disentangle-E ep180–185` (best, W&B `laikh8dr`)
- `inactivity-G ep100` (W&B `9lj20so0`) — cardiomegaly head collapsed (AUC 0.476)
- `independence-I ep100` (W&B `41nce8qq`) — both heads in KL dead zone

The only checkpoint where **both** disease heads carry meaningful class information is disentangle-E. Despite its leakage issues (cross_head_score = 0.935), it is the only viable starting point.

We use **epoch 180** rather than epoch 185 (stated best in the results table) because `save_every=10` — checkpoints exist at ep170, ep180, ep190, ep200. Epoch 185 was not saved. ep190 and ep199 have lower probe AUC than the ep185 peak. ep180 is the nearest clean checkpoint.

| Epoch | Mean probe AUC | Status |
|-------|---------------|--------|
| ep170 | ~0.75 (est.) | Available |
| **ep180** | **~0.77 (est.)** | **Selected** |
| ep190 | ~0.74 | Degraded |
| ep199 | 0.737 | Last valid |
| ep200 | NaN | Unusable |

### 6.3 Pre-encoding pipeline

Pre-encoding extracts the disease-head latents ($\mu$ only, not a reparameterised sample) from the SepVAE encoder and writes them to `.npy` files indexed by `manifest.jsonl`. A scale factor $s = 1/\text{std}(\mu)$ is computed over the full dataset and stored in `latent_meta.json`; all `.npy` files are rescaled to unit variance so the LDM receives standard-normal inputs.

**Submit both simultaneously (CPU-only, no GPU contention):**

```bash
# Cardiomegaly latents
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio \
  slurm_scripts/preencode_sepvae.slurm

# Effusion latents
sbatch --nodelist=mscluster76 \
  --job-name=preencode-effusion \
  --export=ALL,\
DISEASE=effusion,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_effusion \
  slurm_scripts/preencode_sepvae.slurm
```

If interrupted, add `RESUME=1` to skip already-encoded image IDs:
```bash
sbatch ... --export=ALL,...,RESUME=1 slurm_scripts/preencode_sepvae.slurm
```

Read scale factor after job completes:
```bash
python -c "import json; d=json.load(open('preencoded_latents/disentangle_cardio/latent_meta.json')); print(d['latent_scale_factor'])"
```

### 6.4 LDM training

After pre-encoding, launch LDMs. `SAMPLE_EVERY=9999` disables in-loop sampling (no VAE decoder is loaded — training runs on stored latents only). Key parameters match the SepVAE disease sub-block: `latent_size=64`, `vae_z_channels=2`, `ldm_z_channels=2`.

```bash
./launchers/single_runs/ldm/train_ldm_vinbig_cardio.sh full_train
./launchers/single_runs/ldm/train_ldm_vinbig_effusion.sh full_train
```

### 6.5 What these runs are NOT

These are **not** the final composition LDMs. They do not condition on $z_{\text{common}}$. They do not account for the double-counting problem ([Chapter 09](09_composition_theory.md)). They will produce imperfect compositions because:
- $z_{\text{common}}$ must be provided externally (fixed from a real image) — no generation of anatomy
- The leakage in disentangle-E means each disease head partially encodes the other disease

They exist to answer the feasibility question and provide a quick visualisation of whether the learned latent subspace is useful for disease-specific generation at all.

---

## Supplementary: Expanded Checkpoint Rationale and Pre-Encoding Detail

*The following provides additional detail on checkpoint selection and the pre-encoding process, particularly the design decisions for `--use_mean` and scale factor computation.*

### Checkpoint comparison table

| | disentangle-E ep180 | inactivity-G ep100 | independence-I ep100 |
|---|---|---|---|
| Cardio probe AUC | **0.776** | 0.476 (collapsed) | 0.532 (dead zone) |
| Effusion probe AUC | **0.772** | 0.767 | 0.812 |
| Mean probe AUC | **0.774** | 0.622 | 0.672 |
| Both heads usable? | **Yes** | No (cardio dead) | No (cardio dead) |

independence-I has `free_bits=2 > KL_inactive≈1.8` — disease heads are permanently in the KL dead zone. inactivity-G collapses the cardiomegaly head by ep100. disentangle-E ep180 is the only checkpoint where both disease heads carry meaningful class information.

### Why `--use_mean` (store μ rather than a reparameterised sample)

Pre-encoded latents are the **training targets** for the LDM. Using the posterior mean $\mu$ rather than a reparameterised sample $\mu + \epsilon\sigma$ is preferred because:
- LDMs trained on samples from the posterior will learn to model $q(z|x)$ rather than the decoder's effective prior $p(z)$ — the scale factor correction partially compensates, but using $\mu$ directly is cleaner
- The slight variance underestimation from using $\mu$ is corrected by the `latent_scale_factor`

### Resume capability

If pre-encoding is interrupted (SLURM time limit, GPU allocation failure), the job can be resumed cleanly:

```bash
# Resume cardiomegaly encoding from where it left off
sbatch --nodelist=mscluster72 \
  --job-name=preencode-cardio-resume \
  --export=ALL,\
DISEASE=cardiomegaly,\
SEPVAE_CKPT=runs_sepvae/sepvae_disentangle-20260217-153031/checkpoints/checkpoint_epoch0180.pkl,\
OUTPUT_DIR=preencoded_latents/disentangle_cardio,\
RESUME=1 \
  slurm_scripts/preencode_sepvae.slurm
```

The script reads the existing `manifest.jsonl`, skips already-encoded `image_id`s, and continues appending.

### What these POC runs are not (long form)

These are **not** the Phase 2 LDMs described in the roadmap (see [Chapter 11](11_roadmap_and_commands.md)), which require:

- Phase 1 quality gate passed (`specificity_ratio > 2.0` on both disease heads)
- `LDM_common` trained on `z_common` 4-channel maps
- Disease LDMs conditioned on `z_common` via cross-attention or concatenation in the ScoreNet
- Full composition pipeline (`run/compose_diseases.py`)

The outputs of these POC runs will inform whether Strategy A is worth building and whether the disentangle-E checkpoint provides sufficient representation quality, or whether further SepVAE training (with R1–R5 applied) is needed before proceeding to the conditional composition architecture.

---

*End of Chapter 07. Continue to [Chapter 08: Full Sweeps](08_full_sweeps.md).*
