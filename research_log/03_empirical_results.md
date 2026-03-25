# Chapter 03 — Empirical Results: Phases 1 & 2

**Previous chapter:** [02 Architecture](02_architecture.md)
**Next chapter:** [04 Diagnoses and Patterns](04_diagnoses_and_patterns.md)

---

## 3. Phase 1 — Initial Disentangle Runs (Feb 16–17, 2026)

### 3.1 What we ran

Five runs under the `sepvae_disentangle` name, forming one continuous training chain split across SLURM jobs:

| Run ID | W&B | Epochs | Notes |
|--------|-----|--------|-------|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | 192 | Old script — no probe AUC logged |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | 1 | **CRASHED** at step 0 |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | 84 | Stopped; resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | 101 | Stopped; resumed as E |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | 200 | **NaN crash at epoch 200** |

> Runs C, D, E are the same training, checkpointed and resubmitted. Together they represent one 200-epoch run (the disentangle-E chain).

**Configuration (disentangle-C/D/E):**

```bash
--free_bits 1.0
--sigma_inactive 0.1         # → KL_inactive ≈ 1.8 nats (above free_bits)
--weight_null 0.01
--weight_orthogonality 0.03
--weight_mi 0.003
--weight_perceptual 0.03
--use_fpn true
--batch_size 6
--lr_vae 6.7e-5
--kl_warmup_epochs 30
--half_precision fp32
--epochs 200
```

### 3.2 What we saw

**Good:** Disease heads learned something useful. Peak probe AUC ≈ 0.774 mean (cardiomegaly: 0.776, effusion: 0.772) at around epoch 185. This is the best any run has achieved.

**Problem 1 — NaN at epoch 200.** Total loss was stable (0.004–0.006) through epoch 199, then all terms simultaneously became `nan`. PCA failed with `ValueError: Input X contains NaN`. The run had no learning rate schedule. After 200 epochs with constant LR, one catastrophic gradient update from an unlucky batch cascaded through the network.

**Problem 2 — Persistent cross-head leakage.** Cross_head_score fluctuated between 0.85 and 1.01 throughout all 200 epochs. The heads never achieved genuine independence. The MI discriminator and orthogonality loss reduce marginal correlation but do not enforce **conditional** independence — a head can still encode the other disease without violating orthogonality or MI constraints.

**Problem 3 — High probe AUC variance.** The 600-sample probe eval produced noisy estimates — the same checkpoint read anywhere from 0.53 to 0.77 on adjacent evaluations. Epoch 185 "best" is partly noise; rolling averages were not implemented.

**Problem 4 — No discrimination between active/inactive KL regions.** The free_bits mechanism clips KL gradients for any channel below $\lambda_{\text{fb}} = 1.0$ nats. Since $\sigma_{\text{inactive}} = 0.1$ gives $\text{KL}_{\text{inactive}} \approx 1.8 > 1.0$, disease channels here are above the threshold and receive gradients — this works. But we had not yet systematically checked what happens when free_bits is raised.

### 3.3 What we learned / questions raised

- Long training (200 epochs) does improve peak performance — but **without LR decay it is unstable**. We need a cosine schedule.
- The current regularisation set ($\mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{MI}} + \mathcal{L}_{\text{null}}$) is insufficient to eliminate cross-head leakage. A stronger structural mechanism is needed.
- The cardiomegaly head seems harder to train — its probe AUC is more volatile and lower than effusion's. Is this a feature of the disease (distributed, subtle signal) or a regularisation imbalance?

**Questions these findings raised that drove Phase 2:**
1. What if we increase the inactivity pressure (tighter $\sigma_{\text{inactive}}$, stronger nulling)?
2. What if we increase the independence pressure (stronger orthogonality, add explicit MI weight)?
3. What is the effect of `free_bits` on the disease heads specifically?

---

## 4. Phase 2 — Targeted Hyperparameter Sweeps (Feb 20, 2026)

We ran two new experiments, each isolating a different hypothesis from the Phase 1 findings. Both used a simplified setup: no FPN, bf16, batch size 10, 100 epochs.

### 4.1 Sweep 1: Inactivity-driven (`sepvae_inactivity_driven`)

**Scientific rationale:** Phase 1 showed leakage despite moderate null/ortho weights. Hypothesis: if the inactive heads are held even tighter to the prior ($\sigma_{\text{inactive}} = 0.05$ instead of 0.1), and the nulling weight is increased 5×, the inactive head will have less capacity to represent the other disease.

$$\text{KL}_{\text{inactive}}(\sigma=0.05) = \frac{1}{2}(0.05^2 - 1 - \log 0.05^2) \approx 2.5 \text{ nats}$$

This is above `free_bits=1.0`, so gradients still flow.

**Run:** `sepvae_inactivity_driven-20260220-085943` | W&B `9lj20so0`
(Note: `62635` / W&B `99s5jhqe` crashed at ~93s from transient GPU conflict; the config was identical, relaunched 23 minutes later.)

```bash
--free_bits 1.0
--sigma_inactive 0.05        # tighter inactive prior (KL_inactive ≈ 2.5)
--weight_null 0.05           # 5× stronger than Phase 1
--weight_orthogonality 0.05
--weight_mi 0.005
--weight_perceptual 0.05
--use_fpn false
--batch_size 10
--lr_vae 1e-4
--kl_warmup_epochs 10
--half_precision bf16
--epochs 100
```

**Results:**

| Metric | Best (~ep50) | Final (ep100) |
|--------|-------------|---------------|
| Probe AUC (cardio) | 0.639 | **0.476** (below chance) |
| Probe AUC (effusion) | 0.627 | 0.767 |
| Cross-head score | **0.744** | 0.809 |

**What we saw:** This run achieved the **best cross-head leakage suppression** of all runs — cross_head_score = 0.744 at its best. However, by epoch 100 the cardiomegaly probe AUC had collapsed to 0.476 (sub-random). The strong nulling pressure drove $\mu_{\text{cardio}} \to 0$ faster than the reconstruction gradient could maintain the cardiomegaly signal.

**Why cardiomegaly specifically?** Cardiomegaly is a distributed, low-contrast change (enlarged cardiac silhouette, subtle mediastinal widening). Effusion is a localised, high-contrast finding (bright pleural fluid). The nulling loss applies equal pressure to both heads regardless of how strong the disease signal is. For cardiomegaly, the signal is weaker — the nulling pressure wins over the reconstruction gradient that tries to preserve it.

**What we learned:** Reducing leakage at the cost of one head dying is not a solution. The regularisation balance needs to be asymmetric or the minimum active KL needs to be floored. This is captured in R4.

---

### 4.2 Sweep 2: Independence-driven (`sepvae_independence_driven`)

**Scientific rationale:** Orthogonality and MI enforce marginal independence. What if we push these much harder (10× stronger orthogonality) while relaxing nulling? And what happens to `free_bits` if we raise it to 2.0?

**Run:** `sepvae_independence_driven-20260220-085943` | W&B `41nce8qq`
(Note: `62742` / W&B `i9cevm1x` crashed at ~91s; same transient GPU issue.)

```bash
--free_bits 2.0              # ← THIS IS THE CRITICAL MISTAKE
--sigma_inactive 0.1         # → KL_inactive ≈ 1.8 nats (below free_bits=2.0!)
--weight_null 0.01
--weight_orthogonality 0.1   # 3× stronger than Phase 1
--weight_mi 0.01             # 3× stronger than Phase 1
--weight_perceptual 0.05
--use_fpn false
--batch_size 10
--lr_vae 1e-4
--kl_warmup_epochs 10
--half_precision bf16
--epochs 100
```

**Results:**

| Metric | Early peak (~ep25) | Final (ep100) |
|--------|--------------------|---------------|
| Probe AUC (cardio) | 0.762 | 0.532 |
| Probe AUC (effusion) | 0.737 | 0.812 |
| Cross-head score | 0.820 | 0.871 |
| Disease head $\mu$-norms | moderate | **≈ 0.025–0.030** (collapsed) |
| KL (disease channels) | variable | **uniformly 1.7–1.8 nats** |

**Root cause — the free_bits dead zone:**

The free-bits mechanism clips KL gradients to zero for any channel with KL below the threshold:

```
KL_inactive ≈ 1.8 nats
─────────────────────────────────────────────────────────
          ◄── gradient = 0 in this zone ──►
    0 ───────────────────────── 2.0 ──────── ∞
                                 ↑ free_bits = 2.0
              disease heads land here permanently
```

Because $\text{KL}_{\text{inactive}}({\sigma=0.1}) \approx 1.8 < \text{free\_bits} = 2.0$, all disease channels are permanently below the threshold. They receive **no KL gradient**. The nulling loss drives $\mu \to 0$, the channels collapse, and the probe AUC degrades to near-chance.

There is a brief window around epoch 20–25 where the channels haven't fully collapsed yet (probe AUC 0.75), but without KL gradients to maintain structure, the heads degrade over the remaining 75 epochs.

**What we learned:** The free_bits mechanism was designed to protect the common head from over-penalisation on low-information channels. Raising it to 2.0 inadvertently killed the disease heads. This is a binary failure condition, not a soft tradeoff. **Any configuration with $\text{KL}_{\text{inactive}} < \text{free\_bits}$ will fail.**

---

### 4.3 Cross-run patterns (summary after Phase 2)

These five patterns were identified by comparing all runs:

**Pattern 1 — `free_bits` / `sigma_inactive` conflict is a hard failure mode**
Any run with $\text{KL}_{\text{inactive}} < \text{free\_bits}$ results in disease heads in a permanent gradient dead zone. All such runs fail. This must be checked before any other hyperparameter analysis.

$$\text{KL}_{\text{inactive}} = \frac{1}{2}\left(\sigma_{\text{inactive}}^2 - 1 - \log \sigma_{\text{inactive}}^2\right)$$

| $\sigma_{\text{inactive}}$ | $\text{KL}_{\text{inactive}}$ | Must keep `free_bits` below |
|--------------------------|-------------------------------|------------------------------|
| 0.20 | ~1.1 nats | 1.0 |
| 0.10 | ~1.8 nats | 1.5 |
| 0.05 | ~2.5 nats | 2.0 (safe up to 2.4) |

**Pattern 2 — Cross-head leakage is universally unsolved**
No run ever achieved cross_head_score < 0.74. The orthogonality + MI + nulling portfolio enforces marginal independence but not conditional independence. The heads can share information about each other's disease without violating any current loss.

**Pattern 3 — Cardiomegaly is consistently harder to train**
Across all runs, cardiomegaly probe AUC is lower, more volatile, and more prone to collapse. Likely because: (a) cardiomegaly is a subtle, distributed feature; (b) CheSS backbone was not trained to separate cardiac pathology from anatomy; (c) the nulling loss applies equal pressure regardless of signal strength.

**Pattern 4 — Probe AUC is too noisy to use single-epoch readings**
Same checkpoint read at adjacent epochs can differ by 0.2+ AUC. The 600-sample probe eval has high variance. Rolling averages or best-of-5 should be the reporting standard.

**Pattern 5 — Long training improves peak but destabilises without LR decay**
200-epoch chain achieves highest peak (0.774) but ends in NaN. No LR schedule in any run.

---

## Supplementary: Complete Run Tables and Group-Level Consensus

*The following tables include all 9 runs (including disentangle-A and crashed runs) and the group-level qualitative analysis.*

### Complete Run Inventory (9 runs)

| Run folder | Wandb ID | Methodology | Epochs | Status |
|---|---|---|---|---|
| `sepvae_disentangle-20260216-154029` | `ir9i9n69` | disentangle-A | 192/200 | ~complete (old script, no probe AUC) |
| `sepvae_disentangle-20260217-054304` | `mlu6en5q` | disentangle-B | 1 | **CRASHED** (step 0 only) |
| `sepvae_disentangle-20260217-054711` | `r588hl2e` | disentangle-C | 84 | Stopped — resumed as D |
| `sepvae_disentangle-20260217-130658` | `hbtxgt0b` | disentangle-D | 101 | Stopped — resumed as E (from C ep80) |
| `sepvae_disentangle-20260217-153031` | `laikh8dr` | disentangle-E | 200 | **NaN crash at final epoch** (from D ep100) |
| `sepvae_inactivity_driven-20260220-062635` | `99s5jhqe` | inactivity-F | 0 | **CRASHED** (~93 s, transient GPU) |
| `sepvae_inactivity_driven-20260220-085943` | `9lj20so0` | inactivity-G | 100 | Completed |
| `sepvae_independence_driven-20260220-062742` | `i9cevm1x` | independence-H | 0 | **CRASHED** (~91 s, transient GPU) |
| `sepvae_independence_driven-20260220-085943` | `41nce8qq` | independence-I | 100 | Completed |

> Note: runs C, D, E form a single continuous training run split across three SLURM jobs.
> Runs F and H crashed during model init — the same configs succeeded 23 min later (G and I), indicating a transient GPU resource conflict.

### Full Hyperparameter Matrix (all groups, including disentangle-A)

| Parameter | disentangle-A | disentangle-C/D/E | inactivity-G | independence-I |
|---|---|---|---|---|
| `free_bits` | 1 | 1 | 1 | **2** |
| `sigma_inactive` | 0.2 | 0.1 | **0.05** | 0.1 |
| KL_inactive (est.) | ~1.1 nats | ~1.8 nats | ~2.5 nats | ~1.8 nats |
| KL_inactive > free_bits? | yes | yes | yes | **NO (1.8 < 2.0)** |
| `weight_null` | 0.005 | 0.01 | **0.05** | 0.01 |
| `weight_orthogonality` | 0.01 | 0.03 | 0.05 | **0.1** |
| `weight_mi` | 0.001 | 0.003 | 0.005 | **0.01** |
| `weight_perceptual` | 0.1 | 0.03 | 0.05 | 0.05 |
| `use_fpn` | yes | yes | no | no |
| `batch_size` | 6 | 6 | 10 | 10 |
| `lr_vae` | 6.7e-5 | 6.7e-5 | 1e-4 | 1e-4 |
| `kl_warmup_epochs` | 30 | 30 | 10 | 10 |
| `half_precision` | fp32 | fp32 | bf16 | bf16 |
| Epochs target | 200 | 200 | 100 | 100 |

KL_inactive estimated as: `0.5 * (sigma^2 - 1 - log(sigma^2))` with mu=0.

### Full Results Table

| Run | Epoch | Probe AUC (cardio) | Probe AUC (effusion) | Probe AUC (mean) | Cross-head score ↓ | Silhouette (disease-only) |
|---|---|---|---|---|---|---|
| disentangle-A | 192 | N/A (old script) | N/A | N/A | N/A | 0.020 |
| disentangle-C/D/E — best checkpoint | ~185 | **0.776** | **0.772** | **0.774** | 0.914 | — |
| disentangle-C/D/E — epoch 101 | 101 | 0.758 | 0.700 | 0.729 | 0.952 | 0.079 |
| disentangle-C/D/E — last valid (ep199) | 199 | 0.731 | 0.744 | 0.737 | 0.935 | — |
| inactivity-G — best checkpoint | ~50 | 0.639 | 0.627 | 0.650 | **0.744** | — |
| inactivity-G — epoch 100 (final) | 100 | 0.476 | 0.767 | 0.622 | **0.809** | 0.079 |
| independence-I — early peak (~ep25) | ~25 | 0.762 | 0.737 | 0.750 | 0.820 | — |
| independence-I — epoch 100 (final) | 100 | 0.532 | 0.812 | 0.672 | 0.871 | 0.124 |

**Cross-head score interpretation:** 1.0 = total leakage (each head predicts the other disease as well as itself); 0.5 = ideal (both heads carry no information about the other disease).

### KL State at Epoch 100 (disease heads)

| Run | KL_cardio | KL_effusion | Status vs free_bits |
|---|---|---|---|
| inactivity-G | ~2.7 (active) / ~2.5 (inactive) | similar | Above free_bits=1 — gradients flowing |
| independence-I | 4.00 (clamped) | 4.00 (clamped) | All below free_bits=2 — **DEAD ZONE** |

### Group-Level Consensus

#### Group 1: `sepvae_disentangle` (Feb 16–17, up to 200 epochs)

**Approach:** Moderate regularization, FPN encoder, slow LR (`6.7e-5`), long KL warmup (30 epochs).

**Findings:**
- **Highest peak performance** — probe AUC mean reaches 0.774 at around epoch 185. Best single-head cardiomegaly AUC = 0.776.
- **Highly unstable** — probe_auc fluctuates from 0.53 to 0.77 across consecutive evaluations within the same run. This is partly eval noise (small 600-sample eval set) and partly genuine training oscillation.
- **NaN explosion at epoch 200** — total loss was stable (0.004–0.006) through epoch 199 before blowing up in the final epoch. The run uses full fp32 precision; the NaN likely reflects a catastrophic gradient from an unlucky batch amplified over time without any LR decay. No LR schedule was used.
- **Persistent high leakage** — cross_head_score stays between 0.85 and 1.01 throughout 200 epochs. The heads learn some disease information but never achieve genuine independence.

#### Group 2: `sepvae_inactivity_driven` (Feb 20, 100 epochs)

**Approach:** Aggressive inactivity penalty — `sigma_inactive=0.05` (KL_inactive ≈ 2.5 nats), 5× stronger nulling (`weight_null=0.05`), no FPN, bf16.

**Findings:**
- **Best cross-head leakage suppression** — reaches cross_head_score = 0.744 mid-training and ends at 0.809, the best of all runs.
- **Weaker but more consistent probe AUC** — peaks around 0.65 (much lower than the 200-epoch disentangle chain), with less volatility.
- **Cardiomegaly head collapse by epoch 100** — final cardiomegaly probe AUC = 0.476 (sub-chance). The strong nulling pressure, when applied consistently over 100 epochs, appears to over-suppress the cardiomegaly head specifically. Cardiomegaly is a subtler, more distributed feature than effusion, so the inactivity pressure wins over the information signal.
- **Key trade-off:** reduces leakage at the cost of one head dying. Shows the right direction but the regularization balance is off.

#### Group 3: `sepvae_independence_driven` (Feb 20, 100 epochs)

**Approach:** Same architecture as Group 2, but `free_bits=2`, weaker nulling (`0.01`), stronger orthogonality (`0.1`).

**Findings:**
- **Confirmed `free_bits` / `sigma_inactive` conflict** — KL_inactive ≈ 1.8 nats < free_bits = 2.0. Disease heads are permanently in the KL dead zone. The free_bits threshold was intended to prevent posterior collapse on common-head channels but instead kills disease-head gradients entirely.
- **Interesting early dynamics** — probe AUC = 0.750 with cross_head_score = 0.820 around epoch 20–25. Some useful structure emerges before the disease heads degrade.
- **Full posterior collapse by epoch 100** — both disease head mu-norms ≈ 0.025–0.030 (essentially zero). KL uniformly 1.7–1.8 nats across all channels and classes (no class-conditional differentiation). Cardiomegaly probe AUC = 0.532 (near chance).
- **Worst configuration** — the `free_bits=2` setting is directly responsible.

---

*End of Chapter 03. Continue to [Chapter 04: Diagnoses and Patterns](04_diagnoses_and_patterns.md).*
