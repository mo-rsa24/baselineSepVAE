# Chapter 04 — Root Cause Diagnoses and Cross-Run Patterns

**Previous chapter:** [03 Empirical Results](03_empirical_results.md)
**Next chapter:** [05 Fixes R1–R7](05_fixes_r1_to_r7.md)

---

## Supplementary: Detailed Root Cause Diagnoses

*The following section provides deeper diagnosis for each failure mode identified in Phase 2. These diagnoses directly motivated the fixes implemented in Phase 3 (R1–R6).*

### Diagnosis 1 — Independence-I: Free-bits dead zone

The central failure mechanism:

$$\text{KL}_{\text{inactive}}(\sigma=0.1) \approx 1.8 \text{ nats} < \text{free\_bits} = 2.0$$

The free-bits threshold clips the KL gradient to zero for any channel with $\text{KL} < \lambda_{\text{fb}}$. Because **all disease channels** are held near $\sigma_{\text{inactive}} = 0.1$, they all land below the threshold permanently.

```
KL_channel ≈ 1.8 nats
────────────────────────────────────────────────────
          ◄── dead zone ──►
    0 ──────────────────── 2.0 ──────────── ∞
                           ↑ free_bits = 2.0
          gradient = 0 here
```

The free-bits mechanism was designed to prevent posterior collapse in the *common* head by protecting low-KL channels from being over-penalised. But it applies globally and inadvertently kills disease-head gradients.

**Fix (R1):** Set `free_bits = 0.0` (removed entirely) with `sigma_inactive = 0.05` to achieve tighter inactive priors instead.

### Diagnosis 2 — Disentangle-E: NaN explosion at epoch 200

Without a learning rate schedule, accumulated small gradient errors over 200 epochs eventually produce a catastrophic update from one unlucky batch. Full fp32 precision increases the risk since there is no implicit gradient scaling. The total loss was stable (0.004–0.006) through epoch 199, then all terms went `nan` simultaneously.

**Fix (R3):** Cosine LR decay from `lr_decay_epochs` onward:

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot (t - t_{\text{decay}})}{T - t_{\text{decay}}}\right)\right)$$

with $\eta_{\min} = 0.1 \cdot \eta_0$.

### Diagnosis 3 — Inactivity-G: Cardiomegaly head collapse under heavy nulling

Strong `weight_null = 0.05` drives $\mu_{\text{cardio}} \to 0$ faster than the reconstruction loss can maintain the cardiomegaly signal, specifically because:
- Cardiomegaly is a distributed, subtle feature (enlarged cardiac silhouette) vs. the localised bright blobs of effusion
- CheSS backbone, trained primarily on normal anatomy, represents cardiomegaly as a weaker feature
- The nulling loss is symmetric — it cannot distinguish "this head should be inactive here" from "this head should be active here"

Final result: `probe_auc/cardiomegaly = 0.476` (below chance).

**Fix (R2):** Balance `weight_null = 0.02`, `weight_orthogonality = 0.02`, `sigma_inactive = 0.05`. Gentler pressure that does not overwhelm the cardiomegaly signal.

### Diagnosis 4 — Universal cross-head leakage

Across **all runs**, cross-head score never falls below 0.74. Even the best run ends at 0.809. The current loss portfolio — nulling + orthogonality + MI discriminator — enforces marginal statistical independence but not **conditional independence**. The heads can share information about the other disease without violating any of these constraints as long as their marginal distributions are uncorrelated.

```
What orthogonality enforces:
  𝔼[z_cardio] ⊥ 𝔼[z_effusion]   (prototype cosine penalty)
  Corr(z_cardio_i, z_effusion_j) ≈ 0   (Barlow-style)

What it does NOT enforce:
  z_cardio ⊥ y_effusion | z_common   (conditional independence of heads given anatomy)
```

**Fix (R5a, R5b):** Paired contrastive loss and cross-head adversarial discriminators — see [Chapter 05](05_fixes_r1_to_r7.md).

---

## Cross-Run Patterns — Extended Analysis

*The following patterns were synthesised across all runs and are recorded as guiding principles for future configuration choices.*

### Pattern 1: `free_bits` / `sigma_inactive` conflict is a hard failure

Only runs with `free_bits > KL_inactive` fail catastrophically. All runs with `free_bits ≤ 1` allow gradients to flow through disease channels. This is not a soft tradeoff — it is a binary failure condition that must be checked **before any other hyperparameter analysis**.

The constraint is:

$$\text{KL}_{\text{inactive}} = \frac{1}{2}\left(\sigma_{\text{inactive}}^2 - 1 - \log \sigma_{\text{inactive}}^2\right) > \text{free\_bits}$$

| $\sigma_{\text{inactive}}$ | $\text{KL}_{\text{inactive}}$ (nats) |
|---------------------------|--------------------------------------|
| 0.20 | ~1.1 |
| 0.10 | ~1.8 |
| 0.05 | ~2.5 |

### Pattern 2: Cross-head leakage is a universal unsolved problem

Cross_head_score never goes below 0.74 across any run at any point. The best leakage suppression (inactivity-G, 0.744) comes at the cost of cardiomegaly head collapse. The current regularisation set does not have a mechanism that directly enforces conditional independence between disease heads.

The orthogonality loss penalises the *output* of the heads (latent vectors) but says nothing about the *routing of input features* — both ConvHeads still see the same full 64×64 backbone feature map and must learn, purely through gradient pressure, to ignore the spatial region belonging to the other disease. This is why spatial attention (R7) was later introduced.

### Pattern 3: Cardiomegaly head is consistently harder to train

Across all runs, cardiomegaly probe AUC is lower, more volatile, and more susceptible to collapse than effusion AUC. Likely causes:
- Cardiomegaly is a subtle, globally distributed change; effusion is a local, high-contrast finding
- After `exclude_cross_disease_overlap=true`, effective cardiomegaly examples may be fewer
- CheSS backbone features do not explicitly separate cardiac silhouette from lung fields
- The nulling loss applies symmetric pressure regardless of signal strength

### Pattern 4: Probe AUC is noisy — rolling averages required

The same checkpoint evaluated at consecutive epochs shows probe AUC ranging from 0.53 to 0.77. The 600-sample probe eval has high variance. Single-epoch readings are unreliable; a **3-evaluation rolling average** should be the reporting standard going forward.

### Pattern 5: Long training improves peak but destabilises without LR decay

The 200-epoch chain achieves the highest peak probe AUC (0.774) but ends in NaN. LR decay was absent in all runs. The combination of long training + no decay + no gradient clip tightening creates a slow-moving instability that eventually triggers catastrophic failure.

**Implication:** A cosine decay schedule starting at epoch 60–80 of a 100–150 epoch run will likely preserve the plateau gains without causing NaN.

---

*End of Chapter 04. Continue to [Chapter 05: Fixes R1–R7](05_fixes_r1_to_r7.md).*
