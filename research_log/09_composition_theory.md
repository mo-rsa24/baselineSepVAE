# Chapter 09 — Composition Theory: Assumptions and Strategies

**Previous chapter:** [08 Full Sweeps](08_full_sweeps.md)
**Next chapter:** [10 Verification and Hypothesis](10_verification_and_hypothesis.md)

---

## 10. Research Theory: Assumptions and Composition Strategies

*This section documents the fundamental assumptions of the project and the three candidate strategies for downstream composition. Read this before designing any LDM training experiment.*

### 10.1 Three core assumptions

#### A1 — Disease factors are approximately separable

$$z_{\text{cardiomegaly}} \perp z_{\text{effusion}}$$

In chest X-rays this is not strictly true. Cardiomegaly and effusion co-occur through shared pathology (right heart failure causes both), shared appearance patterns, and dataset sampling bias. `exclude_cross_disease_overlap=True` removes co-occurring patients from training but creates distribution mismatch at inference when we compose both.

**Evidence from runs:** Cross_head_score never below 0.74. Assumption is violated to a significant degree.

#### A2 — Radiologist labels correspond to clean generative factors

Cardiomegaly is not a primitive visual atom — it is a high-level finding that shifts the mediastinum, displaces the lungs, and creates global shape changes that look like "anatomy variation" to any encoder. The model routes some of this into $z_{\text{common}}$.

**Evidence from runs:** Cardiomegaly probe AUC is consistently lower and more volatile than effusion AUC.

#### A3 — Separated representation implies separable score fields

Even with clean factorised latents, separately trained LDMs will double-count shared content unless the composition strategy accounts for it:

$$p(z \mid c, e) \propto \frac{p(z \mid c)\, p(z \mid e)}{p(z)}$$

If each disease LDM was trained on the full $z$ (including $z_{\text{common}}$), both carry anatomical structure in their scores. Naive addition double-counts anatomy:

$$\nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) = \nabla_z \log p(z \mid c, e) + \nabla_z \log p(z)$$

The extra $\nabla_z \log p(z)$ over-reinforces common structure and produces physically inconsistent compositions.

---

### 10.2 Three candidate composition strategies

#### Strategy A — Sub-block conditional LDMs (RECOMMENDED)

$$p(z) = p(z_{\text{common}}) \cdot p(z_{\text{cardio}} \mid z_{\text{common}}) \cdot p(z_{\text{effusion}} \mid z_{\text{common}})$$

```
Inference (both diseases):

    LDM_common  ──→  z_common ──────────────────────┐
                         │                           │
                         ├──→ LDM_cardio(z_common) ──→ z_cardio   ──┐
                         │                                           │
                         └──→ LDM_effusion(z_common) → z_effusion ──┤
                                                                     │
                    Decoder(z_common, z_cardio, z_effusion) ←────────┘
                         │
                         ▼
                  Composed CXR (cardio + effusion)
```

Each LDM acts on a **different disjoint subspace** — no score addition, no double-counting. Composition is a sequential ancestral sample. Requires: $z_{\text{cardio}} \perp\!\!\!\perp z_{\text{effusion}} \mid z_{\text{common}}$ — exactly what R5b tests and enforces.

#### Strategy B — Full-z score composition (fallback)

$$\nabla_z \log p(z \mid c, e) \approx \nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) - \nabla_z \log p(z)$$

Requires three LDMs and near-perfect disentanglement. The prior subtraction corrects double-counting only if the disease LDMs are truly independent of each other's content in $z$.

#### Strategy C — Sub-block scores only (broken with current decoder)

Strategy C assumes the decoder treats each sub-block as an independent additive perturbation. The current SepVAE decoder concatenates all heads and applies shared convolutions — it mixes all channels. The decoder Jacobian is not block-diagonal. Strategy C is geometrically unsound for this architecture.

#### CFG with unconditional = common expert (Addition P4)

If a single conditional LDM is trained with CFG dropout, the unconditional direction = $z_{\text{common}}$-only output. Composition becomes:

$$\nabla_z \log p^{\text{guided}}(z \mid c, e) = \nabla_z \log p(z \mid \varnothing) + \lambda[\nabla_z \log p(z \mid c) - \nabla_z \log p(z \mid \varnothing)] + \lambda[\nabla_z \log p(z \mid e) - \nabla_z \log p(z \mid \varnothing)]$$

Double-counting is prevented: both guidance directions are subtracted against the same common base. Architecturally simpler than Strategy A (one LDM to train). Natural extension if a joint conditional LDM is trained as the ablation baseline.

| | Strategy A | Strategy B | Strategy C | CFG (P4) |
|--|------------|------------|------------|----------|
| Double-counts anatomy? | No | Requires prior subtraction | No (if block-diagonal decoder) | No |
| Compatible with current decoder | Yes | Yes | **No** | Yes |
| Required disentanglement level | Moderate ($z_d \perp z_d \mid z_c$) | High ($z \perp z$ globally) | High (block-diagonal decoder Jacobian) | Moderate (same as A) |
| Needs unconditional LDM | Yes (`LDM_common`) | Yes (prior $p(z)$) | Yes (`base LDM`) | No (CFG dropout replaces it) |
| Architecturally consistent with current decoder | Yes | Yes | **No** | Yes |
| Implementation complexity | Medium (3 LDMs + cross-attention conditioning) | Medium (3 LDMs + score arithmetic) | High (requires decoder redesign) | Low (1 LDM + CFG dropout) |
| Recommended | **Yes** | Fallback | No | Yes (as variant) |

---

### 10.3 Revised, falsifiable research hypothesis

> **If disease-related variation can be approximately factorised from shared anatomical and acquisition variation in the SepVAE latent space** — as measured by probe AUC > 0.75 per head, cross-head score < 0.65, and edit purity ratio > 2.0 — **then conditional sequential sampling from disease-specific sub-block LDMs conditioned on $z_{\text{common}}$ (Strategy A) will better approximate multi-pathology comorbid synthesis**, as measured by:
> 1. Lower FID against held-out comorbid images vs. a jointly-trained conditional baseline
> 2. Higher dual-disease classifier confidence on synthesised images
> 3. Higher anatomical consistency (SSIM on non-disease regions vs. single-disease reference)

**Fallback claim (if cardiomegaly remains unfactorisable):**
Score composition achieves plausible multi-pathology generation even when factorisation is imperfect, whereas direct joint conditioning fails to generalise to the comorbid case due to data sparsity.

---

## Supplementary: Assumptions with Evidence, Risks, and Full Strategy Mathematics

*The following provides deeper mathematical treatment of each assumption and each composition strategy.*

### Assumption A1 — Disease factors are approximately independent: risks and evidence

**Why it may not hold in VinBigData:**
- Cardiomegaly and pleural effusion co-occur via shared pathology (e.g., right heart failure causes both)
- Shared appearance patterns: enlarged cardiac silhouette pushes against the pleural space, creating apparent blunting at the costophrenic angles
- Projection effects: AP vs PA positioning changes how both conditions manifest
- Dataset sampling bias: patients admitted to hospital with one condition are more likely to have the other

**Mitigation implemented:** `exclude_cross_disease_overlap=true` removes all co-occurring patients from the triplet dataset, so the VAE never sees both simultaneously. However, this creates a distribution mismatch at inference when we want to synthesise a comorbid image.

**Evidence from runs:** Cross-head score consistently above 0.74, suggesting the representations are not fully independent even after training.

### Assumption A2 — Labels correspond to clean generative factors: risks and evidence

A clinical label like "cardiomegaly" is a high-level diagnostic summary, not a primitive visual atom. The cardiac silhouette enlarges, but it also:
- Shifts the mediastinum
- Displaces the lungs
- Creates global shape changes that look like "common anatomy" to any encoder

The model may place some cardiomegaly-relevant variation into $z_{\text{common}}$ because that signal *is* correlated with anatomy from the encoder's perspective.

**Evidence from runs:** `probe_auc/cardiomegaly` is consistently lower and more unstable than `probe_auc/effusion`. Effusion is a localised, high-contrast finding (bright pleural fluid) that is easier to separate from background anatomy. The backbone (CheSS) was trained on CXR but not with a supervision signal that separates cardiac pathology from normal cardiac anatomy.

**Longer-term fix:** Fine-tune the backbone with a cardiac-specific contrastive objective, or use a dedicated cardiomegaly-supervised feature extractor.

### Assumption A3 — Separated representation implies separable score fields: the mathematical risk

Score addition works as a product of experts:

$$p(z \mid c, e) \propto \frac{p(z \mid c) \cdot p(z \mid e)}{p(z)}$$

This identity holds under the conditional independence assumption $p(z \mid c, e) = \frac{p(z \mid c) p(z \mid e)}{p(z)}$, which requires:

$$p(z \mid c) \perp p(z \mid e) \mid z_{\text{common}}$$

If each disease LDM is trained on the *full* z (including $z_{\text{common}}$), they will both encode anatomical structure, and naïvely adding their scores double-counts the shared content:

$$\underbrace{\nabla_z \log p(z \mid c)}_{\text{anatomy + cardio}} + \underbrace{\nabla_z \log p(z \mid e)}_{\text{anatomy + effusion}} - \underbrace{\nabla_z \log p(z)}_{\text{anatomy once}} = \underbrace{\nabla_z \log p(z \mid c, e)}_{\text{target}} + \underbrace{\nabla_z \log p(z)}_{\text{extra anatomy term}}$$

The extra $\nabla_z \log p(z)$ over-sharpens anatomical structure and can cause inconsistencies at the cardiac-pleural interface.

**This is the central mathematical risk of the project.** Strategy A avoids it by design.

---

### Strategy A — Full mathematical treatment

**Factorised generative process:**

$$p(z) = p(z_{\text{common}}) \cdot p(z_{\text{cardio}} \mid z_{\text{common}}) \cdot p(z_{\text{effusion}} \mid z_{\text{common}})$$

**Inference (both diseases simultaneously):**

1. Sample $z_{\text{common}} \sim \text{LDM}_{\text{common}}$
2. Sample $z_{\text{cardio}} \sim \text{LDM}_{\text{cardio}}(z_{\text{common}})$
3. Sample $z_{\text{effusion}} \sim \text{LDM}_{\text{effusion}}(z_{\text{common}})$
4. Decode $\hat{x} = \text{Dec}(z_{\text{common}}, z_{\text{cardio}}, z_{\text{effusion}})$

```
Strategy A — Inference diagram:

    [LDM_common]
         │
    z_common ──────────────────────────┐
         │                             │
         ├──→ [LDM_cardio(z_common)] → z_cardio    │
         │                             │
         └──→ [LDM_effusion(z_common)] → z_effusion │
                                        │
              ┌─────────────────────────┘
              ↓
     Decoder(z_common, z_cardio, z_effusion)
              │
              ↓
       Composed CXR (cardio + effusion)
```

**Why this avoids the double-counting problem:** Each LDM operates on a *different* subspace. There is no score addition; the composition is a sequential ancestral sample from the factorised joint. The conditional independence assumption $z_{\text{cardio}} \perp z_{\text{effusion}} \mid z_{\text{common}}$ is what the SepVAE training must enforce — and this is exactly what R5a and R5b are designed to provide.

**What the SepVAE must provide for this to work:**

$$z_{\text{cardio}} \perp\!\!\!\perp z_{\text{effusion}} \mid z_{\text{common}}$$

The cross-head adversarial discriminator (R5b) is the direct empirical test: if $D_{c \to e}(z_{\text{effusion}}) \approx 0.5$, then $z_{\text{effusion}}$ contains no cardiomegaly information beyond what is in $z_{\text{common}}$.

**Architecture for disease LDMs:**
```
LDM_common: p(z_common) — trained on z_common from normal images only
LDM_cardio: p(z_cardio | z_common) — z_common is the conditioning signal
LDM_effusion: p(z_effusion | z_common) — z_common is the conditioning signal
```

Conditioning on $z_{\text{common}}$ (a 4-channel spatial map) can be done via cross-attention or concatenation in the UNet denoiser, following standard LDM conditioning patterns already in the codebase.

---

### Strategy B — Full mathematical treatment

Train two full-z conditional LDMs and one unconditional LDM, then compose scores:

$$\nabla_z \log p(z \mid c, e) \approx \nabla_z \log p(z \mid c) + \nabla_z \log p(z \mid e) - \nabla_z \log p(z)$$

```
Strategy B — Score composition:

  ∇ log p(z|cardio) ──┐
                       ├──(+)──→ ∇ log p(z|both)
  ∇ log p(z|effusion)─┘
          │
  ∇ log p(z) ────────(−)──→ subtract prior once
```

**The mathematical concern:** The identity only holds when $c \perp e \mid z$. If $z_{\text{common}}$ leaks into $z_{\text{disease}}$ (as observed — cross-head score 0.74–0.87), both disease LDMs encode anatomy, and the prior subtraction does not compensate:

$$\underbrace{\nabla_z \log p(z \mid c)}_{\text{anatomy + cardio}} + \underbrace{\nabla_z \log p(z \mid e)}_{\text{anatomy + effusion}} - \underbrace{\nabla_z \log p(z)}_{\text{anatomy once}} = \underbrace{\nabla_z \log p(z \mid c, e)}_{\text{target}} + \underbrace{\nabla_z \log p(z)}_{\text{extra anatomy term}}$$

**Requires:** Much cleaner disentanglement (cross-head score approaching 0.5) than currently achieved.

---

### Strategy C — Why it is broken with the current decoder

Assume each disease LDM only acts on its own sub-block and $z_{\text{common}}$ is sampled from a separate base model:

$$\nabla_z \log p(z) = \underbrace{\nabla_{z_c} \log p(z_{\text{common}})}_{\text{base}} + \underbrace{\nabla_{z_d} \log p(z_{\text{cardio}})}_{\text{cardio sub-block}} + \underbrace{\nabla_{z_e} \log p(z_{\text{effusion}})}_{\text{effusion sub-block}}$$

```
Strategy C — Sub-block score decomposition:

   z = [ z_common | z_cardio | z_effusion ]
              ↑          ↑           ↑
         base LDM   cardio LDM  effusion LDM
        (score acts (score acts  (score acts
         on this     on this      on this
         block only) block only)  block only)
```

**Why this is broken:** The SepVAE decoder is a shared convolutional network that takes all heads concatenated. It does not treat disease sub-blocks as independent additive perturbations — the decoder mixes all channels before any convolution. Score gradients w.r.t. $z_{\text{cardio}}$ are entangled with $z_{\text{common}}$ and $z_{\text{effusion}}$ through the decoder Jacobian. Sub-block score independence requires a decoder with block-diagonal Jacobian (e.g., independent channel-wise decoders), which is not the current design.

---

### CFG Addition P4 — Full mathematical treatment

An elegant reformulation of Strategy B that avoids explicit score subtraction. Train one conditional LDM with disease labels, where the **unconditional guidance** direction starts from $z_{\text{common}}$-only (disease heads set to their prior $\mathcal{N}(0, \sigma_{\text{inactive}}^2)$):

$$\nabla_z \log p_\theta^{\text{guided}}(z \mid c, e) = \nabla_z \log p_\theta(z \mid \varnothing) + \lambda \left(\nabla_z \log p_\theta(z \mid c) - \nabla_z \log p_\theta(z \mid \varnothing)\right) + \lambda \left(\nabla_z \log p_\theta(z \mid e) - \nabla_z \log p_\theta(z \mid \varnothing)\right)$$

where $\varnothing$ denotes the unconditional (common-only) direction. Expanding:

$$= (1 - 2\lambda) \nabla_z \log p_\theta(z \mid \varnothing) + \lambda \nabla_z \log p_\theta(z \mid c) + \lambda \nabla_z \log p_\theta(z \mid e)$$

**Why this is natural given the architecture:** The SepVAE already defines what "unconditional" means — it is the output when disease heads are clamped to their inactive prior. The common expert is not an abstract baseline; it is the decoder output with $z_{\text{cardio}} = z_{\text{effusion}} = \mathbf{0}$.

**Relationship to Strategy A:** If the SepVAE is well-disentangled, Strategy A and CFG-composition should give approximately equivalent results. If they diverge, the divergence is diagnostic — it reveals how much cross-head information the disease LDMs learned.

**Implementation note:** requires training a single conditional LDM with CFG dropout (where the condition is zeroed with probability $p_\text{uncond}$ during training) and treating the unconditioned output as the common-expert baseline. It is architecturally simpler than Strategy A because there is only one LDM to train.

---

*End of Chapter 09. Continue to [Chapter 10: Verification and Hypothesis](10_verification_and_hypothesis.md).*
