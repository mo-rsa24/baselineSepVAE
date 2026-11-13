# The Composability Gap: A Precise Diagnosis

## What Logical Composition Assumes

Score-based compositional generation relies on:

$$
\nabla_x \log p(x \mid A, B) = \nabla_x \log p(x \mid A) + \nabla_x \log p(x \mid B) - \nabla_x \log p(x)
$$

This is valid *if and only if*:

$$
p(A, B \mid x) = p(A \mid x) \cdot p(B \mid x) \quad \text{[conditional independence given } x\text{]}
$$

Which itself holds *if and only if* the factors $A$, $B$ are **statistically independent in the data-generating process**, and the learned conditionals faithfully reflect this.

Neither condition holds in practice. This is not a diffusion mechanics failure. It is a **representational statistics failure** — the score functions are wrong before composition even begins.

---

## Why Individual Conditionals Are Already Entangled

When you train $p(x \mid \text{cardiomegaly})$ on a real dataset:

- Cardiomegaly co-occurs with pleural effusion, pulmonary edema, and specific patient demographics at non-trivial rates.
- The learned conditional absorbs these co-occurrence statistics — it cannot distinguish "features caused by cardiomegaly" from "features that happen to appear alongside cardiomegaly in the training marginal."
- The conditional is therefore not $p(x \mid do(\text{cardiomegaly}=1))$ in the Pearl sense. It is $p(x \mid \text{cardiomegaly}=1)$, which is an *observational* conditional, not an *interventional* one.

Composing entangled conditionals produces three simultaneous failure modes:

| Failure Mode | Mechanism | Symptom |
|---|---|---|
| **Double-counting** | Features co-occurring with both A and B get scored twice | Over-expression artifacts, exaggerated anatomy |
| **Competing gradients** | Dimensions where $p(x \mid A)$ and $p(x \mid B)$ have contradictory curvature produce conflicting score updates | Incoherent, averaged-out outputs |
| **Missing interaction effects** | Joint effects only present in $p(x \mid A, B)$ are absent from either marginal | Under-expression of comorbid pathology |

The third failure mode is the hardest: it is a fundamental limitation of product-of-experts approaches that persists even after the first two are resolved.

---

## Is This Primarily a Latent Design Problem?

**Yes, with a specific qualification.**

The composability gap is a symptom of violated independence at the representation level. The causal chain is:

```
Entangled training co-occurrences
    → Learned conditionals absorb shared information
        → Independence assumption violated
            → Score composition incoherent
                → Composability gap
```

Intervening at the score-composition step (step 4) treats a symptom.
Intervening at the representation step (steps 1–2) treats the cause.

### What Latent Disentanglement Actually Does

If you enforce $z_A \perp z_B$ by construction via MI minimisation + hard-zero nulling + tight conditional KL, then:

$$
p(z_A, z_B) \approx p(z_A) \cdot p(z_B) \quad \text{[enforced, not assumed]}
$$

Score composition in *latent space* becomes:

$$
\nabla_z \log p(z \mid A, B) = \nabla_z \log p(z_A \mid A) + \nabla_z \log p(z_B \mid B) + \nabla_z \log p(z_0)
$$

This is **exactly valid** — not approximately valid, not valid-with-guidance-tricks — because the independence is a hard architectural constraint on the encoder, not a modelling assumption about the data. Composability is a latent design problem because only latent design can enforce the assumption that composition requires.

---

## What Riemannian Geometry Does and Doesn't Address

Two separate problems must be distinguished:

### Problem 1 — Geometric consistency of composition (what Riemannian geometry addresses)

Euclidean score summation assumes the latent space is flat: that adding gradient vectors from different conditionals produces a result pointing toward a coherent region. On a curved manifold this is wrong — parallel transport matters, geodesic distances matter, the exponential map from one point is not the same as from another.

Riemannian approaches (Riemannian score matching, geodesic interpolation, diffusion on Lie groups) genuinely address this. They make composition operations *geometrically consistent* with the true manifold structure of the data.

### Problem 2 — Statistical independence of learned conditionals (what Riemannian geometry does NOT address)

The independence violation is in the *probability measure* attached to the manifold, not in the manifold's metric structure. Formally: if the fiber over $x$ carries entangled conditional information, changing the Riemannian metric on the base manifold does not change the fiber statistics.

> **Analogy:** If two biased sensors measure the same underlying signal, aggregating their readings with a more sophisticated geometry (Riemannian mean instead of arithmetic mean) produces a geometrically consistent aggregate of two biased measurements — but does not remove the bias. The bias lives in the measurement process, upstream of aggregation.

Riemannian geometry solves a real problem (composition on curved data manifolds) but **leaves the underlying independence violations entirely unresolved**.

---

## The Residual Hard Problem: Non-Additive Interaction Effects

Even with perfect disentanglement ($z_A \perp z_B$ enforced), a fundamental limitation remains:

$$
p(x \mid A, B) = p(x \mid A) \cdot p(x \mid B) \cdot \underbrace{\text{correction}(A, B)}_{\text{silently set to 1}}
$$

The `correction(A, B)` term captures non-additive interaction effects — features produced only by the *joint presence* of both factors and inferable from neither marginal alone. In the CXR setting: cardiomegaly + pleural effusion together may produce hemodynamic patterns in the lung fields that neither condition produces in isolation.

Product-of-experts composition sets $\text{correction}(A, B) = 1$. This is correct if factors affect genuinely separate pathways with no interaction. It is incorrect when factors interact at the image level even if represented independently at the latent level.

This is where the SepVAE framework's validity depends on a **domain assumption**: cardiomegaly and pleural thickening affect spatially and physiologically separable image regions with negligible joint interaction effects. For CXR this is approximately true. For settings with substantial factor interactions, this residual gap persists even after perfect disentanglement.

---

## The Coherent Picture

```
Composability gap
├── Primary cause: Independence assumption violated in learned conditionals
│   └── Fix: Latent disentanglement (MI loss, hard-zero nulling, tight KL)
│       → Makes independence true by construction in z-space
│       → Composition in latent space becomes exactly valid
│
├── Secondary cause: Geometric inconsistency in score composition
│   └── Fix: Riemannian geometry, geodesic composition, Lie group structure
│       → Addresses real but secondary manifold structure problem
│       → Does NOT fix statistical entanglement
│
└── Residual irreducible component: Non-additive factor interaction effects
    └── Fix: Requires joint training data (defeats the purpose)
        → Or: domain knowledge that interactions are negligible (SepVAE's assumption)
        → Or: explicit interaction modelling with sparse comorbid data
```

---

## Practical Implication for the SepVAE / LDM Pipeline

The hierarchy is:

1. **Load-bearing**: Getting disentanglement right (D0→D3). The correctness of compositional generation depends entirely on whether the encoder produced orthogonal factor latents.

2. **Important but secondary**: Reconstruction quality (D4→D5). Affects downstream LDM sample quality but not compositional validity.

3. **Tertiary**: Geometric choices in the LDM (DDPM vs flow matching vs SDE, Euclidean vs Riemannian transport). Affect transport quality given correct score functions, but the correctness of the score functions is determined by the latent encoder — not the diffusion mechanics.

Riemannian approaches are worth attention if the latent manifold is highly curved (likely at 512×512 scale), but as a second-order correction on top of solved disentanglement, not as a substitute for it.
