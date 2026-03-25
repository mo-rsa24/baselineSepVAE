# Composability Thesis Framing

---

## The Scope of the Argument: Hierarchy of Claims

The claim is not "everyone should train factorized VAEs" — that is not scalable or meaningful outside structured domains. The actual hierarchy of claims is:

```
1. Score composition requires conditional independence (theoretical)
        ↓
2. Independence cannot be recovered post-hoc from entangled pretrained models (negative result)
        ↓
3. In domains with known causal factor structure, independence CAN and SHOULD
   be enforced during training (positive result)
        ↓
4. Medical imaging is exactly such a domain — disease labels define factors,
   biological mechanisms justify independence, comorbidity is confounding not causation
        ↓
5. SepVAE instantiates this correctly; pretrained baselines cannot match it on
   semantically coherent pathology composition by design, not by scale
```

The approach **is** context-specific. It requires:
- A well-defined, finite factor vocabulary (pathology labels)
- Supervision signal (annotations or weak labels)
- Domain knowledge that factors are approximately causally independent

This is not a weakness to hide — it is a **scope condition** that makes the claim precise and defensible. The argument is strongest stated as:

> *For medical image synthesis tasks where compositional control over pathological factors is required, post-hoc approaches applied to general pretrained diffusion models are fundamentally insufficient because independence cannot be imposed after training. Explicit factorization during training, guided by domain-specific causal structure, is necessary and sufficient for principled composition.*

---

## How Flux and DALL-E 3 Handle Complex Scenes

It is **not** factorization. The mechanisms are:

1. **DALL-E 3 — Recaptioning at scale**: GPT-4 was used to rewrite all training captions into dense, descriptive, faithful descriptions. The model never had to learn to compose — it learned to match highly specific descriptions to images where every semantic element was explicitly present. Composition is implicit in the data.

2. **Flux (DiT architecture) — Joint attention**: Full bidirectional joint attention between image patches and text tokens — not U-Net cross-attention. This means text and image tokens attend to each other at every layer, allowing much richer spatial grounding. Combined with T5-XXL (not CLIP ViT-L), the model processes relational structure in language much more faithfully.

3. **In both cases — Coverage, not composition**: The model is doing **pattern matching at unprecedented scale** against a training set that happened to contain most realistic combinations. They fail precisely at *counterfactual* combinations and *precise attribute binding* — the things the medical domain requires.

> **They are not solving composition. They are solving coverage.**

Their failure modes (wrong attribute binding, object count errors, contradictory attributes) are exactly the failure modes of learned co-occurrence, not learned independence.

---

## The Core Reframing

The previous framing — "can we improve semantic composition over monolithic prompts via grounded logical decomposition" — is fighting the wrong battle. Flux already won that battle for natural images.

**Foundation models achieve compositional coverage through scale. The medical domain requires compositional control through structure.**

Coverage and control are different. Flux can generate "two dogs with pink bow ties" because it saw enough dogs and bow ties. It cannot generate "cardiomegaly with no effusion, mild left atelectasis, and normal right hemidiaphragm" with diagnostic fidelity — not because of scale, but because:
- No amount of training data gives it pathology-level supervision
- Prompt engineering cannot enforce medical independence constraints
- Coverage of rare disease combinations does not exist in internet data

The value of this work is in **exact counterfactual control** over medically meaningful factors — generating specific combinations that do not appear in training data, with correct independence structure. That is a capability no scaling of DALL-E 3 or Flux delivers.

---

## Why the Supervisor's Structure Has a Problem

The supervisor's RH2 (Riemannian/geodesic composition as the solution) is **directly contradicted by the proposal itself**:

> *"Even if we perfectly model the data manifold and compose along geodesics... you get cleaner, more realistic hybrids, but not logically correct compositions."*

You cannot propose geodesic composition as the solution in RH2 and then refute it in the same document. The self-refutation section is actually the most theoretically sharp part of the proposal, and the supervisor's structure discards it.

| | Supervisor | Revised Proposal |
|---|---|---|
| RH2 claim | Geodesic composition closes the gap | Geometry is the *wrong diagnosis*; factorization is the right one |
| RH2 evidence | Show geodesic > linear composition | Show factorized model closes gap; geodesic doesn't |
| Theoretical stance | Fix the inference | Fix the representation |
| Novelty | Incremental (better manifold navigation) | Reframes the problem class |

---

## The Critical Design Requirement for miniSepVAE

For RH2 to be a real hypothesis test, miniSepVAE needs a controlled ablation — not just showing factorized models work, but showing:

```
(A) Entangled model + linear score addition       → gap exists
(B) Entangled model + geodesic composition        → gap still exists
(C) Factorized model + linear score addition      → gap closes
(D) Factorized model + geodesic composition       → gap closes (same as C, confirms geometry is not the variable)
```

Case (B) is what makes the thesis rigorous. Without it, you have shown factorization helps but not that geometry fails. The supervisor's RH2 would predict (B) closes the gap. Your RH2 predicts it does not. **That is a falsifiable, publishable claim.**

The full 2×2 table makes the design airtight:

| | Entangled model | Factorized model |
|---|---|---|
| **Linear composition** | (A) gap exists | (C) gap closes |
| **Geodesic composition** | (B) gap persists | (D) gap closes (same as C) |

The critical cells are (B) and (C). If (B) persists and (C) closes, you have a clean double dissociation: composition rule does not matter if the representation is wrong; representation determines the outcome regardless of composition rule.

---

### The Design Is Right, and Here Is Why Case (B) Is the Linchpin

**What "geodesic composition" operationally means:**

There are two distinct senses in which geodesic composition could be implemented:

1. **Pullback metric geodesic in latent space** — Given the decoder $p_\theta(x \mid z)$, define the Riemannian metric on $\mathcal{Z}$ as the pullback of pixel-space Euclidean metric through the Jacobian:
   $$G(z) = J_\theta(z)^\top J_\theta(z)$$
   Compose factors by geodesic interpolation under $G$: rather than $z_{A+B} = z_A + z_B - z_0$ (Euclidean), use the exponential map $\exp_{z_0}(v_A + v_B)$ where $v_A, v_B$ are tangent vectors in $T_{z_0}\mathcal{Z}$.

2. **Riemannian score-space composition** — Instead of summing score functions ($\nabla_x \log p(x|A) + \nabla_x \log p(x|B)$), parallel-transport each score vector to a common reference point on the manifold before summing. Ensures the score vectors live in the same tangent space.

Both are meaningful geometric refinements. Neither fixes entanglement.

**Why Case (B) must fail regardless of which geodesic variant is used:**

The composability gap has two components:
1. **Geometric component**: Euclidean score summation is inconsistent with curved data manifold → geodesic composition fixes this.
2. **Statistical component**: $p(x \mid A)$ and $p(x \mid B)$ are observational conditionals that have absorbed co-occurrence statistics from the training data → **neither geodesic variant touches this**.

When you compute the geodesic between the composition of two entangled conditionals, you are computing the geometrically-consistent path between two statistically-wrong starting points. The geodesic is correct relative to the manifold but incorrect relative to the causal query "what does A+B look like independent of co-occurrence?"

Formally: if the entangled score $\nabla_x \log p(x \mid A)$ encodes "cardiomegaly features **plus** co-occurring effusion at rate $r$", then parallel-transporting this score to a different manifold point preserves its direction — including the spurious effusion component. The geometric operation does not filter for causal vs. observational content.

> The bias lives in the content of the score functions, upstream of any composition operation. Changing the composition operation (linear → geodesic) changes how you combine biased measurements but does not remove the bias.

**The SLERP nuance:**

SLERP (spherical linear interpolation) is a tractable approximation to pullback-metric geodesics, valid when $G(z) \approx \lambda I$ (conformally flat latent space). For the miniSepVAE toy domains at 64×64, this is a reasonable first approximation. The implementation:

```python
def slerp(z0, z1, t):
    """Spherical interpolation in latent space."""
    omega = torch.acos((z0 * z1).sum(-1, keepdim=True).clamp(-1, 1)
                       / (z0.norm(dim=-1, keepdim=True) * z1.norm(dim=-1, keepdim=True)))
    return (torch.sin((1-t)*omega) * z0 + torch.sin(t*omega) * z1) / torch.sin(omega)

def geodesic_compose(z_base, z_f1, z_f2, t1=1.0, t2=1.0):
    """Compose two factor latents via geodesic transport from base."""
    v1 = slerp(z_base, z_f1, t1) - z_base   # tangent vector toward factor 1
    v2 = slerp(z_base, z_f2, t2) - z_base   # tangent vector toward factor 2
    return z_base + v1 + v2                  # Euclidean sum in tangent space at z_base
```

For robustness, implement both SLERP and the full pullback-metric version (via finite-difference Jacobian) and confirm Case (B) fails under both. If (B) fails for SLERP but somehow closes with the pullback metric, that is an important qualification — but theoretically, the independence violation argument holds for both.

**Practical implementation note:**

No new training is required. The ablation is purely at evaluation time:

```python
# In train_generalize.py evaluate_composition_stamps() / evaluate_composition_texture()
def evaluate_composition(model, val_ds, compose='linear'):
    """compose ∈ {'linear', 'geodesic_slerp', 'geodesic_pullback'}"""
    ...
    if compose == 'linear':
        z_composed = z_base + z_f1_delta + z_f2_delta
    elif compose == 'geodesic_slerp':
        z_composed = geodesic_compose(z_base, z_f1_active, z_f2_active)
    ...
```

Add `--compose {linear,geodesic_slerp}` flag to `run.train_generalize`. The four conditions (A/B/C/D) are then:

```bash
# (A) Entangled + linear
python -m run.train_generalize --domain B --stage d0 --compose linear

# (B) Entangled + geodesic  ← the critical falsification case
python -m run.train_generalize --domain B --stage d0 --compose geodesic_slerp

# (C) Factorized + linear
python -m run.train_generalize --domain B --stage d5 --compose linear

# (D) Factorized + geodesic  ← confirms representation is the variable
python -m run.train_generalize --domain B --stage d5 --compose geodesic_slerp
```

Domain B (CorrelatedStamps, correlation_rate=0.30) is the right test bed because its entanglement is moderate and recoverable — strong enough to show Case (B) fails, not so extreme that it trivially collapses everything.

---

## One Genuine Risk to Flag

The supervisor may push back that RH2 is "just a theoretical observation" without a strong **methodological contribution**.

The response needs to be ready: the method **is** the factorized VAE design — the independence enforcement objective, the separation of disease-specific and shared latents, and the training procedure. miniSepVAE is the proof-of-concept and RH3 (CXR) is the demonstration that this works under real-world conditions where independence is approximate and medically motivated rather than mathematically exact.

That distinction — **approximate clinical independence is sufficient** — is what makes RH3 non-trivial and not just an application chapter.

---

## Suggested Revised Structure

**RH1 — Characterize the Problem**
The composability gap is characterized by off-manifold drift proportional to relational dependency between concepts. Mutual information between factors predicts composition failure.

**RH2 — Reframe the Diagnosis**
The gap is not a geometric problem and cannot be closed by manifold-aware inference. It is a representation problem: only explicit factorization with independence constraints during training closes the gap.
*(miniSepVAE as controlled experiment: cases A, B, C above)*

**RH3 — Real-World Validation**
Approximate clinical independence — grounded in anatomical and biological structure — is a sufficient condition for factorized composition in real-world medical imaging, enabling controlled synthesis of rare co-morbidities never seen during training.
*(CXR project)*

---

### Narrative Summary

This is a coherent, falsifiable, and honest thesis. It has a clean story:

> *You found the right diagnosis (factorization) where others found the wrong one (geometry). The composability gap is not a navigation problem — it is a representation problem. And you have both the theoretical argument and the controlled experiment to prove it.*
