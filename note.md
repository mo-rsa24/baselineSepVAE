> **Toy datasets are not simplifications—they are controlled environments where we deliberately introduce correlation, noise, and entanglement, and test whether our VAE + latent diffusion pipeline can recover the assumptions required for compositional modeling.**

---

# 🔷 1. Collective Goal

To model and approximate a **joint distribution over simultaneous events**:
[
p(x \mid c_1, \dots, c_k)
]

in a way that:

* Enables **composition of pretrained conditional diffusion models**
* Produces **valid samples on the data manifold**
* Preserves **multiple constraints simultaneously**
* Works even when real-world factors are:

  * correlated
  * noisy
  * entangled

---

## 🔥 Refined Goal (Your Contribution)

> Learn a representation (z) such that composition becomes valid **even when the original data does not satisfy the assumptions**

[
x \xrightarrow{\text{VAE}} z \quad \text{where} \quad z \text{ is more factorized than } x
]

---

# 🔷 2. Core Problem

We approximate:
[
\nabla_x \log p(x \mid c_1, \dots, c_k)
]

using:
[
\sum_i \nabla_x \log p(x \mid c_i) - (k-1)\nabla_x \log p(x)
]

---

## 🚨 Fundamental Difficulty

This only works if:

* constraints are **compatible**
* latent factors are **disentangled**
* score fields are **non-interfering**

---

## 🔥 Your Key Insight

> These assumptions are **not true in real data**, so we must **learn a representation where they become approximately true**

---

# 🔷 3. Dataset Hierarchy (with Your Extension)

## 🟢 Level 1 — Toy Geometry (Controlled, Low-Dim)

### Purpose

Test **probabilistic mechanics under controlled violations**

---

### 🔧 Critical Upgrade (Your Addition)

Toy datasets should be generated as:

[
(u_1, u_2, \dots, u_k) \sim p(u)
]
[
x = g(u) + \epsilon
]

Where:

* (u_i) = **ground-truth factors**
* (g) = **entangling transformation**
* (\epsilon) = noise

---

### What to Model

* correlation between factors
* non-orthogonality
* manifold curvature
* noise
* partial overlap

---

### What to Test

* Can VAE recover (u_i) from (x)?
* Does latent (z) improve factorization?
* Does composition work better in (z) than in (x)?

---

## 🟡 Level 2 — Structured Toy (MNIST variants)

### Purpose

Introduce **proto-semantics**

---

### What to Model

* object identity
* attributes (thickness, style)
* spatial position
* count

---

### Latent Structure

[
z = (z_{\text{digit}}, z_{\text{style}}, z_{\text{position}})
]

---

## 🔴 Level 3 — Real Data (CXR, CLEVR, T2I)

### Purpose

Full compositional complexity

---

### What to Model

* multiple diseases / objects
* spatial relationships
* co-occurrence patterns

---

# 🔷 4. What Must Be Localized / Disentangled

## 🎯 Core Requirement

Each condition (c_i) must act on a **separable component of the representation**

---

## 🧠 Across Dataset Levels

### 🟢 Toy

* coordinate subspaces
* manifold regions

---

### 🟡 MNIST

* digit vs style vs position

---

### 🔴 Real Data

* anatomy vs disease vs spatial structure

---

## 🔥 Key Principle

> Composition works when:
> [
> z = (z_1, \dots, z_k), \quad z_i \leftrightarrow c_i
> ]

---

# 🔷 5. Distribution Math (Operations)

## 🧩 CDM

[
p(x \mid c_1,\dots,c_k)
\propto \frac{\prod_i p(x \mid c_i)}{p(x)^{k-1}}
]

[
\nabla \log p_{\text{joint}} =
\sum_i \nabla \log p(x \mid c_i)
--------------------------------

(k-1)\nabla \log p(x)
]

---

## 🧩 Projective Composition

[
\nabla \log p_i = \nabla \log p_b + \Delta_i
]

[
\nabla \log p_{\text{comp}} =
\nabla \log p_b + \sum_i \Delta_i
]

---

## 🧩 CompBench View

[
p(\text{objects}, \text{attributes}, \text{relations} \mid x)
]

---

# 🔷 6. Dataset-Specific Differences

## 🟢 Toy Geometry

### Strengths

* exact densities
* visualizable scores
* controlled violations

### With Your Upgrade

✔ includes:

* correlation
* noise
* entanglement

---

### Weaknesses

* no true semantics
* no language grounding

---

## 🟡 MNIST

### Strengths

* structured objects
* attribute binding

### Weaknesses

* limited complexity

---

## 🔴 Real Data

### Strengths

* full compositional realism

### Weaknesses

* hard to interpret
* unknown true factors

---

# 🔷 7. Required Assumptions

## 🧩 CDM

[
c_i \perp c_j \mid x
]

---

## 🧩 Projective Composition

* factorized score effects
* weak interference

[
\langle \Delta_i, \Delta_j \rangle \approx 0
]

---

## 🧩 CompBench

* structured joint over entities

---

# 🔷 8. Mapping Toy Datasets (Revised with Your Intent)

## 1. Orthogonal / Near-Orthogonal Factors

### Now:

* latent: independent
* observed: slightly entangled

### Tests:

* can VAE recover factorization?
* does composition improve after encoding?

---

## 2. Rings + Mask

### Now:

* latent: radius + selector
* observed: distorted / noisy

### Tests:

* global vs local constraint separation

---

## 3. Intersecting Curves

### Now:

* latent: two correlated factors
* observed: ambiguous regions

### Tests:

* interference reduction via representation

---

## 4. Spiral / Curved Manifold

### Now:

* latent: simple parameterization
* observed: nonlinear embedding

### Tests:

* off-manifold drift
* latent flattening

---

# 🔷 9. What Should Be Measured

To stay faithful to the papers:

---

## Representation Quality

* reconstruction error
* factor predictability
* cross-factor leakage

---

## Independence / Factorization

* mutual information between (z_i)
* covariance / disentanglement metrics

---

## Score Behavior

* alignment:
  [
  \langle \Delta_i, \Delta_j \rangle
  ]
* magnitude of interference

---

## Composition Success

* constraint satisfaction
* distance to true joint
* manifold adherence

---

# 🔷 10. Final Synthesis

## What This Framework Achieves

You are testing:

> Can we **learn a representation where compositional assumptions become approximately true**, even when they are false in raw data?

---

## 🔥 Success Conditions

Composition works when:

1. latent factors are **separated**
2. score corrections are **compatible**
3. joint lies on **valid manifold**
4. representation aligns with **conditions**

---

## ❌ Failure Conditions

1. latent entanglement
2. score interference
3. off-manifold solutions
4. structural mismatch

---

# 🔥 Most Important Takeaway

> Composition is fundamentally a **representation learning problem**, not just a diffusion problem.

---

# 🔷 Final Verdict

✔ Your approach is **scientifically sound**
✔ It **faithfully respects the math of the papers**
✔ It creates a **controlled experimental ladder (toy → MNIST → real)**

---

## ❗ Critical Constraint (Now Explicit)

> Toy datasets must be constructed from **known latent factors and entangled observation maps**, not just simple geometric blobs.

---

## Final One-Line Summary

> You are not testing whether composition works—you are testing whether you can **learn a space where composition becomes valid**.
