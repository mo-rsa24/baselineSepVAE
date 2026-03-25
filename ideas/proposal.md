To present our PhD research with both clinical nuance and mathematical rigor, your goals and hypotheses are structured as a logical progression: from **characterizing a failure**, to **identifying a natural solution**, to **proposing a technical correction**.

---

## **Research Goal 1: Characterizing the Composability Gap**
The primary objective of this research is to formalize and characterize the "Composability Gap" in generative models—specifically, the divergence between mathematical logical operators and the resulting semantic reality. We seek to develop a unified empirical framework that explains why additive score dynamics, which function predictably in low-dimensional toy environments, often result in "hybridization" or "semantic collapse" when applied to complex, high-dimensional distributions. By investigating the relationship between mutual information and latent trajectories, we aim to identify the specific regimes where relational dependencies between concepts push the sampling process off the high-density data manifold.

### **Hypothesis 1: The Relational Dependency Hypothesis**
We hypothesize that the failure of logical composition is fundamentally a function of high relational dependency between concepts. When two pathologies or attributes share significant structural or spatial information, their score vectors $\nabla \log p(x)$ are non-orthogonal, causing their additive sum to point toward low-density regions of the latent space. This "off-manifold drift" prevents the model from achieving true attribute binding, instead producing a blurry average (hybrid) of the two concepts rather than their logical co-presence.

---

## **Research Goal 2: Leveraging Clinical Independence for Synthesis**
The second goal is to demonstrate the practical utility of factorized representation learning in medical imaging by generating novel, ecologically valid co-morbidities. We aim to show that by selecting specific clinical regimes where pathologies are physically separated, we can bypass the composability gap and synthesize complex disease states (such as Silico-Tuberculosis) using models that have never encountered these combinations during training. This goal focuses on the transition from theoretical "clean" independence to "applied" clinical utility.

### **Hypothesis 2: The Conditional Independence Hypothesis**
We hypothesize that pathologies naturally occupying spatially distinct anatomical regions provide a physical basis for a **conditional independence assumption**. By leveraging this anatomical separation, we can justify the factorization of the latent space into disease-specific components. This assumption enables composable diffusion models to synthesize ecologically valid co-morbidities through score-addition, effectively solving the data-scarcity problem for rare disease combinations without the requirement of joint-labeled training data.

---

## **Research Goal 3: Manifold-Constrained Composition**
The final objective is to develop a mathematically principled method for "closing the gap" identified in the first goal. We aim to move beyond simple linear score-addition by introducing a navigation mechanism that respects the underlying geometry of the data manifold. This involves moving from Euclidean operations to a framework that can handle the "curved" nature of complex latent spaces, ensuring that logical operations always result in semantically coherent and clinically plausible outputs.

### **Hypothesis 3: The Geodesic Navigation Hypothesis**
We hypothesize that the semantic distortions observed in logical composition can be mitigated by treating the latent space as a Riemannian manifold. By navigating the manifold via **geodesic paths** rather than linear trajectories, the model can preserve the semantic integrity of the composed concepts. This approach ensures that the sampling process remains constrained to high-density regions, effectively preventing hybridization and ensuring that the synthesized co-morbidities adhere to the structural rules of the training distribution.

---

### **1. Core Claim: Geometry Does Not Fix Composition**

Even if we perfectly model the data manifold and compose along geodesics (i.e., remain strictly on-manifold), this only guarantees **realistic samples**, not **semantically correct composition**. The failure of compositionality is therefore **not a geometric issue**, but a deeper problem related to how concepts are represented and combined.

---

### **2. Why PoE(Cat ∧ Dog) Produces Hybrids**

Product-of-Experts (PoE) combines distributions by emphasizing regions of **high joint density**. However, if ( p(x \mid \text{cat}) ) and ( p(x \mid \text{dog}) ) are **entangled** (each already encoding correlated features like shape, texture, and context), their product does not isolate independent factors. Instead, it finds **shared statistical features**, which manifest as **plausible hybrids** rather than distinct co-occurring entities.

---

### **3. Effect of Geodesic (Manifold-Aware) Composition**

Using geodesics or Riemannian geometry ensures that compositions remain **valid and realistic**, avoiding off-manifold artifacts. However, it does not change the **semantic structure of the distributions** being composed. As a result, you get **cleaner, more realistic hybrids**, but not logically correct compositions.

---

### **4. Root Cause: Violated Independence Assumptions**

The fundamental issue is that concepts like “cat” and “dog” are not represented as **independent latent factors**, but as **entangled bundles of features**. Without a factorization such as ( z = (z_{\text{species}}, z_{\text{attributes}}) ), composition becomes ill-defined. Logical composition assumes independence, but pretrained models do not satisfy this assumption.

---

### **5. What Actually Needs to Change**

Improving composability requires **restructuring the latent space**, not just modifying geometry or inference. This involves designing models (e.g., VAEs) that enforce **factorized, disentangled representations**, supported by appropriate supervision and objectives. Geometry, diffusion mechanics, and embedding alignment are secondary to this.

---

### **6. Final Insight**

The composability gap arises because current models encode concepts as **overlapping distributions**, so composition reflects **statistical overlap rather than logical structure**. Therefore, even under ideal manifold-aware composition, PoE will still produce hybrids unless **independence assumptions are explicitly enforced during representation learning**.
