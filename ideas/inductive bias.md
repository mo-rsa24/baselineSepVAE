To build a "Full-Stack" pipeline for medical composability, we need to map architectural choices to their specific **Inductive Biases**. In deep learning, an inductive bias is a shortcut the model takes to prioritize certain explanations over others.

---

### 1. CNNs & Residual Blocks (Local Translation Invariance)
* **Structure Imposed:** They assume that the same "feature" (e.g., a sharp edge or a vascular texture) has the same meaning regardless of where it appears in the image.
* **Factorization:** They encourage **spatial local independence**. A filter looking at the top-left doesn't "know" what the filter at the bottom-right is doing.
* **Utility:** Essential for your **"Clean Reconstruction"** goal. Residual blocks (ResNet) allow for deeper networks without gradient vanishing, which is required to capture the high-frequency details (crispness) of an X-ray.
* **Regime:** Baseline for all regimes (Supervised, cVAE, Hybrid).

### 2. Spatial Attention (CBAM / Self-Attention)
* **Structure Imposed:** It assumes that not all pixels are created equal. It creates a **Dynamic Importance Map**.
* **Factorization:** It encourages **Attribute-to-Location** coupling. It tells the model *where* the "Cardiomegaly" information is coming from.
* **Utility:** High. In your **Hybrid cVAE**, attention is the bridge. It allows the model to "attend" specifically to the BBox region to fill the $z_{path}$ latent. Without attention, the model might try to encode heart size using lung volume (the "bleeding" problem).
* **Regime:** Crucial for **cVAE** and **Hybrid** to ensure the "Salient" head doesn't pick up background noise.

### 3. Separate Latent Blocks & Hierarchical Latents
* **Structure Imposed:** These assume the world is **Composational**. They mirror a "Part-Whole" hierarchy (e.g., Patient $\to$ Chest $\to$ Heart $\to$ Disease).
* **Factorization:** They enforce **Structural Independence**. By physically separating $z_{anat}$ and $z_{path}$, you prevent the weights of one from influencing the other.
* **Utility:** This is the **core of your research**. Hierarchical latents (like in NVAE) allow for a "Global-to-Local" flow: the top latents handle the general patient shape, while the bottom (separate) latents handle the specific pathology.
* **Regime:** Best in **Supervised/Hybrid**. It allows you to "turn off" the pathology block for Normal cases.

### 4. Dirichlet & Sparse Priors
* **Structure Imposed:** These assume the world is **Sparse**. A Dirichlet prior (often used in Topic Modeling or LDA) or a Sparse Prior (like $L_1$ penalty) assumes that only a few factors are "active" at any given time.
* **Factorization:** They encourage **Feature Exclusion**. They force the model to explain the image using the *minimum* number of latent dimensions possible.
* **Utility:** Very useful for **separating multiple diseases**. If you have Cardiomegaly and Pleural Thickening, a Sparse Prior forces the model to use $z_1$ for one and $z_2$ for the other, rather than "smearing" both diseases across all latent dims.
* **Regime:** **cVAE**. It acts as a statistical pressure to find the *most* salient difference.

### 5. Slot Attention (The "Object-Centric" Bias)
* **Structure Imposed:** It assumes the image is composed of a fixed number of **Independent Objects (Slots)** that "compete" for pixels via a routing mechanism (like K-Means).
* **Factorization:** It enforces **Mutual Exclusivity**. A pixel belongs to Slot A (Heart) or Slot B (Lungs), but rarely both.
* **Utility:** **Surprisingly low for X-rays.** Slot attention excels at synthetic scenes (Clevr) with distinct objects. In an X-ray, organs are transparent and **overlapping**. A lung pixel is *also* a rib pixel and *also* a skin pixel.
* **Regime:** Avoid for medical imaging unless you are doing instance segmentation of very distinct, non-overlapping masses. Your **BBox + Attention** approach is a "softer," more effective version of slots for medical data.

---

### Critical Comparison Table

| Tool | Inductive Bias | Best For... | Regimes |
| :--- | :--- | :--- | :--- |
| **ResBlocks** | Locality | Crisp Texture | All |
| **Attention** | Relevance | Spatial Localization | cVAE / Hybrid |
| **Sep. Latents** | Modularity | **Composability** | Supervised / Hybrid |
| **Sparse Priors** | Simplicity | Disentanglement | cVAE |
| **Slot Attention** | Objectness | Distinct Entities | (Not recommended) |

### Final Recommendation for Your Objective:
To get **Separation + Crispness**, your "Full-Stack" architecture should look like this:
1.  **Backbone:** CNN with **Residual Blocks** (for crispness).
2.  **Bottleneck:** **Separate Latent Blocks** ($z_{anat}, z_{path}$).
3.  **Bridge:** **Spatial Attention** supervised by your **BBoxes**.
4.  **Prior:** Use a **Sparse/Dirichlet Prior** on $z_{path}$ to ensure it only "activates" when a disease is truly present.

This setup ensures that when you move to Diffusion, your "Cardiomegaly Score" is a clean, spatially-anchored vector that doesn't "know" about the lungs.
