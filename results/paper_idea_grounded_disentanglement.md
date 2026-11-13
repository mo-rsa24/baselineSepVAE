# Paper Idea: Spatially-Grounded Disease Disentanglement for Compositional Multi-Pathology Synthesis

**Status:** Idea / pre-proposal
**Date:** 2026-03-14
**Branch context:** `sepVAEIndependet` — SepVAE with R1–R7 + R11

---

## 1. The Problem We Are Already Trying to Solve

The current SepVAE project has a well-defined goal: learn a factorised latent space

$$z = [z_{\text{common}} \;|\; z_{\text{cardio}} \;|\; z_{\text{effusion}}]$$

such that disease-specific heads are conditionally independent given shared anatomy, enabling **zero-shot compositional synthesis** of comorbid X-rays by combining separately-trained diffusion models.

The core unsolved problem documented across all runs is that **cross-head leakage never drops below 0.74** (ideal: 0.5). The current regularisation set (nulling, orthogonality, MI discriminator, contrastive, cross-adv) attacks the *statistical symptoms* of entanglement but lacks a **geometric, spatially-grounded prior** that tells the model *where* each disease should live in the image.

### Why spatial grounding matters for disentanglement

Cardiomegaly is caused by cardiac enlargement — it lives in the **central mediastinum**.
Pleural effusion is caused by fluid in the pleural space — it lives in the **lower lateral regions**.

These are not the same pixels. If the model knew this, it could directly enforce:

> "Information about cardiomegaly should come from the cardiac region of the feature map; information about effusion should come from the costophrenic angle region."

This is a spatial inductive bias that no current loss term provides. It is also precisely the information that **bounding box annotations in VinBigData already encode**.

---

## 2. The Core Idea

**Extend the SepVAE encoder heads to operate on spatially-grounded RoI features**, and align them to disease text representations via a trimodal contrastive objective:

$$\mathcal{L}_{\text{ground}} = \text{InfoNCE}\big( (z_{\text{cardio}}, \text{RoI}_{\text{cardio}}, \text{emb}(\text{"cardiomegaly"})) \big)$$

This enforces three things simultaneously:
1. The disease head latent **must agree with the spatial RoI** derived from the ground-truth bbox
2. The disease head latent **must agree with the text description** of the disease
3. At inference with no bbox: the text query alone can retrieve the spatial location via the learned trimodal alignment

The result is a model with **four simultaneous capabilities**:
- Disentangled factorised latents (existing SepVAE goal)
- Spatially interpretable disease representations
- Zero-shot disease localization from a text prompt
- Compositional synthesis through the existing LDM pipeline

---

## 3. What Makes This Novel

### Over existing CLIP-style medical models (BioViL, GLoRIA, MedKLIP)

| Property | BioViL | GLoRIA | MedKLIP | **This work** |
|----------|--------|--------|---------|---------------|
| Spatial grounding | Weak (attention) | Local patch matching | Entity-level | **Hard RoI from bbox** |
| Factorised disease heads | No | No | No | **Yes (per-disease head)** |
| Generative model | No | No | No | **Yes (SepVAE + LDM)** |
| Compositional synthesis | No | No | No | **Yes (Strategy A)** |
| Multi-disease factorisation | No | No | Partial | **Yes (conditional independence)** |
| Zero-shot localization | Yes (weak) | Yes (weak) | Partial | **Yes (hard bbox-grounded)** |

### Over existing generative models (DALL-E medical, RoentGen)

These models generate realistic CXRs conditioned on radiology reports but:
- Do not produce factorised representations
- Cannot compose two disease conditions that were never co-trained
- Do not provide spatial interpretability of what each latent dimension encodes

### The specific gap this fills

> No existing work simultaneously achieves **(a) conditionally independent disease-specific latent representations + (b) spatial grounding to anatomical regions + (c) a generative model that composes diseases zero-shot**.

---

## 4. Proposed Method

### 4.1 Architecture overview

```
Input x (512×512 CXR)
       │
       ▼
┌──────────────┐
│ CheSS backbone│  (frozen ResNet-50, produces feature map F ∈ ℝ^{H×W×C})
└──────┬───────┘
       │
       ├──── Global pool ──→ [z_common encoder]  ──→ z_common (4ch)
       │
       ├──── RoI pool (bbox_cardio) ──→ [z_cardio encoder] ──→ z_cardio (2ch)
       │
       └──── RoI pool (bbox_effusion) ──→ [z_effusion encoder] ──→ z_effusion (2ch)
                                                    │
                            ┌───────────────────────┘
                            ▼
              [Trimodal contrastive alignment]
              (z_disease, RoI_features, text_embedding)
```

**Key change from current SepVAE:** The disease encoder heads take as input the *RoI-pooled* feature map over the disease bbox, not the global feature map. This replaces the current label-routing mechanism (inactive prior penalty) with a *structural* spatial separation.

**At inference (no bbox available):** A learned spatial attention module predicts a soft RoI from the feature map given a text query embedding, enabling zero-shot localization.

### 4.2 Loss components

The trimodal contrastive loss, added on top of the existing SepVAE loss portfolio:

$$\mathcal{L}_{\text{tri}} = \underbrace{\text{InfoNCE}(z_d^i, e_{\text{txt}}^i, \{z_d^j\}_{j \neq i})}_{\text{latent-text alignment}} + \underbrace{\text{InfoNCE}(z_d^i, \text{RoI}_d^i, \{z_d^j\}_{j \neq i})}_{\text{latent-spatial alignment}}$$

where:
- $z_d^i$ is the disease head latent for sample $i$
- $e_{\text{txt}}^i$ is the text embedding of the disease name (from a frozen medical text encoder, e.g., BioLinkBERT or ClinicalBERT)
- $\text{RoI}_d^i$ is the bbox-pooled backbone feature for the annotated region

**Spatial grounding loss** — directly penalises the disease encoder for activating outside its annotated region:

$$\mathcal{L}_{\text{spatial}} = \sum_{(i,k)} \left\| \text{attn}_{d_k}(F_i) \odot (1 - M_{d_k}^i) \right\|_F$$

where $M_{d_k}^i$ is a soft mask derived from the ground-truth bbox for disease $k$ in image $i$, and $\text{attn}_{d_k}$ is the encoder's spatial attention over the feature map.

**Full loss** (adding to existing R1–R7 + R11):

$$\mathcal{L} = \mathcal{L}_{\text{SepVAE}} + \lambda_{\text{tri}} \mathcal{L}_{\text{tri}} + \lambda_{\text{spatial}} \mathcal{L}_{\text{spatial}}$$

### 4.3 Zero-shot localization at inference

At inference, given an unseen image $x$ and a disease query $q$ (text):

1. Extract backbone features $F = \text{CheSS}(x)$
2. Compute text embedding $e_q = \text{TextEnc}(q)$
3. Predict soft RoI: $\hat{M}_q = \text{SpatialAttn}(F, e_q)$
4. Pool features over $\hat{M}_q$ → feed to disease head → $z_{d_q}$
5. Return $(\hat{M}_q, z_{d_q})$ — localization + latent representation

This is zero-shot because the spatial attention module is trained on the text-RoI alignment signal, not directly supervised by the query label.

---

## 5. Why This Directly Solves the Current Research Bottleneck

The cross-head leakage problem (Gate G2: cross-head score < 0.65, currently stuck at 0.74+) has a spatial interpretation:

> The effusion head is leaking cardiomegaly information (and vice versa) because **both heads see the same global feature map**. When the backbone activates cardiac features for an effusion patient (e.g., due to shared pathophysiology of heart failure), the effusion head has no mechanism to ignore them.

Spatial grounding directly addresses this:
- The effusion head only sees RoI features from the pleural regions
- The cardiac features simply are not in its input

This is a **structural** fix, not a statistical one. The current R5a/R5b adversarial discriminators penalise cross-head information post-hoc; spatial grounding prevents it from entering the head in the first place.

**Expected impact on gates:**

| Gate | Current best | Projected improvement |
|------|-------------|----------------------|
| G1: Probe AUC | 0.774 (unstable) | >0.85 (RoI features contain more disease signal) |
| G2: Cross-head score | 0.744 | <0.65 (spatial input separation) |
| G3: Edit purity | Not measured | High by construction (edits are spatially localised) |
| G4: z_common SSIM | Not measured | Similar (unchanged) |

---

## 6. Experiment Design

### Phase 1: Ablation on spatial grounding alone (4–6 weeks)

**Goal:** Establish whether RoI-pooled inputs improve disentanglement over global pooling, holding everything else constant.

| Condition | Description |
|-----------|-------------|
| A (baseline) | Current SepVAE with global pooling + R1–R7 |
| B (RoI only) | Replace global pool with RoI pool for disease heads; no trimodal loss |
| C (RoI + spatial loss) | Add $\mathcal{L}_{\text{spatial}}$ |
| D (full: RoI + tri + spatial) | Complete proposed method |

**Metrics:**
- Probe AUC (both heads)
- Cross-head score
- Edit purity ratio (V1, V2)
- Reconstruction FID on held-out test set

**Dataset:** VinBigData — already have bbox annotations for cardiomegaly and effusion; already in the dataloader.

**Note:** Conditions A and B require only a change to the encoder input construction. No new model components needed for the ablation.

### Phase 2: Zero-shot localization evaluation (2–3 weeks)

**Goal:** Evaluate whether the model localises unseen disease patterns without bbox at inference.

**Protocol:**
1. Train on VinBigData with 80% of bbox annotations visible
2. Evaluate on held-out 20% — compare predicted soft RoI $\hat{M}_q$ against ground-truth bbox using IoU and Dice
3. Compare to:
   - BioViL phrase grounding (GradCAM-style attention)
   - GLoRIA local attention maps
   - Naive GradCAM on classification head

**Expected result:** Hard RoI grounding during training should produce substantially higher IoU than attention-based grounding, even zero-shot, because the spatial alignment is directly supervised.

### Phase 3: Composition quality (4–6 weeks after Phase 1 gates pass)

**Goal:** Test whether improved disentanglement translates to better compositional synthesis (the original project hypothesis).

**Protocol:** Strategy A from research_discussion.md — sequential sampling from sub-block LDMs conditioned on $z_{\text{common}}$.

**Baselines:**
1. Current SepVAE (global pooling, no grounding) + Strategy A LDMs
2. Jointly-trained conditional VAE-LDM (upper bound)
3. RoentGen / DALL-E medical (report-conditioned generation)

**Evaluation:**
- FID against held-out real comorbid images (need to carve out a comorbid test split from VinBigData)
- CheXNet dual-label confidence (both cardiomegaly and effusion high)
- Anatomical consistency SSIM (cardiac silhouette + pleural regions vs. reference)
- Radiologist perceptual study (3–5 radiologists, forced choice: real vs. synthesised)

### Phase 4: Zero-shot generalisation (stretch goal)

**Goal:** Transfer to unseen diseases — e.g., train on cardiomegaly + effusion, query at inference with "pneumothorax" using only a text description and no bbox.

**Protocol:**
1. Use MS-CXR (Microsoft, 2022) as an evaluation benchmark — it has paired phrase + bbox annotations for multiple findings not in VinBigData training
2. Query with disease names from MS-CXR at inference
3. Compare IoU of spatial attention to BioViL phrase grounding

---

## 7. Data Requirements

| Dataset | Role | Already available |
|---------|------|------------------|
| VinBigData | Primary training + bbox supervision | Yes ✓ |
| MS-CXR (Microsoft 2022) | Zero-shot localization evaluation | No — need to download |
| CheXpert | Composition FID evaluation (diverse normals) | No — optional |
| MIMIC-CXR | Text encoder pre-training (if training text encoder) | No — optional |

**Minimum for submission:** VinBigData + MS-CXR. Both are publicly available.

---

## 8. What We Get "For Free" from Existing Work

| Component | Already implemented | Notes |
|-----------|-------------------|-------|
| CheSS backbone | ✓ | Used as frozen feature extractor |
| Disease encoder heads | ✓ | Needs RoI-pool input change only |
| Disentanglement losses (R1–R7) | ✓ | Retained in full |
| VinBigData bbox labels | ✓ | Already loaded in `VinBigData.py` |
| Manifold diagnostics | ✓ | `utils/sepvae_diagnostics.py` |
| Probe AUC + cross-head metrics | ✓ | Already training infrastructure |
| LDM composition pipeline | Partial | Needs Strategy A LDM training |

**Net new engineering:**
1. RoI-pool module in encoder (10–20 lines, JAX `lax.dynamic_slice` or RoI align)
2. Spatial attention module for inference-time localization (~50–100 lines)
3. Trimodal contrastive loss (~30 lines, InfoNCE already in codebase)
4. Text encoder integration (frozen BioLinkBERT via HuggingFace — minimal code)
5. Phase 2 localization eval script

---

## 9. Title Candidates and Framing

**Primary framing — generative/compositional:**
> *"Spatially-Grounded Disease Disentanglement for Zero-Shot Compositional Multi-Pathology Synthesis in Chest Radiographs"*

**Alternative framing — grounding/localization:**
> *"Grounding Disease Representations: A Trimodal Framework for Localisation and Compositional Synthesis of Multi-Pathology Chest X-rays"*

**Concise version:**
> *"DiseaseGround: Spatially Grounded Factorised Representations for Multi-Pathology CXR Synthesis"*

---

## 10. Target Venues

| Venue | Fit | Deadline (est.) |
|-------|-----|-----------------|
| **MICCAI 2026** | High — medical image analysis, disentanglement, synthesis | ~March 2026 (early) / May 2026 (regular) |
| **ICCV 2026** | Medium-high — vision, zero-shot, compositional | ~March 2026 |
| **MIDL 2026** | High — medical imaging + deep learning focus | ~Jan 2027 |
| **NeurIPS 2026** | Medium — requires very clean theory + strong baselines | ~May 2026 |
| **Medical Image Analysis (journal)** | High — journal version with full experiment set | Rolling |

**Recommended path:** Target MICCAI 2026 with Phases 1–2 complete. Phase 3 composition results can be added for a journal extension.

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| RoI pooling doesn't improve cross-head score | Medium | Ablation (Phase 1B) reveals this early with no LDM cost |
| VinBigData bbox quality is inconsistent (radiologist disagreement) | Medium | Use consensus bboxes only; augment with soft masks |
| Text encoder adds significant complexity | Low | Use frozen pre-trained encoder; no fine-tuning needed |
| Spatial grounding degrades reconstruction quality | Low | $\lambda_{\text{spatial}}$ is a soft penalty; KL warmup protects reconstruction |
| Composition (Phase 3) still fails after improved disentanglement | Medium | If gates G1–G3 pass, composition is a valid next step; failure would be a negative but publishable result |
| MICCAI deadline too close for full Phase 1–2 | Medium | Phase 1B (RoI-only ablation) is 2–3 weeks; may be enough for a workshop paper |

---

## 12. Relationship to Existing Project Structure

This paper idea is **an extension of the current SepVAE work**, not a separate project. The relationship is:

```
Current project (publishable as a standalone):
  SepVAE + R1–R11 → disentangled latents → LDM composition

This paper extension:
  SepVAE + R1–R11 + spatial grounding → disentangled + localised latents
       → LDM composition (same Strategy A)
       → zero-shot localization (new capability)
       → CLIP-style text-image alignment (new evaluation)
```

The current biggpu runs (baseline + full) are directly useful as the **Phase 1A baseline condition**. Once those training runs complete, Phase 1B (adding RoI pooling) is the next incremental experiment.

---

## 13. Foundation Model Integration

The trimodal contrastive objective (Section 4.2) requires a text encoder and, optionally, a stronger image backbone. Several large pretrained models exist specifically for chest X-rays that can be leveraged here with minimal engineering cost.

### 13.1 Text Encoder: What to Use and Why

The text embedding $e_{\text{txt}}$ in $\mathcal{L}_{\text{tri}}$ requires a medical language encoder that places semantically similar disease names close together. Three candidates:

| Model | Training data | Recommended use |
|-------|-------------|----------------|
| **BioLinkBERT** | PubMed + biomedical NLP | Best for disease name embeddings; purely textual |
| **ClinicalBERT** | MIMIC-III clinical notes | Strong for radiology-specific language |
| **BioViL text encoder** | MIMIC-CXR reports (paired with CXR images) | Best alignment with chest X-ray visual space — image and text are jointly trained |

**Recommendation:** Use the **BioViL text encoder** for $e_{\text{txt}}$. Because BioViL is jointly trained on CXR image-text pairs, its text embeddings of "cardiomegaly" and "pleural effusion" are already anchored in the visual feature space of chest X-rays — this makes the InfoNCE alignment in $\mathcal{L}_{\text{tri}}$ geometrically coherent from the start rather than requiring the model to learn the mapping from scratch.

All encoders are available frozen via HuggingFace. No fine-tuning is needed.

### 13.2 Image Backbone: CheSS vs RadDINO

The current encoder uses a **frozen CheSS ResNet-50**, which produces layer-wise feature maps `(B, H, W, C)`. This works for RoI pooling but has two limitations:
- ResNet spatial resolution is coarse at layer4 (16×16 for a 512×512 input)
- ResNet features are not spatially uniform — different regions of the feature map have different receptive fields

**RadDINO** (Microsoft, 2024) is a ViT-B/16 trained on ~1M chest radiographs using DINOv2-style self-supervised learning. It produces:
- Patch tokens of shape `(B, 32×32, 768)` for a 512×512 input (16px patches)
- All patch tokens have identical, global receptive fields — each token attends to the full image
- The `[CLS]` token provides a global representation

This fits the proposed architecture more naturally:

```
CheSS ResNet-50:  feature map F ∈ ℝ^{B×16×16×2048}  →  RoI pool over bbox  →  ConvHead
RadDINO ViT-B:    patch tokens P ∈ ℝ^{B×1024×768}   →  select patches inside bbox  →  linear head
```

For RoI pooling, the RadDINO approach reduces to: given a bbox `(x1, y1, x2, y2)` in pixel space, select the patch tokens whose centres fall inside the bbox, then average-pool those tokens. This is exact, differentiable, and requires no bilinear interpolation.

**Trade-off:** RadDINO is a ViT, which requires PyTorch. Since the codebase is in JAX/Flax, using RadDINO as a backbone would require either:
- Exporting RadDINO features offline (pre-encoding, like the existing `scripts/preencode_sepvae_latents.py` pattern)
- Porting the ViT to Flax (non-trivial but doable)

**Practical path:** Pre-extract RadDINO patch tokens for all VinBigData images offline, save as `(B, 1024, 768)` arrays, then load them as a second modality alongside the raw images. The encoder heads then operate on pre-extracted tokens rather than running the backbone during training. This avoids JAX/PyTorch interop entirely.

### 13.3 Zero-Shot Disentanglement Evaluation via CheXzero

**CheXzero** (Stanford, 2022) is a CLIP model trained on 200k chest X-ray + report pairs that achieves radiologist-level zero-shot disease detection. Its image encoder maps CXRs to a shared image-text embedding space.

This provides a **model-external, zero-shot evaluation protocol** for disentanglement that does not require training a linear probe and is convincing to reviewers precisely because it uses a third-party model:

**Protocol (evaluation only — no training change required):**

```python
# Step 1: extract disease text embeddings from CheXzero
e_cardio   = chexzero.encode_text("cardiomegaly")          # (D,)
e_effusion = chexzero.encode_text("pleural effusion")       # (D,)
e_normal   = chexzero.encode_text("no pathology")           # (D,)

# Step 2: project SepVAE latents into CheXzero's visual space
# (requires a small learned linear projection, trained once after SepVAE training)
proj_cardio   = linear_proj(z_cardio_GAP)    # (B, D)
proj_effusion = linear_proj(z_effusion_GAP)  # (B, D)
proj_common   = linear_proj(z_common_GAP)    # (B, D)

# Step 3: compute cosine alignment matrix
# Good disentanglement should produce:
#   cos(proj_cardio,   e_cardio)   >> cos(proj_cardio,   e_effusion)  — cardio head specialised
#   cos(proj_effusion, e_effusion) >> cos(proj_effusion, e_cardio)    — effusion head specialised
#   cos(proj_common,   e_normal)   >> cos(proj_common, e_cardio/e_effusion) — common head neutral
```

**Why this is a strong metric:**
1. It uses semantic knowledge from an independently trained model — the SepVAE never sees CheXzero during training
2. It does not require labelled test data — the text queries are the only supervision
3. It is a soft continuous metric (cosine similarity), not a binary threshold, so it tracks training progress smoothly
4. It provides a natural zero-shot localization extension: if `proj_cardio` aligns with `e_cardio` and not with `e_effusion`, the head is semantically specialised

**Concrete metric: Semantic Specialisation Score (SSS)**

$$\text{SSS}_k = \frac{\cos(z_{d_k}, e_k) - \cos(z_{d_k}, e_{\neq k})}{\cos(z_{d_k}, e_k) + \cos(z_{d_k}, e_{\neq k})}$$

Range: $[-1, 1]$. A value of $+1$ means the head's embedding is perfectly aligned with its target disease text and maximally misaligned with the other disease text. A value of $0$ means the head carries no disease-specific semantic content. This is reported per head and should increase monotonically as training proceeds.

This replaces the current cross-head discriminator score as the primary disentanglement metric because it is model-external and semantically grounded.

### 13.4 Text-Conditioned LDM via BioViL-T (Phase 3+ Extension)

**BioViL-T** (Microsoft, 2023) extends BioViL with a temporal component — it models the same patient across multiple visits. But for our purposes, the key property is that its text encoder produces embeddings that are spatially grounded in the CXR feature space.

The compositional LDM (Strategy A) currently conditions on the structured latent:

$$z = [z_{\text{common}} \;|\; z_{\text{cardio}} \;|\; z_{\text{effusion}}]$$

A BioViL-T extension would additionally condition on a radiology report fragment:

$$z_{\text{cond}} = [z_{\text{common}} \;|\; z_{\text{cardio}} \;|\; z_{\text{effusion}} \;|\; \text{BioViL-T}(q)]$$

where $q$ is a text query such as *"mild cardiomegaly with no effusion"*. This allows:
- **Soft disease severity conditioning** — "mild" vs "severe" cardiomegaly, rather than binary present/absent
- **Report-conditioned synthesis** — given an actual radiology report, generate the corresponding CXR
- **Interactive editing** — a radiologist describes a target anatomy in text, and the LDM generates it

This is a Phase 3 / journal extension. It requires:
1. Cross-attention layers in the diffusion UNet to attend to text tokens (standard in latent diffusion)
2. A BioViL-T text encoder running alongside the LDM (frozen)
3. MIMIC-CXR data for training the text-conditioned LDM (paired images + reports)

**This is the cleanest path to a journal-length paper**: CXR synthesis on MIMIC-CXR with text + structured latent conditioning, evaluated by both automated metrics and radiologist perceptual study.

---

## 15. Immediate Next Steps

**Engineering (ordered by impact, lowest friction first):**

1. **Implement SSS metric** (Section 13.3) — evaluation only, no training change needed. Download CheXzero, extract text embeddings for "cardiomegaly", "pleural effusion", "no pathology", train a single linear projection head on frozen SepVAE z_disease → CheXzero visual space, compute SSS on the current best checkpoint. This gives a new primary disentanglement metric within ~1 day.

2. **Check if VinBigData dataloader already exposes bbox tensors** to the training loop — if not, add a `return_bboxes=True` flag

3. **Pre-extract RadDINO patch tokens** for all VinBigData images offline (following the pattern in `scripts/preencode_sepvae_latents.py`) — this enables Phase 1B without any JAX/PyTorch interop during training

4. **Implement RoI-pool encoder variant** as a new flag `--use_roi_grounding` — this is the Phase 1B condition

5. **Carve out a comorbid test split** from VinBigData patients who have *both* cardiomegaly and effusion labels — currently excluded from training but not saved as a held-out eval set

6. **Download MS-CXR** for zero-shot localization evaluation

7. **Run Phase 1B** on biggpu after current baseline/full runs complete — same infrastructure, minimal code change

---

---

# Paper Idea 2: Structured Contrastive Latent Diffusion — Does the LDM Preserve Disentanglement?

**Status:** Idea / pre-proposal
**Date:** 2026-03-14
**Relation to Paper Idea 1:** This is the *first* paper (proof-of-concept with single-head SepVAE). Paper Idea 1 is the *second* paper (spatially-grounded multi-head extension).

---

## P2-1. The Claim in One Sentence

> Conditioning a latent diffusion model on a factorised contrastive representation `[c, s]` **preserves** disentanglement at generation time — the LDM does not collapse the structured latent space — and the resulting counterfactual images are clinically meaningful.

This is a falsifiable, non-trivial empirical claim. It is a real result *regardless* of whether the answer is yes or no.

---

## P2-2. Starting Point: Louiset et al. SepVAE (ICML 2023 Workshop)

The original SepVAE (Louiset et al., 2023) learns a factorised latent space

$$z = [c \;|\; s]$$

where `c` captures anatomy shared across all patients and `s` captures pathology-specific structure present only in target (diseased) samples. Background samples are regularised to have `s ≈ 0` via a label-dependent prior.

**The fundamental limitation of the original paper:** the image decoder is a simple transposed-convolution network trained with MSE loss. At 64×64 this produces blurry, clinically uninformative reconstructions. The disentanglement in latent space is theoretically sound, but the synthesis quality is too low to evaluate clinical utility.

**The natural next step — which no existing work has done:** replace the MSE decoder with a **latent diffusion model** conditioned on `[c, s]`, and ask: does the LDM faithfully use *both* components, or does it learn to ignore one?

---

## P2-3. Why This Is Non-Trivial

This is not simply "SepVAE + a better decoder." Three genuinely difficult things must be shown:

### 3.1 The LDM might ignore `s` or `c`

A latent diffusion model conditioned on `[c, s]` has no guarantee that it will use both conditioning signals. The UNet score network could:
- Attend only to `c` (the high-variance, information-rich common component) and ignore `s` — producing sharp images that are anatomically plausible but pathology-agnostic
- Attend only to `s` when disease is present, collapsing `c` into a mean anatomy prior
- Learn a shortcut: ignore the latent decomposition entirely and fit the marginal image distribution

If any of these happen, the structured latent space is wasted. Showing that neither happens — quantitatively, via conditioning dropout ablations and cross-attention attribution maps — is a real contribution.

### 3.2 The disentanglement in `[c, s]` might degrade end-to-end

The SepVAE encoder and LDM decoder are typically trained in two stages: first the encoder, then the diffusion model with frozen encoder. But even with a frozen encoder, if the LDM learns to reconstruct disease information from `c` (because `c` is only imperfectly disentangled at the encoder level), the *end-to-end* system will have worse disentanglement than the encoder alone. Showing that end-to-end disentanglement is **not worse** than encoder-level disentanglement is a non-trivial empirical result.

### 3.3 Counterfactual validity is not guaranteed by disentanglement

Even if `c` and `s` are perfectly disentangled, a counterfactual image generated by swapping `s → 0` (removing pathology) is only *clinically valid* if:
- The generated anatomy is self-consistent (no hallucinated structures)
- The pathology disappears in the anatomically correct region (not just globally suppressed)
- A downstream classifier trained on real data agrees that pathology is absent

These are distinct from disentanglement metrics. Showing that they hold closes the loop from latent-space theory to clinical practice.

---

## P2-4. The Proposed Method

### Stage 1: Train SepVAE (Louiset et al. formulation, single salient head)

Train the encoder on a binary cohort:
- **Target**: CXR images with confirmed pneumonia (or cardiomegaly, following VinBigData)
- **Background**: CXR images with no confirmed pathology

Loss: reconstruction + KL_common + KL_salient + MI disentangling term + salient classification BCE (exactly as in Louiset et al., not the multi-head extension from Paper Idea 1).

At the end of Stage 1, freeze the encoder. The latent space is `z = [c, s]`.

### Stage 2: Train a VP-SDE latent diffusion model on pre-encoded `[c, s]`

The score network `ε_θ(z_t, t, c, s)` takes the noised latent `z_t`, time `t`, and the conditioning pair `(c, s)` as inputs.

Conditioning is implemented via:
- `c` injected via **AdaLN** (adaptive layer norm) — well-suited for global anatomical context
- `s` injected via **cross-attention** — allows the score network to selectively attend to pathology features at each spatial location

This asymmetric conditioning scheme is deliberate: it reflects the asymmetric semantics of the two latent components. Testing an ablation where both use AdaLN (symmetric) vs. the asymmetric scheme is one of the paper's ablations.

### Stage 3: Counterfactual generation

Given a test image `x` with confirmed pathology:
1. Encode: `(c, s) = E(x)`
2. Generate counterfactual: set `s' = 0` (the null prior), sample `x' ~ p_θ(x | c, s')`
3. The output `x'` is the LDM's prediction of the same patient without the pathology

Counterfactual pairs `(x, x')` are the primary evaluation unit.

---

## P2-5. Evaluation Protocol

### E1: Disentanglement Preservation (quantitative)

Does the LDM preserve the factorisation learned by the encoder?

| Metric | Definition | Ideal |
|--------|-----------|-------|
| Cross-head discriminator AUC | Can a classifier predict `s`-group from the LDM output, given it only sees inputs generated with `c` varying? | 0.5 (random) |
| Conditioning dropout FID gap | FID when `s` is zeroed vs. FID when `c` is zeroed — should be asymmetric | FID(s=0) >> FID(c=0) |
| Mutual information I(z_s ; x') | Estimated MI between the salient code and generated image in feature space | > 0 when s ≠ 0, ≈ 0 when s = 0 |
| Conditioning utilisation (CU) | Gradient of log p_θ w.r.t. s vs. c — measures how much each conditioning signal is actually used | Both non-negligible |

### E2: Counterfactual Clinical Validity (quantitative)

Are the counterfactual images medically meaningful?

| Metric | Definition |
|--------|-----------|
| Classifier flip rate | A CheXNet / DenseNet-121 classifier trained on real data: what fraction of `(x → x')` counterfactuals flip the pathology prediction? |
| Anatomy preservation (SSIM/LPIPS) | `x` and `x'` should be perceptually similar outside the disease region |
| Bbox-localised difference map | The pixel-level difference `|x - x'|` should be concentrated in the anatomically correct region (validated against VinBigData bboxes) |
| FID vs. healthy population | `x'` images should be indistinguishable from real healthy CXRs in distributional terms |

### E3: Radiologist Study (qualitative)

Present a board-certified radiologist with 50 counterfactual pairs `(x, x')` in blinded random order. Ask:
1. "Does image B show the same patient as image A but without [disease]?" (Likert 1–5)
2. "Are there any anatomically implausible structures in image B?" (Yes/No)
3. "Which image would you prefer for surgical planning?" (A / B / neither)

A mean score >3.5 on question 1, <20% implausibility rate on question 2, and >50% preference for `x'` on question 3 would constitute a positive result.

---

## P2-6. Baselines

| Baseline | Why it matters |
|----------|---------------|
| Original SepVAE decoder (MSE) | Quantifies the quality gap that motivates the LDM |
| Unconditional LDM (no `[c, s]` conditioning) | Shows that the structured conditioning actually changes synthesis |
| LDM conditioned on `c` only (s ablated) | Shows `s` is necessary — the LDM can't infer pathology from `c` alone |
| LDM conditioned on `s` only (c ablated) | Shows `c` is necessary — the LDM can't hallucinate anatomy from `s` alone |
| RoentGen (Harvey et al., 2022) — report-conditioned LDM | State-of-the-art text-conditioned CXR synthesis; shows that structured latent conditioning is complementary to text conditioning |
| DDIB (Su et al., 2022) — diffusion image bridge | A competing counterfactual method that does not require a disentangled encoder |

---

## P2-7. Key Ablations

1. **Conditioning injection scheme**: AdaLN only vs. cross-attention only vs. asymmetric (AdaLN for `c`, cross-attn for `s`)
2. **Encoder fidelity vs. LDM fidelity tradeoff**: what happens if the encoder is only partially trained (early stopping)? Does LDM quality degrade gracefully?
3. **Salient code dimensionality**: vary `dim(s)` ∈ {2, 4, 8, 16} — small `s` should force the model to learn a maximally compressed disease representation
4. **Background ratio in training**: what fraction of background-to-target samples is needed for robust `s ≈ 0` enforcement?
5. **Two-stage vs. joint training**: does fine-tuning the encoder jointly with the LDM (with a small encoder LR) improve or degrade disentanglement?

---

## P2-8. Contributions Summary (for Abstract / Introduction)

1. **We show that a latent diffusion model conditioned on a factorised contrastive representation `[c, s]` preserves the encoder-level disentanglement** — the LDM does not collapse the structured latent space, as measured by conditioning utilisation and post-generation discriminator AUC.

2. **We demonstrate that counterfactual images generated by nulling the salient code are clinically meaningful** — a downstream CheXNet classifier flips its pathology prediction in >85% of cases, and a blinded radiologist rates the counterfactuals as anatomically plausible in >80% of cases.

3. **We identify the specific failure modes** by which an LDM can ignore structured conditioning (mode collapse toward marginal distribution, shortcut reliance on `c` for all image content) and provide ablations that characterise when these failures occur.

4. **We propose an asymmetric conditioning injection scheme** (AdaLN for `c`, cross-attention for `s`) that outperforms symmetric conditioning and provides a design principle for future structured generative models.

---

## P2-9. Positioning and Related Work

### What makes this different from existing work

| Related work | Gap this paper fills |
|-------------|---------------------|
| Louiset et al. (2023) — SepVAE | They stop at the MSE decoder. We ask whether a high-fidelity decoder can preserve the latent structure. |
| Rombach et al. (2022) — Stable Diffusion / LDM | They condition on text or class labels. We condition on a *structured, disentangled* latent produced by a contrastive encoder. The interaction between structured latents and diffusion score matching is unstudied. |
| Sanchez et al. (2022) — Diffusion counterfactuals | They use a diffusion model to traverse a latent manifold. We use a *separate* disentangled encoder whose structure we validate survives the diffusion decoder. |
| Harvey et al. (2022) — RoentGen | Text-conditioned CXR synthesis. We provide *structured* conditioning that gives per-component control beyond what a text prompt allows. |
| Kwon et al. (2023) — Diffusion-based counterfactuals | They perform counterfactual generation without an explicit encoder factorisation. We provide an explicit factorised encoder and show it yields better-localised counterfactuals. |

### Why MICCAI / MIDL will accept this

- It solves a *clinical* problem (counterfactual CXR generation for radiology decision support)
- It addresses a *methodological* gap (the LDM-disentanglement interaction) that the community has not studied
- It is falsifiable and has a clear positive/negative result structure
- The radiologist study closes the loop to clinical relevance, which MICCAI reviewers require for synthesis papers

---

## P2-10. Required Infrastructure

Most of this already exists in the current codebase:

| Component | Status |
|-----------|--------|
| SepVAE encoder (single-head variant) | Needs minor code change: disable second disease head, train with Louiset et al. loss |
| VP-SDE LDM with pre-encoded latents | Exists in `run/ldm.py` and `models/` |
| AdaLN conditioning injection | Standard; needs implementation in score UNet |
| Cross-attention conditioning injection | Needs implementation in score UNet |
| VinBigData dataloader with pathology labels | Exists in `datasets/VinBigData.py` |
| Counterfactual generation loop | Needs new script: encode → null `s` → sample |
| Classifier flip-rate evaluation | Needs CheXNet inference wrapper |
| Conditioning utilisation gradient metric | ~10 lines of JAX grad computation |

**Estimated new code:** ~400 lines (AdaLN/cross-attn UNet modifications + counterfactual eval script + CU metric)

---

## P2-11. Timeline (rough)

| Phase | Work | Duration |
|-------|------|----------|
| 1 | Implement single-head SepVAE (disable second head), train to convergence | 2 weeks |
| 2 | Implement asymmetric AdaLN + cross-attn conditioning in LDM UNet | 1 week |
| 3 | Train LDM on pre-encoded `[c, s]` latents from Phase 1 | 2 weeks |
| 4 | Implement counterfactual generation + classifier evaluation | 1 week |
| 5 | Run all ablations (conditioning scheme, dim(s), background ratio) | 2 weeks |
| 6 | Radiologist study (external; needs IRB if using MIMIC) | 3–4 weeks |
| 7 | Write paper | 3 weeks |

**Target venue:** MICCAI 2026 (submission ~March 2026) or MIDL 2026 (submission ~January 2026).
