# Mask Supervision Journey — From Bbox Collapse to CheXmask

**Related documents:** [09 Forward Plan](09_forward_plan.md) | [08 Current State — D3](08_current_state_d3.md) | [05 Objective Functions](05_objective_functions.md) | [Index](INDEX.md)

**Last updated:** 2026-03-28
**Status:** Exploration complete. Integration into training loop is the next step.
**Git branch:** `feature/mask-supervision` | **Tag:** `explore/mask-supervision-v1`

---

## 1. The Question: Bboxes or Masks?

The original supervision signal for z_disease in SepVAEV2 comes from the VinBigData cardiomegaly
annotations. These annotations are **CTR width measurement boxes** — radiologists draw two vertical
lines marking the maximum cardiac width and maximum thoracic width to compute the Cardiothoracic
Ratio (CTR). They are not cardiac silhouettes.

Concretely, the union of these boxes across radiologists yields:

| Statistic | Value |
|---|---|
| Mean width (normalised) | 0.467 |
| Mean height (normalised) | 0.238 |
| Mean aspect ratio | 2.51 |

A flat rectangle with AR=2.51 covering nearly half the image width. The cross-attention head
(`BboxCrossAttnHead`) trained on this region attends to a region containing ribs, aortic arch,
lung parenchyma, and heart simultaneously.

The theoretical justification for needing better supervision is Locatello et al. (2019): without
inductive bias or supervision, disentanglement is provably unidentifiable — any of infinitely
many representations satisfies the training objective equally. The question is not *whether* to
supervise, but *how precisely*. Noisy supervision (flat bbox) produces noisy representations.

---

## 2. What D3 Actually Revealed

D3 (`d3_gan_fix-20260325-143813`, ep 55→120) is a genuine reconstruction success. It is not a
failed experiment. It demonstrated:

**Reconstruction quality achieved:**
- Crisp rib cortex edges and sharp cardiac borders
- Pulmonary vessel detail visible in hilar region
- No stripe banding, no checkerboard artifacts, no mode collapse
- Stable GAN training after fixing three configuration bugs (see [08 Current State §5](08_current_state_d3.md#5-what-was-fixed-to-get-here))

**Latent separation signals (positive):**
- `z_cardio_norm_ratio > 2.0` — z_disease carries larger-norm representations for cardiomegaly images vs. Normal
- `loss/bbox_attn ≈ 0.20–0.25` — attention map concentrated within the bbox region
- `loss/masked_rec ≈ 0.02` — z_common reconstructs the non-cardiac region without z_disease

**What traversal diagnostics revealed (the problem):**

Running `scripts/latent_traversal.py` at ep105 showed `diff_scale = 0.0001` — the decoder's
pixel-level response to z_disease perturbation was almost unmeasurably small.

Running `scripts/latent_traversal_medsam.py` at ep120 gave a quantitative measurement:
MedSAM-measured cardiac mask area across the full traversal α = 0 → 2.0 changed by **+4 to −7 pixels**.
On a 512×512 image this is a rounding error, not a controllable cardiac size dial.

**Interpretation:** z_disease learned to satisfy the bbox_attn loss (attend within the bbox region)
and produce z_cardio_norm_ratio > 2.0 (different norms for different classes), but it did not
encode a *geometrically meaningful* representation of cardiac size. The model found a shortcut:
encode texture differences within the bbox region, which is sufficient to satisfy every training
objective without producing a latent that controls cardiac geometry at inference.

**Three root cause hypotheses (not yet disambiguated):**

| Hypothesis | Description | Test |
|---|---|---|
| H1: Noisy signal | Flat bbox encodes ribs/aorta alongside heart; z_disease collapses to average texture | Replace bbox with precise cardiac mask → retrain |
| H2: Weak loss weight | `weight_bbox_attn=0.10` dominated by `weight_rec=1.0`; bbox supervision signal too quiet | Raise `weight_bbox_attn` 10× → retrain 20 epochs → re-measure |
| H3: Capacity insufficiency | z_disease has no architectural advantage over z_common for cardiac geometry | Change architecture so only z_disease can access bbox-gated features |

H2 is the cheapest to test and should be done first (see §5).

---

## 3. MedSAM Investigation

### 3.1 Motivation

The immediate diagnostic question was: *can we measure whether z_disease encodes cardiac size
without modifying the training loop?*

If we can extract accurate cardiac silhouettes from decoded images at different traversal points,
we can measure mask area as a function of α and directly answer whether z_disease controls
cardiac geometry. This led to investigating Segment Anything Model for Medical Images (MedSAM).

The longer-term question was whether MedSAM could provide training supervision (mask-overlap
loss or CTR regression targets) to replace the flat bbox supervision.

### 3.2 What MedSAM Is

`wanglab/medsam-vit-base` — SAM fine-tuned on ~1.6M medical image–mask pairs across modalities.
Requires a bounding-box prompt. Returns 3 mask candidates with predicted IoU scores.

### 3.3 Attempt 1 — Union Bbox as Prompt

The VinBigData union bbox was used directly as the MedSAM prompt.

**Result:** Fragmented masks, area 871–2,808 px. Many images had no cardiac mask at all.

**Root cause:** The flat AR≈2.51 bbox confuses MedSAM. The model attempts to segment the largest
coherent object within the box, which in a flat box spanning the cardiac region is often a rib or
the aorta, not the heart.

### 3.4 Attempt 2 — Expanded Bbox

`expand_bbox(x0, y0, x1, y1, up=0.12, down=0.10, sides=0.03)` — added 12% height above and
10% below the union bbox to convert the flat CTR measurement box into something that could
encompass the full cardiac silhouette.

**Result:** Improved — area 9,526–27,902 px, top IoU=0.954 for the best images. But one image
(`e1c7cdc2`) had cardiac_width = 421 px = full image width. The mask spread across the entire
image. MedSAM's predicted IoU for this mask was high — it was *confidently wrong*.

This is a fundamental limitation: MedSAM's IoU score is self-reported confidence that the mask
fits the prompted region. It is not a measure of anatomical correctness.

### 3.5 Attempt 3 — Anatomical Prior (No Annotations)

**Key insight:** The heart occupies a predictable anatomical region in every PA (posteroanterior)
chest X-ray. We do not need annotations to define a reasonable prompt.

```python
CARDIAC_PRIOR = (0.28, 0.35, 0.76, 0.83)  # (x0, y0, x1, y1) normalised
# Covers: aortic arch → diaphragm, right cardiac border → left cardiac border
# Works for normal and cardiomegaly images, PA view only
```

Additional improvements:
- `multimask_output=True` — MedSAM returns 3 candidates; select highest predicted IoU
- Pool-based selection: sample 60 random cardiomegaly images, run all through MedSAM,
  display top 5 by IoU score (filters out ambiguous images automatically)

**Result:** Top 5 IoU = 0.972, 0.959, 0.949, 0.946, 0.942. Mask areas 5,727–34,105 px.
Visually clean cardiac silhouettes in 4/5 images. One image (row 5) showed mild
diaphragm bleed-through despite a high IoU score — another example of IoU not being
a reliable quality indicator.

This is the version implemented in `scripts/test_medsam_cardiac.py`.

### 3.6 Latent Traversal with MedSAM Measurement

`scripts/latent_traversal_medsam.py` combined SepVAEV2 (JAX, GPU) and MedSAM (PyTorch, CPU)
in a single process:

```
For each image:
  1. Encode with SepVAEV2 → (z_common, z_disease, skip features)
  2. For α in [0.0, 0.25, 0.50, ..., 2.0]:
       decoded = SepVAE_decoder(z_common, α × z_disease, skip)
  3. Run MedSAM on each decoded image (anatomical prior prompt)
  4. Record mask area, cardiac width, approximate CTR at each α step
  5. Plot: traversal frames | mask overlays | area/width curve per image
```

**Key result:** Mask area Δ = +4 to −7 px across the full α = 0 → 2.0 range.
Across multiple images, the mean mask area curve is nearly flat with near-zero variance.

This confirmed quantitatively that z_disease at D3/ep120 does not control cardiac size.

### 3.7 Engineering Obstacles Solved

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: CVE-2025-32434` | `transformers ≥4.51` blocks `.bin` loading with `torch<2.6` | Download `.bin` manually, clone all tensors (`v.clone().contiguous()`), convert to safetensors in tmpdir, load from there |
| `AttributeError: torch.get_default_device` | `torch 2.2.2` in `cxr` env (added in 2.3) | Monkey-patch: `if not hasattr(torch, 'get_default_device'): torch.get_default_device = lambda: None` |
| `RuntimeError: Some tensors share memory` | SAM shares `shared_image_embedding.positional_embedding` | `{k: v.clone().contiguous() for k, v in state_dict.items()}` before saving |
| JAX preallocates all GPU memory | XLA default allocates full GPU; PyTorch finds no memory | `os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.55'`; run MedSAM on CPU |
| `ModuleNotFoundError: accelerate` | Required by transformers model loader | `pip install accelerate` in `cxr` env |

### 3.8 Conclusion on MedSAM

MedSAM is a useful post-hoc diagnostic tool and remains valuable for datasets not covered by
purpose-built CXR segmentation datasets. For VinBigData training supervision it is the wrong tool:

1. **Pseudo-label noise is diverse:** Different images fail in different ways. There is no
   systematic noise pattern to account for in the loss function.

2. **IoU ≠ quality:** The only available quality signal is self-reported and unreliable, as
   demonstrated by the full-width mask with high IoU.

3. **Throughput:** ~2s/image on GPU. Offline generation of 14,029 masks would take ~8 hours
   and produces static pseudo-labels — any noise is baked into training permanently.

4. **Supervision redundancy:** CheXmask (§4) provides the same cardiac silhouettes at higher
   quality, with physician-validated quality scores, at zero inference cost.

---

## 4. CheXmask: Ground Truth Supervision Signal

### 4.1 Discovery

CheXmask (Gaggion et al., *Nature Scientific Data*, 2024) is a large-scale dataset of anatomical
segmentation masks for chest X-rays. It covers six public datasets including **VinDr-CXR** —
which is our VinBigData training set.

| Property | Value |
|---|---|
| Paper | Gaggion et al. 2024, *Nature Scientific Data* |
| Source | PhysioNet, CC-BY 4.0 (no authentication required) |
| Segmentation model | HybridGNet — graph neural network purpose-built for CXR anatomy |
| Physician validation | DSC 0.955–0.967 vs. two experienced physicians on a gold-standard subset |
| VinDr-CXR coverage | 18,000 images |
| Masks provided | Left Lung (RLE), Right Lung (RLE), Heart (RLE), Landmarks |
| Quality score | Dice RCA (Mean) — calibrated reverse classification accuracy per mask |
| File | `Preprocessed/VinDr-CXR.csv` — 1024×1024 resolution, 309 MB |

### 4.2 Why the Size Numbers Are Not Contradictory

A natural question arose: our VinBigData zip is 142 GB, and the full CheXmask is 15.6 GB.
How can CheXmask cover the same dataset if it is smaller?

- **142 GB zip:** Raw DICOM files at native acquisition resolution (~8–10 MB/image × 15,000 images).
  DICOM stores uncompressed 16-bit pixel arrays at 2000×3000+ pixel resolution.
- **7 GB cache:** Our pre-processed 512×512 uint16 `.npy` files for 14,029 images.
- **15.6 GB CheXmask (all 6 datasets):** CSV files containing only RLE-encoded mask strings —
  no images. RLE for a 1024×1024 binary mask compresses to ~23 KB. Images are not redistributed.

CheXmask and our cache use identical image IDs (32-char hex strings from the VinDr-CXR Kaggle
release). The join is direct.

### 4.3 Download

```bash
# Scripts: scripts/download_chexmask.sh
# Downloads only VinDr-CXR CSV (~309 MB), not the full 15.6 GB zip

wget -N -c \
  "https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/Preprocessed/VinDr-CXR.csv" \
  -O /datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv
```

### 4.4 RLE Decoding

CheXmask uses 1-indexed, row-major RLE:

```python
def decode_rle(rle_string: str, height: int, width: int) -> np.ndarray:
    runs = np.array(rle_string.split(), dtype=np.int64)
    starts  = runs[::2] - 1   # 1-indexed → 0-indexed
    lengths = runs[1::2]
    flat = np.zeros(height * width, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        flat[s : s + l] = 1
    return flat.reshape(height, width)
```

Height and Width columns are included per row. Masks are 1024×1024 in the Preprocessed CSV;
resize to 512×512 with nearest-neighbour for alignment with our cached images.

### 4.5 CTR from Masks

```python
def compute_ctr(heart, left_lung, right_lung):
    thorax = (left_lung | right_lung).astype(bool)
    cols_h = np.where(heart.astype(bool).any(axis=0))[0]
    cols_t = np.where(thorax.any(axis=0))[0]
    cardiac_w  = int(cols_h.max() - cols_h.min())
    thoracic_w = int(cols_t.max() - cols_t.min())
    return cardiac_w / thoracic_w
```

This gives true CTR (not an approximation from annotation boxes) for every training image.

### 4.6 Validation Results

Running `scripts/verify_chexmask.py`:

```
Loaded CSV: 18,000 rows
Matched to cache: 14,029 / 18,000   ← 100% hit rate on training set
After Dice RCA (Mean) ≥ 0.70: 13,970 usable   ← only 59 removed
Top 8 by quality: Dice RCA range 0.887–0.913
```

The 3,971 unmatched rows are VinDr-CXR test-set images not in our training cache — expected.

Visual inspection confirmed:
- Clean, convex cardiac silhouettes on all 8 displayed images
- Accurate lung boundaries, well-separated from cardiac region
- No spurious inclusions (no rib or aorta bleeding into heart mask)
- One lateral-view image (unusual for VinDr-CXR) handled gracefully by HybridGNet

### 4.7 CheXmask vs. MedSAM: Final Comparison

| Axis | MedSAM (anatomical prior) | CheXmask |
|---|---|---|
| Signal quality | Heuristic (fixed prior box) | Physician-validated (DSC 0.955–0.967) |
| Noise characterisation | Diverse, unpredictable | Known, quantified via Dice RCA |
| Noise filterability | None (IoU unreliable) | Yes — filter by Dice RCA threshold |
| VinDr-CXR coverage | Any image (run inference) | 14,029 training images (pre-generated) |
| Inference cost | ~2s/image on GPU | Zero |
| CTR derivability | Approximate (no thorax mask) | Exact (heart + lung masks both available) |

CheXmask is strictly superior for training supervision on VinBigData.

---

## 5. Next Steps — Mask Supervision Track

### 5.1 Root Cause Disambiguation First (Cheap)

Before implementing mask supervision, test H2 (loss weight, not signal quality):

```bash
# Resume from D3 ep120 checkpoint
# Raise weight_bbox_attn: 0.10 → 1.0 (10× increase)
# Train for 20 epochs
# Re-run scripts/latent_traversal_medsam.py
```

If mask area Δ increases beyond ±50 px → H2 confirmed → raise weight and continue on main track.
If still flat → H1/H3 → proceed to mask supervision below.

### 5.2 Offline Mask Generation

Decode CheXmask RLE for all 13,970 usable images and save as binary `.npy` masks:

```python
# Output: /datasets/mmolefe/chexmask/masks/{image_id}_heart.npy  (512×512 uint8)
#         /datasets/mmolefe/chexmask/masks/{image_id}_lung.npy   (512×512 uint8)
#         /datasets/mmolefe/chexmask/ctr_labels.csv             (image_id, ctr, dice_rca)
```

### 5.3 CTR Regression Head on z_disease

Add a small MLP head to the SepVAEV2 training loop:

```python
# During training (for images with cardiomegaly annotation):
z_d_gap = z_disease.mean(axis=(1, 2))          # (B, C) — global average pool
ctr_pred = MLP(z_d_gap[:, 0:1])                # single scalar per image
loss_ctr = weight_ctr_reg * |ctr_pred - ctr_gt|  # L1 against CheXmask-derived CTR
```

The weight `weight_ctr_reg` needs to be tuned to compete with KL + reconstruction.
Start at 1.0 and monitor whether z_disease traversal begins to produce mask area changes.

### 5.4 Optional — Mask Overlap Loss

Replace the flat-rectangle attention target in `BboxCrossAttnHead` with the CheXmask
heart mask directly. The attention map is already a spatial 2D field — comparing it to
a binary mask with a cross-entropy or Dice loss is architecturally clean.

```python
# Instead of: attend within bbox rectangle
# Do:         match attention map to heart mask (downsampled to attention resolution)
loss_mask_attn = DiceLoss(attention_map, heart_mask_downsampled)
```

This is the highest-information supervision variant and closes the geometric specificity gap
completely, but requires more implementation work and a new attention map logging diagnostic.

---

## 6. Branch and Tag Registry

| Artifact | Name | Description |
|---|---|---|
| Branch | `feature/mask-supervision` | All mask investigation work — scripts, docs, results |
| Tag | `explore/mask-supervision-v1` | End of exploration phase. Diagnostic complete. |
| Tag | `explore/mask-supervision-v2` | GT heart-mask supervision integrated into the shared-decoder SepVAEV2 training path |
| Milestone tag | `milestone/mask-supervision-gt-v1` | First successful GT heart-mask pass replacing bbox-guided spatial supervision in the encoder path |

### 6.1 Milestone Status

`milestone/mask-supervision-gt-v1` records the first successful pass where CheXmask
heart masks replace coarse bbox-guided spatial supervision in the encoder-side routing.

This milestone is intentionally recorded before the compositional V3 redesign. The
current SepVAEV2 still uses a shared decoder, so leakage from heart-specific structure
back into `z_common` remains structurally possible even though the encoder branches are
now spatially mask-routed by GT heart masks.

**Scripts on this branch:**

| Script | Purpose |
|---|---|
| `scripts/test_medsam_cardiac.py` | MedSAM cardiac segmentation (anatomical prior, pool IoU ranking) |
| `scripts/latent_traversal_medsam.py` | SepVAEV2 latent traversal + MedSAM area measurement |
| `scripts/download_chexmask.sh` | PhysioNet VinDr-CXR mask download (~309 MB) |
| `scripts/verify_chexmask.py` | CheXmask ID matching, RLE decode, CTR computation, visualisation |

**Results produced:**

| File | Description |
|---|---|
| `results/medsam_cardiac_anatomical.png` | MedSAM with anatomical prior — top 5 by IoU |
| `results/medsam_cardiac_expanded.png` | MedSAM with expanded bbox — top 5 by IoU |
| `results/medsam_cardiac_seed123.png` | MedSAM anatomical prior — different seed, 5 new images |
| `results/verify_chexmask.png` | CheXmask VinDr-CXR — top 8 by Dice RCA |

**Dataset on cluster:**

| Path | Description |
|---|---|
| `/datasets/mmolefe/chexmask/VinDr-CXR_preprocessed.csv` | CheXmask masks, 309 MB, 18,000 rows |

---

*End of document. Return to [Index](INDEX.md).*
