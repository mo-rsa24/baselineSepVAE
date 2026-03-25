# Data Preprocessing Summary

> Referenced from: [PLAN_D5.md](PLAN_D5.md)

All preprocessing logic lives in [datasets/VinBigData.py](datasets/VinBigData.py) and [scripts/cache_dicoms.py](scripts/cache_dicoms.py).

---

## Pipeline Overview

Raw VinBigData DICOMs go through the following steps before reaching the model. The net result is a `float32 (1, H, W)` tensor in `[-1, 1]`, polarity-corrected, windowed, and size-standardized.

---

### Step 1 — MONOCHROME1 Polarity Fix

Some VinBigData DICOMs use `MONOCHROME1` encoding where intensity is inverted (bones dark, air bright — opposite of clinical convention). We detect this from the DICOM header and invert:

```python
if photometric == 'MONOCHROME1':
    arr = arr.max() - arr
```

Applied in both `_load_dicom()` ([datasets/VinBigData.py:338](datasets/VinBigData.py#L338)) and `cache_dicoms.py` ([scripts/cache_dicoms.py:49](scripts/cache_dicoms.py#L49)) **before** any windowing, ensuring all images share the same `bones-bright / air-dark` profile regardless of scanner origin.

---

### Step 2 — VOI-LUT Windowing

Raw DICOM X-rays have pixel values spanning 0–4095. We map them to `[0, 1]` using windowing:

- **Preferred**: use `WindowCenter` / `WindowWidth` from the DICOM metadata
- **Fallback**: use 1st–99th percentile of pixel values

The fallback is critical — the hard-coded CT defaults (`wc=40 / ww=400`) would clip all X-ray pixels to white if used directly.

```python
windowed = np.clip(pixel_array, lower, upper)
windowed = (windowed - lower) / (upper - lower + 1e-8)  # → [0, 1]
```

See [datasets/VinBigData.py:344-356](datasets/VinBigData.py#L344-L356).

---

### Step 3 — Blank / Corrupt Image Rejection

Images with `std < 0.02` after windowing are rejected as blank or uniform. DICOMs that fail to parse entirely are also dropped. Both are logged and replaced via retry sampling from the same class pool (up to 10 retries).

```python
if windowed.std() < 0.02:
    return None  # triggers retry
```

See [datasets/VinBigData.py:359](datasets/VinBigData.py#L359) and [datasets/VinBigData.py:305](datasets/VinBigData.py#L305).

---

### Step 4 — Resize

Images are resized to `img_size × img_size` (default 512×512, training uses 256×256):

- **Live DICOM path**: PIL `BICUBIC` interpolation ([datasets/VinBigData.py:377](datasets/VinBigData.py#L377))
- **Cache generation**: PIL `LANCZOS` interpolation ([scripts/cache_dicoms.py:59](scripts/cache_dicoms.py#L59))

---

### Step 5 — Normalization to `[-1, 1]`

`_preprocess_image()` maps the `[0, 1]` float array to `[-1, 1]`:

```python
x = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]
x = x * 2.0 - 1.0                               # [-1, 1]
x = x[None, ...]                                 # (1, H, W)
```

This is the expected input range for the CheSS backbone and the VAE decoder. See [datasets/VinBigData.py:380-382](datasets/VinBigData.py#L380-L382).

---

### Step 6 — Cache Path (`.npy` files)

`cache_dicoms.py` pre-applies Steps 1–4 and saves as `uint16` numpy arrays (pixel values in `[0, 65535]`):

```python
arr = arr / arr.max() * 65535.0
np.save(dest, arr.astype(np.uint16))
```

At load time, `_load_npy()` recovers `[0, 1]` by dividing by `65535.0`, then `_preprocess_image()` maps to `[-1, 1]`. This avoids re-reading and re-windowing DICOMs on every training step. See [datasets/VinBigData.py:301](datasets/VinBigData.py#L301).

---

### Step 7 — Bounding Box Normalization

Bounding boxes from the raw CSV are stored in pixel coordinates. We normalize to `[0, 1]` by dividing by the original DICOM `H` / `W`:

```python
bbox = [x0 / W_orig, y0 / H_orig, x1 / W_orig, y1 / H_orig]
```

The cache CSV (`train_filtered.csv`) stores these pre-normalized values directly so no runtime division is needed. See [datasets/VinBigData.py:261-264](datasets/VinBigData.py#L261-L264) and [scripts/cache_dicoms.py:155-163](scripts/cache_dicoms.py#L155-L163).

---

## Two Loading Modes

| Mode | `use_cache` | Source | I/O cost |
|---|---|---|---|
| Live DICOM | `False` | `.dicom` files, VOI-LUT on the fly | High (pydicom per sample) |
| Cache NPY | `True` | Pre-cached `.npy` uint16 arrays | Low (numpy load only) |

Use `scripts/cache_dicoms.py` to generate the cache before training.
