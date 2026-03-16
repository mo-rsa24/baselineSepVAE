"""
sepvae_diagnostics.py
=====================
Diagnostic visualisations for SepVAE disentanglement evaluation.

During-training  (low overhead, called from train_sep_vae.py every N epochs):
    plot_latent_swap_grid(...)       – 3×6 compositional decode grid
    plot_per_channel_kl_heatmap(...) – per-channel KL split by disease class
    plot_head_norm_by_class(...)     – head L2-norms per class (bar chart)

Post-training only (run once after training is complete):
    plot_ablation_heatmap_gt(...)    – decoder ablation diff-map vs GT bounding boxes + IoU
    plot_gradcam_disease_latent(...) – GradCAM of disease latent norm w.r.t. input pixels
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")                 # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers shared across all diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _make_variables(vae_params, vae_batch_stats: dict) -> dict:
    """Bundle params + batch_stats into a Flax variables dict."""
    variables = {"params": vae_params}
    if vae_batch_stats:
        variables["batch_stats"] = vae_batch_stats
    return variables


def _encode_one(model, variables: dict, x: jnp.ndarray) -> dict:
    """
    Encode a single-image batch through the model encoder.

    Args:
        x: (1, H, W, 1) JAX array in [-1, 1]

    Returns:
        dict with keys 'mu_common', 'mu_cardio', 'mu_plthick',
        each (1, 64, 64, C) JAX array.
    """
    ld = model.apply(variables, x, method=model.encode)
    return {
        "mu_common":   ld["common"][0],
        "mu_cardio":   ld["cardiomegaly"][0],
        "mu_plthick": ld["plthick"][0],
    }


def _encode_three_classes(model, variables: dict, batch: dict) -> dict:
    """
    Encode one representative from each class (Normal / Effusion / Cardiomegaly).

    batch keys expected: 'x_norm', 'x_disease1', 'x_disease2'
    All values are (B, H, W, 1) JAX arrays.

    Returns dict keyed by 'normal', 'plthick', 'cardiomegaly', each with
    'input' (1, H, W, 1) and 'mu_common', 'mu_cardio', 'mu_plthick' latents.
    """
    result = {}
    for key, cls_name in [
        ("x_norm",     "normal"),
        ("x_disease1", "plthick"),
        ("x_disease2", "cardiomegaly"),
    ]:
        x = batch[key][:1]        # take first image of the class: (1, H, W, 1)
        enc = _encode_one(model, variables, x)
        enc["input"] = x
        result[cls_name] = enc
    return result


def _decode(model, variables: dict,
            mu_common, mu_cardio, mu_plthick) -> jnp.ndarray:
    """
    Decode from three latent head means (no reparameterisation sampling).

    Concatenates along the channel axis and calls model.decode.
    z layout: [z_common | z_cardio | z_effusion] → (B, 64, 64, C_c+2*C_d)
    """
    z = jnp.concatenate([mu_common, mu_cardio, mu_plthick], axis=-1)
    return model.apply(variables, z, method=model.decode)


def _to_uint8(x_jax, input_range: str = "01") -> np.ndarray:
    """
    Convert a (H, W) or (H, W, 1) JAX/numpy array to uint8 HWC for display.

    input_range: '01'  → input already in [0, 1]
                 '11'  → input in [-1, 1], shifted to [0, 1]
    """
    x = np.array(x_jax, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, :, 0]             # (H, W, 1) → (H, W)
    if input_range == "11":
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DURING-TRAINING DIAGNOSTIC 1: Latent Swap Grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_swap_grid(
    model,
    vae_params,
    vae_batch_stats: dict,
    batch: dict,
    z_channels_common: int = 4,
    z_channels_disease: int = 2,
    epoch: int = 0,
) -> plt.Figure:
    """
    3-row × 6-column compositional latent swap grid.

    Rows:  source image  (Normal / Effusion / Cardiomegaly)
    Cols:
        0  Input (ground truth)
        1  Recon  — decode(z_c_src, z_card_src, z_eff_src)   identity check
        2  Common only — decode(z_c_src, 0, 0)               what common encodes alone
        3  +Eff head   — decode(z_c_src, 0, z_eff_E)         transplant effusion from E
        4  +Cardio     — decode(z_c_src, z_card_C, 0)        transplant cardio from C
        5  Composite   — decode(z_c_src, z_card_C, z_eff_E)  both disease heads active

    Interpretation:
      Row Normal,    cols 3/4/5: do transplanted heads visibly add disease?
      Row Effusion,  col  2:     anatomy without disease (nulling check)
      Row Cardio,    col  2:     same nulling check
      All rows, col 1 vs 0:      reconstruction quality baseline
    """
    variables = _make_variables(vae_params, vae_batch_stats)
    enc = _encode_three_classes(model, variables, batch)

    N = enc["normal"]
    E = enc["plthick"]
    C = enc["cardiomegaly"]

    zeros_card = jnp.zeros_like(N["mu_cardio"])
    zeros_eff  = jnp.zeros_like(N["mu_plthick"])

    col_titles = [
        "Input",
        "Recon\n(own latents)",
        "Common only\n(no disease)",
        "+Eff head\n(from E img)",
        "+Cardio head\n(from C img)",
        "Composite\n(+Eff +Cardio)",
    ]
    row_labels = ["Normal (N)", "Effusion (E)", "Cardiomegaly (C)"]

    rows_data = []
    for src in [N, E, C]:
        zc   = src["mu_common"]
        zk   = src["mu_cardio"]
        ze   = src["mu_plthick"]

        cells = [
            _to_uint8(src["input"][0], input_range="11"),
            _to_uint8(_decode(model, variables, zc, zk, ze)[0], input_range="01"),
            _to_uint8(_decode(model, variables, zc, zeros_card, zeros_eff)[0], input_range="01"),
            _to_uint8(_decode(model, variables, zc, zeros_card, E["mu_plthick"])[0], input_range="01"),
            _to_uint8(_decode(model, variables, zc, C["mu_cardio"], zeros_eff)[0], input_range="01"),
            _to_uint8(_decode(model, variables, zc, C["mu_cardio"], E["mu_plthick"])[0], input_range="01"),
        ]
        rows_data.append(cells)

    n_rows, n_cols = 3, 6
    cell_px = 2.3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_px, n_rows * cell_px + 1.0),
        gridspec_kw={"wspace": 0.04, "hspace": 0.08},
    )

    for r, (row_cells, row_lbl) in enumerate(zip(rows_data, row_labels)):
        for c, img in enumerate(row_cells):
            ax = axes[r][c]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=7.5, pad=4)
            if c == 0:
                ax.set_ylabel(row_lbl, fontsize=9, labelpad=6)

            # Highlight the identity column with a faint border
            if c == 1:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#888888")
                    spine.set_linewidth(0.8)

    fig.suptitle(
        f"Latent Swap Grid — Epoch {epoch}\n"
        r"Tests compositional disentanglement: rows = source image, "
        r"cols = latent combination",
        fontsize=9, y=1.01,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DURING-TRAINING DIAGNOSTIC 2: Per-Channel KL Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_channel_kl_heatmap(
    model,
    vae_params,
    vae_batch_stats: dict,
    batch: dict,
    epoch: int = 0,
) -> plt.Figure:
    """
    Per-channel KL divergence heatmap, split by input disease class.

    Layout: 3 subplots side by side (one per head: Common / Cardiomegaly / Effusion).
      Rows (Y-axis): input class  (Normal / Effusion / Cardiomegaly)
      Cols (X-axis): latent channel index
      Colour:        mean KL value, averaged over batch and spatial dimensions

    Expected disentanglement pattern:
      - Common head:       roughly uniform KL across all three classes
      - Cardiomegaly head: HIGH KL for Cardiomegaly inputs, low for Normal/Effusion
      - Effusion head:     HIGH KL for Effusion inputs, low for Normal/Cardiomegaly

    A channel with KL ≈ 0 across all classes is dead (posterior collapse).
    """
    variables = _make_variables(vae_params, vae_batch_stats)

    class_inputs = {
        "Normal":       batch["x_norm"],
        "Effusion":     batch["x_disease1"],
        "Cardiomegaly": batch["x_disease2"],
    }
    class_order = ["Normal", "PlThick", "Cardiomegaly"]
    head_keys   = ["common", "cardiomegaly", "plthick"]
    head_titles = {
        "common":       "Common head",
        "cardiomegaly": "Cardiomegaly head",
        "plthick":      "PlThick head",
    }

    # kl_by_head_class[head][class] = (C,) numpy array
    kl_by_head_class: Dict[str, Dict[str, np.ndarray]] = {h: {} for h in head_keys}

    for cls_name, x in class_inputs.items():
        ld = model.apply(variables, x, method=model.encode)
        for head in head_keys:
            mu, logvar = ld[head]
            # Element-wise KL: 0.5*(μ² + exp(logvar) - 1 - logvar)
            kl_elem = 0.5 * (jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar)
            # Average over batch + spatial dims → per-channel scalar (C,)
            reduce_axes = tuple(range(kl_elem.ndim - 1))   # all dims except last
            kl_per_ch = np.array(jnp.mean(kl_elem, axis=reduce_axes))
            kl_by_head_class[head][cls_name] = kl_per_ch

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                             gridspec_kw={"wspace": 0.35})

    for ax, head in zip(axes, head_keys):
        mat = np.stack(
            [kl_by_head_class[head][cls] for cls in class_order], axis=0
        )  # (3, C)

        vmax = float(np.percentile(mat, 98)) if mat.max() > 0 else 1.0

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax,
                       interpolation="nearest")

        ax.set_title(head_titles[head], fontsize=10, pad=5)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f"ch{i}" for i in range(mat.shape[1])], fontsize=8)
        ax.set_yticks(range(len(class_order)))
        ax.set_yticklabels(class_order, fontsize=9)
        ax.set_xlabel("Channel", fontsize=9)

        # Annotate each cell with its KL value
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                txt_col = "white" if val > 0.65 * vmax else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_col, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("KL (nats)", fontsize=8)

    fig.suptitle(
        f"Per-Channel KL Divergence by Disease Class — Epoch {epoch}\n"
        "Ideal: disease heads show HIGH KL only for their active class",
        fontsize=9, y=1.03,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DURING-TRAINING DIAGNOSTIC 3: Head L2-Norm by Class
# ─────────────────────────────────────────────────────────────────────────────

def plot_head_norm_by_class(
    latents_common:   np.ndarray,
    latents_cardio:   np.ndarray,
    latents_plthick: np.ndarray,
    labels:           np.ndarray,
    epoch: int = 0,
) -> plt.Figure:
    """
    Grouped bar chart of mean ||μ_head||₂ per input disease class.

    Args:
        latents_*: (N, D) numpy arrays of spatially-pooled μ vectors,
                   as returned by _collect_latents_from_triplet_loader().
        labels:    (N,) integer class labels {0=Normal, 1=Effusion, 2=Cardiomegaly}

    Expected disentanglement pattern:
      - Cardiomegaly head: highest norm for class=2, low for 0 and 1
      - Effusion head:     highest norm for class=1, low for 0 and 2
      - Common head:       roughly equal across all classes (anatomy, not disease)
    """
    head_data = {
        "Common\nhead":        latents_common,
        "Cardiomegaly\nhead":  latents_cardio,
        "Effusion\nhead":      latents_plthick,
    }
    class_names = ["Normal (0)", "Effusion (1)", "Cardiomegaly (2)"]
    class_colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    x_pos = np.arange(len(head_data))
    n_classes = 3
    bar_w = 0.22
    offsets = np.array([-bar_w, 0.0, bar_w])

    fig, ax = plt.subplots(figsize=(9, 4.2))

    for cls_id, (cls_name, color, offset) in enumerate(
        zip(class_names, class_colors, offsets)
    ):
        means, errs = [], []
        for latents in head_data.values():
            mask = labels == cls_id
            if mask.sum() > 0:
                norms = np.linalg.norm(latents[mask], axis=1)
                means.append(norms.mean())
                errs.append(norms.std() / max(np.sqrt(mask.sum()), 1))
            else:
                means.append(0.0)
                errs.append(0.0)

        ax.bar(
            x_pos + offset, means, bar_w,
            label=cls_name, color=color, alpha=0.82,
            yerr=errs, capsize=4, ecolor="grey", error_kw={"linewidth": 1.2},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(head_data.keys()), fontsize=10)
    ax.set_ylabel("Mean  ‖μ‖₂  (±1 SEM)", fontsize=10)
    ax.set_title(
        f"Head L₂-Norm by Disease Class — Epoch {epoch}\n"
        "Ideal: each disease head is active (high norm) only for its own class",
        fontsize=9,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_consensus_boxes(
    csv_path:       str,
    disease_class_id: int,
    min_annotators: int = 3,
) -> Dict[str, dict]:
    """
    Load averaged bounding boxes from VinBigData CSV.
    Only returns images with >= min_annotators agreeing annotations.

    Returns:
        dict: image_id → {'x_min', 'x_max', 'y_min', 'y_max'}
              Coordinates are in native DICOM pixel space.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    disease_df = df[(df["class_id"] == disease_class_id) & df["x_min"].notna()].copy()

    counts = disease_df.groupby("image_id")["rad_id"].count()
    valid_ids = counts[counts >= min_annotators].index
    disease_df = disease_df[disease_df["image_id"].isin(valid_ids)]

    boxes = {}
    for img_id, grp in disease_df.groupby("image_id"):
        boxes[img_id] = {
            "x_min": float(grp["x_min"].mean()),
            "x_max": float(grp["x_max"].mean()),
            "y_min": float(grp["y_min"].mean()),
            "y_max": float(grp["y_max"].mean()),
        }
    return boxes


def _load_dicom_preprocessed(
    image_id: str,
    dicom_dir: str,
    img_size:  int = 512,
) -> Tuple[jnp.ndarray, int, int]:
    """
    Load a DICOM, apply windowing, resize to img_size × img_size.

    Returns:
        img_jax:   (1, img_size, img_size, 1) JAX array in [-1, 1]
        native_h:  original DICOM height in pixels
        native_w:  original DICOM width in pixels
    """
    import pydicom
    from PIL import Image as PILImage

    path = Path(dicom_dir) / f"{image_id}.dicom"
    dcm = pydicom.dcmread(str(path), force=True)
    pixel_array = dcm.pixel_array.astype(np.float32)
    native_h, native_w = pixel_array.shape

    wc = float(getattr(dcm, "WindowCenter", 40.0))
    ww = float(getattr(dcm, "WindowWidth", 400.0))
    if isinstance(wc, list): wc = float(wc[0])
    if isinstance(ww, list): ww = float(ww[0])

    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    img = np.clip(pixel_array, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)

    pil = PILImage.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((img_size, img_size), PILImage.BICUBIC)
    img_512 = np.array(pil, dtype=np.float32) / 255.0  # [0, 1]

    img_jax = jnp.array(img_512 * 2.0 - 1.0)[None, :, :, None]  # (1, S, S, 1)
    return img_jax, native_h, native_w


def _scale_box(box: dict, native_h: int, native_w: int, target: int = 512) -> dict:
    """Scale bounding box from native DICOM resolution to target × target."""
    sx, sy = target / native_w, target / native_h
    return {
        "x_min": box["x_min"] * sx,  "x_max": box["x_max"] * sx,
        "y_min": box["y_min"] * sy,  "y_max": box["y_max"] * sy,
    }


def _heatmap_iou(
    heatmap:       np.ndarray,
    box:           dict,
    threshold_pct: float = 20.0,
) -> float:
    """
    Compute IoU between the top-N% activation mask and the ground-truth box mask.

    threshold_pct = 20 → keep the brightest 20% of heatmap pixels as foreground.
    """
    H, W = heatmap.shape
    thresh = np.percentile(heatmap, 100.0 - threshold_pct)
    pred_mask = (heatmap >= thresh).astype(np.uint8)

    gt_mask = np.zeros((H, W), dtype=np.uint8)
    x0 = max(0, int(box["x_min"]));  x1 = min(W, int(box["x_max"]))
    y0 = max(0, int(box["y_min"]));  y1 = min(H, int(box["y_max"]))
    gt_mask[y0:y1, x0:x1] = 1

    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    return inter / (union + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING DIAGNOSTIC 4: Ablation Heatmap + GT Bounding Boxes
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_heatmap_gt(
    model,
    vae_params,
    vae_batch_stats:  dict,
    csv_path:         str,
    dicom_dir:        str,
    disease:          str  = "cardiomegaly",
    disease_class_id: int  = 3,
    n_samples:        int  = 8,
    min_annotators:   int  = 3,
    img_size:         int  = 512,
    threshold_pct:    float = 20.0,
    seed:             int  = 42,
) -> Tuple[plt.Figure, dict]:
    """
    Decoder ablation heatmap vs ground-truth bounding boxes (post-training).

    For each sampled image:
      1. Encode  →  (z_common, z_cardio, z_effusion)
      2. Full recon    = decode(z_common, z_cardio, z_effusion)
      3. Ablated recon = decode(z_common, ZEROS,    z_effusion)   [for cardiomegaly]
         or             decode(z_common, z_cardio,  ZEROS   )    [for effusion]
      4. heatmap = |full_recon − ablated_recon|  (absolute pixel difference)
      5. Scale GT box from native DICOM → 512 px and overlay
      6. IoU: thresholded heatmap mask vs GT box mask (top-threshold_pct% activations)

    Returns:
        fig:     matplotlib Figure with n_samples × 4 panel layout
        metrics: dict with 'iou_per_image' (dict) and 'mean_iou' (float)
    """
    variables = _make_variables(vae_params, vae_batch_stats)

    boxes_db = _load_consensus_boxes(csv_path, disease_class_id, min_annotators)
    rng = np.random.default_rng(seed)
    image_ids = list(boxes_db.keys())
    chosen = list(rng.choice(image_ids,
                             size=min(n_samples, len(image_ids)),
                             replace=False))

    col_titles = [
        f"Input ({img_size}×{img_size})",
        f"|Full recon − Ablated|\n({disease} head zeroed)",
        "Heatmap overlay\n(cyan = GT box)",
        "GT box only\n(radiologist consensus)",
    ]

    n_rows = len(chosen)
    fig = plt.figure(figsize=(4 * 3.2, n_rows * 3.2 + 0.6))
    gs  = gridspec.GridSpec(n_rows, 4, figure=fig, wspace=0.05, hspace=0.28)

    metrics_iou = {}

    for row_idx, img_id in enumerate(chosen):
        try:
            img_jax, native_h, native_w = _load_dicom_preprocessed(
                img_id, dicom_dir, img_size
            )
        except Exception as e:
            print(f"  [ablation] Could not load {img_id}: {e}")
            continue

        box_native = boxes_db[img_id]
        box_scaled = _scale_box(box_native, native_h, native_w, target=img_size)

        # Encode
        ld = model.apply(variables, img_jax, method=model.encode)
        mu_c    = ld["common"][0]
        mu_card = ld["cardiomegaly"][0]
        mu_ef  = ld["plthick"][0]

        # Full and ablated decode
        recon_full = np.array(
            _decode(model, variables, mu_c, mu_card, mu_ef)[0, :, :, 0]
        )
        if disease == "cardiomegaly":
            zeros = jnp.zeros_like(mu_card)
            recon_abl = np.array(
                _decode(model, variables, mu_c, zeros, mu_ef)[0, :, :, 0]
            )
        else:
            zeros = jnp.zeros_like(mu_ef)
            recon_abl = np.array(
                _decode(model, variables, mu_c, mu_card, zeros)[0, :, :, 0]
            )

        heatmap = np.abs(recon_full - recon_abl)
        iou = _heatmap_iou(heatmap, box_scaled, threshold_pct)
        metrics_iou[img_id] = iou

        img_display = np.clip(
            (np.array(img_jax[0, :, :, 0]) + 1.0) / 2.0, 0.0, 1.0
        )
        hmap_norm = heatmap / (heatmap.max() + 1e-8)

        bx_min = box_scaled["x_min"];  bx_max = box_scaled["x_max"]
        by_min = box_scaled["y_min"];  by_max = box_scaled["y_max"]
        bw = bx_max - bx_min;          bh = by_max - by_min

        def _make_rect():
            return Rectangle(
                (bx_min, by_min), bw, bh,
                linewidth=1.8, edgecolor="#00FFFF", facecolor="none",
            )

        # Col 0: input image
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")
        if row_idx == 0:
            ax0.set_title(col_titles[0], fontsize=8, pad=4)
        ax0.set_ylabel(
            f"…{img_id[-8:]}", fontsize=6,
            rotation=0, labelpad=55, va="center",
        )

        # Col 1: ablation heatmap (normalised)
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(hmap_norm, cmap="hot", vmin=0, vmax=1)
        ax1.axis("off")
        if row_idx == 0:
            ax1.set_title(col_titles[1], fontsize=8, pad=4)

        # Col 2: overlay (heatmap α-blended + GT box)
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax2.imshow(hmap_norm, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        ax2.add_patch(_make_rect())
        ax2.axis("off")
        ax2.set_title(
            col_titles[2] if row_idx == 0 else f"IoU = {iou:.3f}",
            fontsize=8, pad=4,
            color=("#00FFFF" if row_idx > 0 else "black"),
        )

        # Col 3: clean image + GT box only
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax3.add_patch(_make_rect())
        ax3.axis("off")
        if row_idx == 0:
            ax3.set_title(col_titles[3], fontsize=8, pad=4)

    mean_iou = float(np.mean(list(metrics_iou.values()))) if metrics_iou else float("nan")
    fig.suptitle(
        f"Decoder Ablation Heatmap vs Radiologist GT — {disease.title()}  "
        f"(≥{min_annotators}-annotator consensus)\n"
        f"Mean IoU (top-{threshold_pct:.0f}% activations): {mean_iou:.3f}  "
        f"[n={len(metrics_iou)} images]",
        fontsize=9, y=1.002,
    )
    return fig, {"iou_per_image": metrics_iou, "mean_iou": mean_iou}


# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING DIAGNOSTIC 5: GradCAM of Disease Latent Norm w.r.t. Input
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_disease_latent(
    vae_params,
    vae_batch_stats:    dict,
    images:             List[jnp.ndarray],
    disease_head:       str  = "cardiomegaly",
    z_channels_disease: int  = 2,
    gt_boxes:           Optional[List[dict]] = None,
    img_size:           int  = 512,
) -> plt.Figure:
    """
    GradCAM of disease latent norm with respect to input pixels (post-training).

    Directly calls backbone + ConvHead without the model's stop_gradient wrapper,
    so that gradients flow back to the input image.

    Score function: S(x) = mean(|μ_disease(x)|²)
    GradCAM map:   G(x) = |∂S/∂x|  (absolute input gradient, no ReLU needed
                                      since we're differentiating a norm)
    Smoothed and normalised to [0, 1] for display.

    Args:
        vae_params:         Full VAE parameter pytree
        vae_batch_stats:    Full VAE batch_stats pytree
        images:             List of (1, H, W, 1) JAX arrays in [-1, 1]
        disease_head:       'cardiomegaly' or 'plthick'
        z_channels_disease: Number of channels in disease latent heads
        gt_boxes:           Optional list of dicts {'x_min', 'x_max', 'y_min', 'y_max'}
                            in img_size pixel coordinates
        img_size:           Input image spatial size (default 512)

    Note:
        The frozen backbone applies stop_gradient during normal training.
        This function bypasses that by calling ResNet50CheSS and ConvHead
        directly with the stored parameters, enabling input gradients.
    """
    from models.resnet_jax import ResNet50CheSS
    from models.sep_vae_jax import ConvHead

    backbone = ResNet50CheSS()
    conv_head = ConvHead(out_channels=z_channels_disease)

    backbone_params = vae_params["encoder"]["backbone"]
    backbone_bs     = vae_batch_stats.get("encoder", {}).get("backbone", {})
    head_params     = vae_params["encoder"][f"head_{disease_head}"]

    backbone_vars = {"params": backbone_params, "batch_stats": backbone_bs}

    def _score(x_input: jnp.ndarray) -> jnp.ndarray:
        """Scalar score: mean squared activation of disease latent head."""
        h = backbone.apply(backbone_vars, x_input, return_spatial=True)
        mu, _ = conv_head.apply({"params": head_params}, h)
        return jnp.mean(jnp.square(mu))

    grad_fn = jax.grad(_score)

    n = len(images)
    n_cols = 4 if gt_boxes is not None else 3
    col_titles_base = ["Input", "GradCAM", "Overlay"]
    col_titles = col_titles_base + (["GT box"] if gt_boxes is not None else [])

    fig, axes = plt.subplots(
        n, n_cols,
        figsize=(n_cols * 3.0, n * 3.2 + 0.5),
        gridspec_kw={"wspace": 0.05, "hspace": 0.18},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, x in enumerate(images):
        grad   = grad_fn(x)                             # (1, H, W, 1)
        gcam   = np.abs(np.array(grad[0, :, :, 0]))    # (H, W)
        gcam   = gcam / (gcam.max() + 1e-8)

        img_display = _to_uint8(x[0], input_range="11")

        ax_in  = axes[i, 0]
        ax_gc  = axes[i, 1]
        ax_ov  = axes[i, 2]

        ax_in.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax_in.axis("off")

        ax_gc.imshow(gcam, cmap="hot", vmin=0, vmax=1)
        ax_gc.axis("off")

        ax_ov.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax_ov.imshow(gcam, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        ax_ov.axis("off")

        if gt_boxes is not None and i < len(gt_boxes):
            ax_gt = axes[i, 3]
            box   = gt_boxes[i]
            ax_gt.imshow(img_display, cmap="gray", vmin=0, vmax=1)
            rect = Rectangle(
                (box["x_min"], box["y_min"]),
                box["x_max"] - box["x_min"],
                box["y_max"] - box["y_min"],
                linewidth=1.8, edgecolor="#00FFFF", facecolor="none",
            )
            ax_gt.add_patch(rect)
            # IoU for the GradCAM map
            iou = _heatmap_iou(gcam, box, threshold_pct=20.0)
            ax_gt.axis("off")
            ax_gt.set_title(f"IoU={iou:.3f}", fontsize=8, color="#00FFFF")

        if i == 0:
            for ax, title in zip(axes[0, :n_cols], col_titles):
                ax.set_title(title, fontsize=8, pad=4)

    fig.suptitle(
        f"GradCAM: ∂‖μ_{disease_head}‖²/∂x  (encoder-side spatial attribution)\n"
        "Shows which input regions most drive the disease latent activation",
        fontsize=9, y=1.002,
    )
    return fig
