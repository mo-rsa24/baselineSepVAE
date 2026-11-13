"""
visualize_domains.py — Dataset Visualisation for the d0–d5 Generalisation Suite
=================================================================================
Produces figures for all 6 domains (A–F) showing what each dataset looks like
and why it is a generalised analog of the CXR compositional disentanglement problem.

Figures generated
-----------------
    domain_{A-F}_panel.png    — Full panel per domain: Normal/F1/F2/Both rows + info row
    overview_all_domains.png  — 1 sample per class per domain, side-by-side
    composition_task.png      — What the encoder must separate (source images + GT Both)
    cxr_analogy_table.png     — Assumption satisfaction table (CXR → each domain)

Usage
-----
    cd /home-mscluster/mmolefe/Playground/PhD/miniSepVAE
    conda run -n cxr python scripts/visualize_domains.py --out_dir figures/domains

    # Include CelebA domains (requires download ~1.7 GB):
    conda run -n cxr python scripts/visualize_domains.py --celeba_root ~/.cache/celeba
"""

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

LABEL_NORMAL  = 0
LABEL_FACTOR1 = 1
LABEL_FACTOR2 = 2
LABEL_BOTH    = 3

PALETTE = {
    "normal":  "#888888",
    "factor1": "#2196F3",
    "factor2": "#E91E63",
    "both":    "#9C27B0",
}


# =============================================================================
# Display helpers
# =============================================================================

def _to_display(x_tensor):
    """
    Convert (C, H, W) tensor in [-1,1] to numpy suitable for imshow.
    Returns (arr, cmap) where arr is (H,W) for greyscale or (H,W,3) for RGB.
    """
    arr = np.clip((x_tensor.squeeze(0).numpy() + 1) / 2, 0, 1)
    if arr.ndim == 2:                          # (H, W) — squeezed greyscale
        return arr, "gray"
    if arr.shape[0] == 1:                      # (1, H, W)
        return arr[0], "gray"
    return arr.transpose(1, 2, 0), None        # (3, H, W) → (H, W, 3)


def _gather(ds, label, n=5):
    """Return up to n display images (arr, cmap) for the given label."""
    idxs = np.where(ds.labels == label)[0][:n]
    return [_to_display(ds[i][0]) for i in idxs]


def _add_bbox_rect(ax, bbox_norm, H, W, color, lw=2):
    x0, y0, x1, y1 = bbox_norm
    ax.add_patch(patches.Rectangle(
        (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
        linewidth=lw, edgecolor=color, facecolor="none", linestyle="--"))


def _channel_hist(ax, ds, label_colors):
    """
    Plot R/B channel histograms for each class — used for ColouredMNIST info row.
    Shows clearly that F1 shifts R (warm) and F2 shifts B (cool).
    """
    bins = np.linspace(0, 1, 40)
    label_map = {LABEL_NORMAL: ("Normal", label_colors[0]),
                 LABEL_FACTOR1: ("F1 warm", label_colors[1]),
                 LABEL_FACTOR2: ("F2 cool", label_colors[2]),
                 LABEL_BOTH: ("Both", label_colors[3])}
    for lbl, (name, lc) in label_map.items():
        idxs = np.where(ds.labels == lbl)[0][:30]
        if len(idxs) == 0:
            continue
        # Stack raw tensors
        imgs = np.stack([(ds[i][0].numpy() + 1) / 2 for i in idxs])  # (N,3,H,W)
        r_mean = imgs[:, 0].mean(axis=(1, 2))
        b_mean = imgs[:, 2].mean(axis=(1, 2))
        ax.hist(r_mean, bins=bins, alpha=0.55, color="tomato",
                density=True, histtype="step", linewidth=1.5)
        ax.hist(b_mean, bins=bins, alpha=0.55, color="steelblue",
                density=True, histtype="step", linewidth=1.5)
    ax.set_xlabel("channel mean", fontsize=6)
    ax.set_yticks([])
    ax.set_title("R(─) B(─) per class", fontsize=6.5)


# =============================================================================
# Per-domain panel
# =============================================================================

def plot_domain(ds_val, domain_letter, title, subtitle,
                cxr_analogs, assumption_status, has_bbox, is_rgb,
                bbox_f1, bbox_f2, slot_rows_f1, slot_cols_f1,
                slot_rows_f2, slot_cols_f2,
                out_path: Path, n_cols: int = 5):
    """
    5-row panel:
      Rows 0–3: Normal / F1 / F2 / Both, n_cols examples each
      Row 4:    Spatial info (bbox overlay or channel histogram)
    """
    labels_to_show = [LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH]
    row_names  = ["Normal", "Factor-1 only", "Factor-2 only", "Both (GT)"]
    row_colors = [PALETTE["normal"], PALETTE["factor1"],
                  PALETTE["factor2"], PALETTE["both"]]
    label_colors = list(row_colors)

    n_rows = 5
    fig = plt.figure(figsize=((n_cols + 1) * 2.0, n_rows * 2.3 + 1.4))
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  hspace=0.10, wspace=0.07,
                  left=0.09, right=0.98, top=0.92, bottom=0.04)

    fig.suptitle(f"Domain {domain_letter}  —  {title}\n{subtitle}",
                 fontsize=12, fontweight="bold", y=0.97, color="#1A1A2E")

    # ── Image rows ────────────────────────────────────────────────────────
    for row, (lbl, rname, rcol) in enumerate(
            zip(labels_to_show, row_names, row_colors)):
        imgs = _gather(ds_val, lbl, n=n_cols)
        for col, (img_arr, cmap) in enumerate(imgs):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_arr, cmap=cmap, vmin=0, vmax=1, aspect="equal")
            if has_bbox:
                H, W = (img_arr.shape[:2] if img_arr.ndim == 2
                        else img_arr.shape[:2])
                if lbl in (LABEL_FACTOR1, LABEL_BOTH) and bbox_f1[2] > 0:
                    _add_bbox_rect(ax, bbox_f1, H, W, PALETTE["factor1"])
                if lbl in (LABEL_FACTOR2, LABEL_BOTH) and bbox_f2[2] > 0:
                    _add_bbox_rect(ax, bbox_f2, H, W, PALETTE["factor2"])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(rcol); sp.set_linewidth(2.5)

        # Row annotation
        ax_ann = fig.add_subplot(gs[row, n_cols])
        ax_ann.axis("off")
        ax_ann.text(0.05, 0.78, rname, fontsize=9, fontweight="bold",
                    color=rcol, va="top")
        ax_ann.text(0.05, 0.52, f"CXR:\n{cxr_analogs[row]}",
                    fontsize=7, color="#444444", va="top",
                    style="italic", linespacing=1.4)

    # ── Spatial info row ──────────────────────────────────────────────────
    imgs_f1 = _gather(ds_val, LABEL_FACTOR1, n=n_cols)
    for col in range(min(n_cols, len(imgs_f1))):
        ax_sp = fig.add_subplot(gs[4, col])
        img_arr, cmap = imgs_f1[col]
        H = W = img_arr.shape[0]

        if is_rgb and not has_bbox:
            # For ColouredMNIST: show the RGB image itself (colour is the factor)
            ax_sp.imshow(img_arr, vmin=0, vmax=1, aspect="equal")
        elif has_bbox:
            ax_sp.imshow(img_arr, cmap=cmap, vmin=0, vmax=1, aspect="equal")
            _add_bbox_rect(ax_sp, bbox_f1, H, W, PALETTE["factor1"])
            _add_bbox_rect(ax_sp, bbox_f2, H, W, PALETTE["factor2"])
            if slot_rows_f1 is not None:
                r0, r1 = slot_rows_f1; c0, c1 = slot_cols_f1
                ax_sp.add_patch(patches.Rectangle(
                    (c0, r0), c1-c0, r1-r0,
                    facecolor=PALETTE["factor1"], alpha=0.20, linewidth=0))
            if slot_rows_f2 is not None:
                r0, r1 = slot_rows_f2; c0, c1 = slot_cols_f2
                ax_sp.add_patch(patches.Rectangle(
                    (c0, r0), c1-c0, r1-r0,
                    facecolor=PALETTE["factor2"], alpha=0.20, linewidth=0))
        else:
            ax_sp.imshow(img_arr, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        ax_sp.set_xticks([]); ax_sp.set_yticks([])
        for sp in ax_sp.spines.values():
            sp.set_edgecolor("#888888"); sp.set_linewidth(1)

    # Spatial info annotation
    ax_sp_ann = fig.add_subplot(gs[4, n_cols])
    ax_sp_ann.axis("off")
    if is_rgb and not has_bbox:
        sp_label = "RGB colour\n(global factor)"
    elif has_bbox:
        sp_label = "Known bbox\n(spatial prior)"
    else:
        sp_label = "No bbox\n(global factor)"

    ax_sp_ann.text(0.05, 0.80, sp_label, fontsize=9, fontweight="bold",
                   color="#555555", va="top")
    ax_sp_ann.text(0.05, 0.50,
                   f"A1 ({'✓' if assumption_status[0] else '✗'})\n"
                   f"A2 ({'✓' if assumption_status[1] else '✗'})\n"
                   f"A3 ✓\nA4 ✓",
                   fontsize=8.5, color="#222222", va="top",
                   fontfamily="monospace", linespacing=1.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# Overview figure
# =============================================================================

def plot_overview(datasets_info: list, out_path: Path):
    """
    One canonical sample per class per domain.
    Rows = domains, Cols = Normal / F1 / F2 / Both / Info
    """
    col_titles = ["Normal\n(no factor)", "Factor-1 only\n(cardiomegaly)",
                  "Factor-2 only\n(pleural)", "Both\n(comorbid)", "Spatial\ninfo"]

    n_rows = len(datasets_info)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.7 + 1.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("#F8F9FA")

    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=9, fontweight="bold",
                               color="#1A1A2E", pad=6)

    for row, info in enumerate(datasets_info):
        ds       = info["ds_val"]
        has_bbox = info["has_bbox"]
        is_rgb   = info["is_rgb"]
        a1_ok    = info["a1_ok"]
        a2_ok    = info["a2_ok"]
        bbox_f1  = info["bbox_f1"]
        bbox_f2  = info["bbox_f2"]

        row_color = ("#2E7D32" if (a1_ok and a2_ok) else
                     "#E65100" if (a1_ok or a2_ok) else "#B71C1C")
        axes[row, 0].set_ylabel(
            f"Domain {info['domain']}\n{info['row_title']}\n({info['row_subtitle']})",
            fontsize=8, fontweight="bold", color=row_color, labelpad=6)

        for col, lbl in enumerate([LABEL_NORMAL, LABEL_FACTOR1,
                                   LABEL_FACTOR2, LABEL_BOTH]):
            ax = axes[row, col]
            imgs = _gather(ds, lbl, n=1)
            if imgs:
                img_arr, cmap = imgs[0]
                H = W = img_arr.shape[0]
                ax.imshow(img_arr, cmap=cmap, vmin=0, vmax=1, aspect="equal")
                if has_bbox:
                    if lbl in (LABEL_FACTOR1, LABEL_BOTH) and bbox_f1[2] > 0:
                        _add_bbox_rect(ax, bbox_f1, H, W, PALETTE["factor1"])
                    if lbl in (LABEL_FACTOR2, LABEL_BOTH) and bbox_f2[2] > 0:
                        _add_bbox_rect(ax, bbox_f2, H, W, PALETTE["factor2"])
            ax.set_xticks([]); ax.set_yticks([])
            ec = [PALETTE["normal"], PALETTE["factor1"],
                  PALETTE["factor2"], PALETTE["both"]][col]
            for sp in ax.spines.values():
                sp.set_edgecolor(ec); sp.set_linewidth(2.5)

        # Info column (col 4)
        ax_s = axes[row, 4]
        imgs_f1 = _gather(ds, LABEL_FACTOR1, n=1)
        if imgs_f1:
            img_arr, cmap = imgs_f1[0]
            H = W = img_arr.shape[0]
            ax_s.imshow(img_arr, cmap=cmap, vmin=0, vmax=1, aspect="equal")
            if has_bbox:
                if bbox_f1[2] > 0:
                    _add_bbox_rect(ax_s, bbox_f1, H, W, PALETTE["factor1"], lw=1.5)
                if bbox_f2[2] > 0:
                    _add_bbox_rect(ax_s, bbox_f2, H, W, PALETTE["factor2"], lw=1.5)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        # Label info col with what the prior knows
        label = ("bbox" if has_bbox else "global RGB" if is_rgb else "global")
        ax_s.set_xlabel(label, fontsize=7, color="#555555")
        for sp in ax_s.spines.values():
            sp.set_edgecolor("#888888"); sp.set_linewidth(1.5)

    plt.suptitle(
        "Generalisation Suite  —  6 Domains Testing d0–d5 Framework Assumptions\n"
        "Green = all assumptions met   Orange = one violation   Red = double violation",
        fontsize=11, fontweight="bold", y=1.01, color="#1A1A2E")
    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# Composition illustration figure
# =============================================================================

def plot_composition_logic(datasets_info: list, out_path: Path):
    """
    For each domain: Normal / F1 / F2 / GT-Both / |Both−Normal| difference map.
    """
    fig, axes_all = plt.subplots(
        len(datasets_info), 5,
        figsize=(5 * 2.3, len(datasets_info) * 2.5 + 0.9))
    if len(datasets_info) == 1:
        axes_all = axes_all[np.newaxis, :]
    fig.patch.set_facecolor("#F8F9FA")

    col_titles = ["Normal\n(z_c source)", "Factor-1\n(z_d1 source)",
                  "Factor-2\n(z_d2 source)", "GT Both\n(ideal output)",
                  "|Both − Normal|\n(what model must add)"]
    for col, ct in enumerate(col_titles):
        axes_all[0, col].set_title(ct, fontsize=8.5, fontweight="bold",
                                    color="#1A1A2E", pad=5)

    for row, info in enumerate(datasets_info):
        ds       = info["ds_val"]
        has_bbox = info["has_bbox"]
        is_rgb   = info["is_rgb"]
        bbox_f1  = info["bbox_f1"]
        bbox_f2  = info["bbox_f2"]
        axes     = axes_all[row]

        row_color = ("#2E7D32" if info["a1_ok"] and info["a2_ok"] else
                     "#E65100" if info["a1_ok"] or info["a2_ok"] else "#B71C1C")
        axes[0].set_ylabel(f"Domain {info['domain']}\n{info['row_title']}",
                           fontsize=8, fontweight="bold", color=row_color)

        srcs = []
        for lbl in [LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH]:
            imgs = _gather(ds, lbl, n=1)
            srcs.append(imgs[0] if imgs else (np.zeros((64, 64)), "gray"))

        bcs = [PALETTE["normal"], PALETTE["factor1"],
               PALETTE["factor2"], PALETTE["both"]]

        for col_i, ((img_arr, cmap), bc) in enumerate(zip(srcs, bcs)):
            ax = axes[col_i]
            H = W = img_arr.shape[0]
            ax.imshow(img_arr, cmap=cmap, vmin=0, vmax=1)
            if has_bbox:
                if col_i in (1, 3) and bbox_f1[2] > 0:
                    _add_bbox_rect(ax, bbox_f1, H, W, PALETTE["factor1"])
                if col_i in (2, 3) and bbox_f2[2] > 0:
                    _add_bbox_rect(ax, bbox_f2, H, W, PALETTE["factor2"])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2.5)

        # Difference column
        ax_diff = axes[4]
        img_n, _ = srcs[0]
        img_b, _ = srcs[3]
        if img_b is not None and img_n is not None:
            if img_n.ndim == 3:   # RGB: take luminance
                diff = np.abs(img_b.mean(-1) - img_n.mean(-1))
            else:
                diff = np.abs(img_b - img_n)
            ax_diff.imshow(diff, cmap="hot", vmin=0, vmax=0.6)
        ax_diff.set_xticks([]); ax_diff.set_yticks([])
        for sp in ax_diff.spines.values():
            sp.set_edgecolor("#555555"); sp.set_linewidth(1.5)

    plt.suptitle(
        "Composition Task:  Decode(z_c[Normal] + z_d1[F1] + z_d2[F2])  →  Both factors\n"
        "Column 5 shows what the encoder must learn to inject (|Both − Normal| luminance)",
        fontsize=10, fontweight="bold", y=1.01, color="#1A1A2E")
    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# CXR analogy table
# =============================================================================

def plot_cxr_analogy(datasets_info: list, out_path: Path):
    """
    Assumption satisfaction table: CXR + all domains.
    """
    domains_meta = [
        ("CXR\n(VinBigData)", "#78909C", True,  True),
    ] + [
        (f"Domain {i['domain']}\n{i['row_title']}",
         "#2E7D32" if (i['a1_ok'] and i['a2_ok']) else
         "#E65100" if (i['a1_ok'] or i['a2_ok']) else "#B71C1C",
         i['a1_ok'], i['a2_ok'])
        for i in datasets_info
    ]

    n_dom = len(domains_meta)
    col_w = 2.4
    fig, ax = plt.subplots(figsize=(1.8 + n_dom * col_w, 6.5))
    ax.set_xlim(0, 1.8 + n_dom * col_w)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    col_xs = [1.0 + (i + 0.5) * col_w for i in range(n_dom)]

    # Headers
    for cx, (name, col, _, _) in zip(col_xs, domains_meta):
        ax.text(cx, 6.05, name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE",
                          edgecolor=col, linewidth=1.5))

    # Row labels
    rows = [
        ("A1: Statistical\nindependence",   5.0,
         "CXR: classes\ncorrelated ~30%"),
        ("A2: Spatial\nlocality",            4.0,
         "CXR: cardiac bbox\npleural region"),
        ("A3: Null class\nexists",           3.0,
         "CXR: Label 14\n= Normal"),
        ("A4: Additive\ncomposition",        2.0,
         "CXR: pathology =\ndeviation from anatomy"),
    ]
    for label, ry, cxr_note in rows:
        ax.text(0.08, ry, label, ha="left", va="center",
                fontsize=8.5, fontweight="bold", color="#333333")

    # CXR reference column details
    cxr_details = [
        (True,  "Labels weakly\ncorrelated (~30%)"),
        (True,  "Cardiac bbox /\npleural region"),
        (True,  "Label 14 =\nNo disease"),
        (True,  "Pathology =\ndeviation"),
    ]

    # Per-domain details per assumption
    domain_details = {
        "A": [(True, "Independently\ngenerated"),
              (True, "Fixed slot\nexact bbox"),
              (True, "Label 0\n= no digit"),
              (True, "Pixel-level\naddition")],
        "B": [(False, "30% hidden\nco-occurrence"),
              (True, "Same as A\nbbox valid"),
              (True, "Label 0\n= no digit"),
              (True, "Pixel-level\naddition")],
        "C": [(True, "Scale ⊥ Orient\nindependent"),
              (False, "Global shape\nno bbox"),
              (True, "Label 0\nsmall+low-θ"),
              (True, "Geom.\nindependent")],
        "D": [(False, "50% hidden\nco-occurrence"),
              (False, "Global shape\nno bbox"),
              (True, "Label 0\nsmall+low-θ"),
              (True, "Geom.\nindependent")],
        "E": [(True, "Balanced\n(corr removed)"),
              (True, "Glasses bbox\nbald bbox"),
              (True, "Label 0\nno glasses, hair"),
              (True, "Attribute\nindependent")],
        "F": [(False, "Natural demo-\ngraphic corr."),
              (True, "Glasses bbox\nbald bbox"),
              (True, "Label 0\nno glasses, hair"),
              (True, "Attribute\nindependent")],
    }

    ry_map  = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
    all_details = [cxr_details] + [
        domain_details.get(i["domain"], [(True, "")] * 4)
        for i in datasets_info
    ]

    for col_i, (cx, (_, col, _, _)) in enumerate(zip(col_xs, domains_meta)):
        for ai, (sat, detail) in enumerate(all_details[col_i]):
            ry = ry_map[ai]
            bg   = "#E8F5E9" if sat else "#FFEBEE"
            fcol = "#1B5E20" if sat else "#B71C1C"
            sym  = "✓" if sat else "✗"
            ax.add_patch(patches.FancyBboxPatch(
                (cx - col_w * 0.46, ry - 0.43), col_w * 0.92, 0.86,
                boxstyle="round,pad=0.05",
                facecolor=bg, edgecolor=fcol, linewidth=1.1, zorder=2))
            ax.text(cx - col_w * 0.36, ry + 0.05, sym, fontsize=11,
                    fontweight="bold", color=fcol, va="center", zorder=3)
            ax.text(cx - col_w * 0.14, ry, detail, fontsize=6.2,
                    color="#333333", va="center", linespacing=1.3, zorder=3)

    # Grid
    for ry in [4.5, 3.5, 2.5]:
        ax.axhline(ry, color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.axvline(1.0 + col_w * 0.5, color="#AAAAAA", linewidth=1.2,
               linestyle=":", zorder=1)

    ax.text((1.8 + n_dom * col_w) / 2, 6.50,
            "d0–d5 Framework Assumptions: CXR → Each Domain",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#1A1A2E")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# CelebA bbox verification figure
# =============================================================================

def plot_celeba_bbox_verify(ds_E, ds_F, bbox_f1, bbox_f2, out_path: Path,
                             n_per_class: int = 6):
    """
    Grid of real CelebA images with F1 (glasses, blue) and F2 (bald, pink) bboxes
    overlaid, for each class in both balanced (E) and natural (F) splits.

    Purpose: visually confirm bbox coordinates sit correctly on the actual faces
    before running the full d0-d5 training.

    Layout:
        Rows  0–3: Domain E (balanced)  — Normal / F1 / F2 / Both
        Rows  4–7: Domain F (natural)   — Normal / F1 / F2 / Both
        Bbox key: blue dashed = glasses region, pink dashed = bald/crown region
    """
    row_labels = ["Normal", "F1\n(Eyeglasses)", "F2\n(Bald)", "Both"]
    row_colors = [PALETTE["normal"], PALETTE["factor1"],
                  PALETTE["factor2"], PALETTE["both"]]
    split_labels = ["Domain E\n(balanced)", "Domain F\n(natural)"]

    n_rows = 8
    n_cols = n_per_class + 1   # +1 for row annotation
    fig = plt.figure(figsize=(n_cols * 1.55, n_rows * 1.65 + 1.5))
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.08, wspace=0.06,
                  left=0.10, right=0.98, top=0.92, bottom=0.03)

    fig.suptitle(
        "CelebA Bbox Verification\n"
        f"Blue dashed = Eyeglasses region (F1)   ·   Pink dashed = Bald/crown region (F2)\n"
        f"Coords — F1: {tuple(round(v,2) for v in bbox_f1)}   "
        f"F2: {tuple(round(v,2) for v in bbox_f2)}",
        fontsize=11, fontweight="bold", y=0.98, color="#1A1A2E")

    for split_i, (ds, split_label) in enumerate([(ds_E, "E (balanced)"),
                                                   (ds_F, "F (natural)")]):
        for class_i, (lbl, rname, rcol) in enumerate(
                zip([LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH],
                    row_labels, row_colors)):
            row = split_i * 4 + class_i
            idxs = np.where(ds.labels == lbl)[0][:n_per_class]

            for col_i, idx in enumerate(idxs):
                ax = fig.add_subplot(gs[row, col_i])
                x_tensor = ds[int(idx)][0]
                img_arr, _ = _to_display(x_tensor)
                H, W = img_arr.shape[:2]

                ax.imshow(img_arr, vmin=0, vmax=1, aspect="equal")

                # Always draw both bboxes — viewer can judge alignment
                _add_bbox_rect(ax, bbox_f1, H, W, PALETTE["factor1"], lw=1.8)
                _add_bbox_rect(ax, bbox_f2, H, W, PALETTE["factor2"], lw=1.8)

                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(rcol); sp.set_linewidth(2.0)

            # Fill any empty columns with blank axes
            for col_i in range(len(idxs), n_per_class):
                ax = fig.add_subplot(gs[row, col_i])
                ax.axis("off")

            # Row annotation
            ax_ann = fig.add_subplot(gs[row, n_per_class])
            ax_ann.axis("off")
            ax_ann.text(0.05, 0.80, rname, fontsize=8, fontweight="bold",
                        color=rcol, va="top")
            if class_i == 0:
                ax_ann.text(0.05, 0.48, f"Split:\n{split_label}",
                            fontsize=6.5, color="#555555", va="top",
                            style="italic", linespacing=1.4)

    # Horizontal divider between E and F
    fig.add_artist(plt.Line2D(
        [0.10, 0.98], [0.505, 0.505],
        transform=fig.transFigure,
        color="#888888", linewidth=1.5, linestyle="--"))

    # Split labels on left margin
    fig.text(0.01, 0.73, "Domain E\n(balanced)", ha="left", va="center",
             fontsize=9, fontweight="bold", color="#2E7D32", rotation=90)
    fig.text(0.01, 0.27, "Domain F\n(natural)", ha="left", va="center",
             fontsize=9, fontweight="bold", color="#E65100", rotation=90)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# Factor space visualisation for dSprites (C/D)
# =============================================================================

def _pixel_mean(ds, lbl, n=80):
    idxs = np.where(ds.labels == lbl)[0][:n]
    return np.array([(ds[i][0].numpy() + 1).mean() / 2 for i in idxs])


def _orient_proxy(ds, lbl, n=80):
    """Principal-axis angle for each sample (in radians)."""
    idxs = np.where(ds.labels == lbl)[0][:n]
    angles = []
    H = W = 64
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float64)
    for i in idxs:
        m = ((ds[i][0].numpy() + 1) / 2)[0].astype(np.float64)  # (H,W)
        total = m.sum() + 1e-8
        cx = (m * xg).sum() / total
        cy = (m * yg).sum() / total
        mxx = (m * (xg - cx) ** 2).sum() / total
        myy = (m * (yg - cy) ** 2).sum() / total
        mxy = (m * (xg - cx) * (yg - cy)).sum() / total
        angles.append(0.5 * abs(np.arctan2(2 * mxy, mxx - myy + 1e-10)))
    return np.array(angles)


def plot_dsprites_factor_analysis(ds_c, ds_d, out_path: Path):
    """
    2-row × 2-col figure confirming factor separability in dSprites (C/D).

    Top row    — Domain C (independent):
        Left:  pixel mean distribution per class  → F1=large has higher mean
        Right: principal-axis angle per class     → F2=high-θ has larger angle

    Bottom row — Domain D (50% correlated):
        Same metrics; F1/F2 distributions partially overlap due to corruption.
    """
    labels  = [LABEL_NORMAL, LABEL_FACTOR1, LABEL_FACTOR2, LABEL_BOTH]
    names   = ["Normal\n(small,low-θ)", "F1\n(large,low-θ)",
               "F2\n(small,high-θ)", "Both\n(large,high-θ)"]
    palette = [PALETTE["normal"], PALETTE["factor1"],
               PALETTE["factor2"], PALETTE["both"]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "dSprites Factor Analysis — Domain C (independent) vs D (50% correlated)\n"
        "Left: pixel mean (scale proxy)   ·   Right: principal-axis angle (orientation proxy)",
        fontsize=10, fontweight="bold", y=1.02, color="#1A1A2E")

    col_titles = ["Pixel mean (↑ = larger shape = F1)",
                  "Principal-axis angle rad (↑ = rotated = F2)"]
    axes[0, 0].set_title(col_titles[0], fontsize=9)
    axes[0, 1].set_title(col_titles[1], fontsize=9)

    for ds_row, (ds, ds_name) in enumerate([
            (ds_c, "Domain C — independent"),
            (ds_d, "Domain D — 50% correlated")]):
        axes[ds_row, 0].set_ylabel(ds_name, fontsize=9, fontweight="bold")

        means_ax  = axes[ds_row, 0]
        angles_ax = axes[ds_row, 1]

        bins_m = np.linspace(0.0,  0.6, 35)
        bins_a = np.linspace(0.0,  0.8, 35)

        for lbl, name, pal in zip(labels, names, palette):
            pm = _pixel_mean(ds, lbl)
            oa = _orient_proxy(ds, lbl)
            if len(pm) == 0:
                continue
            means_ax.hist(pm, bins=bins_m, alpha=0.60, color=pal,
                          label=name, density=True, histtype="stepfilled")
            angles_ax.hist(oa, bins=bins_a, alpha=0.60, color=pal,
                           label=name, density=True, histtype="stepfilled")

        for ax, xlabel in [(means_ax, "pixel mean"),
                           (angles_ax, "angle (rad)")]:
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_yticks([])
            ax.legend(fontsize=6.5, ncol=2, loc="upper right")
            for sp in ax.spines.values():
                sp.set_color("#BBBBBB")

    plt.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",       type=str, default="figures/domains")
    p.add_argument("--n_cols",        type=int, default=5)
    p.add_argument("--seed",          type=int, default=99)
    p.add_argument("--mnist_root",    type=str, default="~/.cache/mnist")
    p.add_argument("--dsprites_root", type=str, default="~/.cache/dsprites")
    p.add_argument("--celeba_root",   type=str, default=None,
                   help="Path for CelebA. Omit to skip domains E/F.")
    args = p.parse_args()
    out = Path(args.out_dir)

    print("\n=== Building datasets (val splits) ===")
    kw = dict(n_normal=300, n_factor1=150, n_factor2=150, n_both=150,
              img_size=64, global_seed=args.seed, mode='flat')
    mnist_kw = dict(mnist_root=args.mnist_root)

    from datasets.SplatteredMNIST import (
        SplatteredMNISTDataset,
        BBOX_F1 as A_BBOX_F1, BBOX_F2 as A_BBOX_F2,
        SLOT_F1_ROWS, SLOT_F1_COLS, SLOT_F2_ROWS, SLOT_F2_COLS,
    )
    from datasets.CorrelatedSplatteredMNIST import CorrelatedSplatteredMNISTDataset
    from datasets.DSprites import DSpritesDataset
    from datasets.CorrelatedDSprites import CorrelatedDSpritesDataset

    dsprites_kw = dict(dsprites_root=args.dsprites_root)
    ds_A = SplatteredMNISTDataset(**kw, **mnist_kw)
    ds_B = CorrelatedSplatteredMNISTDataset(**kw, **mnist_kw, correlation_rate=0.30)
    ds_C = DSpritesDataset(**kw, **dsprites_kw)
    ds_D = CorrelatedDSpritesDataset(**kw, **dsprites_kw, correlation_rate=0.50)
    print(f"  A: {len(ds_A)} | B: {len(ds_B)} | C: {len(ds_C)} | D: {len(ds_D)}")

    _ZERO = (0.0, 0.0, 0.0, 0.0)

    datasets_info = [
        dict(ds_val=ds_A, domain="A", has_bbox=True,  is_rgb=False,
             a1_ok=True,  a2_ok=True,
             bbox_f1=A_BBOX_F1, bbox_f2=A_BBOX_F2,
             row_title="SplatteredMNIST", row_subtitle="All assumptions met"),
        dict(ds_val=ds_B, domain="B", has_bbox=True,  is_rgb=False,
             a1_ok=False, a2_ok=True,
             bbox_f1=A_BBOX_F1, bbox_f2=A_BBOX_F2,
             row_title="CorrSplatteredMNIST", row_subtitle="A1 violated (30% corr)"),
        dict(ds_val=ds_C, domain="C", has_bbox=False, is_rgb=False,
             a1_ok=True,  a2_ok=False,
             bbox_f1=_ZERO, bbox_f2=_ZERO,
             row_title="dSprites", row_subtitle="A2 violated (global)"),
        dict(ds_val=ds_D, domain="D", has_bbox=False, is_rgb=False,
             a1_ok=False, a2_ok=False,
             bbox_f1=_ZERO, bbox_f2=_ZERO,
             row_title="CorrelatedDSprites", row_subtitle="A1+A2 violated"),
    ]

    # Optionally add CelebA
    if args.celeba_root is not None:
        try:
            from datasets.CelebAFactor import (
                CelebAFactorDataset,
                BBOX_F1 as E_BBOX_F1, BBOX_F2 as E_BBOX_F2,
            )
            print("  Loading CelebA (may download ~1.7 GB on first run)...")
            ds_E = CelebAFactorDataset(celeba_root=args.celeba_root,
                                       split="valid", balanced=True,
                                       global_seed=args.seed)
            ds_F = CelebAFactorDataset(celeba_root=args.celeba_root,
                                       split="valid", balanced=False,
                                       global_seed=args.seed)
            print(f"  E: {len(ds_E)} | F: {len(ds_F)}")
            datasets_info.append(dict(
                ds_val=ds_E, domain="E", has_bbox=True, is_rgb=True,
                a1_ok=True, a2_ok=True,
                bbox_f1=E_BBOX_F1, bbox_f2=E_BBOX_F2,
                row_title="CelebA (balanced)", row_subtitle="Real, approx. independent"))
            datasets_info.append(dict(
                ds_val=ds_F, domain="F", has_bbox=True, is_rgb=True,
                a1_ok=False, a2_ok=True,
                bbox_f1=E_BBOX_F1, bbox_f2=E_BBOX_F2,
                row_title="CelebA (natural)", row_subtitle="Real, natural correlation"))

            # Bbox verification figure
            print("  Generating CelebA bbox verification figure...")
            plot_celeba_bbox_verify(
                ds_E, ds_F, E_BBOX_F1, E_BBOX_F2,
                out / "celeba_bbox_verify.png", n_per_class=6)

        except Exception as e:
            print(f"  WARNING: CelebA failed ({e}), skipping E/F.")

    # ── Per-domain panels ─────────────────────────────────────────────────
    print("\n=== Generating per-domain panels ===")
    panel_extra = {
        "A": dict(slot_rows_f1=SLOT_F1_ROWS, slot_cols_f1=SLOT_F1_COLS,
                  slot_rows_f2=SLOT_F2_ROWS, slot_cols_f2=SLOT_F2_COLS),
        "B": dict(slot_rows_f1=SLOT_F1_ROWS, slot_cols_f1=SLOT_F1_COLS,
                  slot_rows_f2=SLOT_F2_ROWS, slot_cols_f2=SLOT_F2_COLS),
    }
    domain_titles = {
        "A": ("SplatteredMNIST — Positive Control",
              "Digit-1 (top-left) + Digit-7 (bottom-right) · Soft edges · Splatters · Independent"),
        "B": ("CorrelatedSplatteredMNIST — A1 Violated",
              "Same as A but 30% of single-label images secretly contain both digits"),
        "C": ("dSprites — A2 Violated",
              "Scale (F1=large) + Orientation (F2=high) · Global factors · Independent · No bbox"),
        "D": ("CorrelatedDSprites — A1 + A2 Violated",
              "Scale × Orientation + 50% spurious co-occurrence · Double violation"),
        "E": ("CelebA Balanced — Real, Independent",
              "Eyeglasses (F1) + Bald (F2) · Balanced subsample · Approx. independent"),
        "F": ("CelebA Natural — Real, Correlated",
              "Eyeglasses (F1) + Bald (F2) · Natural split · Demographic co-occurrence preserved"),
    }
    cxr_analogs_map = {
        "A": ["No pathology",
              "Cardiomegaly\n(enlarged silhouette)",
              "Pleural effusion\n(costophrenic blunting)",
              "Comorbid CXR", "Slot bbox overlay"],
        "B": ["No pathology",
              "Cardiomegaly label\n(30% hide effusion)",
              "Effusion label\n(30% hide cardiomegaly)",
              "Comorbid (test only)", "Bbox valid; labels corrupt"],
        "C": ["No pathology",
              "Disease-1 analog\n(large scale)",
              "Disease-2 analog\n(high orientation)",
              "Comorbid analog", "Global geometry\n(no bbox)"],
        "D": ["No pathology",
              "D1 label\n(50% also high-θ)",
              "D2 label\n(50% also large)",
              "Comorbid (test only)", "Global; correlated"],
        "E": ["No glasses, hair",
              "Eyeglasses\n(face accessory)",
              "Bald\n(hair absence)",
              "Glasses + Bald", "Glasses / crown bbox"],
        "F": ["No glasses, hair",
              "Eyeglasses\n(nat. corr.)",
              "Bald\n(nat. corr.)",
              "Glasses + Bald", "Glasses / crown bbox"],
    }

    for info in datasets_info:
        d = info["domain"]
        print(f"  Domain {d}...")
        title, subtitle = domain_titles.get(d, (f"Domain {d}", ""))
        extra = panel_extra.get(d, dict(slot_rows_f1=None, slot_cols_f1=None,
                                        slot_rows_f2=None, slot_cols_f2=None))
        plot_domain(
            ds_val=info["ds_val"],
            domain_letter=d,
            title=title, subtitle=subtitle,
            cxr_analogs=cxr_analogs_map.get(d, [""] * 5),
            assumption_status=(info["a1_ok"], info["a2_ok"]),
            has_bbox=info["has_bbox"], is_rgb=info["is_rgb"],
            bbox_f1=info["bbox_f1"], bbox_f2=info["bbox_f2"],
            **extra,
            out_path=out / f"domain_{d}_panel.png",
            n_cols=args.n_cols,
        )

    print("\n=== Generating overview figure ===")
    plot_overview(datasets_info, out / "overview_all_domains.png")

    print("\n=== Generating composition illustration ===")
    plot_composition_logic(datasets_info, out / "composition_task.png")

    print("\n=== Generating CXR analogy table ===")
    plot_cxr_analogy(datasets_info, out / "cxr_analogy_table.png")

    print("\n=== Generating dSprites factor analysis ===")
    plot_dsprites_factor_analysis(ds_C, ds_D, out / "dsprites_factor_analysis.png")

    print(f"\nAll figures saved to {out}/")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
