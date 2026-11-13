"""
visualize_syndis_pedagogy.py
============================
Three pedagogical figures illustrating SynDis and its relationship to the
VinBigData / SepVAE disentanglement task.

  fig1_dataset_anatomy.png   — Sample grid with clinical labels + bbox +
                               schematic CXR anatomy sketch
  fig2_latent_scenarios.png  — Simulated 2-D latent distributions under 4
                               disentanglement regimes (bad → ideal)
  fig3_composition_test.png  — Region-swap composition under each regime,
                               with per-region signal scores

No GPU or trained model required — runs in seconds.

Usage
-----
    # from repo root
    python run/visualize_syndis_pedagogy.py
    python run/visualize_syndis_pedagogy.py --out_dir results/pedagogy
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from datasets.SynDis import (
    SynDisDataset,
    _make_background, _add_disease1, _add_disease2,
    BLOB_ROWS, BLOB_COLS, STRIPE_ROWS, STRIPE_COLS,
    LABEL_NORMAL, LABEL_DISEASE1, LABEL_DISEASE2, LABEL_BOTH,
)

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
C_NORMAL   = "#7f8c8d"   # gray     — healthy
C_CARDIO   = "#c0392b"   # red      — cardiomegaly / disease-1 / blob
C_EFFUSION = "#1a6fa8"   # blue     — effusion / disease-2 / stripe
C_BOTH     = "#7d3c98"   # purple   — comorbid
C_BG       = "#fafafa"

SCENARIO_META = [
    # (key,        display_title,              accent_color,   alpha_blob, alpha_stripe, leak)
    ("bad",        "Bad\n(no regularisation)",  "#e74c3c",      1.00,       0.00,         0.90),
    ("uncertain",  "Uncertain\n(weak null+ortho)", "#e67e22",   0.90,       0.35,         0.40),
    ("good",       "Good\n(moderate null+ortho+clf)", "#27ae60",0.95,       0.80,         0.10),
    ("ideal",      "Ideal\n(strong null+ortho+clf)", "#2980b9", 1.00,       1.00,         0.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gen(seed: int, label: int) -> np.ndarray:
    """Generate one SynDis image in [0, 1]."""
    rng = np.random.RandomState(seed)
    img = _make_background(rng)
    if label in (LABEL_DISEASE1, LABEL_BOTH):
        img = _add_disease1(img, rng)
    if label in (LABEL_DISEASE2, LABEL_BOTH):
        img = _add_disease2(img, rng)
    return img.astype(np.float32)


def _region_mask(rows, cols, H=64, W=64) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.float32)
    m[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
    return m


BLOB_MASK   = _region_mask(BLOB_ROWS,   BLOB_COLS)
STRIPE_MASK = _region_mask(STRIPE_ROWS, STRIPE_COLS)


def compose_image(seed_n: int, seed_d1: int, seed_d2: int,
                  alpha_blob: float  = 1.0,
                  alpha_stripe: float = 1.0,
                  leak_blob: float   = 0.0) -> np.ndarray:
    """
    Region-swap composition simulating decode(z_c_normal, z_d1_cardio, z_d2_effusion).

    alpha_blob    fraction of blob region taken from the d1 source
    alpha_stripe  fraction of stripe region taken from the d2 source
    leak_blob     extra blob carried by the z_d2 head (head entanglement)

    Returns image in [0, 1].
    """
    bg = _gen(seed_n,  LABEL_NORMAL)
    d1 = _gen(seed_d1, LABEL_DISEASE1)
    d2 = _gen(seed_d2, LABEL_DISEASE2)

    composed = bg.copy()
    # Blob region: blend bg ↔ d1 pixels
    composed = composed * (1 - BLOB_MASK * alpha_blob) + d1 * BLOB_MASK * alpha_blob
    # Stripe region: blend bg ↔ d2 pixels
    composed = composed * (1 - STRIPE_MASK * alpha_stripe) + d2 * STRIPE_MASK * alpha_stripe
    # Leakage: z_d2 accidentally encoded blob → extra blob signal appears
    if leak_blob > 0:
        blob_signal = np.clip(d1 - bg, 0, None)   # extra brightness from d1
        composed = composed + blob_signal * BLOB_MASK * leak_blob
    return np.clip(composed, 0, 1)


def region_score(img: np.ndarray, reference: np.ndarray,
                 mask: np.ndarray) -> float:
    """Mean pixel lift over `reference` within `mask` (positive = signal present)."""
    return float(((img - reference) * mask).sum() / mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Simulated latent-space data for the 4 scenarios
# ─────────────────────────────────────────────────────────────────────────────

def simulate_latents(scenario: str, rng: np.random.RandomState,
                     n_normal=300, n_d1=150, n_d2=150, n_both=150):
    """
    Return simulated 2-D GAP(z_d1) vectors mimicking what you would see
    after PCA of the disease-1 head under each disentanglement regime.

    Axis interpretation
    -------------------
    dim-0  "how much disease-1 (blob) signal is encoded in z_d1"
    dim-1  "how much disease-2 (stripe) signal is encoded in z_d1"
                                                      ^^^^ should be 0 in ideal

    Scenario | d1 shifts dim-0 | d2 shifts dim-1 (LEAKAGE)
    ---------|-----------------|----------------------------
    bad      |  yes (large)   |  yes (large)  — fully entangled
    uncertain|  yes (large)   |  yes (moderate)
    good     |  yes (large)   |  yes (small)
    ideal    |  yes (large)   |  NO           — perfectly disentangled
    """
    # (mu_normal, mu_d1, mu_d2, mu_both, sigma)
    params = {
        "bad":       ([0, 0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0], 0.55),
        "uncertain": ([0, 0], [2.8, 0.1], [0.1, 1.5], [2.9, 1.6], 0.62),
        "good":      ([0, 0], [2.5, 0.0], [0.0, 0.5], [2.5, 0.5], 0.50),
        "ideal":     ([0, 0], [2.5, 0.0], [0.0, 0.0], [2.5, 0.0], 0.50),
    }
    mu_n, mu_d1, mu_d2, mu_b, sigma = params[scenario]

    def _c(mu, n): return rng.randn(n, 2) * sigma + np.array(mu, dtype=float)

    z = np.vstack([_c(mu_n, n_normal), _c(mu_d1, n_d1),
                   _c(mu_d2, n_d2),    _c(mu_b,  n_both)])
    labels = np.concatenate([
        np.full(n_normal, LABEL_NORMAL),
        np.full(n_d1,     LABEL_DISEASE1),
        np.full(n_d2,     LABEL_DISEASE2),
        np.full(n_both,   LABEL_BOTH),
    ])
    return z, labels


def _auc(z, binary_labels) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return float("nan")
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(z, binary_labels)
    prob = clf.predict_proba(z)[:, 1]
    return roc_auc_score(binary_labels, prob)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Dataset Anatomy
# ─────────────────────────────────────────────────────────────────────────────

def _draw_cxr_sketch(ax):
    """Schematic chest X-ray annotated with the clinical analogies."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#eef2f7")

    # Thorax outline
    ax.add_patch(mpatches.Ellipse((5, 6.2), 9.2, 11.2,
                                  fc="#f7fbff", ec="#bbb", lw=1.5, zorder=0))
    # Lung fields
    for cx, cy in [(3.2, 6.8), (6.8, 6.8)]:
        ax.add_patch(mpatches.Ellipse((cx, cy), 3.0, 6.2,
                                      fc="#ddeeff", ec="#99b", lw=0.8,
                                      alpha=0.7, zorder=1))
    # Diaphragm arc
    ax.add_patch(mpatches.Arc((5, 3.2), 7.5, 3.5,
                              angle=0, theta1=0, theta2=180,
                              color="#999", lw=2, zorder=3))
    # Spine
    ax.add_patch(mpatches.Rectangle((4.7, 1.0), 0.6, 10,
                                    fc="#e0d0b0", ec="#bbb", lw=0.5,
                                    alpha=0.6, zorder=2))

    # ── Cardiac silhouette ──
    heart = mpatches.Ellipse((4.3, 5.6), 3.0, 3.6,
                             fc="#f5c6c6", ec=C_CARDIO, lw=2.0,
                             alpha=0.85, zorder=4)
    ax.add_patch(heart)
    ax.text(4.3, 5.5, "♥", ha="center", va="center",
            fontsize=15, color=C_CARDIO, fontweight="bold", zorder=6)

    # Cardio enlargement indicator (dashed outer ellipse)
    heart_big = mpatches.Ellipse((4.3, 5.6), 4.2, 5.0,
                                 fc="none", ec=C_CARDIO, lw=1.5,
                                 ls="--", alpha=0.6, zorder=4)
    ax.add_patch(heart_big)

    # ── Effusion band ──
    effusion = mpatches.Rectangle((0.5, 0.5), 9.0, 1.5,
                                  fc="#aed6f1", ec=C_EFFUSION, lw=1.5,
                                  alpha=0.75, zorder=5)
    ax.add_patch(effusion)

    # ── Annotations ──
    ax.annotate("Cardiac silhouette\n→ Disease-1 (blob)\n  Cardiomegaly",
                xy=(3.8, 3.8), xytext=(0.3, 1.8),
                fontsize=7.5, color=C_CARDIO, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_CARDIO, lw=1.2),
                bbox=dict(fc="white", ec=C_CARDIO, lw=0.8, pad=2, alpha=0.9),
                zorder=10)
    ax.annotate("Pleural effusion\n→ Disease-2 (stripe)\n  Costophrenic blunting",
                xy=(6.0, 1.2), xytext=(6.8, 3.5),
                fontsize=7.5, color=C_EFFUSION, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_EFFUSION, lw=1.2),
                bbox=dict(fc="white", ec=C_EFFUSION, lw=0.8, pad=2, alpha=0.9),
                zorder=10)
    ax.annotate("Lung field\n→ Common latent z_c\n  Shared anatomy",
                xy=(7.5, 8.0), xytext=(7.2, 10.5),
                fontsize=7.5, color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2),
                bbox=dict(fc="white", ec="#aaa", lw=0.8, pad=2, alpha=0.9),
                zorder=10)
    ax.set_title("VinBigData\nClinical CXR Anatomy", fontsize=9.5,
                 fontweight="bold", pad=6, color="#2c3e50")


def figure1_dataset_anatomy(out_path: Path, n_per_class: int = 5,
                             rng_seed: int = 7):
    ds = SynDisDataset(n_normal=40, n_disease1=30, n_disease2=30, n_both=30,
                       global_seed=rng_seed)

    class_rows = [
        # (label, row_title, border_color, bbox_description)
        (LABEL_NORMAL,   "Normal\n(healthy)",          C_NORMAL,
         "No disease signal — shared anatomy only (z_c)"),
        (LABEL_DISEASE1, "Disease-1\n(cardio analog)",  C_CARDIO,
         "Gaussian blob at centre-left  ↔  enlarged cardiac silhouette (z_d1)"),
        (LABEL_DISEASE2, "Disease-2\n(effusion analog)", C_EFFUSION,
         "Gaussian stripe at bottom     ↔  pleural fluid at costophrenic angle (z_d2)"),
        (LABEL_BOTH,     "Both\n(comorbid — val only)", C_BOTH,
         "Blob + stripe  ↔  cardiomegaly + effusion (composition evaluation target)"),
    ]

    N_COLS_IMG = n_per_class       # image columns
    N_ROWS     = len(class_rows)

    fig = plt.figure(figsize=(17, 9.5), facecolor=C_BG)
    # Layout: [images: 5 cols] [spacer] [description: 1 col] [spacer] [CXR sketch: 1.8 col]
    gs = gridspec.GridSpec(
        N_ROWS, N_COLS_IMG + 3,
        figure=fig,
        width_ratios=[1] * N_COLS_IMG + [0.1, 2.2, 1.8],
        hspace=0.08, wspace=0.05,
        left=0.05, right=0.98, top=0.91, bottom=0.04,
    )

    for row, (label, row_title, color, desc) in enumerate(class_rows):
        idxs = np.where(ds.labels == label)[0][:n_per_class]

        for ci, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[row, ci])
            img_t, _ = ds[idx]
            img_np = (img_t.squeeze().numpy() + 1) / 2
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")

            # Bounding box overlays on first image of each disease row
            if ci == 0:
                if label in (LABEL_DISEASE1, LABEL_BOTH):
                    br, bc = BLOB_ROWS, BLOB_COLS
                    ax.add_patch(mpatches.Rectangle(
                        (bc[0] - 0.5, br[0] - 0.5),
                        bc[1] - bc[0], br[1] - br[0],
                        lw=2.0, ec=C_CARDIO, fc="none", ls="--"))
                    ax.text(bc[0], br[0] - 3.5, "cardio\nbbox",
                            fontsize=5.5, color=C_CARDIO, ha="left", va="top",
                            bbox=dict(fc="white", ec=C_CARDIO, lw=0.5,
                                      pad=0.5, alpha=0.8))
                if label in (LABEL_DISEASE2, LABEL_BOTH):
                    sr, sc = STRIPE_ROWS, STRIPE_COLS
                    ax.add_patch(mpatches.Rectangle(
                        (sc[0] - 0.5, sr[0] - 0.5),
                        sc[1] - sc[0], sr[1] - sr[0],
                        lw=2.0, ec=C_EFFUSION, fc="none", ls=":"))
                    ax.text(sc[0] + 1, sr[0] - 3.5, "effusion\nbbox",
                            fontsize=5.5, color=C_EFFUSION, ha="left", va="top",
                            bbox=dict(fc="white", ec=C_EFFUSION, lw=0.5,
                                      pad=0.5, alpha=0.8))

            # Coloured border per class
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(2.2)

            # Column header on row 0
            if row == 0:
                ax.set_title(f"{'(+bbox)' if ci == 0 else f'sample {ci + 1}'}",
                             fontsize=7, color="#555", pad=2)

        # Row label in description column
        ax_desc = fig.add_subplot(gs[row, N_COLS_IMG + 1])
        ax_desc.axis("off")
        ax_desc.set_facecolor(C_BG)
        ax_desc.text(0.0, 0.78, row_title, fontsize=9.5, fontweight="bold",
                     color=color, va="top", transform=ax_desc.transAxes)
        ax_desc.text(0.0, 0.45, desc, fontsize=7.2,
                     color="#444", va="top", transform=ax_desc.transAxes,
                     wrap=True, style="italic")
        # Disease label badge
        badge_txt = {
            LABEL_NORMAL:   "y = 0  (normal)",
            LABEL_DISEASE1: "y = 1  (disease-1 / blob)",
            LABEL_DISEASE2: "y = 2  (disease-2 / stripe)",
            LABEL_BOTH:     "y = 3  (both — withheld from training)",
        }[label]
        ax_desc.text(0.0, 0.12, badge_txt, fontsize=7,
                     color="white", va="top", transform=ax_desc.transAxes,
                     bbox=dict(fc=color, ec="none", pad=3, alpha=0.9))

    # CXR sketch spanning all rows
    ax_cxr = fig.add_subplot(gs[:, N_COLS_IMG + 2])
    _draw_cxr_sketch(ax_cxr)

    fig.suptitle(
        "Figure 1 — SynDis Dataset Anatomy  ↔  VinBigData Clinical Analogy\n"
        "Dashed boxes = disease bounding regions (analogous to radiologist annotations in VinBigData)  ·  "
        "Solid colours = class border",
        fontsize=10.5, fontweight="bold", y=0.975, color="#1a1a2e",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Latent-Space Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_latent(ax, z, labels,
                    color_by: str = "d1",
                    title: str = "",
                    show_auc: bool = True):
    """
    Plot 2-D simulated z_d1 scatter.
    color_by = 'd1' → colour by disease-1 label (on-diagonal probe)
    color_by = 'd2' → colour by disease-2 label (off-diagonal probe)
    """
    CLASS_COLORS = {
        LABEL_NORMAL:   C_NORMAL,
        LABEL_DISEASE1: C_CARDIO,
        LABEL_DISEASE2: C_EFFUSION,
        LABEL_BOTH:     C_BOTH,
    }

    # binary label used for coloring and AUC
    if color_by == "d1":
        is_pos = ((labels == LABEL_DISEASE1) | (labels == LABEL_BOTH)).astype(int)
        pos_color, neg_color = C_CARDIO, C_NORMAL
        pos_label, neg_label = "has blob (d1)", "no blob"
    else:
        is_pos = ((labels == LABEL_DISEASE2) | (labels == LABEL_BOTH)).astype(int)
        pos_color, neg_color = C_EFFUSION, C_NORMAL
        pos_label, neg_label = "has stripe (d2)", "no stripe"

    ax.scatter(z[is_pos == 0, 0], z[is_pos == 0, 1],
               c=neg_color, s=8, alpha=0.45, linewidths=0, label=neg_label)
    ax.scatter(z[is_pos == 1, 0], z[is_pos == 1, 1],
               c=pos_color, s=8, alpha=0.55, linewidths=0, label=pos_label)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#ccc")

    if show_auc:
        auc = _auc(z, is_pos)
        auc_str = f"AUC = {auc:.3f}" if not np.isnan(auc) else "AUC = n/a"
        # colour the AUC text to signal goodness
        if color_by == "d1":
            auc_color = "#27ae60" if auc > 0.85 else "#e67e22" if auc > 0.65 else "#e74c3c"
            want = "↑ want high"
        else:
            auc_color = "#27ae60" if auc < 0.6 else "#e67e22" if auc < 0.75 else "#e74c3c"
            want = "↓ want ~0.5"
        ax.text(0.97, 0.04, f"{auc_str}\n{want}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=7, color=auc_color,
                bbox=dict(fc="white", ec=auc_color, lw=0.7, pad=2, alpha=0.92))

    if title:
        ax.set_title(title, fontsize=8, pad=3)


def figure2_latent_scenarios(out_path: Path, rng_seed: int = 0):
    rng = np.random.RandomState(rng_seed)

    SCENARIOS = [s[0] for s in SCENARIO_META]
    TITLES    = [s[1] for s in SCENARIO_META]
    ACCENTS   = [s[2] for s in SCENARIO_META]

    fig, axes = plt.subplots(
        2, len(SCENARIOS),
        figsize=(15, 7),
        facecolor=C_BG,
        gridspec_kw=dict(hspace=0.35, wspace=0.12,
                         left=0.06, right=0.98,
                         top=0.88, bottom=0.10),
    )

    row_labels = [
        "z_d1  coloured by  d1 label\n(on-diagonal probe — want separation)",
        "z_d1  coloured by  d2 label\n(off-diagonal probe — want NO separation)",
    ]
    row_keys   = ["d1", "d2"]
    row_y      = [0.955, 0.490]

    for ri, (rkey, rlabel, ry) in enumerate(zip(row_keys, row_labels, row_y)):
        fig.text(0.01, ry, rlabel, va="top", ha="left",
                 fontsize=8.5, color="#333", fontweight="bold",
                 rotation=0)

    for ci, (scen, title, accent) in enumerate(zip(SCENARIOS, TITLES, ACCENTS)):
        z, labels = simulate_latents(scen, rng)

        # Row 0: on-diagonal (z_d1 ↔ d1 label)
        ax0 = axes[0, ci]
        _scatter_latent(ax0, z, labels, color_by="d1", show_auc=True)
        ax0.set_title(title, fontsize=9, fontweight="bold", color=accent, pad=6)
        # Accent top border
        for sp in ["top", "bottom", "left", "right"]:
            ax0.spines[sp].set_linewidth(2.5 if sp == "top" else 0.6)
            ax0.spines[sp].set_color(accent if sp == "top" else "#ccc")

        # Row 1: off-diagonal (z_d1 ↔ d2 label) — the critical separability test
        ax1 = axes[1, ci]
        _scatter_latent(ax1, z, labels, color_by="d2", show_auc=True)
        for sp in ["top", "bottom", "left", "right"]:
            ax1.spines[sp].set_linewidth(2.5 if sp == "bottom" else 0.6)
            ax1.spines[sp].set_color(accent if sp == "bottom" else "#ccc")

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=C_NORMAL,   label="Normal (no disease)"),
        mpatches.Patch(color=C_CARDIO,   label="Disease-1 / blob present"),
        mpatches.Patch(color=C_EFFUSION, label="Disease-2 / stripe present"),
        mpatches.Patch(color=C_BOTH,     label="Both present"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
               bbox_to_anchor=(0.5, 0.01))

    # Key insight annotation
    fig.text(
        0.5, 0.95,
        "Figure 2 — Simulated z_d1 latent space under 4 disentanglement regimes\n"
        "Row 1 (on-diagonal): z_d1 should always classify disease-1 well  ·  "
        "Row 2 (off-diagonal): z_d1 must NOT classify disease-2 — this is what orthogonality buys",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
        color="#1a1a2e",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Composition Test
# ─────────────────────────────────────────────────────────────────────────────

def figure3_composition_test(out_path: Path,
                              seed_n: int  = 42_000_001,
                              seed_d1: int = 42_100_001,
                              seed_d2: int = 42_200_001,
                              seed_gt: int = 42_300_001):
    """
    Shows: decode(z_c_normal, z_d1_cardio, z_d2_effusion)
    under each of the 4 regimes, with a score panel per row.

    Columns: Normal src | Cardio src | Effusion src | → | Composed | Scores
    """
    bg_ref = _gen(seed_n, LABEL_NORMAL)

    rows = []
    for key, title, accent, ab, as_, leak in SCENARIO_META:
        composed = compose_image(seed_n, seed_d1, seed_d2,
                                 alpha_blob=ab, alpha_stripe=as_,
                                 leak_blob=leak)
        blob_score   = region_score(composed, bg_ref, BLOB_MASK)
        stripe_score = region_score(composed, bg_ref, STRIPE_MASK)
        rows.append(dict(key=key, title=title, accent=accent,
                         composed=composed,
                         blob_score=blob_score, stripe_score=stripe_score))

    # Fixed source images (same for all rows)
    src_n  = _gen(seed_n,  LABEL_NORMAL)
    src_d1 = _gen(seed_d1, LABEL_DISEASE1)
    src_d2 = _gen(seed_d2, LABEL_DISEASE2)
    src_gt = _gen(seed_gt, LABEL_BOTH)       # ground-truth comorbid sample

    N_ROWS   = len(rows)
    N_COLS   = 6   # normal | cardio | effusion | composed | score bar | gt
    fig = plt.figure(figsize=(15, 3.2 * N_ROWS + 0.8), facecolor=C_BG)
    gs_outer = gridspec.GridSpec(
        N_ROWS, N_COLS,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1.5, 1],
        hspace=0.18, wspace=0.06,
        left=0.05, right=0.98,
        top=0.91, bottom=0.06,
    )

    col_headers = [
        "Normal source\n(provides z_c)",
        "Cardio source\n(provides z_d1)",
        "Effusion source\n(provides z_d2)",
        "Composed output\ndecode(z_c, z_d1, z_d2)",
        "Region signal scores\n(blob ↑  stripe ↑  = success)",
        "GT comorbid\n(reference)",
    ]
    for ci, hdr in enumerate(col_headers):
        ax_h = fig.add_subplot(gs_outer[0, ci])
        ax_h.axis("off")
        ax_h.set_title(hdr, fontsize=8, fontweight="bold",
                       color="#333", pad=4)

    for ri, row in enumerate(rows):
        imgs_and_borders = [
            (src_n,  C_NORMAL),
            (src_d1, C_CARDIO),
            (src_d2, C_EFFUSION),
            (row["composed"], row["accent"]),
            (None,   None),               # score panel
            (src_gt, C_BOTH),
        ]

        for ci, (img, bcolor) in enumerate(imgs_and_borders):
            ax = fig.add_subplot(gs_outer[ri, ci])

            if img is None:
                # Score bar panel
                ax.set_facecolor("#f8f9fa")
                ax.axis("off")
                bscore = row["blob_score"]
                sscore = row["stripe_score"]
                bar_colors = [C_CARDIO, C_EFFUSION]
                vals = [max(bscore, 0), max(sscore, 0)]
                bar_labels = [f"blob Δ = {bscore:+.3f}", f"stripe Δ = {sscore:+.3f}"]
                ax.barh([1, 0], vals, color=bar_colors, height=0.45,
                        left=0, align="center")
                ax.set_xlim(0, 0.45)
                ax.set_ylim(-0.5, 1.5)
                # Negative score indicators (when composing failed)
                for vi, (v, label, col) in enumerate(zip(vals, bar_labels, bar_colors)):
                    ax.text(0.01, [1, 0][vi], label, ha="left", va="center",
                            fontsize=7.5, color="white" if v > 0.04 else "#555",
                            zorder=5)
                # Raw value for negative bars
                if bscore <= 0:
                    ax.barh([1], [abs(min(bscore, 0))], color="#ffcccc",
                            height=0.45, left=0, align="center")
                    ax.text(0.01, 1, f"blob Δ = {bscore:+.3f} ✗",
                            ha="left", va="center", fontsize=7.5, color=C_CARDIO)
                if sscore <= 0:
                    ax.barh([0], [abs(min(sscore, 0))], color="#cceeff",
                            height=0.45, left=0, align="center")
                    ax.text(0.01, 0, f"stripe Δ = {sscore:+.3f} ✗",
                            ha="left", va="center", fontsize=7.5, color=C_EFFUSION)

                # Verdict
                both_positive = bscore > 0.02 and sscore > 0.02
                verdict = "✓ BOTH signals" if both_positive else (
                    "~ partial" if (bscore > 0.02 or sscore > 0.02) else "✗ failed")
                verdict_color = "#27ae60" if both_positive else (
                    "#e67e22" if "partial" in verdict else "#e74c3c")
                ax.text(0.5, -0.35, verdict, ha="center", va="center",
                        transform=ax.transAxes, fontsize=8.5,
                        fontweight="bold", color=verdict_color)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.axis("off")

                # bbox overlays on source images in first row only
                if ri == 0 and ci == 1:   # cardio source
                    br, bc = BLOB_ROWS, BLOB_COLS
                    ax.add_patch(mpatches.Rectangle(
                        (bc[0] - 0.5, br[0] - 0.5), bc[1] - bc[0], br[1] - br[0],
                        lw=1.8, ec=C_CARDIO, fc="none", ls="--"))
                if ri == 0 and ci == 2:   # effusion source
                    sr, sc = STRIPE_ROWS, STRIPE_COLS
                    ax.add_patch(mpatches.Rectangle(
                        (sc[0] - 0.5, sr[0] - 0.5), sc[1] - sc[0], sr[1] - sr[0],
                        lw=1.8, ec=C_EFFUSION, fc="none", ls=":"))

                # Coloured border
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(bcolor if bcolor else "#ccc")
                    sp.set_linewidth(2.0)

        # Row label (scenario title) — placed left of the row using fig.text
        # We compute position from the gridspec row rather than adding a duplicate subplot
        row_pos = gs_outer[ri, 0].get_position(fig)
        row_mid_y = row_pos.y0 + row_pos.height / 2
        fig.text(
            0.002, row_mid_y,
            row["title"].replace("\n", " "),
            ha="left", va="center", fontsize=8.0,
            fontweight="bold", color=row["accent"],
            rotation=90,
        )

    fig.suptitle(
        "Figure 3 — Composition Test: decode(z_c_normal , z_d1_cardio , z_d2_effusion)\n"
        "Success = both blob Δ > 0 and stripe Δ > 0 in composed output  "
        "·  Score = mean pixel lift over normal baseline within each bbox region",
        fontsize=10, fontweight="bold", y=0.985, color="#1a1a2e",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("SynDis pedagogical figures")
    parser.add_argument("--out_dir", type=str, default="results/pedagogy",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures → {out_dir}/")
    figure1_dataset_anatomy(out_dir / "fig1_dataset_anatomy.png")
    figure2_latent_scenarios(out_dir / "fig2_latent_scenarios.png")
    figure3_composition_test(out_dir / "fig3_composition_test.png")
    print("Done.")


if __name__ == "__main__":
    main()
