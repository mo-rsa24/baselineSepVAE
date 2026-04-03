"""
Training scenario overlay diagnostic.

Compares the current run against known good/bad reference runs to provide
visual context about whether training is healthy, collapsing, or spiking.

Exported function (called from train_sep_vae.py every sample_every epochs):
    make_scenario_overlay(current_metrics_path, epoch, workdir=None) -> PIL.Image

Logged to W&B as "diagnostics/loss_scenarios".
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Known reference runs (dir names under <workdir>/runs_sepvae/) ─────────────
# Each entry maps a scenario label to its run-directory name.
SCENARIO_RUNS = {
    "healthy_d1":   "d1_recon_bbox_xattn-20260321-004241",
    "kl_spike_d2":  "d2_perceptual_bbox-20260322-050230",
    "gan_collapse": "d5_gan-20260323-042442",
}

SCENARIO_STYLE = {
    "healthy_d1":   {"color": "#2ca02c", "label": "healthy D1",      "alpha": 0.30, "lw": 1.2, "ls": "-"},
    "kl_spike_d2":  {"color": "#ff7f0e", "label": "KL-spike D2",     "alpha": 0.30, "lw": 1.2, "ls": "--"},
    "gan_collapse": {"color": "#d62728", "label": "GAN collapse D5",  "alpha": 0.30, "lw": 1.2, "ls": "-."},
}

# (metric_key, subplot_title, healthy_lo, healthy_hi, bad_threshold, bad_is_above)
# healthy_lo/hi = None means no shaded corridor for that metric
SUBPLOT_SPEC = [
    ("loss/kl_total",
     "KL total (nats)",
     50, 400, 600, True),
    ("loss/reconstruction",
     "Reconstruction MSE",
     0.005, 0.04, 0.06, True),
    ("loss/ctr_reg",
     "CTR regression loss",
     0.0, 0.05, 0.10, True),
    ("loss/mi_factor",
     "MI factor loss",
     None, None, None, None),
    ("metrics/disc_acc",
     "FactorVAE disc accuracy",
     0.55, 0.75, 0.85, True),
    ("loss/masked_rec",
     "Masked rec loss",
     0.0, 0.05, 0.15, True),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_metrics(path: Path) -> list[dict]:
    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def _extract_series(records: list[dict], key: str):
    """Return (epochs_array, values_array) for a metric key."""
    epochs, vals = [], []
    for r in records:
        if key in r and r[key] is not None:
            try:
                epochs.append(float(r.get("epoch", len(epochs))))
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                pass
    return np.array(epochs, dtype=float), np.array(vals, dtype=float)


# ── Public API ────────────────────────────────────────────────────────────────

def make_scenario_overlay(
    current_metrics_path,
    epoch: int,
    workdir=None,
) -> Image.Image:
    """
    Build a 3×2 scenario overlay plot and return as a PIL Image.

    Args:
        current_metrics_path: Path to the current run's metrics_history.jsonl
        epoch: Current training epoch (for title)
        workdir: Repo root directory. Scenario run dirs are resolved relative to
                 <workdir>/runs_sepvae/. Defaults to grandparent of metrics file.
    """
    current_metrics_path = Path(current_metrics_path)
    if workdir is None:
        # Typical layout: runs_sepvae/<run_name>/metrics_history.jsonl
        workdir = current_metrics_path.parent.parent.parent
    runs_root = Path(workdir) / "runs_sepvae"

    current_records = _load_metrics(current_metrics_path)
    scenario_records: dict[str, list[dict]] = {}
    for key, run_dir_name in SCENARIO_RUNS.items():
        p = runs_root / run_dir_name / "metrics_history.jsonl"
        recs = _load_metrics(p)
        if recs:
            scenario_records[key] = recs

    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 10),
        gridspec_kw={"hspace": 0.50, "wspace": 0.32},
    )
    axes_flat = axes.flatten()

    for ax_idx, (metric_key, title, h_lo, h_hi, bad_thresh, bad_above) in enumerate(SUBPLOT_SPEC):
        ax = axes_flat[ax_idx]

        # Scenario faint lines
        for sc_key, recs in scenario_records.items():
            st = SCENARIO_STYLE[sc_key]
            ep, vals = _extract_series(recs, metric_key)
            if len(vals) > 0:
                ax.plot(ep, vals,
                        color=st["color"], alpha=st["alpha"],
                        lw=st["lw"], ls=st["ls"])

        # Healthy corridor (shaded green band)
        if h_lo is not None and h_hi is not None:
            ax.axhspan(h_lo, h_hi, alpha=0.10, color="#2ca02c", zorder=0)

        # Bad threshold dotted line
        if bad_thresh is not None:
            line_color = "#d62728" if bad_above else "#2ca02c"
            ax.axhline(bad_thresh, ls=":", lw=1.3, color=line_color, alpha=0.8)
            ax.text(0.01, bad_thresh, f"  thresh={bad_thresh}", va="bottom",
                    ha="left", transform=ax.get_yaxis_transform(),
                    fontsize=7, color=line_color, alpha=0.8)

        # Current run (bold blue)
        ep_cur, vals_cur = _extract_series(current_records, metric_key)
        if len(vals_cur) > 0:
            ax.plot(ep_cur, vals_cur, color="#1f77b4", lw=2.2, zorder=5)
        elif current_records:
            ax.text(0.5, 0.5, "not yet logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray",
                    style="italic")

        # ── Y-axis limits: anchor to current run, ensure bands are visible ──
        if len(vals_cur) > 0:
            cur_min = float(vals_cur.min())
            cur_max = float(vals_cur.max())
            cur_range = max(cur_max - cur_min, abs(cur_max) * 0.05, 1e-8)

            y_lo = cur_min - 0.10 * cur_range
            y_hi = cur_max + 0.30 * cur_range

            # Ensure the bad threshold line is always visible
            if bad_thresh is not None:
                if bad_above:
                    y_hi = max(y_hi, bad_thresh * 1.15)
                else:
                    y_lo = min(y_lo, bad_thresh * 0.85)

            # Ensure the healthy corridor is always visible
            if h_hi is not None:
                y_hi = max(y_hi, h_hi * 1.20)
            if h_lo is not None:
                y_lo = min(y_lo, h_lo * 0.80)

            # Don't go below zero for non-negative metrics
            if metric_key not in ("loss/mi_factor",):
                y_lo = max(y_lo, 0.0)

            ax.set_ylim(y_lo, y_hi)

        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend on last subplot
    handles, labels = [], []
    for sc_key, st in SCENARIO_STYLE.items():
        if sc_key in scenario_records:
            handles.append(plt.Line2D([0], [0], color=st["color"],
                                       lw=1.5, ls=st["ls"], alpha=0.8))
            labels.append(st["label"])
    handles.append(plt.Line2D([0], [0], color="#1f77b4", lw=2.2))
    labels.append("current run")
    axes_flat[-1].legend(handles, labels, fontsize=7, loc="best",
                          framealpha=0.8)

    fig.suptitle(
        f"Training Scenario Overlay — Epoch {epoch}\n"
        "Faint lines = reference runs  |  green band = healthy corridor  |  dotted = threshold",
        fontsize=9, y=1.01,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── CLI (optional standalone use) ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("plot_training_scenarios")
    p.add_argument("--metrics", required=True,
                   help="Path to current run's metrics_history.jsonl")
    p.add_argument("--epoch",   type=int, default=0)
    p.add_argument("--out",     default="scenario_overlay.png")
    p.add_argument("--workdir", default=None)
    args = p.parse_args()
    img = make_scenario_overlay(args.metrics, args.epoch, workdir=args.workdir)
    img.save(args.out)
    print(f"Saved → {args.out}")
