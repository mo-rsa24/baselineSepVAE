from pathlib import Path
import urllib.request

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "c2_factor_targets.pdf"
CACHE_DIR = Path.home() / ".cache" / "dsprites"
NPZ_PATH = CACHE_DIR / "dsprites.npz"
URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)

COL_SHAPE = 1
COL_SCALE = 2
COL_ORIENT = 3
COL_POSX = 4
COL_POSY = 5


def ensure_dataset() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not NPZ_PATH.exists():
        urllib.request.urlretrieve(URL, NPZ_PATH)
    return NPZ_PATH


def find_index(latents_cls, shape_idx, scale_idx, orient_idx, posx_idx=16, posy_idx=16):
    mask = (
        (latents_cls[:, COL_SHAPE] == shape_idx)
        & (latents_cls[:, COL_SCALE] == scale_idx)
        & (latents_cls[:, COL_ORIENT] == orient_idx)
        & (latents_cls[:, COL_POSX] == posx_idx)
        & (latents_cls[:, COL_POSY] == posy_idx)
    )
    idxs = np.where(mask)[0]
    if len(idxs) != 1:
        raise ValueError(f"Expected 1 match, got {len(idxs)}")
    return idxs[0]


def main():
    data = np.load(ensure_dataset(), allow_pickle=True, encoding="latin1")
    imgs = data["imgs"]
    latents_cls = data["latents_classes"]
    latents_vals = data["latents_values"]

    scale_vals = np.unique(latents_vals[:, COL_SCALE])
    orient_vals = np.unique(latents_vals[:, COL_ORIENT])
    step_rad = orient_vals[1] - orient_vals[0]
    step_deg = np.degrees(step_rad)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 14,
        }
    )

    fig = plt.figure(figsize=(12.0, 4.4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.2])

    fig.suptitle("C2 supervised factor targets", fontweight="bold")

    # Left panel: scale classes rendered as actual dSprites samples.
    gs_left = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0], wspace=0.15)
    left_axes = [fig.add_subplot(gs_left[i]) for i in range(6)]
    for sc, ax in enumerate(left_axes):
        idx = find_index(latents_cls, shape_idx=0, scale_idx=sc, orient_idx=0)
        ax.imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"class {sc}\n({scale_vals[sc]:.3f})", fontsize=9, pad=6)
        ax.axis("off")
    left_axes[0].text(
        0.0,
        1.20,
        "Scale head $g_{\\mathrm{scale}}$: 6 classes",
        transform=left_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    left_axes[0].text(
        0.0,
        1.07,
        "Fixed context: square shape, orientation = 0, centred position",
        transform=left_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    left_axes[0].text(
        0.0,
        -0.24,
        "Class indices 0--5 map to physical scale values 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.",
        transform=left_axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    # Right panel: compact orientation discretisation.
    ax_polar = fig.add_subplot(gs[1], projection="polar")
    ax_polar.set_title("Orientation head $g_{\\mathrm{orient}}$: 40 classes", pad=18, fontweight="bold")
    for ori in range(len(orient_vals)):
        angle = orient_vals[ori]
        ax_polar.annotate(
            "",
            xy=(angle, 1.0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.9),
        )
    ax_polar.set_ylim(0, 1.0)
    ax_polar.set_yticklabels([])
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax_polar.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    ax_polar.grid(alpha=0.35)
    ax_polar.text(
        0.5,
        -0.17,
        (
            "Class indices 0--39 correspond to\n"
            f"0, {orient_vals[1]:.4f}, {orient_vals[2]:.4f}, ... radians\n"
            f"in uniform {step_rad:.4f} rad (~{step_deg:.1f}$^\\circ$) steps."
        ),
        transform=ax_polar.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.savefig(OUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
