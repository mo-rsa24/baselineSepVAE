from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "spvae_architecture.pdf"


def add_box(ax, xy, width, height, text, fc="#F8FAFC", ec="#334155", fontsize=10, dashed=False):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return box


def add_arrow(ax, start, end, color="#475569", dashed=False, lw=1.4, connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    return arrow


def add_sprite(ax, origin, size, label):
    x0, y0 = origin
    pad = 0.03 * size
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            size,
            size,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor="#FFFFFF",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0 + pad, y0 + pad),
            size - 2 * pad,
            size - 2 * pad,
            boxstyle="round,pad=0.00,rounding_size=0.01",
            linewidth=0.0,
            facecolor="#0F172A",
        )
    )

    sprite = np.zeros((64, 64))
    yy, xx = np.mgrid[:64, :64]
    sprite[((xx - 32) ** 2) / (18**2) + ((yy - 28) ** 2) / (11**2) <= 1.0] = 1.0
    sprite[((xx - 25) ** 2) / (8**2) + ((yy - 39) ** 2) / (7**2) <= 1.0] = 1.0
    sprite[((xx - 39) ** 2) / (8**2) + ((yy - 39) ** 2) / (7**2) <= 1.0] = 1.0

    inset = ax.inset_axes([x0 + pad * 1.6, y0 + pad * 1.6, size - 3.2 * pad, size - 3.2 * pad], transform=ax.transData)
    inset.imshow(sprite, cmap="gray", vmin=0, vmax=1)
    inset.axis("off")

    ax.text(x0 + size / 2, y0 - 0.05, label, ha="center", va="top", fontsize=10)


def add_latent(ax, center, label, subtitle=None):
    circ = Circle(center, 0.04, facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.4)
    ax.add_patch(circ)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=10)
    if subtitle:
        ax.text(center[0], center[1] - 0.07, subtitle, ha="center", va="top", fontsize=8.8, color="#334155")
    return circ


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )

    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "SP-VAE architecture for Regime D",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    ax.text(
        0.5,
        0.93,
        "Same correlated dSprites support and matched supervision as Regime C2; different latent organisation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
    )

    add_sprite(ax, (0.03, 0.54), 0.12, r"input $x$")
    add_box(ax, (0.20, 0.50), 0.16, 0.20, "encoder\n$E_\\phi(x)$", fc="#E0F2FE", ec="#0369A1", fontsize=11)
    add_arrow(ax, (0.15, 0.60), (0.20, 0.60))

    param_y = 0.79
    param_xs = [0.43, 0.56, 0.69]
    param_labels = [
        r"$q_\phi(z_c\mid x)$",
        r"$q_\phi(z_{d1}\mid x)$",
        r"$q_\phi(z_{d2}\mid x)$",
    ]
    for x, label in zip(param_xs, param_labels):
        add_box(ax, (x - 0.055, param_y - 0.04), 0.11, 0.08, label, fc="#F8FAFC", ec="#64748B", fontsize=9.5)
        add_arrow(ax, (0.36, 0.64), (x, param_y - 0.04), connectionstyle="arc3,rad=0.0")

    ax.text(0.56, 0.69, "sample via reparameterization", ha="center", va="center", fontsize=9, color="#475569")

    add_latent(ax, (0.43, 0.55), r"$z_c$", "preserved context")
    add_latent(ax, (0.56, 0.55), r"$z_{d1}$", "orientation head")
    add_latent(ax, (0.69, 0.55), r"$z_{d2}$", "scale head")

    for x in param_xs:
        add_arrow(ax, (x, param_y - 0.04), (x, 0.59))

    ax.text(0.56, 0.46, r"$z=(z_c, z_{d1}, z_{d2})$", ha="center", va="center", fontsize=10.5)
    add_box(ax, (0.77, 0.50), 0.14, 0.20, "decoder\n$D_\\theta(z)$", fc="#DCFCE7", ec="#15803D", fontsize=11)
    add_arrow(ax, (0.73, 0.55), (0.77, 0.60))

    add_sprite(ax, (0.93, 0.54), 0.06, r"recon. $\hat{x}$")
    add_arrow(ax, (0.91, 0.60), (0.93, 0.60))

    ax.text(
        0.60,
        0.39,
        r"training-time heads read encoder means $\mu_{d1}, \mu_{d2}$",
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
    )
    add_box(
        ax,
        (0.46, 0.22),
        0.13,
        0.08,
        "training-only\n$g_{\\mathrm{orient}}$",
        fc="#FEF3C7",
        ec="#B45309",
        fontsize=9.4,
        dashed=True,
    )
    add_box(
        ax,
        (0.66, 0.22),
        0.13,
        0.08,
        "training-only\n$g_{\\mathrm{scale}}$",
        fc="#FEF3C7",
        ec="#B45309",
        fontsize=9.4,
        dashed=True,
    )
    add_arrow(ax, (0.56, 0.51), (0.525, 0.30), dashed=True, color="#B45309")
    add_arrow(ax, (0.69, 0.51), (0.725, 0.30), dashed=True, color="#B45309")
    add_arrow(ax, (0.69, 0.51), (0.535, 0.30), dashed=True, color="#B91C1C", connectionstyle="arc3,rad=-0.20")
    add_arrow(ax, (0.56, 0.51), (0.725, 0.30), dashed=True, color="#B91C1C", connectionstyle="arc3,rad=0.20")
    ax.text(0.61, 0.34, "GRL", ha="center", va="center", fontsize=8.8, color="#B91C1C", fontweight="bold")
    ax.text(0.50, 0.315, "CE", ha="center", va="bottom", fontsize=8.6, color="#B45309")
    ax.text(0.74, 0.315, "CE", ha="center", va="bottom", fontsize=8.6, color="#B45309")
    ax.text(0.61, 0.30, "wrong-head suppression", ha="center", va="top", fontsize=8.4, color="#B91C1C")
    ax.text(0.61, 0.265, "orange = correct head, red = GRL path", ha="center", va="top", fontsize=8.2, color="#475569")

    add_box(
        ax,
        (0.18, 0.08),
        0.64,
        0.10,
        (
            r"$\mathcal{L}_{\mathrm{ELBO}}"
            r" + \mathcal{L}_{\mathrm{align}}\;(\mathrm{CE}+\mathrm{GRL})"
            r" + \lambda_{\mathrm{mi}}\hat{I}(z_{d1};z_{d2})"
            r" + \lambda_{\mathrm{orth}}\|\mathrm{Cov}(z_{d1}, z_{d2})\|_F^2$"
        ),
        fc="#F1F5F9",
        ec="#334155",
        fontsize=10.5,
    )
    ax.text(
        0.50,
        0.03,
        "Backpropagate summed objective through encoder, decoder, and auxiliary heads.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    add_arrow(ax, (0.96, 0.54), (0.74, 0.18), connectionstyle="arc3,rad=-0.15")
    add_arrow(ax, (0.525, 0.22), (0.525, 0.18), dashed=True, color="#B45309")
    add_arrow(ax, (0.725, 0.22), (0.725, 0.18), dashed=True, color="#B45309")
    add_arrow(ax, (0.43, 0.50), (0.43, 0.18), dashed=True, color="#2563EB", connectionstyle="arc3,rad=0.05")
    add_arrow(ax, (0.69, 0.50), (0.69, 0.18), dashed=True, color="#2563EB", connectionstyle="arc3,rad=-0.05")

    fig.savefig(OUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
