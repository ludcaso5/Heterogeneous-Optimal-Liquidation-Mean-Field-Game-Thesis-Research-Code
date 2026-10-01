"""Plotting utilities matching the figures used in the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from .solver import compute_aggregated_flow

FIGURE_WIDTH_IN = 10.0
FIGURE_HEIGHT_IN = 6.0

POPULATION_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
]
AGGREGATE_COLOR = "#334155"


def setup_plot_theme() -> None:
    """Apply the visual conventions used for the thesis figures."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.9,
            "grid.alpha": 0.85,
            "grid.linestyle": "-",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": "#0f172a",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "lines.linewidth": 2.4,
            "lines.antialiased": True,
            "lines.solid_capstyle": "round",
            "axes.prop_cycle": cycler(
                color=[
                    "#2563eb",
                    "#dc2626",
                    "#059669",
                    "#d97706",
                    "#7c3aed",
                    "#0891b2",
                    "#be123c",
                    "#4b5563",
                ]
            ),
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#cbd5e1",
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "savefig.facecolor": "#f8fafc",
            "savefig.edgecolor": "#f8fafc",
            "figure.dpi": 110,
            "svg.fonttype": "none",
        }
    )


def _apply_axis_style(ax, ylabel: str, xlabel: str = "Temps") -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(True, axis="y", which="major", alpha=0.85, linewidth=0.9)
    ax.grid(True, axis="x", which="major", alpha=0.28, linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=10, colors="#334155", length=0, pad=6)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.margins(x=0.02)


def _save_svg(fig, filename: str | Path | None) -> None:
    if filename is None:
        return
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN, forward=False)
    fig.savefig(
        filename,
        format="svg",
        bbox_inches=None,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )


def plot_inventory_trajectories(
    t_grid: np.ndarray,
    qbar: np.ndarray,
    *,
    reference_qbar: np.ndarray | None = None,
    filename: str | Path | None = None,
):
    """Plot mean inventories."""

    setup_plot_theme()
    p = qbar.shape[0]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=110)
    fig.patch.set_facecolor("#f8fafc")

    for i in range(p):
        color = POPULATION_COLORS[i % len(POPULATION_COLORS)]
        ax.plot(
            t_grid,
            qbar[i, :, 0],
            color=color,
            linestyle="-",
            linewidth=2.4,
            alpha=1.0,
            label=f"Sous-population {i + 1}",
            zorder=4,
        )
        if reference_qbar is not None:
            ax.plot(
                t_grid,
                reference_qbar[i, :, 0],
                color=color,
                linestyle=(0, (5, 3)),
                linewidth=1.9,
                alpha=0.80,
                label=f"Réf. homogène — population {i + 1}",
                zorder=2,
            )

    title = (
        "Interaction hétérogène et références homogènes"
        if reference_qbar is not None
        else "Trajectoires d'inventaire moyen"
    )
    ax.set_title(title, loc="left", pad=12)
    _apply_axis_style(ax, "Inventaire moyen")
    ax.legend(loc="upper right", ncol=1, frameon=True, fancybox=True, framealpha=0.95)
    fig.tight_layout()
    _save_svg(fig, filename)
    return fig, ax


def plot_speed_trajectories(
    t_grid: np.ndarray,
    vbar: np.ndarray,
    rho_list,
    *,
    reference_vbar: np.ndarray | None = None,
    filename: str | Path | None = None,
):
    """Plot population trading speeds and the aggregate flow."""

    setup_plot_theme()
    p = vbar.shape[0]
    mu = compute_aggregated_flow(vbar, rho_list)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=110)
    fig.patch.set_facecolor("#f8fafc")

    for i in range(p):
        color = POPULATION_COLORS[i % len(POPULATION_COLORS)]
        ax.plot(
            t_grid,
            vbar[i, :, 0],
            color=color,
            linestyle="-",
            linewidth=2.4,
            alpha=1.0,
            label=f"Sous-population {i + 1}",
            zorder=4,
        )
        if reference_vbar is not None:
            ax.plot(
                t_grid,
                reference_vbar[i, :, 0],
                color=color,
                linestyle=(0, (5, 3)),
                linewidth=1.9,
                alpha=0.80,
                label=f"Réf. homogène — population {i + 1}",
                zorder=2,
            )

    ax.plot(
        t_grid,
        mu[:, 0],
        color=AGGREGATE_COLOR,
        linestyle="-.",
        linewidth=2.7,
        alpha=0.95,
        label="Flux agrégé μ",
        zorder=5,
    )

    ax.set_title("Trajectoires de μ et des vitesses de négociation", loc="left", pad=12)
    _apply_axis_style(ax, "Vitesse de négociation")
    ax.legend(loc="lower right", ncol=1, frameon=True, fancybox=True, framealpha=0.95)
    fig.tight_layout()
    _save_svg(fig, filename)
    return fig, ax
