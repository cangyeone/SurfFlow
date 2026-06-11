#!/usr/bin/env python3
"""Draw Figure 1 panel A: prior-predictive construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_PATH = Path(
    "/Volumes/lx_exFAT/yzy_directSWI/code_data/results/fair_di_comparison/production/fair_di_diagnostics.npz"
)
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "SurfFlow" / "manuscript" / "figures"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


INK = "#20242B"
SUBTLE = "#697281"
GRID = "#E6E9ED"
STRONG = "#3F6FA7"
WEAK = "#D07C2C"
SHADE = "#EEF1F4"


def finish(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.75, colors=INK)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(color=GRID, linewidth=0.62)


def quantiles(values: np.ndarray, qs: tuple[float, ...]) -> list[np.ndarray]:
    return [np.quantile(values, q, axis=0) for q in qs]


def draw_prior_support(ax: plt.Axes, depth: np.ndarray, prior_vs: np.ndarray) -> None:
    prior_05, prior_50, prior_95 = quantiles(prior_vs, (0.05, 0.50, 0.95))

    ax.axhspan(75, 100, color=SHADE, zorder=0)
    for curve in prior_vs[::57][:18]:
        ax.plot(curve, depth, color=STRONG, lw=0.34, alpha=0.06, zorder=1)
    ax.fill_betweenx(depth, prior_05, prior_95, color=STRONG, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(prior_50, depth, color=STRONG, lw=1.55, zorder=4)

    ax.set_ylim(100, 0)
    ax.set_xlim(1.55, 5.7)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Structural prior support", loc="left", fontsize=7.9, color=INK, pad=3.0)
    style_axis(ax)


def draw_prior_predictive(ax: plt.Axes, prior_disp: np.ndarray) -> None:
    periods = prior_disp[0, 0]
    prior_ray = prior_disp[:, 1]
    prior_10, prior_50, prior_90 = quantiles(prior_ray, (0.10, 0.50, 0.90))

    for curve in prior_ray[::57][:18]:
        ax.plot(periods, curve, color=STRONG, lw=0.34, alpha=0.055, zorder=1)
    ax.fill_between(periods, prior_10, prior_90, color=STRONG, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(periods, prior_50, color=STRONG, lw=1.55, zorder=4)

    ax.set_xlim(2, 60)
    ax.set_ylim(0.65, 4.65)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_title("Prior-predictive Rayleigh data", loc="left", fontsize=7.9, color=INK, pad=3.0)
    style_axis(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--stem", default="fig01_A_panel_ready_prior_predictive")
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    strong_key = "DI_Strong_in_prior"
    strong_target = np.asarray(diagnostics[f"{strong_key}_target"], dtype=float)
    strong_disp = np.asarray(diagnostics[f"{strong_key}_disp"], dtype=float)
    depth = np.linspace(0.0, 127.5, strong_target.shape[-1], dtype=float)

    fig = plt.figure(figsize=(6.75, 2.65))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.08],
        left=0.078,
        right=0.992,
        bottom=0.19,
        top=0.865,
        wspace=0.30,
    )
    ax_prior = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])

    draw_prior_support(ax_prior, depth, strong_target[:, 1])
    draw_prior_predictive(ax_pred, strong_disp)

    fig.text(0.008, 0.982, "A", ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    fig.add_artist(
        FancyArrowPatch(
            (0.475, 0.55),
            (0.522, 0.55),
            transform=fig.transFigure,
            arrowstyle="-|>",
            lw=0.9,
            color=SUBTLE,
            mutation_scale=10,
        )
    )
    finish(fig, args.stem)
    diagnostics.close()


if __name__ == "__main__":
    main()
