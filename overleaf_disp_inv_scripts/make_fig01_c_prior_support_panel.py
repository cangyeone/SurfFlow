#!/usr/bin/env python3
"""Draw Figure 1 panel C: prior-support pull-in audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_PATH = Path(
    "/Volumes/lx_exFAT/yzy_directSWI/code_data/results/fair_di_comparison/production/fair_di_diagnostics.npz"
)
DEFAULT_METRICS_PATH = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "fair_di_metrics.csv"
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
TARGET = "#111111"
POST = "#D07C2C"
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


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(rows: list[dict[str, str]], method: str, regime: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and row["test_set"] == regime)


def draw_profile_audit(ax: plt.Axes, diagnostics: np.lib.npyio.NpzFile, regime: str, index: int) -> None:
    prior = np.asarray(diagnostics["DI_Strong_in_prior_target"][:, 1], dtype=float)
    key = f"DI_Strong_{regime}"
    target = np.asarray(diagnostics[f"{key}_target"][index, 1], dtype=float)
    posterior = np.asarray(diagnostics[f"{key}_samples"][index, :, 1], dtype=float)
    posterior_median = np.median(posterior, axis=0)
    depth = np.linspace(0.0, 127.5, prior.shape[-1], dtype=float)

    prior_05 = np.quantile(prior, 0.05, axis=0)
    prior_50 = np.quantile(prior, 0.50, axis=0)
    prior_95 = np.quantile(prior, 0.95, axis=0)
    posterior_10 = np.quantile(posterior, 0.10, axis=0)
    posterior_90 = np.quantile(posterior, 0.90, axis=0)

    ax.axhspan(75, 100, color=SHADE, zorder=0)
    ax.fill_betweenx(depth, prior_05, prior_95, color=STRONG, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(prior_50, depth, color=STRONG, lw=1.2, alpha=0.95, zorder=3)
    ax.fill_betweenx(depth, posterior_10, posterior_90, color=POST, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(target, depth, color=TARGET, lw=1.35, zorder=5)
    ax.plot(posterior_median, depth, color=POST, lw=1.75, zorder=6)

    ax.set_ylim(100, 0)
    ax.set_xlim(1.55, 5.7)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Prior-support example", loc="left", fontsize=7.9, color=INK, pad=3.0)
    ax.text(0.62, 0.82, "prior", transform=ax.transAxes, color=STRONG, fontsize=6.2, ha="left")
    ax.text(0.62, 0.74, "posterior", transform=ax.transAxes, color=POST, fontsize=6.2, ha="left")
    ax.text(0.62, 0.66, "target", transform=ax.transAxes, color=TARGET, fontsize=6.2, ha="left")
    style_axis(ax)


def draw_pullin_summary(ax: plt.Axes, metrics: list[dict[str, str]]) -> None:
    regimes = ["boundary", "out-of-prior"]
    y_base = np.arange(len(regimes))[::-1]
    offsets = {"DI-Strong": 0.09, "DI-Weak": -0.09}
    colors = {"DI-Strong": STRONG, "DI-Weak": WEAK}

    for method in ["DI-Strong", "DI-Weak"]:
        xs = []
        lo = []
        hi = []
        ys = []
        for regime, y in zip(regimes, y_base):
            row = metric_row(metrics, method, regime)
            value = float(row["pred_inside_given_target_outside"])
            low = float(row["pred_inside_given_target_outside_ci_low"])
            high = float(row["pred_inside_given_target_outside_ci_high"])
            xs.append(value)
            lo.append(value - low)
            hi.append(high - value)
            ys.append(y + offsets[method])
        ax.errorbar(
            xs,
            ys,
            xerr=[lo, hi],
            fmt="o",
            color=colors[method],
            ecolor=colors[method],
            elinewidth=1.15,
            capsize=2.5,
            markersize=4.3,
            zorder=4,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(["Boundary", "Out-of-prior"])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.45, len(regimes) - 0.55)
    ax.set_xlabel("Pulled inside prior support")
    ax.set_title("Prior pull-in audit", loc="left", fontsize=7.9, color=INK, pad=3.0)
    ax.grid(axis="x", color=GRID, linewidth=0.62)
    ax.grid(axis="y", color="#F1F3F5", linewidth=0.45)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4, colors=INK)
    ax.tick_params(axis="x", direction="out", length=3.0, width=0.75, colors=INK)
    ax.text(0.78, 0.82, "DI-Strong", color=STRONG, transform=ax.transAxes, fontsize=6.2, ha="left")
    ax.text(0.78, 0.73, "DI-Weak", color=WEAK, transform=ax.transAxes, fontsize=6.2, ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--regime", choices=["boundary", "out_of_prior"], default="boundary")
    parser.add_argument("--index", type=int, default=491)
    parser.add_argument("--stem", default="fig01_C_panel_ready_prior_support")
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    metrics = read_metrics(args.metrics)

    fig = plt.figure(figsize=(6.75, 2.65))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.03, 0.92],
        left=0.078,
        right=0.992,
        bottom=0.19,
        top=0.865,
        wspace=0.34,
    )
    ax_profile = fig.add_subplot(gs[0, 0])
    ax_pull = fig.add_subplot(gs[0, 1])

    draw_profile_audit(ax_profile, diagnostics, args.regime, args.index)
    draw_pullin_summary(ax_pull, metrics)
    fig.text(0.008, 0.982, "C", ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)

    finish(fig, args.stem)
    diagnostics.close()


if __name__ == "__main__":
    main()
