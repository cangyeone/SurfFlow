#!/usr/bin/env python3
"""Make a compact calibration/reliability figure for the GJI manuscript."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "calibration"
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "SurfFlow" / "manuscript" / "figures"
SURFFLOW_FIG_DIR = ROOT / "SurfFlow" / "figures" / "fair_di_comparison" / "production" / "calibration"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_fig01_asset_panels import GRID, INK, PAPER, POST_BLUE, style_axis  # noqa: E402


WEAK_ORANGE = "#C9823A"
SUBTLE = "#8B95A3"
NOMINAL = 0.68

REGIMES = ["in-prior", "boundary", "out-of-prior"]
REGIME_LABELS = {
    "in-prior": "Inside support",
    "boundary": "Near edge",
    "out-of-prior": "Outside support",
}
METHODS = ["DI-Strong", "DI-Weak"]
METHOD_COLORS = {"DI-Strong": POST_BLUE, "DI-Weak": WEAK_ORANGE}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.72,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for(
    rows: list[dict[str, str]],
    *,
    method: str,
    regime: str,
    scale_label: str | None = None,
    nominal: float | None = None,
) -> list[dict[str, str]]:
    out = [r for r in rows if r["method"] == method and r["test_set"] == regime]
    if scale_label is not None:
        out = [r for r in out if r["scale_label"] == scale_label]
    if nominal is not None:
        out = [r for r in out if abs(float(r["nominal_percent"]) - nominal) < 1e-6]
    return out


def metric_value(
    rows: list[dict[str, str]], *, method: str, regime: str, scale_label: str, nominal: float, field: str
) -> float:
    matches = rows_for(rows, method=method, regime=regime, scale_label=scale_label, nominal=nominal)
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {method}/{regime}/{scale_label}/{nominal}, got {len(matches)}")
    return float(matches[0][field])


def panel_label(ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    ax.figure.text(
        box.x0 - 0.034,
        box.y1 + 0.045,
        f"({label})",
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=INK,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR, SURFFLOW_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(out_dir / f"{stem}.png", dpi=440, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def plot_reliability_curves(ax: plt.Axes, rows: list[dict[str, str]], regime: str, *, show_ylabel: bool) -> None:
    ax.plot([0, 100], [0, 1], color="#23272E", lw=0.72, ls=(0, (2.0, 1.7)), zorder=1)
    for method in METHODS:
        color = METHOD_COLORS[method]
        for scale_label, ls, marker, alpha, fill in (
            ("raw", (0, (1.2, 1.35)), "o", 0.85, PAPER),
            ("temperature_scaled", "-", "s", 0.98, color),
        ):
            subset = rows_for(rows, method=method, regime=regime, scale_label=scale_label)
            subset = sorted(subset, key=lambda r: float(r["nominal_percent"]))
            xs = [float(r["nominal_percent"]) for r in subset]
            ys = [float(r["coverage_mean"]) for r in subset]
            ax.plot(
                xs,
                ys,
                color=color,
                lw=1.05,
                ls=ls,
                marker=marker,
                ms=3.0,
                mfc=fill,
                mec=color,
                mew=0.65,
                alpha=alpha,
                zorder=4,
            )
    ax.set_title(REGIME_LABELS[regime], loc="left", fontsize=7.0, color=INK, pad=2.0)
    ax.set_xlim(45, 94)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([50, 68, 90])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Nominal interval (%)", fontsize=6.4, labelpad=1.0)
    if show_ylabel:
        ax.set_ylabel("Empirical coverage", fontsize=6.4, labelpad=1.0)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    style_axis(ax)


def summary_rows(rows: list[dict[str, str]]) -> list[tuple[str, str, float, float, float]]:
    out = []
    for method in METHODS:
        for regime in REGIMES:
            raw = metric_value(rows, method=method, regime=regime, scale_label="raw", nominal=68.0, field="coverage_mean")
            scaled = metric_value(
                rows, method=method, regime=regime, scale_label="temperature_scaled", nominal=68.0, field="coverage_mean"
            )
            tau = metric_value(
                rows, method=method, regime=regime, scale_label="temperature_scaled", nominal=68.0, field="temperature_scale"
            )
            out.append((method, regime, raw, scaled, tau))
    return out


def plot_coverage_dumbbell(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    summary = summary_rows(rows)
    y_positions = list(range(len(summary)))[::-1]
    for y, (method, regime, raw, scaled, _tau) in zip(y_positions, summary):
        color = METHOD_COLORS[method]
        ax.plot([raw, scaled], [y, y], color="#D2D8E0", lw=1.2, zorder=1)
        ax.scatter(raw, y, s=20, facecolor=PAPER, edgecolor=color, linewidth=0.85, zorder=3)
        ax.scatter(scaled, y, s=22, facecolor=color, edgecolor=color, linewidth=0.65, zorder=4)
    ax.axvline(NOMINAL, color="#667282", lw=0.82, ls=(0, (2.2, 1.8)), zorder=0)
    ax.text(
        NOMINAL + 0.012,
        len(summary) - 0.45,
        "nominal 0.68",
        ha="left",
        va="center",
        fontsize=5.75,
        color="#667282",
    )
    labels = [f"{m.replace('DI-', '')}\n{REGIME_LABELS[r]}" for m, r, *_ in summary]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=5.55)
    ax.set_xlim(0.05, 0.92)
    ax.set_xticks([0.2, 0.5, 0.68, 0.9])
    ax.set_xlabel("Mean pointwise coverage at nominal 68%", fontsize=6.4, labelpad=1.0)
    ax.set_title("Split-sample coverage correction", loc="left", fontsize=7.0, color=INK, pad=2.0)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)


def plot_temperature_factors(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    summary = summary_rows(rows)
    y_positions = list(range(len(summary)))[::-1]
    for y, (method, _regime, _raw, _scaled, tau) in zip(y_positions, summary):
        color = METHOD_COLORS[method]
        ax.plot([1.0, tau], [y, y], color=color, lw=1.5, solid_capstyle="round", alpha=0.88, zorder=2)
        ax.scatter(tau, y, s=23, facecolor=color, edgecolor=color, linewidth=0.5, zorder=3)
        ax.text(tau + 0.08, y, f"{tau:.2f}", ha="left", va="center", fontsize=5.55, color=INK)
    ax.axvline(1.0, color="#667282", lw=0.82, ls=(0, (2.2, 1.8)), zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlim(0.85, 5.15)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel(r"Temperature factor $\tau$", fontsize=6.4, labelpad=1.0)
    ax.set_title("Posterior widening required", loc="left", fontsize=7.0, color=INK, pad=2.0)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)


def plot_coverage_at_68(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x_base = {regime: i for i, regime in enumerate(REGIMES)}
    offsets = {"DI-Strong": -0.13, "DI-Weak": 0.13}
    for method in METHODS:
        color = METHOD_COLORS[method]
        for regime in REGIMES:
            x = x_base[regime] + offsets[method]
            raw = metric_value(rows, method=method, regime=regime, scale_label="raw", nominal=68.0, field="coverage_mean")
            scaled = metric_value(
                rows,
                method=method,
                regime=regime,
                scale_label="temperature_scaled",
                nominal=68.0,
                field="coverage_mean",
            )
            ax.plot([x, x], [raw, scaled], color=color, lw=1.05, alpha=0.45, zorder=2)
            if scaled > raw + 0.025:
                ax.annotate(
                    "",
                    xy=(x, scaled - 0.018),
                    xytext=(x, raw + 0.018),
                    arrowprops=dict(arrowstyle="-|>", lw=0.70, color=color, alpha=0.62, mutation_scale=7),
                    zorder=2,
                )
            ax.scatter(x, raw, s=28, facecolor=PAPER, edgecolor=color, linewidth=1.0, zorder=4)
            ax.scatter(x, scaled, s=30, facecolor=color, edgecolor=color, linewidth=0.70, zorder=5)

    ax.axhline(NOMINAL, color="#667282", lw=0.82, ls=(0, (2.2, 1.8)), zorder=1)
    ax.text(
        len(REGIMES) - 0.52,
        NOMINAL + 0.025,
        "target 0.68",
        ha="left",
        va="bottom",
        fontsize=5.85,
        color="#667282",
    )
    ax.set_xlim(-0.55, len(REGIMES) - 0.45)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=6.1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.68, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.68", "0.75", "1.00"], fontsize=5.85)
    ax.set_ylabel("Truth inside nominal 68% interval", fontsize=6.4, labelpad=1.0)
    ax.set_title("68% interval coverage check", loc="left", fontsize=7.0, color=INK, pad=2.0)
    style_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.grid(axis="x", color="#EEF2F6", lw=0.45)


def plot_tau_bars(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x = list(range(len(REGIMES)))
    width = 0.34
    offsets = {"DI-Strong": -width / 2.0, "DI-Weak": width / 2.0}
    for method in METHODS:
        color = METHOD_COLORS[method]
        values = [
            metric_value(
                rows,
                method=method,
                regime=regime,
                scale_label="temperature_scaled",
                nominal=68.0,
                field="temperature_scale",
            )
            for regime in REGIMES
        ]
        xpos = [i + offsets[method] for i in x]
        ax.bar(xpos, values, width=width * 0.82, color=color, edgecolor=color, linewidth=0.35, alpha=0.88, zorder=3)
        for xx, value in zip(xpos, values):
            ax.text(xx, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=5.65, color=INK)
    ax.axhline(1.0, color="#667282", lw=0.82, ls=(0, (2.2, 1.8)), zorder=1)
    ax.text(
        len(REGIMES) - 0.60,
        1.08,
        r"$\tau=1$: no widening",
        ha="left",
        va="bottom",
        fontsize=5.75,
        color="#667282",
    )
    ax.set_xlim(-0.52, len(REGIMES) - 0.48)
    ax.set_ylim(0.0, 5.15)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=6.1)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel(r"Posterior spread multiplier $\tau$", fontsize=6.4, labelpad=1.0)
    ax.set_title("Fitted posterior widening", loc="left", fontsize=7.0, color=INK, pad=2.0)
    style_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.5)


def plot_reliability_matrix(rows: list[dict[str, str]], stem: str) -> None:
    fig = plt.figure(figsize=(7.15, 3.75), facecolor=PAPER)
    gs = fig.add_gridspec(
        2,
        3,
        left=0.095,
        right=0.992,
        bottom=0.145,
        top=0.855,
        hspace=0.30,
        wspace=0.20,
    )
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
    for i, method in enumerate(METHODS):
        color = METHOD_COLORS[method]
        for j, regime in enumerate(REGIMES):
            ax = axes[i][j]
            ax.plot([0, 100], [0, 1], color="#252A31", lw=0.78, ls=(0, (2.0, 1.7)), zorder=1)
            for scale_label, marker, ls, fill, label in (
                ("raw", "o", (0, (1.25, 1.25)), PAPER, "raw"),
                ("temperature_scaled", "s", "-", color, "temperature scaled"),
            ):
                subset = rows_for(rows, method=method, regime=regime, scale_label=scale_label)
                subset = sorted(subset, key=lambda r: float(r["nominal_percent"]))
                x = [float(r["nominal_percent"]) for r in subset]
                y = [float(r["coverage_mean"]) for r in subset]
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=1.15,
                    ls=ls,
                    marker=marker,
                    ms=3.3,
                    mfc=fill,
                    mec=color,
                    mew=0.72,
                    label=label if (i, j) == (0, 0) else None,
                    zorder=4,
                )
            tau = metric_value(
                rows,
                method=method,
                regime=regime,
                scale_label="temperature_scaled",
                nominal=68.0,
                field="temperature_scale",
            )
            ax.text(
                0.955,
                0.085,
                rf"$\tau={tau:.2f}$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.85,
                color="#5F6B7A",
            )
            if i == 0:
                ax.set_title(REGIME_LABELS[regime], fontsize=7.15, color=INK, pad=2.5)
            ax.set_xlim(45, 94)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([50, 68, 90])
            ax.set_yticks([0.0, 0.5, 1.0])
            if i == 1:
                ax.set_xlabel("Nominal credible interval (%)", fontsize=6.25, labelpad=1.0)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Held-out coverage", fontsize=6.25, labelpad=1.0)
            else:
                ax.set_yticklabels([])
            style_axis(ax)
            if (i, j) == (0, 0):
                ax.text(
                    0.145,
                    0.790,
                    "ideal calibration",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=5.45,
                    color="#404852",
                    rotation=31,
                )

    for i, method in enumerate(METHODS):
        row_box = axes[i][0].get_position()
        fig.text(
            0.025,
            (row_box.y0 + row_box.y1) / 2,
            method,
            ha="left",
            va="center",
            rotation=90,
            fontsize=7.2,
            color=METHOD_COLORS[method],
            fontweight="bold",
        )

    legend_handles = [
        Line2D([0], [0], color="#252A31", lw=0.78, ls=(0, (2.0, 1.7)), label="nominal = empirical"),
        Line2D(
            [0],
            [0],
            color="#4A5564",
            marker="o",
            mfc=PAPER,
            mec="#4A5564",
            lw=1.0,
            ls=(0, (1.25, 1.25)),
            label="raw",
        ),
        Line2D(
            [0],
            [0],
            color="#4A5564",
            marker="s",
            mfc="#4A5564",
            mec="#4A5564",
            lw=1.0,
            label="temperature scaled",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.635, 0.972),
        ncol=3,
        fontsize=6.15,
        columnspacing=1.05,
        handletextpad=0.42,
        frameon=False,
    )
    fig.text(
        0.095,
        0.970,
        "Posterior reliability",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
    )
    save_figure(fig, stem)


def make_figure(rows: list[dict[str, str]], stem: str) -> None:
    fig = plt.figure(figsize=(7.25, 3.05), facecolor=PAPER)
    gs = fig.add_gridspec(
        1,
        5,
        left=0.085,
        right=0.985,
        bottom=0.175,
        top=0.850,
        wspace=0.68,
    )
    ax_cov = fig.add_subplot(gs[0, 0:3])
    ax_tau = fig.add_subplot(gs[0, 3:5])
    plot_coverage_at_68(ax_cov, rows)
    plot_tau_bars(ax_tau, rows)

    panel_label(ax_cov, "a")
    panel_label(ax_tau, "b")

    legend_handles = [
        Line2D([0], [0], color=POST_BLUE, marker="o", lw=1.0, label="DI-Strong"),
        Line2D([0], [0], color=WEAK_ORANGE, marker="o", lw=1.0, label="DI-Weak"),
        Line2D([0], [0], color="#4A5564", marker="o", mfc=PAPER, mec="#4A5564", lw=0, label="raw"),
        Line2D([0], [0], color="#4A5564", marker="o", mfc="#4A5564", mec="#4A5564", lw=0, label="scaled"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=4,
        columnspacing=0.92,
        handletextpad=0.42,
        fontsize=6.15,
        frameon=False,
    )
    save_figure(fig, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "calibration_metrics.csv")
    parser.add_argument("--stem", default="fig04_calibration_reliability_v4")
    parser.add_argument("--layout", choices=("summary", "curves"), default="summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.metrics)
    if args.layout == "curves":
        plot_reliability_matrix(rows, args.stem)
    else:
        make_figure(rows, args.stem)


if __name__ == "__main__":
    main()
