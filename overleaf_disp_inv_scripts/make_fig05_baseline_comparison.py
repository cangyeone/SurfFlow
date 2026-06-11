#!/usr/bin/env python3
"""Draw a publication-style baseline comparison figure for the GJI manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SURFFLOW = ROOT / "SurfFlow"
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = SURFFLOW / "manuscript" / "figures"
SURFFLOW_FIG_DIR = SURFFLOW / "figures" / "fair_di_comparison" / "production" / "baselines"

FAIR_DI = SURFFLOW / "results" / "fair_di_comparison" / "production" / "fair_di_metrics.csv"
BASELINES = SURFFLOW / "results" / "fair_di_comparison" / "production" / "baselines" / "baseline_metrics.csv"

QEDISP_ROOT = Path(
    "/Users/liuxin/CodexWorkspace/Codes/dispersion_curve_calculation/deliverables/"
    "qedispinv_iso_inversion_clean_20260605/joint_rayleigh_love_experiments/"
    "rayleigh_love_joint_inversion_20260605/results"
)
QEDISP_MULTIMODE = QEDISP_ROOT / "rayleigh_multimode_traditional_baseline_s48_nfev80_20260608" / "joint_inversion_summary.json"
QEDISP_M0 = QEDISP_ROOT / "rayleigh_m0_traditional_baseline_s48_nfev80_20260608" / "joint_inversion_summary.json"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_fig01_asset_panels import GRID, INK, PAPER, POST_BLUE, style_axis  # noqa: E402


WEAK_ORANGE = "#C9823A"
GREY = "#727B88"
LIGHT_GREY = "#D9DEE6"
REGIMES = ["in-prior", "boundary", "out-of-prior"]
REGIME_LABELS = {
    "in-prior": "In prior",
    "boundary": "Near boundary",
    "out-of-prior": "Out of prior",
}


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


def metric(rows: list[dict[str, str]], method: str, regime: str, field: str) -> float:
    matches = [r for r in rows if r["method"] == method and r["test_set"] == regime]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {method}/{regime}, got {len(matches)}")
    value = matches[0].get(field, "")
    if value in {"", "nan", "NaN", "None"}:
        return float("nan")
    return float(value)


def qmetric(path: Path, field: str) -> float:
    data = json.loads(path.read_text())
    node: object = data
    for key in field.split("."):
        if not isinstance(node, dict):
            raise KeyError(field)
        node = node[key]
    return float(node)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR, SURFFLOW_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(out_dir / f"{stem}.png", dpi=440, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.975,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        zorder=10,
    )


def bar_style(method: str) -> dict[str, object]:
    color = POST_BLUE if method.endswith("Strong") else WEAK_ORANGE
    if method.startswith("DI"):
        return {"facecolor": color, "edgecolor": color, "hatch": None, "alpha": 0.92}
    return {"facecolor": PAPER, "edgecolor": color, "hatch": "///", "alpha": 1.0}


def plot_metric_bars(ax: plt.Axes, rows: list[dict[str, str]], field: str, ylabel: str, title: str) -> None:
    x = np.arange(len(REGIMES), dtype=float)
    methods = ["DI-Strong", "DET-Strong", "DI-Weak", "DET-Weak"]
    offsets = [-0.255, -0.085, 0.085, 0.255]
    width = 0.145
    for method, offset in zip(methods, offsets):
        vals = [metric(rows, method, regime, field) for regime in REGIMES]
        style = bar_style(method)
        ax.bar(
            x + offset,
            vals,
            width=width,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.78,
            hatch=style["hatch"],
            alpha=style["alpha"],
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=6.0)
    ax.set_xlim(-0.48, len(REGIMES) - 0.52)
    ax.set_ylabel(ylabel, fontsize=6.35, labelpad=1.1)
    ax.set_title(title, loc="left", fontsize=7.0, color=INK, pad=2.5)
    style_axis(ax)


def plot_coverage(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x = np.arange(len(REGIMES), dtype=float)
    methods = ["DI-Strong", "DI-Weak"]
    offsets = [-0.095, 0.095]
    width = 0.165
    for method, offset in zip(methods, offsets):
        color = POST_BLUE if method.endswith("Strong") else WEAK_ORANGE
        vals = [metric(rows, method, regime, "coverage_16_84_mean") for regime in REGIMES]
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.45,
            alpha=0.92,
            zorder=4,
        )
    ax.axhline(0.68, color="#404852", lw=0.82, ls=(0, (2.1, 1.8)), zorder=1)
    ax.text(
        2.33,
        0.697,
        "nominal 0.68",
        ha="right",
        va="bottom",
        fontsize=5.75,
        color="#404852",
    )
    ax.set_xlim(-0.48, len(REGIMES) - 0.52)
    ax.set_ylim(0.0, 0.82)
    ax.set_yticks([0.0, 0.34, 0.68])
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=6.0)
    ax.set_ylabel("Mean 16-84% interval coverage", fontsize=6.35, labelpad=1.1)
    ax.set_title("DI-only interval coverage", loc="left", fontsize=7.0, color=INK, pad=2.5)
    style_axis(ax)


def plot_qedisp(ax: plt.Axes) -> None:
    labels = ["RMS", "MAE"]
    values = [qmetric(QEDISP_M0, "best.metrics.rms_mps"), qmetric(QEDISP_M0, "best.metrics.mae_mps")]
    x = np.arange(len(labels), dtype=float)
    colors = ["#4A5564", "#AEB7C3"]
    ax.bar(x, values, width=0.42, color=colors, edgecolor=colors, lw=0.35, zorder=4)
    for xx, yy in zip(x, values):
        ax.text(xx, yy + 0.13, f"{yy:.1f}", ha="center", va="bottom", fontsize=5.85, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("Dispersion residual (m/s)", fontsize=6.35, labelpad=1.0)
    ax.set_ylim(0, max(values) * 1.45)
    ax.set_title("Conventional Rayleigh inversion", loc="left", fontsize=7.0, color=INK, pad=2.5)
    style_axis(ax)


def make_figure(stem: str) -> None:
    di_rows = read_csv(FAIR_DI)
    baseline_rows = read_csv(BASELINES)
    rows = di_rows + baseline_rows

    fig = plt.figure(figsize=(6.95, 3.38), facecolor=PAPER)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.088,
        right=0.985,
        bottom=0.145,
        top=0.765,
        hspace=0.44,
        wspace=0.36,
    )
    ax_vs = fig.add_subplot(gs[0, 0])
    ax_disp = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, :])

    plot_metric_bars(ax_vs, rows, "vs_mae", "$V_S$ MAE (km/s)", "Model error")
    ax_vs.set_ylim(0.0, 0.90)
    ax_vs.set_yticks([0.0, 0.3, 0.6, 0.9])

    plot_metric_bars(ax_disp, rows, "pred_disp_mae", "Dispersion MAE (km/s)", "Forward-modelled dispersion error")
    ax_disp.set_ylim(0.0, 0.42)
    ax_disp.set_yticks([0.0, 0.14, 0.28, 0.42])

    plot_coverage(ax_cov, rows)

    panel_label(ax_vs, "a")
    panel_label(ax_disp, "b")
    panel_label(ax_cov, "c")

    legend_handles = [
        Patch(facecolor=POST_BLUE, edgecolor=POST_BLUE, label="strong prior + DI sampler"),
        Patch(facecolor=PAPER, edgecolor=POST_BLUE, hatch="///", label="strong prior + point estimate"),
        Patch(facecolor=WEAK_ORANGE, edgecolor=WEAK_ORANGE, label="weak prior + DI sampler"),
        Patch(facecolor=PAPER, edgecolor=WEAK_ORANGE, hatch="///", label="weak prior + point estimate"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.992),
        ncol=2,
        fontsize=5.55,
        columnspacing=0.95,
        handlelength=1.00,
        handletextpad=0.35,
        frameon=False,
        title="Panels (a,b): training prior × network output",
        title_fontsize=5.85,
    )
    save_figure(fig, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="fig05_baseline_comparison_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.stem)


if __name__ == "__main__":
    main()
