#!/usr/bin/env python3
"""Make a publication-style Figure 3 for matched DI diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_fig01_asset_panels import (  # noqa: E402
    DEFAULT_DIAGNOSTICS_PATH,
    DEFAULT_METRICS_PATH,
    GRID,
    INK,
    PAPER,
    POST_BLUE,
    POST_BLUE_FILL,
    SUBTLE,
    TRUTH,
    forward_rayleigh,
    quantiles,
    style_axis,
)


PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "SurfFlow" / "manuscript" / "figures"
SURFFLOW_FIG_DIR = ROOT / "SurfFlow" / "figures" / "fair_di_comparison" / "production"
WEAK_ORANGE = "#C9823A"
WEAK_FILL = "#F2DEC8"


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


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(rows: list[dict[str, str]], method: str, regime: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and row["test_set"] == regime)


def panel_label(ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    ax.figure.text(
        box.x0 - 0.030,
        box.y1 + 0.026,
        f"({label})",
        ha="left",
        va="top",
        fontsize=7.6,
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


def plot_metric_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ci_low: str,
    ci_high: str,
    xlabel: str,
    title: str,
    regimes: list[str],
    labels: list[str],
    xlim: tuple[float, float],
    nominal: float | None = None,
    show_case_labels: bool = False,
) -> None:
    y = np.arange(len(regimes), dtype=float)[::-1]
    styles = {
        "DI-Strong": {"color": POST_BLUE, "marker": "o"},
        "DI-Weak": {"color": WEAK_ORANGE, "marker": "s"},
    }
    for band in range(len(regimes)):
        if band % 2 == 0:
            ax.axhspan(y[band] - 0.34, y[band] + 0.34, color="#F7F9FB", lw=0, zorder=0)
    values_by_method: dict[str, list[float]] = {}
    for method in styles:
        values_by_method[method] = [float(metric_row(rows, method, regime)[metric]) for regime in regimes]
    for j, yy in enumerate(y):
        ax.plot(
            [values_by_method["DI-Strong"][j], values_by_method["DI-Weak"][j]],
            [yy, yy],
            color="#CFD6DF",
            lw=1.05,
            solid_capstyle="round",
            zorder=1,
        )
    for method, style in styles.items():
        values = []
        lo = []
        hi = []
        for regime in regimes:
            row = metric_row(rows, method, regime)
            value = float(row[metric])
            values.append(value)
            lo.append(value - float(row[ci_low]))
            hi.append(float(row[ci_high]) - value)
        ax.errorbar(
            values,
            y,
            xerr=[lo, hi],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            ms=3.8,
            capsize=1.9,
            elinewidth=0.78,
            markeredgecolor=PAPER,
            markeredgewidth=0.38,
            label=method,
            zorder=4,
        )
    if nominal is not None:
        ax.axvline(nominal, color="#9AA3AE", lw=0.78, ls=(0, (2.2, 2.0)), zorder=1)
        ax.text(
            nominal + 0.012 * (xlim[1] - xlim[0]),
            2.43,
            "nominal 0.68",
            ha="left",
            va="top",
            fontsize=4.9,
            color=SUBTLE,
        )
    ax.set_yticks(y)
    if show_case_labels:
        ax.set_yticklabels(labels, fontsize=5.75)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.48, len(regimes) - 0.52)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=6.1, labelpad=1.4)
    ax.set_title(title, loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax, grid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.50)


def plot_metric_panel_old(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ci_low: str,
    ci_high: str,
    ylabel: str,
    title: str,
    regimes: list[str],
    labels: list[str],
    ylim: tuple[float, float],
    nominal: float | None = None,
) -> None:
    x = np.arange(len(regimes), dtype=float)
    styles = {
        "DI-Strong": {"color": POST_BLUE, "marker": "o", "ls": (0, (3.0, 1.8))},
        "DI-Weak": {"color": WEAK_ORANGE, "marker": "s", "ls": (0, (1.2, 1.35))},
    }
    for method, style in styles.items():
        values = []
        lo = []
        hi = []
        for regime in regimes:
            row = metric_row(rows, method, regime)
            value = float(row[metric])
            values.append(value)
            lo.append(value - float(row[ci_low]))
            hi.append(float(row[ci_high]) - value)
        ax.errorbar(
            x,
            values,
            yerr=[lo, hi],
            color=style["color"],
            marker=style["marker"],
            ls=style["ls"],
            lw=0.95,
            ms=2.7,
            capsize=1.6,
            elinewidth=0.58,
            label=method,
            zorder=4,
        )
    if nominal is not None:
        ax.axhline(nominal, color="#9AA3AE", lw=0.72, ls=(0, (2.2, 2.0)), zorder=1)
        ax.text(
            0.99,
            nominal + 0.018 * (ylim[1] - ylim[0]),
            "nominal 0.68",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=4.9,
            color=SUBTLE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.35)
    ax.set_xlim(-0.22, len(regimes) - 0.78)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=6.1, labelpad=1.1)
    ax.set_title(title, loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax)


def select_representative_index(diagnostics: np.lib.npyio.NpzFile, depth: np.ndarray) -> int:
    keep = depth <= 100.0
    target = diagnostics["DI_Strong_in_prior_target"][:, 1, :][:, keep]
    samples = diagnostics["DI_Strong_in_prior_samples"][:, :, 1, :][:, :, keep]
    median = np.median(samples, axis=1)
    mae = np.mean(np.abs(median - target), axis=1)
    ranks = np.argsort(mae)
    return int(ranks[int(0.42 * len(ranks))])


def plot_dispersion_example(
    ax: plt.Axes,
    depth: np.ndarray,
    disp: np.ndarray,
    mask: np.ndarray,
    samples: np.ndarray,
) -> None:
    period = disp[0]
    observed = mask[1].astype(bool) & np.isfinite(disp[1])
    predictions = forward_rayleigh(depth, period, samples)[:, observed]
    p = period[observed]
    obs = disp[1, observed]
    q16, q50, q84 = quantiles(predictions, (0.16, 0.50, 0.84))
    ax.fill_between(p, q16, q84, color=POST_BLUE_FILL, linewidth=0, alpha=0.96, zorder=2)
    ax.plot(p, q50, color=POST_BLUE, lw=1.15, ls=(0, (3.0, 1.8)), zorder=4)
    keep = np.zeros(len(p), dtype=bool)
    keep[::2] = True
    keep[-1] = True
    ax.scatter(p[keep], obs[keep], s=8.8, color=TRUTH, edgecolor=PAPER, linewidth=0.25, zorder=6)
    ax.set_xlim(float(p.min()) - 1.5, float(p.max()) + 3.0)
    ax.set_ylim(min(float(q16.min()), float(obs.min())) - 0.12, max(float(q84.max()), float(obs.max())) + 0.10)
    ax.set_xlabel("Period (s)", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel(r"$c$ (km/s)", fontsize=6.4, labelpad=1.0)
    ax.set_title("Representative in-prior fit", loc="left", fontsize=6.8, color=INK, pad=2.0)
    style_axis(ax)
    handles = [
        Line2D([0], [0], color=POST_BLUE, lw=1.15, ls=(0, (3.0, 1.8)), label="posterior median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TRUTH, markeredgecolor=TRUTH, ms=3.2, label="observed"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.3, handlelength=1.7, borderaxespad=0.2)


def plot_vs_example(
    ax: plt.Axes,
    depth: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
) -> None:
    keep = depth <= 100.0
    z = depth[keep]
    vs_samples = samples[:, 1, :][:, keep]
    q16, q50, q84 = quantiles(vs_samples, (0.16, 0.50, 0.84))
    ax.fill_betweenx(z, q16, q84, color=POST_BLUE_FILL, linewidth=0, alpha=0.96, zorder=2)
    ax.plot(target[1, keep], z, color=TRUTH, lw=1.08, zorder=4)
    ax.plot(q50, z, color=POST_BLUE, lw=1.22, ls=(0, (3.0, 1.8)), zorder=5)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.75, 5.35)
    ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("Depth (km)", fontsize=6.4, labelpad=1.0)
    ax.set_title(r"$V_S$ posterior from the same case", loc="left", fontsize=6.8, color=INK, pad=2.0)
    style_axis(ax)
    handles = [
        Line2D([0], [0], color=TRUTH, lw=1.08, label="target"),
        Line2D([0], [0], color=POST_BLUE, lw=1.22, ls=(0, (3.0, 1.8)), label="posterior median"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.3, handlelength=1.7, borderaxespad=0.2)


def plot_grouped_bar_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ci_low: str,
    ci_high: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    nominal: float | None = None,
) -> None:
    regimes = ["in-prior", "boundary", "out-of-prior"]
    labels = ["Inside\nsupport", "Near\nedge", "Outside\nsupport"]
    x = np.arange(len(regimes), dtype=float)
    width = 0.32
    styles = {
        "DI-Strong": {"color": POST_BLUE, "offset": -0.18},
        "DI-Weak": {"color": WEAK_ORANGE, "offset": 0.18},
    }
    for method, style in styles.items():
        values = []
        low = []
        high = []
        for regime in regimes:
            row = metric_row(rows, method, regime)
            value = float(row[metric])
            values.append(value)
            low.append(value - float(row[ci_low]))
            high.append(float(row[ci_high]) - value)
        ax.bar(
            x + style["offset"],
            values,
            width=width,
            color=style["color"],
            alpha=0.86,
            edgecolor="none",
            zorder=3,
        )
        ax.errorbar(
            x + style["offset"],
            values,
            yerr=[low, high],
            fmt="none",
            ecolor=INK,
            elinewidth=0.58,
            capsize=1.5,
            capthick=0.58,
            zorder=5,
        )
    if nominal is not None:
        ax.axhline(nominal, color="#9AA3AE", lw=0.76, ls=(0, (2.2, 2.0)), zorder=1)
        ax.text(
            2.47,
            nominal + 0.022 * (ylim[1] - ylim[0]),
            "nominal 0.68",
            ha="right",
            va="bottom",
            fontsize=4.9,
            color=SUBTLE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=6.1, labelpad=1.1)
    ax.set_title(title, loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax, grid=False)
    ax.grid(axis="y", color=GRID, linewidth=0.50)


def median_model(samples: np.ndarray) -> np.ndarray:
    return np.median(samples, axis=0)


def safe_forward_rayleigh(depth: np.ndarray, period: np.ndarray, samples: np.ndarray) -> np.ndarray:
    predictions = []
    for sample in samples:
        try:
            pred = forward_rayleigh(depth, period, sample[None, :, :])[0]
        except Exception:
            pred = np.full_like(period, np.nan, dtype=float)
        predictions.append(pred)
    return np.stack(predictions, axis=0)


def plot_case_dispersion(
    ax: plt.Axes,
    diagnostics: np.lib.npyio.NpzFile,
    depth: np.ndarray,
    regime: str,
    index: int,
    *,
    title: str,
    ylabel: bool,
    show_observed_legend: bool = False,
) -> None:
    disp = diagnostics[f"DI_Strong_{regime}_disp"][index]
    mask = diagnostics[f"DI_Strong_{regime}_mask"][index]
    period = disp[0]
    ok = mask[1].astype(bool) & np.isfinite(disp[1])
    p = period[ok]
    obs = disp[1, ok]
    methods = [
        ("DI_Strong", POST_BLUE, POST_BLUE_FILL, (0, (3.0, 1.8))),
        ("DI_Weak", WEAK_ORANGE, WEAK_FILL, (0, (1.2, 1.35))),
    ]
    all_pred = []
    for method, color, fill, ls in methods:
        samples = diagnostics[f"{method}_{regime}_samples"][index]
        pred = safe_forward_rayleigh(depth, period, samples)[:, ok]
        q10 = np.nanquantile(pred, 0.05, axis=0)
        q50 = np.nanquantile(pred, 0.50, axis=0)
        q90 = np.nanquantile(pred, 0.95, axis=0)
        all_pred.extend([q10, q90])
        ax.fill_between(p, q10, q90, color=fill, linewidth=0, alpha=0.38, zorder=1)
        ax.plot(p, q50, color=color, lw=1.12, ls=ls, zorder=4)
    keep = np.zeros(len(p), dtype=bool)
    keep[::3] = True
    keep[-1] = True
    ax.scatter(p[keep], obs[keep], s=8.2, color=TRUTH, edgecolor=PAPER, linewidth=0.25, zorder=6)
    stacked = np.concatenate([obs, *all_pred])
    ax.set_xlim(float(p.min()) - 1.5, float(p.max()) + 3.0)
    ax.set_ylim(float(stacked.min()) - 0.16, float(stacked.max()) + 0.14)
    ax.set_xlabel("Period (s)", fontsize=6.1, labelpad=0.9)
    if ylabel:
        ax.set_ylabel(r"$c$ (km/s)", fontsize=6.1, labelpad=1.0)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax)
    if show_observed_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TRUTH,
                markeredgecolor=TRUTH,
                ms=3.1,
                label="observed",
            )
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=5.2, handlelength=1.0, borderaxespad=0.18)


def plot_case_vs(
    ax: plt.Axes,
    diagnostics: np.lib.npyio.NpzFile,
    depth: np.ndarray,
    regime: str,
    index: int,
    *,
    ylabel: bool,
    show_target_legend: bool = False,
) -> None:
    keep = depth <= 100.0
    z = depth[keep]
    target = diagnostics[f"DI_Strong_{regime}_target"][index, 1, keep]
    ax.plot(target, z, color=TRUTH, lw=1.15, zorder=6)
    methods = [
        ("DI_Strong", POST_BLUE, POST_BLUE_FILL, (0, (3.0, 1.8))),
        ("DI_Weak", WEAK_ORANGE, WEAK_FILL, (0, (1.2, 1.35))),
    ]
    for method, color, fill, ls in methods:
        vs = diagnostics[f"{method}_{regime}_samples"][index, :, 1, :][:, keep]
        q10, q50, q90 = quantiles(vs, (0.05, 0.50, 0.95))
        ax.fill_betweenx(z, q10, q90, color=fill, linewidth=0, alpha=0.42, zorder=1)
        ax.plot(q50, z, color=color, lw=1.18, ls=ls, zorder=5)
    ax.set_ylim(100, 0)
    ax.set_xlim(0.55, 5.35)
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.1, labelpad=0.9)
    if ylabel:
        ax.set_ylabel("Depth (km)", fontsize=6.1, labelpad=1.0)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(0.72)
    if show_target_legend:
        handles = [Line2D([0], [0], color=TRUTH, lw=1.15, label="target")]
        ax.legend(handles=handles, loc="lower left", fontsize=5.2, handlelength=1.55, borderaxespad=0.18)


def make_figure(args: argparse.Namespace) -> None:
    metrics = read_metrics(args.metrics)
    diagnostics = np.load(args.diagnostics)
    depth = np.linspace(0.0, 127.5, diagnostics["DI_Strong_in_prior_target"].shape[-1], dtype=float)
    case_indices = {
        "in_prior": args.in_prior_index,
        "boundary": args.boundary_index,
        "out_of_prior": args.out_of_prior_index,
    }

    fig = plt.figure(figsize=(7.25, 6.35), facecolor=PAPER)
    gs = fig.add_gridspec(
        3,
        6,
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.930,
        height_ratios=[0.82, 1.02, 1.45],
        hspace=0.45,
        wspace=0.46,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    plot_grouped_bar_panel(
        ax_a,
        metrics,
        metric="vs_mae",
        ci_low="vs_mae_ci_low",
        ci_high="vs_mae_ci_high",
        ylabel=r"$V_S$ error (km/s)",
        title="Model error",
        ylim=(0.0, 0.86),
    )
    plot_grouped_bar_panel(
        ax_b,
        metrics,
        metric="pred_disp_mae",
        ci_low="pred_disp_mae_ci_low",
        ci_high="pred_disp_mae_ci_high",
        ylabel=r"$c$ residual (km/s)",
        title="Forward fit",
        ylim=(0.0, 0.38),
    )
    plot_grouped_bar_panel(
        ax_c,
        metrics,
        metric="coverage_vs",
        ci_low="coverage_vs_ci_low",
        ci_high="coverage_vs_ci_high",
        ylabel=r"$V_S$ coverage",
        title="Interval check",
        ylim=(0.0, 0.90),
        nominal=0.68,
    )

    handles = [
        Line2D([0], [0], color=POST_BLUE, lw=1.28, ls=(0, (3.0, 1.8)), label="DI-Strong"),
        Line2D([0], [0], color=WEAK_ORANGE, lw=1.28, ls=(0, (1.2, 1.35)), label="DI-Weak"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.065, 0.688), ncol=2, fontsize=6.1, handlelength=1.8)
    fig.text(0.286, 0.706, "shaded bands: p5-p95", ha="left", va="center", fontsize=5.45, color=SUBTLE)

    cases = [
        ("in_prior", "Inside support"),
        ("boundary", "Near edge"),
        ("out_of_prior", "Outside support"),
    ]
    dispersion_axes = []
    profile_axes = []
    for j, (regime, title) in enumerate(cases):
        ax_disp = fig.add_subplot(gs[1, 2 * j : 2 * j + 2])
        ax_vs = fig.add_subplot(gs[2, 2 * j : 2 * j + 2])
        plot_case_dispersion(
            ax_disp,
            diagnostics,
            depth,
            regime,
            case_indices[regime],
            title=title,
            ylabel=j == 0,
            show_observed_legend=j == 0,
        )
        plot_case_vs(
            ax_vs,
            diagnostics,
            depth,
            regime,
            case_indices[regime],
            ylabel=j == 0,
            show_target_legend=j == 0,
        )
        dispersion_axes.append(ax_disp)
        profile_axes.append(ax_vs)

    for label, axis in zip("abcdefghi", [ax_a, ax_b, ax_c, *dispersion_axes, *profile_axes]):
        panel_label(axis, label)

    save_figure(fig, args.stem)
    diagnostics.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--stem", default="fig03_direct_inversion_results_v9")
    parser.add_argument("--example-index", type=int, default=186)
    parser.add_argument("--in-prior-index", type=int, default=227)
    parser.add_argument("--boundary-index", type=int, default=292)
    parser.add_argument("--out-of-prior-index", type=int, default=850)
    args = parser.parse_args()
    make_figure(args)


if __name__ == "__main__":
    main()
