#!/usr/bin/env python3
"""Draw the unified Figure 1: prior-predictive posterior surrogate and audit."""

from __future__ import annotations

import argparse
import csv
import sys
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
DEFAULT_METRICS_PATH = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "fair_di_metrics.csv"
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "SurfFlow" / "manuscript" / "figures"

SURFFLOW_ROOT = ROOT / "SurfFlow"
if str(SURFFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(SURFFLOW_ROOT))

from utils.generate_data import compute_phase_dispersion


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


INK = "#20242B"
SUBTLE = "#697281"
GRID = "#E6E9ED"
PRIOR = "#3F6FA7"
POST = "#D07C2C"
TRUTH = "#111111"
SHADE = "#EEF1F4"


def finish(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(out_dir / f"{stem}.png", dpi=360, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.tick_params(direction="out", length=2.8, width=0.7, colors=INK, pad=2.0)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.56)


def quantiles(values: np.ndarray, qs: tuple[float, ...]) -> list[np.ndarray]:
    return [np.quantile(values, q, axis=0) for q in qs]


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(rows: list[dict[str, str]], method: str, regime: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and row["test_set"] == regime)


def forward_rayleigh(depth: np.ndarray, periods: np.ndarray, samples: np.ndarray) -> np.ndarray:
    predictions = []
    depth64 = np.asarray(depth, dtype=np.float64)
    periods64 = np.asarray(periods, dtype=np.float64)
    for sample in np.asarray(samples, dtype=np.float64):
        result = compute_phase_dispersion(
            depth64,
            sample[0],
            sample[1],
            sample[2],
            periods=periods64,
            modes=(0,),
            wave="rayleigh",
        )[0]
        predictions.append(np.asarray(result.velocity, dtype=np.float64))
    return np.stack(predictions, axis=0)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.052,
        box.y1 + 0.012,
        label,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )


def draw_prior_support(ax: plt.Axes, depth: np.ndarray, prior_vs: np.ndarray) -> None:
    q05, q50, q95 = quantiles(prior_vs, (0.05, 0.50, 0.95))
    ax.axhspan(75, 100, color=SHADE, zorder=0)
    for curve in prior_vs[::64][:16]:
        ax.plot(curve, depth, color=PRIOR, lw=0.30, alpha=0.055, zorder=1)
    ax.fill_betweenx(depth, q05, q95, color=PRIOR, alpha=0.19, linewidth=0, zorder=2)
    ax.plot(q50, depth, color=PRIOR, lw=1.45, zorder=4)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.55, 5.7)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Structural prior support", loc="left", fontsize=7.4, color=INK, pad=2.0)
    style_axis(ax)


def draw_prior_predictive(ax: plt.Axes, prior_disp: np.ndarray) -> None:
    periods = prior_disp[0, 0]
    ray = prior_disp[:, 1]
    q10, q50, q90 = quantiles(ray, (0.10, 0.50, 0.90))
    for curve in ray[::64][:16]:
        ax.plot(periods, curve, color=PRIOR, lw=0.30, alpha=0.052, zorder=1)
    ax.fill_between(periods, q10, q90, color=PRIOR, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(periods, q50, color=PRIOR, lw=1.45, zorder=4)
    ax.set_xlim(2, 60)
    ax.set_ylim(0.65, 4.65)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_title("Prior-predictive Rayleigh data", loc="left", fontsize=7.4, color=INK, pad=2.0)
    style_axis(ax)


def draw_posterior_dispersion(
    ax: plt.Axes,
    depth: np.ndarray,
    dispersion: np.ndarray,
    mask: np.ndarray,
    samples: np.ndarray,
) -> None:
    periods = dispersion[0]
    keep = mask[1].astype(bool)
    pred = forward_rayleigh(depth, periods, samples)[:, keep]
    p = periods[keep]
    obs = dispersion[1, keep]
    q10, q50, q90 = quantiles(pred, (0.10, 0.50, 0.90))
    for curve in pred[::2]:
        ax.plot(p, curve, color=POST, lw=0.30, alpha=0.042, zorder=1)
    ax.fill_between(p, q10, q90, color=POST, alpha=0.23, linewidth=0, zorder=2)
    ax.plot(p, q50, color=POST, lw=1.45, zorder=4)
    ax.scatter(p, obs, s=6.2, facecolor="white", edgecolor=TRUTH, linewidth=0.55, zorder=5)
    ax.set_xlim(max(0, float(p.min()) - 1.5), float(p.max()) + 3.0)
    ax.set_ylim(min(float(q10.min()), float(obs.min())) - 0.12, max(float(q90.max()), float(obs.max())) + 0.10)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_title("Rayleigh fit", loc="left", fontsize=7.4, color=INK, pad=2.0)
    style_axis(ax)


def draw_posterior_vs(ax: plt.Axes, depth: np.ndarray, target: np.ndarray, samples: np.ndarray) -> None:
    vs_samples = samples[:, 1]
    q10, q50, q90 = quantiles(vs_samples, (0.10, 0.50, 0.90))
    ax.axhspan(75, 100, color=SHADE, zorder=0)
    for curve in vs_samples[::2]:
        ax.plot(curve, depth, color=POST, lw=0.30, alpha=0.05, zorder=1)
    ax.fill_betweenx(depth, q10, q90, color=POST, alpha=0.23, linewidth=0, zorder=2)
    ax.plot(target[1], depth, color=TRUTH, lw=1.20, zorder=4)
    ax.plot(q50, depth, color=POST, lw=1.60, zorder=5)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.8, 5.35)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(r"$V_S$ posterior", loc="left", fontsize=7.4, color=INK, pad=2.0)
    style_axis(ax)


def draw_prior_support_example(
    ax: plt.Axes,
    depth: np.ndarray,
    prior_vs: np.ndarray,
    target_vs: np.ndarray,
    posterior_vs: np.ndarray,
) -> None:
    prior_05, prior_50, prior_95 = quantiles(prior_vs, (0.05, 0.50, 0.95))
    post_10, post_50, post_90 = quantiles(posterior_vs, (0.10, 0.50, 0.90))
    ax.axhspan(75, 100, color=SHADE, zorder=0)
    ax.fill_betweenx(depth, prior_05, prior_95, color=PRIOR, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(prior_50, depth, color=PRIOR, lw=1.10, zorder=3)
    ax.fill_betweenx(depth, post_10, post_90, color=POST, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(target_vs, depth, color=TRUTH, lw=1.20, zorder=5)
    ax.plot(post_50, depth, color=POST, lw=1.55, zorder=6)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.55, 5.7)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Prior-support example", loc="left", fontsize=7.4, color=INK, pad=2.0)
    ax.text(0.62, 0.82, "prior", transform=ax.transAxes, color=PRIOR, fontsize=5.8, ha="left")
    ax.text(0.62, 0.74, "posterior", transform=ax.transAxes, color=POST, fontsize=5.8, ha="left")
    ax.text(0.62, 0.66, "target", transform=ax.transAxes, color=TRUTH, fontsize=5.8, ha="left")
    style_axis(ax)


def draw_pullin_audit(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    regimes = ["boundary", "out-of-prior"]
    y_base = np.arange(len(regimes))[::-1]
    offsets = {"DI-Strong": 0.09, "DI-Weak": -0.09}
    colors = {"DI-Strong": PRIOR, "DI-Weak": POST}
    for method in ["DI-Strong", "DI-Weak"]:
        xs, xlo, xhi, ys = [], [], [], []
        for regime, y in zip(regimes, y_base):
            row = metric_row(rows, method, regime)
            value = float(row["pred_inside_given_target_outside"])
            low = float(row["pred_inside_given_target_outside_ci_low"])
            high = float(row["pred_inside_given_target_outside_ci_high"])
            xs.append(value)
            xlo.append(value - low)
            xhi.append(high - value)
            ys.append(y + offsets[method])
        ax.errorbar(
            xs,
            ys,
            xerr=[xlo, xhi],
            fmt="o",
            color=colors[method],
            ecolor=colors[method],
            elinewidth=1.05,
            capsize=2.0,
            markersize=3.8,
            zorder=4,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(["Boundary", "Out-of-prior"])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.45, len(regimes) - 0.55)
    ax.set_xlabel("Pull-in fraction")
    ax.set_title("Prior pull-in audit", loc="left", fontsize=7.4, color=INK, pad=2.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=0.56)
    ax.grid(axis="y", color="#F1F3F5", linewidth=0.42)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3.0, colors=INK)
    ax.tick_params(axis="x", direction="out", length=2.8, width=0.7, colors=INK, pad=2.0)
    ax.text(0.78, 0.83, "DI-Strong", color=PRIOR, transform=ax.transAxes, fontsize=6.0, ha="left")
    ax.text(0.78, 0.74, "DI-Weak", color=POST, transform=ax.transAxes, fontsize=6.0, ha="left")


def add_forward_arrow(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    left = ax_left.get_position()
    right = ax_right.get_position()
    y = 0.5 * (left.y0 + left.y1)
    x0 = left.x1 + 0.014
    x1 = right.x0 - 0.014
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            lw=0.85,
            color=SUBTLE,
            mutation_scale=9.5,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--posterior-index", type=int, default=743)
    parser.add_argument("--audit-index", type=int, default=491)
    parser.add_argument("--stem", default="fig01_prior_posterior_audit_draft")
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    metrics = read_metrics(args.metrics)
    depth = np.linspace(0.0, 127.5, diagnostics["DI_Strong_in_prior_target"].shape[-1], dtype=float)

    prior_target = np.asarray(diagnostics["DI_Strong_in_prior_target"], dtype=float)
    prior_disp = np.asarray(diagnostics["DI_Strong_in_prior_disp"], dtype=float)

    post_key = "DI_Strong_in_prior"
    post_i = args.posterior_index
    post_target = np.asarray(diagnostics[f"{post_key}_target"][post_i], dtype=float)
    post_samples = np.asarray(diagnostics[f"{post_key}_samples"][post_i], dtype=float)
    post_disp = np.asarray(diagnostics[f"{post_key}_disp"][post_i], dtype=float)
    post_mask = np.asarray(diagnostics[f"{post_key}_mask"][post_i], dtype=float)

    audit_key = "DI_Strong_boundary"
    audit_i = args.audit_index
    audit_target_vs = np.asarray(diagnostics[f"{audit_key}_target"][audit_i, 1], dtype=float)
    audit_posterior_vs = np.asarray(diagnostics[f"{audit_key}_samples"][audit_i, :, 1], dtype=float)

    fig = plt.figure(figsize=(7.35, 6.45))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.86, 1.00, 0.93],
        left=0.075,
        right=0.985,
        bottom=0.060,
        top=0.965,
        hspace=0.58,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.94, 1.12], wspace=0.32)
    middle = outer[1].subgridspec(1, 2, width_ratios=[1.14, 0.86], wspace=0.28)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.34)

    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1])
    ax_b1 = fig.add_subplot(middle[0, 0])
    ax_b2 = fig.add_subplot(middle[0, 1])
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1])

    draw_prior_support(ax_a1, depth, prior_target[:, 1])
    draw_prior_predictive(ax_a2, prior_disp)
    draw_posterior_dispersion(ax_b1, depth, post_disp, post_mask, post_samples)
    draw_posterior_vs(ax_b2, depth, post_target, post_samples)
    draw_prior_support_example(ax_c1, depth, prior_target[:, 1], audit_target_vs, audit_posterior_vs)
    draw_pullin_audit(ax_c2, metrics)

    panel_label(fig, ax_a1, "A")
    panel_label(fig, ax_b1, "B")
    panel_label(fig, ax_c1, "C")
    add_forward_arrow(fig, ax_a1, ax_a2)

    finish(fig, args.stem)
    diagnostics.close()


if __name__ == "__main__":
    main()
