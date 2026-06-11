#!/usr/bin/env python3
"""Generate a three-row Figure 1 draft using real prior, posterior and audit panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from make_fig01_asset_panels import (
    DEFAULT_DIAGNOSTICS_PATH,
    DEFAULT_METRICS_PATH,
    GRID,
    INK,
    MANUSCRIPT_ASSET_DIR,
    PAPER,
    PAPER_ASSET_DIR,
    POST_BLUE,
    POST_BLUE_FILL,
    SUPPORT_FILL,
    TRUTH,
    draw_sampler_block,
    plot_posterior_predictive_fit,
    plot_posterior_vs,
    plot_prior_predictive_dispersion,
    plot_prior_vs,
    quantiles,
    read_metrics,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "SurfFlow" / "manuscript" / "figures"
WEAK_LINE = "#C9823A"
ARROW = "#737C88"


def save_composite(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR, PAPER_ASSET_DIR, MANUSCRIPT_ASSET_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.png", dpi=440, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def panel_title(fig: plt.Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, f"({letter})", ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    fig.text(x + 0.041, y, title, ha="left", va="top", fontsize=7.6, fontweight="bold", color=INK)


def row_separator(fig: plt.Figure, y: float) -> None:
    fig.add_artist(
        plt.Line2D([0.015, 0.985], [y, y], transform=fig.transFigure, color="#C9D0D8", lw=0.55, alpha=0.82)
    )


def fig_arrow(
    fig: plt.Figure,
    ax0: plt.Axes,
    ax1: plt.Axes,
    *,
    shrink: float = 0.018,
    end_pad: float | None = None,
) -> None:
    b0 = ax0.get_position()
    b1 = ax1.get_position()
    y = 0.5 * (b0.y0 + b0.y1)
    gap = max(float(b1.x0 - b0.x1), 0.001)
    pad = min(shrink, max(0.006, 0.24 * gap))
    target_pad = pad if end_pad is None else end_pad
    fig.add_artist(
        FancyArrowPatch(
            (b0.x1 + pad, y),
            (b1.x0 - target_pad, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8.4,
            lw=0.78,
            color=ARROW,
            clip_on=False,
            zorder=50,
        )
    )


def solver_block(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.10, 0.33), 0.80, 0.36, facecolor="#F8FAFC", edgecolor=INK, lw=0.72, transform=ax.transAxes))
    ax.text(0.50, 0.565, "surface-wave", ha="center", va="center", fontsize=5.35, color=INK, transform=ax.transAxes)
    ax.text(0.50, 0.450, "modal solver", ha="center", va="center", fontsize=5.35, color=INK, transform=ax.transAxes)


def plot_conditioning_data(ax: plt.Axes, disp: np.ndarray, mask: np.ndarray) -> None:
    period = disp[0]
    ray = disp[1]
    ok = mask[1].astype(bool) & np.isfinite(ray)
    missing = ~ok & np.isfinite(ray)
    ax.plot(period[np.isfinite(ray)], ray[np.isfinite(ray)], color="#B8C4D2", lw=0.55, alpha=0.58, zorder=1)
    ax.scatter(period[ok], ray[ok], s=7.2, color=TRUTH, linewidth=0, zorder=4)
    if missing.any():
        ax.scatter(
            period[missing],
            ray[missing],
            s=7.2,
            facecolor=PAPER,
            edgecolor="#8F98A4",
            linewidth=0.50,
            alpha=0.72,
            zorder=3,
        )
    valid = ok | missing
    ax.set_xlim(float(period[valid].min()) - 1.5, float(period[valid].max()) + 3.0)
    ax.set_ylim(float(ray[valid].min()) - 0.18, float(ray[valid].max()) + 0.18)
    ax.set_xlabel("Period (s)", fontsize=6.1, labelpad=1.0)
    ax.set_ylabel(r"$c$ (km/s)", fontsize=6.1, labelpad=0.8)
    ax.set_title(r"Observed dispersion $d_{\rm obs}$", loc="left", fontsize=6.4, color=INK, pad=2.0)
    style_axis(ax)


def support_profile(
    ax: plt.Axes,
    depth: np.ndarray,
    p05: np.ndarray,
    p95: np.ndarray,
    target: np.ndarray,
    strong_samples: np.ndarray,
    weak_samples: np.ndarray,
    title: str,
    *,
    ylabel: bool,
    legend: bool = False,
) -> None:
    q05, q50, q95 = quantiles(strong_samples, (0.05, 0.50, 0.95))
    weak_q50 = quantiles(weak_samples, (0.50,))[0]
    ax.fill_betweenx(depth, p05, p95, color=SUPPORT_FILL, lw=0, zorder=1)
    ax.fill_betweenx(depth, q05, q95, color=POST_BLUE_FILL, lw=0, alpha=0.88, zorder=2)
    target_line, = ax.plot(target, depth, color=TRUTH, lw=1.02, zorder=4, label="target")
    strong_line, = ax.plot(q50, depth, color=POST_BLUE, lw=1.12, ls=(0, (3.0, 1.8)), zorder=5, label="DI-Strong")
    weak_line, = ax.plot(weak_q50, depth, color=WEAK_LINE, lw=0.98, ls=(0, (1.2, 1.35)), zorder=6, label="DI-Weak")
    ax.set_ylim(100, 0)
    ax.set_xlim(0.55, 5.35)
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.0, labelpad=1.0)
    ax.set_title(title, loc="left", fontsize=6.25, color=INK, pad=2.0)
    if ylabel:
        ax.set_ylabel("Depth (km)", fontsize=6.0, labelpad=1.0)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    style_axis(ax)
    if legend:
        ax.legend(
            handles=[target_line, strong_line, weak_line],
            loc="lower left",
            bbox_to_anchor=(0.025, 0.02),
            fontsize=5.25,
            handlelength=1.65,
            borderaxespad=0.0,
        )


def plot_audit_summary(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    order = ["in-prior", "boundary", "out-of-prior"]
    labels = ["in\nprior", "near\nboundary", "out of\nprior"]
    x = np.arange(len(order), dtype=float)
    styles = {
        "DI-Strong": {"color": POST_BLUE, "ls": (0, (3.0, 1.8)), "marker": "o"},
        "DI-Weak": {"color": WEAK_LINE, "ls": (0, (1.2, 1.35)), "marker": "s"},
    }
    for method, style in styles.items():
        values = []
        low = []
        high = []
        for test_set in order:
            row = next(r for r in rows if r["method"] == method and r["test_set"] == test_set)
            value = float(row["vs_mae"])
            values.append(value)
            low.append(value - float(row["vs_mae_ci_low"]))
            high.append(float(row["vs_mae_ci_high"]) - value)
        ax.errorbar(
            x,
            values,
            yerr=[low, high],
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            ms=2.4,
            lw=0.95,
            capsize=1.4,
            elinewidth=0.55,
            label=method,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.1)
    ax.set_ylabel(r"$V_S$ error (km/s)", fontsize=6.0, labelpad=1.0)
    ax.set_title("Prior test sets", loc="left", fontsize=6.25, color=INK, pad=2.0)
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(0.0, 0.86)
    style_axis(ax)
    ax.text(
        0.98,
        0.08,
        "lower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#6E7682",
    )
    ax.legend(loc="upper left", fontsize=5.25, handlelength=1.65, borderaxespad=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--stem", default="fig01_three_row_main_draft")
    parser.add_argument("--training-index", type=int, default=941)
    parser.add_argument("--posterior-index", type=int, default=186)
    parser.add_argument("--audit-index", type=int, default=703)
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    metrics = read_metrics(args.metrics)
    depth = np.linspace(0.0, 127.5, diagnostics["DI_Strong_in_prior_target"].shape[-1], dtype=float)
    prior_vs = diagnostics["DI_Strong_in_prior_target"][:, 1]
    key = "DI_Strong_in_prior"
    prior_pair_indices = np.array([], dtype=int)

    fig = plt.figure(figsize=(7.25, 6.25), facecolor=PAPER)
    panel_title(fig, 0.018, 0.972, "a", r"Data: synthetic training pairs $(m,d)$")
    panel_title(fig, 0.018, 0.656, "b", r"Model: posterior sampling from dispersion data")
    panel_title(fig, 0.018, 0.348, "c", r"Reliability checks: fit, error and prior support")
    row_separator(fig, 0.684)
    row_separator(fig, 0.378)

    ax_a_prior = fig.add_axes([0.055, 0.745, 0.160, 0.175])
    ax_a_solver = fig.add_axes([0.265, 0.770, 0.085, 0.125])
    ax_a_disp = fig.add_axes([0.425, 0.745, 0.220, 0.175])
    ax_a_note = fig.add_axes([0.715, 0.760, 0.225, 0.130])
    plot_prior_vs(
        ax_a_prior,
        depth,
        prior_vs,
        example=diagnostics["DI_Strong_in_prior_target"][args.training_index, 1],
        ensemble_indices=prior_pair_indices,
    )
    ax_a_prior.set_title(r"Synthetic prior $m$", loc="left", fontsize=6.4, color=INK, pad=2.0)
    solver_block(ax_a_solver)
    plot_prior_predictive_dispersion(
        ax_a_disp,
        diagnostics["DI_Strong_in_prior_disp"],
        diagnostics["DI_Strong_in_prior_mask"],
        args.training_index,
        prior_pair_indices,
    )
    ax_a_disp.set_ylabel(r"$c$ (km/s)", fontsize=6.0, labelpad=0.8)
    ax_a_disp.set_title(r"Synthetic Rayleigh dispersion $d$", loc="left", fontsize=6.4, color=INK, pad=2.0)
    ax_a_note.set_axis_off()
    ax_a_note.add_patch(Rectangle((0.08, 0.20), 0.84, 0.58, facecolor="#FFFFFF", edgecolor="#A9B2BE", lw=0.62, transform=ax_a_note.transAxes))
    ax_a_note.text(
        0.50,
        0.50,
        "Training pairs define the\nsynthetic inversion problem;\nbands show p5-p95 quantiles.",
        ha="center",
        va="center",
        fontsize=5.85,
        color=INK,
        transform=ax_a_note.transAxes,
    )
    fig_arrow(fig, ax_a_prior, ax_a_solver, shrink=0.006)
    fig_arrow(fig, ax_a_solver, ax_a_disp, shrink=0.006, end_pad=0.034)

    ax_b_data = fig.add_axes([0.055, 0.440, 0.165, 0.170])
    ax_b_flow = fig.add_axes([0.295, 0.455, 0.180, 0.140])
    ax_b_post = fig.add_axes([0.550, 0.425, 0.145, 0.205])
    ax_b_fit = fig.add_axes([0.805, 0.440, 0.165, 0.170])
    plot_conditioning_data(ax_b_data, diagnostics[f"{key}_disp"][args.posterior_index], diagnostics[f"{key}_mask"][args.posterior_index])
    draw_sampler_block(ax_b_flow)
    plot_posterior_predictive_fit(
        ax_b_fit,
        depth,
        diagnostics[f"{key}_disp"][args.posterior_index],
        diagnostics[f"{key}_mask"][args.posterior_index],
        diagnostics[f"{key}_samples"][args.posterior_index],
        title=r"Forward dispersion check $G(m)$",
        ylabel=False,
    )
    plot_posterior_vs(
        ax_b_post,
        depth,
        diagnostics[f"{key}_target"][args.posterior_index],
        diagnostics[f"{key}_samples"][args.posterior_index],
        title=r"Posterior $V_S$",
    )
    fig_arrow(fig, ax_b_data, ax_b_flow)
    fig_arrow(fig, ax_b_flow, ax_b_post)
    fig_arrow(fig, ax_b_post, ax_b_fit)

    ax_c_audit = fig.add_axes([0.055, 0.080, 0.175, 0.230])
    ax_c_fit = fig.add_axes([0.280, 0.080, 0.205, 0.230])
    ax_c_inside = fig.add_axes([0.565, 0.080, 0.160, 0.230])
    ax_c_mismatch = fig.add_axes([0.805, 0.080, 0.160, 0.230])
    plot_audit_summary(ax_c_audit, metrics)
    plot_posterior_predictive_fit(
        ax_c_fit,
        depth,
        diagnostics[f"{key}_disp"][args.posterior_index],
        diagnostics[f"{key}_mask"][args.posterior_index],
        diagnostics[f"{key}_samples"][args.posterior_index],
        title="Dispersion fit",
    )
    p05, p95 = quantiles(prior_vs, (0.05, 0.95))
    support_profile(
        ax_c_inside,
        depth,
        p05,
        p95,
        diagnostics["DI_Strong_in_prior_target"][args.posterior_index, 1],
        diagnostics["DI_Strong_in_prior_samples"][args.posterior_index, :, 1],
        diagnostics["DI_Weak_in_prior_samples"][args.posterior_index, :, 1],
        "In prior",
        ylabel=True,
        legend=False,
    )
    support_profile(
        ax_c_mismatch,
        depth,
        p05,
        p95,
        diagnostics["DI_Strong_boundary_target"][args.audit_index, 1],
        diagnostics["DI_Strong_boundary_samples"][args.audit_index, :, 1],
        diagnostics["DI_Weak_boundary_samples"][args.audit_index, :, 1],
        "Near boundary",
        ylabel=False,
        legend=True,
    )

    save_composite(fig, args.stem)
    diagnostics.close()


if __name__ == "__main__":
    main()
