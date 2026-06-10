#!/usr/bin/env python3
"""Generate separate publication-style assets for the redesigned Figure 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from make_fig01_prior_posterior_audit import forward_rayleigh


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_PATH = Path(
    "/Volumes/lx_exFAT/yzy_directSWI/code_data/results/fair_di_comparison/production/fair_di_diagnostics.npz"
)
DEFAULT_METRICS_PATH = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "fair_di_metrics.csv"
DEFAULT_FIELD_VOLUME_PATH = Path(
    "/Volumes/lx_exFAT/yzy_directSWI/code_data/field_masw_results_fair_weak/bayan_obo_masw_dnn_posterior_volume.npz"
)
DEFAULT_FIELD_SUMMARY_PATH = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "field" / "field_summary.csv"
PAPER_ASSET_DIR = ROOT / "paper-overleaf" / "figures" / "fig01_assets"
MANUSCRIPT_ASSET_DIR = ROOT / "SurfFlow" / "manuscript" / "figures" / "fig01_assets"


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


INK = "#20242B"
SUBTLE = "#6E7682"
GRID = "#E8EBEF"
AXIS = "#A8B0BA"
PRIOR = "#2F6F9F"
PRIOR_FILL = "#DCE9F4"
SUPPORT_FILL = "#E9EEF3"
EXAMPLE = "#171A1F"
POST = "#CC7A29"
POST_FILL = "#F3DDC7"
POST_BLUE = "#2F6F9F"
POST_BLUE_FILL = "#DCE9F4"
OBS = "#2E6FBB"
LOVE = "#4E9A8A"
TRUTH = "#111111"
WEAK = "#8A6F9B"
STRONG = "#3B7F87"
WARN = "#B2473E"
PAPER = "#FFFFFF"


def save_asset(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_ASSET_DIR, MANUSCRIPT_ASSET_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.png", dpi=440, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.tick_params(direction="out", length=2.4, width=0.65, colors=INK, pad=1.6, labelsize=6.2)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid:
        ax.grid(color=GRID, linewidth=0.5)


def tiny_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.plot([0.08, 0.08], [0.10, 0.90], transform=ax.transAxes, color=AXIS, lw=0.62, clip_on=False)
    ax.plot([0.08, 0.92], [0.10, 0.10], transform=ax.transAxes, color=AXIS, lw=0.62, clip_on=False)


def quantiles(values: np.ndarray, qs: tuple[float, ...]) -> list[np.ndarray]:
    return [np.quantile(values, q, axis=0) for q in qs]


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(rows: list[dict[str, str]], method: str, regime: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and row["test_set"] == regime)


def arrow_between(fig: plt.Figure, ax0: plt.Axes, ax1: plt.Axes, *, label: str | None = None) -> None:
    b0 = ax0.get_position()
    b1 = ax1.get_position()
    y = 0.5 * (b0.y0 + b0.y1)
    x0 = b0.x1 + 0.010
    x1 = b1.x0 - 0.010
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9.2,
            lw=0.82,
            color=SUBTLE,
        )
    )
    if label:
        fig.text(0.5 * (x0 + x1), y + 0.030, label, ha="center", va="bottom", fontsize=6.0, color=SUBTLE)


def panel_title(fig: plt.Figure, letter: str, title: str) -> None:
    fig.text(0.012, 0.965, letter, ha="left", va="top", fontsize=7.8, fontweight="bold", color=INK)
    fig.text(0.055, 0.965, title, ha="left", va="top", fontsize=7.4, fontweight="bold", color=INK)


def plot_prior_vs(
    ax: plt.Axes,
    depth: np.ndarray,
    vs: np.ndarray,
    *,
    example: np.ndarray | None = None,
    ensemble_indices: np.ndarray | None = None,
    xlabel: bool = True,
) -> None:
    q10, q90 = quantiles(vs, (0.10, 0.90))
    ax.fill_betweenx(depth, q10, q90, color=PRIOR_FILL, linewidth=0, zorder=2, label="Prior ensemble p10-p90")
    if ensemble_indices is not None:
        for idx in ensemble_indices[:8]:
            ax.plot(vs[int(idx)], depth, color=PRIOR, lw=0.38, alpha=0.18, zorder=3)
    if example is not None:
        ax.plot(example, depth, color=EXAMPLE, lw=1.48, zorder=4, label="Highlighted pair")
    ax.set_ylim(100, 0)
    ax.set_xlim(1.55, 5.7)
    if xlabel:
        ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.4)
    ax.set_ylabel("Depth (km)", fontsize=6.4)
    style_axis(ax)


def plot_masked_dispersion(
    ax: plt.Axes,
    disp: np.ndarray,
    mask: np.ndarray,
    *,
    show_missing: bool = True,
    ylabel: bool = True,
) -> None:
    period = disp[0]
    ray_ok = mask[1].astype(bool)
    love_ok = mask[2].astype(bool)
    ax.scatter(period[ray_ok], disp[1, ray_ok], s=8.5, facecolor=PAPER, edgecolor=OBS, linewidth=0.65, label="Rayleigh", zorder=4)
    if love_ok.any():
        ax.scatter(period[love_ok], disp[2, love_ok], s=8.5, facecolor=PAPER, edgecolor=LOVE, linewidth=0.65, label="Love", zorder=4)
    if show_missing:
        miss = ~(ray_ok | love_ok)
        if miss.any():
            ymin, ymax = ax.get_ylim()
            ax.scatter(period[miss], np.full(miss.sum(), ymin + 0.08 * (ymax - ymin)), marker="x", s=8, color="#9AA3AE", linewidth=0.55, zorder=3)
    valid = ray_ok | love_ok
    ax.set_xlim(float(period[valid].min()) - 2, float(period[valid].max()) + 3)
    vals = []
    if ray_ok.any():
        vals.append(disp[1, ray_ok])
    if love_ok.any():
        vals.append(disp[2, love_ok])
    v = np.concatenate(vals)
    ax.set_ylim(float(v.min()) - 0.18, float(v.max()) + 0.18)
    ax.set_xlabel("Period (s)", fontsize=6.4)
    if ylabel:
        ax.set_ylabel(r"$c$ (km/s)", fontsize=6.4)
    style_axis(ax)


def finite_quantiles(values: np.ndarray, qs: tuple[float, ...]) -> list[np.ndarray]:
    out = []
    for q in qs:
        out.append(np.nanquantile(values, q, axis=0))
    return out


def plot_prior_predictive_dispersion(
    ax: plt.Axes,
    disp_all: np.ndarray,
    mask_all: np.ndarray,
    example_index: int,
    ensemble_indices: np.ndarray,
) -> None:
    period = disp_all[0, 0]
    ray = np.where(np.isfinite(disp_all[:, 1, :]), disp_all[:, 1, :], np.nan)
    ray_ok = np.isfinite(ray).sum(axis=0) >= 32
    q10, q90 = finite_quantiles(ray[:, ray_ok], (0.10, 0.90))
    ax.fill_between(period[ray_ok], q10, q90, color=PRIOR_FILL, lw=0, alpha=0.88, zorder=2)

    for idx in ensemble_indices[:8]:
        curve = disp_all[int(idx), 1]
        curve_ok = np.isfinite(curve) & ray_ok
        if curve_ok.any():
            ax.plot(period[curve_ok], curve[curve_ok], color=PRIOR, lw=0.36, alpha=0.16, zorder=3)

    example = disp_all[example_index]
    example_ok = np.isfinite(example[1]) & ray_ok
    ax.plot(period[example_ok], example[1, example_ok], color=EXAMPLE, lw=1.18, zorder=4)

    values = ray[:, ray_ok].ravel()
    values = values[np.isfinite(values)]
    ax.set_xlim(float(period[ray_ok].min()) - 1.5, float(period[ray_ok].max()) + 2.0)
    ax.set_ylim(float(np.nanpercentile(values, 1)) - 0.10, float(np.nanpercentile(values, 99)) + 0.12)
    ax.set_xlabel("Period (s)", fontsize=6.4)
    ax.set_ylabel("")
    style_axis(ax)


def plot_posterior_predictive_fit(
    ax: plt.Axes,
    depth: np.ndarray,
    disp: np.ndarray,
    mask: np.ndarray,
    samples: np.ndarray,
    *,
    title: str = "Rayleigh data and fit",
    ylabel: bool = True,
) -> None:
    period = disp[0]
    ok = mask[1].astype(bool)
    pred = forward_rayleigh(depth, period, samples)[:, ok]
    p = period[ok]
    obs = disp[1, ok]
    q16, q50, q84 = quantiles(pred, (0.16, 0.50, 0.84))
    ax.fill_between(p, q16, q84, color=POST_BLUE_FILL, linewidth=0, alpha=0.92, zorder=2)
    ax.plot(p, q16, color=POST_BLUE, lw=0.34, alpha=0.34, zorder=3)
    ax.plot(p, q84, color=POST_BLUE, lw=0.34, alpha=0.34, zorder=3)
    ax.plot(p, q50, color=POST_BLUE, lw=1.15, zorder=4)
    keep = np.zeros(len(p), dtype=bool)
    keep[::2] = True
    keep[-1] = True
    ax.scatter(p[keep], obs[keep], s=7.4, facecolor=EXAMPLE, edgecolor=PAPER, linewidth=0.34, zorder=6)
    ax.set_xlim(float(p.min()) - 1.5, float(p.max()) + 3.0)
    ax.set_ylim(min(float(q16.min()), float(obs.min())) - 0.12, max(float(q84.max()), float(obs.max())) + 0.10)
    ax.set_xlabel("Period (s)", fontsize=6.4)
    if ylabel:
        ax.set_ylabel(r"$c$ (km/s)", fontsize=6.4)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", fontsize=6.7, color=INK, pad=2.0)
    style_axis(ax)


def make_asset_a(diagnostics: np.lib.npyio.NpzFile, depth: np.ndarray, index: int) -> None:
    prior_vs = diagnostics["DI_Strong_in_prior_target"][:, 1]
    example_vs = diagnostics["DI_Strong_in_prior_target"][index, 1]
    disp_all = diagnostics["DI_Strong_in_prior_disp"]
    mask_all = diagnostics["DI_Strong_in_prior_mask"]
    ensemble_indices = np.array([22, 154, 258, 395, 526, 825, 930, 941, 993], dtype=int)
    ensemble_indices = ensemble_indices[ensemble_indices != index]

    fig = plt.figure(figsize=(5.65, 1.80), facecolor=PAPER)
    panel_title(fig, "a", "Prior-predictive training generator")
    gs = fig.add_gridspec(1, 3, left=0.070, right=0.985, bottom=0.215, top=0.750, width_ratios=[0.95, 0.48, 1.15], wspace=0.42)
    ax_prior = fig.add_subplot(gs[0, 0])
    ax_solver = fig.add_subplot(gs[0, 1])
    ax_disp = fig.add_subplot(gs[0, 2])

    plot_prior_vs(ax_prior, depth, prior_vs, example=example_vs, ensemble_indices=ensemble_indices)
    ax_prior.set_title("Synthetic model prior", loc="left", fontsize=6.7, color=INK, pad=2.0)

    ax_solver.set_axis_off()
    ax_solver.add_patch(Rectangle((0.12, 0.37), 0.76, 0.30, facecolor="#F8FAFC", edgecolor=INK, lw=0.75, transform=ax_solver.transAxes))
    ax_solver.text(0.50, 0.550, "surface-wave", ha="center", va="center", fontsize=6.0, color=INK, transform=ax_solver.transAxes)
    ax_solver.text(0.50, 0.450, "forward model", ha="center", va="center", fontsize=6.0, color=INK, transform=ax_solver.transAxes)

    plot_prior_predictive_dispersion(ax_disp, disp_all, mask_all, index, ensemble_indices)
    ax_disp.set_title("Rayleigh prior predictions", loc="left", fontsize=6.7, color=INK, pad=2.0)

    arrow_between(fig, ax_prior, ax_solver)
    arrow_between(fig, ax_solver, ax_disp)
    save_asset(fig, "fig01_asset_A_prior_pairs")


def draw_sampler_block(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.13, 0.32), 0.74, 0.36, facecolor="#F8FAFC", edgecolor=INK, lw=0.82, transform=ax.transAxes))
    ax.text(0.50, 0.565, "posterior", ha="center", va="center", fontsize=6.5, color=INK, transform=ax.transAxes)
    ax.text(0.50, 0.455, "sampler", ha="center", va="center", fontsize=6.5, color=INK, transform=ax.transAxes)
    ax.text(0.50, 0.235, r"$q_\theta(m\mid d)$", ha="center", va="top", fontsize=5.9, color=SUBTLE, transform=ax.transAxes)


def plot_posterior_vs(ax: plt.Axes, depth: np.ndarray, target: np.ndarray, samples: np.ndarray, *, title: str) -> None:
    vs = samples[:, 1]
    q16, q50, q84 = quantiles(vs, (0.16, 0.50, 0.84))
    for curve in vs[::4]:
        ax.plot(curve, depth, color=POST_BLUE, lw=0.24, alpha=0.030, zorder=1)
    ax.fill_betweenx(depth, q16, q84, color=POST_BLUE_FILL, linewidth=0, alpha=0.90, zorder=2)
    ax.plot(target[1], depth, color=TRUTH, lw=1.08, zorder=4)
    ax.plot(q50, depth, color=POST_BLUE, lw=1.36, ls=(0, (3.0, 1.8)), zorder=5)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.80, 5.35)
    ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.4)
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=6.7, color=INK, pad=2.0)
    style_axis(ax)


def make_asset_b(diagnostics: np.lib.npyio.NpzFile, depth: np.ndarray, index: int) -> None:
    key = "DI_Strong_in_prior"
    disp = diagnostics[f"{key}_disp"][index]
    mask = diagnostics[f"{key}_mask"][index]
    target = diagnostics[f"{key}_target"][index]
    samples = diagnostics[f"{key}_samples"][index]

    fig = plt.figure(figsize=(5.65, 1.76), facecolor=PAPER)
    panel_title(fig, "b", "Conditional-flow posterior inference")
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.220, top=0.760, width_ratios=[1.15, 0.62, 0.92], wspace=0.38)
    ax_in = fig.add_subplot(gs[0, 0])
    ax_sampler = fig.add_subplot(gs[0, 1])
    ax_out = fig.add_subplot(gs[0, 2])

    plot_posterior_predictive_fit(ax_in, depth, disp, mask, samples)
    draw_sampler_block(ax_sampler)
    plot_posterior_vs(ax_out, depth, target, samples, title=r"Posterior $V_S$")

    arrow_between(fig, ax_in, ax_sampler)
    arrow_between(fig, ax_sampler, ax_out)
    save_asset(fig, "fig01_asset_B_posterior_sampler")


def make_asset_c(diagnostics: np.lib.npyio.NpzFile, metrics: list[dict[str, str]], depth: np.ndarray, inside_index: int, boundary_index: int) -> None:
    prior_vs = diagnostics["DI_Strong_in_prior_target"][:, 1]
    inside_target = diagnostics["DI_Strong_in_prior_target"][inside_index, 1]
    inside_samples = diagnostics["DI_Strong_in_prior_samples"][inside_index, :, 1]
    boundary_target = diagnostics["DI_Strong_boundary_target"][boundary_index, 1]
    boundary_samples = diagnostics["DI_Strong_boundary_samples"][boundary_index, :, 1]

    p05, p95 = quantiles(prior_vs, (0.05, 0.95))

    fig = plt.figure(figsize=(5.65, 1.88), facecolor=PAPER)
    panel_title(fig, "c", "Prior-support audit")
    gs = fig.add_gridspec(1, 2, left=0.078, right=0.985, bottom=0.245, top=0.740, wspace=0.30)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    def profile_panel(ax: plt.Axes, target: np.ndarray, samples: np.ndarray, title: str, *, ylabel: bool) -> None:
        q16, q50, q84 = quantiles(samples, (0.16, 0.50, 0.84))
        ax.fill_betweenx(depth, p05, p95, color=SUPPORT_FILL, lw=0, zorder=1)
        ax.fill_betweenx(depth, q16, q84, color=POST_BLUE_FILL, lw=0, alpha=0.90, zorder=2)
        ax.plot(target, depth, color=TRUTH, lw=1.10, zorder=4)
        ax.plot(q50, depth, color=POST_BLUE, lw=1.22, ls=(0, (3.0, 1.8)), zorder=5)
        ax.set_ylim(100, 0)
        ax.set_xlim(0.55, 5.35)
        ax.set_xticks([1, 3, 5])
        ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.2, labelpad=1.0)
        ax.set_title(title, loc="left", fontsize=6.55, color=INK, pad=2.0)
        if ylabel:
            ax.set_ylabel("Depth (km)", fontsize=6.2, labelpad=1.0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        style_axis(ax)

    profile_panel(axes[0], inside_target, inside_samples, "inside support", ylabel=True)
    profile_panel(axes[1], boundary_target, boundary_samples, "support mismatch", ylabel=False)

    save_asset(fig, "fig01_asset_C_prior_support_audit")


def _meta_dicts(meta: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "subarray": int(row["subarray"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "n_periods_used": int(row["n_periods_used"]),
        }
        for row in meta
    ]


def smooth_section(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(values, ((0, 0), (width // 2, width // 2)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def smooth_vector(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_asset_d(field_volume_path: Path, field_summary_path: Path) -> None:
    data = np.load(field_volume_path, allow_pickle=True)
    depth = data["depth_km"]
    period = data["period_s"]
    disp = data["disp"]
    mask = data["mask"]
    median = data["median"][:, 1, :]
    q16 = data["q16"][:, 1, :]
    q84 = data["q84"][:, 1, :]
    ray = np.where(mask[:, 1, :].astype(bool), disp[:, 1, :], np.nan)
    valid = np.isfinite(ray).any(axis=0)
    period_window = valid & (period >= 6.0) & (period <= 40.0)
    rep = int(np.argsort(np.nanmean(ray[:, valid], axis=1))[len(ray) // 2])

    fig = plt.figure(figsize=(5.65, 1.88), facecolor=PAPER)
    panel_title(fig, "d", "Rayleigh-wave field workflow")
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.230, top=0.740, width_ratios=[1.10, 0.78, 0.92], wspace=0.36)
    ax_pick = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_range = fig.add_subplot(gs[0, 2])

    pick_p = period[period_window]
    pick_ray = ray[:, period_window]
    pick_q16, pick_q50, pick_q84 = np.nanpercentile(pick_ray, [16, 50, 84], axis=0)
    for i in range(0, len(pick_ray), 8):
        ax_pick.plot(pick_p, pick_ray[i], color="#9AA3AE", lw=0.30, alpha=0.10, zorder=1)
    ax_pick.fill_between(pick_p, pick_q16, pick_q84, color="#D8E7F4", lw=0, zorder=2)
    ax_pick.plot(pick_p, pick_q50, color=OBS, lw=1.35, zorder=4)
    ax_pick.set_xlim(float(pick_p.min()), float(pick_p.max()))
    ax_pick.set_ylim(float(np.nanpercentile(pick_ray, 2)) - 0.06, float(np.nanpercentile(pick_ray, 98)) + 0.06)
    ax_pick.set_xlabel("Period (s)", fontsize=6.4)
    ax_pick.set_ylabel(r"Phase velocity (km s$^{-1}$)", fontsize=6.4)
    ax_pick.set_title("field Rayleigh dispersion", loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax_pick)

    ax_profile.fill_betweenx(depth, q16[rep], q84[rep], color=POST_FILL, lw=0, zorder=2)
    ax_profile.plot(median[rep], depth, color=POST, lw=1.35, zorder=4)
    ax_profile.set_ylim(90, 0)
    ax_profile.set_xlim(1.6, 5.3)
    ax_profile.set_xlabel(r"$V_S$ (km s$^{-1}$)", fontsize=6.4)
    ax_profile.set_ylabel("Depth (km)", fontsize=6.4)
    ax_profile.set_title("local ensemble", loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax_profile)

    depth_window = depth <= 90.0
    z = depth[depth_window]
    field_profiles = median[:, depth_window]
    q10 = smooth_vector(np.quantile(field_profiles, 0.10, axis=0), width=7)
    q50 = smooth_vector(np.quantile(field_profiles, 0.50, axis=0), width=7)
    q90 = smooth_vector(np.quantile(field_profiles, 0.90, axis=0), width=7)
    for i in np.linspace(0, field_profiles.shape[0] - 1, 8, dtype=int):
        ax_range.plot(smooth_vector(field_profiles[i], width=7), z, color="#9AA3AE", lw=0.38, alpha=0.22, zorder=1)
    ax_range.fill_betweenx(z, q10, q90, color=POST_FILL, lw=0, zorder=2)
    ax_range.plot(q50, z, color=POST, lw=1.35, zorder=4)
    ax_range.set_ylim(90, 0)
    ax_range.set_xlim(1.8, 4.8)
    ax_range.set_xlabel(r"$V_S$ (km s$^{-1}$)", fontsize=6.4)
    ax_range.set_ylabel("")
    ax_range.set_title("field profile range", loc="left", fontsize=6.55, color=INK, pad=2.0)
    style_axis(ax_range)

    save_asset(fig, "fig01_asset_D_field_workflow")
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--field-volume", type=Path, default=DEFAULT_FIELD_VOLUME_PATH)
    parser.add_argument("--field-summary", type=Path, default=DEFAULT_FIELD_SUMMARY_PATH)
    parser.add_argument("--posterior-index", type=int, default=186)
    parser.add_argument("--training-index", type=int, default=941)
    parser.add_argument("--audit-index", type=int, default=703)
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    metrics = read_metrics(args.metrics)
    depth = np.linspace(0.0, 127.5, diagnostics["DI_Strong_in_prior_target"].shape[-1], dtype=float)

    make_asset_a(diagnostics, depth, args.training_index)
    make_asset_b(diagnostics, depth, args.posterior_index)
    make_asset_c(diagnostics, metrics, depth, args.posterior_index, args.audit_index)
    if args.field_volume.exists():
        make_asset_d(args.field_volume, args.field_summary)
    else:
        raise FileNotFoundError(f"Field volume not found: {args.field_volume}")
    diagnostics.close()


if __name__ == "__main__":
    main()
