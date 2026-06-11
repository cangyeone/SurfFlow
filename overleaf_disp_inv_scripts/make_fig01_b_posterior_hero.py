#!/usr/bin/env python3
"""Draw Figure 1 panel B as a standalone posterior-sampling hero asset."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SAMPLE_PATH = ROOT / "SurfFlow" / "results" / "fair_di_comparison" / "production" / "figure_samples" / "posterior_figure_samples.npz"
FALLBACK_SAMPLE_PATH = Path(
    "/Users/liuxin/Documents/submissions/co-authored/yzy_swi_gradient/"
    "overleaf_surface_wave_flow/figures/posterior_figure_samples.npz"
)
DEFAULT_DIAGNOSTICS_PATH = Path(
    "/Volumes/lx_exFAT/yzy_directSWI/code_data/results/fair_di_comparison/production/fair_di_diagnostics.npz"
)
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
RAYLEIGH = "#C94C4C"
LOVE = "#537AA3"
POST = "#D07C2C"
POST_SOFT = "#EBC9A8"
TRUTH = "#111111"
MEDIAN = "#D07C2C"
PROFILE = "#D07C2C"
SHADE = "#EEF1F4"

WAVES = (
    ("Rayleigh", "rayleigh", 1, RAYLEIGH, "o"),
    ("Love", "love", 2, LOVE, "s"),
)


def sample_path() -> Path:
    if LOCAL_SAMPLE_PATH.exists():
        return LOCAL_SAMPLE_PATH
    if FALLBACK_SAMPLE_PATH.exists():
        return FALLBACK_SAMPLE_PATH
    raise FileNotFoundError(
        "No posterior figure samples found. Expected either "
        f"{LOCAL_SAMPLE_PATH} or {FALLBACK_SAMPLE_PATH}."
    )


def finish(fig: plt.Figure, stem: str) -> None:
    stems = [stem]
    if stem == "fig01_B_posterior_hero_synthetic_draft":
        stems.append("fig01_B_posterior_hero_draft")
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        for out_stem in stems:
            fig.savefig(out_dir / f"{out_stem}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{out_stem}.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.tick_params(direction="out", length=3.0, width=0.75, colors=INK)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.62)


def draw_dispersion(ax: plt.Axes, periods: np.ndarray, dispersion: np.ndarray, mask: np.ndarray) -> None:
    ray_valid = mask[1].astype(bool)
    love_valid = mask[2].astype(bool)
    if ray_valid.any():
        ax.plot(periods[ray_valid], dispersion[1, ray_valid], color=RAYLEIGH, lw=1.25)
        ax.scatter(
            periods[ray_valid],
            dispersion[1, ray_valid],
            s=16,
            facecolor="white",
            edgecolor=RAYLEIGH,
            linewidth=0.8,
            zorder=4,
        )
    if love_valid.any():
        ax.plot(periods[love_valid], dispersion[2, love_valid], color=LOVE, lw=1.25)
        ax.scatter(
            periods[love_valid],
            dispersion[2, love_valid],
            s=16,
            facecolor="white",
            edgecolor=LOVE,
            linewidth=0.8,
            zorder=4,
        )
    else:
        ax.text(0.98, 0.87, "Love masked", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=SUBTLE)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_xlim(1, 61)
    ax.set_ylim(2.35, 4.15)
    style_axis(ax)

    strip = ax.inset_axes([0.08, -0.31, 0.84, 0.12])
    strip.set_xlim(periods.min(), periods.max())
    strip.set_ylim(0, 2)
    strip.set_yticks([1.5, 0.5])
    strip.set_yticklabels(["R", "L"], fontsize=5.8)
    strip.set_xticks([])
    strip.tick_params(length=0, pad=1.0, colors=SUBTLE)
    for spine in strip.spines.values():
        spine.set_visible(False)
    for row, color, yy in [(1, RAYLEIGH, 1.1), (2, LOVE, 0.1)]:
        valid = mask[row].astype(bool)
        if valid.any():
            strip.broken_barh([(float(periods[valid].min()), float(np.ptp(periods[valid])))], (yy, 0.62), facecolors=color, alpha=0.72)
        else:
            strip.broken_barh([(float(periods.min()), float(np.ptp(periods)))], (yy, 0.62), facecolors="#E6E9ED", alpha=0.95)
    strip.text(0.0, -0.32, "input mask", transform=strip.transAxes, ha="left", va="top", fontsize=5.6, color=SUBTLE)


def forward_wave_predictions(depth: np.ndarray, periods: np.ndarray, samples: np.ndarray, wave: str) -> np.ndarray:
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
            wave=wave,
        )[0]
        predictions.append(np.asarray(result.velocity, dtype=np.float64))
    return np.stack(predictions, axis=0)


def draw_dispersion_fit(
    ax: plt.Axes,
    periods: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    predicted_by_wave: dict[str, np.ndarray],
) -> dict[str, float]:
    y_minima = []
    y_maxima = []
    x_valid = []
    residual_metrics: dict[str, float] = {}
    for label, _wave, row, color, marker in WAVES:
        keep = mask[row].astype(bool)
        if not keep.any() or label not in predicted_by_wave:
            continue
        pred = predicted_by_wave[label][:, keep]
        p = periods[keep]
        obs = observed[row, keep]
        q16 = np.quantile(pred, 0.16, axis=0)
        q50 = np.quantile(pred, 0.50, axis=0)
        q84 = np.quantile(pred, 0.84, axis=0)
        for curve in pred:
            ax.plot(p, curve, color=color, lw=0.45, alpha=0.11, zorder=1)
        ax.fill_between(p, q16, q84, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(p, q50, color=color, lw=1.45, zorder=4)
        ax.scatter(p, obs, s=16, marker=marker, facecolor="white", edgecolor=TRUTH, linewidth=0.78, zorder=5)
        y_minima.extend([float(np.min(q16)), float(np.min(obs))])
        y_maxima.extend([float(np.max(q84)), float(np.max(obs))])
        x_valid.extend([float(p.min()), float(p.max())])
        residual = q50 - obs
        residual_metrics[label] = float(np.mean(np.abs(residual)))
    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_xlim(max(0, min(x_valid) - 1.5), max(x_valid) + 3.0)
    ax.set_ylim(min(y_minima) - 0.08, max(y_maxima) + 0.08)
    style_axis(ax)
    return residual_metrics


def draw_rayleigh_fit_clean(
    ax: plt.Axes,
    periods: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    predicted_samples: np.ndarray,
) -> None:
    keep = mask[1].astype(bool)
    pred = predicted_samples[:, keep]
    p = periods[keep]
    obs = observed[1, keep]
    q10 = np.quantile(pred, 0.10, axis=0)
    q50 = np.quantile(pred, 0.50, axis=0)
    q90 = np.quantile(pred, 0.90, axis=0)

    for curve in pred[::2]:
        ax.plot(p, curve, color=RAYLEIGH, lw=0.34, alpha=0.045, zorder=1)
    ax.fill_between(p, q10, q90, color=RAYLEIGH, alpha=0.24, linewidth=0, zorder=2)
    ax.plot(p, q50, color=RAYLEIGH, lw=1.55, zorder=4)
    ax.scatter(p, obs, s=9.0, facecolor="white", edgecolor=TRUTH, linewidth=0.62, zorder=5)

    ax.set_xlabel("Period (s)")
    ax.set_ylabel(r"Phase velocity (km s$^{-1}$)")
    ax.set_xlim(max(0, float(p.min()) - 1.5), float(p.max()) + 3.0)
    ax.set_ylim(min(float(q10.min()), float(obs.min())) - 0.12, max(float(q90.max()), float(obs.max())) + 0.10)
    ax.set_title("Rayleigh dispersion", loc="left", fontsize=7.9, color=INK, pad=3.0)
    style_axis(ax)


def draw_vs_profile(
    ax: plt.Axes,
    depth: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
) -> float:
    channel_samples = samples[:, 1, :]
    median = np.median(channel_samples, axis=0)
    q16 = np.quantile(channel_samples, 0.16, axis=0)
    q84 = np.quantile(channel_samples, 0.84, axis=0)

    ax.axhspan(75, 127.5, color=SHADE, zorder=0)
    for curve in channel_samples:
        ax.plot(curve, depth, color=PROFILE, lw=0.55, alpha=0.16, zorder=1)
    ax.fill_betweenx(depth, q16, q84, color=PROFILE, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(target[1], depth, color=TRUTH, lw=1.25, zorder=4)
    ax.plot(median, depth, color=PROFILE, lw=1.65, zorder=5)
    ax.set_ylim(100, 0)
    ax.set_xlim(1.8, 5.7)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.text(0.98, 0.17, "weakly constrained\ndeep interval", transform=ax.transAxes, ha="right", va="center", fontsize=6.0, color=SUBTLE)
    style_axis(ax)
    shallow = depth <= 75.0
    return float(np.mean(np.abs(median[shallow] - target[1, shallow])))


def draw_vs_profile_clean(
    ax: plt.Axes,
    depth: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
) -> None:
    channel_samples = samples[:, 1, :]
    median = np.median(channel_samples, axis=0)
    q10 = np.quantile(channel_samples, 0.10, axis=0)
    q90 = np.quantile(channel_samples, 0.90, axis=0)

    ax.axhspan(75, 100, color=SHADE, zorder=0)
    for curve in channel_samples[::2]:
        ax.plot(curve, depth, color=PROFILE, lw=0.34, alpha=0.055, zorder=1)
    ax.fill_betweenx(depth, q10, q90, color=PROFILE, alpha=0.24, linewidth=0, zorder=2)
    ax.plot(target[1], depth, color=TRUTH, lw=1.35, zorder=4)
    ax.plot(median, depth, color=PROFILE, lw=1.75, zorder=5)

    ax.set_ylim(100, 0)
    ax.set_xlim(1.8, 5.35)
    ax.set_xlabel(r"$V_S$ (km s$^{-1}$)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(r"$V_S$ posterior", loc="left", fontsize=7.9, color=INK, pad=3.0)
    style_axis(ax)


def draw_arrays_clean(
    *,
    depth: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
    dispersion: np.ndarray,
    mask: np.ndarray,
    stem: str,
) -> None:
    periods = dispersion[0]
    rayleigh_predictions = forward_wave_predictions(depth, periods, samples, "rayleigh")

    fig = plt.figure(figsize=(6.75, 2.65))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.18],
        left=0.078,
        right=0.992,
        bottom=0.19,
        top=0.865,
        wspace=0.28,
    )
    ax_disp = fig.add_subplot(gs[0, 0])
    ax_vs = fig.add_subplot(gs[0, 1])

    draw_rayleigh_fit_clean(ax_disp, periods, dispersion, mask, rayleigh_predictions)
    draw_vs_profile_clean(ax_vs, depth, target, samples)

    fig.text(0.008, 0.982, "B", ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    finish(fig, stem)


def draw_arrays(
    *,
    prefix: str,
    depth: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
    dispersion: np.ndarray,
    mask: np.ndarray,
    posterior_samples: int,
    stem: str,
) -> None:
    periods = dispersion[0]
    predicted_by_wave = {
        label: forward_wave_predictions(depth, periods, samples, wave)
        for label, wave, row, _color, _marker in WAVES
        if mask[row].astype(bool).any()
    }

    fig = plt.figure(figsize=(7.45, 3.55))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.34],
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.78,
        wspace=0.28,
    )
    ax_disp = fig.add_subplot(gs[0, 0])
    ax_vs = fig.add_subplot(gs[0, 1])

    dispersion_mae = draw_dispersion_fit(ax_disp, periods, dispersion, mask, predicted_by_wave)
    vs_mae = draw_vs_profile(ax_vs, depth, target, samples)

    fig.text(0.012, 0.965, "B", ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.055, 0.958, r"$V_S$ posterior and dispersion fit", ha="left", va="top", fontsize=10.2, fontweight="bold", color=INK)
    valid_waves = [label for label, _wave, row, _color, _marker in WAVES if mask[row].astype(bool).any()]
    period_limits = [
        (float(periods[mask[row].astype(bool)][0]), float(periods[mask[row].astype(bool)][-1]))
        for _label, _wave, row, _color, _marker in WAVES
        if mask[row].astype(bool).any()
    ]
    period_text = "; ".join(
        f"{label[0]} {lo:.0f}-{hi:.0f} s"
        for label, (lo, hi) in zip(valid_waves, period_limits)
    )
    fig.text(
        0.055,
        0.895,
        rf"{' + '.join(valid_waves)} input, posterior-predictive dispersion, and the corresponding $V_S$ ensemble",
        ha="left",
        va="top",
        fontsize=7.3,
        color=SUBTLE,
    )
    fig.text(0.405, 0.635, r"$q_\theta(m|d)$", ha="center", va="center", fontsize=8.6, color=INK)
    fig.add_artist(
        FancyArrowPatch(
            (0.425, 0.635),
            (0.462, 0.635),
            transform=fig.transFigure,
            arrowstyle="-|>",
            lw=0.9,
            color=SUBTLE,
            mutation_scale=11,
        )
    )

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=TRUTH, markeredgewidth=0.78, markersize=4.2)]
    labels = ["observed / truth"]
    for label, _wave, _row, color, _marker in WAVES:
        if label in valid_waves:
            handles.append(Line2D([0], [0], color=color, lw=1.45))
            labels.append(f"{label} median")
    handles.extend(
        [
            Line2D([0], [0], color=PROFILE, lw=1.55),
            Line2D([0], [0], color=PROFILE, lw=6.0, alpha=0.22),
        ]
    )
    labels.extend([r"$V_S$ median", "16-84% sample quantile"])
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
        ncol=min(5, len(handles)),
        fontsize=6.7,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    fig.text(
        0.075,
        0.065,
        f"{prefix}; {period_text}; "
        + "; ".join(f"{label} MAE {value:.3f}" for label, value in dispersion_mae.items())
        + rf"; $V_S$ MAE 0-75 km {vs_mae:.3f} km s$^{{-1}}$",
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=SUBTLE,
    )
    fig.text(
        0.985,
        0.065,
        f"{posterior_samples} posterior samples; raw sample quantiles, not field-calibrated intervals",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=SUBTLE,
    )
    finish(fig, stem)


def draw_archive_case(prefix: str) -> None:
    data = np.load(sample_path())
    draw_arrays(
        prefix=prefix,
        depth=np.asarray(data["depth_km"], dtype=float),
        target=np.asarray(data[f"{prefix}_target"], dtype=float),
        samples=np.asarray(data[f"{prefix}_posterior_samples"], dtype=float),
        dispersion=np.asarray(data[f"{prefix}_dispersion"], dtype=float),
        mask=np.asarray(data[f"{prefix}_mask"], dtype=float),
        posterior_samples=int(data["posterior_samples"]),
        stem=f"fig01_B_posterior_hero_{prefix}_draft",
    )
    data.close()


def draw_diagnostics_case(path: Path, method: str, regime: str, index: int, stem_override: str | None = None) -> None:
    diagnostics = np.load(path)
    key = f"{method}_{regime}"
    target = np.asarray(diagnostics[f"{key}_target"][index], dtype=float)
    samples = np.asarray(diagnostics[f"{key}_samples"][index], dtype=float)
    dispersion = np.asarray(diagnostics[f"{key}_disp"][index], dtype=float)
    mask = np.asarray(diagnostics[f"{key}_mask"][index], dtype=float)
    depth = np.linspace(0.0, 127.5, target.shape[-1], dtype=float)
    label = f"{method.replace('_', '-')}, {regime.replace('_', '-')}, sample {index}"
    stem = stem_override or f"fig01_B_posterior_hero_{method}_{regime}_i{index}_draft"
    draw_arrays(
        prefix=label,
        depth=depth,
        target=target,
        samples=samples,
        dispersion=dispersion,
        mask=mask,
        posterior_samples=int(samples.shape[0]),
        stem=stem,
    )
    diagnostics.close()


def draw_diagnostics_case_clean(path: Path, method: str, regime: str, index: int, stem: str) -> None:
    diagnostics = np.load(path)
    key = f"{method}_{regime}"
    target = np.asarray(diagnostics[f"{key}_target"][index], dtype=float)
    samples = np.asarray(diagnostics[f"{key}_samples"][index], dtype=float)
    dispersion = np.asarray(diagnostics[f"{key}_disp"][index], dtype=float)
    mask = np.asarray(diagnostics[f"{key}_mask"][index], dtype=float)
    depth = np.linspace(0.0, 127.5, target.shape[-1], dtype=float)
    draw_arrays_clean(
        depth=depth,
        target=target,
        samples=samples,
        dispersion=dispersion,
        mask=mask,
        stem=stem,
    )
    diagnostics.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["archive", "diagnostics"], default="archive")
    parser.add_argument("--case", choices=["synthetic", "ak135", "all"], default="all")
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--method", choices=["DI_Strong", "DI_Weak"], default="DI_Weak")
    parser.add_argument("--regime", choices=["in_prior", "boundary", "out_of_prior"], default="boundary")
    parser.add_argument("--index", type=int, default=983)
    parser.add_argument("--stem", help="Optional output filename stem for diagnostics cases.")
    parser.add_argument("--clean-panel", action="store_true", help="Draw a compact panel-ready version with minimal annotations.")
    args = parser.parse_args()
    if args.source == "diagnostics":
        if args.clean_panel:
            draw_diagnostics_case_clean(
                args.diagnostics,
                args.method,
                args.regime,
                args.index,
                args.stem or f"fig01_B_posterior_hero_{args.method}_{args.regime}_i{args.index}_clean",
            )
            return
        draw_diagnostics_case(args.diagnostics, args.method, args.regime, args.index, args.stem)
        return
    cases = ("synthetic", "ak135") if args.case == "all" else (args.case,)
    for case in cases:
        draw_archive_case(case)


if __name__ == "__main__":
    main()
