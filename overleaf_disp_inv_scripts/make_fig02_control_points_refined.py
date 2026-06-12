#!/usr/bin/env python3
"""Generate a publication-style Figure 2 for the depth-control representation."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SURFFLOW = ROOT / "SurfFlow"
PAPER_FIG_DIR = ROOT / "paper-overleaf" / "figures"
MANUSCRIPT_FIG_DIR = SURFFLOW / "manuscript" / "figures"
PAPER_SCRIPT = SURFFLOW / "overleaf_disp_inv_scripts" / "make_paper_figures.py"

INK = "#20242B"
SUBTLE = "#6E7682"
GRID = "#E8EBEF"
PAPER = "#FFFFFF"
PRIOR = "#2F6F9F"
PRIOR_FILL = "#DCE9F4"
CONTROL = "#C9823A"

PDF_METADATA = {
    "Creator": "make_fig02_control_points_refined.py",
    "CreationDate": datetime(2026, 6, 10, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 6, 10, tzinfo=timezone.utc),
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=2.4, width=0.65, colors=INK, pad=1.6, labelsize=6.2)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.grid(color=GRID, linewidth=0.5)


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left", fontsize=6.8, fontweight="bold", color=INK, pad=1.6)


def representative_index(vs_profiles: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> int:
    width = np.maximum(q90 - q10, 0.08)
    centre_score = np.mean(((vs_profiles - q50) / width) ** 2, axis=1)
    outside = np.maximum(q10 - vs_profiles, 0.0) + np.maximum(vs_profiles - q90, 0.0)
    outside_score = np.mean(outside / width, axis=1)
    return int(np.argmin(centre_score + 8.0 * outside_score))


def control_depth_grid(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    controls = np.array(
        list(np.arange(0.0, 5.0 + 0.5, 0.5))
        + list(np.arange(6.0, 10.0 + 1.0, 1.0))
        + list(np.arange(15.0, 50.0 + 5.0, 5.0))
        + [70.0, 90.0, 110.0, float(depth[-1])],
        dtype=float,
    )
    indices = np.array([int(np.argmin(np.abs(depth - z))) for z in controls], dtype=int)
    return controls, indices


def save_all(fig: plt.Figure, stem: str) -> None:
    for out_dir in (PAPER_FIG_DIR, MANUSCRIPT_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025, metadata=PDF_METADATA)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.png", dpi=440, bbox_inches="tight", pad_inches=0.025)


def main() -> None:
    if str(SURFFLOW) not in sys.path:
        sys.path.insert(0, str(SURFFLOW))

    paper = load_module("make_paper_figures_for_fig02", PAPER_SCRIPT)
    weak_data_mod = load_module("generate_data_weak_prior_for_fig02", SURFFLOW / "utils" / "generate_data_weak_prior.py")

    model_batch, _, _ = paper.collect_dataset(weak_data_mod, n=128)
    target = model_batch[:, 1:4, :].float()

    depth = model_batch[0, 0, :].detach().cpu().numpy()
    control_depths, control_indices = control_depth_grid(depth)
    vs_profiles = target[:, 1, :].detach().cpu().numpy()
    q10, q50, q90 = np.quantile(vs_profiles, [0.10, 0.50, 0.90], axis=0)
    median_controls = q50[control_indices]
    median_control_interp = np.interp(depth, control_depths, median_controls)

    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.42), sharey=True, facecolor=PAPER)
    ax = axes[0]
    ax.fill_betweenx(depth, q10, q90, color=PRIOR_FILL, linewidth=0, zorder=1, label="p10-p90")
    ax.plot(q50, depth, color=PRIOR, lw=1.16, alpha=0.98, zorder=3, label="median profile")
    ax.set_ylim(float(depth.max()), 0.0)
    ax.set_xlim(0.75, 5.45)
    ax.set_ylabel("Depth (km)", fontsize=6.6, labelpad=1.0)
    panel_title(ax, "a", "Weak-prior ensemble")
    style_axis(ax)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        fontsize=5.3,
        handlelength=1.45,
        borderaxespad=0.0,
        frameon=False,
    )

    ax = axes[1]
    ax.plot(median_control_interp, depth, color=INK, lw=1.10, zorder=3, label="interpolated profile")
    ax.plot(
        median_controls,
        control_depths,
        "o-",
        color=CONTROL,
        ms=2.5,
        lw=0.85,
        zorder=4,
        label="28 depth controls",
    )
    ax.set_ylim(float(depth.max()), 0.0)
    ax.set_xlim(0.75, 5.45)
    ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.8, labelpad=1.4)
    ax.set_ylabel("")
    panel_title(ax, "b", "Depth-control representation")
    style_axis(ax)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        fontsize=5.3,
        handlelength=1.45,
        borderaxespad=0.0,
        frameon=False,
    )

    for ax in axes:
        ax.set_xlabel(r"$V_S$ (km/s)", fontsize=6.8, labelpad=1.4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.185, top=0.905, wspace=0.18)
    save_all(fig, "fig02_control_points")
    plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(2026)
    np.random.seed(2026)
    main()
