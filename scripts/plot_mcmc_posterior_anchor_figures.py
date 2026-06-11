#!/usr/bin/env python3
"""Plot clean DI-vs-MCMC posterior-anchor figures from saved diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 7.0
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_METRICS = (
    ROOT
    / "results"
    / "same_prior_mcmc_posterior"
    / "strong_gmm6_k10_validated_cases0_5"
    / "same_prior_mcmc_posterior_metrics_combined.csv"
)
DEFAULT_EDGE_METRICS = (
    ROOT
    / "results"
    / "same_prior_mcmc_posterior"
    / "strong_gmm6_k10_boundary_out_cases0_1_resume_s2600"
    / "same_prior_mcmc_posterior_metrics.csv"
)
DEFAULT_EDGE_DIAG = (
    ROOT
    / "results"
    / "same_prior_mcmc_posterior"
    / "strong_gmm6_k10_boundary_out_cases0_1_resume_s2600"
    / "same_prior_mcmc_posterior_diagnostics.npz"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "same_prior_mcmc_posterior" / "posterior_anchor_publication"

COLORS = {
    "target": "#111111",
    "mcmc": "#4f8f58",
    "mcmc_fill": "#b8d6bd",
    "di": "#2f76b7",
    "di_fill": "#c9dcec",
    "prior": "#9aa9b7",
    "prior_fill": "#e9eef3",
    "grid": "#e6ebf0",
    "bad": "#b84b4b",
}

REGIME_LABELS = {
    "in-prior": "inside support",
    "boundary": "near edge",
    "out-of-prior": "outside support",
}


def style(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def key_prefix(regime: str, case_index: int) -> str:
    if regime == "in-prior":
        return f"case{case_index}"
    return f"{regime}_case{case_index}".replace("-", "_")


def diag_path_from_row(row: pd.Series) -> Path:
    source = row.get("source_file", "")
    if isinstance(source, str) and source:
        return Path(source).parent / "same_prior_mcmc_posterior_diagnostics.npz"
    raise ValueError("Row does not include a source_file diagnostics pointer")


def load_case(row: pd.Series, diag_path: Path | None = None) -> Dict[str, object]:
    regime = str(row["regime"])
    case_index = int(row["case_index"])
    path = diag_path if diag_path is not None else diag_path_from_row(row)
    z = np.load(path, allow_pickle=True)
    prefix = key_prefix(regime, case_index)
    legacy = f"case{case_index}"
    if f"{prefix}_target" not in z and f"{legacy}_target" in z:
        prefix = legacy
    target = np.asarray(z[f"{prefix}_target"], dtype=float)
    depth = np.arange(target.shape[-1], dtype=float) * 0.5
    out: Dict[str, object] = {
        "row": row,
        "regime": regime,
        "case_index": case_index,
        "target": target,
        "mcmc_q": np.asarray(z[f"{prefix}_mcmc_profile_qs"], dtype=float),
        "di_q": np.asarray(z[f"{prefix}_di_profile_qs"], dtype=float),
        "depth": depth,
        "prior_q": None,
    }
    if "prior_params" in z and "knot_depths_km" in z:
        prior_params = np.asarray(z["prior_params"], dtype=float)
        knot_depths = np.asarray(z["knot_depths_km"], dtype=float)
        prior_vs = np.stack([np.interp(depth, knot_depths, p[:-1]) for p in prior_params[:2500]])
        out["prior_q"] = np.quantile(prior_vs, (0.05, 0.95), axis=0)
    return out


def save_multi(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=360)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_profile_panel(ax, case: Dict[str, object], title: str, show_ylabel: bool, show_prior: bool) -> None:
    depth = case["depth"]
    target = case["target"]
    m_q = case["mcmc_q"]
    d_q = case["di_q"]
    prior_q = case["prior_q"]
    if show_prior and prior_q is not None:
        ax.fill_betweenx(depth, prior_q[0], prior_q[1], color=COLORS["prior_fill"], lw=0, zorder=0)
    ax.fill_betweenx(depth, m_q[0, 1], m_q[4, 1], color=COLORS["mcmc_fill"], alpha=0.70, lw=0, zorder=1)
    ax.fill_betweenx(depth, d_q[0, 1], d_q[4, 1], color=COLORS["di_fill"], alpha=0.72, lw=0, zorder=2)
    ax.plot(m_q[2, 1], depth, color=COLORS["mcmc"], lw=1.55, zorder=4)
    ax.plot(d_q[2, 1], depth, color=COLORS["di"], lw=1.45, ls=(0, (3.0, 1.7)), zorder=5)
    ax.plot(target[1], depth, color=COLORS["target"], lw=1.45, zorder=6)
    ax.set_ylim(float(depth[-1]), 0)
    ax.set_xlim(1.1, 5.35)
    ax.set_xlabel("$V_S$ (km/s)")
    if show_ylabel:
        ax.set_ylabel("Depth (km)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold")
    style(ax)


def plot_inside_grid(metrics: pd.DataFrame, out_dir: Path) -> Path:
    rows = metrics.sort_values("case_index").to_dict("records")
    cases = [load_case(pd.Series(row)) for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.815, wspace=0.18, hspace=0.30)
    for ax, case in zip(axes.flat, cases):
        row = case["row"]
        title = f"case {int(row['case_index'])}: diff {row['vs_median_absdiff_mcmc_di_km_s']:.2f}, overlap {row['vs_p05_p95_interval_overlap']:.2f}"
        plot_profile_panel(ax, case, title, show_ylabel=ax in axes[:, 0], show_prior=False)
    handles = [
        plt.Line2D([0], [0], color=COLORS["target"], lw=1.6, label="target"),
        plt.Line2D([0], [0], color=COLORS["mcmc"], lw=1.6, label="MCMC median"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["mcmc_fill"], ec="none", alpha=0.70, label="MCMC p5-p95"),
        plt.Line2D([0], [0], color=COLORS["di"], lw=1.6, ls=(0, (3.0, 1.7)), label="DI median"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["di_fill"], ec="none", alpha=0.72, label="DI p5-p95"),
    ]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 0.885), columnspacing=0.9)
    fig.suptitle(
        "Reduced Bayesian reference versus DI posterior: inside-support cases",
        x=0.085,
        y=0.955,
        ha="left",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.905,
        "Bands show pointwise p5-p95 intervals for $V_S$; lines show posterior medians.",
        ha="left",
        fontsize=7.0,
        color="0.30",
    )
    out = out_dir / "mcmc_anchor_inside_support_profiles"
    save_multi(fig, out)
    return out.with_suffix(".png")


def parse_transition(spec: str) -> Iterable[Tuple[str, int]]:
    for part in spec.split(","):
        regime, case_id = part.split(":")
        yield regime.strip(), int(case_id)


def plot_transition(in_metrics: pd.DataFrame, edge_metrics: pd.DataFrame, edge_diag: Path, spec: str, out_dir: Path) -> Path:
    cases = []
    for regime, case_index in parse_transition(spec):
        if regime == "in-prior":
            row = in_metrics.loc[in_metrics["case_index"].astype(int) == case_index].iloc[0]
            cases.append(load_case(row))
        else:
            subset = edge_metrics[(edge_metrics["regime"] == regime) & (edge_metrics["case_index"].astype(int) == case_index)]
            if subset.empty:
                raise ValueError(f"No metrics row for {regime}:{case_index}")
            cases.append(load_case(subset.iloc[0], edge_diag))

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.86])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.74, wspace=0.20)

    for i, (ax, case) in enumerate(zip(axes, cases)):
        row = case["row"]
        regime_label = REGIME_LABELS.get(str(row["regime"]), str(row["regime"]))
        usable = bool(row.get("mcmc_usable", True))
        title = f"{regime_label}\ncase {int(row['case_index'])}"
        plot_profile_panel(ax, case, title, show_ylabel=i == 0, show_prior=True)
        note_color = "0.25" if usable else COLORS["bad"]
        note = (
            f"Rhat {row['mcmc_rhat_max']:.2f}\n"
            f"ESS {row['mcmc_ess_min']:.0f}\n"
            f"fit {row['mcmc_pred_rms_km_s']:.3f}\n"
            f"diff {row['vs_median_absdiff_mcmc_di_km_s']:.2f}"
        )
        ax.text(0.04, 0.04, note, transform=ax.transAxes, fontsize=5.7, color=note_color, va="bottom")

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=COLORS["target"], lw=1.6, label="target"),
        plt.Line2D([0], [0], color=COLORS["mcmc"], lw=1.6, label="MCMC median"),
        plt.Line2D([0], [0], color=COLORS["di"], lw=1.6, ls=(0, (3.0, 1.7)), label="DI median"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["mcmc_fill"], ec="none", alpha=0.70, label="MCMC p5-p95"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["di_fill"], ec="none", alpha=0.72, label="DI p5-p95"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["prior_fill"], ec="none", alpha=1.0, label="projected prior p5-p95"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.2, handlelength=1.8)
    ax.text(
        0.0,
        0.05,
        "MCMC reference uses a\nlow-dimensional control-point\nparameterization fitted to the\nstrong synthetic prior.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="0.30",
        linespacing=1.25,
    )
    fig.suptitle(
        "Posterior anchor across prior-support regimes",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.895,
        "Inside support tests posterior approximation; near/outside support tests why reliability audits are needed.",
        ha="left",
        fontsize=7.0,
        color="0.30",
    )
    out = out_dir / "mcmc_anchor_support_transition_profiles"
    save_multi(fig, out)
    return out.with_suffix(".png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-prior-metrics", type=Path, default=DEFAULT_IN_METRICS)
    p.add_argument("--edge-metrics", type=Path, default=DEFAULT_EDGE_METRICS)
    p.add_argument("--edge-diagnostics", type=Path, default=DEFAULT_EDGE_DIAG)
    p.add_argument("--transition", default="in-prior:3,boundary:0,out-of-prior:1")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    in_metrics = pd.read_csv(args.in_prior_metrics)
    edge_metrics = pd.read_csv(args.edge_metrics)
    inside = plot_inside_grid(in_metrics, args.out_dir)
    transition = plot_transition(in_metrics, edge_metrics, args.edge_diagnostics, args.transition, args.out_dir)
    print(f"[done] wrote {inside}")
    print(f"[done] wrote {transition}")


if __name__ == "__main__":
    main()
