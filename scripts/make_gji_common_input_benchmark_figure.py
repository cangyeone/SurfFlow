#!/usr/bin/env python3
"""Draw the GJI main-text common-input benchmark figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8.2
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_METRICS = ROOT / "results" / "dispformer_openswi_csrm" / "sites128_official" / "same_sites_all_methods_metrics.csv"
DEFAULT_SUMMARY = ROOT / "results" / "dispformer_openswi_csrm" / "sites128_official" / "same_sites_all_methods_summary.json"
DEFAULT_OUT = WORKSPACE / "paper-overleaf" / "figures" / "fig05_common_input_benchmark"

METHODS: Sequence[Tuple[str, str, str, str]] = (
    ("QEDisp m0 multistart", "QEDisp", "#6f6f6f", "Rayleigh phase"),
    ("DI-Strong", "DI-Strong", "#2f76b7", "Rayleigh phase"),
    ("DI-Weak", "DI-Weak", "#d07b28", "Rayleigh phase"),
    ("DispFormer phase-only", "DispFormer\nphase-only", "#70aa7b", "Rayleigh phase"),
    ("DispFormer phase+group", "DispFormer\nphase+group", "#147a3d", "phase + group"),
)

GRID = "#e7ebf0"
TEXT = "#20242a"


def read_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out: Dict[str, object] = dict(row)
            for key in [
                "vs_mae_km_s",
                "reference_in_p05_p95",
                "disp_best_rms_km_s",
                "phase_forward_rms_km_s",
                "group_forward_rms_km_s",
            ]:
                val = row.get(key)
                out[key] = np.nan if val in ("", None, "nan") else float(val)
            rows.append(out)
    return rows


def values(rows: List[Dict[str, object]], method: str, key: str) -> np.ndarray:
    return np.asarray(
        [float(r[key]) for r in rows if r["method"] == method and np.isfinite(float(r[key]))],
        dtype=float,
    )


def panel_label(ax, label: str, text: str) -> None:
    ax.text(
        -0.08,
        1.03,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT,
    )


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2.0, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def draw_box_strip(ax, rows: List[Dict[str, object]], rng: np.random.Generator) -> None:
    data = [values(rows, method, "vs_mae_km_s") for method, _label, _color, _input in METHODS]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.1},
        whiskerprops={"color": "0.25", "linewidth": 0.9},
        capprops={"color": "0.25", "linewidth": 0.9},
    )
    for patch, (_method, _label, color, input_label) in zip(bp["boxes"], METHODS):
        alpha = 0.24 if input_label == "Rayleigh phase" else 0.18
        patch.set(facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.0)
        if input_label != "Rayleigh phase":
            patch.set_hatch("//")

    for j, ((method, _label, color, input_label), vals) in enumerate(zip(METHODS, data), start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        edge = "white" if input_label == "Rayleigh phase" else color
        face = color if input_label == "Rayleigh phase" else "white"
        ax.scatter(
            np.full(len(vals), j) + jitter,
            vals,
            s=13,
            color=face,
            edgecolor=edge,
            linewidth=0.25,
            alpha=0.70,
            zorder=6,
        )

    ax.set_xticks(np.arange(1, len(METHODS) + 1))
    ax.set_xticklabels([label for _method, label, _color, _input in METHODS])
    ax.set_ylabel("$V_S$ MAE to CSRM reference (km/s)")
    ax.set_ylim(0.025, 0.36)
    ax.axvspan(0.55, 4.45, color="#f7f9fb", zorder=-5)
    ax.axvspan(4.55, 5.45, color="#f2f7f2", zorder=-5)
    ax.text(2.5, 0.345, "common Rayleigh-phase input", ha="center", va="top", fontsize=7.2, color="0.35")
    ax.text(5.0, 0.345, "extra group input", ha="center", va="top", fontsize=7.2, color="0.35")
    panel_label(ax, "a", "same-site model error")
    style(ax)


def draw_coverage(ax, rows: List[Dict[str, object]], rng: np.random.Generator) -> None:
    methods = [
        ("DI-Strong", "DI-Strong", "#2f76b7"),
        ("DI-Weak", "DI-Weak", "#d07b28"),
    ]
    data = [values(rows, method, "reference_in_p05_p95") for method, _label, _color in methods]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.1},
        whiskerprops={"color": "0.25", "linewidth": 0.9},
        capprops={"color": "0.25", "linewidth": 0.9},
    )
    for patch, (_method, _label, color) in zip(bp["boxes"], methods):
        patch.set(facecolor=color, alpha=0.24, edgecolor=color, linewidth=1.0)
    for j, ((_method, _label, color), vals) in enumerate(zip(methods, data), start=1):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full(len(vals), j) + jitter,
            vals,
            s=13,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.70,
            zorder=6,
        )
    ax.axhline(0.90, color="0.25", lw=0.85, ls=":", zorder=2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label for _method, label, _color in methods])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Reference profile inside p05--p95 band")
    panel_label(ax, "b", "DI posterior coverage")
    style(ax)


def plot(args: argparse.Namespace) -> None:
    rows = read_rows(args.metrics)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    n = int(summary["DI-Strong"]["n"])
    rng = np.random.default_rng(20260611)

    fig = plt.figure(figsize=(7.1, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 0.88], left=0.08, right=0.985, bottom=0.18, top=0.88, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_box_strip(ax0, rows, rng)
    draw_coverage(ax1, rows, rng)
    fig.suptitle(f"Common-input benchmark on CSRM reference sites (n={n})", x=0.08, ha="left", y=0.985, fontsize=10.0, fontweight="bold", color=TEXT)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=350)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    plt.close(fig)
    print(args.out.with_suffix(".png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
