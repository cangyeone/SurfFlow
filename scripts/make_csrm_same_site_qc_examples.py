#!/usr/bin/env python3
"""Draw same-site CSRM QC examples for SurfFlow and QEDispInv.

Each row shows one representative CSRM site:
left, Rayleigh phase-velocity fit; right, Vs profile comparison against the
CSRM reference model.  The figure is intended for scientific QC before deciding
which field/baseline panels should enter the manuscript.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
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

from run_openswi_csrm_transfer import forward_rayleigh_prediction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURF_DIAG = ROOT / "results" / "openswi_csrm_transfer" / "n512_s32_step24" / "csrm_transfer_diagnostics.npz"
DEFAULT_QEDISP_DIR = ROOT / "results" / "qedisp_csrm_multistart" / "sites32_starts64"
DEFAULT_DISPFORMER = ROOT / "results" / "dispformer_openswi_csrm" / "sites32_official" / "dispformer_same_sites_predictions.npz"
DEFAULT_OUT = ROOT / "figures" / "openswi_csrm_transfer" / "n512_s32_step24" / "csrm_same_site_qc_examples"

COLORS = {
    "obs": "#111111",
    "qedisp": "#6f6f6f",
    "strong": "#2f76b7",
    "weak": "#d07b28",
    "dispformer": "#147a3d",
    "dispformer_light": "#70aa7b",
    "grid": "#e7ebf0",
}


def read_combined(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "site_index": int(row["site_index"]),
                    "method": row["method"],
                    "vs_mae_km_s": float(row["vs_mae_km_s"]),
                }
            )
    return rows


def choose_sites(rows: List[Dict[str, object]], max_sites: int = 3) -> List[int]:
    strong = [r for r in rows if r["method"] == "DI-Strong"]
    strong = sorted(strong, key=lambda r: float(r["vs_mae_km_s"]))
    if len(strong) <= max_sites:
        return [int(r["site_index"]) for r in strong]
    quantiles = np.linspace(0.15, 0.85, max_sites)
    selected = []
    for q in quantiles:
        idx = min(len(strong) - 1, max(0, int(round(q * (len(strong) - 1)))))
        selected.append(int(strong[idx]["site_index"]))
    return selected


def qedisp_best_dispersion(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        data = h5["data"][...]
        lookup = {(round(float(row[0]), 8), int(round(float(row[2])))): float(row[1]) for row in data}
        best_key = None
        best_rms = np.inf
        for key in h5["disp"].keys():
            pred = h5["disp"][key][...]
            residuals = []
            for row in pred:
                obs = lookup.get((round(float(row[0]), 8), int(round(float(row[2])))))
                if obs is not None:
                    residuals.append(float(row[1]) - obs)
            if residuals:
                rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    best_key = key
        if best_key is None:
            return np.empty(0), np.empty(0)
        pred = h5["disp"][best_key][...]
        period = 1.0 / pred[:, 0]
        order = np.argsort(period)
        return period[order], pred[:, 1][order]


def read_qedisp_profile(h5_path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        return {
            "z": h5["z_sample"][...].astype(float),
            "median": h5["vs_median"][...].astype(float),
            "p10": h5["vs_cred10"][...].astype(float),
            "p90": h5["vs_cred90"][...].astype(float),
        }


def style(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def panel_label(ax, label: str, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.4},
        zorder=30,
    )


def plot(args: argparse.Namespace) -> None:
    surf = np.load(args.surf_diag)
    dispformer = np.load(args.dispformer) if args.dispformer is not None and args.dispformer.exists() else None
    site_index = surf["site_index"].astype(int)
    site_to_row = {int(site): i for i, site in enumerate(site_index)}
    dispformer_site_to_row = {}
    if dispformer is not None:
        dispformer_site_to_row = {int(site): i for i, site in enumerate(dispformer["site_index"].astype(int))}
    combined = read_combined(args.qedisp_dir / "combined_same_sites_metrics.csv")
    sites = [int(x) for x in args.sites] if args.sites else choose_sites(combined, max_sites=3)

    n = len(sites)
    fig, axes = plt.subplots(n, 2, figsize=(7.1, 2.35 * n + 0.45), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.90, wspace=0.32, hspace=0.36)

    model_depth = surf["model_depth"].astype(float)
    period_grid = surf["period_grid"].astype(float)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, site in enumerate(sites):
        i = site_to_row[site]
        qh5 = args.qedisp_dir / f"site_{site:05d}" / "qedispinv.h5"
        if not qh5.exists():
            raise FileNotFoundError(qh5)

        valid = surf["mask"][i, 1] > 0
        obs_period = period_grid[valid]
        obs_c = surf["disp"][i, 1, valid]

        ax = axes[r, 0]
        ax.scatter(obs_period, obs_c, s=15, color=COLORS["obs"], edgecolor="white", lw=0.25, zorder=8, label="Observed")
        q_period, q_c = qedisp_best_dispersion(qh5)
        if q_period.size:
            ax.plot(q_period, q_c, color=COLORS["qedisp"], lw=1.35, label="QEDisp best")
        for key, label, color in [
            ("di_strong_median_profile", "DI-Strong median", COLORS["strong"]),
            ("di_weak_median_profile", "DI-Weak median", COLORS["weak"]),
        ]:
            pred = forward_rayleigh_prediction(surf[key][i], model_depth, obs_period.astype(np.float32))
            if pred is not None:
                ax.plot(obs_period, pred, color=color, lw=1.45, label=label)
        if dispformer is not None and site in dispformer_site_to_row:
            di = dispformer_site_to_row[site]
            raw_p = dispformer["raw_disp"][di, :, 0]
            for key, label, color, ls, lw in [
                ("dispformer_phase_only_phase_pred", "DispFormer phase-only", COLORS["dispformer_light"], "--", 1.25),
                ("dispformer_phaseplusgroup_phase_pred", "DispFormer phase+group", COLORS["dispformer"], "-", 1.30),
            ]:
                y = dispformer[key][di]
                ok = np.isfinite(y) & (raw_p >= obs_period.min() - 1e-4) & (raw_p <= obs_period.max() + 1e-4)
                if np.any(ok):
                    ax.plot(raw_p[ok], y[ok], color=color, lw=lw, ls=ls, label=label)
        ax.set_xlim(max(0, obs_period.min() - 2), min(62, obs_period.max() + 2))
        pad = max(0.03, 0.07 * (float(np.nanmax(obs_c)) - float(np.nanmin(obs_c))))
        ax.set_ylim(float(np.nanmin(obs_c)) - pad, float(np.nanmax(obs_c)) + pad)
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("$c$ (km/s)")
        panel_label(ax, next(letters), f"site {site}: Rayleigh phase")
        style(ax)

        ax = axes[r, 1]
        ref_depth = surf["ref_depth"][i].astype(float)
        ref_vs = surf["ref_vs"][i].astype(float)
        qprof = read_qedisp_profile(qh5)
        ax.plot(ref_vs, ref_depth, color=COLORS["obs"], lw=1.35, label="CSRM reference")
        ax.fill_betweenx(qprof["z"], qprof["p10"], qprof["p90"], color=COLORS["qedisp"], alpha=0.16, lw=0)
        ax.plot(qprof["median"], qprof["z"], color=COLORS["qedisp"], lw=1.35, label="QEDisp median")
        for prefix, label, color in [
            ("di_strong", "DI-Strong median", COLORS["strong"]),
            ("di_weak", "DI-Weak median", COLORS["weak"]),
        ]:
            med = surf[f"{prefix}_median_vs_ref_depth"][i]
            q05 = surf[f"{prefix}_q05_vs_ref_depth"][i]
            q95 = surf[f"{prefix}_q95_vs_ref_depth"][i]
            ax.fill_betweenx(ref_depth, q05, q95, color=color, alpha=0.13, lw=0)
            ax.plot(med, ref_depth, color=color, lw=1.45, label=label)
        if dispformer is not None and site in dispformer_site_to_row:
            di = dispformer_site_to_row[site]
            ax.plot(
                dispformer["dispformer_phase_only_pred_vs_ref_depth"][di],
                ref_depth,
                color=COLORS["dispformer_light"],
                lw=1.25,
                ls="--",
                label="DispFormer phase-only",
            )
            ax.plot(
                dispformer["dispformer_phaseplusgroup_pred_vs_ref_depth"][di],
                ref_depth,
                color=COLORS["dispformer"],
                lw=1.30,
                label="DispFormer phase+group",
            )
        ax.set_ylim(120, 0)
        ax.set_xlim(2.0, 5.0)
        ax.set_xlabel("$V_S$ (km/s)")
        ax.set_ylabel("Depth (km)")
        panel_label(ax, next(letters), f"site {site}: model")
        style(ax)
    handles, labels = [], []
    for ax in (axes[0, 0], axes[0, 1]):
        h, lab = ax.get_legend_handles_labels()
        for one_h, one_lab in zip(h, lab):
            if one_lab not in labels:
                handles.append(one_h)
                labels.append(one_lab)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        fontsize=6.8,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=350)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    plt.close(fig)
    print(args.out.with_suffix(".png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surf-diag", type=Path, default=DEFAULT_SURF_DIAG)
    parser.add_argument("--qedisp-dir", type=Path, default=DEFAULT_QEDISP_DIR)
    parser.add_argument("--dispformer", type=Path, default=DEFAULT_DISPFORMER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sites", type=int, nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
