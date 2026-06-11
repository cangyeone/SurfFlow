#!/usr/bin/env python3
"""Run SurfFlow transfer tests on OpenSWI-real/CSRM.

This is a real-data benchmark/reference comparison, not a calibrated field
validation.  The CSRM files provide observed Rayleigh phase/group dispersion
curves and a reference Vs model.  SurfFlow is trained for Rayleigh/Love phase
velocity on a fixed 2--60 s grid, so this script uses only Rayleigh phase
velocities in the overlapping period band and masks the Love channel.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_DIR = WORKSPACE / "field_data_candidates" / "openswi_real_csrm"
DEFAULT_CKPT_ROOTS = [
    ROOT / "ckpt",
    WORKSPACE / "code_data" / "ckpt",
    Path("/Volumes/lx_exFAT/yzy_directSWI/code_data/ckpt"),
]
DEFAULT_OUT_DIR = ROOT / "results" / "openswi_csrm_transfer"
DEFAULT_FIG_DIR = ROOT / "figures" / "openswi_csrm_transfer"

METHODS = {
    "DI-Strong": "fair_di_strong_full_seed642026/best.pt",
    "DI-Weak": "fair_di_weak_full_seed642026/best.pt",
}

COLORS = {
    "DI-Strong": "#2f76b7",
    "DI-Weak": "#d07b28",
    "reference": "#111111",
    "interval": "#9ecae1",
    "grid": "#e5e9ef",
}


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_checkpoint(label: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"{label} checkpoint does not exist: {explicit}")
    rel = METHODS[label]
    tried = []
    for root in DEFAULT_CKPT_ROOTS:
        path = root / rel
        tried.append(path)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label} checkpoint. Tried:\n" + "\n".join(f"  - {p}" for p in tried)
    )


def load_model(checkpoint: Path, train_script: Path, device: torch.device):
    train_mod = import_from_path(f"surf_flow_train_{checkpoint.parent.name}", train_script)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = train_mod.Disp2StructCRF(
        H=len(ckpt["depth_grid"]),
        T=59,
        profile_channels=3,
        cond_base_channels=int(cfg["cond_base_channels"]),
        cond_dim=int(cfg["cond_dim"]),
        flow_hidden=int(cfg["flow_hidden"]),
        time_dim=int(cfg["time_dim"]),
        dropout=0.0,
        reference_profile=ckpt["reference_profile"],
        profile_scale=ckpt["profile_scale"],
        depth_grid=ckpt["depth_grid"],
        control_indices=ckpt["control_indices"],
        period_minmax=tuple(float(x) for x in ckpt["period_minmax"].reshape(-1).tolist()),
        disp_mean=ckpt["disp_mean"],
        disp_scale=ckpt["disp_scale"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def read_csrm(data_dir: Path) -> Dict[str, np.ndarray]:
    disp_path = data_dir / "obs_period_phase_group.npz"
    vs_path = data_dir / "obs_depth_vs.npz"
    loc_path = data_dir / "obs_depth_vs_loc.npz"
    missing = [p for p in (disp_path, vs_path, loc_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSRM file(s): " + ", ".join(str(p) for p in missing))
    disp = np.load(disp_path)["data"].astype(np.float32)
    vs = np.load(vs_path)["data"].astype(np.float32)
    loc = np.load(loc_path)["data"].astype(np.float32)
    return {"disp_raw": disp, "vs_ref_raw": vs, "loc": loc}


def build_conditions(
    csrm: Dict[str, np.ndarray],
    period_min: float,
    period_max: float,
    min_period_count: int,
    max_sites: int,
    seed: int,
    site_stride: int,
) -> Dict[str, np.ndarray]:
    raw = csrm["disp_raw"]
    periods_raw = raw[:, :, 0]
    phase_raw = raw[:, :, 1]
    grid = np.arange(2.0, 61.0, 1.0, dtype=np.float32)
    n_sites = raw.shape[0]
    disp = np.zeros((n_sites, 3, grid.size), dtype=np.float32)
    mask = np.zeros_like(disp)
    disp[:, 0, :] = grid[None, :]

    valid_count = np.zeros(n_sites, dtype=np.int32)
    period_span = np.zeros((n_sites, 2), dtype=np.float32)
    for i in range(n_sites):
        ok = (
            np.isfinite(periods_raw[i])
            & np.isfinite(phase_raw[i])
            & (phase_raw[i] > 0.0)
            & (periods_raw[i] >= period_min)
            & (periods_raw[i] <= period_max)
        )
        p = periods_raw[i, ok]
        c = phase_raw[i, ok]
        if p.size:
            order = np.argsort(p)
            p = p[order]
            c = c[order]
            for period in np.unique(p):
                j = int(round(float(period) - 2.0))
                if 0 <= j < grid.size and abs(grid[j] - float(period)) < 1e-4:
                    disp[i, 1, j] = float(np.median(c[p == period]))
                    mask[i, 1, j] = 1.0
            valid_count[i] = int(mask[i, 1].sum())
            if valid_count[i] > 0:
                used = grid[mask[i, 1] > 0]
                period_span[i] = (float(used.min()), float(used.max()))
    mask[:, 0, :] = (mask[:, 1:3, :].sum(axis=1) > 0).astype(np.float32)

    keep = np.where(valid_count >= min_period_count)[0]
    if site_stride > 1:
        keep = keep[::site_stride]
    if max_sites > 0 and keep.size > max_sites:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, size=max_sites, replace=False))

    ref_depth = csrm["vs_ref_raw"][keep, :, 0].astype(np.float32)
    ref_vs = csrm["vs_ref_raw"][keep, :, 1].astype(np.float32)
    loc = csrm["loc"][keep].astype(np.float32)
    return {
        "site_index": keep.astype(np.int32),
        "period_grid": grid,
        "disp": disp[keep],
        "mask": mask[keep],
        "loc": loc,
        "ref_depth": ref_depth,
        "ref_vs": ref_vs,
        "valid_count": valid_count[keep],
        "period_span": period_span[keep],
    }


def batched_indices(n: int, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, n, batch_size):
        yield np.arange(start, min(n, start + batch_size))


@torch.no_grad()
def run_sampler(
    model,
    disp: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    posterior_samples: int,
    euler_steps: int,
    batch_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = None if device.type == "mps" else torch.Generator(device=device).manual_seed(seed)
    samples = []
    medians = []
    stds = []
    for idx in batched_indices(len(disp), batch_size):
        d = torch.from_numpy(disp[idx]).float().to(device)
        m = torch.from_numpy(mask[idx]).float().to(device)
        out = model.sample(
            d,
            m,
            num_samples=posterior_samples,
            num_steps=euler_steps,
            temperature=1.0,
            generator=generator,
        )
        samples.append(out["profile_samples"].detach().cpu().numpy())
        medians.append(out["profile_median"].detach().cpu().numpy())
        stds.append(out["profile_std"].detach().cpu().numpy())
    sample_arr = np.concatenate(samples, axis=0)
    return {
        "samples": sample_arr,
        "median": np.concatenate(medians, axis=0),
        "std": np.concatenate(stds, axis=0),
        "q05": np.quantile(sample_arr, 0.05, axis=1).astype(np.float32),
        "q95": np.quantile(sample_arr, 0.95, axis=1).astype(np.float32),
    }


def align_to_reference_depth(
    model_depth: np.ndarray,
    ref_depth: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate model-depth values to each site's reference-depth grid."""
    if ref_depth.ndim != 2:
        raise ValueError(f"Expected ref_depth [N,D], got {ref_depth.shape}")
    n_site, n_depth = ref_depth.shape
    if values.ndim == 3:
        out = np.zeros((values.shape[0], values.shape[1], n_depth), dtype=np.float32)
        for i in range(n_site):
            for s in range(values.shape[1]):
                out[i, s] = np.interp(ref_depth[i], model_depth, values[i, s]).astype(np.float32)
        return out
    if values.ndim == 2:
        out = np.zeros((values.shape[0], n_depth), dtype=np.float32)
        for i in range(n_site):
            out[i] = np.interp(ref_depth[i], model_depth, values[i]).astype(np.float32)
        return out
    raise ValueError(f"Expected values [N,D] or [N,S,D], got {values.shape}")


def metrics_for_method(
    label: str,
    posterior: Dict[str, np.ndarray],
    model_depth: np.ndarray,
    ref_depth: np.ndarray,
    ref_vs: np.ndarray,
    runtime_s: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    med_vs = align_to_reference_depth(model_depth, ref_depth, posterior["median"][:, 1, :])
    q05_vs = align_to_reference_depth(model_depth, ref_depth, posterior["q05"][:, 1, :])
    q95_vs = align_to_reference_depth(model_depth, ref_depth, posterior["q95"][:, 1, :])
    err = med_vs - ref_vs
    inside = (ref_vs >= q05_vs) & (ref_vs <= q95_vs)
    row: Dict[str, float] = {
        "method": label,
        "n_sites": int(ref_vs.shape[0]),
        "n_depth": int(ref_vs.shape[1]),
        "runtime_s": float(runtime_s),
        "vs_mae_km_s": float(np.mean(np.abs(err))),
        "vs_rmse_km_s": float(np.sqrt(np.mean(err**2))),
        "vs_bias_km_s": float(np.mean(err)),
        "reference_in_p05_p95": float(np.mean(inside)),
        "reference_in_p05_p95_shallow_0_40km": float(np.mean(inside[:, ref_depth[0] <= 40.0])),
        "vs_mae_0_40km_km_s": float(np.mean(np.abs(err[:, ref_depth[0] <= 40.0]))),
        "vs_mae_40_120km_km_s": float(np.mean(np.abs(err[:, (ref_depth[0] > 40.0) & (ref_depth[0] <= 120.0)]))),
    }
    diagnostics = {
        "median_vs_ref_depth": med_vs.astype(np.float32),
        "q05_vs_ref_depth": q05_vs.astype(np.float32),
        "q95_vs_ref_depth": q95_vs.astype(np.float32),
        "err_vs_ref_depth": err.astype(np.float32),
        "inside_p05_p95": inside.astype(np.uint8),
        "mae_by_depth": np.mean(np.abs(err), axis=0).astype(np.float32),
        "bias_by_depth": np.mean(err, axis=0).astype(np.float32),
        "inside_by_depth": np.mean(inside, axis=0).astype(np.float32),
    }
    return row, diagnostics


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def style_axis(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.2")


def panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.6},
        zorder=20,
    )


def select_examples(rows: List[Dict[str, object]], diagnostics: Dict[str, Dict[str, np.ndarray]]) -> List[int]:
    if "DI-Strong" in diagnostics:
        site_mae = np.mean(np.abs(diagnostics["DI-Strong"]["err_vs_ref_depth"]), axis=1)
    else:
        site_mae = np.mean(np.abs(next(iter(diagnostics.values()))["err_vs_ref_depth"]), axis=1)
    qs = [0.15, 0.50, 0.85]
    examples = []
    order = np.argsort(site_mae)
    for q in qs:
        examples.append(int(order[min(len(order) - 1, max(0, int(round(q * (len(order) - 1)))))]))
    return sorted(set(examples))


def forward_rayleigh_prediction(
    profile: np.ndarray,
    model_depth: np.ndarray,
    periods: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        gen = import_from_path("csrm_generate_data_forward", ROOT / "utils" / "generate_data.py")
        out = gen.compute_phase_dispersion(
            model_depth.astype(float),
            profile[0].astype(float),
            profile[1].astype(float),
            profile[2].astype(float),
            periods.astype(float),
            modes=(0,),
            wave="rayleigh",
        )
        return out[0].velocity.astype(np.float32)
    except Exception:
        return None


def make_summary_figure(
    fig_path_base: Path,
    field: Dict[str, np.ndarray],
    model_depth: np.ndarray,
    posteriors: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, Dict[str, np.ndarray]],
    rows: List[Dict[str, object]],
) -> None:
    fig_path_base.parent.mkdir(parents=True, exist_ok=True)
    ref_depth = field["ref_depth"]
    ref_vs = field["ref_vs"]
    lon = field["loc"][:, 0]
    lat = field["loc"][:, 1]
    period_grid = field["period_grid"]
    disp = field["disp"]
    mask = field["mask"]
    examples = select_examples(rows, metrics)

    fig = plt.figure(figsize=(10.2, 7.0))
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.985, bottom=0.08, top=0.965, wspace=0.38, hspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    ref_60 = np.array([np.interp(60.0, ref_depth[i], ref_vs[i]) for i in range(len(ref_vs))])
    sc = ax.scatter(lon, lat, c=ref_60, s=11, cmap="RdBu", edgecolor="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(sc, ax=ax, shrink=0.76, pad=0.02)
    cb.set_label("Reference $V_S$ at 60 km (km/s)")
    panel_label(ax, "a", "CSRM benchmark sites")
    style_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(20260611)
    thin = rng.choice(len(disp), size=min(220, len(disp)), replace=False)
    for i in thin:
        valid = mask[i, 1] > 0
        ax.plot(period_grid[valid], disp[i, 1, valid], color="0.70", lw=0.45, alpha=0.22)
    for ex in examples:
        valid = mask[ex, 1] > 0
        ax.plot(period_grid[valid], disp[ex, 1, valid], lw=1.25, alpha=0.95, label=f"site {field['site_index'][ex]}")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("$c$ (km/s)")
    ax.set_xlim(7, 61)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.2)
    panel_label(ax, "b", "Rayleigh phase inputs")
    style_axis(ax)

    ax = fig.add_subplot(gs[0, 2])
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods))
    vals = [float(row["vs_mae_km_s"]) for row in rows]
    inc = [float(row["reference_in_p05_p95"]) for row in rows]
    bars = ax.bar(x - 0.17, vals, width=0.34, color=[COLORS[m] for m in methods], alpha=0.82)
    ax.set_ylabel("$V_S$ MAE vs reference (km/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax2 = ax.twinx()
    ax2.scatter(x + 0.20, inc, marker="D", s=34, color="0.15", label="Reference in p5-p95")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Reference in p5-p95")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    panel_label(ax, "c", "aggregate reference comparison")
    style_axis(ax)
    ax2.grid(False)

    ax = fig.add_subplot(gs[1, 0])
    z = ref_depth[0]
    for label in methods:
        ax.plot(metrics[label]["mae_by_depth"], z, color=COLORS[label], lw=1.6, label=label)
    ax.invert_yaxis()
    ax.set_xlabel("$V_S$ MAE (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(120, 0)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "d", "depth-dependent error")
    style_axis(ax)

    ax = fig.add_subplot(gs[1, 1])
    for label in methods:
        ax.plot(metrics[label]["inside_by_depth"], z, color=COLORS[label], lw=1.6, label=label)
    ax.axvline(0.90, color="0.45", lw=0.85, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel("Reference in p5-p95")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(120, 0)
    panel_label(ax, "e", "posterior interval inclusion")
    style_axis(ax)

    ax = fig.add_subplot(gs[1, 2])
    ex = examples[len(examples) // 2]
    ax.plot(ref_vs[ex], ref_depth[ex], color=COLORS["reference"], lw=1.35, label="CSRM reference")
    for label in methods:
        diag = metrics[label]
        ax.fill_betweenx(
            ref_depth[ex],
            diag["q05_vs_ref_depth"][ex],
            diag["q95_vs_ref_depth"][ex],
            color=COLORS[label],
            alpha=0.16,
            lw=0,
        )
        ax.plot(diag["median_vs_ref_depth"][ex], ref_depth[ex], color=COLORS[label], lw=1.35, label=label)
    ax.invert_yaxis()
    ax.set_xlabel("$V_S$ (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(120, 0)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    panel_label(ax, "f", f"example profile, site {field['site_index'][ex]}")
    style_axis(ax)

    fig.savefig(fig_path_base.with_suffix(".pdf"))
    fig.savefig(fig_path_base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    # Separate example dispersion-fit figure for sanity checking only.
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.0, 2.4 * len(examples)), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.96, wspace=0.34, hspace=0.38)
    for row, ex in enumerate(examples):
        valid = mask[ex, 1] > 0
        obs_periods = period_grid[valid]
        obs_vel = disp[ex, 1, valid]
        ax = axes[row, 0]
        ax.scatter(obs_periods, obs_vel, s=13, color=COLORS["reference"], label="Observed Rayleigh")
        for label in methods:
            med_profile = posteriors[label]["median"][ex]
            pred = forward_rayleigh_prediction(med_profile, model_depth, obs_periods)
            if pred is not None:
                ax.plot(obs_periods, pred, color=COLORS[label], lw=1.35, label=label)
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("$c$ (km/s)")
        ax.legend(frameon=False, fontsize=6.8, loc="lower right")
        panel_label(ax, chr(ord("a") + 2 * row), f"site {field['site_index'][ex]} dispersion")
        style_axis(ax)

        ax = axes[row, 1]
        ax.plot(ref_vs[ex], ref_depth[ex], color=COLORS["reference"], lw=1.25, label="CSRM reference")
        for label in methods:
            diag = metrics[label]
            ax.fill_betweenx(
                ref_depth[ex],
                diag["q05_vs_ref_depth"][ex],
                diag["q95_vs_ref_depth"][ex],
                color=COLORS[label],
                alpha=0.16,
                lw=0,
            )
            ax.plot(diag["median_vs_ref_depth"][ex], ref_depth[ex], color=COLORS[label], lw=1.25, label=label)
        ax.invert_yaxis()
        ax.set_ylim(120, 0)
        ax.set_xlabel("$V_S$ (km/s)")
        ax.set_ylabel("Depth (km)")
        ax.legend(frameon=False, fontsize=6.8, loc="lower right")
        panel_label(ax, chr(ord("b") + 2 * row), f"site {field['site_index'][ex]} profile")
        style_axis(ax)
    fig.savefig(fig_path_base.with_name(fig_path_base.name + "_examples").with_suffix(".pdf"))
    fig.savefig(fig_path_base.with_name(fig_path_base.name + "_examples").with_suffix(".png"), dpi=300)
    plt.close(fig)


def save_outputs(
    out_dir: Path,
    field: Dict[str, np.ndarray],
    model_depth: np.ndarray,
    posteriors: Dict[str, Dict[str, np.ndarray]],
    diagnostics: Dict[str, Dict[str, np.ndarray]],
    rows: List[Dict[str, object]],
    protocol: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "csrm_transfer_metrics.csv", rows)
    write_json(out_dir / "csrm_transfer_metrics.json", rows)
    write_json(out_dir / "csrm_transfer_protocol.json", protocol)
    arrays: Dict[str, np.ndarray] = {
        "site_index": field["site_index"],
        "period_grid": field["period_grid"],
        "disp": field["disp"],
        "mask": field["mask"],
        "loc": field["loc"],
        "ref_depth": field["ref_depth"],
        "ref_vs": field["ref_vs"],
        "valid_count": field["valid_count"],
        "period_span": field["period_span"],
        "model_depth": model_depth.astype(np.float32),
    }
    for label, diag in diagnostics.items():
        prefix = label.lower().replace("-", "_")
        for key, value in diag.items():
            arrays[f"{prefix}_{key}"] = value
    for label, posterior in posteriors.items():
        prefix = label.lower().replace("-", "_")
        arrays[f"{prefix}_median_profile"] = posterior["median"].astype(np.float32)
        arrays[f"{prefix}_q05_profile"] = posterior["q05"].astype(np.float32)
        arrays[f"{prefix}_q95_profile"] = posterior["q95"].astype(np.float32)
        # Full samples are useful but can be large; keep them for the selected
        # benchmark size because reproducibility matters more here.
        arrays[f"{prefix}_samples"] = posterior["samples"].astype(np.float32)
    np.savez_compressed(out_dir / "csrm_transfer_diagnostics.npz", **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-script", type=Path, default=ROOT / "disp_inv_train.v1.3.py")
    parser.add_argument("--strong-ckpt", type=Path, default=None)
    parser.add_argument("--weak-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--period-min", type=float, default=8.0)
    parser.add_argument("--period-max", type=float, default=60.0)
    parser.add_argument("--min-period-count", type=int, default=10)
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--site-stride", type=int, default=1)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--euler-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", nargs="+", default=["DI-Strong", "DI-Weak"], choices=sorted(METHODS))
    parser.add_argument("--tag", default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    field = build_conditions(
        read_csrm(args.data_dir),
        period_min=args.period_min,
        period_max=args.period_max,
        min_period_count=args.min_period_count,
        max_sites=args.max_sites,
        seed=args.seed,
        site_stride=args.site_stride,
    )
    if len(field["site_index"]) == 0:
        raise RuntimeError("No CSRM sites passed the period-count QC.")

    out_dir = args.out_dir / args.tag
    fig_dir = args.fig_dir / args.tag
    ckpts = {
        "DI-Strong": resolve_checkpoint("DI-Strong", args.strong_ckpt),
        "DI-Weak": resolve_checkpoint("DI-Weak", args.weak_ckpt),
    }
    rows: List[Dict[str, object]] = []
    diagnostics: Dict[str, Dict[str, np.ndarray]] = {}
    posteriors: Dict[str, Dict[str, np.ndarray]] = {}
    model_depth: Optional[np.ndarray] = None

    for label in args.methods:
        tic = time.time()
        model, ckpt = load_model(ckpts[label], args.train_script, device)
        if model_depth is None:
            model_depth = model.depth_grid.detach().cpu().numpy().astype(np.float32)
        posterior = run_sampler(
            model,
            field["disp"],
            field["mask"],
            device=device,
            posterior_samples=args.posterior_samples,
            euler_steps=args.euler_steps,
            batch_size=args.batch_size,
            seed=args.seed + (17 if label == "DI-Weak" else 0),
        )
        runtime_s = time.time() - tic
        row, diag = metrics_for_method(label, posterior, model_depth, field["ref_depth"], field["ref_vs"], runtime_s)
        row.update(
            {
                "checkpoint": str(ckpts[label]),
                "posterior_samples": int(args.posterior_samples),
                "euler_steps": int(args.euler_steps),
                "period_min_s": float(args.period_min),
                "period_max_s": float(args.period_max),
                "min_period_count": int(args.min_period_count),
                "site_stride": int(args.site_stride),
            }
        )
        rows.append(row)
        diagnostics[label] = diag
        posteriors[label] = posterior

    assert model_depth is not None
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "SurfFlow transfer/reference comparison on OpenSWI-real/CSRM.",
        "data_dir": str(args.data_dir),
        "n_available_sites": int(read_csrm(args.data_dir)["disp_raw"].shape[0]),
        "n_used_sites": int(len(field["site_index"])),
        "site_selection": "Random subset after minimum valid Rayleigh phase-period count QC.",
        "rayleigh_phase_only": True,
        "group_velocity_used": False,
        "love_channel": "masked",
        "period_grid": "SurfFlow fixed 2--60 s grid; only observed 8--60 s Rayleigh phase samples are unmasked.",
        "comparison_reference": "OpenSWI-real/CSRM reference Vs profile; treated as an external benchmark reference, not exact ground truth.",
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoints": {label: str(path) for label, path in ckpts.items()},
    }
    save_outputs(out_dir, field, model_depth, posteriors, diagnostics, rows, protocol)
    make_summary_figure(fig_dir / "csrm_transfer_summary", field, model_depth, posteriors, diagnostics, rows)
    print(json.dumps({"rows": rows, "out_dir": str(out_dir), "fig_dir": str(fig_dir)}, indent=2))


if __name__ == "__main__":
    main()
