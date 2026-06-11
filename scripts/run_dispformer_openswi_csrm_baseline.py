#!/usr/bin/env python3
"""Run the official OpenSWI/DispFormer CSRM baseline on SurfFlow sites.

The official OpenSWI deep model is deterministic: it predicts one Vs profile
from the input dispersion curve.  This script therefore compares it as a point
estimator against the same CSRM reference profiles used for the SurfFlow and
QEDisp checks.  It runs two input protocols:

* phase_only: Rayleigh phase velocities only, matching the SurfFlow/QEDisp
  input information as closely as possible.
* phase_group: OpenSWI's official real-data input, using both phase and group
  velocities when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8.5
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DATA_DIR = WORKSPACE / "field_data_candidates" / "openswi_real_csrm"
DEFAULT_OPENSWI_ROOT = WORKSPACE / "external" / "OpenSWI"
DEFAULT_MODEL_DIR = (
    DEFAULT_OPENSWI_ROOT
    / "Script"
    / "Training"
    / "OpenSWI-deep"
    / "Article-Figure"
    / "model"
    / "dispformer_local_global_v1"
    / "dispformer_local_global_v1_md=128_nh=8_nl=3_od=301_lt=MSE_lr=0.0001_ne=1000_bs=512_ni=False"
)
DEFAULT_QEDISP_DIR = ROOT / "results" / "qedisp_csrm_multistart" / "sites32_starts64"
DEFAULT_SURF_DIAG = ROOT / "results" / "openswi_csrm_transfer" / "n512_s32_step24" / "csrm_transfer_diagnostics.npz"
DEFAULT_OUT_DIR = ROOT / "results" / "dispformer_openswi_csrm" / "sites32_official"
DEFAULT_FIG_DIR = ROOT / "figures" / "openswi_csrm_transfer" / "n512_s32_step24"

COLORS = {
    "DispFormer phase-only": "#4a9b62",
    "DispFormer phase+group": "#147a3d",
    "grid": "#e7ebf0",
    "reference": "#111111",
}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_site_indices(qedisp_dir: Path, surf_diag: Path, max_sites: int) -> np.ndarray:
    combined = qedisp_dir / "combined_same_sites_metrics.csv"
    if combined.exists():
        sites: List[int] = []
        with combined.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                site = int(row["site_index"])
                if site not in sites:
                    sites.append(site)
        return np.asarray(sites[:max_sites], dtype=np.int32)
    if surf_diag.exists():
        sites = np.load(surf_diag)["site_index"].astype(np.int32)
        return sites[:max_sites]
    raise FileNotFoundError(
        "No site list found. Expected either "
        f"{combined} or {surf_diag}."
    )


def read_csrm(data_dir: Path, site_index: np.ndarray) -> Dict[str, np.ndarray]:
    disp = np.load(data_dir / "obs_period_phase_group.npz")["data"].astype(np.float32)
    ref = np.load(data_dir / "obs_depth_vs.npz")["data"].astype(np.float32)
    loc = np.load(data_dir / "obs_depth_vs_loc.npz")["data"].astype(np.float32)
    return {
        "disp": disp[site_index],
        "ref_depth": ref[site_index, :, 0],
        "ref_vs": ref[site_index, :, 1],
        "loc": loc[site_index],
    }


def load_model(openswi_root: Path, model_dir: Path, device: torch.device):
    sys.path.insert(0, str(openswi_root))
    # OpenSWI's dispformer_local_global_v1.py imports einops.rearrange but does
    # not use it.  Avoid modifying the user's conda environment just for this
    # unused import.
    if "einops" not in sys.modules:
        stub = types.ModuleType("einops")

        def _unused_rearrange(*_args, **_kwargs):
            raise RuntimeError("einops.rearrange was unexpectedly called")

        stub.rearrange = _unused_rearrange
        sys.modules["einops"] = stub
    from SWInversion.model.dispformer_local_global_v1 import DispersionTransformer

    ckpt = model_dir / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = DispersionTransformer(
        model_dim=128,
        num_heads=8,
        num_layers=3,
        output_dim=301,
        scale_factor=5,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def make_input(raw_disp: np.ndarray, protocol: str, period_min: float, period_max: float) -> Tuple[np.ndarray, np.ndarray]:
    data = raw_disp.copy()
    data[~np.isfinite(data)] = -1.0
    data[data <= 0.0] = -1.0
    outside = (data[:, :, 0] < period_min) | (data[:, :, 0] > period_max)
    data[:, :, 1][outside] = -1.0
    data[:, :, 2][outside] = -1.0
    if protocol == "phase_only":
        data[:, :, 2] = -1.0
    elif protocol == "phase_group":
        pass
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    tensor = np.transpose(data, (0, 2, 1)).astype(np.float32)
    mask = (tensor[:, 1, :] <= 0.0) & (tensor[:, 2, :] <= 0.0)
    return tensor, mask


def batched(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


@torch.no_grad()
def predict(
    model,
    raw_disp: np.ndarray,
    protocol: str,
    device: torch.device,
    batch_size: int,
    period_min: float,
    period_max: float,
) -> np.ndarray:
    inputs, masks = make_input(raw_disp, protocol, period_min, period_max)
    outputs = []
    for sl in batched(inputs.shape[0], batch_size):
        x = torch.from_numpy(inputs[sl]).to(device)
        m = torch.from_numpy(masks[sl]).to(device)
        y = model(x, m).detach().cpu().numpy()
        outputs.append(y.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def import_openswi_dispersion(openswi_root: Path):
    sys.path.insert(0, str(openswi_root))
    from SWInversion.dispersion import calculate_dispersion, transform_vs_to_vel_model

    return calculate_dispersion, transform_vs_to_vel_model


def forward_dispersion(
    vs: np.ndarray,
    depth: np.ndarray,
    periods: np.ndarray,
    openswi_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    calculate_dispersion, transform_vs_to_vel_model = import_openswi_dispersion(openswi_root)
    ok_depth = depth <= 120.0
    if ok_depth.sum() < 3:
        ok_depth = np.ones_like(depth, dtype=bool)
    vel_model = transform_vs_to_vel_model(vs[ok_depth].astype(float), depth[ok_depth].astype(float))
    out = calculate_dispersion(vel_model.astype(float), t=periods.astype(float), dc=0.001)
    return out[:, 1].astype(np.float32), out[:, 2].astype(np.float32)


def compute_metrics(
    site_index: np.ndarray,
    csrm: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    openswi_root: Path,
    period_min: float,
    period_max: float,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    pred_depth = np.arange(301, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    arrays: Dict[str, np.ndarray] = {
        "site_index": site_index.astype(np.int32),
        "pred_depth": pred_depth,
    }

    ref_depth = csrm["ref_depth"]
    ref_vs = csrm["ref_vs"]
    raw_disp = csrm["disp"]

    for label, pred_vs in predictions.items():
        pred_on_ref = np.zeros_like(ref_vs, dtype=np.float32)
        phase_pred = np.full((len(site_index), raw_disp.shape[1]), np.nan, dtype=np.float32)
        group_pred = np.full((len(site_index), raw_disp.shape[1]), np.nan, dtype=np.float32)
        phase_rms = np.full(len(site_index), np.nan, dtype=np.float32)
        group_rms = np.full(len(site_index), np.nan, dtype=np.float32)
        vs_mae = np.full(len(site_index), np.nan, dtype=np.float32)
        vs_mae_0_40 = np.full(len(site_index), np.nan, dtype=np.float32)
        vs_mae_40_120 = np.full(len(site_index), np.nan, dtype=np.float32)

        for i in range(len(site_index)):
            pred_on_ref[i] = np.interp(ref_depth[i], pred_depth, pred_vs[i])
            err = pred_on_ref[i] - ref_vs[i]
            depth = ref_depth[i]
            full = (depth >= 0.0) & (depth <= 120.0)
            shallow = (depth >= 0.0) & (depth <= 40.0)
            deep = (depth > 40.0) & (depth <= 120.0)
            vs_mae[i] = float(np.mean(np.abs(err[full])))
            vs_mae_0_40[i] = float(np.mean(np.abs(err[shallow])))
            vs_mae_40_120[i] = float(np.mean(np.abs(err[deep])))

            in_window = (raw_disp[i, :, 0] >= period_min) & (raw_disp[i, :, 0] <= period_max)
            phase_ok = in_window & (raw_disp[i, :, 1] > 0.0)
            group_ok = in_window & (raw_disp[i, :, 2] > 0.0)
            any_ok = in_window & ((raw_disp[i, :, 1] > 0.0) | (raw_disp[i, :, 2] > 0.0))
            if np.any(any_ok):
                periods = raw_disp[i, any_ok, 0]
                ph, gr = forward_dispersion(pred_vs[i], pred_depth, periods, openswi_root)
                phase_pred[i, any_ok] = ph
                group_pred[i, any_ok] = gr
                if np.any(phase_ok):
                    phase_rms[i] = float(
                        np.sqrt(np.mean((phase_pred[i, phase_ok] - raw_disp[i, phase_ok, 1]) ** 2))
                    )
                if np.any(group_ok):
                    group_rms[i] = float(
                        np.sqrt(np.mean((group_pred[i, group_ok] - raw_disp[i, group_ok, 2]) ** 2))
                    )

            rows.append(
                {
                    "site_index": int(site_index[i]),
                    "method": label,
                    "vs_mae_km_s": float(vs_mae[i]),
                    "vs_mae_0_40km_km_s": float(vs_mae_0_40[i]),
                    "vs_mae_40_120km_km_s": float(vs_mae_40_120[i]),
                    "phase_forward_rms_km_s": float(phase_rms[i]) if np.isfinite(phase_rms[i]) else None,
                    "group_forward_rms_km_s": float(group_rms[i]) if np.isfinite(group_rms[i]) else None,
                    "input_protocol": "phase only" if "phase-only" in label else "phase + group",
                }
            )

        prefix = label.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        arrays[f"{prefix}_pred_vs"] = pred_vs.astype(np.float32)
        arrays[f"{prefix}_pred_vs_ref_depth"] = pred_on_ref
        arrays[f"{prefix}_phase_pred"] = phase_pred
        arrays[f"{prefix}_group_pred"] = group_pred
        arrays[f"{prefix}_vs_mae"] = vs_mae
        arrays[f"{prefix}_vs_mae_0_40"] = vs_mae_0_40
        arrays[f"{prefix}_vs_mae_40_120"] = vs_mae_40_120
        arrays[f"{prefix}_phase_forward_rms"] = phase_rms
        arrays[f"{prefix}_group_forward_rms"] = group_rms
    return rows, arrays


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    methods = sorted({str(r["method"]) for r in rows})
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        summary: Dict[str, object] = {"n": len(vals)}
        for key in [
            "vs_mae_km_s",
            "vs_mae_0_40km_km_s",
            "vs_mae_40_120km_km_s",
            "phase_forward_rms_km_s",
            "group_forward_rms_km_s",
        ]:
            arr = np.asarray([np.nan if r[key] is None else float(r[key]) for r in vals], dtype=float)
            summary[f"mean_{key}"] = float(np.nanmean(arr))
            summary[f"median_{key}"] = float(np.nanmedian(arr))
            summary[f"p25_{key}"] = float(np.nanpercentile(arr, 25))
            summary[f"p75_{key}"] = float(np.nanpercentile(arr, 75))
        out[method] = summary
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def make_summary_figure(out: Path, rows: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.88, wspace=0.35)

    methods = ["DispFormer phase-only", "DispFormer phase+group"]
    metrics = [
        ("vs_mae_km_s", "Vs MAE (km/s)"),
        ("phase_forward_rms_km_s", "Rayleigh phase RMS (km/s)"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        data = []
        for method in methods:
            vals = [float(r[key]) for r in rows if r["method"] == method and r[key] is not None]
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, method in zip(bp["boxes"], methods):
            patch.set(facecolor=COLORS[method], alpha=0.28, edgecolor=COLORS[method], linewidth=1.1)
        for part in ["whiskers", "caps", "medians"]:
            for artist in bp[part]:
                artist.set(color="0.22", linewidth=1.0)
        x = np.arange(1, len(methods) + 1)
        for j, method in enumerate(methods, start=1):
            vals = np.asarray(data[j - 1], dtype=float)
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), j) + jitter, vals, s=10, color=COLORS[method], alpha=0.65, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(["phase only", "phase+group"])
        ax.set_ylabel(ylabel)
        ax.grid(color=COLORS["grid"], lw=0.55)
        ax.tick_params(direction="out", length=3, width=0.7)
    axes[0].set_title("(a) model fit", loc="left", fontweight="bold")
    axes[1].set_title("(b) forward fit", loc="left", fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def make_all_methods_figure(out: Path, rows: List[Dict[str, object]]) -> None:
    labels = [
        ("QEDisp m0 multistart", "QEDisp"),
        ("DI-Strong", "DI-Strong"),
        ("DI-Weak", "DI-Weak"),
        ("DispFormer phase-only", "DispFormer\nphase-only"),
        ("DispFormer phase+group", "DispFormer\nphase+group"),
    ]
    colors = {
        "QEDisp m0 multistart": "#6f6f6f",
        "DI-Strong": "#2f76b7",
        "DI-Weak": "#d07b28",
        "DispFormer phase-only": "#70aa7b",
        "DispFormer phase+group": "#147a3d",
    }
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.88)
    data = []
    for method, _ in labels:
        vals = []
        for row in rows:
            if row.get("method") == method and row.get("vs_mae_km_s", "") not in ("", None):
                vals.append(float(row["vs_mae_km_s"]))
        data.append(vals)
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, (method, _) in zip(bp["boxes"], labels):
        patch.set(facecolor=colors[method], alpha=0.25, edgecolor=colors[method], linewidth=1.1)
    for part in ["whiskers", "caps", "medians"]:
        for artist in bp[part]:
            artist.set(color="0.22", linewidth=1.0)
    for j, vals in enumerate(data, start=1):
        vals = np.asarray(vals, dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) else []
        ax.scatter(
            np.full(len(vals), j) + jitter,
            vals,
            s=10,
            color=colors[labels[j - 1][0]],
            alpha=0.70,
            lw=0,
            zorder=5,
        )
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([label for _, label in labels])
    ax.set_ylabel("Vs MAE to CSRM reference (km/s)")
    n_sites = max((len(vals) for vals in data), default=0)
    ax.set_title(f"Same-site CSRM benchmark (n={n_sites})", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], lw=0.55, axis="y")
    ax.tick_params(direction="out", length=3, width=0.7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--openswi-root", type=Path, default=DEFAULT_OPENSWI_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--qedisp-dir", type=Path, default=DEFAULT_QEDISP_DIR)
    parser.add_argument("--surf-diag", type=Path, default=DEFAULT_SURF_DIAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--max-sites", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--period-min", type=float, default=8.0)
    parser.add_argument("--period-max", type=float, default=60.0)
    parser.add_argument("--protocols", nargs="+", default=["phase_only", "phase_group"], choices=["phase_only", "phase_group"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    device = choose_device(args.device)
    site_index = read_site_indices(args.qedisp_dir, args.surf_diag, args.max_sites)
    csrm = read_csrm(args.data_dir, site_index)
    model = load_model(args.openswi_root, args.model_dir, device)

    predictions: Dict[str, np.ndarray] = {}
    for protocol in args.protocols:
        label = "DispFormer phase-only" if protocol == "phase_only" else "DispFormer phase+group"
        predictions[label] = predict(
            model,
            csrm["disp"],
            protocol,
            device,
            args.batch_size,
            args.period_min,
            args.period_max,
        )

    rows, arrays = compute_metrics(site_index, csrm, predictions, args.openswi_root, args.period_min, args.period_max)
    summary = summarize(rows)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "device": str(device),
        "data_dir": str(args.data_dir),
        "openswi_root": str(args.openswi_root),
        "model_dir": str(args.model_dir),
        "site_source": str(args.qedisp_dir / "combined_same_sites_metrics.csv"),
        "n_sites": int(len(site_index)),
        "period_min": float(args.period_min),
        "period_max": float(args.period_max),
        "note": (
            "DispFormer is deterministic. phase_only is the strict same-information "
            "comparison with SurfFlow/QEDisp; phase_group is the official OpenSWI "
            "real-data input protocol."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "dispformer_same_sites_metrics.csv", rows)
    write_json(args.out_dir / "dispformer_same_sites_summary.json", summary)
    write_json(args.out_dir / "dispformer_same_sites_protocol.json", protocol)
    existing_summary_path = args.qedisp_dir / "combined_same_sites_summary.json"
    if existing_summary_path.exists():
        existing = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        write_json(args.out_dir / "same_sites_all_methods_summary.json", {**existing, **summary})
    existing_metrics_path = args.qedisp_dir / "combined_same_sites_metrics.csv"
    if existing_metrics_path.exists():
        existing_rows = read_csv_rows(existing_metrics_path)
        all_keys = list(existing_rows[0].keys())
        for key in rows[0].keys():
            if key not in all_keys:
                all_keys.append(key)
        normalized = []
        for row in existing_rows + rows:
            normalized.append({key: row.get(key, "") for key in all_keys})
        with (args.out_dir / "same_sites_all_methods_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(normalized)
        make_all_methods_figure(args.fig_dir / "same_sites_all_methods_vs_mae", normalized)
    np.savez_compressed(
        args.out_dir / "dispformer_same_sites_predictions.npz",
        **arrays,
        raw_disp=csrm["disp"],
        ref_depth=csrm["ref_depth"],
        ref_vs=csrm["ref_vs"],
        loc=csrm["loc"],
    )
    make_summary_figure(args.fig_dir / "dispformer_same_sites_summary", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.out_dir / "dispformer_same_sites_predictions.npz")


if __name__ == "__main__":
    main()
