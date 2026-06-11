#!/usr/bin/env python3
"""Controlled long-period Rayleigh/Love m0 benchmark.

This benchmark uses the current SurfFlow synthetic prior and fixed 2--60 s
period grid.  It compares three input protocols on the same samples:

* Rayleigh m0 phase only
* Love m0 phase only
* Rayleigh+Love m0 phase jointly

SurfFlow is evaluated as a posterior sampler.  The QEDisp comparison is the
local wrapper-level joint inversion that calls the QEDisp Rayleigh/Love forward
executables; it is not native mixed-wave QEDispInv.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import importlib.util
import json
import subprocess
import sys
import time
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
DEFAULT_QEDISP_ROOT = Path(
    "/Users/liuxin/Documents/Backups/Codes/dispersion_curve_calculation/deliverables/"
    "qedispinv_iso_inversion_clean_20260605"
)
DEFAULT_OUT_DIR = ROOT / "results" / "synthetic_m0_joint_controlled"
DEFAULT_FIG_DIR = ROOT / "figures" / "synthetic_m0_joint_controlled"
DEFAULT_STRONG_CKPT = Path("/Volumes/lx_exFAT/yzy_directSWI/code_data/ckpt/fair_di_strong_full_seed642026/best.pt")
DEFAULT_WEAK_CKPT = Path("/Volumes/lx_exFAT/yzy_directSWI/code_data/ckpt/fair_di_weak_full_seed642026/best.pt")

PROTOCOLS = {
    "rayleigh": {"waves": "rayleigh", "mask_rayleigh": True, "mask_love": False},
    "love": {"waves": "love", "mask_rayleigh": False, "mask_love": True},
    "joint": {"waves": "rayleigh,love", "mask_rayleigh": True, "mask_love": True},
}

COLORS = {
    "SurfFlow DI-Strong": "#2f76b7",
    "SurfFlow DI-Weak": "#d07b28",
    "QEDisp wrapper": "#6f6f6f",
    "grid": "#e7ebf0",
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_surfflow_model(checkpoint: Path, device: torch.device):
    transfer = import_from_path("csrm_transfer_for_synth_joint", ROOT / "scripts" / "run_openswi_csrm_transfer.py")
    return transfer.load_model(checkpoint, ROOT / "disp_inv_train.v1.3.py", device)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def generate_synthetic_samples(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = import_from_path("surf_gen_for_synth_joint", ROOT / "utils" / "generate_data.py")
    ds = gen.SurfaceWaveDataset(n_samples=n, z_max_km=150.0, z_max_num=256, dz_km=0.5, seed=seed)
    models, disps = [], []
    for i in range(n):
        model, disp, _mask = ds[i]
        models.append(model.numpy().astype(np.float32))
        disps.append(disp.numpy().astype(np.float32))
    models_np = np.stack(models)
    disps_np = np.stack(disps)
    return models_np, disps_np, models_np[:, 0, :].astype(np.float32)


def selected_period_indices(periods: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.arange(periods.size, dtype=int)
    return np.arange(0, periods.size, stride, dtype=int)


def make_mask(disp: np.ndarray, idx: np.ndarray, protocol: str) -> np.ndarray:
    mask = np.zeros_like(disp, dtype=np.float32)
    cfg = PROTOCOLS[protocol]
    mask[:, 0, idx] = 1.0
    if cfg["mask_rayleigh"]:
        mask[:, 1, idx] = 1.0
    if cfg["mask_love"]:
        mask[:, 2, idx] = 1.0
    return mask


def surfflow_metrics(
    label: str,
    protocol: str,
    posterior: Dict[str, np.ndarray],
    true_models: np.ndarray,
    depth: np.ndarray,
    runtime_s: float,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    med_vs = posterior["median"][:, 1, :]
    q05 = posterior["q05"][:, 1, :]
    q95 = posterior["q95"][:, 1, :]
    true_vs = true_models[:, 2, :]
    err = med_vs - true_vs
    inside = (true_vs >= q05) & (true_vs <= q95)
    shallow = depth <= 40.0
    deep = (depth > 40.0) & (depth <= 120.0)
    row = {
        "method": label,
        "protocol": protocol,
        "n": int(true_vs.shape[0]),
        "runtime_s": float(runtime_s),
        "vs_mae_km_s": float(np.mean(np.abs(err))),
        "vs_rmse_km_s": float(np.sqrt(np.mean(err**2))),
        "vs_bias_km_s": float(np.mean(err)),
        "vs_mae_0_40km_km_s": float(np.mean(np.abs(err[:, shallow]))),
        "vs_mae_40_120km_km_s": float(np.mean(np.abs(err[:, deep]))),
        "reference_in_p05_p95": float(np.mean(inside)),
        "reference_in_p05_p95_0_40km": float(np.mean(inside[:, shallow])),
        "posterior_width_p05_p95_mean_km_s": float(np.mean(q95 - q05)),
    }
    diag = {
        "median_vs": med_vs.astype(np.float32),
        "q05_vs": q05.astype(np.float32),
        "q95_vs": q95.astype(np.float32),
        "err_vs": err.astype(np.float32),
        "inside_p05_p95": inside.astype(np.uint8),
    }
    return row, diag


def write_qedisp_model(path: Path, depth: np.ndarray, profile: np.ndarray, control_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = []
    for j, idx in enumerate(control_indices, start=1):
        z = float(depth[idx])
        vp = float(profile[0, idx])
        vs = float(profile[1, idx])
        rho = float(profile[2, idx])
        arr.append([j, z, rho, vs, vp])
    np.savetxt(path, np.asarray(arr), fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])


def write_picks(path: Path, disp_one: np.ndarray, idx: np.ndarray, protocol: str, sigma_mps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    waves = []
    if PROTOCOLS[protocol]["mask_rayleigh"]:
        waves.append(("rayleigh", "ZZ", 1))
    if PROTOCOLS[protocol]["mask_love"]:
        waves.append(("love", "TT", 2))
    for wave, component, col in waves:
        for k in idx:
            period_s = float(disp_one[0, k])
            observed_mps = float(disp_one[col, k] * 1000.0)
            rows.append(
                {
                    "wave": wave,
                    "component": component,
                    "mode": 0,
                    "frequency_hz": 1.0 / period_s,
                    "qedisp_phase_velocity_mps": observed_mps,
                    "clean_energy_peak_velocity_mps": observed_mps,
                    "picked_phase_velocity_mps": observed_mps,
                    "noise_mps": 0.0,
                    "sigma_mps": sigma_mps,
                    "sigma_ln": sigma_mps / max(observed_mps, 1.0),
                    "energy_peak_db": 0.0,
                    "local_peak_offset_mps": 0.0,
                    "source_psd_power": 1.0,
                    "span_id": f"{wave}_m0_synth",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_qedisp_best_model(path: Path, depth: np.ndarray) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    top = arr[:, 1]
    vs = arr[:, 3]
    return np.interp(depth, top, vs, left=vs[0], right=vs[-1]).astype(np.float32)


def run_qedisp_case(
    script: Path,
    package_root: Path,
    sample_dir: Path,
    model_path: Path,
    reference_path: Path,
    picks_path: Path,
    protocol: str,
    starts: int,
    max_nfev: int,
    seed: int,
) -> Tuple[Path, float]:
    outdir = sample_dir / f"qedisp_{protocol}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--picks",
        str(picks_path),
        "--model",
        str(model_path),
        "--reference-model",
        str(reference_path),
        "--waves",
        PROTOCOLS[protocol]["waves"],
        "--modes",
        "0",
        "--starts",
        str(starts),
        "--max-nfev",
        str(max_nfev),
        "--random-seed",
        str(seed),
        "--outdir",
        str(outdir),
    ]
    tic = time.time()
    proc = subprocess.run(
        cmd,
        cwd=package_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.time() - tic
    (outdir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (outdir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"QEDisp wrapper failed for {outdir}: {proc.stderr[-1000:]}")
    return outdir, elapsed


def run_qedisp_job(job: Dict[str, object]) -> Dict[str, object]:
    outdir, elapsed = run_qedisp_case(
        script=Path(job["script"]),
        package_root=Path(job["package_root"]),
        sample_dir=Path(job["sample_dir"]),
        model_path=Path(job["model_path"]),
        reference_path=Path(job["reference_path"]),
        picks_path=Path(job["picks_path"]),
        protocol=str(job["protocol"]),
        starts=int(job["starts"]),
        max_nfev=int(job["max_nfev"]),
        seed=int(job["seed"]),
    )
    return {
        "protocol": str(job["protocol"]),
        "sample": int(job["sample"]),
        "outdir": str(outdir),
        "elapsed": float(elapsed),
    }


def qedisp_metrics_for_protocol(
    protocol: str,
    true_models: np.ndarray,
    depth: np.ndarray,
    q_vs: np.ndarray,
    runtimes: List[float],
) -> Dict[str, object]:
    true_vs = true_models[:, 2, :]
    err = q_vs - true_vs
    shallow = depth <= 40.0
    deep = (depth > 40.0) & (depth <= 120.0)
    return {
        "method": "QEDisp wrapper",
        "protocol": protocol,
        "n": int(true_vs.shape[0]),
        "runtime_s": float(np.sum(runtimes)),
        "vs_mae_km_s": float(np.mean(np.abs(err))),
        "vs_rmse_km_s": float(np.sqrt(np.mean(err**2))),
        "vs_bias_km_s": float(np.mean(err)),
        "vs_mae_0_40km_km_s": float(np.mean(np.abs(err[:, shallow]))),
        "vs_mae_40_120km_km_s": float(np.mean(np.abs(err[:, deep]))),
        "reference_in_p05_p95": None,
        "reference_in_p05_p95_0_40km": None,
        "posterior_width_p05_p95_mean_km_s": None,
    }


def plot_summary(fig_base: Path, rows: List[Dict[str, object]]) -> None:
    fig_base.parent.mkdir(parents=True, exist_ok=True)
    protocols = ["rayleigh", "love", "joint"]
    methods = ["SurfFlow DI-Strong", "SurfFlow DI-Weak", "QEDisp wrapper"]
    x = np.arange(len(protocols), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.88)
    for j, method in enumerate(methods):
        vals = []
        for protocol in protocols:
            match = [r for r in rows if r["method"] == method and r["protocol"] == protocol]
            vals.append(float(match[0]["vs_mae_km_s"]) if match else np.nan)
        ax.bar(x + (j - 1) * width, vals, width=width, color=COLORS[method], alpha=0.72, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(["Rayleigh", "Love", "Rayleigh+Love"])
    ax.set_ylabel("Vs MAE (km/s)")
    ax.set_title("Controlled synthetic m0 input benchmark", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="upper left")
    fig.savefig(fig_base.with_suffix(".png"), dpi=350)
    fig.savefig(fig_base.with_suffix(".pdf"))
    fig.savefig(fig_base.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--period-stride", type=int, default=2)
    p.add_argument("--posterior-samples", type=int, default=32)
    p.add_argument("--euler-steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--qedisp-starts", type=int, default=8)
    p.add_argument("--qedisp-max-nfev", type=int, default=30)
    p.add_argument("--qedisp-workers", type=int, default=1)
    p.add_argument("--qedisp-root", type=Path, default=DEFAULT_QEDISP_ROOT)
    p.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    p.add_argument("--weak-ckpt", type=Path, default=DEFAULT_WEAK_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--tag", default="pilot_n8_s8")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-qedisp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_dir / args.tag
    fig_dir = args.fig_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    true_models, disp, depth_profiles = generate_synthetic_samples(args.n_samples, args.seed)
    depth = depth_profiles[0].astype(np.float32)
    periods = disp[0, 0]
    period_idx = selected_period_indices(periods, args.period_stride)

    strong_model, strong_ckpt = load_surfflow_model(args.strong_ckpt, device)
    weak_model, weak_ckpt = load_surfflow_model(args.weak_ckpt, device)
    model_depth = strong_ckpt["depth_grid"].detach().cpu().numpy().astype(np.float32)

    rows: List[Dict[str, object]] = []
    arrays: Dict[str, np.ndarray] = {
        "true_models": true_models.astype(np.float32),
        "disp": disp.astype(np.float32),
        "depth": depth.astype(np.float32),
        "period_idx": period_idx.astype(np.int32),
    }
    for protocol in PROTOCOLS:
        mask = make_mask(disp, period_idx, protocol)
        for label, model, seed_offset in [
            ("SurfFlow DI-Strong", strong_model, 1000),
            ("SurfFlow DI-Weak", weak_model, 2000),
        ]:
            tic = time.time()
            transfer = import_from_path("csrm_transfer_metrics_for_synth_joint", ROOT / "scripts" / "run_openswi_csrm_transfer.py")
            posterior = transfer.run_sampler(
                model,
                disp,
                mask,
                device,
                posterior_samples=args.posterior_samples,
                euler_steps=args.euler_steps,
                batch_size=args.batch_size,
                seed=args.seed + seed_offset,
            )
            row, diag = surfflow_metrics(label, protocol, posterior, true_models, depth, time.time() - tic)
            rows.append(row)
            prefix = f"{label.lower().replace(' ', '_').replace('-', '_')}_{protocol}"
            for key, value in diag.items():
                arrays[f"{prefix}_{key}"] = value
        arrays[f"mask_{protocol}"] = mask.astype(np.float32)

    q_script = args.qedisp_root / "joint_rayleigh_love_experiments" / "rayleigh_love_joint_inversion_20260605" / "scripts" / "run_joint_rayleigh_love_multistart.py"
    control_indices = strong_ckpt["control_indices"].detach().cpu().numpy().astype(int)
    reference_profile = strong_ckpt["reference_profile"].detach().cpu().numpy().astype(np.float32)
    ref_model_path = run_dir / "qedisp_reference_prior_mean_model.txt"
    write_qedisp_model(ref_model_path, model_depth, reference_profile, control_indices)

    if not args.skip_qedisp:
        jobs: List[Dict[str, object]] = []
        for protocol in PROTOCOLS:
            for i in range(args.n_samples):
                sample_dir = run_dir / f"sample_{i:03d}"
                picks_path = sample_dir / f"picks_{protocol}.csv"
                model_path = sample_dir / "qedisp_start_prior_mean_model.txt"
                write_qedisp_model(model_path, model_depth, reference_profile, control_indices)
                write_picks(picks_path, disp[i], period_idx, protocol, sigma_mps=20.0)
                jobs.append(
                    {
                        "script": str(q_script),
                        "package_root": str(args.qedisp_root),
                        "sample_dir": str(sample_dir),
                        "model_path": str(model_path),
                        "reference_path": str(ref_model_path),
                        "picks_path": str(picks_path),
                        "protocol": protocol,
                        "sample": i,
                        "starts": args.qedisp_starts,
                        "max_nfev": args.qedisp_max_nfev,
                        "seed": args.seed + i * 17,
                    }
                )

        if args.qedisp_workers > 1:
            results = []
            with cf.ProcessPoolExecutor(max_workers=args.qedisp_workers) as pool:
                for result in pool.map(run_qedisp_job, jobs):
                    results.append(result)
        else:
            results = [run_qedisp_job(job) for job in jobs]

        for protocol in PROTOCOLS:
            protocol_results = sorted(
                [r for r in results if r["protocol"] == protocol],
                key=lambda r: int(r["sample"]),
            )
            q_vs = []
            runtimes = []
            for result in protocol_results:
                outdir = Path(str(result["outdir"]))
                q_vs.append(parse_qedisp_best_model(outdir / "best_model_qedisp_km.txt", depth))
                runtimes.append(float(result["elapsed"]))
            q_vs_arr = np.stack(q_vs).astype(np.float32)
            arrays[f"qedisp_wrapper_{protocol}_vs"] = q_vs_arr
            rows.append(qedisp_metrics_for_protocol(protocol, true_models, depth, q_vs_arr, runtimes))

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "periods_used_s": periods[period_idx].astype(float).tolist(),
        "period_stride": args.period_stride,
        "posterior_samples": args.posterior_samples,
        "euler_steps": args.euler_steps,
        "qedisp_starts": args.qedisp_starts,
        "qedisp_max_nfev": args.qedisp_max_nfev,
        "qedisp_workers": args.qedisp_workers,
        "qedisp_root": str(args.qedisp_root),
        "note": (
            "QEDisp wrapper results use the local wrapper-level Rayleigh/Love joint objective. "
            "They are not native mixed-wave QEDispInv results."
        ),
    }
    write_csv(run_dir / "synthetic_m0_joint_metrics.csv", rows)
    write_json(run_dir / "synthetic_m0_joint_metrics.json", {"protocol": protocol, "rows": rows})
    np.savez_compressed(run_dir / "synthetic_m0_joint_diagnostics.npz", **arrays)
    plot_summary(fig_dir / "synthetic_m0_joint_vs_mae", rows)
    print(json.dumps({"rows": rows, "run_dir": str(run_dir), "fig_dir": str(fig_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
