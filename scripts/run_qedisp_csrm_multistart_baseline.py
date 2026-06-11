#!/usr/bin/env python3
"""Run a small QEDispInv multistart baseline on OpenSWI-real/CSRM sites.

The purpose is a conventional Rayleigh fundamental-mode comparison on the same
observed CSRM phase-velocity curves used by the SurfFlow transfer test.  The
QEDispInv reference model is a smooth regional mean from the CSRM reference
profiles, not a site-specific reference, to avoid giving the optimizer the
answer for each benchmark site.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DATA_DIR = WORKSPACE / "field_data_candidates" / "openswi_real_csrm"
DEFAULT_QEDISP_ROOT = Path(
    "/Users/liuxin/CodexWorkspace/Codes/dispersion_curve_calculation/deliverables/"
    "qedispinv_iso_inversion_clean_20260605"
)
DEFAULT_SURFFLOW_DIAG = ROOT / "results" / "openswi_csrm_transfer" / "n512_s32_step24" / "csrm_transfer_diagnostics.npz"
DEFAULT_OUT_DIR = ROOT / "results" / "qedisp_csrm_multistart"


def brocher_rho(vp: np.ndarray) -> np.ndarray:
    rho = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    return np.clip(rho, 1.8, 3.6)


def smooth(y: np.ndarray, passes: int = 4) -> np.ndarray:
    out = y.astype(float).copy()
    for _ in range(passes):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def write_reference_model(path: Path, data_dir: Path, dz_km: float, zmax_km: float) -> None:
    raw = np.load(data_dir / "obs_depth_vs.npz")["data"].astype(float)
    depth = raw[0, :, 0]
    vs = np.nanmedian(raw[:, :, 1], axis=0)
    target_depth = np.arange(0.0, zmax_km + 0.5 * dz_km, dz_km, dtype=float)
    vs_i = smooth(np.interp(target_depth, depth, vs), passes=5)
    vp_i = np.maximum(vs_i * 1.76, vs_i + 0.25)
    rho_i = brocher_rho(vp_i)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (z, rho, vs0, vp) in enumerate(zip(target_depth, rho_i, vs_i, vp_i), start=1):
            # QEDispInv model columns are: index, top depth, rho, Vs, Vp.
            f.write(f"{i:5d} {z:14.8f} {rho:10.5f} {vs0:10.5f} {vp:10.5f}\n")


def read_csrm(data_dir: Path) -> Dict[str, np.ndarray]:
    return {
        "disp_raw": np.load(data_dir / "obs_period_phase_group.npz")["data"].astype(np.float32),
        "vs_ref_raw": np.load(data_dir / "obs_depth_vs.npz")["data"].astype(np.float32),
        "loc": np.load(data_dir / "obs_depth_vs_loc.npz")["data"].astype(np.float32),
    }


def valid_phase_for_site(raw: np.ndarray, site: int, period_min: float, period_max: float) -> np.ndarray:
    p = raw[site, :, 0]
    c = raw[site, :, 1]
    ok = np.isfinite(p) & np.isfinite(c) & (c > 0.0) & (p >= period_min) & (p <= period_max)
    # QEDispInv data files use frequency in Hz as the first column.  The CSRM
    # file stores period in seconds, so convert here while retaining the same
    # Rayleigh phase-velocity samples used by SurfFlow.
    freq_hz = 1.0 / p[ok]
    arr = np.column_stack([freq_hz, c[ok], np.zeros(int(ok.sum()), dtype=np.float32)])
    if arr.size == 0:
        return arr.reshape(0, 3)
    order = np.argsort(arr[:, 0])
    return arr[order]


def select_sites(
    data_dir: Path,
    surf_flow_diag: Optional[Path],
    max_sites: int,
    seed: int,
    period_min: float,
    period_max: float,
    min_period_count: int,
) -> np.ndarray:
    csrm = read_csrm(data_dir)
    if surf_flow_diag is not None and surf_flow_diag.exists():
        diag = np.load(surf_flow_diag)
        site_index = diag["site_index"].astype(int)
        if "di_strong_err_vs_ref_depth" in diag:
            mae = np.mean(np.abs(diag["di_strong_err_vs_ref_depth"]), axis=1)
            order = np.argsort(mae)
            qs = np.linspace(0.08, 0.92, max_sites)
            chosen = [int(site_index[order[min(len(order) - 1, int(round(q * (len(order) - 1))))]]) for q in qs]
            return np.asarray(sorted(set(chosen)), dtype=int)
        return site_index[:max_sites]

    raw = csrm["disp_raw"]
    valid = []
    for site in range(raw.shape[0]):
        arr = valid_phase_for_site(raw, site, period_min, period_max)
        if len(arr) >= min_period_count:
            valid.append(site)
    rng = np.random.default_rng(seed)
    valid = np.asarray(valid, dtype=int)
    if len(valid) > max_sites:
        valid = np.sort(rng.choice(valid, size=max_sites, replace=False))
    return valid


def write_site_inputs(
    site_dir: Path,
    site: int,
    csrm: Dict[str, np.ndarray],
    period_min: float,
    period_max: float,
    mref_path: Path,
    num_init: int,
    vs_width: float,
    reg_lambda: float,
    zmax_km: float,
) -> Dict[str, Path]:
    site_dir.mkdir(parents=True, exist_ok=True)
    data = valid_phase_for_site(csrm["disp_raw"], site, period_min, period_max)
    data_path = site_dir / "data.txt"
    np.savetxt(data_path, data, fmt="%12.5f %12.5f %4d")
    config_path = site_dir / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[secfunc]",
                f'file_model = "{mref_path}"',
                "nc = 10000",
                "",
                "[forward]",
                f'file_model = "{mref_path}"',
                "fmin = 0.01",
                "fmax = 0.20",
                "nf = 300",
                "",
                "[inversion]",
                'vs2model = "FixVpRho"',
                f'model_ref = "{mref_path}"',
                f"vs_width = {vs_width:.6g}",
                f"lambda = {reg_lambda:.6g}",
                "reg_type = 2",
                f"num_init = {num_init:d}",
                "num_noise = 1",
                "rand_depth = true",
                "rand_vs = false",
                f"zmax = {zmax_km:.6f}",
                "r0 = 0.5",
                "rmin = 1.0",
                "rmax = 1.5",
                "weight = [1.0]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"data": data_path, "config": config_path, "h5": site_dir / "qedispinv.h5"}


def step_value(model: np.ndarray, depth_km: np.ndarray) -> np.ndarray:
    tops = model[:, 1]
    values = model[:, 3]
    idx = np.searchsorted(tops, depth_km, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def analyze_site(site: int, h5_path: Path, csrm: Dict[str, np.ndarray]) -> Dict[str, float | int | str]:
    ref_depth = csrm["vs_ref_raw"][site, :, 0].astype(float)
    ref_vs = csrm["vs_ref_raw"][site, :, 1].astype(float)
    with h5py.File(h5_path, "r") as h5:
        z = h5["z_sample"][...].astype(float)
        vs_median = h5["vs_median"][...].astype(float)
        vs_mean = h5["vs_mean"][...].astype(float)
        valid = (ref_depth >= z.min()) & (ref_depth <= z.max())
        med_on_ref = np.interp(ref_depth[valid], z, vs_median)
        mean_on_ref = np.interp(ref_depth[valid], z, vs_mean)
        ref = ref_vs[valid]

        data = h5["data"][...]
        best_rms = np.nan
        best_mae = np.nan
        n_pred = 0
        n_matched_best = 0
        if "disp" in h5 and len(h5["disp"].keys()) > 0:
            data_lookup = {
                (round(float(row[0]), 8), int(round(float(row[2])))): float(row[1])
                for row in data
            }
            rms_values = []
            mae_values = []
            matched_values = []
            for key in h5["disp"].keys():
                pred = h5["disp"][key][...]
                residuals = []
                for row in pred:
                    obs = data_lookup.get((round(float(row[0]), 8), int(round(float(row[2])))))
                    if obs is not None:
                        residuals.append(float(row[1]) - obs)
                if not residuals:
                    continue
                residual = np.asarray(residuals, dtype=float)
                rms_values.append(float(np.sqrt(np.mean(residual**2))))
                mae_values.append(float(np.mean(np.abs(residual))))
                matched_values.append(int(len(residual)))
            if rms_values:
                n_pred = len(rms_values)
                best_i = int(np.argmin(rms_values))
                best_rms = float(rms_values[best_i])
                best_mae = float(mae_values[best_i])
                n_matched_best = int(matched_values[best_i])
        return {
            "site_index": int(site),
            "lon": float(csrm["loc"][site, 0]),
            "lat": float(csrm["loc"][site, 1]),
            "status": "ok",
            "num_valid": int(np.asarray(h5["num_valid"])),
            "num_pred": int(n_pred),
            "niter_median": float(np.median(h5["niter"][...])) if "niter" in h5 and len(h5["niter"]) else np.nan,
            "fitness_min": float(np.min(h5["fitness"][...])) if "fitness" in h5 and len(h5["fitness"]) else np.nan,
            "disp_best_rms_km_s": best_rms,
            "disp_best_mae_km_s": best_mae,
            "disp_best_matched_points": n_matched_best,
            "vs_median_mae_km_s": float(np.mean(np.abs(med_on_ref - ref))),
            "vs_mean_mae_km_s": float(np.mean(np.abs(mean_on_ref - ref))),
            "vs_median_bias_km_s": float(np.mean(med_on_ref - ref)),
            "z_sample_max_km": float(z.max()),
        }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_combined(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        vals = [row for row in rows if row["method"] == method and row.get("status") == "ok"]
        out: Dict[str, object] = {"n": len(vals)}
        for key in ["vs_mae_km_s", "disp_best_rms_km_s", "reference_in_p05_p95"]:
            arr = np.asarray(
                [
                    np.nan
                    if row.get(key) in (None, "", "nan")
                    else float(row.get(key, np.nan))
                    for row in vals
                ],
                dtype=float,
            )
            if np.all(~np.isfinite(arr)):
                out[f"mean_{key}"] = None
                out[f"median_{key}"] = None
                out[f"p25_{key}"] = None
                out[f"p75_{key}"] = None
            else:
                out[f"mean_{key}"] = float(np.nanmean(arr))
                out[f"median_{key}"] = float(np.nanmedian(arr))
                out[f"p25_{key}"] = float(np.nanpercentile(arr, 25))
                out[f"p75_{key}"] = float(np.nanpercentile(arr, 75))
        summary[method] = out
    return summary


def write_combined_same_site_outputs(out_dir: Path, qedisp_rows: List[Dict[str, object]], surf_diag: Path) -> None:
    if surf_diag is None or not surf_diag.exists():
        return
    diag = np.load(surf_diag)
    site_index = diag["site_index"].astype(int)
    site_to_row = {int(site): i for i, site in enumerate(site_index)}
    combined: List[Dict[str, object]] = []
    for row in qedisp_rows:
        if row.get("status") != "ok":
            continue
        site = int(row["site_index"])
        if site not in site_to_row:
            continue
        combined.append(
            {
                "site_index": site,
                "method": "QEDisp m0 multistart",
                "vs_mae_km_s": float(row["vs_median_mae_km_s"]),
                "disp_best_rms_km_s": float(row["disp_best_rms_km_s"]),
                "reference_in_p05_p95": np.nan,
                "status": "ok",
            }
        )
        i = site_to_row[site]
        for label, prefix in [("DI-Strong", "di_strong"), ("DI-Weak", "di_weak")]:
            combined.append(
                {
                    "site_index": site,
                    "method": label,
                    "vs_mae_km_s": float(np.mean(np.abs(diag[f"{prefix}_err_vs_ref_depth"][i]))),
                    "disp_best_rms_km_s": np.nan,
                    "reference_in_p05_p95": float(np.mean(diag[f"{prefix}_inside_p05_p95"][i])),
                    "status": "ok",
                }
            )
    if not combined:
        return
    write_csv(out_dir / "combined_same_sites_metrics.csv", combined)
    (out_dir / "combined_same_sites_summary.json").write_text(
        json.dumps(summarize_combined(combined), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--qedisp-root", type=Path, default=DEFAULT_QEDISP_ROOT)
    parser.add_argument("--surf-flow-diagnostics", type=Path, default=DEFAULT_SURFFLOW_DIAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="sites6_starts16")
    parser.add_argument("--max-sites", type=int, default=6)
    parser.add_argument("--period-min", type=float, default=8.0)
    parser.add_argument("--period-max", type=float, default=60.0)
    parser.add_argument("--min-period-count", type=int, default=10)
    parser.add_argument("--num-init", type=int, default=16)
    parser.add_argument("--vs-width", type=float, default=1.2)
    parser.add_argument("--lambda-reg", type=float, default=1.0e-2)
    parser.add_argument("--zmax-km", type=float, default=120.0)
    parser.add_argument("--ref-dz-km", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csrm = read_csrm(args.data_dir)
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mref_path = out_dir / "regional_mean_mref.txt"
    write_reference_model(mref_path, args.data_dir, dz_km=args.ref_dz_km, zmax_km=args.zmax_km)
    sites = select_sites(
        args.data_dir,
        args.surf_flow_diagnostics,
        max_sites=args.max_sites,
        seed=args.seed,
        period_min=args.period_min,
        period_max=args.period_max,
        min_period_count=args.min_period_count,
    )
    rows: List[Dict[str, object]] = []
    binary = args.qedisp_root / "bin" / "qedispinv_inversion"
    if not binary.exists():
        raise FileNotFoundError(f"Missing QEDispInv binary: {binary}")

    for site in sites:
        site_dir = out_dir / f"site_{site:05d}"
        paths = write_site_inputs(
            site_dir,
            int(site),
            csrm,
            period_min=args.period_min,
            period_max=args.period_max,
            mref_path=mref_path,
            num_init=args.num_init,
            vs_width=args.vs_width,
            reg_lambda=args.lambda_reg,
            zmax_km=args.zmax_km,
        )
        tic = time.time()
        status = "ok"
        stderr = ""
        if not args.skip_run:
            proc = subprocess.run(
                [str(binary), "-c", str(paths["config"]), "-d", str(paths["data"]), "-o", str(paths["h5"])],
                cwd=args.qedisp_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                status = f"failed:{proc.returncode}"
                stderr = proc.stderr.strip()
                (site_dir / "qedisp_stdout.txt").write_text(proc.stdout, encoding="utf-8")
                (site_dir / "qedisp_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if paths["h5"].exists() and status == "ok":
            row = analyze_site(int(site), paths["h5"], csrm)
        else:
            row = {
                "site_index": int(site),
                "lon": float(csrm["loc"][site, 0]),
                "lat": float(csrm["loc"][site, 1]),
                "status": status,
                "stderr": stderr[:500],
            }
        row.update(
            {
                "runtime_s": float(time.time() - tic),
                "num_init_requested": int(args.num_init),
                "vs_width": float(args.vs_width),
                "lambda_reg": float(args.lambda_reg),
            }
        )
        rows.append(row)
        write_csv(out_dir / "qedisp_csrm_site_metrics_partial.csv", rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "QEDispInv Rayleigh m0 multistart baseline on OpenSWI-real/CSRM.",
        "data_dir": str(args.data_dir),
        "qedisp_root": str(args.qedisp_root),
        "reference_model": str(mref_path),
        "site_indices": [int(s) for s in sites],
        "num_init": int(args.num_init),
        "period_min_s": float(args.period_min),
        "period_max_s": float(args.period_max),
        "notes": [
            "Only Rayleigh phase velocity is used; mode is set to 0.",
            "QEDispInv data column 1 is frequency in Hz, converted from CSRM periods.",
            "The reference model is a regional mean CSRM profile, not a site-specific target.",
            "QEDispInv ensemble intervals are multistart ensembles, not calibrated posterior probabilities.",
        ],
    }
    ok = [row for row in rows if row.get("status") == "ok"]
    if ok:
        summary.update(
            {
                "ok_sites": len(ok),
                "median_vs_median_mae_km_s": float(np.median([float(r["vs_median_mae_km_s"]) for r in ok])),
                "mean_vs_median_mae_km_s": float(np.mean([float(r["vs_median_mae_km_s"]) for r in ok])),
                "median_best_disp_rms_km_s": float(np.median([float(r["disp_best_rms_km_s"]) for r in ok])),
            }
        )
    write_csv(out_dir / "qedisp_csrm_site_metrics.csv", rows)
    (out_dir / "qedisp_csrm_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_combined_same_site_outputs(out_dir, rows, args.surf_flow_diagnostics)
    print(json.dumps({"summary": summary, "out_dir": str(out_dir), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
