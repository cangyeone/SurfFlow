#!/usr/bin/env python3
"""Rayleigh-only, Love-only and joint-input ablation for SurfFlow DI samplers."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 8.0
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EXTERNAL_CKPT = Path("/Volumes/lx_exFAT/yzy_directSWI/code_data/ckpt")
DEFAULT_OUT_DIR = ROOT / "results" / "rayleigh_love_input_ablation"
DEFAULT_FIG_DIR = ROOT / "figures" / "rayleigh_love_input_ablation"
DEFAULT_STRONG_CKPT = EXTERNAL_CKPT / "fair_di_strong_full_seed642026" / "best.pt"
DEFAULT_WEAK_CKPT = EXTERNAL_CKPT / "fair_di_weak_full_seed642026" / "best.pt"

METHODS = {
    "DI-Strong": {"prior": "strong", "color": "#2f76b7"},
    "DI-Weak": {"prior": "weak", "color": "#d07b28"},
}
INPUT_MODES = ("rayleigh", "love", "joint")
MODE_LABELS = {"rayleigh": "Rayleigh", "love": "Love", "joint": "Rayleigh+Love"}


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


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dataset_to_arrays(dataset) -> Tuple[np.ndarray, np.ndarray]:
    models, dispersions = [], []
    for i in range(len(dataset)):
        model, disp, _mask = dataset[i]
        models.append(model.numpy()[1:4].astype(np.float32))
        dispersions.append(disp.numpy().astype(np.float32))
    return np.stack(models), np.stack(dispersions)


def make_full_mask(disp: np.ndarray, mode: str, period_min: float, period_max: float) -> np.ndarray:
    periods = disp[:, 0, :]
    mask = np.zeros_like(disp, dtype=np.float32)
    in_window = (periods >= period_min) & (periods <= period_max)
    mask[:, 0, :] = in_window.astype(np.float32)
    if mode in {"rayleigh", "joint"}:
        mask[:, 1, :] = in_window.astype(np.float32)
    if mode in {"love", "joint"}:
        mask[:, 2, :] = in_window.astype(np.float32)
    return mask


def make_test_data(prior: str, n: int, seed: int, strong_mod, weak_mod) -> Tuple[np.ndarray, np.ndarray]:
    module = strong_mod if prior == "strong" else weak_mod
    dataset = module.SurfaceWaveDataset(n_samples=n, z_max_km=150.0, z_max_num=256, dz_km=0.5, seed=seed)
    return dataset_to_arrays(dataset)


def summarize(samples: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    median = np.median(samples, axis=1)
    q16 = np.quantile(samples, 0.16, axis=1)
    q84 = np.quantile(samples, 0.84, axis=1)
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    vs_err = median[:, 1, :] - target[:, 1, :]
    return {
        "vp_mae_km_s": float(np.mean(np.abs(median[:, 0, :] - target[:, 0, :]))),
        "vs_mae_km_s": float(np.mean(np.abs(vs_err))),
        "rho_mae_g_cm3": float(np.mean(np.abs(median[:, 2, :] - target[:, 2, :]))),
        "vs_rmse_km_s": float(np.sqrt(np.mean(vs_err**2))),
        "vs_16_84_coverage": float(((target[:, 1, :] >= q16[:, 1, :]) & (target[:, 1, :] <= q84[:, 1, :])).mean()),
        "vs_05_95_coverage": float(((target[:, 1, :] >= q05[:, 1, :]) & (target[:, 1, :] <= q95[:, 1, :])).mean()),
        "mean_vs_std_km_s": float(np.mean(np.std(samples[:, :, 1, :], axis=1))),
        "mean_vs_p16_p84_width_km_s": float(np.mean(q84[:, 1, :] - q16[:, 1, :])),
        "mean_vs_p05_p95_width_km_s": float(np.mean(q95[:, 1, :] - q05[:, 1, :])),
    }


def plot_summary(rows: List[Dict[str, object]], fig_dir: Path) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("vs_mae_km_s", "$V_S$ MAE (km/s)"),
        ("vs_16_84_coverage", "$V_S$ 16-84% coverage"),
        ("mean_vs_std_km_s", "mean $V_S$ posterior std (km/s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.83, wspace=0.34)
    width = 0.34
    x = np.arange(len(INPUT_MODES))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for j, method in enumerate(METHODS):
            vals = [
                float(next(r[metric] for r in rows if r["method"] == method and r["input_mode"] == mode))
                for mode in INPUT_MODES
            ]
            ax.bar(
                x + (j - 0.5) * width,
                vals,
                width=width,
                color=METHODS[method]["color"],
                alpha=0.82,
                label=method,
            )
        if metric == "vs_16_84_coverage":
            ax.axhline(0.68, color="0.25", ls=":", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in INPUT_MODES], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e7ebf0", lw=0.6)
        ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left")
    fig.suptitle("Input-channel ablation on matched synthetic examples", x=0.08, ha="left", fontsize=9.7, fontweight="bold")
    out = fig_dir / "rayleigh_love_input_ablation"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    parser.add_argument("--weak-ckpt", type=Path, default=DEFAULT_WEAK_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--euler-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--period-min", type=float, default=2.0)
    parser.add_argument("--period-max", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=642026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", default="DI-Strong,DI-Weak")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    strong_mod = import_from_path("rl_ablation_strong_data", ROOT / "utils" / "generate_data.py")
    weak_mod = import_from_path("rl_ablation_weak_data", ROOT / "utils" / "generate_data_weak_prior.py")
    boundary_mod = import_from_path("rl_ablation_boundary_helpers", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ckpts = {"DI-Strong": args.strong_ckpt, "DI-Weak": args.weak_ckpt}
    rows: List[Dict[str, object]] = []
    protocol: Dict[str, object] = {
        "created_unix_time": time.time(),
        "device": str(device),
        "n_test": args.n_test,
        "posterior_samples": args.posterior_samples,
        "euler_steps": args.euler_steps,
        "period_min": args.period_min,
        "period_max": args.period_max,
        "seed": args.seed,
        "checkpoints": {method: str(ckpts[method]) for method in methods},
    }
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        ckpt = ckpts[method]
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        prior = METHODS[method]["prior"]
        print(f"[info] loading {method}: {ckpt}")
        model, _cfg = boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", ckpt, device)
        if model is None:
            raise RuntimeError(f"Could not load {method}")
        target, disp = make_test_data(prior, args.n_test, args.seed + (10 if prior == "strong" else 20), strong_mod, weak_mod)
        for mode in INPUT_MODES:
            mask = make_full_mask(disp, mode, args.period_min, args.period_max)
            tic = time.time()
            print(f"[info] {method} {mode}: sampling n={args.n_test}, samples={args.posterior_samples}")
            samples = boundary_mod.direct_samples(
                model,
                disp.astype(np.float32),
                mask.astype(np.float32),
                device,
                n_samples=args.posterior_samples,
                steps=args.euler_steps,
                batch_size=args.batch_size,
            )
            row: Dict[str, object] = {
                "method": method,
                "prior": prior,
                "input_mode": mode,
                "n_test": args.n_test,
                "posterior_samples": args.posterior_samples,
                "euler_steps": args.euler_steps,
                "runtime_s": float(time.time() - tic),
                "period_min": args.period_min,
                "period_max": args.period_max,
            }
            row.update(summarize(samples, target))
            rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "rayleigh_love_input_ablation_metrics.csv", rows)
    fig_path = plot_summary(rows, args.fig_dir)
    protocol["figure"] = str(fig_path)
    write_json(args.out_dir / "rayleigh_love_input_ablation_protocol.json", protocol)
    print(f"[done] wrote {args.out_dir / 'rayleigh_love_input_ablation_metrics.csv'}")
    print(f"[done] wrote {fig_path}")


if __name__ == "__main__":
    main()
