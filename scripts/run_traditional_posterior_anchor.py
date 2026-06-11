#!/usr/bin/env python3
"""Small traditional posterior anchor for the GJI SurfFlow manuscript.

This script builds an approximate Bayesian reference posterior for a few
synthetic surface-wave inversions using the same local generators and forward
solver as the SurfFlow benchmark.  It deliberately avoids comparing against an
external INN implementation with a different prior or parameterisation.

The reference posterior is computed by prior-predictive importance sampling:

    m_j ~ p_syn(m)
    d_j = F(m_j)
    w_j proportional to exp(-0.5 ||E_q(d_j)-d_obs||^2 / sigma_c^2)

The result is not intended to replace a full MCMC/transdimensional inversion.
It is a compact posterior anchor that tests whether the learned posterior is
being compared with a physically explicit sampler under the same inversion
problem.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "traditional_posterior_anchor"
DEFAULT_FIG_DIR = ROOT / "figures" / "traditional_posterior_anchor"

COLORS = {
    "target": "#111111",
    "strong": "#2f76b7",
    "weak": "#d07b28",
    "best": "#58606b",
    "grid": "#e7ebf0",
}
REGIME_LABELS = {
    "in-prior": "In-prior",
    "boundary": "Near-boundary",
    "out-of-prior": "Out-of-prior",
}


@dataclass
class Case:
    regime: str
    case_index: int
    target: np.ndarray  # [3, H] = vp, vs, rho
    disp: np.ndarray  # [3, T] = period, Rayleigh, Love
    mask: np.ndarray  # [3, T]


@dataclass
class CandidateBank:
    prior: str
    models: np.ndarray  # [N, 3, H]
    disp: np.ndarray  # [N, 3, T]
    runtime_s: float


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def observation_mask(periods: np.ndarray, input_mode: str, period_min: float, period_max: float) -> np.ndarray:
    mask = np.zeros((3, len(periods)), dtype=np.float32)
    ok = (periods >= period_min) & (periods <= period_max)
    mask[0, ok] = 1.0
    if input_mode in {"rayleigh", "joint"}:
        mask[1, ok] = 1.0
    if input_mode in {"love", "joint"}:
        mask[2, ok] = 1.0
    return mask


def compute_dispersion(module, depth: np.ndarray, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, periods: np.ndarray) -> np.ndarray:
    # disba's numba kernels are sensitive to mixed float32/float64 arrays.
    # The training datasets call the forward solver before casting to float32,
    # so we keep the same convention here.
    depth64 = np.asarray(depth, dtype=float)
    vp64 = np.asarray(vp, dtype=float)
    vs64 = np.asarray(vs, dtype=float)
    rho64 = np.asarray(rho, dtype=float)
    periods64 = np.asarray(periods, dtype=float)
    ray = module.compute_phase_dispersion(depth64, vp64, vs64, rho64, periods=periods64, modes=(0,), wave="rayleigh")[0]
    love = module.compute_phase_dispersion(depth64, vp64, vs64, rho64, periods=periods64, modes=(0,), wave="love")[0]
    return np.stack([ray.period.astype(np.float32), ray.velocity.astype(np.float32), love.velocity.astype(np.float32)])


def sample_one(prior: str, strong_mod, weak_mod, rng: np.random.Generator, periods: np.ndarray, n_depth: int) -> Tuple[np.ndarray, np.ndarray]:
    module = strong_mod if prior == "strong" else weak_mod
    if prior == "strong":
        depth, vs, vp, rho, _meta = strong_mod.sample_global_1d_model(z_max_km=150.0, dz_km=0.5, rng=rng)
    elif prior == "weak":
        depth, vs, vp, rho, _meta = weak_mod.sample_weak_prior_1d_model(z_max_km=150.0, dz_km=0.5, rng=rng)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    depth = depth[:n_depth]
    vp = vp[:n_depth]
    vs = vs[:n_depth]
    rho = rho[:n_depth]
    disp = compute_dispersion(module, depth, vp, vs, rho, periods)
    return np.stack([vp, vs, rho]).astype(np.float32), disp.astype(np.float32)


def make_cases(boundary_mod, strong_mod, args: argparse.Namespace) -> List[Case]:
    periods = np.arange(args.period_min, args.period_max + 0.1, args.period_step, dtype=np.float32)
    full_mask = observation_mask(periods, args.input_mode, args.period_min, args.period_max)
    dataset = boundary_mod.strong_dataset(strong_mod, args.cases_per_regime, args.seed + 10)
    in_models, in_disp, _in_mask = boundary_mod.dataset_to_arrays(dataset)
    in_profiles = in_models[:, 1:4, :].astype(np.float32)
    boundary_profiles, boundary_disp, _boundary_mask = boundary_mod.parametric_dataset(
        strong_mod,
        "boundary",
        args.cases_per_regime,
        args.seed + 20,
        in_profiles.shape[-1],
        periods,
    )
    out_profiles, out_disp, _out_mask = boundary_mod.parametric_dataset(
        strong_mod,
        "out-of-prior",
        args.cases_per_regime,
        args.seed + 30,
        in_profiles.shape[-1],
        periods,
    )
    cases: List[Case] = []
    for regime, profiles, disp in [
        ("in-prior", in_profiles, in_disp.astype(np.float32)),
        ("boundary", boundary_profiles.astype(np.float32), boundary_disp.astype(np.float32)),
        ("out-of-prior", out_profiles.astype(np.float32), out_disp.astype(np.float32)),
    ]:
        for i in range(args.cases_per_regime):
            cases.append(Case(regime=regime, case_index=i, target=profiles[i], disp=disp[i], mask=full_mask.copy()))
    return cases


def build_candidate_bank(prior: str, strong_mod, weak_mod, args: argparse.Namespace) -> CandidateBank:
    rng = np.random.default_rng(args.seed + (1000 if prior == "strong" else 2000))
    periods = np.arange(args.period_min, args.period_max + 0.1, args.period_step, dtype=np.float32)
    models: List[np.ndarray] = []
    dispersions: List[np.ndarray] = []
    tries = 0
    tic = time.time()
    while len(models) < args.n_prior_draws and tries < args.n_prior_draws * args.max_tries_factor:
        tries += 1
        try:
            model, disp = sample_one(prior, strong_mod, weak_mod, rng, periods, args.n_depth)
        except Exception:
            continue
        models.append(model)
        dispersions.append(disp)
        if args.progress_every and len(models) % args.progress_every == 0:
            print(f"[info] {prior} prior candidates: {len(models)}/{args.n_prior_draws}")
    if len(models) < args.n_prior_draws:
        raise RuntimeError(f"Only generated {len(models)} valid {prior} candidates after {tries} tries")
    return CandidateBank(prior=prior, models=np.stack(models), disp=np.stack(dispersions), runtime_s=time.time() - tic)


def log_weights(bank: CandidateBank, case: Case, sigma_c: float) -> Tuple[np.ndarray, np.ndarray]:
    wave_mask = case.mask[1:3].astype(bool)
    residual = bank.disp[:, 1:3, :] - case.disp[None, 1:3, :]
    residual_obs = residual[:, wave_mask]
    chi2 = np.sum((residual_obs / sigma_c) ** 2, axis=1)
    logw = -0.5 * (chi2 - np.nanmin(chi2))
    return logw, residual_obs


def normalise_log_weights(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - np.nanmax(logw))
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(w, 1.0 / len(w), dtype=np.float64)
    return (w / total).astype(np.float64)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return np.interp(np.asarray(qs), cdf, v)


def weighted_profile_quantiles(models: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    out = np.empty((len(qs), models.shape[1], models.shape[2]), dtype=np.float32)
    for c in range(models.shape[1]):
        for z in range(models.shape[2]):
            out[:, c, z] = weighted_quantile(models[:, c, z], weights, qs)
    return out


def weighted_disp_quantiles(disp: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    out = np.empty((len(qs), 2, disp.shape[2]), dtype=np.float32)
    for wave in range(2):
        for j in range(disp.shape[2]):
            out[:, wave, j] = weighted_quantile(disp[:, wave + 1, j], weights, qs)
    return out


def weighted_corr(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean[None, :]
    cov = (centered * weights[:, None]).T @ centered
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return cov / np.outer(std, std)


def posterior_weights(bank: CandidateBank, case: Case, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float, str]:
    logw, residual_obs = log_weights(bank, case, args.sigma_c)
    rms = np.sqrt(np.mean(residual_obs**2, axis=1))
    if args.abc_keep > 0:
        k = min(int(args.abc_keep), len(rms))
        idx = np.argpartition(rms, k - 1)[:k]
        weights = np.zeros(len(rms), dtype=np.float64)
        weights[idx] = 1.0 / float(k)
        epsilon = float(np.max(rms[idx]))
        return weights, residual_obs, epsilon, f"abc_top_{k}"
    if args.abc_keep_fraction > 0:
        k = max(1, int(round(float(args.abc_keep_fraction) * len(rms))))
        idx = np.argpartition(rms, k - 1)[:k]
        weights = np.zeros(len(rms), dtype=np.float64)
        weights[idx] = 1.0 / float(k)
        epsilon = float(np.max(rms[idx]))
        return weights, residual_obs, epsilon, f"abc_top_fraction_{args.abc_keep_fraction:g}"
    weights = normalise_log_weights(logw)
    epsilon = float(np.sqrt(np.average(rms**2, weights=weights)))
    return weights, residual_obs, epsilon, "gaussian_importance"


def summarize_case(bank: CandidateBank, case: Case, args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    weights, residual_obs, epsilon, weighting = posterior_weights(bank, case, args)
    ess = 1.0 / np.sum(weights**2)
    best_idx = int(np.argmax(weights))
    qs = weighted_profile_quantiles(bank.models, weights, (0.05, 0.16, 0.50, 0.84, 0.95))
    disp_qs = weighted_disp_quantiles(bank.disp, weights, (0.05, 0.16, 0.50, 0.84, 0.95))
    median = qs[2]
    err_vs = median[1] - case.target[1]
    p05, p95 = qs[0, 1], qs[4, 1]
    p16, p84 = qs[1, 1], qs[3, 1]
    target_in_90 = (case.target[1] >= p05) & (case.target[1] <= p95)
    target_in_68 = (case.target[1] >= p16) & (case.target[1] <= p84)
    best_resid = residual_obs[best_idx]
    key_depths = np.asarray(args.correlation_depths_km, dtype=float)
    depth = np.arange(case.target.shape[-1], dtype=np.float32) * 0.5
    key_idx = np.clip(np.searchsorted(depth, key_depths), 0, len(depth) - 1)
    corr = weighted_corr(bank.models[:, 1, key_idx], weights)
    row: Dict[str, object] = {
        "prior": bank.prior,
        "regime": case.regime,
        "case_index": case.case_index,
        "n_prior_draws": int(len(weights)),
        "sigma_c_km_s": float(args.sigma_c),
        "weighting": weighting,
        "abc_epsilon_rms_km_s": float(epsilon),
        "input_mode": args.input_mode,
        "ess": float(ess),
        "ess_fraction": float(ess / len(weights)),
        "best_disp_rms_km_s": float(np.sqrt(np.mean(best_resid**2))),
        "median_vs_mae_km_s": float(np.mean(np.abs(err_vs))),
        "median_vs_rmse_km_s": float(np.sqrt(np.mean(err_vs**2))),
        "vs_p05_p95_coverage": float(target_in_90.mean()),
        "vs_p16_p84_coverage": float(target_in_68.mean()),
        "mean_vs_p05_p95_width_km_s": float(np.mean(p95 - p05)),
        "mean_vs_p16_p84_width_km_s": float(np.mean(p84 - p16)),
        "candidate_runtime_s": float(bank.runtime_s),
    }
    diag = {
        "weights": weights.astype(np.float32),
        "target": case.target.astype(np.float32),
        "disp": case.disp.astype(np.float32),
        "mask": case.mask.astype(np.float32),
        "profile_qs": qs.astype(np.float32),
        "disp_qs": disp_qs.astype(np.float32),
        "best_model": bank.models[best_idx].astype(np.float32),
        "best_disp": bank.disp[best_idx].astype(np.float32),
        "corr_depths_km": key_depths.astype(np.float32),
        "vs_corr": corr.astype(np.float32),
    }
    return row, diag


def style(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def draw_case_figure(case: Case, prior_diags: Dict[str, Dict[str, np.ndarray]], args: argparse.Namespace) -> Path:
    depth = np.arange(case.target.shape[-1], dtype=np.float32) * 0.5
    period = case.disp[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.1, 1.0, 0.85]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.21, top=0.82, wspace=0.38)
    ax = axes[0]
    obs_ray = case.mask[1] > 0.5
    obs_love = case.mask[2] > 0.5
    if np.any(obs_ray):
        ax.scatter(period[obs_ray], case.disp[1, obs_ray], s=13, color=COLORS["target"], label="Rayleigh obs", zorder=10)
    if np.any(obs_love):
        ax.scatter(period[obs_love], case.disp[2, obs_love], s=13, facecolor="white", edgecolor=COLORS["target"], label="Love obs", zorder=10)
    for prior, diag in prior_diags.items():
        color = COLORS[prior]
        q = diag["disp_qs"]
        ray = q[:, 0]
        if np.any(obs_ray):
            ax.fill_between(period, ray[0], ray[4], color=color, alpha=0.12, lw=0)
            ax.plot(period, ray[2], color=color, lw=1.25, label=f"{prior} Rayleigh")
        if np.any(obs_love):
            love = q[:, 1]
            ax.fill_between(period, love[0], love[4], color=color, alpha=0.06, lw=0)
            ax.plot(period, love[2], color=color, lw=1.05, ls="--", label=f"{prior} Love")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("$c$ (km/s)")
    ax.set_title("Posterior predictive", loc="left", fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    style(ax)

    ax = axes[1]
    ax.plot(case.target[1], depth, color=COLORS["target"], lw=1.3, label="target")
    for prior, diag in prior_diags.items():
        color = COLORS[prior]
        q = diag["profile_qs"][:, 1]
        ax.fill_betweenx(depth, q[0], q[4], color=color, alpha=0.15, lw=0)
        ax.plot(q[2], depth, color=color, lw=1.3, label=f"{prior} median")
    ax.set_ylim(120, 0)
    ax.set_xlabel("$V_S$ (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("$V_S$ marginal posterior", loc="left", fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    style(ax)

    ax = axes[2]
    prior = "weak" if "weak" in prior_diags else next(iter(prior_diags))
    corr = prior_diags[prior]["vs_corr"]
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
    ticks = np.arange(len(args.correlation_depths_km))
    labels = [f"{d:g}" for d in args.correlation_depths_km]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"{prior} $V_S$ correlation", loc="left", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.suptitle(
        f"{REGIME_LABELS.get(case.regime, case.regime)} case {case.case_index}: traditional posterior anchor",
        x=0.075,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"posterior_anchor_{case.regime}_case{case.case_index}_{args.input_mode}"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--n-prior-draws", type=int, default=2048)
    parser.add_argument("--cases-per-regime", type=int, default=1)
    parser.add_argument("--priors", default="strong,weak", help="Comma-separated reference priors: strong,weak")
    parser.add_argument("--input-mode", choices=["rayleigh", "love", "joint"], default="joint")
    parser.add_argument("--sigma-c", type=float, default=0.04, help="Gaussian phase-velocity error in km/s")
    parser.add_argument("--abc-keep", type=int, default=0, help="Use ABC rejection with the K closest prior-predictive particles")
    parser.add_argument(
        "--abc-keep-fraction",
        type=float,
        default=0.0,
        help="Use ABC rejection with this fraction of closest prior-predictive particles",
    )
    parser.add_argument("--period-min", type=float, default=2.0)
    parser.add_argument("--period-max", type=float, default=60.0)
    parser.add_argument("--period-step", type=float, default=1.0)
    parser.add_argument("--n-depth", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--max-tries-factor", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--correlation-depths-km", type=float, nargs="+", default=[10.0, 30.0, 60.0, 100.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    strong_mod = import_from_path("surf_anchor_strong_prior", ROOT / "utils" / "generate_data.py")
    weak_mod = import_from_path("surf_anchor_weak_prior", ROOT / "utils" / "generate_data_weak_prior.py")
    boundary_mod = import_from_path("surf_anchor_boundary", ROOT / "scripts" / "eval_prior_boundary_effect.py")

    priors = [p.strip().lower() for p in args.priors.split(",") if p.strip()]
    for prior in priors:
        if prior not in {"strong", "weak"}:
            raise ValueError(f"Unknown prior in --priors: {prior}")

    print("[info] generating target cases")
    cases = make_cases(boundary_mod, strong_mod, args)
    banks = {}
    for prior in priors:
        print(f"[info] building {prior} prior candidate bank: n={args.n_prior_draws}")
        banks[prior] = build_candidate_bank(prior, strong_mod, weak_mod, args)

    rows: List[Dict[str, object]] = []
    npz_payload: Dict[str, np.ndarray] = {}
    figure_paths: List[str] = []
    for case in cases:
        prior_diags = {}
        for prior, bank in banks.items():
            row, diag = summarize_case(bank, case, args)
            rows.append(row)
            prior_diags[prior] = diag
            prefix = f"{case.regime}_case{case.case_index}_{prior}".replace("-", "_")
            for key, value in diag.items():
                npz_payload[f"{prefix}_{key}"] = value
        figure_paths.append(str(draw_case_figure(case, prior_diags, args)))

    write_csv(args.out_dir / "posterior_anchor_metrics.csv", rows)
    write_json(
        args.out_dir / "posterior_anchor_protocol.json",
        {
            "created_unix_time": time.time(),
            "n_prior_draws": args.n_prior_draws,
            "cases_per_regime": args.cases_per_regime,
            "priors": priors,
            "input_mode": args.input_mode,
            "sigma_c_km_s": args.sigma_c,
            "abc_keep": args.abc_keep,
            "abc_keep_fraction": args.abc_keep_fraction,
            "period_min": args.period_min,
            "period_max": args.period_max,
            "period_step": args.period_step,
            "seed": args.seed,
            "interpretation": (
                "Prior-predictive importance-sampling posterior under the local generator, "
                "surface-wave forward solver, mask and Gaussian phase-velocity error."
            ),
            "figure_paths": figure_paths,
        },
    )
    np.savez_compressed(args.out_dir / "posterior_anchor_diagnostics.npz", **npz_payload)
    print(f"[done] wrote {args.out_dir / 'posterior_anchor_metrics.csv'}")
    print(f"[done] wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
