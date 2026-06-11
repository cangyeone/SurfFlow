#!/usr/bin/env python3
"""Small MCMC posterior reference for the GJI SurfFlow manuscript.

The goal is not to build a production Bayesian inversion package.  It is to
check whether a transparent, low-dimensional MCMC posterior is usable as a
reference for a few synthetic cases before putting it in the paper.

The MCMC state is a Vs-only spline/control profile plus a global Vp/Vs ratio.
Vp and density are derived from Vs and Vp using a Brocher-style relation.  This
keeps the state low-dimensional enough for trace and R-hat diagnostics to mean
something.  If the diagnostics fail, the output should be treated as a failed
reference posterior rather than a comparison result.
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
matplotlib.rcParams["font.size"] = 7.0
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_CKPT = Path("/Volumes/lx_exFAT/yzy_directSWI/code_data/ckpt")
DEFAULT_STRONG_CKPT = EXTERNAL_CKPT / "fair_di_strong_full_seed642026" / "best.pt"
DEFAULT_WEAK_CKPT = EXTERNAL_CKPT / "fair_di_weak_full_seed642026" / "best.pt"
DEFAULT_OUT_DIR = ROOT / "results" / "mcmc_posterior_reference"
DEFAULT_FIG_DIR = ROOT / "figures" / "mcmc_posterior_reference"

COLORS = {
    "target": "#111111",
    "mcmc": "#4a8f5a",
    "di_strong": "#2f76b7",
    "di_weak": "#d07b28",
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
    target: np.ndarray
    disp: np.ndarray
    mask: np.ndarray


@dataclass
class MCMCResult:
    samples: np.ndarray  # [chains, draws, n_params]
    logp: np.ndarray
    accept_rate: np.ndarray
    step_scale: float
    runtime_s: float


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


def style(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def observation_mask(periods: np.ndarray, input_mode: str, period_min: float, period_max: float) -> np.ndarray:
    mask = np.zeros((3, len(periods)), dtype=np.float32)
    ok = (periods >= period_min) & (periods <= period_max)
    mask[0, ok] = 1.0
    if input_mode in {"rayleigh", "joint"}:
        mask[1, ok] = 1.0
    if input_mode in {"love", "joint"}:
        mask[2, ok] = 1.0
    return mask


def compute_dispersion(strong_mod, depth: np.ndarray, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, periods: np.ndarray):
    depth64 = np.asarray(depth, dtype=float)
    vp64 = np.asarray(vp, dtype=float)
    vs64 = np.asarray(vs, dtype=float)
    rho64 = np.asarray(rho, dtype=float)
    periods64 = np.asarray(periods, dtype=float)
    ray = strong_mod.compute_phase_dispersion(depth64, vp64, vs64, rho64, periods=periods64, modes=(0,), wave="rayleigh")[0]
    love = strong_mod.compute_phase_dispersion(depth64, vp64, vs64, rho64, periods=periods64, modes=(0,), wave="love")[0]
    disp = np.stack([ray.period.astype(np.float32), ray.velocity.astype(np.float32), love.velocity.astype(np.float32)])
    if not np.all(np.isfinite(disp)):
        raise ValueError("non-finite dispersion")
    return disp


def make_cases(boundary_mod, strong_mod, args: argparse.Namespace) -> List[Case]:
    periods = np.arange(args.period_min, args.period_max + 0.1, args.period_step, dtype=np.float32)
    full_mask = observation_mask(periods, args.input_mode, args.period_min, args.period_max)
    dataset = boundary_mod.strong_dataset(strong_mod, args.cases_per_regime, args.seed + 10)
    in_models, in_disp, _ = boundary_mod.dataset_to_arrays(dataset)
    in_profiles = in_models[:, 1:4, :].astype(np.float32)
    boundary_profiles, boundary_disp, _ = boundary_mod.parametric_dataset(
        strong_mod, "boundary", args.cases_per_regime, args.seed + 20, in_profiles.shape[-1], periods
    )
    out_profiles, out_disp, _ = boundary_mod.parametric_dataset(
        strong_mod, "out-of-prior", args.cases_per_regime, args.seed + 30, in_profiles.shape[-1], periods
    )
    cases = []
    for regime, profiles, disp in [
        ("in-prior", in_profiles, in_disp.astype(np.float32)),
        ("boundary", boundary_profiles.astype(np.float32), boundary_disp.astype(np.float32)),
        ("out-of-prior", out_profiles.astype(np.float32), out_disp.astype(np.float32)),
    ]:
        for i in range(args.cases_per_regime):
            cases.append(Case(regime=regime, case_index=i, target=profiles[i], disp=disp[i], mask=full_mask.copy()))
    if args.regimes != "all":
        keep = {r.strip() for r in args.regimes.split(",") if r.strip()}
        cases = [c for c in cases if c.regime in keep]
    return cases


def brocher_rho_from_vp(vp: np.ndarray) -> np.ndarray:
    rho = 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
    return np.clip(rho, 1.2, 3.8).astype(np.float32)


def reflect_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    out = x.copy()
    width = hi - lo
    for _ in range(3):
        below = out < lo
        out[below] = lo[below] + (lo[below] - out[below])
        above = out > hi
        out[above] = hi[above] - (out[above] - hi[above])
    return np.clip(out, lo + 1e-6 * width, hi - 1e-6 * width)


def knot_bounds(knot_depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.empty(len(knot_depths), dtype=np.float32)
    hi = np.empty(len(knot_depths), dtype=np.float32)
    for i, z in enumerate(knot_depths):
        if z <= 5.0:
            lo[i], hi[i] = 0.25, 3.2
        elif z <= 15.0:
            lo[i], hi[i] = 1.0, 4.2
        elif z <= 50.0:
            lo[i], hi[i] = 2.2, 4.8
        else:
            lo[i], hi[i] = 3.0, 5.8
    return lo, hi


def params_to_profile(params: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vs_knots = params[:-1]
    vpvs = float(params[-1])
    vs = np.interp(depth, knot_depths, vs_knots).astype(np.float32)
    vs = np.clip(vs, 0.15, 6.0)
    vp = np.maximum(vs * vpvs, vs + 0.25).astype(np.float32)
    rho = brocher_rho_from_vp(vp)
    return vp, vs, rho


def log_smooth_prior(vs_knots: np.ndarray, smooth_sigma: float) -> float:
    if len(vs_knots) < 3:
        return 0.0
    d2 = vs_knots[2:] - 2.0 * vs_knots[1:-1] + vs_knots[:-2]
    return float(-0.5 * np.sum((d2 / smooth_sigma) ** 2))


def log_posterior(
    params: np.ndarray,
    case: Case,
    strong_mod,
    depth: np.ndarray,
    knot_depths: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    if np.any(params < lo) or np.any(params > hi) or not np.all(np.isfinite(params)):
        return -np.inf, np.inf
    vs_knots = params[:-1]
    vpvs = float(params[-1])
    logp = log_smooth_prior(vs_knots, args.smooth_sigma)
    logp += -0.5 * ((vpvs - args.vpvs_mean) / args.vpvs_sigma) ** 2
    try:
        vp, vs, rho = params_to_profile(params, depth, knot_depths)
        pred = compute_dispersion(strong_mod, depth, vp, vs, rho, case.disp[0])
    except Exception:
        return -np.inf, np.inf
    wave_mask = case.mask[1:3].astype(bool)
    residual = (pred[1:3] - case.disp[1:3])[wave_mask]
    rms = float(np.sqrt(np.mean(residual**2)))
    loglike = -0.5 * float(np.sum((residual / args.sigma_c) ** 2))
    return logp + loglike, rms


def random_initial_points(
    rng: np.random.Generator,
    n: int,
    lo: np.ndarray,
    hi: np.ndarray,
    knot_depths: np.ndarray,
) -> np.ndarray:
    params = rng.uniform(lo, hi, size=(n, len(lo))).astype(np.float32)
    # Slightly prefer increasing-with-depth structures while keeping broad support.
    for i in range(n):
        if rng.random() < 0.6:
            anchors = np.linspace(params[i, 0], params[i, -2], len(knot_depths))
            params[i, :-1] = 0.55 * params[i, :-1] + 0.45 * anchors
    params[:, -1] = rng.normal(1.80, 0.06, size=n).clip(lo[-1], hi[-1])
    return params


def select_starts(
    case: Case,
    strong_mod,
    depth: np.ndarray,
    knot_depths: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    draws = random_initial_points(rng, args.n_init, lo, hi, knot_depths)
    scored = []
    for i, x in enumerate(draws):
        lp, rms = log_posterior(x, case, strong_mod, depth, knot_depths, lo, hi, args)
        if np.isfinite(lp):
            scored.append((lp, rms, i))
    if len(scored) < args.n_chains:
        raise RuntimeError(f"Only {len(scored)} finite initial points for {case.regime} case {case.case_index}")
    scored.sort(reverse=True, key=lambda t: t[0])
    starts = []
    for _, _, i in scored[: args.n_chains]:
        starts.append(draws[i])
    return np.stack(starts).astype(np.float32)


def run_mcmc(
    case: Case,
    strong_mod,
    depth: np.ndarray,
    knot_depths: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    starts: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> MCMCResult:
    tic = time.time()
    n_params = len(lo)
    samples = np.empty((args.n_chains, args.n_steps, n_params), dtype=np.float32)
    logps = np.empty((args.n_chains, args.n_steps), dtype=np.float32)
    accepted = np.zeros(args.n_chains, dtype=np.int64)
    params = starts.copy()
    current = np.array(
        [log_posterior(p, case, strong_mod, depth, knot_depths, lo, hi, args)[0] for p in params],
        dtype=np.float64,
    )
    proposal = np.full(n_params, args.proposal_vs, dtype=np.float32)
    proposal[-1] = args.proposal_vpvs
    step_scale = 1.0
    window_accept = np.zeros(args.n_chains, dtype=np.int64)
    window_count = 0
    for step in range(args.n_steps):
        for c in range(args.n_chains):
            prop = params[c] + rng.normal(0.0, proposal * step_scale, size=n_params)
            prop = reflect_bounds(prop.astype(np.float32), lo, hi)
            prop_lp, _ = log_posterior(prop, case, strong_mod, depth, knot_depths, lo, hi, args)
            if np.isfinite(prop_lp) and math.log(rng.random()) < prop_lp - current[c]:
                params[c] = prop
                current[c] = prop_lp
                accepted[c] += 1
                window_accept[c] += 1
        samples[:, step] = params
        logps[:, step] = current
        if args.adapt and step < args.burnin:
            window_count += 1
            if window_count == args.adapt_window:
                rate = float(window_accept.sum() / max(args.n_chains * window_count, 1))
                if rate < 0.15:
                    step_scale *= 0.80
                elif rate > 0.40:
                    step_scale *= 1.20
                step_scale = float(np.clip(step_scale, 0.15, 4.0))
                window_accept[:] = 0
                window_count = 0
        if args.progress_every and (step + 1) % args.progress_every == 0:
            rate = accepted.sum() / float(args.n_chains * (step + 1))
            print(
                f"[mcmc] {case.regime} case {case.case_index}: step {step + 1}/{args.n_steps}, "
                f"accept={rate:.3f}, best_logp={np.max(current):.1f}"
            )
    return MCMCResult(
        samples=samples,
        logp=logps,
        accept_rate=accepted / float(args.n_steps),
        step_scale=step_scale,
        runtime_s=time.time() - tic,
    )


def sample_stretch_z(rng: np.random.Generator, a: float) -> float:
    lo = 1.0 / math.sqrt(a)
    hi = math.sqrt(a)
    return float((lo + (hi - lo) * rng.random()) ** 2)


def run_stretch_mcmc(
    case: Case,
    strong_mod,
    depth: np.ndarray,
    knot_depths: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    starts: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> MCMCResult:
    tic = time.time()
    n_walkers, n_params = starts.shape
    samples = np.empty((n_walkers, args.n_steps, n_params), dtype=np.float32)
    logps = np.empty((n_walkers, args.n_steps), dtype=np.float32)
    accepted = np.zeros(n_walkers, dtype=np.int64)
    params = starts.copy()
    current = np.array(
        [log_posterior(p, case, strong_mod, depth, knot_depths, lo, hi, args)[0] for p in params],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(current)):
        raise RuntimeError("Stretch sampler received non-finite initial walkers")
    for step in range(args.n_steps):
        order = rng.permutation(n_walkers)
        for i in order:
            candidates = np.delete(np.arange(n_walkers), i)
            j = int(rng.choice(candidates))
            z = sample_stretch_z(rng, args.stretch_a)
            prop = params[j] + z * (params[i] - params[j])
            if np.any(prop < lo) or np.any(prop > hi):
                samples[i, step] = params[i]
                logps[i, step] = current[i]
                continue
            prop_lp, _ = log_posterior(prop.astype(np.float32), case, strong_mod, depth, knot_depths, lo, hi, args)
            log_alpha = (n_params - 1) * math.log(z) + prop_lp - current[i]
            if np.isfinite(prop_lp) and math.log(rng.random()) < log_alpha:
                params[i] = prop.astype(np.float32)
                current[i] = prop_lp
                accepted[i] += 1
        samples[:, step] = params
        logps[:, step] = current
        if args.progress_every and (step + 1) % args.progress_every == 0:
            rate = accepted.sum() / float(n_walkers * (step + 1))
            print(
                f"[mcmc] {case.regime} case {case.case_index}: step {step + 1}/{args.n_steps}, "
                f"accept={rate:.3f}, best_logp={np.max(current):.1f}"
            )
    return MCMCResult(
        samples=samples,
        logp=logps,
        accept_rate=accepted / float(args.n_steps),
        step_scale=float(args.stretch_a),
        runtime_s=time.time() - tic,
    )


def posterior_draws(result: MCMCResult, burnin: int, thin: int) -> np.ndarray:
    return result.samples[:, burnin::thin, :]


def split_rhat(draws: np.ndarray) -> np.ndarray:
    chains, n, p = draws.shape
    if n < 4:
        return np.full(p, np.nan)
    half = n // 2
    split = np.concatenate([draws[:, :half], draws[:, -half:]], axis=0)
    m, n2, _ = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    b = n2 * chain_means.var(axis=0, ddof=1)
    w = chain_vars.mean(axis=0)
    var_hat = ((n2 - 1) / n2) * w + b / n2
    return np.sqrt(var_hat / np.maximum(w, 1e-12))


def autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return out


def ess_per_param(draws: np.ndarray) -> np.ndarray:
    chains, n, p = draws.shape
    merged = draws.reshape(chains * n, p)
    max_lag = min(200, max(1, chains * n // 4))
    ess = np.empty(p, dtype=np.float64)
    for j in range(p):
        ac = autocorr_1d(merged[:, j], max_lag)
        positive = ac[1:]
        cutoff = len(positive)
        for k, val in enumerate(positive):
            if val < 0:
                cutoff = k
                break
        tau = 1.0 + 2.0 * float(np.sum(positive[:cutoff]))
        ess[j] = chains * n / max(tau, 1.0)
    return ess


def sample_profiles(params_draws: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> np.ndarray:
    flat = params_draws.reshape(-1, params_draws.shape[-1])
    profiles = []
    for p in flat:
        vp, vs, rho = params_to_profile(p, depth, knot_depths)
        profiles.append(np.stack([vp, vs, rho]).astype(np.float32))
    return np.stack(profiles)


def predictive_quantiles(strong_mod, profiles: np.ndarray, periods: np.ndarray, max_samples: int) -> Tuple[np.ndarray, float, int]:
    if len(profiles) > max_samples:
        idx = np.linspace(0, len(profiles) - 1, max_samples, dtype=int)
        use_profiles = profiles[idx]
    else:
        use_profiles = profiles
    depth = np.arange(profiles.shape[-1], dtype=np.float32) * 0.5
    preds = []
    for profile in use_profiles:
        try:
            preds.append(compute_dispersion(strong_mod, depth, profile[0], profile[1], profile[2], periods)[1:3])
        except Exception:
            continue
    if not preds:
        return np.full((5, 2, len(periods)), np.nan, dtype=np.float32), np.inf, 0
    stack = np.stack(preds)
    qs = np.quantile(stack, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
    return qs, 0.0, len(preds)


def summarize_profiles(samples: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    qs = np.quantile(samples, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    err_vs = qs[2, 1] - target[1]
    return {
        "median_vs_mae_km_s": float(np.mean(np.abs(err_vs))),
        "median_vs_rmse_km_s": float(np.sqrt(np.mean(err_vs**2))),
        "vs_p05_p95_coverage": float(((target[1] >= qs[0, 1]) & (target[1] <= qs[4, 1])).mean()),
        "vs_p16_p84_coverage": float(((target[1] >= qs[1, 1]) & (target[1] <= qs[3, 1])).mean()),
        "mean_vs_p05_p95_width_km_s": float(np.mean(qs[4, 1] - qs[0, 1])),
        "mean_vs_p16_p84_width_km_s": float(np.mean(qs[3, 1] - qs[1, 1])),
    }


def sample_di(boundary_mod, model, case: Case, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    samples = boundary_mod.direct_samples(
        model,
        case.disp[None].astype(np.float32),
        case.mask[None].astype(np.float32),
        device,
        n_samples=args.di_samples,
        steps=args.di_steps,
        batch_size=1,
    )[0]
    return samples.astype(np.float32)


def draw_case_figure(
    case: Case,
    depth: np.ndarray,
    mcmc_profiles: np.ndarray,
    mcmc_disp_qs: np.ndarray,
    di_profiles: Dict[str, np.ndarray],
    row: Dict[str, object],
    args: argparse.Namespace,
) -> Path:
    period = case.disp[0]
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 2.55), gridspec_kw={"width_ratios": [1.05, 1.0, 0.9, 0.75]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.79, wspace=0.38)

    ax = axes[0]
    obs_ray = case.mask[1] > 0.5
    obs_love = case.mask[2] > 0.5
    if np.any(obs_ray):
        ax.scatter(period[obs_ray], case.disp[1, obs_ray], s=12, color=COLORS["target"], label="Rayleigh obs", zorder=10)
    if np.any(obs_love):
        ax.scatter(period[obs_love], case.disp[2, obs_love], s=12, facecolor="white", edgecolor=COLORS["target"], label="Love obs", zorder=10)
    if np.all(np.isfinite(mcmc_disp_qs)):
        ax.fill_between(period, mcmc_disp_qs[0, 0], mcmc_disp_qs[4, 0], color=COLORS["mcmc"], alpha=0.18, lw=0)
        ax.plot(period, mcmc_disp_qs[2, 0], color=COLORS["mcmc"], lw=1.25, label="MCMC R")
        ax.fill_between(period, mcmc_disp_qs[0, 1], mcmc_disp_qs[4, 1], color=COLORS["mcmc"], alpha=0.10, lw=0)
        ax.plot(period, mcmc_disp_qs[2, 1], color=COLORS["mcmc"], lw=1.05, ls="--", label="MCMC L")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("$c$ (km/s)")
    ax.set_title("Posterior predictive", loc="left", fontsize=8.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.7)
    style(ax)

    ax = axes[1]
    ax.plot(case.target[1], depth, color=COLORS["target"], lw=1.4, label="target")
    qs = np.quantile(mcmc_profiles[:, 1], (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    ax.fill_betweenx(depth, qs[0], qs[4], color=COLORS["mcmc"], alpha=0.18, lw=0)
    ax.plot(qs[2], depth, color=COLORS["mcmc"], lw=1.35, label="MCMC")
    for name, profiles in di_profiles.items():
        c = COLORS["di_strong"] if name == "DI-Strong" else COLORS["di_weak"]
        q = np.quantile(profiles[:, 1], (0.05, 0.50, 0.95), axis=0)
        ax.fill_betweenx(depth, q[0], q[2], color=c, alpha=0.06, lw=0)
        ax.plot(q[1], depth, color=c, lw=1.05, ls=(0, (2.3, 1.5)), label=name)
    ax.set_ylim(float(depth[-1]), 0)
    ax.set_xlabel("$V_S$ (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("$V_S$ posterior", loc="left", fontsize=8.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.7, loc="lower left", handlelength=1.5)
    style(ax)

    ax = axes[2]
    labels = ["MCMC"] + list(di_profiles.keys())
    maes = [float(row["mcmc_median_vs_mae_km_s"])]
    widths = [float(row["mcmc_mean_vs_p05_p95_width_km_s"])]
    colors = [COLORS["mcmc"]]
    for name, profiles in di_profiles.items():
        s = summarize_profiles(profiles, case.target)
        maes.append(s["median_vs_mae_km_s"])
        widths.append(s["mean_vs_p05_p95_width_km_s"])
        colors.append(COLORS["di_strong"] if name == "DI-Strong" else COLORS["di_weak"])
    x = np.arange(len(labels))
    bw = 0.34
    for i, c in enumerate(colors):
        ax.bar(x[i] - bw / 2, maes[i], width=bw, color=c, alpha=0.9, lw=0)
        ax.bar(x[i] + bw / 2, widths[i], width=bw, color=c, alpha=0.28, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("-", "\n") for l in labels], fontsize=5.8)
    ax.set_ylabel("$V_S$ (km/s)")
    ax.set_title("MAE / width", loc="left", fontsize=8.0, fontweight="bold")
    ax.text(0.02, 0.97, "dark: MAE\npale: p05-p95", transform=ax.transAxes, ha="left", va="top", fontsize=5.6)
    style(ax)

    ax = axes[3]
    ax.axis("off")
    usable = str(row["mcmc_usable"])
    diagnostics = [
        f"usable: {usable}",
        f"accept: {float(row['mcmc_accept_mean']):.2f}",
        f"max R-hat: {float(row['mcmc_rhat_max']):.2f}",
        f"min ESS: {float(row['mcmc_ess_min']):.0f}",
        f"pred RMS: {float(row['mcmc_pred_median_rms_km_s']):.3f}",
    ]
    ax.text(0.0, 0.95, "\n".join(diagnostics), ha="left", va="top", fontsize=6.3, linespacing=1.45)
    fig.suptitle(
        f"{REGIME_LABELS.get(case.regime, case.regime)} case {case.case_index}: MCMC posterior reference",
        x=0.07,
        ha="left",
        fontsize=8.8,
        fontweight="bold",
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"mcmc_reference_{case.regime}_case{case.case_index}_{args.input_mode}"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def plot_summary(rows: List[Dict[str, object]], args: argparse.Namespace) -> Path:
    regimes = ["in-prior", "boundary", "out-of-prior"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.78, wspace=0.34)
    metrics = [
        ("mcmc_rhat_max", "max split R-hat"),
        ("mcmc_ess_min", "min ESS"),
        ("mcmc_pred_median_rms_km_s", "median predictive RMS (km/s)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = []
        labels = []
        colors = []
        for regime in regimes:
            group = [r for r in rows if r["regime"] == regime]
            if not group:
                continue
            vals.append(float(np.mean([float(r[metric]) for r in group])))
            labels.append(REGIME_LABELS[regime])
            colors.append(COLORS["mcmc"] if all(r["mcmc_usable"] for r in group) else "#a55a5a")
        ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.85, lw=0)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        if metric == "mcmc_rhat_max":
            ax.axhline(args.rhat_threshold, color="0.25", ls=":", lw=0.9)
        if metric == "mcmc_ess_min":
            ax.axhline(args.ess_threshold, color="0.25", ls=":", lw=0.9)
        if metric == "mcmc_pred_median_rms_km_s":
            ax.axhline(args.pred_rms_threshold_factor * args.sigma_c, color="0.25", ls=":", lw=0.9)
        style(ax)
    fig.suptitle("MCMC reference posterior diagnostic", x=0.075, ha="left", fontsize=8.8, fontweight="bold")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"mcmc_reference_summary_{args.input_mode}"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    p.add_argument("--weak-ckpt", type=Path, default=DEFAULT_WEAK_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--input-mode", choices=["rayleigh", "love", "joint"], default="joint")
    p.add_argument("--regimes", default="all", help="all or comma-separated subset: in-prior,boundary,out-of-prior")
    p.add_argument("--cases-per-regime", type=int, default=1)
    p.add_argument("--period-min", type=float, default=2.0)
    p.add_argument("--period-max", type=float, default=60.0)
    p.add_argument("--period-step", type=float, default=1.0)
    p.add_argument("--sigma-c", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--knot-depths-km", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 127.5])
    p.add_argument("--vpvs-mean", type=float, default=1.80)
    p.add_argument("--vpvs-sigma", type=float, default=0.08)
    p.add_argument("--smooth-sigma", type=float, default=0.45)
    p.add_argument("--n-init", type=int, default=256)
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--sampler", choices=["rw", "stretch"], default="stretch")
    p.add_argument("--n-steps", type=int, default=1600)
    p.add_argument("--burnin", type=int, default=600)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--proposal-vs", type=float, default=0.055)
    p.add_argument("--proposal-vpvs", type=float, default=0.012)
    p.add_argument("--stretch-a", type=float, default=2.0)
    p.add_argument("--adapt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adapt-window", type=int, default=100)
    p.add_argument("--pred-max-samples", type=int, default=96)
    p.add_argument("--di-samples", type=int, default=64)
    p.add_argument("--di-steps", type=int, default=24)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--rhat-threshold", type=float, default=1.20)
    p.add_argument("--ess-threshold", type=float, default=80.0)
    p.add_argument("--accept-min", type=float, default=0.10)
    p.add_argument("--accept-max", type=float, default=0.60)
    p.add_argument("--pred-rms-threshold-factor", type=float, default=1.75)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    strong_mod = import_from_path("mcmc_ref_strong_prior", ROOT / "utils" / "generate_data.py")
    boundary_mod = import_from_path("mcmc_ref_boundary_helpers", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    device = choose_device(args.device)
    print("[info] generating test cases")
    cases = make_cases(boundary_mod, strong_mod, args)
    depth = np.arange(cases[0].target.shape[-1], dtype=np.float32) * 0.5
    knot_depths = np.asarray(args.knot_depths_km, dtype=np.float32)
    if knot_depths[0] != depth[0] or knot_depths[-1] != depth[-1]:
        knot_depths = np.unique(np.concatenate([[depth[0]], knot_depths, [depth[-1]]])).astype(np.float32)
    vs_lo, vs_hi = knot_bounds(knot_depths)
    lo = np.concatenate([vs_lo, [args.vpvs_mean - 3 * args.vpvs_sigma]]).astype(np.float32)
    hi = np.concatenate([vs_hi, [args.vpvs_mean + 3 * args.vpvs_sigma]]).astype(np.float32)
    print(f"[info] MCMC params: {len(knot_depths)} Vs knots + Vp/Vs = {len(lo)} dimensions")
    if args.sampler == "stretch" and args.n_chains < 2 * len(lo):
        print(
            f"[warn] stretch sampler usually needs at least 2 x ndim walkers; "
            f"n_chains={args.n_chains}, ndim={len(lo)}"
        )

    di_models = {}
    for label, ckpt in [("DI-Strong", args.strong_ckpt), ("DI-Weak", args.weak_ckpt)]:
        if ckpt.exists():
            print(f"[info] loading {label}: {ckpt}")
            model, _cfg = boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", ckpt, device)
            if model is not None:
                di_models[label] = model

    rows: List[Dict[str, object]] = []
    figure_paths: List[str] = []
    npz_payload: Dict[str, np.ndarray] = {}
    for case in cases:
        print(f"[info] selecting starts for {case.regime} case {case.case_index}")
        starts = select_starts(case, strong_mod, depth, knot_depths, lo, hi, args, rng)
        print(f"[info] running MCMC for {case.regime} case {case.case_index}")
        if args.sampler == "stretch":
            result = run_stretch_mcmc(case, strong_mod, depth, knot_depths, lo, hi, starts, args, rng)
        else:
            result = run_mcmc(case, strong_mod, depth, knot_depths, lo, hi, starts, args, rng)
        draws = posterior_draws(result, args.burnin, args.thin)
        rhat = split_rhat(draws)
        ess = ess_per_param(draws)
        mcmc_profiles = sample_profiles(draws, depth, knot_depths)
        mcmc_disp_qs, _, n_pred = predictive_quantiles(strong_mod, mcmc_profiles, case.disp[0], args.pred_max_samples)
        if np.all(np.isfinite(mcmc_disp_qs)):
            wave_mask = case.mask[1:3].astype(bool)
            pred_resid = (mcmc_disp_qs[2] - case.disp[1:3])[wave_mask]
            pred_rms = float(np.sqrt(np.mean(pred_resid**2)))
        else:
            pred_rms = math.inf
        mcmc_summary = summarize_profiles(mcmc_profiles, case.target)
        accept_mean = float(np.mean(result.accept_rate))
        usable = bool(
            accept_mean >= args.accept_min
            and accept_mean <= args.accept_max
            and float(np.nanmax(rhat)) <= args.rhat_threshold
            and float(np.nanmin(ess)) >= args.ess_threshold
            and pred_rms <= args.pred_rms_threshold_factor * args.sigma_c
        )
        row: Dict[str, object] = {
            "regime": case.regime,
            "case_index": case.case_index,
            "input_mode": args.input_mode,
            "n_chains": args.n_chains,
            "sampler": args.sampler,
            "n_steps": args.n_steps,
            "burnin": args.burnin,
            "thin": args.thin,
            "n_post_draws": int(np.prod(draws.shape[:2])),
            "n_params": int(len(lo)),
            "sigma_c_km_s": args.sigma_c,
            "mcmc_runtime_s": result.runtime_s,
            "mcmc_accept_mean": accept_mean,
            "mcmc_accept_min": float(np.min(result.accept_rate)),
            "mcmc_accept_max": float(np.max(result.accept_rate)),
            "mcmc_rhat_max": float(np.nanmax(rhat)),
            "mcmc_rhat_median": float(np.nanmedian(rhat)),
            "mcmc_ess_min": float(np.nanmin(ess)),
            "mcmc_ess_median": float(np.nanmedian(ess)),
            "mcmc_pred_median_rms_km_s": pred_rms,
            "mcmc_pred_valid_samples": int(n_pred),
            "mcmc_usable": usable,
        }
        for key, value in mcmc_summary.items():
            row[f"mcmc_{key}"] = value
        di_profiles = {}
        for label, model in di_models.items():
            samples = sample_di(boundary_mod, model, case, device, args)
            di_profiles[label] = samples
            for key, value in summarize_profiles(samples, case.target).items():
                row[f"{label.lower().replace('-', '_')}_{key}"] = value
        rows.append(row)
        fig_path = draw_case_figure(case, depth, mcmc_profiles, mcmc_disp_qs, di_profiles, row, args)
        figure_paths.append(str(fig_path))
        prefix = f"{case.regime}_case{case.case_index}".replace("-", "_")
        npz_payload[f"{prefix}_mcmc_param_draws"] = draws.astype(np.float32)
        npz_payload[f"{prefix}_mcmc_profile_qs"] = np.quantile(mcmc_profiles, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
        npz_payload[f"{prefix}_mcmc_disp_qs"] = mcmc_disp_qs.astype(np.float32)
        npz_payload[f"{prefix}_target"] = case.target.astype(np.float32)
        npz_payload[f"{prefix}_disp"] = case.disp.astype(np.float32)
        npz_payload[f"{prefix}_mask"] = case.mask.astype(np.float32)
    figure_paths.append(str(plot_summary(rows, args)))
    write_csv(args.out_dir / "mcmc_posterior_reference_metrics.csv", rows)
    write_json(
        args.out_dir / "mcmc_posterior_reference_protocol.json",
        {
            "created_unix_time": time.time(),
            "interpretation": (
                "Low-dimensional Vs-spline Metropolis posterior reference. Treat rows with "
                "mcmc_usable=false as failed reference posteriors, not as paper evidence."
            ),
            "input_mode": args.input_mode,
            "regimes": args.regimes,
            "cases_per_regime": args.cases_per_regime,
            "sigma_c_km_s": args.sigma_c,
            "knot_depths_km": [float(x) for x in knot_depths],
            "n_chains": args.n_chains,
            "n_steps": args.n_steps,
            "burnin": args.burnin,
            "thin": args.thin,
            "thresholds": {
                "rhat_max": args.rhat_threshold,
                "ess_min": args.ess_threshold,
                "accept_min": args.accept_min,
                "accept_max": args.accept_max,
                "pred_rms_max": args.pred_rms_threshold_factor * args.sigma_c,
            },
            "figure_paths": figure_paths,
            "checkpoints": {label: str(ckpt) for label, ckpt in [("DI-Strong", args.strong_ckpt), ("DI-Weak", args.weak_ckpt)]},
        },
    )
    np.savez_compressed(args.out_dir / "mcmc_posterior_reference_diagnostics.npz", **npz_payload)
    print(f"[done] wrote {args.out_dir / 'mcmc_posterior_reference_metrics.csv'}")
    print(f"[done] wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
