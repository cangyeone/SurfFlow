#!/usr/bin/env python3
"""Same-prior MCMC posterior comparison for SurfFlow DI.

This script targets the narrow validation claim:

    Does the learned DI posterior resemble a Bayesian posterior under the same
    synthetic prior family, forward solver, observation mask and noise model?

It projects the SurfFlow synthetic training prior to a low-dimensional state
vector, fits an empirical Gaussian-mixture prior in that space, and runs an
affine-invariant stretch MCMC sampler for selected in-prior synthetic cases.
The output reports posterior-similarity diagnostics between the MCMC reference
and DI posterior samples.
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
from typing import Dict, Iterable, List, Tuple

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
DEFAULT_OUT_DIR = ROOT / "results" / "same_prior_mcmc_posterior"
DEFAULT_FIG_DIR = ROOT / "figures" / "same_prior_mcmc_posterior"

COLORS = {
    "target": "#111111",
    "mcmc": "#5f9d68",
    "di": "#2f76b7",
    "prior": "#a7b6c7",
    "grid": "#e7ebf0",
}

REGIME_LABELS = {
    "in-prior": "inside support",
    "boundary": "near edge of support",
    "out-of-prior": "outside support",
}


@dataclass
class EmpiricalGMM:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    means_z: np.ndarray
    covs_z: np.ndarray
    inv_covs_z: np.ndarray
    logdets_z: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module():
    return import_from_path("same_prior_mcmc_base", ROOT / "scripts" / "run_mcmc_posterior_reference.py")


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


def case_prefix(case) -> str:
    return f"{case.regime}_case{case.case_index}".replace("-", "_")


def style(ax) -> None:
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.tick_params(direction="out", length=3.0, width=0.7, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def project_profile_to_params(profile: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> np.ndarray:
    vp, vs, _rho = profile
    vs_knots = np.interp(knot_depths, depth, vs).astype(np.float32)
    vpvs = np.median(vp / np.maximum(vs, 1e-3)).astype(np.float32)
    return np.concatenate([vs_knots, [vpvs]]).astype(np.float32)


def params_to_profile(base, params: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return base.params_to_profile(params, depth, knot_depths)


def sample_prior_params(
    strong_mod,
    n: int,
    depth: np.ndarray,
    knot_depths: np.ndarray,
    seed: int,
    progress_every: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 10:
        tries += 1
        try:
            d, vs, vp, rho, _meta = strong_mod.sample_global_1d_model(z_max_km=150.0, dz_km=0.5, rng=rng)
            profile = np.stack([vp[: len(depth)], vs[: len(depth)], rho[: len(depth)]]).astype(np.float32)
            out.append(project_profile_to_params(profile, depth, knot_depths))
        except Exception:
            continue
        if progress_every and len(out) % progress_every == 0:
            print(f"[prior] projected samples {len(out)}/{n}")
    if len(out) < n:
        raise RuntimeError(f"Only projected {len(out)} prior samples after {tries} attempts")
    return np.stack(out).astype(np.float32)


def fit_gmm_full(
    x: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
    n_iter: int,
    reg_covar: float,
) -> EmpiricalGMM:
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(1e-4)
    z = (x - mean) / scale
    init_idx = rng.choice(n, size=n_components, replace=False)
    means = z[init_idx].copy()
    covs = np.stack([np.cov(z.T) + reg_covar * np.eye(p) for _ in range(n_components)])
    weights = np.full(n_components, 1.0 / n_components)
    for _ in range(n_iter):
        log_resp = np.empty((n, n_components), dtype=np.float64)
        invs = []
        logdets = []
        for k in range(n_components):
            cov = covs[k] + reg_covar * np.eye(p)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + 10 * reg_covar * np.eye(p)
                sign, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.inv(cov)
            invs.append(inv)
            logdets.append(logdet)
            dz = z - means[k]
            q = np.einsum("ij,jk,ik->i", dz, inv, dz)
            log_resp[:, k] = np.log(weights[k] + 1e-12) - 0.5 * (p * np.log(2 * np.pi) + logdet + q)
        norm = np.apply_along_axis(logsumexp, 1, log_resp)
        resp = np.exp(log_resp - norm[:, None])
        nk = resp.sum(axis=0).clip(1e-6)
        weights = nk / n
        means = (resp.T @ z) / nk[:, None]
        for k in range(n_components):
            dz = z - means[k]
            covs[k] = (dz * resp[:, k : k + 1]).T @ dz / nk[k] + reg_covar * np.eye(p)
    inv_covs = np.empty_like(covs)
    logdets = np.empty(n_components, dtype=np.float64)
    for k in range(n_components):
        sign, logdet = np.linalg.slogdet(covs[k])
        if sign <= 0:
            covs[k] += 10 * reg_covar * np.eye(p)
            sign, logdet = np.linalg.slogdet(covs[k])
        inv_covs[k] = np.linalg.inv(covs[k])
        logdets[k] = logdet
    q_lo = np.quantile(x, 0.001, axis=0)
    q_hi = np.quantile(x, 0.999, axis=0)
    pad = 0.10 * (q_hi - q_lo).clip(1e-3)
    lo = q_lo - pad
    hi = q_hi + pad
    # Keep broad physical sanity bounds around the empirical support.
    phys_lo = np.array([0.15] * (p - 1) + [1.55], dtype=np.float64)
    phys_hi = np.array([6.20] * (p - 1) + [2.10], dtype=np.float64)
    lo = np.maximum(lo, phys_lo).astype(np.float32)
    hi = np.minimum(hi, phys_hi).astype(np.float32)
    return EmpiricalGMM(
        weights=weights.astype(np.float64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        means_z=means.astype(np.float64),
        covs_z=covs.astype(np.float64),
        inv_covs_z=inv_covs.astype(np.float64),
        logdets_z=logdets.astype(np.float64),
        lo=lo,
        hi=hi,
    )


def gmm_logpdf(gmm: EmpiricalGMM, x: np.ndarray) -> float:
    if np.any(x < gmm.lo) or np.any(x > gmm.hi) or not np.all(np.isfinite(x)):
        return -np.inf
    z = (np.asarray(x, dtype=np.float64) - gmm.mean) / gmm.scale
    p = len(z)
    terms = np.empty(len(gmm.weights), dtype=np.float64)
    for k in range(len(gmm.weights)):
        dz = z - gmm.means_z[k]
        q = float(dz @ gmm.inv_covs_z[k] @ dz)
        terms[k] = np.log(gmm.weights[k] + 1e-12) - 0.5 * (p * np.log(2 * np.pi) + gmm.logdets_z[k] + q)
    # The Jacobian constant is irrelevant for MCMC ratios, but keeping it makes
    # reported log probabilities interpretable.
    return logsumexp(terms) - float(np.sum(np.log(gmm.scale)))


def gmm_sample(gmm: EmpiricalGMM, n: int, rng: np.random.Generator) -> np.ndarray:
    comps = rng.choice(len(gmm.weights), size=n, p=gmm.weights)
    out = np.empty((n, len(gmm.mean)), dtype=np.float32)
    for i, k in enumerate(comps):
        z = rng.multivariate_normal(gmm.means_z[k], gmm.covs_z[k])
        out[i] = (gmm.mean + z * gmm.scale).astype(np.float32)
    return out


def compute_dispersion_residual(base, strong_mod, params, case, depth, knot_depths) -> Tuple[np.ndarray, np.ndarray]:
    vp, vs, rho = params_to_profile(base, params, depth, knot_depths)
    pred = base.compute_dispersion(strong_mod, depth, vp, vs, rho, case.disp[0])
    wave_mask = case.mask[1:3].astype(bool)
    residual = (pred[1:3] - case.disp[1:3])[wave_mask]
    return pred, residual


def log_posterior_same_prior(base, strong_mod, gmm, params, case, depth, knot_depths, sigma_c) -> Tuple[float, float]:
    lp = gmm_logpdf(gmm, params)
    if not np.isfinite(lp):
        return -np.inf, np.inf
    try:
        _pred, residual = compute_dispersion_residual(base, strong_mod, params, case, depth, knot_depths)
    except Exception:
        return -np.inf, np.inf
    rms = float(np.sqrt(np.mean(residual**2)))
    ll = -0.5 * float(np.sum((residual / sigma_c) ** 2))
    return lp + ll, rms


def select_starts_same_prior(base, strong_mod, gmm, case, depth, knot_depths, args, rng) -> np.ndarray:
    draws = []
    while len(draws) < args.n_init:
        cand = gmm_sample(gmm, args.n_init, rng)
        ok = np.all((cand >= gmm.lo) & (cand <= gmm.hi), axis=1)
        draws.extend(cand[ok].tolist())
    draws = np.asarray(draws[: args.n_init], dtype=np.float32)
    scored = []
    for i, x in enumerate(draws):
        lp, rms = log_posterior_same_prior(base, strong_mod, gmm, x, case, depth, knot_depths, args.sigma_c)
        if np.isfinite(lp):
            scored.append((lp, rms, i))
    if len(scored) < args.n_walkers:
        raise RuntimeError(f"Only {len(scored)} finite prior starts; need {args.n_walkers}")
    scored.sort(reverse=True, key=lambda t: t[0])
    return draws[[i for _lp, _rms, i in scored[: args.n_walkers]]].astype(np.float32)


def sample_stretch_z(rng: np.random.Generator, a: float) -> float:
    lo = 1.0 / math.sqrt(a)
    hi = math.sqrt(a)
    return float((lo + (hi - lo) * rng.random()) ** 2)


def run_stretch_same_prior(base, strong_mod, gmm, case, depth, knot_depths, starts, args, rng):
    tic = time.time()
    n_walkers, n_params = starts.shape
    samples = np.empty((n_walkers, args.n_steps, n_params), dtype=np.float32)
    logps = np.empty((n_walkers, args.n_steps), dtype=np.float32)
    accepted = np.zeros(n_walkers, dtype=np.int64)
    params = starts.copy()
    current = np.array(
        [log_posterior_same_prior(base, strong_mod, gmm, p, case, depth, knot_depths, args.sigma_c)[0] for p in params],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(current)):
        raise RuntimeError("non-finite initial walkers")
    for step in range(args.n_steps):
        for i in rng.permutation(n_walkers):
            candidates = np.delete(np.arange(n_walkers), i)
            j = int(rng.choice(candidates))
            z = sample_stretch_z(rng, args.stretch_a)
            prop = params[j] + z * (params[i] - params[j])
            if np.any(prop < gmm.lo) or np.any(prop > gmm.hi):
                continue
            prop_lp, _ = log_posterior_same_prior(
                base, strong_mod, gmm, prop.astype(np.float32), case, depth, knot_depths, args.sigma_c
            )
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
    return base.MCMCResult(
        samples=samples,
        logp=logps,
        accept_rate=accepted / float(args.n_steps),
        step_scale=float(args.stretch_a),
        runtime_s=time.time() - tic,
    )


def params_to_profiles(base, draws: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> np.ndarray:
    flat = draws.reshape(-1, draws.shape[-1])
    profiles = []
    for p in flat:
        vp, vs, rho = params_to_profile(base, p, depth, knot_depths)
        profiles.append(np.stack([vp, vs, rho]).astype(np.float32))
    return np.stack(profiles)


def project_profiles(profiles: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> np.ndarray:
    return np.stack([project_profile_to_params(p, depth, knot_depths) for p in profiles]).astype(np.float32)


def posterior_similarity(mcmc_profiles: np.ndarray, di_profiles: np.ndarray, target: np.ndarray, depth: np.ndarray, knot_depths: np.ndarray) -> Dict[str, float]:
    m_q = np.quantile(mcmc_profiles[:, 1], (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    d_q = np.quantile(di_profiles[:, 1], (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    overlap = np.maximum(0.0, np.minimum(m_q[4], d_q[4]) - np.maximum(m_q[0], d_q[0]))
    union = np.maximum(m_q[4], d_q[4]) - np.minimum(m_q[0], d_q[0])
    m_width = np.maximum(m_q[4] - m_q[0], 1e-6)
    d_width = d_q[4] - d_q[0]
    qs = np.linspace(0.01, 0.99, 99)
    m_param = project_profiles(mcmc_profiles, depth, knot_depths)
    d_param = project_profiles(di_profiles, depth, knot_depths)
    wdist = []
    for j in range(len(knot_depths)):
        wdist.append(float(np.mean(np.abs(np.quantile(m_param[:, j], qs) - np.quantile(d_param[:, j], qs)))))
    return {
        "vs_median_absdiff_mcmc_di_km_s": float(np.mean(np.abs(m_q[2] - d_q[2]))),
        "vs_q05_q50_q95_absdiff_mcmc_di_km_s": float(np.mean(np.abs(m_q[[0, 2, 4]] - d_q[[0, 2, 4]]))),
        "vs_p05_p95_interval_overlap": float(np.mean(overlap / np.maximum(union, 1e-6))),
        "vs_p05_p95_width_ratio_di_to_mcmc": float(np.mean(d_width / m_width)),
        "vs_control_wasserstein_mean_km_s": float(np.mean(wdist)),
        "mcmc_target_vs_mae_km_s": float(np.mean(np.abs(m_q[2] - target[1]))),
        "di_target_vs_mae_km_s": float(np.mean(np.abs(d_q[2] - target[1]))),
    }


def profile_summary(samples: np.ndarray, target: np.ndarray, prefix: str) -> Dict[str, float]:
    qs = np.quantile(samples[:, 1], (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    return {
        f"{prefix}_median_vs_mae_km_s": float(np.mean(np.abs(qs[2] - target[1]))),
        f"{prefix}_mean_vs_p05_p95_width_km_s": float(np.mean(qs[4] - qs[0])),
        f"{prefix}_vs_p05_p95_coverage": float(((target[1] >= qs[0]) & (target[1] <= qs[4])).mean()),
        f"{prefix}_vs_p16_p84_coverage": float(((target[1] >= qs[1]) & (target[1] <= qs[3])).mean()),
    }


def draw_case_figure(base, case, depth, knot_depths, mcmc_profiles, di_profiles, prior_params, row, args) -> Path:
    period = case.disp[0]
    m_q = np.quantile(mcmc_profiles[:, 1], (0.05, 0.50, 0.95), axis=0)
    d_q = np.quantile(di_profiles[:, 1], (0.05, 0.50, 0.95), axis=0)
    prior_vs = np.stack([np.interp(depth, knot_depths, p[:-1]) for p in prior_params[: min(len(prior_params), 2000)]])
    p_q = np.quantile(prior_vs, (0.05, 0.95), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(7.3, 2.55), gridspec_kw={"width_ratios": [1.0, 1.02, 0.82, 0.74]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.23, top=0.79, wspace=0.40)

    ax = axes[0]
    obs_ray = case.mask[1] > 0.5
    obs_love = case.mask[2] > 0.5
    ax.scatter(period[obs_ray], case.disp[1, obs_ray], s=12, color=COLORS["target"], label="Rayleigh obs", zorder=10)
    ax.scatter(period[obs_love], case.disp[2, obs_love], s=12, facecolor="white", edgecolor=COLORS["target"], label="Love obs", zorder=10)
    mcmc_disp_qs, _, _n = base.predictive_quantiles(base.import_from_path("same_prior_fig_strong", ROOT / "utils" / "generate_data.py"), mcmc_profiles, period, args.pred_max_samples)
    if np.all(np.isfinite(mcmc_disp_qs)):
        ax.fill_between(period, mcmc_disp_qs[0, 0], mcmc_disp_qs[4, 0], color=COLORS["mcmc"], alpha=0.16, lw=0)
        ax.plot(period, mcmc_disp_qs[2, 0], color=COLORS["mcmc"], lw=1.2, label="MCMC R")
        ax.fill_between(period, mcmc_disp_qs[0, 1], mcmc_disp_qs[4, 1], color=COLORS["mcmc"], alpha=0.08, lw=0)
        ax.plot(period, mcmc_disp_qs[2, 1], color=COLORS["mcmc"], lw=1.0, ls="--", label="MCMC L")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("$c$ (km/s)")
    ax.set_title("Posterior predictive", loc="left", fontsize=8.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.4, loc="lower right", ncol=2, handlelength=1.3, columnspacing=0.6)
    style(ax)

    ax = axes[1]
    ax.fill_betweenx(depth, p_q[0], p_q[1], color=COLORS["prior"], alpha=0.16, lw=0, label="projected prior")
    ax.fill_betweenx(depth, m_q[0], m_q[2], color=COLORS["mcmc"], alpha=0.20, lw=0)
    ax.plot(m_q[1], depth, color=COLORS["mcmc"], lw=1.35, label="MCMC")
    ax.fill_betweenx(depth, d_q[0], d_q[2], color=COLORS["di"], alpha=0.12, lw=0)
    ax.plot(d_q[1], depth, color=COLORS["di"], lw=1.15, ls=(0, (2.3, 1.5)), label="DI-Strong")
    ax.plot(case.target[1], depth, color=COLORS["target"], lw=1.35, label="target")
    ax.set_ylim(float(depth[-1]), 0)
    ax.set_xlabel("$V_S$ (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("$V_S$ posterior", loc="left", fontsize=8.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.4, loc="lower left", handlelength=1.4)
    style(ax)

    ax = axes[2]
    labels = ["median\ndiff", "q05/q50/q95\ndiff", "interval\noverlap", "width\nratio"]
    vals = [
        row["vs_median_absdiff_mcmc_di_km_s"],
        row["vs_q05_q50_q95_absdiff_mcmc_di_km_s"],
        row["vs_p05_p95_interval_overlap"],
        row["vs_p05_p95_width_ratio_di_to_mcmc"],
    ]
    colors = [COLORS["di"], COLORS["di"], COLORS["mcmc"], COLORS["mcmc"]]
    ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.84, lw=0)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=5.3)
    ax.set_title("Posterior similarity", loc="left", fontsize=8.0, fontweight="bold")
    style(ax)

    ax = axes[3]
    ax.axis("off")
    text = [
        f"usable: {row['mcmc_usable']}",
        f"accept: {row['mcmc_accept_mean']:.2f}",
        f"max R-hat: {row['mcmc_rhat_max']:.2f}",
        f"min ESS: {row['mcmc_ess_min']:.0f}",
        f"pred RMS: {row['mcmc_pred_rms_km_s']:.3f}",
        f"overlap: {row['vs_p05_p95_interval_overlap']:.2f}",
    ]
    ax.text(0, 0.95, "\n".join(text), ha="left", va="top", fontsize=6.1, linespacing=1.45)
    regime_label = REGIME_LABELS.get(case.regime, case.regime)
    fig.suptitle(
        f"Reduced Bayesian reference vs DI posterior: {regime_label}, case {case.case_index}",
        x=0.07,
        ha="left",
        fontsize=8.8,
        fontweight="bold",
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"same_prior_mcmc_{case_prefix(case)}_{args.input_mode}"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def plot_summary(rows: List[Dict[str, object]], args: argparse.Namespace) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.78, wspace=0.34)
    x = np.arange(len(rows))
    labels = [
        f"{REGIME_LABELS.get(str(r.get('regime', '')), str(r.get('regime', '')))}\ncase {r['case_index']}"
        for r in rows
    ]
    metrics = [
        ("vs_median_absdiff_mcmc_di_km_s", "median $V_S$ diff (km/s)", COLORS["di"]),
        ("vs_p05_p95_interval_overlap", "p05-p95 overlap", COLORS["mcmc"]),
        ("vs_p05_p95_width_ratio_di_to_mcmc", "DI/MCMC p05-p95 width", COLORS["mcmc"]),
    ]
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        vals = [float(r[metric]) for r in rows]
        ax.bar(x, vals, color=color, alpha=0.85, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        style(ax)
    fig.suptitle(
        "Reduced Bayesian reference posterior similarity summary",
        x=0.075,
        ha="left",
        fontsize=8.8,
        fontweight="bold",
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"same_prior_mcmc_summary_{args.input_mode}"
    fig.savefig(out.with_suffix(".png"), dpi=350)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--strong-ckpt", type=Path, default=DEFAULT_STRONG_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--cases", type=int, default=2)
    p.add_argument("--regimes", default="in-prior", help="all or comma-separated subset: in-prior,boundary,out-of-prior")
    p.add_argument(
        "--case-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional in-prior case indices to process after generating the deterministic case list.",
    )
    p.add_argument("--input-mode", choices=["rayleigh", "love", "joint"], default="joint")
    p.add_argument("--period-min", type=float, default=2.0)
    p.add_argument("--period-max", type=float, default=60.0)
    p.add_argument("--period-step", type=float, default=1.0)
    p.add_argument("--sigma-c", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--knot-depths-km", type=float, nargs="+", default=[0.0, 5.0, 15.0, 30.0, 60.0, 127.5])
    p.add_argument("--n-prior-samples", type=int, default=3000)
    p.add_argument("--gmm-components", type=int, default=5)
    p.add_argument("--gmm-iters", type=int, default=80)
    p.add_argument("--gmm-reg-covar", type=float, default=2e-4)
    p.add_argument("--n-init", type=int, default=512)
    p.add_argument(
        "--resume-starts-npz",
        type=Path,
        default=None,
        help="Optional diagnostics npz from a previous run. Uses the last retained draw for matching cases as walker starts.",
    )
    p.add_argument("--n-walkers", type=int, default=14)
    p.add_argument("--n-steps", type=int, default=1400)
    p.add_argument("--burnin", type=int, default=500)
    p.add_argument("--thin", type=int, default=5)
    p.add_argument("--stretch-a", type=float, default=2.0)
    p.add_argument("--di-samples", type=int, default=96)
    p.add_argument("--di-steps", type=int, default=24)
    p.add_argument("--pred-max-samples", type=int, default=128)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--rhat-threshold", type=float, default=1.20)
    p.add_argument("--ess-threshold", type=float, default=80.0)
    p.add_argument("--accept-min", type=float, default=0.10)
    p.add_argument("--accept-max", type=float, default=0.65)
    p.add_argument("--pred-rms-threshold-factor", type=float, default=1.75)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = load_base_module()
    strong_mod = import_from_path("same_prior_generate_data", ROOT / "utils" / "generate_data.py")
    boundary_mod = import_from_path("same_prior_boundary_helpers", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    device = base.choose_device(args.device)
    n_cases_to_generate = args.cases
    if args.case_indices:
        n_cases_to_generate = max(n_cases_to_generate, max(args.case_indices) + 1)
    case_args = argparse.Namespace(
        period_min=args.period_min,
        period_max=args.period_max,
        period_step=args.period_step,
        input_mode=args.input_mode,
        cases_per_regime=n_cases_to_generate,
        seed=args.seed,
        regimes=args.regimes,
    )
    print("[info] generating in-prior target cases")
    cases = base.make_cases(boundary_mod, strong_mod, case_args)
    if args.case_indices:
        wanted = set(args.case_indices)
        cases = [case for case in cases if case.case_index in wanted]
        if not cases:
            raise RuntimeError(f"No generated cases matched --case-indices {args.case_indices}")
    depth = np.arange(cases[0].target.shape[-1], dtype=np.float32) * 0.5
    knot_depths = np.asarray(args.knot_depths_km, dtype=np.float32)
    if knot_depths[0] != depth[0] or knot_depths[-1] != depth[-1]:
        knot_depths = np.unique(np.concatenate([[depth[0]], knot_depths, [depth[-1]]])).astype(np.float32)
    print(f"[info] projecting strong training prior: n={args.n_prior_samples}")
    prior_params = sample_prior_params(strong_mod, args.n_prior_samples, depth, knot_depths, args.seed + 1000, args.progress_every)
    print(f"[info] fitting GMM prior: k={args.gmm_components}")
    gmm = fit_gmm_full(prior_params, args.gmm_components, rng, args.gmm_iters, args.gmm_reg_covar)
    if args.n_walkers < 2 * prior_params.shape[1]:
        print(f"[warn] n_walkers={args.n_walkers} < 2*ndim={2*prior_params.shape[1]}")
    if not args.strong_ckpt.exists():
        raise FileNotFoundError(args.strong_ckpt)
    print(f"[info] loading DI-Strong: {args.strong_ckpt}")
    di_model, _cfg = boundary_mod.load_direct_model(ROOT / "disp_inv_train.v1.3.py", args.strong_ckpt, device)
    rows: List[Dict[str, object]] = []
    figure_paths: List[str] = []
    npz_payload: Dict[str, np.ndarray] = {
        "prior_params": prior_params.astype(np.float32),
        "gmm_lo": gmm.lo.astype(np.float32),
        "gmm_hi": gmm.hi.astype(np.float32),
        "knot_depths_km": knot_depths.astype(np.float32),
    }
    resume_npz = None
    if args.resume_starts_npz is not None:
        if not args.resume_starts_npz.exists():
            raise FileNotFoundError(args.resume_starts_npz)
        resume_npz = np.load(args.resume_starts_npz)
        print(f"[info] resuming starts from {args.resume_starts_npz}")
    for case in cases:
        resume_source = ""
        starts = None
        if resume_npz is not None:
            key = f"case{case.case_index}_mcmc_param_draws"
            legacy_key = f"case{case.case_index}_mcmc_param_draws"
            key = f"{case_prefix(case)}_mcmc_param_draws"
            found_key = key if key in resume_npz else legacy_key if legacy_key in resume_npz else ""
            if found_key:
                prev = np.asarray(resume_npz[found_key], dtype=np.float32)
                if prev.ndim == 3 and prev.shape[0] == args.n_walkers and prev.shape[-1] == prior_params.shape[1]:
                    starts = prev[:, -1, :].astype(np.float32)
                    resume_source = str(args.resume_starts_npz)
                    print(f"[info] using resumed starts for case {case.case_index}")
                else:
                    print(
                        f"[warn] cannot resume {found_key}: expected ({args.n_walkers}, *, {prior_params.shape[1]}), "
                        f"got {prev.shape}"
                    )
            else:
                print(f"[warn] resume npz has no {key}; selecting starts")
        if starts is None:
            print(f"[info] selecting same-prior MCMC starts for case {case.case_index}")
            starts = select_starts_same_prior(base, strong_mod, gmm, case, depth, knot_depths, args, rng)
        print(f"[info] running same-prior MCMC for case {case.case_index}")
        result = run_stretch_same_prior(base, strong_mod, gmm, case, depth, knot_depths, starts, args, rng)
        draws = result.samples[:, args.burnin :: args.thin]
        rhat = base.split_rhat(draws)
        ess = base.ess_per_param(draws)
        mcmc_profiles = params_to_profiles(base, draws, depth, knot_depths)
        mcmc_disp_qs, _, n_pred = base.predictive_quantiles(strong_mod, mcmc_profiles, case.disp[0], args.pred_max_samples)
        wave_mask = case.mask[1:3].astype(bool)
        if np.all(np.isfinite(mcmc_disp_qs)):
            pred_rms = float(np.sqrt(np.mean((mcmc_disp_qs[2] - case.disp[1:3])[wave_mask] ** 2)))
        else:
            pred_rms = math.inf
        di_profiles = base.sample_di(boundary_mod, di_model, case, device, args)
        accept_mean = float(np.mean(result.accept_rate))
        chain_usable = bool(
            accept_mean >= args.accept_min
            and accept_mean <= args.accept_max
            and float(np.nanmax(rhat)) <= args.rhat_threshold
            and float(np.nanmin(ess)) >= args.ess_threshold
        )
        predictive_fit_ok = bool(pred_rms <= args.pred_rms_threshold_factor * args.sigma_c)
        usable = bool(chain_usable and predictive_fit_ok)
        row: Dict[str, object] = {
            "case_index": case.case_index,
            "regime": case.regime,
            "input_mode": args.input_mode,
            "prior_model": f"GMM-{args.gmm_components} projected strong prior",
            "n_prior_samples": args.n_prior_samples,
            "n_params": int(prior_params.shape[1]),
            "n_walkers": args.n_walkers,
            "n_steps": args.n_steps,
            "burnin": args.burnin,
            "thin": args.thin,
            "n_post_draws": int(np.prod(draws.shape[:2])),
            "sigma_c_km_s": args.sigma_c,
            "mcmc_runtime_s": result.runtime_s,
            "mcmc_accept_mean": accept_mean,
            "mcmc_accept_min": float(np.min(result.accept_rate)),
            "mcmc_accept_max": float(np.max(result.accept_rate)),
            "mcmc_rhat_max": float(np.nanmax(rhat)),
            "mcmc_rhat_median": float(np.nanmedian(rhat)),
            "mcmc_ess_min": float(np.nanmin(ess)),
            "mcmc_ess_median": float(np.nanmedian(ess)),
            "mcmc_pred_rms_km_s": pred_rms,
            "mcmc_pred_valid_samples": int(n_pred),
            "mcmc_chain_usable": chain_usable,
            "mcmc_predictive_fit_ok": predictive_fit_ok,
            "mcmc_usable": usable,
            "resume_source": resume_source,
        }
        row.update(profile_summary(mcmc_profiles, case.target, "mcmc"))
        row.update(profile_summary(di_profiles, case.target, "di"))
        row.update(posterior_similarity(mcmc_profiles, di_profiles, case.target, depth, knot_depths))
        rows.append(row)
        fig_path = draw_case_figure(base, case, depth, knot_depths, mcmc_profiles, di_profiles, prior_params, row, args)
        figure_paths.append(str(fig_path))
        prefix = case_prefix(case)
        legacy_prefix = f"case{case.case_index}"
        npz_payload[f"{prefix}_mcmc_param_draws"] = draws.astype(np.float32)
        npz_payload[f"{prefix}_mcmc_profile_qs"] = np.quantile(mcmc_profiles, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
        npz_payload[f"{prefix}_di_profile_qs"] = np.quantile(di_profiles, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
        npz_payload[f"{prefix}_target"] = case.target.astype(np.float32)
        npz_payload[f"{prefix}_disp"] = case.disp.astype(np.float32)
        npz_payload[f"{prefix}_mask"] = case.mask.astype(np.float32)
        if case.regime == "in-prior":
            npz_payload[f"{legacy_prefix}_mcmc_param_draws"] = draws.astype(np.float32)
            npz_payload[f"{legacy_prefix}_mcmc_profile_qs"] = np.quantile(mcmc_profiles, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
            npz_payload[f"{legacy_prefix}_di_profile_qs"] = np.quantile(di_profiles, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0).astype(np.float32)
            npz_payload[f"{legacy_prefix}_target"] = case.target.astype(np.float32)
            npz_payload[f"{legacy_prefix}_disp"] = case.disp.astype(np.float32)
            npz_payload[f"{legacy_prefix}_mask"] = case.mask.astype(np.float32)
    figure_paths.append(str(plot_summary(rows, args)))
    write_csv(args.out_dir / "same_prior_mcmc_posterior_metrics.csv", rows)
    write_json(
        args.out_dir / "same_prior_mcmc_posterior_protocol.json",
        {
            "created_unix_time": time.time(),
            "interpretation": (
                "Projected same-prior MCMC reference. The empirical prior is fitted to low-dimensional "
                "parameters projected from the SurfFlow strong synthetic training generator. This is the "
                "appropriate test for posterior similarity, distinct from broad-prior stress MCMC."
            ),
            "prior_projection": "Vs at knot depths plus median Vp/Vs",
            "knot_depths_km": [float(x) for x in knot_depths],
            "n_prior_samples": args.n_prior_samples,
            "gmm_components": args.gmm_components,
            "input_mode": args.input_mode,
            "regimes": args.regimes,
            "cases": args.cases,
            "sigma_c_km_s": args.sigma_c,
            "mcmc": {
                "sampler": "affine-invariant stretch move",
                "n_walkers": args.n_walkers,
                "n_steps": args.n_steps,
                "burnin": args.burnin,
                "thin": args.thin,
            },
            "thresholds": {
                "rhat_max": args.rhat_threshold,
                "ess_min": args.ess_threshold,
                "accept_min": args.accept_min,
                "accept_max": args.accept_max,
                "pred_rms_max": args.pred_rms_threshold_factor * args.sigma_c,
            },
            "figure_paths": figure_paths,
            "checkpoint": str(args.strong_ckpt),
        },
    )
    np.savez_compressed(args.out_dir / "same_prior_mcmc_posterior_diagnostics.npz", **npz_payload)
    print(f"[done] wrote {args.out_dir / 'same_prior_mcmc_posterior_metrics.csv'}")
    print(f"[done] wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
