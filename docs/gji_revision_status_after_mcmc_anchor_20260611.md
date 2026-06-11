# GJI revision status after posterior-anchor experiments

Date: 2026-06-11

## Core framing

The paper should not claim that SurfFlow is a more general or more accurate replacement for DispFormer or QEDispInv. The strongest and most defensible GJI framing is:

> SurfFlow is an amortized probabilistic posterior sampler for a stated synthetic surface-wave inverse problem, and the paper shows how to audit that learned posterior through prior support, empirical coverage, posterior predictive checks, MCMC anchoring and common-input baselines.

In manuscript wording, use "learned posterior under the stated synthetic training distribution", "synthetic-distribution posterior", or "reduced Bayesian reference". Avoid unqualified claims such as "the Bayesian posterior" or "field-calibrated posterior".

## Evidence now in place

### 1. Prior-support and calibration diagnostics

Main manuscript figures:

- `manuscript/figures/fig03_direct_inversion_results_v9.pdf`
- `manuscript/figures/fig04_calibration_reliability_v4.pdf`

These support the central reliability-audit story:

- DI-Strong is best inside its own structural support.
- DI-Weak is less accurate inside support but degrades less near and outside the strong-prior support.
- Raw DI intervals are not automatically calibrated; posterior-temperature scaling is a diagnostic, not a physical likelihood.

Use the terminology consistently:

- `inside support`
- `near edge`
- `outside support`

Avoid the older terms `in-prior`, `near-boundary`, and `out-of-prior` in reader-facing manuscript text.

### 2. Same-site CSRM common-input benchmark

Main manuscript figure:

- `manuscript/figures/fig05_common_input_benchmark.pdf`

Optional QC figure:

- `manuscript/figures/fig06_common_input_qc_examples.pdf`

Current n=128 summary:

| Method | Input | Mean Vs MAE (km/s) | Median Vs MAE (km/s) |
| --- | --- | ---: | ---: |
| QEDisp m0 multistart | Rayleigh phase | 0.104 | 0.100 |
| DI-Strong | Rayleigh phase | 0.108 | 0.106 |
| DI-Weak | Rayleigh phase | 0.138 | 0.128 |
| DispFormer phase-only | Rayleigh phase | 0.111 | 0.108 |
| DispFormer phase+group | Rayleigh phase + group | 0.105 | 0.101 |

Interpretation:

- DI-Strong is point-estimate competitive, not clearly better.
- QEDisp and DispFormer remain strong baselines.
- The manuscript claim should be "comparable point error plus auditable posterior samples", not "best accuracy".

### 3. Reduced same-prior MCMC posterior anchor

Appendix figures:

- `manuscript/figures/mcmc_anchor_inside_support_profiles.pdf`
- `manuscript/figures/mcmc_anchor_support_transition_profiles.pdf`

Inside-support validated summary:

| Quantity | Value |
| --- | ---: |
| usable MCMC cases | 6 / 6 |
| mean max R-hat | 1.14 |
| mean min ESS | 194 |
| mean posterior-predictive RMS | 0.013 km/s |
| mean DI-MCMC median Vs difference | 0.057 km/s |
| mean p05-p95 interval overlap | 0.58 |
| mean DI/MCMC p05-p95 width ratio | 1.21 |
| mean DI median Vs MAE to target | 0.072 km/s |
| mean MCMC median Vs MAE to target | 0.082 km/s |

Interpretation:

- Inside support, DI-Strong gives posterior medians and p05-p95 intervals broadly consistent with a reduced Bayesian MCMC reference.
- Near and outside support, the agreement degrades and one outside-support MCMC chain fails the convergence gate.
- This supports the audit framing: posterior samples need prior-support and calibration checks.

Important limitation:

- This is not an exact full-dimensional posterior for the original procedural generator. It uses a 10-knot Vs parameterisation, one median Vp/Vs parameter, a GMM prior fitted to projected DI-Strong prior samples and a Gaussian phase-velocity error scale.

### 4. Rayleigh/Love/joint input ablation

Supplement candidate:

- `figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.pdf`
- `docs/rayleigh_love_input_ablation_20260611.md`

Current n=128 summary:

| Method | Rayleigh-only Vs MAE | Love-only Vs MAE | Joint Vs MAE |
| --- | ---: | ---: | ---: |
| DI-Strong | 0.0786 | 0.0852 | 0.0757 |
| DI-Weak | 0.1446 | 0.1706 | 0.1417 |

Interpretation:

- Joint Rayleigh+Love input modestly helps or ties the best single-wave case.
- This is useful capability evidence, but it should not become the main paper claim.

## Recommended figure placement

Main text:

1. Fig. 1: method concept and prior/posterior audit logic.
2. Fig. 2: depth-control / parameterisation.
3. Fig. 3: prior-support production diagnostics.
4. Fig. 4: calibration/reliability summary.
5. Fig. 5: common-input QEDisp/DispFormer benchmark.

Optional main or supplement:

- Fig. 6 common-input QC examples. Keep in main only if the editor/reviewer needs visual proof of dispersion and model fits.

Appendix or supplement:

- Reduced same-prior MCMC posterior anchor.
- Rayleigh/Love/joint input ablation.
- Full reliability curves, rank/PIT and depth-binned tables.
- Field-data audit or short-period interpolation warning, if retained at all.

## Current manuscript build status

Both local manuscript directories compile:

- `SurfFlow/manuscript/gjilguid2e.pdf`
- `paper-overleaf/gjilguid2e.pdf`

The current compile has no undefined citations, undefined references, or missing figures. The only remaining layout warning is a tiny 0.32 pt overfull line near the observation-encoding sentence, which can be fixed in a final typography pass.

