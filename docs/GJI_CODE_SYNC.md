# GJI Code, Figure and Experiment Sync Notes

Date: 2026-06-12

This note is the repository-side handoff for the current GJI draft. It records
which scripts, small result summaries and figures should stay aligned between
GitHub, the local manuscript tree and Overleaf.

## Manuscript Position

The paper should be read as an audit-oriented method paper. SurfFlow is a
conditional rectified-flow sampler for a stated synthetic surface-wave inverse
problem. The main claim is not that it is universally more accurate or more
general than QEDispInv or DispFormer. The main claim is that learned posterior
samples can be generated quickly and then checked through:

- prior-support stress tests,
- empirical coverage and posterior-temperature diagnostics,
- posterior-predictive dispersion residuals,
- missing-band and noise sensitivity checks,
- reduced prior-matched MCMC posterior anchors,
- common-input QEDisp/DispFormer baselines.

Use qualified wording such as:

- `learned posterior under the stated synthetic training distribution`
- `synthetic-distribution posterior`
- `conditional posterior sampler`
- `posterior samples conditioned on the prior, forward solver, mask and noise assumption`
- `reduced prior-matched MCMC reference`

Avoid unqualified claims that the network returns a field-calibrated Bayesian
posterior or a general-purpose surface-wave foundation model.

## Current Main-Text Figure Map

Run commands from the repository root. The local working environment is the
`seisloc` conda environment, normally accessed as
`/opt/miniconda3/envs/seisloc/bin/python`. If `disba`/`numba` cache writes fail
on this machine, set:

```bash
export MPLCONFIGDIR=/private/tmp/yzy_mpl_cache
export NUMBA_CACHE_DIR=/private/tmp/yzy_numba_cache
```

| Figure | Manuscript file | Regeneration entry point | Role |
| --- | --- | --- | --- |
| Fig. 1 | `manuscript/figures/fig01_workflow_prior_posterior_audit_p5p95.pdf` | `overleaf_disp_inv_scripts/make_fig01_three_row_main.py` | Conceptual method and audit workflow |
| Fig. 2 | `manuscript/figures/fig02_control_points.pdf` | `overleaf_disp_inv_scripts/make_fig02_control_points_refined.py` | Depth controls and model parameterisation |
| Fig. 3 | `manuscript/figures/fig03_direct_inversion_results_v9.pdf` | `overleaf_disp_inv_scripts/make_fig03_direct_inversion_results.py` | DI-Strong/DI-Weak production diagnostics |
| Fig. 4 | `manuscript/figures/fig04_calibration_reliability_v4.pdf` | `overleaf_disp_inv_scripts/make_fig04_calibration_reliability.py` | Calibration/reliability summary |
| Fig. 5 | `manuscript/figures/fig05_common_input_benchmark.pdf` | `scripts/make_gji_common_input_benchmark_figure.py` | Same-site QEDisp/DispFormer/DI benchmark |
| Fig. 6 | `manuscript/figures/fig06_common_input_qc_examples.pdf` | `scripts/make_csrm_same_site_qc_examples.py` | Visual QC examples for common-input benchmark |

Missing-band and additive-noise diagnostic figures are now Appendix figures, not
main-text figures.

The current manuscript text uses the reader-facing regime labels:

- `inside support`
- `near edge`
- `outside support`

Do not reintroduce the older labels `in-prior`, `near-boundary` or
`out-of-prior` in captions or prose unless referring to internal file names.

## Common-Input Benchmark

Small tracked summaries:

```text
results/openswi_csrm_transfer/n512_s32_step24/
results/qedisp_csrm_multistart/sites128_starts64/
results/dispformer_openswi_csrm/sites128_official/
```

Current n=128 summary:

| Method | Input | Mean Vs MAE (km/s) | Median Vs MAE (km/s) |
| --- | --- | ---: | ---: |
| QEDisp m0 multistart | Rayleigh phase | 0.104 | 0.100 |
| DI-Strong | Rayleigh phase | 0.108 | 0.106 |
| DI-Weak | Rayleigh phase | 0.138 | 0.128 |
| DispFormer phase-only | Rayleigh phase | 0.111 | 0.108 |
| DispFormer phase+group | Rayleigh phase + group | 0.105 | 0.101 |

Interpretation:

- DI-Strong is point-estimate competitive, not clearly more accurate.
- QEDisp and DispFormer remain strong baselines.
- The manuscript claim should be "comparable point error plus auditable
  posterior samples", not "best accuracy".

## Reduced Same-Prior MCMC Posterior Anchor

This is now the preferred posterior-reference experiment. It supersedes the
earlier ABC-style posterior anchor for the core manuscript claim.

Scripts:

```text
scripts/run_mcmc_posterior_reference.py
scripts/run_same_prior_mcmc_posterior.py
scripts/plot_mcmc_posterior_anchor_figures.py
```

Clean publication figures:

```text
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_inside_support_profiles.*
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_support_transition_profiles.*
manuscript/figures/mcmc_anchor_inside_support_profiles.pdf
manuscript/figures/mcmc_anchor_support_transition_profiles.pdf
```

Small metrics/protocol outputs to keep:

```text
results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_metrics_combined.csv
results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_summary.json
results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_metrics.csv
results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_protocol.json
```

Large local-only output:

```text
results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_diagnostics.npz
```

The `.npz` file is required to replot the transition figure without rerunning
MCMC, but it is intentionally omitted from GitHub. Put it in an external
archive or rerun the MCMC command if a fully reproducible package is needed.

Replot command from saved diagnostics:

```bash
MPLCONFIGDIR=/private/tmp/yzy_mpl_cache \
NUMBA_CACHE_DIR=/private/tmp/yzy_numba_cache \
/opt/miniconda3/envs/seisloc/bin/python scripts/plot_mcmc_posterior_anchor_figures.py
```

Boundary/outside-support MCMC resume command used for the current diagnostics:

```bash
MPLCONFIGDIR=/private/tmp/yzy_mpl_cache \
NUMBA_CACHE_DIR=/private/tmp/yzy_numba_cache \
PYTHONUNBUFFERED=1 \
/opt/miniconda3/envs/seisloc/bin/python scripts/run_same_prior_mcmc_posterior.py \
  --cases 2 \
  --regimes boundary,out-of-prior \
  --n-prior-samples 2500 \
  --gmm-components 6 \
  --gmm-iters 80 \
  --knot-depths-km 0 2 5 10 20 35 50 75 100 127.5 \
  --n-init 768 \
  --n-walkers 24 \
  --n-steps 2600 \
  --burnin 300 \
  --thin 5 \
  --resume-starts-npz results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_s1600/same_prior_mcmc_posterior_diagnostics.npz \
  --di-samples 96 \
  --di-steps 24 \
  --pred-max-samples 128 \
  --out-dir results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600 \
  --fig-dir figures/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600 \
  --progress-every 200
```

Inside-support summary:

| Quantity | Value |
| --- | ---: |
| usable MCMC cases | 6 / 6 |
| mean max R-hat | 1.14 |
| mean min ESS | 194 |
| mean posterior-predictive RMS | 0.013 km/s |
| mean DI-MCMC median Vs difference | 0.057 km/s |
| mean p05-p95 interval overlap | 0.58 |
| mean DI/MCMC p05-p95 width ratio | 1.21 |

Interpretation:

- Inside support, DI-Strong gives posterior medians and p05-p95 bands broadly
  consistent with the reduced MCMC reference.
- Near and outside support, agreement degrades, which supports the prior-support
  audit message.
- This is not the exact full-dimensional Bayesian posterior for the original
  procedural Earth generator.

## Rayleigh/Love Input-Channel Ablation

This is a supplement candidate. It checks that the same DI architecture can use
Rayleigh-only, Love-only or joint Rayleigh+Love m0 inputs under the matched
synthetic setup.

Script:

```text
scripts/run_rayleigh_love_input_ablation.py
```

Formal n=128 command:

```bash
MPLCONFIGDIR=/private/tmp/yzy_mpl_cache \
NUMBA_CACHE_DIR=/private/tmp/yzy_numba_cache \
PYTHONUNBUFFERED=1 \
/opt/miniconda3/envs/seisloc/bin/python scripts/run_rayleigh_love_input_ablation.py \
  --n-test 128 \
  --posterior-samples 64 \
  --euler-steps 24 \
  --batch-size 16 \
  --out-dir results/rayleigh_love_input_ablation/n128_s64_steps24 \
  --fig-dir figures/rayleigh_love_input_ablation/n128_s64_steps24
```

Small outputs:

```text
results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_metrics.csv
results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_protocol.json
figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.*
```

Current n=128 result:

| Method | Rayleigh-only Vs MAE | Love-only Vs MAE | Joint Vs MAE |
| --- | ---: | ---: | ---: |
| DI-Strong | 0.079 | 0.085 | 0.076 |
| DI-Weak | 0.145 | 0.171 | 0.142 |

Interpretation:

- Joint input modestly helps or ties the best single-wave case.
- This is capability evidence, not the main contribution.
- Do not present this as a general multimode Rayleigh/Love foundation model.

## Files Not Suitable for GitHub

The following products should stay local or move to a separate reproducibility
archive:

- model checkpoint weights: `best.pt`, `latest.pt`, `*.ckpt`, `*.pth`
- large arrays: `*.npz`, `*.npy`
- QEDisp per-site `qedispinv.h5` products
- external OpenSWI/DispFormer repositories and official checkpoints
- external QEDisp/QEDispInv build trees
- large exploratory figure batches that are not cited by the manuscript

The repository should track scripts, configs, small CSV/JSON summaries, final
small manuscript figures and the LaTeX source. Large binary diagnostics should
be listed here and archived separately.

## Current Overleaf State

The local Overleaf clone at:

```text
/Users/liuxin/Documents/yzy_directSWI/paper-overleaf
```

has been updated and pushed to the Overleaf git remote with commit:

```text
870105a Add MCMC posterior anchor and updated audit figures
```

The Overleaf version compiles locally with no undefined citations, references or
missing figures. The only remaining warning is a tiny 0.32 pt overfull hbox near
the observation-encoding sentence.

## Next Work

1. Keep Fig. 6 as main-text QC only if the manuscript needs visual support for
   the common-input benchmark; otherwise move it to supplement.
2. Keep the Rayleigh/Love ablation in supplement unless reviewers ask for more
   input-channel evidence.
3. Add a small depth-control or smoothness-loss ablation only if the narrow
   posterior intervals become a reviewer concern.
4. Prepare a separate archive for omitted `.npz`, checkpoint and QEDisp per-site
   files before final submission.
