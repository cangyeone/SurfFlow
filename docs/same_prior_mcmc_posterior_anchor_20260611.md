# Reduced Bayesian posterior anchor for SurfFlow DI

Date: 2026-06-11

## Purpose

This experiment tests a narrow but important validation claim:

> Does the DI-Strong posterior sampler give posterior medians and intervals that are broadly consistent with a conventional Bayesian posterior reference when the prior, forward solver, observation mask and noise model are matched as closely as practical?

The answer is yes for inside-support synthetic cases, with important limits.

The reference is not an exact full-dimensional Bayesian posterior for the original procedural Earth generator. It is a reduced-dimensional Bayesian reference:

- Vs at 10 depth knots: 0, 2, 5, 10, 20, 35, 50, 75, 100 and 127.5 km
- one median Vp/Vs parameter
- a full-covariance GMM fitted to 2500 projections from the DI-Strong synthetic prior
- the same Rayleigh-Love m0 period mask
- Gaussian phase-velocity error with sigma_c = 0.10 km/s
- affine-invariant stretch-move MCMC

Use the wording "reduced Bayesian reference" or "reduced same-prior MCMC reference". Do not call it the exact full Bayesian posterior.

## Conceptual framing

The GPT assessment is basically right: SurfFlow DI is not a deterministic regression from dispersion curve to one velocity model. It is better framed as amortized neural posterior estimation for a simulator-defined surface-wave inverse problem.

The learned target is:

```text
q_theta(m | d, q) approximates p_syn(m | d, q)
```

where `p_syn` is defined by the synthetic structural prior, surface-wave forward solver, observation mask and noise model. The posterior is therefore conditional on the synthetic data-generating assumptions. It is not an unconditional posterior for the real Earth.

This is exactly why the manuscript needs prior-support, calibration, posterior predictive and MCMC-anchor checks. Rectified flow supplies a conditional generative sampler; it does not by itself prove that the samples are calibrated posterior samples.

## Key outputs

Inside-support validated metrics:

```text
results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_metrics_combined.csv
results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_summary.json
```

Boundary/outside-support metrics:

```text
results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_metrics.csv
results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_diagnostics.npz
```

Clean publication-style figures:

```text
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_inside_support_profiles.png
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_inside_support_profiles.pdf
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_inside_support_profiles.svg
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_support_transition_profiles.png
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_support_transition_profiles.pdf
figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_support_transition_profiles.svg
```

Replot command:

```bash
MPLCONFIGDIR=/private/tmp/yzy_mpl_cache \
/opt/miniconda3/envs/seisloc/bin/python scripts/plot_mcmc_posterior_anchor_figures.py
```

## Result summary

Six inside-support synthetic cases passed the MCMC diagnostic gates.

| Quantity | Inside support |
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

Near-edge cases also passed the MCMC diagnostic gates, but DI and MCMC intervals differ more.

| Quantity | Near edge |
| --- | ---: |
| usable MCMC cases | 2 / 2 |
| mean max R-hat | 1.12 |
| mean min ESS | 220 |
| mean posterior-predictive RMS | 0.032 km/s |
| mean DI-MCMC median Vs difference | 0.124 km/s |
| mean p05-p95 interval overlap | 0.48 |
| mean DI/MCMC p05-p95 width ratio | 2.15 |

Outside-support behavior is mixed, which is scientifically useful rather than a failure of the experiment.

| Quantity | Outside support |
| --- | ---: |
| usable MCMC cases | 1 / 2 |
| mean posterior-predictive RMS | 0.042 km/s |
| mean DI-MCMC median Vs difference | 0.186 km/s |
| mean p05-p95 interval overlap | 0.53 |
| mean DI/MCMC p05-p95 width ratio | 1.64 |
| note | one case has max R-hat about 30.9, so it is diagnostic-only |

## Manuscript interpretation

The strongest supported claim is:

> For inside-support synthetic examples, DI-Strong provides posterior medians and p5-p95 intervals that are broadly consistent with a reduced Bayesian MCMC reference, while avoiding a new MCMC run at test time.

The experiment does not support:

> DI-Strong exactly recovers the full high-dimensional Bayesian posterior of the procedural Earth generator.

For near-edge and outside-support examples, the result supports a different claim:

> Posterior samples must be interpreted together with prior-support and calibration diagnostics. A good posterior predictive fit alone does not guarantee that the inferred Earth model is geologically reliable when the target lies near or outside the synthetic support.

This distinction is central to the GJI story. The paper should be positioned as probabilistic posterior sampling plus reliability auditing, not as a universal foundation model for all periods and all Earth structures.

## Recommended placement

Main text:

- Mention the reduced Bayesian reference in the Results as a validation anchor.
- Use one concise sentence in the Methods explaining the 10-knot + Vp/Vs MCMC parameterization.
- Keep the main claim moderate: "broadly consistent", not "identical".

Supplement:

- Put `mcmc_anchor_inside_support_profiles` as the main MCMC-anchor figure.
- Put `mcmc_anchor_support_transition_profiles` as a support-regime diagnostic.
- Include the full metrics table with R-hat, ESS, posterior predictive RMS, median differences, interval overlap and width ratio.

Suggested caption language:

> Reduced Bayesian posterior anchor. The MCMC reference uses a 10-knot Vs parameterization and a GMM prior fitted to projections from the DI-Strong synthetic prior. Inside the synthetic support, DI-Strong and MCMC give similar posterior medians and comparable p5-p95 intervals. Near and outside the support, posterior similarity degrades and one outside-support MCMC chain fails the convergence gate, illustrating why prior-support and calibration diagnostics are needed before interpreting posterior samples.
