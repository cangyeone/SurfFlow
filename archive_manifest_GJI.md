# GJI Reproducibility Archive Manifest

This manifest lists the files that should accompany the GJI submission
"Auditing Learned Posterior Samples for Surface-Wave Dispersion Inversion".
GitHub contains the manuscript-facing source, small CSV/JSON outputs, plotting
code and metadata. A separate DOI archive should hold the large model weights
and large diagnostic arrays that are intentionally omitted from GitHub.

## Manuscript Source

- `manuscript/gjilguid2e.tex`
- `manuscript/gjilguid2e.pdf`
- `manuscript/gjilguid2e.bbl`
- `manuscript/references.bib`
- `manuscript/gji.cls`
- `manuscript/gji.bst`
- `manuscript/gji_extra.sty`
- `manuscript/timet.sty`
- `manuscript/cover_letter_GJI.md`

## Manuscript Figures Used By The Current PDF

- `manuscript/figures/fig01_workflow_prior_posterior_audit_p5p95.pdf`
- `manuscript/figures/fig02_control_points.pdf`
- `manuscript/figures/fig03_direct_inversion_results_v9.pdf`
- `manuscript/figures/fig04_calibration_reliability_v4.pdf`
- `manuscript/figures/fig05_common_input_benchmark.pdf`
- `manuscript/figures/fig06_common_input_qc_examples.pdf`
- `manuscript/figures/mcmc_anchor_inside_support_profiles.pdf`
- `manuscript/figures/mcmc_anchor_support_transition_profiles.pdf`
- `manuscript/figures/fair_missing_band_uncertainty.pdf`
- `manuscript/figures/fair_missing_band_random_scatter.pdf`
- `manuscript/figures/fair_noise_sensitivity.pdf`

## Core Code, Configs And Plotting Scripts

- `README.md`
- `OMITTED_LARGE_CHECKPOINTS.txt`
- `archive_manifest_GJI.md`
- `configs/fair_di_strong_full.yaml`
- `configs/fair_di_weak_full.yaml`
- `configs/det_di_strong_full.yaml`
- `configs/det_di_weak_full.yaml`
- `scripts/train_di_fair.py`
- `scripts/eval_fair_di_comparison.py`
- `scripts/eval_fair_calibration.py`
- `scripts/eval_fair_missing_band.py`
- `scripts/eval_fair_noise_sensitivity.py`
- `scripts/eval_fair_sampling_sensitivity.py`
- `scripts/train_deterministic_di_fair.py`
- `scripts/eval_fair_baselines.py`
- `scripts/run_openswi_csrm_transfer.py`
- `scripts/run_qedisp_csrm_multistart_baseline.py`
- `scripts/make_gji_common_input_benchmark_figure.py`
- `scripts/run_mcmc_posterior_reference.py`
- `scripts/run_same_prior_mcmc_posterior.py`
- `scripts/plot_mcmc_posterior_anchor_figures.py`
- `scripts/run_rayleigh_love_input_ablation.py`
- `overleaf_disp_inv_scripts/make_fig03_direct_inversion_results.py`
- `overleaf_disp_inv_scripts/make_fig04_calibration_reliability.py`
- `overleaf_disp_inv_scripts/make_archive_inventory.py`
- `overleaf_disp_inv_scripts/plan_archive_bundle.py`
- `overleaf_disp_inv_scripts/prepare_archive_bundles.py`

## Small Result Tables And Protocol Files

- `results/fair_di_comparison/README.md`
- `results/fair_di_comparison/production/fair_di_metrics.csv`
- `results/fair_di_comparison/production/fair_di_metrics.json`
- `results/fair_di_comparison/production/fair_di_protocol.json`
- `results/fair_di_comparison/production/calibration/calibration_metrics.csv`
- `results/fair_di_comparison/production/calibration/calibration_protocol.json`
- `results/fair_di_comparison/production/calibration/depth_binned_coverage.csv`
- `results/fair_di_comparison/production/calibration/rank_diagnostics.csv`
- `results/fair_di_comparison/production/calibration/temperature_scaling.json`
- `results/fair_di_comparison/production/missing_band/missing_band_uncertainty.csv`
- `results/fair_di_comparison/production/missing_band/missing_band_uncertainty.json`
- `results/fair_di_comparison/production/noise/noise_sensitivity.csv`
- `results/fair_di_comparison/production/noise/noise_sensitivity.json`
- `results/fair_di_comparison/production/sampling_sensitivity/sampling_sensitivity.csv`
- `results/fair_di_comparison/production/sampling_sensitivity/sampling_sensitivity.json`
- `results/fair_di_comparison/production/baselines/baseline_metrics.csv`
- `results/fair_di_comparison/production/baselines/baseline_metrics.json`
- `results/openswi_csrm_transfer/n512_s32_step24/csrm_transfer_metrics.csv`
- `results/openswi_csrm_transfer/n512_s32_step24/csrm_transfer_metrics.json`
- `results/openswi_csrm_transfer/n512_s32_step24/csrm_transfer_protocol.json`
- `results/qedisp_csrm_multistart/sites128_starts64/combined_same_sites_metrics.csv`
- `results/qedisp_csrm_multistart/sites128_starts64/combined_same_sites_summary.json`
- `results/qedisp_csrm_multistart/sites128_starts64/qedisp_csrm_site_metrics.csv`
- `results/qedisp_csrm_multistart/sites128_starts64/qedisp_csrm_summary.json`
- `results/dispformer_openswi_csrm/sites128_official/dispformer_same_sites_metrics.csv`
- `results/dispformer_openswi_csrm/sites128_official/dispformer_same_sites_protocol.json`
- `results/dispformer_openswi_csrm/sites128_official/dispformer_same_sites_summary.json`
- `results/dispformer_openswi_csrm/sites128_official/same_sites_all_methods_metrics.csv`
- `results/dispformer_openswi_csrm/sites128_official/same_sites_all_methods_summary.json`

## Reduced MCMC Anchor And Rayleigh/Love Ablation

- `docs/same_prior_mcmc_posterior_anchor_20260611.md`
- `docs/rayleigh_love_input_ablation_20260611.md`
- `results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_metrics_combined.csv`
- `results/same_prior_mcmc_posterior/strong_gmm6_k10_validated_cases0_5/same_prior_mcmc_posterior_summary.json`
- `results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_metrics.csv`
- `results/same_prior_mcmc_posterior/strong_gmm6_k10_boundary_out_cases0_1_resume_s2600/same_prior_mcmc_posterior_protocol.json`
- `figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_inside_support_profiles.pdf`
- `figures/same_prior_mcmc_posterior/posterior_anchor_publication/mcmc_anchor_support_transition_profiles.pdf`
- `results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_metrics.csv`
- `results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_protocol.json`
- `figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.pdf`

## Checkpoint Metadata Committed To GitHub

Include all files under:

- `checkpoint_metadata/fair_di_strong_full_seed642026/`
- `checkpoint_metadata/fair_di_weak_full_seed642026/`
- `checkpoint_metadata/det_di_strong_full_seed642026/`
- `checkpoint_metadata/det_di_weak_full_seed642026/`

These directories contain configs, training logs, normalization summaries,
runtime metadata and best-checkpoint selection metadata, but not the model
weights.

## Large Artifacts For DOI Archive

The following files are intentionally omitted from GitHub and should be placed
in a DOI archive, or regenerated from the recorded commands:

- `ckpt/fair_di_strong_full_seed642026/best.pt`
- `ckpt/fair_di_weak_full_seed642026/best.pt`
- `ckpt/det_di_strong_full_seed642026/best.pt`
- `ckpt/det_di_weak_full_seed642026/best.pt`
- `results/fair_di_comparison/production/fair_di_diagnostics.npz`
- `results/openswi_csrm_transfer/*/csrm_transfer_diagnostics.npz`
- `results/dispformer_openswi_csrm/*/dispformer_same_sites_predictions.npz`
- `results/qedisp_csrm_multistart/**/qedispinv.h5`
- `results/same_prior_mcmc_posterior/**/same_prior_mcmc_posterior_diagnostics.npz`

## External Data And Baseline Software

- OpenSWI/OpenSWI-real is cited in the manuscript and used for the same-site
  CSRM benchmark. Do not duplicate it in the archive unless the data license
  permits redistribution.
- QEDispInv and DispFormer are external baselines. Archive the wrapper scripts,
  extracted CSV/JSON summaries and exact run protocols used in this manuscript;
  cite the original projects/papers for the upstream software.

## Field-Data Scope

No calibrated field-data posterior validation is used as manuscript evidence.
The short-period field products generated during development are retained as
internal stress-test artifacts only and are not part of the main GJI claim.
