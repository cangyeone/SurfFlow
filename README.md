# SurfFlow

Probabilistic surface-wave dispersion inversion using a depth-controlled
conditional rectified flow.

This repository contains the manuscript-facing code and small artifacts for the
GJI submission on probabilistic surface-wave dispersion inversion. The training,
evaluation, comparison, plotting, manuscript and reproducibility files are kept
at the repository root rather than inside an extra publish subdirectory.

The current manuscript framing is an audit-oriented one: SurfFlow is treated as
a learned posterior sampler for a stated synthetic inversion problem, not as a
general field-data foundation model. The most recent code, figure and experiment
sync notes are in `docs/GJI_CODE_SYNC.md`; the current GJI revision status after
the reduced MCMC posterior-anchor experiment is in
`docs/gji_revision_status_after_mcmc_anchor_20260611.md`.

## Contents

- `configs/`: matched DI-Strong, DI-Weak and deterministic baseline configs.
- `scripts/`: training, evaluation, calibration, baseline, missing-band,
  noise-sensitivity, common-input benchmark and manuscript-update scripts.
- `training_entrypoints/`: top-level historical and v1.3 training entrypoints.
- `models/` and `utils/`: model and synthetic-prior/data-generation support.
- `overleaf_disp_inv_scripts/`: plotting and diagnostic helpers used by
  manuscript figure generation.
- `results/`: CSV/JSON/log outputs used by the manuscript tables and diagnostics.
- `figures/`: generated comparison, calibration, noise, missing-band and field
  figures copied from the production run.
- `manuscript/`: GJI LaTeX source, bibliography, cover letter and compiled PDF.
- `field_masw_results_fair_weak/`: small field-summary CSV/JSON products.
- `checkpoint_metadata/`: configs, training logs, best-selection metadata,
  normalization summaries and runtime metadata. Large model weights are omitted.

## Large Files Omitted

Model checkpoint weights larger than 2 MB are intentionally not committed to
GitHub. The omitted files are listed in `OMITTED_LARGE_CHECKPOINTS.txt`.
The repository keeps small checkpoint metadata such as `source_config.yaml`,
`config_resolved.json`, `epoch_metrics.csv`, `runtime_metadata.json`,
`best_selection.json`, `training_complete.json` and normalization summaries.

Large diagnostic arrays and per-site inversion products are also omitted from
this GitHub package:

- `results/fair_di_comparison/production/fair_di_diagnostics.npz`
- `field_masw_results_fair_weak/bayan_obo_masw_dnn_posterior_volume.npz`
- `results/openswi_csrm_transfer/*/csrm_transfer_diagnostics.npz`
- `results/dispformer_openswi_csrm/*/dispformer_same_sites_predictions.npz`
- `results/qedisp_csrm_multistart/**/qedispinv.h5`

These files should be placed in an external reproducibility archive if needed.

## Reproduction Entry Points

From the project root, the primary production commands are:

```bash
python scripts/train_di_fair.py --config configs/fair_di_strong_full.yaml
python scripts/train_di_fair.py --config configs/fair_di_weak_full.yaml
python scripts/eval_fair_di_comparison.py --help
python scripts/eval_fair_calibration.py --help
python scripts/eval_fair_noise_sensitivity.py --help
python scripts/eval_fair_missing_band.py --help
python scripts/eval_fair_sampling_sensitivity.py --help
python overleaf_disp_inv_scripts/make_fig01_three_row_main.py
python overleaf_disp_inv_scripts/make_fig02_control_points_refined.py
python overleaf_disp_inv_scripts/make_fig03_direct_inversion_results.py
python overleaf_disp_inv_scripts/make_fig04_calibration_reliability.py
python scripts/make_gji_common_input_benchmark_figure.py
python scripts/run_same_prior_mcmc_posterior.py --help
python scripts/plot_mcmc_posterior_anchor_figures.py --help
python scripts/run_rayleigh_love_input_ablation.py --help
```

The full production run used matched strong/weak training budgets and is
documented in `results/fair_di_comparison/README.md`.

The same-site CSRM benchmark against QEDisp and DispFormer is documented by
the wrapper scripts in `scripts/` and the small CSV/JSON summaries under
`results/openswi_csrm_transfer/`, `results/qedisp_csrm_multistart/` and
`results/dispformer_openswi_csrm/`.

The reduced same-prior MCMC posterior anchor and Rayleigh/Love input ablation
are documented in:

- `docs/same_prior_mcmc_posterior_anchor_20260611.md`
- `docs/rayleigh_love_input_ablation_20260611.md`
