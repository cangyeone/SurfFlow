# SurfFlow phase figures and saved-sample checks — v3

Six main and five supplementary figures for the regional fundamental-mode Rayleigh phase study. This update adds the fixed-subset admission sensitivity and period-design-matched prior-width comparison. The core inference/results record remains https://github.com/cangyeone/SurfFlow/releases/tag/phase-20261117-v1 . Earlier figure v2 remains unchanged.

## Reproduce from saved samples
From the extracted directory:
```
python tools/analyze_admission_and_prior_20261117.py
python tools/build_redesigned_figures_20261117.py
python tools/package_figure_review_20261117.py
```
These commands use shipped inputs and require no network, neural inference, forward solver, Monte Carlo chain or training run. They regenerate the admission/prior CSVs, eleven figures and an eleven-page vector figure book, and check 27 frozen summaries. `SHA256.json` records the shipped files. PDF creation metadata can differ after regeneration.

Tested environment: Python 3.12.12; numpy 2.4.2; scipy 1.17.0; pandas 3.0.1; matplotlib 3.10.8; cartopy 0.25.0; xarray 2026.4.0; pypdf 6.12.2; netCDF4. The scripts install nothing.

## Interpretation and provenance
The main structural comparisons retain all 4096 neural outputs at each detailed field case. Four original prediction files preserve the 512 fixed evaluated indices, latent values and success/failure outcomes. The admission check compares those same 512 before/after removing unavailable predictions, without replacing samples. It does not certify the untested outputs. The public files do not contain the per-draw failure-status subclasses, which are not inferred.

The prior-width display divides pointwise network 5–95% widths by widths from the matching 27- or 17-period admitted prior. It uses 256 outputs/location and 512 models/prior design. It measures empirical interval contraction, not vertical resolution, a formal information-gain quantity or calibrated field coverage. The short-design shallow-average undercoverage and 96-degree residual mismatch remain.

`figure_data.npz` preserves regional/reference summaries and two previously documented synthetic display replays. `presentation_context.npz` preserves the earlier prior/error-scale summaries. The new evidence directory contains saved-input hashes, fixed-subset outcomes, result tables and width-ratio arrays. The extraction/export scripts document original sources but are not needed for the offline workflow. Full checkpoint replay and reference chains remain in the v1 core package.

## Geographic background
`map_source.json` identifies the small GMT Earth Relief crop (Tozer et al., 2019; doi:10.1029/2019EA000658). The public-domain Natural Earth 110 m coastline is bundled to avoid downloads (https://www.naturalearthdata.com/about/terms-of-use/). The CSRM Moho is published context, not an inferred boundary or independent truth.

This archive contains no manuscript or confidential correspondence. GitHub main and previous archives remain unchanged.

Archive SHA256: `68fa713f2d546b07a6b8fd1cd9318930e4b39741fc436f8f9e5199c16fca0a2e`.
