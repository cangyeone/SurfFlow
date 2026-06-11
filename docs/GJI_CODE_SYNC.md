# GJI Code and Figure Sync Notes

This note records the manuscript-facing files that should stay aligned between
GitHub and Overleaf while the GJI draft is being revised.

## Manuscript Position

The current paper should be read as an audit-oriented method paper. SurfFlow is
used as a conditional-flow sampler for a stated synthetic inversion problem. The
main claim is not that rectified flow is universally more accurate than QEDisp
or DispFormer. The main claim is that learned posterior samples can be generated
quickly and then checked through prior-support tests, empirical coverage,
posterior-temperature diagnostics, missing-band sensitivity, noise sensitivity
and common-input baselines.

Use qualified language such as:

- `learned posterior under the stated synthetic training distribution`
- `synthetic-distribution posterior`
- `conditional posterior sampler`
- `posterior samples conditioned on the prior, forward solver, mask and noise assumption`

Avoid unqualified claims that the network returns a field-calibrated Bayesian
posterior.

## Figure Regeneration

Run commands from the repository root. The local working convention is the
`seisloc` conda environment:

```bash
conda run -n seisloc python overleaf_disp_inv_scripts/make_fig01_three_row_main.py
conda run -n seisloc python overleaf_disp_inv_scripts/make_fig02_control_points_refined.py
conda run -n seisloc python overleaf_disp_inv_scripts/make_fig03_direct_inversion_results.py
conda run -n seisloc python overleaf_disp_inv_scripts/make_fig04_calibration_reliability.py
conda run -n seisloc python scripts/make_gji_common_input_benchmark_figure.py
```

The current manuscript also uses a same-site CSRM QC example figure. Regenerate
it with:

```bash
conda run -n seisloc python scripts/make_csrm_same_site_qc_examples.py
```

That script needs diagnostic arrays that are intentionally not stored in GitHub
because they are large. See the omitted-file list below.

## Common-Input Benchmark Files

Small CSV/JSON result summaries used by the current manuscript are tracked in
GitHub:

- `results/openswi_csrm_transfer/n512_s32_step24/`
- `results/qedisp_csrm_multistart/sites128_starts64/`
- `results/dispformer_openswi_csrm/sites128_official/`

The benchmark currently compares the following:

- SurfFlow DI-Strong and DI-Weak on the same CSRM sites.
- QEDisp fundamental-mode Rayleigh multistart inversion.
- DispFormer phase-only on the same Rayleigh phase input.
- DispFormer phase+group as a method-native reference with extra input.

Interpretation: the comparison supports comparable point-estimate performance
for DI-Strong, while the main SurfFlow contribution remains posterior sampling
and reliability checking.

## Large Files Not Tracked by GitHub

The following products are local/archive files rather than GitHub files:

- model checkpoints, including `best.pt`, `latest.pt`, `*.ckpt` and `*.pth`
- large diagnostic arrays such as `*.npz` and `*.npy`
- QEDisp per-site `qedispinv.h5` files
- external OpenSWI/DispFormer repositories and their official checkpoints
- external QEDisp/QEDispInv build trees

These files should be regenerated from the scripts or placed in a separate
reproducibility archive before final submission.

## Next Experiments

The next GJI-strengthening experiments should be kept separate from the current
code-sync commit:

1. A small traditional Bayesian posterior anchor for 3-6 synthetic cases.
2. A Rayleigh-only, Love-only and Rayleigh+Love synthetic input ablation.
3. A lightweight depth-control or smoothness-loss ablation.

For the Bayesian posterior anchor, we do not need to reproduce Zhang and Curtis'
INN implementation directly. The useful comparison is a traditional sampler run
under our own prior, forward solver, parameterisation and noise assumption, so
that SurfFlow and the reference posterior share the same inversion problem.
