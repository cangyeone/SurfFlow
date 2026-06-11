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

## Posterior Anchor and Input Ablation

Two manuscript-supporting pilot experiments are now tracked in GitHub.

### Traditional ABC posterior anchor vs DI posterior samples

Script:

```bash
/opt/miniconda3/envs/seisloc/bin/python scripts/run_traditional_posterior_anchor.py \
  --n-prior-draws 4096 \
  --cases-per-regime 2 \
  --priors strong,weak \
  --input-mode joint \
  --sigma-c 0.10 \
  --abc-keep 128 \
  --di-samples 64 \
  --di-steps 24 \
  --di-forward-max-samples 0 \
  --out-dir results/traditional_posterior_anchor/joint_n4096_abc128_di64_modelonly \
  --fig-dir figures/traditional_posterior_anchor/joint_n4096_abc128_di64_modelonly \
  --progress-every 512
```

Tracked outputs:

- `results/traditional_posterior_anchor/joint_n4096_abc128_di64_modelonly/posterior_anchor_metrics.csv`
- `results/traditional_posterior_anchor/joint_n4096_abc128_di64_modelonly/posterior_anchor_protocol.json`
- `figures/traditional_posterior_anchor/joint_n4096_abc128_di64_modelonly/posterior_anchor_vs_di_summary_joint.*`
- six case-level figures under the same figure directory.

The large `posterior_anchor_diagnostics.npz` file is intentionally local-only.

Interpretation: this is a same-prior, same-forward prior-predictive ABC anchor,
not a full transdimensional MCMC reference. It shows that DI median profiles are
often more accurate than the ABC posterior median in these cases, but DI
p05-p95 intervals are narrower than the same-prior ABC anchor. This supports
using DI as a fast learned posterior sampler only when paired with calibration
and prior-support audits.

We do not need to download or depend on Zhang and Curtis' VIP code for this
specific comparison. An external variational-inference package would introduce
a different parameterisation and prior, whereas this anchor is designed to keep
the inversion problem identical to SurfFlow.

### Rayleigh/Love input-channel ablation

Script:

```bash
/opt/miniconda3/envs/seisloc/bin/python scripts/run_rayleigh_love_input_ablation.py \
  --n-test 64 \
  --posterior-samples 32 \
  --euler-steps 16 \
  --batch-size 16 \
  --methods DI-Strong,DI-Weak \
  --out-dir results/rayleigh_love_input_ablation/n64_s32_steps16 \
  --fig-dir figures/rayleigh_love_input_ablation/n64_s32_steps16
```

Tracked outputs:

- `results/rayleigh_love_input_ablation/n64_s32_steps16/rayleigh_love_input_ablation_metrics.csv`
- `results/rayleigh_love_input_ablation/n64_s32_steps16/rayleigh_love_input_ablation_protocol.json`
- `figures/rayleigh_love_input_ablation/n64_s32_steps16/rayleigh_love_input_ablation.*`

The current pilot shows a modest but consistent improvement from joint
Rayleigh+Love input relative to either single-wave input.

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

1. Scale the posterior anchor beyond the current 2 cases/regime only if it is
   needed for a supplement table.
2. Re-run the Rayleigh/Love input-channel ablation at n=128, samples=64,
   steps=24 if the ablation moves from supplement to main text.
3. Add a lightweight depth-control or smoothness-loss ablation if reviewers are
   likely to question whether narrow posteriors are caused by the parameter
   basis or loss regularisation rather than the data.
