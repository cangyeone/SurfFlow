# Rayleigh/Love input-channel ablation

Date: 2026-06-11

## Purpose

This experiment checks whether SurfFlow DI can condition on Rayleigh-only, Love-only and Rayleigh+Love m0 inputs under the same synthetic setup.

This is an auxiliary capability check. It should not be presented as the main contribution of the paper. The main GJI story remains amortized posterior sampling plus reliability auditing.

## Run

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

## Outputs

```text
results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_metrics.csv
results/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation_protocol.json
figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.png
figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.pdf
figures/rayleigh_love_input_ablation/n128_s64_steps24/rayleigh_love_input_ablation.svg
```

## Result summary

| Method | Input | Vs MAE (km/s) | Vs RMSE (km/s) | p05-p95 coverage | mean Vs std (km/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| DI-Strong | Rayleigh | 0.079 | 0.118 | 0.845 | 0.088 |
| DI-Strong | Love | 0.085 | 0.121 | 0.822 | 0.093 |
| DI-Strong | Rayleigh+Love | 0.076 | 0.110 | 0.851 | 0.086 |
| DI-Weak | Rayleigh | 0.145 | 0.204 | 0.825 | 0.152 |
| DI-Weak | Love | 0.171 | 0.235 | 0.798 | 0.172 |
| DI-Weak | Rayleigh+Love | 0.142 | 0.196 | 0.820 | 0.152 |

## Interpretation

Rayleigh+Love joint input gives the best or tied-best Vs MAE and RMSE for both DI-Strong and DI-Weak, but the improvement is modest.

This supports a restrained statement:

> Under the matched synthetic m0 setup, the same DI architecture can condition on Rayleigh-only, Love-only or joint Rayleigh+Love observations, and joint input gives a small improvement in median Vs accuracy.

It does not support a broad statement that SurfFlow is a general multimode Rayleigh/Love inversion foundation model. The current comparison is m0 only and within the synthetic training assumptions.

## Recommended placement

Supplement is the safest placement. In the main text, mention it briefly if discussing observation masks or input-channel flexibility. Do not let this ablation compete with the main reliability-audit narrative.
