# SurfFlow phase-only reproducibility record — 2026-11-17 v1

This frozen record supports the regional Rayleigh-wave phase-velocity study. Use this release asset, not older results on the repository's main branch, for the current 28-Vs-control models.

Included: the exact two trained checkpoints; their source and model configuration; observed curves/scales at all 85 CSRM grid locations; saved regional and detailed neural samples; long/short synthetic calibration truths and structural samples; the two qualified same-target real-data reference chains and diagnostics; the CheckedQED source/runtime and its notices; compact figure data and figure/table scripts; SHA256 manifest; a portable result/checkpoint replay entry.

## Run

Use Python 3.12 with dependencies in `environment.txt`. No installation or network access occurs during replay.

```
python reproduce.py --output replay-summary --summaries-only
python reproduce.py --output replay-gpu --device cuda:0 --forward-check
```

The first command verifies every packaged hash and reconstructs calibration scores, regional depth-average summaries and reference comparisons from arrays. The second additionally runs both checkpoints for all 85 locations and the two 4096-draw real examples, and checks two phase predictions with the packaged Linux binary. Output directories must be new. Run these commands from any working directory; paths are relative to this script, not a laboratory server path.

On the tested A100/PyTorch 2.5.1+cu121 environment, reproduction of saved draws is checked at 2e-6 km/s. CPU uses a different random stream and is supported for generating samples, without promising elementwise agreement with CUDA samples. Hardware/software changes can alter floating-point results. The checked runtime is Linux x86_64; `build_checked_runtime.py` reproduces its build commands with relocated source paths (requires g++ and gfortran). Preserve the supplied binary if rebuilding; its manifest will then need a separate provenance record.

`paper_assets/tools/build_phase_figures_20261117.py` and the table scripts regenerate the plotted summaries. They need pandas, Matplotlib and Cartopy; the map panel uses Natural Earth 110m coastlines. This record's main verification does not require those plotting dependencies or external map downloads.

## Scientific scope

The main problem is fundamental-mode Rayleigh phase dispersion, 28 fixed Vs controls, Vp=1.75Vs and Brocher density. A/B are averages over 10–30/30–50 km; Delta=B−A. CSRM locations are map grid points, not independent subarrays. The source training bank also contains group-only and mixed designs; known group-label issues and the one-seed-per-model limitation are retained in the study. Short-design A coverage is imperfect. This package does not certify all posterior coordinates or actual Earth truth.

The million-model training bank is not duplicated in this inference/results archive. Its SHA256 is `76e21e0f9d52b49785c2c15f8188cbe8ec9d490fff7cb534019738b9d4135f17`. Frozen generation/training code and config are included as provenance; their historical orchestration contains machine-specific paths. From-scratch training regeneration and the historical development experiments are outside the tested portable entry. The published-scale reference chains are included; provisional and historical chains are not claimed as qualified references here.

Data sources: CSRM Level-2 dispersion maps and their released error scales; OpenSWI observation interface; published CSRM structure at https://doi.org/10.5281/zenodo.11098135. Exact source references and compact evidence are in `paper_assets/evidence/phase_20261117`. The chains are posterior reference samples under the study's working likelihood, not additional field observations.

See `THIRD_PARTY.md` for preserved component notices. No new blanket license is assigned to upstream materials.


Release asset SHA256: `559e4f9a37fc862156874921215132cf1f8f10c68c1d30e51ab8285fa4b88f29`.
