# Overleaf Manuscript Workflow

The manuscript is maintained in
[Overleaf](https://www.overleaf.com/project/6af12bf703ca6c90d7b4f430).
GitHub holds the code, small result tables and experiment documentation, not the
manuscript source, bibliography, journal styles or submission PDF.

## Separate Checkout

Clone the manuscript outside the SurfFlow code repository:

```bash
git clone https://git@git.overleaf.com/6af12bf703ca6c90d7b4f430 SurfFlow-paper
cd SurfFlow-paper
```

The Overleaf Git remote is the only manuscript remote. Keep paper commits and
code commits separate; do not copy the manuscript back into GitHub.

## Compile And Edit

Select `main.tex` as the Overleaf main document and pdfLaTeX as the compiler.
The entry point includes `gjilguid2e.tex`, which contains the manuscript body.
Bibliography, journal styles and `figures/` are in the Overleaf project root.

For a local TeX installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The generated PDF is `build/main.pdf`. The imported `gjilguid2e.pdf` and
`gjilguid2e.bbl` are pre-migration snapshots, not automatically updated outputs.
Before editing locally, pull changes from Overleaf; push manuscript commits only
to that checkout's Overleaf remote.

## Migration Record

On 2026-11-09, all 64 files from `manuscript/` at SurfFlow commit `d792a76`
were copied unchanged and verified by SHA256. Overleaf commit `04a9f01` adds
those files, the `main.tex` wrapper and project instructions. Local compilation
completed with 38 pages and no undefined citations or references.

This is a location-only migration of the existing draft. New Vs-only, CSRM
transect and CUDA acceptance results were not automatically inserted into the
manuscript. The original local `manuscript/` directory remains as an ignored
backup; it is no longer tracked in the current GitHub tree. Historical GitHub
commits are unchanged.

Older experiment notes and helper defaults may reference `manuscript/` or
`gji_dnn_posterior_inversion/`. Those are historical locations, not the active
paper checkout. Use explicit external output paths when supported and review
generated figures before copying them into Overleaf. The build command above
supersedes the old code-repository manuscript build target.

Large training data and checkpoints stay in their existing external storage;
neither this migration nor paper updates should upload them to Overleaf or GitHub.
