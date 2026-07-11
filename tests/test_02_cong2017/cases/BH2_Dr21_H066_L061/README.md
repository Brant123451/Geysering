# B-H2 — Cong, Chan & Lee (2017)

Series B case with `Dr=21 mm`, `H0=0.66 m`, and `L0=0.61 m`.

- Measured geyser classification: `1`
- Model classification: `0` (`MISMATCH`)
- Re-run: `python scripts/run_series_case.py`
- Original campaign scan: `../../studies/criterion_map`

## Paper-audited three-dimensional model

`openfoam/3d` contains an independent OpenFOAM v2512
`compressibleInterIsoFoam` model with a circular 6.59 m main, circular 21 mm
riser, true T junction, compressible ideal-gas pocket, physical 1.8 m riser,
and a separate external atmosphere.  The 3-D model uses the paper geometry
(`xT=3.47 m`) rather than inheriting the reduced model's 6.0 m / 2.88 m
layout.

Start with `openfoam/3d/PAPER_AUDIT.md`; execution and compact-output commands
are in `openfoam/3d/README.md`.  The measured geyser classification is retained
only as an evaluation target and is not used to select numerical parameters.
