# B-H4 — Cong, Chan & Lee (2017)

Series B case with `Dr=31 mm`, `H0=0.66 m`, and `L0=0.61 m`.

- Measured geyser classification: `0`
- Model classification: `0` (`OK`)
- Re-run: `python scripts/run_series_case.py`
- Original campaign scan: `../../studies/criterion_map`

## Three-dimensional OpenFOAM validation

`openfoam/3d` contains the independent `compressibleInterFoam` validation.
It uses the paper's 6.59 m dimension chain and `x_tee=3.47 m`, circular
50/31 mm pipes, a true 3-D T-junction, a compressible atmospheric pocket, and
an external air domain above the physical 1.8 m riser.

Start with `openfoam/3d/PAPER_AUDIT.md`; run instructions and the complete
initial/boundary-condition ledger are in `openfoam/3d/README.md`.  The known
no-geyser label is comparison data only.  A 3-D no-geyser result is accepted
only after base/refined mesh, valve-time, interface-mixing, and mass-balance
checks rule out a numerical false negative.
