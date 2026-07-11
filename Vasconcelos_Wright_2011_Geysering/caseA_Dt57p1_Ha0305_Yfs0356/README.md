# Vasconcelos-Wright 2011 geysering: Case A

This self-contained Case reproduces Case A from Vasconcelos and Wright (2011)
with a circular 3-D compressible air-water OpenFOAM model. The validated planar
model is retained as a legacy comparison. See `manifest.yaml` for the
machine-readable inventory and `reference/README.md` for the citation.

## Contents

```text
config/      documented physical and numerical parameters
data/        digitized experimental comparison traces
model/       executable OpenFOAM initial fields and dictionaries
scripts/     validation, run, resume, clean, and post-processing entry points
reference/   source citation and data provenance
outputs/     compact validated metrics, series, and comparison figures
HANDOFF.md   copy-ready prompt for a replacement Cursor Cloud Agent
```

Every repository path used by the scripts is resolved relative to this Case,
so the Case can be moved or invoked from another working directory.

## Geometry and initial state

- Horizontal acrylic pipe: `D = 0.094 m`.
- Upstream air chamber: `0.546 m`.
- Water-filled middle pipe: `2.970 m`.
- Downstream closed pipe: `0.490 m`.
- Tower: centre `x = 3.516 m`, `Dt = 0.0571 m`, `L = 0.610 m`, open top.
- Initial tower water level: `0.356 m` above the pipe crown.
- Initial chamber gauge-pressure head: `Ha0 = 0.305 m`.
- Pressure transducer: `x = 1.616 m`, near the pipe invert.

These values give the paper's Case A nominal initial air volume
`pi D^2 (0.546)/4 = 3.79 L`, `Dt/D = 0.607`, `Ha0/L = 0.500`, and
`Yfs0/L = 0.584`. The executable initial fields use zero velocity,
`T = 293.15 K`, atmospheric pressure `101325 Pa`, chamber pressure
`104311.7 Pa`, and hydrostatic water pressure `105271.3 Pa` at the pipe
centreline.

The valve is represented by an initially open interface at `x = 0.546 m`.
The initial pressure and phase discontinuity therefore approximates an
instantaneous opening; the experiment reports a manual opening in under one
second.

## Numerical model

- OpenFOAM v2512 `compressibleInterFoam`.
- Laminar, isothermal-start air-water VOF.
- Ideal-gas air and weakly compressible water (`rho0 = 998.2 kg/m3`,
  `c = sqrt(2.2e6) = 1483 m/s`).
- Primary model: circular 3-D pipe and tower, approximately `6 mm` core
  resolution and 138,292 cells.
- Legacy model: vertical-plane geometry, approximately `4 mm` in-plane
  resolution and 26,128 cells.
- Serial or automatic Scotch decomposition, capped at six MPI ranks.

## Reproduce

OpenFOAM v2512 must be installed. The scripts use an already loaded
environment or discover the packaged `openfoam2512` launcher; for other
installations, set `OPENFOAM_BASHRC` to the installation's `etc/bashrc`.
Python 3, NumPy, and Matplotlib are required for plots.

To validate and run the circular 3-D model from this directory:

```bash
./scripts/validate_3d.sh
./scripts/run_3d.sh
```

`validate_3d.sh` checks the watertight surface, builds the mesh, verifies the
physical chamber volume, and runs a short solver start-up in a temporary
directory. `run_3d.sh` performs the full 9 s simulation and regenerates the
3-D files in `outputs/`. It automatically uses the available processors,
capped at six. To continue an interrupted 3-D simulation or start over:

```bash
./scripts/resume_3d.sh
./scripts/clean_3d.sh
./scripts/run_3d.sh
```

The original `validate.sh`, `run.sh`, `resume.sh`, and `clean.sh` commands
operate on the legacy planar model.

## Validated results

Both tracked models completed 9 s OpenFOAM v2512 runs on four MPI ranks:

| Quantity | Circular 3-D | Planar 2-D | Experimental target |
| --- | ---: | ---: | ---: |
| Mean pressure plateau, `H*` (`1 <= T* <= 7`) | 0.698 | 0.710 | 0.54 |
| Pressure RMSE, `H*` | 0.148 | 0.206 | — |
| Maximum free-surface level, `Y*` | 0.651 | 0.628 | 0.63 |
| Free-surface RMSE, `Y*` | 0.0319 | 0.0122 | — |
| Interface RMSE, `Y*` | 0.207 | 0.347 | — |
| Interface lift-off, `T*` | 7.45 | 8.03 | 7.3, 7.8, 7.9 |
| Interface catches free surface, `T*` | 8.34 | not captured | 8.4 |
| Geysering | false | false | false |

The circular model materially improves pressure RMSE and interface timing, and
it captures the observed interface/free-surface catch. It does not eliminate
the discrepancy: its mean pressure plateau remains 29% above the experimental
target, and its free-surface RMSE is larger than the planar result. Level RMSE
uses only the finite model/experiment overlap before the centreline water slug
disappears. Detailed values are in `outputs/openfoam_3d_metrics.json`.

## Why the 3-D model is needed

The vertical-plane geometry preserves the published lengths and diameters but
cannot preserve the circular pipe/tower area ratio. Its ratio is
`Dt/D = 0.607`, whereas the physical circular ratio is
`(Dt/D)^2 = 0.369`; consequently the physical `3.79 L` air volume is not a
literal volume in the planar mesh. The model also cannot resolve the
three-dimensional annular wall film. This run is therefore a morphology and
time-history diagnostic, not a geometry-exact substitute for a circular 3-D
T-junction simulation. The primary 3-D model uses the physical circular ratio
`(Dt/D)^2 = 0.369`, gives a discrete initial chamber volume of `3.769 L`
versus the paper's `3.789 L` (0.53% mesh error), and resolves azimuthal flow.

OpenFOAM time directories, `processor*`, raw probes, and logs are deliberately
excluded. Recreate the 3-D run with `./scripts/run_3d.sh`.
