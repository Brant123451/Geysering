# Vasconcelos-Wright 2011 geysering: Case A

This self-contained Case reproduces Case A from Vasconcelos and Wright (2011)
with a planar, compressible air-water OpenFOAM model. See `manifest.yaml` for
the machine-readable inventory and `reference/README.md` for the citation.

## Contents

```text
config/      documented physical and numerical parameters
data/        digitized experimental comparison traces
model/       executable OpenFOAM initial fields and dictionaries
scripts/     validation, run, resume, clean, and post-processing entry points
reference/   source citation and data provenance
outputs/     compact validated metrics, series, and comparison figures
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

The valve is represented by an initially open interface at `x = 0.546 m`.
The initial pressure and phase discontinuity therefore approximates an
instantaneous opening; the experiment reports a manual opening in under one
second.

## Numerical model

- OpenFOAM v2512 `compressibleInterFoam`.
- Laminar, isothermal-start air-water VOF.
- Ideal-gas air and weakly compressible water (`rho0 = 998.2 kg/m3`,
  `c = sqrt(2.2e6) = 1483 m/s`).
- Approximately `4 mm` in-plane resolution, 26,208 cells.
- Serial or automatic Scotch decomposition, capped at six MPI ranks.

## Reproduce

OpenFOAM v2512 must be loaded in the shell, or `OPENFOAM_BASHRC` must point to
its `etc/bashrc`. Python 3, NumPy, and Matplotlib are required for plots.

From this directory:

```bash
./scripts/validate.sh
OPENFOAM_NP=6 ./scripts/run.sh
```

`validate.sh` builds the mesh and runs a short solver start-up in a temporary
directory. `run.sh` performs the full 9 s simulation and regenerates
`outputs/`. Set `OPENFOAM_NP=1` for a serial run. To continue an interrupted
simulation or start over:

```bash
./scripts/resume.sh
./scripts/clean.sh
./scripts/run.sh
```

## Validated result

The tracked outputs are from a completed 9 s OpenFOAM v2512 run:

| Quantity | Model | Experimental target |
| --- | ---: | ---: |
| Mean pressure plateau, `H*` (`1 <= T* <= 7`) | 0.710 | 0.54 |
| Maximum free-surface level, `Y*` | 0.623 | 0.63 |
| Interface lift-off, `T*` | 8.03 | 7.3, 7.8, 7.9 |
| Geysering | false | false |

The free-surface maximum and no-geyser outcome agree closely, while pressure
and interface timing retain substantial model discrepancy. Detailed RMSE
values are in `outputs/openfoam_2d_metrics.json`.

## Important 2-D limitation

The vertical-plane geometry preserves the published lengths and diameters but
cannot preserve the circular pipe/tower area ratio. Its ratio is
`Dt/D = 0.607`, whereas the physical circular ratio is
`(Dt/D)^2 = 0.369`. It also cannot resolve the three-dimensional annular wall
film. This run is therefore a morphology and time-history diagnostic, not a
geometry-exact substitute for a circular 3-D T-junction simulation.

OpenFOAM time directories, `processor*`, raw probes, and logs are deliberately
excluded. Recreate all of them with `./scripts/run.sh`.
