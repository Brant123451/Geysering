# C9 three-dimensional compressible OpenFOAM model

This directory contains a source-complete OpenFOAM v2512 model for Liu et al.
(2020) Case C9. It is intentionally separate from the validated phase-1
one-dimensional model. A 1-D trace is never relabelled as a 3-D result.

The governing solver is `compressibleInterFoam`: two compressible phases,
VOF interface capture, gravity, surface tension, laminar viscous stress,
perfect-gas air, and weakly compressible water (`perfectFluid`, bulk modulus
2.2 GPa). `compressibleInterIsoFoam` is available as the declared thermal/EOS
sensitivity. The model includes the full 5.80 m upstream pipe, 0.30 m junction
chamber, 5.95 m downstream pipe, 1.22 m riser, an external atmospheric plume
region, a resolved tailgate opening, and the upstream crown air pocket.

Read `PAPER_AUDIT.md` first. In particular, the paper does **not** report the
initial trapped-air length, volume, nose/tail coordinates, or tailgate
opening. Those quantities are named sensitivity parameters, not measurements.
The default air volume is 12.642 L and the two bracketing priors are 4.641 and
21.608 L. The default 0.0084 m² effective gate opening is rated once from
the documented initial state, downstream-full closure, and \(Q_0\); it is not
fitted to any geyser timing or peak.

## Build and staged execution

OpenFOAM v2512 must be installed at `/usr/lib/openfoam/openfoam2512`.

```bash
cd tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d
python3 prepare_case.py
cd case
./Allrun.mesh
./Allrun.initialize   # solver 0–0.25 s: Q=25 L/s, no valve change
./Allrun.resume smoke # to 1.25 s; paper time is solver time - 0.25 s
./Allrun.resume phase1
./Allrun.resume full
./Allrun.postprocess
```

Convenience wrappers are `Allrun.smoke`, `Allrun.phase1`, `Allrun.phase2`, and
`Allrun.solve`. The case runs on four MPI ranks by default. Every stage starts
from `latestTime`; interrupted runs can be continued with:

```bash
./Allrun.resume full
```

`Allrun.reconstruct` reconstructs only the latest field time for inspection.
Function-object histories are written directly to `postProcessing`. Runtime
time directories, decomposed processors, meshes, and logs are ignored by Git.

The time convention is deliberate:

- solver time 0–0.25 s is initialization at \(Q_0=25\) L/s;
- paper \(t=0\) is solver time 0.25 s;
- flow ramps linearly to 40 L/s over the reported 0.4 s;
- phase 1 ends at solver time 6.75 s (paper time 6.50 s);
- the full run ends at solver time 20.25 s (paper time 20.00 s).

## Initial and boundary conditions

- The chamber/downstream HGL is 0.75 m: chamber roof 0.45 m plus the reported
  initial 0.30 m riser column.
- The riser and pipe water are initialized hydrostatically through uniform
  `p_rgh`; the pocket is initialized at the local water pressure at its main
  interface. Air temperature is 293.15 K.
- The inlet is pure water with a tabulated volumetric flow-rate boundary.
- The downstream pipe is full. The tailgate is a resolved circular opening
  against a 0.28 m tailwater HGL (`Dd`); the omitted experimental opening is
  represented by the documented effective-area sensitivity.
- The riser opens into a 0.6 × 0.6 × 1.0 m atmospheric plume region. Its side,
  top, and floor outside the riser use pressure/open boundaries, allowing
  expelled water and air to leave the computational domain.
- No-slip and a 90° equilibrium contact angle are defaults. Contact angle and
  interface compression are explicit sensitivity dimensions.

`setFields` creates the selected pocket as a long crown body followed by the
thin crown layer described qualitatively in the paper. Its volume and initial
pressure are recorded in `case/generated_case.json`.

## Mesh and numerical controls

`blockMesh` and parallel `snappyHexMesh` generate the mesh. The base background
is 306 × 16 × 69 cells, with local refinement at the pocket body and
interfaces, chamber, riser, initial riser free surface, and tailgate. The
`refined` profile raises the background to 408 × 22 × 92. `checkMesh
-allGeometry -allTopology` is mandatory and its log is consumed by
post-processing.

The default transient limits are `maxCo=0.20`, `maxAlphaCo=0.15`, and
`maxDeltaT=2.5e-4 s`. MULES uses two alpha corrections and two subcycles.
Three pressure correctors and one non-orthogonal corrector are used. A
12 m/s velocity limiter is a numerical safety bound, more than twice the
5.75 m/s experimental maximum quoted by the paper.

## Diagnostics and required artifacts

The solver records PT1–PT4, 111 riser-centreline probes, 60 upstream-crown
probes, zone water/air inventories, boundary volume/mass fluxes, and extrema.
After a run:

```bash
python3 postprocess_openfoam.py --case case
```

creates in the C9 `outputs/` directory:

- `openfoam_3d_PT1_PT2_PT3_PT4.csv`
- `openfoam_3d_riser_height.csv`
- `openfoam_3d_air_pocket.csv`
- `openfoam_3d_event_table.csv`
- `openfoam_3d_metrics.json`
- pressure, riser, and air-pocket comparison PNGs
- `openfoam_3d_mesh_sensitivity.csv`

The event table is generated only from actual `alpha.water` crossing the
physical riser rim. Missing stages remain `not_run`, `smoke_only`, or
`complete_phase1_only`; the postprocessor does not invent phase-2 eruptions.

## Sensitivities

The matrix covers mesh, timestep/Courant limits, small/base/large pocket
volumes, energy versus isothermal gas treatment, gate area ±20%, contact angle
60°/120°, and interface compression 0.5/1.5.

```bash
# Materialize all source cases without running:
python3 run_sensitivity.py --stage prepare --fresh

# Example executable checks:
python3 run_sensitivity.py --stage mesh \
  --variants base,mesh_refined --fresh
python3 run_sensitivity.py --stage smoke \
  --variants base,time_tight,pocket_small,pocket_large --fresh
```

Sensitivity cases live in ignored `runs/` directories. The aggregate CSV
contains real status and metrics; prepared but unexecuted rows have blank
result fields.
