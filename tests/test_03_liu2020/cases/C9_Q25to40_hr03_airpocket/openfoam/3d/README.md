# C9 three-dimensional compressible OpenFOAM model

This directory contains a source-complete OpenFOAM v2512 model for Liu et al.
(2020) Case C9. It is intentionally separate from the validated phase-1
one-dimensional model. A 1-D trace is never relabelled as a 3-D result.

For cross-account continuation, read `HANDOFF.md` and paste the complete
`CONTINUATION_PROMPT.md` into the replacement Cloud Agent.

The governing solver is `compressibleInterFoam`: two compressible phases,
VOF interface capture, gravity, surface tension, laminar viscous stress,
perfect-gas air, and weakly compressible water (`perfectFluid`, bulk modulus
2.2 GPa). `compressibleInterIsoFoam` is available strictly as an
isoAdvector-versus-MULES **interface transport** sensitivity; it is not
misidentified as an isothermal gas solver. The model includes the full 5.80 m
upstream pipe, 0.30 m junction
chamber, 5.95 m downstream pipe, 1.22 m riser, an external atmospheric plume
region, a resolved tailgate opening, and the upstream crown air pocket.

Read `PAPER_AUDIT.md` first. In particular, the paper does **not** report the
initial trapped-air length, volume, nose/tail coordinates, or tailgate
opening. Those quantities are named sensitivity parameters, not measurements.
The default analytic air volume is 12.642 L and the two bracketing priors are
4.641 and 21.608 L; the mesh-integrated value is reported separately. The
target gate discharge area is 0.00823 m² from the documented initial state,
downstream-full closure, and \(Q_0\). Because the resolved sharp opening has
\(C_d=0.817\) in the no-ramp hydraulic check, its geometric area is
0.01008 m². This one-time initial-condition closure is not fitted to any
geyser timing or peak.

## Build and staged execution

OpenFOAM v2512 must be installed at `/usr/lib/openfoam/openfoam2512`.

```bash
cd tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d
python3 -m pip install -r requirements.txt
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
  represented by the explicitly derived geometric/effective-area closure.
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
`refined` profile adds one local level in the chamber, riser, pocket
interfaces, and gate while retaining the same far-field background. Both
standard `checkMesh` and `checkMesh -allGeometry -allTopology` are run and
their logs are consumed by post-processing. The current 142,343-cell base mesh
passes the standard check (maximum nonorthogonality 60.43 and skewness 2.88);
the strict check reports 2,228 concave cells, so it is not represented as a
strict `Mesh OK`.

The default transient limits are `maxCo=0.35`, `maxAlphaCo=0.20`, and
`maxDeltaT=5e-4 s`; the tighter timestep case halves these limits. MULES uses
one alpha correction with two subcycles. Two pressure correctors and one
non-orthogonal corrector are used, following the supplied v2512
`compressibleInterFoam` tutorial structure. A
12 m/s velocity limiter is a numerical safety bound, more than twice the
5.75 m/s experimental maximum quoted by the paper. Its activation is recorded
in `openfoam_3d_metrics.json`; a result that depends on clipping is treated as
numerically qualified, not silently accepted.

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

## Current validation status

The committed artifacts cover the initialization, the complete 1.00 s
paper-time smoke window, and an interrupted phase-1 attempt through paper time
1.504 s. They are **not** a completed phase-1 or phase-2 validation. Measured
against the paper targets, the run gives:

- initialized PT2 = 2.853 kPa gauge versus 2.970 kPa;
- first PT2 peak = 10.818 kPa at 0.392 s versus 10.690 kPa at 0.500 s;
- first mixture crossing of the riser rim = 0.640 s versus 0.730 s;
- total- and gas-mass residuals through 1.504 s =
  \(6.13\times10^{-6}\) and \(4.06\times10^{-4}\);
- mesh-integrated initial upstream gas volume = 14.065 L versus the
  12.642 L analytic pocket construction;
- upstream-zone gas mass falls from 17.557 g to 1.441 g by 1.504 s, showing
  that the baseline pocket is transported/released much earlier than the
  paper's 6.46 s main-pocket arrival. The declared 20%-mass-transfer criterion
  gives 0.620 s (−90.4%).

The velocity limiter activated during the transient and reached 18,217 cells
(12.8%) during the partial phase-1 continuation; the smoke-window final
maximum-velocity location was in the atmospheric plume. This is an unresolved
numerical qualification and requires the declared control sensitivity before
claiming eruption-count agreement. No eight-eruption claim is made from this
short run. The early upstream gas loss is a physical-model discrepancy, not a
domain mass-conservation failure.

## Sensitivities

The matrix covers mesh, timestep/Courant limits, small/base/large pocket
volumes, MULES versus isoAdvector interface transport, adiabatic-like versus
near-isothermal heat-capacity limits for pocket compression, liquid bulk
modulus ±20%, gate area ±20%, contact angle 60°/120°, and interface
compression 0.5/1.5. The thermal limits are declared closure sensitivities,
not fitted air properties.

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
