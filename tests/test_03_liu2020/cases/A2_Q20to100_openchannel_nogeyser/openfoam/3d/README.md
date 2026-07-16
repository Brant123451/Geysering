# Case A2 three-dimensional OpenFOAM assessment

This directory contains the reproducible 3-D assessment of Liu, Shao & Zhu
(2020), Case A2 (`Q=20→100 L/s`, downstream open channel). The geometry and
paper evidence are audited in [PAPER_AUDIT.md](PAPER_AUDIT.md). Generated
meshes, numerical time directories, logs, `processor*`, and `postProcessing`
are intentionally not versioned.

## Model

* Solver: OpenFOAM v2512 `interFoam`, VOF water–air, transient RANS
  `kOmegaSST`.
* Domain: full circular upstream/downstream pipes, full 0.30 m-wide chamber,
  the 0.057 m A2 riser (rounded to 0.06 m in the journal), and the reported
  receiving tank/circular overflow weir; no symmetry, thin-layer, or 2-D
  approximation.
* Mesh: conformal first-order tetrahedra generated with Gmsh/OpenCASCADE.
  `base` and `refined` profiles change the chamber/riser and far-field sizes.
* Gravity: `(0 0 -9.81) m/s²`.
* Inlet: wet-fraction-weighted flow at the reported upstream-pipe end,
  0.020 m³/s until `t=0`, linear to 0.100 m³/s by `t=0.4 s`. The former
  unreported numerical headbox is removed.
* Initialization: approximate steady `Q0` velocities, 0.08 m upstream depth,
  chamber surface `z=0.12 m` inferred from PT3=0.99 kPa, and downstream
  `hd=Dd/4=0.070 m`. The simulation allocates `t=-12…0 s` to establish Q0.
* Downstream: the 0.57 × 0.61 × 0.89 m tank and 0.30 m-diameter, 0.40 m-high
  movable circular weir reported in Liu's 2018 thesis for this apparatus.
  Its crest is calibrated with a Q0-only hydraulic pilot to reproduce the
  reported `(Q0=20 L/s, hd=0.070 m)` operating point, not from any transient
  pressure or geyser result; tank stage then evolves freely.
* Vents: receiving-tank atmosphere, weir drain, and physical riser outlet are
  distinct patches. Water crossing each opening is audited independently.
* Pressure comparison: sampled reconstructed gauge `p`, not `p_rgh`.

The required clock has `t=0` at ramp start. The paper defines zero at the
fully-open instant; experimental times are therefore shifted by +0.4 s in
the output comparison.

The deterministic Gmsh target sizes are:

| Region | base | refined |
|---|---:|---:|
| Far field maximum | 0.0500 m | 0.0425 m |
| Chamber | 0.0180 m | 0.0153 m |
| Riser | 0.0120 m | 0.0102 m |
| Upstream pipe | 0.0280 m | 0.0238 m |
| Downstream pipe | 0.0400 m | 0.0340 m |
| Receiving tank | 0.0400 m | 0.0340 m |
| Weir crest region | 0.0180 m | 0.0153 m |

## Requirements

The validated environment uses:

* OpenFOAM.com v2512, launched by `/usr/bin/openfoam2512`;
* Python 3 with NumPy and Matplotlib;
* Gmsh Python API 4.15.2 and its `libGLU.so.1` runtime;
* four MPI ranks for the reported runs.

`OPENFOAM_LAUNCH`, `NP`, and `GMSH_THREADS` can be overridden. Mesh generation
defaults to one Gmsh thread so the tetrahedral mesh is deterministic.

## Reproduce

From `openfoam/3d/case`:

```bash
NP=4 ./Allrun base
NP=4 ./Allrun refined
```

This clean-clone entrypoint is equivalent to running `Allclean`,
`Allrun.mesh base`, `Allrun.solve smoke`, `Allrun.solve full`, and the
postprocessor in sequence.

`Allrun.mesh` requires both the independent combined STL to pass
`surfaceCheck` and the converted volume mesh to pass:

```bash
checkMesh -allGeometry -allTopology
```

The smoke run is a fresh 0.2 s `Q0` run. `Allrun.solve full` deliberately
starts fresh afterward, performs the complete `-12…14.4 s` run, and keeps only
three field checkpoints while retaining high-frequency compact function
outputs.

If a full decomposed solve is interrupted, do not run `Allrun.solve` again;
resume its latest processor checkpoint with:

```bash
NP=4 ./Allrun.resume
```

For a diagnostic split run, `NP=4 ./Allrun.solve initialize` stops cleanly at
the ramp start after the full Q0 interval. Inspect inlet/weir fluxes, tank
stage, and PT3 there, then continue the same decomposed state with
`NP=4 ./Allrun.resume`. The standard `Allrun` still executes one fresh full
solve.

The second command cleans generated base runtime state but retains its compact
outputs. The equivalent manual grid-sensitivity sequence is:

```bash
./Allclean
./Allrun.mesh refined
NP=4 ./Allrun.solve full
python3 ../postprocess_compare.py --profile refined --no-primary
```

The postprocessor reruns the frozen Case A2 1-D model, reads all OpenFOAM
restart segments, applies the paper/simulation clock conversion, and writes
compact profile-specific series plus the required primary deliverables:

* `outputs/openfoam_3d_pressure_series.csv`
* `outputs/openfoam_3d_riser_series.csv`
* `outputs/openfoam_3d_metrics.json`
* `outputs/openfoam_3d_pressure_comparison.png`
* `outputs/openfoam_3d_riser_comparison.png`

The primary files and plots intentionally retain the base profile. Profile
files are named `openfoam_3d_base_*` and `openfoam_3d_refined_*`; after base is
run first and refined second, `openfoam_3d_metrics.json` also receives the
base/refined grid-sensitivity block.

## Numerical observables

`controlDict` records:

* PT1, PT2, and PT3 `p`, `p_rgh`, and phase fraction;
* 61 riser elevations with five radial samples at each elevation;
* two five-point filling-bore stations, chamber phase probes, and a vertical
  receiving-tank stage line;
* integrated wet area at three downstream-pipe cross-sections, converted to
  equivalent circular-segment depths so `hd` is not confused with tank stage;
* water volume and phase-weighted water flux through every open boundary.

Riser results distinguish water-equivalent height, contiguous mixture-column
height, and highest mixture front. A geyser requires the mixture to reach the
1.22 m top and non-negligible water to leave the physical `riserOutlet`.
Liquid continuity is checked independently as

`V(t)-V(t0)+integral(sum(outward water fluxes) dt)`.

Bore arrival uses the five phase probes 10 mm before the chamber: their mean
`alpha.water` must remain at least 0.5 for 80% of a 20 ms window. The
experimental target is 1.60 s on the ramp-start clock. A PT3 rise of 0.20 kPa
is retained separately as a pressure-response time, not relabeled as visual
bore arrival.

## Replacement-model Q0 check

A base-grid Q0-only `-8…0 s` initialization followed by a four-second restart
was used to decide the required source initialization length. Over its final
one second, the inlet and weir flows were 19.987 and 20.091 L/s and the water
volume slope was -0.058 L/s. The 0.5% flow mismatch is much smaller than at
eight seconds and supports a canonical `-12…0 s` initialization.

At the same endpoint, integrated wet areas gave equivalent downstream-pipe
depths of 0.0627, 0.0710, and 0.0781 m at `x=0.60`, `3.25`, and `6.00 m`.
The source prescribes `hd=0.070 m` but does not identify an axial measurement
station; the full result therefore reports all three sections rather than
calling the 0.0749 m receiving-tank stage `hd`. PT3 was 0.793 kPa, consistent
with the paper's literal 0.10 m chamber depth but below its internally
inconsistent 0.99 kPa pressure statement. No transient pressure, riser
response, or geyser classification was used in this Q0 check. The fresh full
run must reproduce these initialization diagnostics before acceptance.

## Replacement-model base and refined full results

Both tank/weir replacement fulls (`-12…14.4 s`) completed on OpenFOAM.com
v2512 with four MPI ranks. Compact metrics are in
`outputs/openfoam_3d_base_metrics.json` and
`outputs/openfoam_3d_refined_metrics.json`.

| Metric | Experiment / target | base | refined |
|---|---:|---:|---:|
| Tetrahedra | — | 158,507 | 251,664 |
| Pre-ramp inlet / weir | 20 / 20 L/s | 19.981 / 20.079 | 19.453 / 20.106 |
| Pre-ramp water-volume slope | ≈0 | −0.134 L/s | −0.655 L/s |
| Downstream depths `x=0.60/3.25/6.00` | `hd=0.070 m` | 0.062/0.068/0.078 | 0.058/0.067/0.068 |
| PT3 initial | 0.99 kPa | 0.796 kPa | 0.823 kPa |
| Bore arrival, ramp-start clock | 1.60 s | 1.538 s | 1.525 s |
| PT2 / PT3 mean, paper 7–14 s | 2.15 / 4.99 kPa | 0.517 / 2.562 | 0.417 / 2.750 |
| First contiguous mixture column | ≈0.13 m | 0.06 m | 0.04 m |
| Max contiguous column / mixture front | <1.22 m | 0.22 / 0.38 | 0.24 / 0.32 |
| Reached riser top / geyser | no | no / no | no / no |
| Max Co / interface Co | interface ≤0.5 | 0.492 / 0.478 | 0.538 / 0.473 |
| Final liquid-balance residual / inflow | — | +0.054% | +0.039% |

Bore timing and no-geyser classification are robust under refinement. Steady
PT2/PT3 remain underpredicted on both grids. Refined briefly overshoots the
all-field Courant ceiling (0.538) while keeping interface Co ≤0.5. No crest
or BC was retuned from transient pressure or no-geyser outcome.

## Superseded fixed-stage run results

The values below belong to the former fixed-stage/headbox model. They are
retained as a diagnosed baseline but are **not** evidence for the replacement
tank/weir model. The replacement base/refined results above supersede them.

Both four-rank OpenFOAM.com v2512 runs reached the complete
`-4…14.4 s` window and both meshes passed
`checkMesh -allGeometry -allTopology`:

| Metric | base | refined |
|---|---:|---:|
| Tetrahedra | 118,321 | 187,195 |
| Maximum non-orthogonality | 54.637 | 55.874 |
| Maximum skewness | 0.712 | 0.693 |
| Minimum cell determinant | 0.00412 | 0.00461 |
| Minimum time step | 6.11e-5 s | 6.12e-5 s |
| Maximum Courant number | 0.506 | 0.491 |
| Maximum interface Courant number | 0.471 | 0.474 |
| OpenFOAM ClockTime | 29,420 s | 33,579 s |
| Final liquid-balance residual / inflow | +0.00151% | -0.00631% |

The task's interface-Courant acceptance ceiling of 0.5 was met, but the
adaptive-step dictionary targets (`maxCo=0.47`, `maxAlphaCo=0.35`) were
briefly overshot: interface peaks were 0.471/0.474 and the base all-field peak
was 0.506. Refined's all-field peak was 0.491. In the retained refined log,
the 10 m/s safety limiter acted on at most 19 cells (0.01%); its minimum time
step occurred near `t=2.58 s`. The base limiter-location history was removed
by the documented clean step before this additional diagnostic was requested,
so it is not reconstructed.

The runs falsify the earlier assumption that four seconds provides a converged
Q0 initialization:

| Pre-ramp metric | base | refined |
|---|---:|---:|
| Inlet liquid flow | 20.00 L/s | 20.00 L/s |
| Outlet liquid flow | 22.05 L/s | 23.75 L/s |
| Water-volume slope | -1.93 L/s | -3.68 L/s |
| PT3 initial gauge pressure | 0.622 kPa | 0.591 kPa |

Quantitative experiment comparison uses reconstructed atmospheric-gauge `p`
and shifts paper times by +0.4 s:

| Observable | Experiment | base | refined |
|---|---:|---:|---:|
| Bore arrival, ramp-start clock | 1.60 s | 2.805 s | 2.849 s |
| PT2 mean, paper 7–14 s window | 2.15 kPa | -0.034 kPa | -0.041 kPa |
| PT3 mean, paper 7–14 s window | 4.99 kPa | 1.643 kPa | 1.788 kPa |
| PT1 RMSE vs digitized trace | — | 0.149 kPa | 0.148 kPa |
| PT2 RMSE vs digitized trace | — | 2.068 kPa | 2.065 kPa |
| PT3 RMSE vs digitized trace | — | 2.865 kPa | 2.775 kPa |
| First contiguous column, `t=0…3 s` | first column 0.13 m | 0 m | 0 m |
| Maximum contiguous column, all `t>=0` | — | 0.020 m | 0.020 m |
| Maximum mixture front, all `t>=0` | — | 0.020 m | 0.080 m |
| Maximum water-equivalent riser height | — | 0.0051 m | 0.0070 m |

Neither mesh reaches the 1.22 m riser top. Integrated riser water discharge is
numerically zero (`1.19e-35 m3` base and `2.99e-54 m3` refined), so the
implemented model classifies both runs as no-geyser. This is not accepted as a
faithful validation merely because it matches the experimental branch: the
model misses the initial state, bore timing, chamber pressurization, and riser
response by large margins. Grid refinement changes PT3's final-window mean by
0.144 kPa (8.1%) and bore timing by 0.043 s (1.5%), but does not remove those
systematic discrepancies. PT2 changes by only 0.0073 kPa; its reported 17.6%
relative grid change is not meaningful because both means are close to zero.
The near-zero, slightly negative PT2 means indicate that the modeled chamber
never develops the measured positive lid pressure. The mixture-front metric is
also grid-sensitive (0.02 to 0.08 m), although both values remain far below
the riser top and the experimental first-column scalar.

## Local agent handoff

**Start here:** [`../LOCAL_AGENT_HANDOFF.md`](../LOCAL_AGENT_HANDOFF.md)

## Simulation data for local rendering

See [`SIMDATA.md`](SIMDATA.md). The refined run products under `case/`
(`VTK/`, reconstructed times `-12/12/13/14`, `postProcessing/`, `polyMesh/`)
are force-committed on this branch so you can rebuild front elevations locally
without re-solving. Large probe/VTK blobs use Git LFS.

## Front-elevation renders

`render_front_water_air.py` writes complete front views under `outputs/`:

* `openfoam_3d_refined_front_water_air.png` — true VTK `y=0` `alpha.water`
  collage at the retained volume dumps (`t=-12, 12, 13, 14 s`; `purgeWrite=3`);
* `openfoam_3d_refined_front_full_t*.png` — dual panel (full apparatus +
  horizontal-pipe zoom) for each retained dump;
* `openfoam_3d_refined_front_complete_strip.png` — four-time strip of the same
  dual panels;
* `openfoam_3d_refined_front_motion_timeline.png` / `_motion.gif` — full
  `-12…14.4 s` dual-panel motion from continuous probes (volume fields between
  dumps were purged);
* `openfoam_3d_refined_front_complete_motion.gif` — denser complete dual-panel
  animation (~130 frames, adaptive cadence, timeline scrubber); true VTK
  `y=0` fields at retained dumps, probe reconstruction elsewhere.

The pipe-zoom panel is required because open-channel `hd≈0.07–0.19 m` is only
a thin band when the full 1.22 m riser is drawn to scale.

## Limitations

The journal article omits the downstream tank/weir geometry; the replacement
uses dimensions reported in the corresponding open-access thesis. The crest
position is inferred from the reported `(Q0, hd)` operating point, and the
unreported weir wall thickness is a mesh-resolved numerical detail. Neither is
fitted to transient pressures. The 20 mm ambiguity between the article's
stated chamber depth and its PT3 pressure is retained in the audit, and probe
in-plane positions remain unreported. The mesh has no resolved viscous
sublayer, so wall-function friction is checked only through base/refined
sensitivity. Finally, incompressible single-velocity VOF does not resolve
acoustic water hammer, compressible trapped gas, or subgrid bubble
slip/breakup. These restrictions are material to the measured aerated pressure
oscillations even if the mean vented Series A branch is reproduced.
