# Liu, Shao & Zhu (2020) Case A2 paper audit

## Scope and citation convention

This audit was made against the original article, *Experimental Study on
Stormwater Geyser in Vertical Shaft above Junction Chamber*, Journal of
Hydraulic Engineering 146(2), article 04019055, DOI
10.1061/(ASCE)HY.1943-7900.0001660. Page references below are the printed
article footers `04019055-N`, not PDF viewer indices. The Case README,
manifest, JSON config, digitized Fig. 3 traces, paper scans, frozen 1-D model,
and the complete pre-existing 3-D pilot were checked independently.

The article's corresponding experimental source is Lujia Liu's open-access
MSc thesis, *Experimental Study on Edmonton's Storm Geyser Formation Mechanism
and Mitigation Measures* (University of Alberta, 2018), DOI
10.7939/R30R9MK98. Its Sec. 3.1 and Figs. 3.1–3.2 report apparatus details
omitted from the condensed journal article. These primary-source additions are
identified explicitly below rather than treated as article measurements.

## Reported apparatus

| Item | Value represented in 3-D | Paper basis |
|---|---:|---|
| Upstream pipe | length 5.80 m; diameter 0.20 m; slope 1:100 toward chamber | p. 04019055-2, Experimental Setup; Fig. 2 |
| Junction chamber | 0.30 × 0.30 × 0.45 m clear cuboid | p. 04019055-2 |
| Pipe invert drop | upstream invert 0.18 m above downstream invert | p. 04019055-2 |
| Downstream pipe | length 5.95 m; diameter 0.28 m; horizontal | p. 04019055-2 |
| Riser | inside diameter 0.057 m; length 1.22 m; centered on chamber top | The journal rounds the diameter to 0.06 m (p. 04019055-2); Liu (2018) Sec. 3.3.1 and Table 3.1 identify the A2 test riser as `Dr=57 mm` |
| Riser opening | open to the laboratory atmosphere | Fig. 2 and the description of PT1/PT2 initially exposed to open air, pp. 04019055-2 and -4 |
| Receiving tank | 0.57 × 0.61 × 0.89 m, open top | Liu (2018), Sec. 3.1, p. 10; Fig. 3.1 |
| Circular overflow weir | diameter 0.30 m; height 0.40 m; movable at tank-bottom center | Liu (2018), Sec. 3.1, p. 10; Fig. 3.1 |

All pipe cross-sections are circular and the chamber is a full-width cuboid;
the final model is not a 2-D or thin-layer surrogate. Coordinates use the
downstream invert/chamber floor as `z=0`, the chamber upstream wall as `x=0`,
and the chamber centerline as `y=0`.

Two numerical details are not apparatus measurements:

1. The mathematically tangent chamber-floor/downstream-pipe contact produces
   zero-angle tetrahedra. The OCC mesh volume therefore has a local
   20 × 80 × 8 mm recess under the outlet mouth (12.8 mL, about 0.0014% of the
   computational volume). The reported chamber dimensions remain unchanged
   elsewhere. This regularization is included in the geometric uncertainty.
2. The thesis reports the overflow weir outside diameter but not its wall
   thickness. The mesh uses a 10 mm wall solely to resolve separate outer and
   inner wetted surfaces; the reported 0.30 m overflow perimeter is unchanged.
3. Neither source dimensions the tank floor relative to the downstream-pipe
   invert. Its absolute vertical placement is inferred from the reported
   0.40 m movable-weir height and the calibrated initial crest position; this
   does not change the reported tank dimensions.

The earlier unreported upstream headbox has been removed. `Q(t)` is now applied
at the reported upstream-pipe end immediately downstream of the ball valve,
only over the instantaneous wet fraction.

The source STL builder remains an independent dimensional/watertightness
check. The simulation mesh itself is generated as one Boolean-unioned OCC
fluid volume by `make_gmsh_mesh.py`.

## Case A2 conditions

| Quantity | Value | Paper basis and status |
|---|---:|---|
| Initial inflow `Q0` | 20 L/s | Table 1, p. 04019055-3; direct |
| Final inflow `Q1` | 100 L/s | Table 1, p. 04019055-3; direct |
| Valve opening | 0.2–0.4 s over all tests | p. 04019055-3; direct range. The simulation uses the conservative reported upper value, 0.4 s. |
| Initial upstream depth | about 0.08 m | Case A2 narrative, p. 04019055-3; direct |
| Initial downstream state | open channel, `hd/Dd=1/4`, hence `hd=0.070 m` | pp. 04019055-2 and -3; Table 1; direct |
| Downstream control | movable circular overflow weir in downstream tank | p. 04019055-2; direct |

The journal article alone does **not** report the tank or weir dimensions. The
thesis resolves that omission and also states that the crest was adjusted to
hold `hd/Dd=1/4` for Series A. It still does not tabulate the crest elevation
or a rating curve. A standard circular sharp-crested estimate for the reported
0.30 m perimeter first gave `z_crest=0.019 m`. Short Q0-only mesh pilots were
then used exactly as the experimenters used the movable weir: to set the known
initial downstream-pipe depth. The first translation to 0.036 m settled toward
a 0.0754 m tank stage; its final 1 s had 19.998 L/s inlet, 19.833 L/s weir
outflow, and 0.224 L/s volume rise. Translating the crest by
`0.0700-0.0754=-0.0054 m` gives `z_crest=0.0306 m`, represented as 0.031 m.
After 12 s at Q0, that geometry had 19.987 L/s inlet, 20.091 L/s weir
outflow, and a -0.058 L/s water-volume slope over the final second. Integrated
wet areas at downstream-pipe `x=0.60`, `3.25`, and `6.00 m` corresponded to
equivalent depths 0.0627, 0.0710, and 0.0781 m. The source does not identify
the axial `hd` station, so all three are retained; the 0.0749 m receiving-tank
stage is not silently treated as pipe depth. The canonical full calculation
therefore uses a fresh `-12…0 s` initialization and must independently
reproduce this closure. No transient pressure, riser response, or no-geyser
outcome enters the calibration.

The initial chamber condition needs special care. The paper states that
PT3 measured 0.99 kPa and that this “indicated a water depth of 0.10 m”
(p. 04019055-4), while PT3 is stated to be 0.02 m above the bottom. Taken
literally, a 0.10 m floor depth gives only about 0.78 kPa at PT3. The 3-D
initial free surface is therefore `z=0.12 m`, an explicitly documented
inference that gives 0.10 m pressure head above the actual PT3 elevation and
honors the requested 0.99 kPa observable. The alternative literal-depth
interpretation is a 20 mm initial-level uncertainty.

The near-balanced 12 s Q0 pilot relaxed PT3 to about 0.793 kPa while preserving
the reported downstream depth at the pipe midpoint. That agrees closely with
the paper's literal 0.10 m floor-depth interpretation but not with its
0.99 kPa pressure. The crest is not altered to hide this inconsistency; the
final initialization reports both observables independently.

## Measurement locations and pressure definition

| Probe | Reported location | 3-D sample |
|---|---|---|
| PT1 | riser wall, 0.80 m above chamber top | `(0.15, 0.025, 1.25)`, 3.5 mm inside the 0.0285 m-radius wall |
| PT2 | chamber top | `(0.08, 0.080, 0.445)`, 5 mm below the lid |
| PT3 | chamber front wall, 0.02 m above bottom | `(0.15, -0.145, 0.020)`, 5 mm inside the wall |

Locations are reported on p. 04019055-2 and shown in Fig. 2. The paper does
not give PT1 circumferential position or PT2/PT3 in-plane coordinates; those
coordinates are numerical choices and are not presented as measured geometry.
PT4 exists in the apparatus (upstream crown, 0.30 m before the chamber), but
the requested Fig. 3 validation uses PT1–PT3 only.

The transducers report pressure relative to laboratory atmosphere in kPa.
OpenFOAM solves `p_rgh = p - rho*gh`; therefore `p_rgh` is **not** directly a
transducer reading at nonzero elevation. The comparison samples OpenFOAM's
reconstructed `p` field, which is gauge pressure because all atmospheric
`prghTotalPressure` boundaries use `p0=0`, and divides Pa by 1000. Absolute
pressure is not used.

## Time origin

The paper defines `t=0` as the instant the ball valve is **fully open**
(p. 04019055-2). It also reports a 0.2–0.4 s manual opening interval. The task
requires `t=0` at the **start** of the flow ramp. These clocks cannot both be
used without a shift:

* simulation/required clock: ramp starts at 0.0 s and ends at 0.4 s;
* paper/Fig. 3 clock: fully open at 0.0 s;
* conversion used here: `t_ramp = t_paper + 0.4 s`.

Thus the directly reported 1.20 s bore arrival is a 1.60 s target on the
required ramp-start clock. Output CSVs and plots use the ramp-start clock and
shift the digitized experimental traces by +0.4 s. Both values are retained
in the metrics JSON.

The simulation measures bore arrival from five initially dry phase probes
10 mm upstream of the chamber; at least 80% of a 0.02 s window must have a
five-probe mean `alpha.water>=0.5`. A second five-probe station at `x=-0.30 m`
provides a local propagation-speed check. The older PT3 threshold is retained
separately as a pressure-response time and is not called the visual bore
arrival.

## Direct experimental outcomes

| Observable | Paper result | Source |
|---|---:|---|
| Bore velocity | about 4.56 m/s | p. 04019055-3 |
| Bore reaches chamber | 1.20 s on paper clock | p. 04019055-3 |
| Bore strikes opposite wall | about 1.25 s | p. 04019055-3 |
| Strong chamber mixing | 1.25–7.00 s | pp. 04019055-3 and -4 |
| PT1/PT2 initial | about 0 kPa | p. 04019055-4 |
| PT3 initial | 0.99 kPa | p. 04019055-4 |
| PT2 mean, 4–7 s | 2.22 kPa | p. 04019055-4 |
| PT3 mean, 4–7 s | 4.94 kPa | p. 04019055-4 |
| PT2 mean, 7–14 s | 2.15 kPa | p. 04019055-4 |
| PT3 mean, 7–14 s | 4.99 kPa | p. 04019055-4 |
| PT2/PT3 oscillation period, 4–7 s | about 0.30 s | p. 04019055-4 |
| Outcome | no geyser | p. 04019055-3; Series A discussion |

The paper defines a geyser as release of a water slug or air–water mixture
from the riser top (p. 04019055-3). Fig. 4 defines `h` as the maximum height
of the **first** mixture column. The Case A2 value near 0.13 m is digitized
from Fig. 4; it is not tabulated in the text and therefore has figure-reading
uncertainty. The text gives a zero-loss estimate of 0.33 m for `Q1=100 L/s`
and says estimates exceed measurements by roughly 0.10–0.15 m
(pp. 04019055-4 and -5). The 3-D postprocessor separately reports contiguous
mixture-column height, highest mixture front, and water-equivalent height so
that entrained droplets are not silently equated with the video definition.

## Solver audit

OpenFOAM v2512 provides `interFoam`, `interIsoFoam`,
`compressibleInterFoam`, and `compressibleInterIsoFoam`. `interFoam` models
two incompressible immiscible phases with VOF. The compressible variants add
phase pressure–density response; iso variants change interface advection, not
the underlying compressibility.

For A2, water density changes at the reported 1–10 kPa pressures are of order
`delta_p/K_water < 5e-6`. Series A has no deliberately trapped air pocket,
and air can leave through both the downstream free surface and open riser.
The reported 1.20 s event is a convective filling bore, not an acoustic
water-hammer arrival. `interFoam` is therefore selected for free-surface
timing, mean gauge pressures, column height, and no-geyser classification.

This choice cannot validate liquid acoustics, pipe-wall compliance, sealed
air-pocket compression, cavitation, dissolved gas, or subgrid bubble
slip/breakup/coalescence. In particular, the measured 0.30 s aerated pressure
oscillation may not be reproduced quantitatively by a single-velocity VOF
model. A compressible calculation would itself require unreported/calibrated
water wave speed and gas thermodynamics; copying an equation of state from a
tutorial would add assumptions rather than independently validate A2.

## Pilot discrepancies corrected

The original pilot was not accepted as evidence. Audit found and corrected:

* overlapping/non-watertight STL junctions and a `snappyHexMesh` mesh that
  failed full topology/geometry quality checks;
* one combined atmospheric patch, preventing riser overflow from being
  distinguished from numerical-headbox overflow;
* generic `totalPressure` applied directly to `p_rgh`, which omitted the
  hydrostatic conversion;
* an unconstrained full-disk pressure outlet rather than the reported
  `hd/Dd=1/4` condition;
* zero initial velocity despite a nominal steady 20 L/s initial state;
* stale 2 s time offset, reversed PT1/PT3 columns, use of `p_rgh` as gauge
  pressure, use of digitized lower envelopes as medians, and a 27-probe riser
  parser despite the current 61×5 sampling layout;
* combined or inconsistent legacy README metrics from several 1-D model
  revisions.

The fixed-stage outlet and numerical headbox were subsequently identified as
the dominant nonphysical assumptions and replaced using the thesis geometry
and the reported pipe-end inlet. The local recess, inferred movable-crest
position, numerical weir-wall thickness, probe-coordinate,
turbulence/wall-resolution, and incompressible-VOF limitations remain explicit.

## Superseded fixed-stage runtime audit

The following clean-source runs document the failed **former fixed-stage
model** and motivated the evidence-based correction above. They must not be
reported as results of the tank/weir model. Replacement base/refined runs are
required before final validation.

The former acceptance runs were completed with OpenFOAM.com v2512
`interFoam` on four MPI ranks. The 118,321-cell base and 187,195-cell refined
tetrahedral meshes both report `Mesh OK` under
`checkMesh -allGeometry -allTopology`, and both solves cover the complete
`t=-4…14.4 s` window. Compact series and metrics are committed under
`outputs/openfoam_3d_*`.

The run evidence does not support the earlier claim that the `-4…0 s` interval
is a converged Q0 initialization:

| Observable immediately before ramp | Paper target | base | refined |
|---|---:|---:|---:|
| Liquid inlet | 20 L/s | 20.00 L/s | 20.00 L/s |
| Liquid outlet | steady with inlet | 22.05 L/s | 23.75 L/s |
| Water-volume slope | approximately zero | -1.93 L/s | -3.68 L/s |
| PT3 gauge pressure | 0.99 kPa | 0.622 kPa | 0.591 kPa |

This is a physical-state imbalance, not the numerical continuity residual:
the final integrated liquid-balance residual is only +0.00151% of inflow for
base and -0.00631% for refined. The discrepancy is therefore retained as a
failed initialization/model check, not relabeled as mass loss.

Pressure comparison samples reconstructed atmospheric-gauge `p`; no result
below uses raw `p_rgh`. Experimental timestamps are shifted by +0.4 s:

| Observable | Paper | base | refined |
|---|---:|---:|---:|
| Bore arrival on ramp-start clock | 1.60 s | 2.805 s | 2.849 s |
| PT2 mean over paper 7–14 s window | 2.15 kPa | -0.034 kPa | -0.041 kPa |
| PT3 mean over paper 7–14 s window | 4.99 kPa | 1.643 kPa | 1.788 kPa |
| First contiguous column, first 3 s | about 0.13 m | 0 m | 0 m |
| Maximum contiguous column | — | 0.020 m | 0.020 m |
| Maximum mixture front | — | 0.020 m | 0.080 m |

Neither calculation reaches the 1.22 m riser top. Integrated outward water
volume at the physical riser opening is effectively zero (`1.19e-35 m3` base,
`2.99e-54 m3` refined), so both computations classify A2 as no-geyser. This
agreement in branch classification is not sufficient validation: chamber
pressurization, bore timing, and the first riser response are all substantially
underpredicted. Refinement changes the PT3 final-window mean by 0.144 kPa
(8.1%) and bore timing by 0.043 s (1.5%), so ordinary grid sensitivity does not
explain the systematic errors.

The interface Courant maxima are 0.471 and 0.474; minimum time steps are
`6.11e-5 s` and `6.12e-5 s`. Base briefly reaches an all-field Courant number
of 0.506, while refined remains at 0.491. Thus the task's 0.5 interface-Co
ceiling is met, but the adaptive-step dictionary targets
(`maxCo=0.47`, `maxAlphaCo=0.35`) are briefly overshot. The retained refined
log shows the 10 m/s safety limiter acting on at most 19 cells (0.01%), with
its minimum time step near `t=2.58 s`; the base limiter-location history was
removed by the documented clean-before-refined workflow and is not
reconstructed.

The strongest evidence-based explanation was the fixed 0.070 m terminal stage,
which prevented the receiving-tank level from responding as flow rose
fivefold. The unreported-headbox inlet and unconverged pre-ramp state were
additional independent defects. The replacement uses primary-source geometry
and a twelve-second Q0 initialization; no transient pressure or geyser
classification is used to set its parameters.
