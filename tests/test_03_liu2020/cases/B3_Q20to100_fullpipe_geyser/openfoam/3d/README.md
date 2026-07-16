# Liu2020 Case B3 — three-dimensional OpenFOAM validation

This directory contains the source-only, full-width 3-D model for B3
(`20 -> 100 L/s`, downstream initially full). Read `PAPER_AUDIT.md` before
interpreting a result. A user-supplied `references/liu2020.pdf` matching the
pre-migration SHA-256 was read directly. The paper still does not report the
detailed Series-B overflow-weir rating, exact numerical B3 chamber stage, or
the in-plane coordinates of every pressure tap.

## Solver choice

The case uses OpenFOAM v2512 `compressibleInterIsoFoam`, not incompressible
`interFoam`. B3's approximately 55 kPa slam and -20 kPa rebound are pressure
wave/gas-compression observables. An incompressible VOF solver can reproduce
free-surface displacement but cannot validate the water-hammer amplitude.
`compressibleInterIsoFoam` retains the two-phase compressible thermodynamics
but transports the interface with geometric isoAdvector reconstruction and
explicit clipping. This was selected after `compressibleInterFoam` screening
developed local negative-temperature failures at mixed tetrahedral interface
cells.

- Water/system compliance is a `perfectFluid` with `R=93025 m2/s2`
  (`c=sqrt(R)=305 m/s`, the acrylic-pipe wave speed stated on paper p. 11).
  `rho0=997.1107767 kg/m3` gives `rho=998.2 kg/m3` at 101325 Pa.
- Air is a 293.15 K `perfectGas`.
- The A2 `kOmegaSST` closure and standard coefficients are retained; no
  coefficient is fitted to B3.
- The fluid mesh walls remain rigid. The paper-sourced 305 m/s effective EOS
  represents the compliance of the clear acrylic pipes in this fluid-only
  calculation; differences between the acrylic pipes/riser and clear PVC
  chamber remain uncertain because wall thickness and modulus are not reported.
- `fvOptions` bounds numerical velocity overshoots to `|U| <= 50 m/s`, far
  outside the expected B3 state. Velocity and temperature are monitored by the
  `extrema` function object; the velocity bound is a solver safeguard, not a
  fitted physical parameter.
- Adaptive stepping limits bulk-flow Courant number to 0.50 (the v2512
  isoAdvector tutorial value), interface Courant number to the stricter 0.15,
  and `deltaT` to 1 ms.
- Each step uses the v2512 reference solver's single PIMPLE outer loop, with
  three pressure correctors and one non-orthogonal correction.

## Geometry and boundary mapping

`make_mesh.py` constructs one connected Boolean fluid volume from the audited
A2 dimensions:

- upstream pipe: `L=5.80 m`, `D=0.20 m`, slope `1:100`;
- chamber: `0.30 x 0.30 x 0.45 m`, invert drop `0.18 m`;
- downstream pipe: `L=5.95 m`, `D=0.28 m`, horizontal;
- physical riser: `D=0.06 m`, `L=1.22 m`.

The numerical inlet plenum is a compact, fully water-filled volume whose top
is a wall. It represents the paper's pressurised feed tank and forces the
prescribed flow through the upstream pipe; an open headbox would create an
unphysical overflow bypass. It lies outside the reported 5.80 m pipe length.
The riser opens into a `0.60 x 0.60 m` external atmosphere extending to
`z=5.25 m`. Its bottom annulus, sides and top are atmospheric boundaries. Thus the physical riser still ends at
`z=1.67 m`; water is not deleted there and can rise to the approximately
`4.21 m` Fig. 7(a) regression height above the chamber lid.

B3 does **not** contain A2's open-channel outfall box. Series B used the
downstream-tank overflow weir at `hd/Dd=1`; the downstream circular end is
therefore water-filled and uses a hydrostatic `p_rgh` with
`H_tail=Dd=0.28 m`. This is the direct hydrostatic equivalent of the reported
initial full-pipe depth. It avoids inventing an unreported weir rating or
fitting the transient.

Initial water is the reported approximately `0.08 m`-deep A2 upstream state, a
full downstream pipe, and a declared `0.30 m` chamber stage consistent with
the Fig. 5(a) `t=0` image and 0.28 m downstream crown. The riser/plume headspace contains
atmospheric air and remains vented. Liquid velocity is seeded at `Q0/A` in the
upstream pipe, chamber, and downstream pipe so the reported steady-Q0 initial
condition is not approximated by stagnant water. A 2 s `Q0` relaxation
interval precedes a linear 0.4 s ramp; comparison time is
`OpenFOAM time - 2.0 s`.

## Mesh profiles

The Gmsh HXT tetrahedral profiles are defined in `make_mesh.py`:

| profile | pipe | chamber/junctions | riser | free surface | jet core | far atmosphere |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 65 mm | 32 mm | 17 mm | 45 mm | 65 mm | 160 mm |
| `baseline` | 50 mm | 22 mm | 13 mm | 35 mm | 50 mm | 120 mm |
| `refined` | 45 mm | 20 mm | 11.5 mm | 30 mm | 45 mm | 110 mm |

The sensitivity profile refines all required critical regions: chamber,
upstream/downstream junctions, riser connection, free surface, physical nozzle,
and plume core. `Allrun.mesh` always runs
`checkMesh -allGeometry -allTopology`.

The baseline/refined pair was selected by mesh quality and runtime stability,
not cell count alone. With the compact inlet plenum they contain approximately
105k and 142k tetrahedra. The 22 mm baseline junction size avoids a
below-threshold interpolation-weight face at the chamber/downstream junction
that appears with the 23--24 mm candidates.
More aggressive 209k and 292k candidates failed their Q0 smoke solves through
localized thermodynamic instability; they are therefore not presented as
evidence of convergence.

## Run

The repository cloud image declares OpenFOAM v2512, Gmsh, NumPy and
Matplotlib. From `openfoam/3d/case`:

```bash
chmod +x Allrun.* Allclean

# Fast mesh and actual 0.02 s Q0 smoke solve
B3_MESH_PROFILE=smoke ./Allrun.mesh
./Allrun.smoke

# Full baseline: 2 s settle + 0.4 s ramp + 14 s after ramp completion
./Allclean
B3_MESH_PROFILE=baseline ./Allrun.mesh
OPENFOAM_NP=6 ./Allrun.solve
python3 ../postprocess_compare.py --case . --mesh-label baseline --primary
```

Run the critical-region sensitivity in a clean copy so the baseline probe
history remains available:

```bash
cp -a case /tmp/liu2020-b3-refined
cd /tmp/liu2020-b3-refined
./Allclean
B3_MESH_PROFILE=refined ./Allrun.mesh
# Through the principal peak/rebound and subsequent oscillations:
B3_END_TIME=6.5 OPENFOAM_NP=6 ./Allrun.solve
python3 /path/to/B3/openfoam/3d/postprocess_compare.py \
  --case . --mesh-label refined
```

`Allrun.resume` switches the decomposed case to `startFrom latestTime` and
continues an interrupted solve; `B3_END_TIME` can be supplied again when
resuming a deliberately shortened/refined run. `Allclean` removes all
generated fields, meshes, processor directories, function-object output, and
logs.

## Committed outputs

The postprocessor creates only the requested compact artefacts:

- `outputs/openfoam_3d_pressure_series.csv`
- `outputs/openfoam_3d_riser_series.csv`
- `outputs/openfoam_3d_metrics.json`
- `outputs/openfoam_3d_pressure_comparison.png`
- `outputs/openfoam_3d_geyser_height_comparison.png`
- `outputs/openfoam_3d_mesh_sensitivity.csv`

The pressure plot overlays paper digitisation, the existing frozen 1-D model,
and 3-D OpenFOAM, with the quoted extrema and event times. Pressure is
`p - 101325 Pa`, never `p_rgh`. Riser height is the highest centreline sample
with `alpha.water >= 0.05`, at 50 mm spacing; geysering requires water at least
one sample above the physical rim.

Mass residual is computed from the change in integrated `alpha.water` volume
plus the time integral of `alpha.water`-weighted `phi` over inlet, submerged
outlet and all atmosphere faces. Net atmosphere liquid outflow is also
reported as spilled volume against Table 2's B3 repeats
(`0.65/0.78/0.82 L`, mean `0.72 L`); gross outward crossings are retained
separately.

Generated OpenFOAM time directories, `processor*`, `postProcessing`,
`constant/polyMesh`, `.msh` files, logs and frame data are ignored and must not
be committed.

## Baseline result status

The production baseline has been completed through solver time `16.4 s`
(`Q0` settle + `0.4 s` ramp + full `14 s` post-ramp window). Compact artefacts
above were written with `postprocess_compare.py --primary`.

Honest scalar summary versus the paper B3 targets:

| quantity | paper | OpenFOAM 3-D baseline |
|---|---:|---:|
| PT2 peak | 55.03 kPa @ 1.47 s | 26.04 kPa @ 1.936 s |
| PT3 peak | 51.76 kPa | 29.96 kPa @ 1.932 s |
| PT2 / PT3 rebound min | -20.26 / -17.77 kPa | -12.47 / -9.29 kPa |
| geyser | yes | yes (`0.10 m` above rim; `1.32 m` above lid) |
| spilled water | mean 0.72 L | 1.47 L |
| mass residual | — | 6.32% |

Geometry, `Q0/Q1`, ramp duration and air volume were not altered to chase the
55 kPa peak. Strict `checkMesh` still fails one check with 565 low-determinant
boundary cells; the 50 m/s velocity limiter activated (max 8989 cells in one
correction) and is reported as a numerical-quality warning. A refined mesh
sensitivity run is executed independently to at least solver time `6.5 s`.

## Refined mesh sensitivity status

An independent refined copy (`~141681` tets) was solved to `6.5 s` and appended
to `openfoam_3d_mesh_sensitivity.csv` (critical-region window only;
`end_time_after_ramp_s = 4.5`). Primary baseline CSV/PNG/JSON artefacts were not
replaced.

| quantity | baseline (16.4 s) | refined (6.5 s) |
|---|---:|---:|
| PT2 peak | 26.04 kPa @ 1.936 s | 19.17 kPa @ 2.002 s |
| PT3 peak | 29.96 kPa | 23.71 kPa @ 1.998 s |
| PT2 / PT3 rebound min | -12.47 / -9.29 kPa | -10.26 / -7.16 kPa |
| geyser height above lid | 1.32 m | 1.27 m |
| spilled water | 1.47 L | 0.69 L |
| mass residual | 6.32% | 1.15% |
| max velocity-limited cells | 8989 | 11649 |

Refinement does **not** close the gap to the paper 55 kPa peak; the shortfall is
therefore not cured by this mesh densification alone. Mass closure improves on
the shorter refined window, while the velocity limiter still activates and
strict `checkMesh` still fails (626 low-determinant cells).

## Front-view water/air motion animation

`controlDict` uses `purgeWrite 4`, so intermediate volume fields were discarded
and a VTK cutting-plane movie of every write interval is not recoverable from
the finished run. The completed baseline does retain the full `16.4 s`
`riserCentreline` `alpha.water` series (`Δt = 0.002 s`) plus PT probes.

Render the front elevation (x–z) animation from those probes:

```bash
python3 render_front_view_animation.py --case case --focus geyser
python3 render_front_view_animation.py --case case --focus full \
  --output ../../outputs/openfoam_3d_front_view_full_apparatus.mp4 --dt 0.08
```

Outputs (under `outputs/`):

- `openfoam_3d_front_view_motion.mp4` — chamber/riser zoom, full `16.4 s`
- `openfoam_3d_front_view_motion.gif` — geyser-window preview
- `openfoam_3d_front_view_full_apparatus.mp4` — full pipe/chamber elevation
- `openfoam_3d_front_view_t_*.png` — stills near ramp / peak / late time

Pipe free-surface evolution away from the riser centreline is shown only as the
declared initial fill (upstream `0.08 m`, downstream full); the animated jet is
the measured 3-D centreline `α.water`.

## Uploaded simulation data (`sim_data/`)

Retained meshes, reconstructed volume fields, full `postProcessing` series and
ParaView `VTK/` exports are stored at:

`tests/test_03_liu2020/cases/B3_Q20to100_fullpipe_geyser/sim_data/`

See `sim_data/README.md` for the purgeWrite limitation and how to open the data
locally. Compact validation plots/CSVs remain under `outputs/`.
