# Liu2020 Case B3 — three-dimensional OpenFOAM validation

This directory contains the source-only, full-width 3-D model for B3
(`20 -> 100 L/s`, downstream initially full). Read `PAPER_AUDIT.md` before
interpreting a result: the required base commit omits the paper PDF and page
scans, and the paper does not report the tail-gate opening, exact B3 chamber
level, or exact in-plane coordinates of every pressure tap.

## Solver choice

The case uses OpenFOAM v2512 `compressibleInterFoam`, not incompressible
`interFoam`. B3's approximately 55 kPa slam and -20 kPa rebound are pressure
wave/gas-compression observables. An incompressible VOF solver can reproduce
free-surface displacement but cannot validate the water-hammer amplitude.

- Water is a `perfectFluid` with `rho0=998.2 kg/m3` and `R=2.2e6 m2/s2`
  (`c=sqrt(R)=1483 m/s`, bulk modulus about 2.2 GPa).
- Air is a 293.15 K `perfectGas`.
- The A2 `kOmegaSST` closure and standard coefficients are retained; no
  coefficient is fitted to B3.
- The PVC walls are rigid because wall thickness and modulus are not reported.
  Wall compliance and unresolved dispersed air are therefore uncertainty
  sources, not tunable effective wave speeds.
- `fvOptions` bounds numerical overshoots to `250 <= T <= 400 K` and
  `|U| <= 50 m/s`. Both limits are well outside the expected B3 state and are
  monitored by the `extrema` function object; they are solver safeguards, not
  fitted physical parameters.

## Geometry and boundary mapping

`make_mesh.py` constructs one connected Boolean fluid volume from the audited
A2 dimensions:

- upstream pipe: `L=5.80 m`, `D=0.20 m`, slope `1:100`;
- chamber: `0.30 x 0.30 x 0.45 m`, invert drop `0.18 m`;
- downstream pipe: `L=5.95 m`, `D=0.28 m`, horizontal;
- physical riser: `D=0.06 m`, `L=1.22 m`.

The numerical inlet headbox is retained from A2 to admit water while the
upstream pipe has a free surface. The riser opens into a `0.60 x 0.60 m`
external atmosphere extending to `z=5.25 m`. Its bottom annulus, sides and top
are atmospheric boundaries. Thus the physical riser still ends at
`z=1.67 m`; water is not deleted there and can rise to the approximately
`4.21 m` Fig. 7(a) regression height above the chamber lid.

B3 does **not** contain A2's outfall box or weir. The downstream circular end
is water-filled and uses a hydrostatic `p_rgh` for a submerged reservoir with
`H_tail=Dd=0.28 m`. This is the minimum-head realisation already declared by
the frozen B3 1-D model. It preserves the reported full-pipe condition without
inventing a gate opening or fitting the transient.

Initial water is A2's free-surface upstream state, a full downstream pipe, and
a declared `0.30 m` chamber stage. The riser/plume headspace contains
atmospheric air and remains vented. Liquid velocity is seeded at `Q0/A` in the
upstream pipe, chamber, and downstream pipe so the reported steady-Q0 initial
condition is not approximated by stagnant water. A 2 s `Q0` relaxation
interval precedes a linear 0.4 s ramp; comparison time is
`OpenFOAM time - 2.0 s`.

## Mesh profiles

The Gmsh HXT tetrahedral profiles are defined in `make_mesh.py`:

| profile | pipe | chamber/junctions | riser | free surface | jet core | far atmosphere |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 50 mm | 24 mm | 13 mm | 35 mm | 50 mm | 120 mm |
| `baseline` | 35 mm | 16 mm | 9 mm | 25 mm | 32 mm | 90 mm |
| `refined` | 27 mm | 11 mm | 6.5 mm | 18 mm | 22 mm | 70 mm |

The sensitivity profile refines all required critical regions: chamber,
upstream/downstream junctions, riser connection, free surface, physical nozzle,
and plume core. `Allrun.mesh` always runs
`checkMesh -allGeometry -allTopology`.

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

`Allrun.resume` continues an interrupted decomposed solve. `Allclean` removes
all generated fields, meshes, processor directories, function-object output,
and logs.

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
outlet and all atmosphere faces.

Generated OpenFOAM time directories, `processor*`, `postProcessing`,
`constant/polyMesh`, `.msh` files, logs and frame data are ignored and must not
be committed.
