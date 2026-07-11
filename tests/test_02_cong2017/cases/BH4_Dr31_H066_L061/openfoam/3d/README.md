# B-H4 three-dimensional compressible VOF case

This directory models Cong, Chan & Lee (2017) Series-B run B-H4 without using
the known no-geyser classification as an input.  `PAPER_AUDIT.md` is the
geometry/parameter gate and `case_parameters.json` is the machine-readable
case record.

## Physical domain

- circular main pipe: `D=0.050 m`, `x=0...6.59 m`;
- circular riser: `Dr=0.031 m`, centre at `x=3.47 m`;
- genuine Boolean-union 3-D T-junction;
- Valve #4 at `x=5.98 m`, followed by the `0.61 m` atmospheric air pocket;
- closed cap at `x=6.59 m`;
- physical riser wall from the pipe crown (`z=0.05 m`) to `z=1.85 m`;
- a separate `0.30 x 0.30 m` external atmosphere volume extends to
  `z=3.05 m`.  Its sides and top are open.  The riser rim is not a pressure
  outlet, so a computed jet must enter the external domain.

Gmsh OCC constructs and Boolean-unions the exact cylinders, then exports a
multi-solid triangulation to OpenFOAM's `cartesianMesh` (cfMesh).  This retains
the circular geometry while producing solver-grade Cartesian/polyhedral cells
that pass the complete topology/geometry checks.  It is not a 2-D, wedge,
thin-layer, or rectangular-area model.  The base mesh resolves both diameters
with at least eight nominal cells; the refined mesh uses at least twelve.

## Solver and initial state

OpenFOAM v2512 `compressibleInterFoam` is used.  Air is an ideal gas and water
uses `rhoConst` at `998 kg/m3`.  Liquid compressibility is negligible on this
seconds-long gravity/air-pocket event, while the ideal-gas phase retains the
required pressure-volume coupling.  This avoids resolving irrelevant
microsecond liquid-acoustic transients without making the air incompressible.
No pressure or velocity forcing source is used.

At `t=0`:

- `U=(0 0 0) m/s`;
- `T=296.15 K` (the reported laboratory temperature, 23 degC);
- the pipe up to Valve #4 and the riser up to global `z=0.66 m` contain water;
- `x=5.98...6.59 m` contains `1.197732 L` of atmospheric air;
- the remaining riser/external domain contains the same hydrostatic air
  column, referenced to 101325 Pa at `z=H0`;
- water pressure is hydrostatic below the `H0=0.66 m` free surface;
- the upstream patch maintains the same 0.66 m piezometric head.

## Boundary-condition ledger

| Patch | `U` | `p_rgh` / `p` | `alpha.water` | `T` |
|---|---|---|---|---|
| `reservoir` | `pressureInletOutletVelocity` | fixed `p_rgh=107786.6508 Pa` (0.66 m head); `p` calculated | fixed 1 | fixed 296.15 K |
| `walls` | no slip | `fixedFluxPressure`; `p` calculated | `constantAlphaContactAngle`, 90 deg | zero gradient |
| `closedEnd` | no slip | `fixedFluxPressure`; `p` calculated | `constantAlphaContactAngle`, 90 deg | zero gradient |
| `atmosphere` | `pressureInletOutletVelocity` | fixed `p_rgh=101332.717209 Pa`, giving static `p=101325 Pa` at `z=H0`; `p` calculated | `inletOutlet`, air on inflow | `inletOutlet`, 296.15 K on inflow |
| `valveCouple*` | `cyclicACMI` | `cyclicACMI`; `p` coupled | `cyclicACMI` | `cyclicACMI` |
| `valveWall*` | no slip | `fixedFluxPressure`; `p` calculated | `constantAlphaContactAngle`, 90 deg | zero gradient |

The paper gives acrylic walls but no contact angle.  The 90-degree value is a
neutral documented numerical closure, not an experimental measurement.
The air is initialised with
`p=101325+rho_air*g*(H0-z)` and constant
`p_rgh=101325+rho_air*g*H0`.  Referencing ambient pressure at the experimental
free-surface height keeps both phases continuous there and prevents the open
external air from undergoing gravitational free fall.  The 7.7 Pa correction
between `z=0` and `H0` is below 0.01% of ambient pressure but is retained for a
true static hold.

## Ball-valve process

`configure_valve.py` writes a `cyclicACMI` interface and a coincident
non-overlap wall on the full circular cross-section at `x=5.98 m`.  The ACMI
coupled area is the prescribed open area and the complementary area is a real
no-slip wall.  Thus the closed state supports the physical initial pressure
difference without replacing a solid valve by an extreme, lagged porous
coefficient.  For the baseline manual opening:

```text
open area / full area = 3 s^2 - 2 s^3
s = clamp(t/t_open, 0, 1)
t_open = 0.20 s
```

The paper reports the total opening time but not ball angle or area versus
time.  The monotonic smoothstep is therefore an explicit closure assumption
with zero endpoint slopes, not a fitted loss curve.  The required duration
study uses 0.10, 0.20, and 0.30 s with every other input unchanged.  Closed
hold, valve flux, and pressure on both sides are recorded independently before
the experimental no-geyser label is examined.

## Run

The environment needs OpenFOAM v2512, Gmsh, NumPy, and Matplotlib.

```bash
# Closed-valve static hold
./Allclean
BH4_VALVE_MODE=closed BH4_END_TIME=0.5 BH4_LABEL=closed_static ./Allrun

# Open-valve smoke
./Allclean
BH4_END_TIME=0.3 BH4_LABEL=open_smoke ./Allrun

# 13 s base event
./Allclean
BH4_END_TIME=13 BH4_LABEL=base_topen0p20 ./Allrun

# Refined mesh
./Allclean
BH4_PIPE_SIZE=0.004166667 BH4_RISER_SIZE=0.002583333 \
BH4_ATMOSPHERE_SIZE=0.01875 BH4_END_TIME=13 \
BH4_LABEL=refined_topen0p20 ./Allrun

# Opening-time sensitivity (base mesh)
./Allclean
BH4_VALVE_TIME=0.10 BH4_END_TIME=13 BH4_LABEL=base_topen0p10 ./Allrun
./Allclean
BH4_VALVE_TIME=0.30 BH4_END_TIME=13 BH4_LABEL=base_topen0p30 ./Allrun
```

Set `BH4_DRY_RUN=1` for the solver's one-step parser/smoke check.  Parallel
runs use at most six ranks by default.  `Allrun.resume` continues an existing
decomposed event.  `BH4_MAX_DELTA_T` can impose a stricter time-step cap for
the closed-valve startup diagnostic without rewriting `controlDict`.

`Allrun` always executes:

```text
Gmsh OCC multi-solid STL -> cartesianMesh -> topoSet
                         -> checkMesh -allGeometry -allTopology
                         -> setFields -> setExprFields -> createBaffles
                         -> checkMesh
                         -> compressibleInterFoam
```

The complete geometry/topology check is performed before ACMI creation.
OpenFOAM represents the closed fraction by scaling coincident coupled/wall
face-area vectors, so `-allGeometry` intentionally flags those runtime masks as
warped faces even though the underlying cell geometry is unchanged.  A second
basic `checkMesh` after baffle creation must still report `Mesh OK`; both stages
are recorded in `mesh_metadata.json`.

## Diagnostics and compact outputs

Function objects record:

- PT2 pressure and the gas-volume-weighted pocket pressure used as the PT1
  proxy (the paper does not report PT1's exact axial coordinate);
- riser-centreline `alpha.water` for `Yfs`, `Yint`, and `Ta`;
- pocket gas volume, water volume above the physical rim, outlet water flux,
  and cumulative ejected water;
- water volume and gas mass, including open-boundary flux balances;
- mixed-interface volume in the riser to expose numerical gas diffusion;
- valve flow and both face-averaged pressures to prove that the closed state
  does not leak or add energy.

`postprocess_compare.py` writes compact CSV, JSON, and one PNG under
`../../outputs/openfoam3d/`.  Generated meshes, time directories,
`processor*`, `postProcessing`, and logs are intentionally not committed.
Runs shorter than the required 13 s event window are explicitly labelled
`INDETERMINATE`; closed-hold and open-smoke diagnostics cannot establish a
no-geyser classification.

For this critical no-geyser case, a matching classification is insufficient:
base/refined agreement, an unblocked external atmosphere, bounded mixed
interface volume, and water/gas conservation must all be reported.
