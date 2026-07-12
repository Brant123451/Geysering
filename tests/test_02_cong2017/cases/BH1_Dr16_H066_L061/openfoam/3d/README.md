# Cong et al. (2017) B-H1 — true 3-D compressible VOF

This is the Case-owned OpenFOAM v2512 model for Run B-H1. It uses
`bh1CompressibleInterFoam`, a source-included `compressibleInterFoam`
derivative; the air pocket is an ideal gas and is initialized at laboratory
atmospheric pressure. No pressure source, velocity source, or
outcome-dependent parameter is present.

The repository-wide source adjudication is in
`../../../../../../openfoam/3d/PAPER_AUDIT.md`. Machine-readable values and
all non-paper numerical choices are in `model_config.json`.

## Geometry and coordinates

`x` follows the horizontal main from the constant-head inlet to the cap, `y`
is upward, and `z` is transverse. The main axis is `y=z=0`.

| Quantity | Model value |
|---|---:|
| Circular main ID and length | `0.050 m`, `6.600 m` |
| Tee axis | `x=3.470 m` |
| Selected valve | `x=5.990 m` |
| Initial full-bore pocket | `5.990 <= x <= 6.600 m` |
| Capped end | `x=6.600 m` |
| Circular B-H1 riser ID | `0.016 m` |
| Physical rim above pipe crown | `1.800 m` |
| External atmosphere | `0.30 x 0.30 x 1.20 m` above the rim |

`make_geometry.py` uses OpenCASCADE Boolean unions and exports a watertight CAD
boundary for `snappyHexMesh`, so the circular main and riser form one conformal
three-dimensional T-junction. The external box begins only at the physical
rim; it is not a confined `3.0 m` riser.

## Initial state

- `U=(0 0 0) m/s`.
- `H0=0.66 m` above the main invert. Since the invert is `y=-0.025 m`,
  the initial free surface is `y=0.635 m`, or `Y=0.610 m` above the crown.
- The main is water-filled except for the final `L0=0.61 m`.
- The isolated pocket target is `1.1977322e-3 m3` and `1.4276e-3 kg` of air.
- The isolated pocket starts at `101325 Pa` absolute on the pipe axis.
  Riser/exterior air is hydrostatic from `101325 Pa` at the initial free
  surface (`p_rgh=101332.42484 Pa`). All phases start at `296.15 K`.
- Water starts at `296.15 K`, `rho0=998.2 kg/m3`, with physical bulk modulus
  `2.2 GPa`; air uses `perfectGas`.
- The energy equation is active. The local solver uses total sensible enthalpy
  with kinetic-energy transport and pressure-time-derivative work, the standard
  OpenFOAM compressible enthalpy transformation. This is energetically
  equivalent to total internal energy but avoids cancellation of the absolute
  atmospheric-pressure datum against `p*div(U)` in low-Mach open air. It does
  not constrain or clip `T`. `fieldMinMax` records temperature extrema, and
  leaving the broad `200--600 K` validity interval invalidates the run.
- Hydrostatic water has `p_rgh=107543.13717 Pa` for the chosen coordinate
  origin. The pocket keeps `p_rgh=101325 Pa`; the disconnected headspace uses
  its hydrostatic value above.

The exact mesh-integrated inventory is emitted for every run. The tunnel-gas
volume is checked against the analytic pocket before an event result is used.

## Patch conditions

| Patch | `U` | `p_rgh` / `p` | `alpha.water` | `T` |
|---|---|---|---|---|
| `inlet` | `pressureInletOutletVelocity` | fixed hydrostatic `p_rgh=107543.13717 Pa`; `p` calculated | fixed water `1` | `inletOutlet`, `296.15 K` inflow |
| `walls` (acrylic, floor, cap, static-check valve disk) | `noSlip` | `fixedFluxPressure`; `p` calculated | `zeroGradient` | adiabatic `zeroGradient` |
| `atmosphere` (external sides/top) | `pressureInletOutletVelocity` | fixed hydrostatic `p_rgh=101332.42484 Pa`, referenced to `101325 Pa` at `y=0.635 m`; `p` calculated | `inletOutlet`, air on inflow | `inletOutlet`, `296.15 K` inflow |

The papers report no static, advancing, or receding acrylic contact angle.
The baseline therefore does **not** prescribe or fit an angle:
`alpha.water` is `zeroGradient` on smooth no-slip walls. It must not be
described as a measured `90 deg` contact angle.

## Valve treatments

1. The event baseline is fully open at `t=0`, matching the instantaneous
   opening used by Chan et al. (2018).
2. The closed-valve static check converts the `x=5.99 m` face zone into a
   zero-thickness impermeable no-slip wall baffle. This is a numerical
   closed-disk representation and is not continued into the event run.
3. The `0.2 s` and `0.5 s` sensitivities use a `25 mm` cell zone immediately
   upstream of the valve. Its declared effective area is
   `A/A0=sin(pi*t/(2*tau))^2`; the loss is `K=(A0/A)^2-1`.
   This is a monotone uncertainty envelope, not a measured ball-angle law.
   The solver adds the passive Forchheimer loss
   `-0.5*rho*K*|U|*U/L` through a non-negative, Picard-linearized momentum
   diagonal. The source is exactly zero at zero velocity: no velocity floor,
   pressure jump, or prescribed flow is used. At startup this implicit coupling
   avoids the unstable old-flux feedback of `porousBafflePressure`. The solver
   also checks that the selected zone volume represents `25 mm` of the 50 mm
   pipe within `20%`, so the integrated loss is consistent on both meshes.

The baffle is a passive loss, not a pressure or velocity source. No pressure,
velocity, or thermal forcing is used. No result is tuned to the known
`GEYSER` classification.

## Meshes and execution

The declared nominal sizes are:

| Mesh | Main | 16 mm riser | Exterior far field |
|---|---:|---:|---:|
| base | `6.25 mm` | `1.5625 mm` | `12.5 mm` plume |
| refined | `3.125 mm` | `0.78125 mm` | `6.25 mm` plume |

Riser-resolution cells continue `50 mm` above the physical rim so that its
sharp edge is not colocated with a three-level refinement transition.

Every mesh runs both standard `checkMesh` and
`checkMesh -allGeometry -allTopology`. Standard checks must report `Mesh OK`.
The extended report is retained even when it flags the small population of
non-convex polyhedra created at body-fitted local-refinement transitions.
Acceptance additionally limits those cells to `1%`, low-determinant cells to
10, the minimum determinant to `4e-4`, non-orthogonality to `70 deg`, and
skewness to 4; any negative-volume cell is fatal. These are explicit mesh
diagnostics, not silently converted into a strict-check pass.

The bounded transport/PIMPLE settings follow OpenFOAM v2512's
`compressibleInterFoam/laminar/depthCharge3D` water-air reference: upwind
momentum transport, uncorrected Laplacian/normal gradients, one outer
corrector, two pressure correctors, and `maxCo=maxAlphaCo=0.5`. This declared
baseline and the total-enthalpy equation are used unchanged for all meshes and
valve cases.

Each run is created under ignored `runs/`; source templates remain clean.
OpenFOAM v2512, Gmsh, NumPy, and Matplotlib are required.
`Allrun.solve` incrementally compiles the source under `solver/` with `wmake`.
Install the Python tools with
`python3 -m pip install --upgrade -r openfoam/3d/requirements.txt`;
Matplotlib 3.9 or newer is required when the environment supplies NumPy 2.

```bash
cd openfoam/3d/case

# Both commands run checkMesh -allGeometry -allTopology.
BH1_MESH_LEVEL=base BH1_RUN_NAME=base-open ./Allrun.mesh
BH1_MESH_LEVEL=refined BH1_RUN_NAME=refined-open ./Allrun.mesh

# Independent acceptance checks.
./Allrun.static
./Allrun.smoke

# 13 s baseline, or all mesh/valve sensitivities.
./Allrun.full
./Allrun.sensitivity
```

Set `OPENFOAM_NP` to change the MPI rank count (capped at eight). Set
`BH1_REBUILD=1` only to replace a generated run of the same name.

## Recorded observables

The runtime function objects record:

- `PT1_proxy` one pipe radius upstream of the cap and `1 mm` below the crown;
- PT2 at the tee invert;
- 10 mm-spaced riser/exterior centreline `alpha.water`, `p`, `U`, and `T`;
- full-domain water/gas volume and mass;
- tunnel gas volume and end-pocket gas-weighted pressure;
- physical-riser and exterior-water inventories;
- inlet, physical-rim, and atmospheric phase/total fluxes;
- pressure, temperature, velocity, phase-fraction extrema and continuity.

`postprocess_openfoam.py` writes compact CSV/JSON/PNG only. `Yfs` is the
highest centreline point with `alpha.water>=0.5`; `Yint` is the top of the
`alpha.water<0.5` gas core connected to the tee. The plots compare these
against this Case's digitized Fig.9(a), the explicitly labelled
`PT1_proxy` against Fig.10(a), and the frozen existing 1-D result. The 1-D
heights are shifted by `-D=-0.05 m` only for display so that all level curves
share the above-crown datum. That comparison remains qualitative because the
frozen 1-D effective pipe and tee geometry differ from the audited 3-D model.
The 3-D pipeline consumes the tracked legacy CSV without regenerating it; its
threshold-sensitive solver requires `--overwrite-frozen` before replacing
those environment-dependent reference artifacts.

Reported `vfs` and `vint` are first-passage averages over fixed, predeclared
height windows (`0.65--1.70 m` and `0.05--1.65 m`, respectively), with linear
interpolation at each threshold after the gas first enters the riser. This
excludes release-stage water sloshing before the eruption and prevents later
fallback through the same heights from cancelling the eruption speed. A 3-D
geyser requires either the centreline free surface to reach `98%` of the
physical rim height or at least `1e-9 m3` of positive upward water transfer
through the rim.

Water, gas, and total-mass conservation use

`final inventory + integrated outward boundary flux - initial inventory`.

The external atmosphere is open, so raw gas inventory alone is not a
conservation test. The gas boundary budget uses the solver's compressible
`rhoPhi` minus the water-volume flux times `998.2 kg/m3`; it does not assume
that expelled or ingested air remains at atmospheric density. The independent
total-mass budget integrates `rhoPhi` directly. Static-hold relative errors
must not exceed `0.1%`; smoke and each complete 13 s event must not exceed
`1%` for any of the three budgets.

The CSV distinguishes instantaneous exterior water inventory from water
ejected through the physical rim. The latter is the time integral of the
positive upward `alphaPhi0.water` flux through `riserMouth`; the net integral
is also retained so fallback is not counted as additional ejection.

Generated `processor*`, `polyMesh`, time directories, `postProcessing`, logs,
and `.msh` files are ignored and must never be committed.
