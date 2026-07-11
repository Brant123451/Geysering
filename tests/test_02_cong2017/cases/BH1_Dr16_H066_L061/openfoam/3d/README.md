# Cong et al. (2017) B-H1 — true 3-D compressible VOF

This is the Case-owned OpenFOAM v2512 model for Run B-H1. It uses
`compressibleInterFoam`; the air pocket is an ideal gas and is initialized at
laboratory atmospheric pressure. No pressure source, velocity source, or
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
- Pocket and riser/exterior air start at `101325 Pa` absolute and `296.15 K`.
- Water starts at `296.15 K`, `rho0=998.2 kg/m3`, with physical bulk modulus
  `2.2 GPa`; air uses `perfectGas`.
- The energy equation is active and `fieldMinMax` records temperature
  extrema. A non-positive temperature invalidates the run; temperature is not
  clipped to conceal a failed thermo inversion.
- Hydrostatic water has `p_rgh=107543.13717 Pa` for the chosen coordinate
  origin. `setFieldsDict` assigns atmospheric `p/p_rgh` to the pocket and
  headspace.

The exact mesh-integrated inventory is emitted for every run. The tunnel-gas
volume is checked against the analytic pocket before an event result is used.

## Patch conditions

| Patch | `U` | `p_rgh` / `p` | `alpha.water` | `T` |
|---|---|---|---|---|
| `inlet` | `pressureInletOutletVelocity` | fixed hydrostatic `p_rgh=107543.13717 Pa`; `p` calculated | fixed water `1` | `inletOutlet`, `296.15 K` inflow |
| `walls` (acrylic, floor, cap, static-check valve disk) | `noSlip` | `fixedFluxPressure`; `p` calculated | `zeroGradient` | adiabatic `zeroGradient` |
| `atmosphere` (external sides/top) | `pressureInletOutletVelocity` | `prghTotalPressure`, `p0=101325 Pa`; `p` calculated | `inletOutlet`, air on inflow | `inletOutlet`, `296.15 K` inflow |
| equivalent-valve cyclic pair | cyclic | time-varying `porousBafflePressure`; `p` cyclic | cyclic | cyclic |

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
3. The `0.2 s` and `0.5 s` sensitivities use a zero-thickness cyclic pressure
   jump. Its declared effective area is
   `A/A0=sin(pi*t/(2*tau))^2`; the loss is `K=(A0/A)^2-1`.
   This is a monotone uncertainty envelope, not a measured ball-angle law.

The baffle is a passive loss, not a pressure or velocity source. No pressure,
velocity, or thermal forcing is used. No result is tuned to the known
`GEYSER` classification.

## Meshes and execution

The declared nominal sizes are:

| Mesh | Main | 16 mm riser | Exterior far field |
|---|---:|---:|---:|
| base | `6.25 mm` | `1.5625 mm` | `12.5 mm` plume |
| refined | `3.125 mm` | `0.78125 mm` | `6.25 mm` plume |

Every mesh runs both standard `checkMesh` and
`checkMesh -allGeometry -allTopology`. Standard checks must report `Mesh OK`.
The extended report is retained even when it flags the small population of
non-convex polyhedra created at body-fitted local-refinement transitions.
Acceptance additionally limits those cells to `1%`, low-determinant cells to
10, the minimum determinant to `4e-4`, non-orthogonality to `70 deg`, and
skewness to 4; any negative-volume cell is fatal. These are explicit mesh
diagnostics, not silently converted into a strict-check pass.

Each run is created under ignored `runs/`; source templates remain clean.
OpenFOAM v2512, Gmsh, NumPy, and Matplotlib are required.

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
`PT1_proxy` against Fig.10(a), and the frozen existing 1-D result.

Gas and water conservation use

`final inventory + integrated outward boundary flux - initial inventory`.

The external atmosphere is open, so raw gas inventory alone is not a
conservation test. The gas boundary budget uses the solver's compressible
`rhoPhi` minus the water-volume flux times `998.2 kg/m3`; it does not assume
that expelled or ingested air remains at atmospheric density.

Generated `processor*`, `polyMesh`, time directories, `postProcessing`, logs,
and `.msh` files are ignored and must never be committed.
