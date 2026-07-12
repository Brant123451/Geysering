# B-H3 three-dimensional compressible VOF case

This directory is the paper-audited OpenFOAM v2512 model for Cong, Chan &
Lee (2017), Series B Run B-H3. It uses a real circular `D=0.050 m`
horizontal pipe, a real circular `Dr=0.026 m` riser, a Boolean-union
three-dimensional tee, and an expanded external atmosphere above the physical
`1.8 m` riser. It does not use a 2-D, wedge, thin-layer, or rectangular
equivalent-area model.

Read `PAPER_AUDIT.md` before running. `MODELING_CONTRACT.json` is the BH3/BH4
parity contract: a paired BH4 model may change only `riser_diameter_m`.

## Coordinate and geometry contract

- `x=0`: upstream constant-head pressure boundary.
- Horizontal pipe invert: `z=0`; centreline `z=0.025 m`; soffit `z=0.050 m`.
- Tee/riser axis: `x=3.47 m`.
- Valve/release plane: `x=5.98 m`.
- Closed downstream cap and PT1: `x=6.59 m`.
- Initial pocket: full circular section, `5.98 <= x <= 6.59 m`.
- Physical riser rim: `z=1.850 m` (`1.8 m` above the pipe soffit).
- External air domain: `0.30 x 0.30 m`, from the physical rim to `z=3.0 m`.

The physical rim remains a geometric expansion into external air; the
computational extension is not represented as a longer physical riser.

## Solver and materials

- Solver: `compressibleInterFoam`.
- Air: `perfectGas`, `Mw=28.966 kg/kmol`, initially `101325 Pa`.
- Water: weakly compressible `perfectFluid` with physical `2.2 GPa` bulk
  modulus and density `998 kg/m3` at `101325 Pa`, `296.15 K`.
- Initial temperature: `296.15 K` (the measured laboratory `23 degC`).
- Surface tension: `0.072 N/m`.
- Standard `kEpsilon`, matching the paired 2018 CFD study.
- Acrylic wall roughness: `Ks=1e-6 m`.
- Static contact angle: neutral `90 deg`; neither source paper reports or
  enables preferential wall adhesion.

No pressure source, velocity source, or geyser-forcing source term is used.
The only driving force is the audited upstream constant head acting on the
compressible initial air pocket.

## Initial fields

| Region | `U` | `p_rgh` / absolute `p` | `alpha.water` | `T` |
|---|---|---|---:|---:|
| Water-filled pipe and riser below `z=0.66 m` | `(0 0 0)` | hydrostatic from `H0=0.66 m` | 1 | 296.15 K |
| Pocket `5.98<=x<=6.59 m` | `(0 0 0)` | isothermal hydrostatic air, `101325 Pa` at pipe centreline | 0 | 296.15 K |
| Riser headspace and external atmosphere | `(0 0 0)` | isothermal hydrostatic air, `101325 Pa` at `H0` | 0 | 296.15 K |

After `setFields`, `setExprFields` imposes
the exact isothermal hydrostatic `p(z)` for the configured `perfectFluid`
water and `perfectGas` air equations of state, followed by
`p_rgh=p-rhoMix*(g dot x)`. The open air column is referenced to `101325 Pa`
at `H0=0.66 m`; the initially isolated pocket is referenced to the same
pressure at the pipe centreline. This avoids the gravitational free fall that
would result from a height-independent gas pressure. `setExprBoundaryFields`
applies the same phase, `p`, and `p_rgh` state to every wall face, including
the water/air transitions along the riser wall and downstream pocket.

Analytic pocket target: `1.1977322 L`; ideal-gas mass target at the stated
pressure and temperature: `1.427641 g`.

## Patch contract

| Patch | `U` | `p_rgh` / `p` | `alpha.water` | `T` | Contact angle / role |
|---|---|---|---|---|---|
| `inlet` | `pressureInletOutletVelocity` | fixed constant-head `p_rgh=107786.65 Pa`; `p` calculated | fixed 1 | fixed 296.15 K | Upstream water reservoir |
| `closedEnd` | no slip | `fixedFluxPressure`; `p` calculated | zero gradient | zero gradient | Permanently capped downstream end |
| `walls` | no slip | `fixedFluxPressure`; `p` calculated | static 90 deg | zero gradient | Circular pipe and external floor |
| `riserWall` | no slip | `fixedFluxPressure`; `p` calculated | static 90 deg | zero gradient | Circular physical riser |
| `atmosphere` | `pressureInletOutletVelocity` | expression-fixed isothermal ambient `p_rgh`, equivalent to `p=101325 Pa` at `z=0.66 m`; `p` calculated | `inletOutlet`, inflow 0 | `inletOutlet`, inflow 296.15 K | Open sides/top of external air domain |
| Valve baffle | coupled cyclic; wall for closed-hold test | zero-jump for instantaneous baseline or time-varying porous pressure loss | coupled | coupled | Published instantaneous opening; 0.2/0.5 s sensitivities |

## Mesh profiles

`make_geometry.py` builds one exact OpenCASCADE Boolean fluid volume and
generates a boundary-fitted Delaunay tetrahedral mesh with named inlet, cap, wall,
riser-wall, and atmosphere physical groups. The base and refined profiles use
independent pipe/riser/atmosphere target sizes; no cut-cell background or
rectangular equivalent flow area enters the solution. Both the initial free
surface and the Valve #4 circular cross-section are conformal internal mesh
surfaces. Every run repeats the strict mesh check after creating its valve
baffle and aborts before solving unless that final mesh reports `Mesh OK`.

```bash
MESH_PROFILE=base ./Allmesh
MESH_PROFILE=refined ./Allmesh
```

Neither generated STL surfaces nor `constant/polyMesh` are committed.

## Runs

```bash
# Closed-valve static hold
RUN_MODE=closed END_TIME=1.0 ./Allrun

# Open-valve numerical smoke
RUN_MODE=event VALVE_OPENING=instant END_TIME=0.02 ./Allrun

# Full first-event window
RUN_MODE=event VALVE_OPENING=instant END_TIME=13 ./Allrun
```

`VALVE_OPENING` accepts `instant`, `0.2`, or `0.5`. `C_ALPHA`, `MAX_CO`,
`MAX_ALPHA_CO`, and `MAX_DELTA_T` expose declared sensitivity controls.
Use clean runtime copies for independent variants; `run_study.py` manages
these copies and writes only compact CSV/JSON/PNG results into `outputs/`.

## Required outputs

`postprocess.py` produces pressure, `Yfs`, `Yint`, pocket/total gas metrics,
physical-rim and atmosphere water flow, cumulative ejected volume, initial
phase-volume errors, total mass residual, gas-mass residual, and an
experiment--existing-1D--3D summary. A geyser is detected only when
`alpha.water >= 0.05` occurs above the physical rim; the known experimental
classification is never supplied to the solver or used to alter parameters.
