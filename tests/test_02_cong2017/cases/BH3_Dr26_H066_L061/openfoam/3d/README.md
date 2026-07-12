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
- Surface tension: primary-paper baseline `0.072 N/m`; `sigma=0` is a
  separately labeled diagnostic sensitivity because the paired CFD momentum
  equation omits a surface-tension term.
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
| Water-filled pipe and riser below `z=0.66 m` | `(0 0 0)` | hydrostatic from `H0=0.66 m` | linear 15 mm transition centred at `z=0.66 m` | 296.15 K |
| Pocket `5.98<=x<=6.59 m` | `(0 0 0)` | isothermal hydrostatic air, `101325 Pa` at pipe centreline | 0 | 296.15 K |
| Riser headspace and external atmosphere | `(0 0 0)` | isothermal hydrostatic air, `101325 Pa` at `H0` | 0 | 296.15 K |

After `setFields`, `setExprFields` first imposes a three-base-cell (`15 mm`)
linear VOF transition centred at the measured free-surface level. It preserves
the analytic water volume while avoiding the discrete curvature impulse of a
one-face jump on tetrahedra; transported-interface diffusion/compression is
still checked through the declared `cAlpha` sensitivities. It is a numerical
control, not a measured interface thickness, and it must pass the closed-hold
gate before event results are accepted. A second pair of expressions imposes
the phase-specific isothermal hydrostatic `p(z)` for the configured
`perfectFluid` water and `perfectGas` air equations of state, followed by
`p_rgh=p-rhoMix*(g dot x)`. The pure-phase regions are hydrostatic; the
alpha-weighted pressure inside the 15 mm transition remains a numerical
initialization and is therefore covered by the closed-hold gate. The open air
column is referenced to `101325 Pa` at `H0=0.66 m`; the initially isolated
pocket is referenced to the same pressure at the pipe centreline. This avoids
the gravitational free fall that would result from a height-independent gas
pressure. `setExprBoundaryFields` applies the same phase, `p`, and `p_rgh`
state to every wall face, including the water/air transitions along the riser
wall and downstream pocket.
The separate `INITIAL_INTERFACE_THICKNESS=0` diagnostic uses the conformal
`z=0.66 m` mesh partition as an exact sharp step. It is an initial-interface
sensitivity, not a replacement for the declared baseline unless the contract
and all paired BH3/BH4 runs are changed together.
A second 15 mm diagnostic uses a symmetric cosine transition with zero slope
at both band edges. It preserves the same analytic phase volume and thickness
as the linear baseline while testing whether the baseline's derivative jumps
are the source of the discrete CSF impulse.

For the closed-hold gate, `balanceInitialPressure` then keeps `alpha.water`
and `U=0` fixed while iterating both phase equations of state and projecting
`p_rgh` against the face operators used for gravity and CSF. It runs only
after the valve has become a closed baffle, so the isolated atmospheric pocket
retains its own pressure reference rather than being numerically equalized
with the upstream water. This is a discrete initialization, not a physical
source term.

Analytic pocket target: `1.1977322 L`; ideal-gas mass target at the stated
pressure and temperature: `1.427641 g`. `initialVolumeAuditDict` samples the
mesh-integrated pocket and non-overlap riser water volumes at exactly `t=0`,
before valve opening, so the audit is not contaminated by the first runtime
sample.

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
generates a boundary-fitted Delaunay tetrahedral mesh with named inlet, cap,
wall, riser-wall, and atmosphere physical groups. The base and refined profiles
use independent pipe/riser/atmosphere target sizes and 40/64 curvature elements
per full circle; no cut-cell background or rectangular equivalent flow area
enters the solution. Both the initial free surface and the Valve #4 circular
cross-section are conformal internal mesh surfaces. Every run repeats the
strict mesh check after creating its valve baffle and aborts before solving
unless that final mesh reports `Mesh OK`.
The paired FLUENT study used a much finer wall-resolved hybrid mesh; the present
tetrahedral profiles are accepted only through their reported base/refined
sensitivity and must not be described as resolving the reported sub-millimetre
falling film a priori.

```bash
MESH_PROFILE=base ./Allmesh
MESH_PROFILE=refined ./Allmesh
```

Neither generated STL surfaces nor `constant/polyMesh` are committed.

## Runs

```bash
# Build the case-local discrete pressure initializer
bash ./Allwmake

# Closed-valve static hold
RUN_MODE=closed END_TIME=1.0 ./Allrun

# Paired-CFD surface-tension omission diagnostic (not the experiment baseline)
python3 run_study.py --variant closed_sigma_zero

# Isolate the initial diffuse-band defect with both a sharp step and sigma=0
python3 run_study.py --variant closed_sharp_sigma_zero

# Measure mesh dependence of the better diffuse/zero-sigma diagnostic
python3 run_study.py --variant closed_refined_sigma_zero

# Restore the measured surface tension on the same refined mesh
python3 run_study.py --variant closed_refined_sigma_072

# Isolate linear-band edge curvature with a volume-preserving cosine profile
python3 run_study.py --variant closed_refined_sigma_072_cosine

# Open-valve numerical smoke
RUN_MODE=event VALVE_OPENING=instant END_TIME=0.02 ./Allrun

# Full first-event window
RUN_MODE=event VALVE_OPENING=instant END_TIME=13 ./Allrun
```

`VALVE_OPENING` accepts `instant`, `0.2`, or `0.5`. `C_ALPHA`, `MAX_CO`,
`MAX_ALPHA_CO`, `MAX_DELTA_T`, `ALPHA_SMOOTH_CURVATURE`, and
`SURFACE_TENSION` expose declared numerical controls.
`INITIAL_INTERFACE_THICKNESS` accepts only the declared `0.015 m` baseline or
the conformal sharp-step value `0`; `INITIAL_INTERFACE_PROFILE` accepts
`linear` or the declared `cosine` diagnostic for the 15 mm band. The baseline
uses zero curvature-smoothing passes:
controlled static tests found that extra passes increased water-side velocity
for this mesh, while the explicit initial VOF transition reduced the startup
impulse. Use clean runtime copies for independent variants;
`run_study.py` manages these copies and writes only compact CSV/JSON/PNG
results into `outputs/`.
It treats a failed closed hold as a hard error and refuses to start core or
sensitivity event windows unless a passing `closed_base` result with the
current source fingerprint exists.
Only a closed run that reaches the full declared `1.0 s` minimum can set
`closed_hold.pass=true`; short isolation diagnostics report threshold status
separately and cannot release the event gate.
The completed sharp/zero-surface-tension diagnostic failed more severely than
the 15 mm diffuse diagnostic (`1.7366` versus `0.02387 m/s` maximum
water-weighted speed at `0.05 s`). It is retained as negative evidence and is
not eligible to replace the baseline. Refining the diffuse/zero-sigma case to
456,068 cells reduced that speed to `0.01760 m/s` and the reconstructed
initial force residual from `475.5` to `319.3`, but the speed was still rising
at `0.05 s`; this is improvement evidence, not a 1 s hold pass. Restoring
the measured `0.072 N/m` on that refined mesh increases the `0.05 s`
water-weighted maximum to `0.12539 m/s` and the reconstructed residual to
`1681.4`. Thus mesh refinement alone does not cure the CSF imbalance, and the
physical-sigma event gate remains closed.

## Required outputs

`postprocess.py` produces pressure, `Yfs`, `Yint`, pocket/total gas metrics,
physical-rim and atmosphere water flow, cumulative ejected volume, initial
phase-volume errors, the paper-defined `Vair/Vw`, total mass residual from
direct `rhoPhi` boundary fluxes, gas-mass residual, and an
experiment--existing-1D--3D summary. A geyser is detected only when
`alpha.water >= 0.05` occurs above the physical rim; the known experimental
classification is never supplied to the solver or used to alter parameters.
Closed-hold acceptance uses free-surface drift, isolated-pocket volume drift,
and the all-domain maximum of `alpha.water*|U|`; unweighted and gas-weighted
speed maxima are retained as explicit diagnostics. For event runs,
`closed_hold.pass` is `null`; an open-valve smoke can never be mistaken for a
closed-hold pass.

At the current revision, the base mesh passes strict `checkMesh` and the
`0.02 s` event smoke completes, but the 1 s closed hold has not passed. No
13 s result is accepted until that gate is resolved without changing the
experimental classification target.
