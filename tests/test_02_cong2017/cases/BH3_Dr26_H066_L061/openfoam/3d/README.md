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
`p_rgh` against the face operators used for gravity and CSF. The projection is
weighted by the startup inverse momentum diagonal
`rAU ~= deltaT/rho`, matching the leading-order pressure operator used by
`compressibleInterFoam` across the water/air density jump. It separately
reports EOS/pressure fixed-point convergence, residual-force acceptance, the
residual maximum location, and the predicted first-step water velocity; a
converged fixed point is not mislabeled as a balanced force state. It runs only
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

`make_geometry.py` builds an exact OpenCASCADE fluid domain and generates a
boundary-fitted mesh with named inlet, cap, wall, riser-wall, and atmosphere
physical groups. The base and refined profiles use independent
pipe/riser/atmosphere target sizes and 40/64 curvature elements per full
circle; no cut-cell background or rectangular equivalent flow area enters the
solution. Both the initial free-surface centre plane and the Valve #4 circular
cross-section are conformal internal mesh surfaces. A separately labeled
`interface` diagnostic inherits the refined profile, adds conformal planes at
both edges of the 15 mm transition (`z=0.6525/0.6675 m`), and targets `2.5 mm`
tetrahedra only in `0.63<=z<=0.69 m` around the riser.

The `prism` diagnostic also inherits the refined targets but replaces only that
60 mm axial interval with 24 conformal `Prism6` layers, each exactly `2.5 mm`
high. Its 25 layer planes include `z=0.6525`, `0.6600`, and `0.6675 m`
exactly. The pipe/tee, riser outside the slab, and expanded external atmosphere
remain tetrahedral. All Booleans finish at the valve plane before the slab is
extruded; shared CAD faces connect its top and bottom to the tetrahedral
regions. Mesh generation asserts the layer coordinates, prism count, shared
faces, and CAD volume. The `prism_atmosphere` diagnostic keeps that interface
slab and additionally extrudes the external air from `z=1.85` to `3.0 m` as
92 exact 12.5 mm vertical prism layers. The initial 46-layer, 25 mm attempt
failed strict `checkMesh` at the physical-rim tetrahedron/prism transition;
halving the layer height reduces that size jump and the prism end-cell aspect
ratio. The resulting 471,331-cell mesh passes the full geometry/topology check
with minimum determinant `0.002026`, minimum interpolation weight `0.06023`,
maximum aspect ratio `17.66`, and maximum non-orthogonality `56.12 deg`; the
closed-valve baffle mesh also passes. This profile removes the top circular-seam
tetrahedron whose 0.352 mm face-to-cell distance produced an audited acoustic
Courant number of `490.5` at `maxDeltaT=5e-4 s`; it does not change the CAD
domain or boundary locations. Every profile starts the square external
atmosphere at the physical rim `z=1.85 m`; the prior tetrahedral construction's 1 mm
annular Boolean overlap below the rim was removed so mesh sensitivities share
the prism profile's exact physical domain. These profiles test the spatially
located CSF defect
without altering physical inputs. Every run repeats the strict mesh check
after creating its valve baffle and aborts before solving unless that final
mesh reports `Mesh OK`.
The paired FLUENT study used a much finer wall-resolved hybrid mesh; the present
tetrahedral profiles are accepted only through their reported base/refined
sensitivity and must not be described as resolving the reported sub-millimetre
falling film a priori.

```bash
MESH_PROFILE=base ./Allmesh
MESH_PROFILE=refined ./Allmesh
MESH_PROFILE=interface ./Allmesh
MESH_PROFILE=prism ./Allmesh
MESH_PROFILE=prism_atmosphere ./Allmesh
```

Neither generated STL surfaces nor `constant/polyMesh` are committed.

## Runs

```bash
# Build the case-local discrete pressure initializer
bash ./Allwmake

# Closed-valve static hold
python3 run_study.py --variant closed_base

# Paired-CFD surface-tension omission diagnostic (not the experiment baseline)
python3 run_study.py --variant closed_sigma_zero

# Isolate the initial diffuse-band defect with both a sharp step and sigma=0
python3 run_study.py --variant closed_sharp_sigma_zero

# Measure mesh dependence of the better diffuse/zero-sigma diagnostic
python3 run_study.py --variant closed_refined_sigma_zero

# Restore the measured surface tension on the same refined mesh
python3 run_study.py --variant closed_refined_sigma_072

# Test the curvature-normal gradient only; all physical inputs stay fixed
python3 run_study.py --variant closed_refined_sigma_072_nhat_ls
python3 run_study.py --variant closed_refined_sigma_zero_nhat_ls
python3 run_study.py --variant closed_refined_sigma_072_nhat_point

# Align and locally refine the full initial transition band
python3 run_study.py --variant closed_interface_sigma_072_nhat_ls
python3 run_study.py --variant closed_interface_sigma_zero_nhat_ls

# Replace the transition neighbourhood with exact axial prism layers
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_repeat
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_serial
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_dt_fine
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_wave
python3 run_study.py --variant closed_prism_atmosphere_sigma_072_nhat_ls_fixed
python3 run_study.py --variant closed_prism_atmosphere_sigma_072_nhat_ls_wave
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_dt_5e6
python3 run_study.py --variant closed_prism_sigma_072_nhat_ls_wave_dt_5e6
python3 run_study.py --variant closed_prism_sigma_zero_nhat_ls
python3 run_study.py --variant closed_prism_sigma_zero_nhat_ls_serial
python3 run_study.py --variant closed_prism_sigma_072_nhat_point
python3 run_study.py --variant closed_prism_sigma_072_nhat_point_serial
python3 run_study.py --variant closed_prism_sigma_zero_nhat_point

# Isolate linear-band edge curvature with a volume-preserving cosine profile
python3 run_study.py --variant closed_refined_sigma_072_cosine

# Open-valve numerical smoke
RUN_MODE=event VALVE_OPENING=instant END_TIME=0.02 ./Allrun

# Full first-event window
RUN_MODE=event VALVE_OPENING=instant END_TIME=13 ./Allrun
```

`VALVE_OPENING` accepts `instant`, `0.2`, or `0.5`. `C_ALPHA`, `MAX_CO`,
`MAX_ALPHA_CO`, `MAX_DELTA_T`, `ALPHA_SMOOTH_CURVATURE`,
`NHAT_GRADIENT_SCHEME`, `SURFACE_TENSION`, and
`ATMOSPHERE_PRESSURE_BOUNDARY` expose declared numerical controls.
`ATMOSPHERE_PRESSURE_BOUNDARY` accepts `fixed-hydrostatic` or the
`wave-transmissive` acoustic-outflow diagnostic; the latter uses
`waveTransmissive` with `gamma=1.4` and no far-field relaxation. Both begin
from the same isothermal hydrostatic face values. `NHAT_GRADIENT_SCHEME`
accepts `gauss-linear`, `least-squares`, or `point-cells-least-squares`; the
latter two are A/B diagnostics for the flat-interface curvature normal on
tetrahedra, not unreported baseline changes.
Parallel runs use deterministic axial `simple` decomposition with
`n=(nProcs 1 1)`. This keeps the long pipe partitions reproducible and avoids
confounding physical/operator comparisons with Scotch's changing partitions.
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
physical-sigma event gate remains closed. The matched cosine-band diagnostic
is not an improvement (`0.12721 m/s`, residual `2171.2`), so the linear
band-edge derivative jumps alone are not a demonstrated cure. Because the
cosine profile also has a larger peak gradient, this comparison is not treated
as a pure one-variable proof about curvature. On the unchanged refined mesh,
correcting `rAU` alone leaves the physical-sigma result essentially unchanged
(`0.12546 m/s`, `1596.3 Pa/m`), while changing only `nHat` to
`leastSquares` reduces the reconstructed residual to `958.5 Pa/m` and the
`0.05 s` water-weighted speed to `0.03173 m/s`. This isolates the main
improvement to the curvature-normal gradient, but still fails the
`0.02 m/s` gate. Its maximum residual is at
`(3.4593,-0.00019,0.66644) m` with `alpha.water=0.0706`, spatially locating
the remaining defect at the upper edge of the transition near the tee.
The first interface-aligned tetrahedral attempt is negative evidence: it
passes strict `checkMesh`, but has 960,212 cells, `67.79 deg` maximum
non-orthogonality, `2741.5 Pa/m` initial residual, and reaches
`0.04948 m/s` by its deliberately stopped `0.028 s` rejection window. Audit
then found that lowering the global Gmsh size floor had also released
curvature refinement along the full pipe; the source now applies a spatial
size-floor callback before repeating this profile. On the unchanged refined
mesh, `pointCellsLeastSquares` further reduces the physical-sigma result to
`0.01598 m/s` and `721.8 Pa/m`; the `0.05 s` thresholds pass, but speed is
again rising at the endpoint and the required 1 s duration is incomplete.
This is a promising short diagnostic, not a formal hold pass. The axial-prism
profile with `leastSquares nHat` is substantially better: over `0.05 s`, its
maximum water-weighted speed is `2.36e-5 m/s`, sampled riser speed is
`6.72e-4 m/s`, and the initial reconstructed residual is `693.4 Pa/m`.
The water-side signal decays to the few-micrometre-per-second range rather than
growing. This short result selects the prism profile for the formal
`closed_base` candidate, but it is not itself a 1 s pass and does not relax
the acceptance threshold. The first formal rerun diverged from the successful
short trajectory at `t=0.031 s` and aborted when the energy equation produced
`T0=-2882.2 K`; no formal metric was accepted. The next declared prism tests
pair `pointCellsLeastSquares` with four-process and serial execution and with
`sigma=0`, separating capillary residual from parallel/energy sensitivity.
The physical-sigma four-process point-gradient run failed earlier, at
`t=0.019 s` with `T0=-3028.1 K`, whereas the otherwise identical serial run
completed `0.05 s` with `0.00784 m/s` maximum water-weighted speed and
`5.45e-8` total-mass residual fraction. The four-process `sigma=0` pair also
completed, with `2.36e-5 m/s` maximum water-weighted speed and `3.09e-9`
total-mass residual fraction. Thus capillary forcing and partition-sensitive
coupling, rather than an incorrect initial `296.15 K` field, are implicated.
The former Scotch decomposition changed cell allocations between nominally
identical runs (`125182/125405/126675/125546` versus
`125566/125685/125677/125880`), so it has been replaced by the deterministic
axial partition before accepting any repeat. Two fixed-partition
physical-sigma repeats now complete with byte-identical time series and
identical metrics: `2.11e-5 m/s` maximum water-weighted speed and `3.28e-9`
total-mass residual fraction. The new extrema audit bounds temperature at
`284.20--307.14 K`; both extrema occur at the top of the external atmosphere,
which locates the remaining gas/energy oscillation. The deterministic 1 s
hold then failed at `t=0.086397 s`. At failure, the water-weighted speed was
still only `2.11e-5 m/s`, with water-volume and total-mass residual fractions
of `2.39e-9` and `3.28e-9`; the external-gas velocity instead grew abruptly
and the temperature inversion exceeded its 100-iteration limit. This rejects
the current fixed atmospheric-pressure boundary for the formal hold without
implicating water-side balance or global mass loss. The matched
`waveTransmissive` diagnostic completes `0.12 s`, crossing the former failure
time. It keeps temperature within `293.62--297.66 K`, water-weighted speed
within `3.24e-5 m/s`, and total-mass residual fraction within `1.23e-9`.
That tetrahedral result still had a rising `1.04 m/s` top-boundary gas speed,
so the outlet mesh remained a confounding factor. A subsequent matched pair on
the strict 92-layer atmosphere mesh now isolates the boundary condition.
Fixed hydrostatic pressure completes `0.12 s` but reaches `0.1487 m/s`
gas-weighted speed and `273.46--318.71 K`; `waveTransmissive` limits these to
`0.00457 m/s` and `295.52--296.80 K`. The latter keeps maximum
water-weighted speed at `3.84e-5 m/s`, with water-volume and total-mass
residual fractions of `7.89e-9` and `7.99e-9`. Its audited atmosphere
acoustic Courant number is `31.64` at the unchanged `maxDeltaT=5e-4 s`,
down from `490.5`. This selects the layered `waveTransmissive` configuration
for the formal hold. The unchanged configuration now completes `1.0 s` and
passes: maximum water-weighted speed is `9.72e-5 m/s`, free-surface and
isolated-pocket volume drifts are zero at the retained resolution, and
water-volume/global-gas-mass/total-mass residual fractions are
`2.81e-8/9.21e-7/3.11e-8`. The all-domain gas-side maximum is
`0.00884 m/s`, while temperature stays within `295.33--297.05 K`.
The pressure initializer reports the exact atmosphere-patch acoustic Courant
number at `maxDeltaT`, and runtime output records both net and absolute
atmosphere mass flux to expose locally cancelling inflow/outflow. Failed
solvers emit compact partial metrics with an explicit failure reason; current
runs also record per-step extrema and locations for temperature, pressure,
velocity, turbulence fields, and volume fraction.

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

At the current revision, the strict layered mesh checks and formal 1 s closed
hold pass. This releases a new open-valve smoke and the 13 s event/sensitivity
runs without changing the experimental classification target.
