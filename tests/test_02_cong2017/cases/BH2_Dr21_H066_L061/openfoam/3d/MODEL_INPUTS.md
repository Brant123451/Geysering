# B-H2 initial and boundary-condition register

This register distinguishes paper inputs from explicit numerical choices.  All
pressures used by `compressibleInterIsoFoam` are absolute.  Coordinates and
paper evidence are defined in `PAPER_AUDIT.md`.

## Material and initial state

| Quantity | Value | Basis |
|---|---:|---|
| gravity | `(0 0 -9.81) m/s2` | Cong et al. notation |
| temperature | `296.15 K` | laboratory temperature 23 °C, main paper p.4 |
| water density | `998 kg/m3` | main paper p.4 |
| water dynamic viscosity | `9.32e-4 Pa s` | water at 23 °C |
| air EOS | ideal gas, `M=28.965 kg/kmol` | physical compressibility; companion 3-D formulation |
| air dynamic viscosity | `1.84e-5 Pa s` | air at 23 °C |
| surface tension | `0.072 N/m` | main paper p.5 |
| wall contact angle | `90 deg`, static | neutral, unfitted choice because the paper reports none |
| wall roughness | geometrically smooth | paper apparatus is acrylic; no fitted roughness |
| turbulence | standard `k-epsilon` RAS | companion 3-D method; no case-dependent coefficients |

Initially `U=(0 0 0)` everywhere and `T=296.15 K`.  Water fills the main for
`x<5.98 m` and the riser below `z=0.635 m`.  Air fills the downstream pocket,
riser headspace, and external domain.  Absolute pressure is referenced as
`p_ref=101325 Pa` at the main centreline `z=0`, where the initial pocket lies.
The quiescent atmospheric profile is
`p_atm(z)=p_ref-rho_air*g*z`.  The water hydrostatic reduced pressure is

```text
p_rgh,w = p_atm(zfs) + rho_w*g*zfs
        = 101325 + (998-1.191912085)*9.81*0.635
        = 107534.4665 Pa.
```

The pocket, riser headspace, and external air start with this hydrostatic
atmospheric profile and `p_rgh=101325 Pa`; water uses the constant hydrostatic
`p_rgh=107534.4665 Pa`.  Mixed interface cells use the corresponding
alpha-weighted expression, which gives continuous absolute pressure at the
initial riser free surface.  The water and pocket-air pressures are allowed to
jump across the closed valve plane.

The analytic initial downstream-pocket audit is:

```text
Vair = pi D^2 L0 / 4 = 1.197732199e-3 m3 = 1.197732 L
rhoair(101325 Pa, 296.15 K) = 1.191912085 kg/m3
mair,pocket = 1.427591483e-3 kg
```

Mesh-discrete volume and mass are recorded separately; they are not forced to
equal the analytic values by adjusting geometry or phase fraction.

## Temporal controls

The solver uses adaptive time stepping with `maxCo=0.35`,
`maxAlphaCo=0.35`, `maxDeltaT=5e-4 s`, and one geometric-alpha subcycle.
Closed-valve-only tests of the OpenFOAM tutorial's 0.5 Courant limits failed
at `t=1.636 s` with `maxDeltaT=1e-3 s` and at `t=1.9695 s` after capping
`maxDeltaT=5e-4 s` (the reported maximum Courant number had reached 0.528).
No open-valve classification result existed when these limits were selected.
The accepted limits are the tested stable lower bound, fixed for the closed
hold and every event/sensitivity run.

Temperature and kinetic-energy convection use bounded first-order upwind,
matching the OpenFOAM v2512 3-D `compressibleInterIsoFoam` tutorial.  A
pre-production run with `limitedLinear` temperature convection generated a
non-physical `T=-102 K` interface cell at `t=8.13985 s`; the run was rejected.
`T` is included in the 0.01 s field-extrema audit for every accepted run.

## Patch conditions

| Patch | Physical meaning | `U` | `p_rgh` | `p` | `alpha.water` | `T` | `k/epsilon/alphat` |
|---|---|---|---|---|---|---|---|
| `inlet` | upstream constant-head tank at `x=0` | `pressureInletOutletVelocity` | `totalPressure`, reduced total `p0=107534.4665 Pa` | `calculated` | `inletOutlet`, inflow 1 | `inletOutlet`, 296.15 K | low-turbulence `inletOutlet` |
| `downstreamCap` | plastic closed end | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |
| `walls` | main, riser and external-domain floor | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |
| `atmosphere` | open sides/top of external air domain | `pressureInletOutletVelocity` | `exprFixedValue`, local atmospheric static `p_rgh` | `calculated` | `inletOutlet`, inflow 0 | `inletOutlet`, 296.15 K | low-turbulence `inletOutlet` |
| `valveCouple0/1` | coupled portion of opening valve | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` |
| `valveWall0/1` | still-blocked portion of opening valve | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |

The upstream boundary specifies head, not velocity or discharge.  Reverse flow
there is water (`alpha.water=1`).  The external boundary specifies atmospheric
total pressure and air on inflow (`alpha.water=0`).  The only downstream end of
the horizontal main is a wall; it never acts as an outlet.

The inlet intentionally prescribes a constant *reduced* total head with
`totalPressure` on `p_rgh`.  At every elevated external-boundary face the
pressure expression is

```text
p_rgh = 101325 + (rho - rho_air)*g*z,
```

so reconstructed absolute pressure is the quiescent atmospheric profile
`101325-rho_air*g*z` for either air or ejected water.  This both avoids a
water-phase pressure deficit and preserves static air under gravity.

Two pre-production event runs were rejected while establishing this mapping.
One using `totalPressure` on `p_rgh` was stopped at `t=7.061 s`, before water
entered the external domain; it would have imposed an 18--30 kPa pressure
deficit on ejected water.  A second using spatially uniform absolute pressure
was stopped at `t=0.840 s`; that condition is not hydrostatic and drove local
air velocities near 5 m/s.  Neither run produced an accepted air-arrival or
geyser-classification result, and all validation variants restart from `t=0`.

## Ball-valve process

The experimental opening duration is approximately 0.2 s.  The baseline ACMI
coupled-area fraction is

```text
Aopen/Apipe = t/0.2  for 0 <= t <= 0.2 s
Aopen/Apipe = 1      for t > 0.2 s.
```

The sensitivity variant uses `Aopen/Apipe=1` from `t=0`.  No pressure or
velocity is prescribed at the valve.  The closed-hold case uses two true wall
baffles, since a finite porous resistance cannot guarantee zero static
leakage.

## Physical rim and external air

The circular riser wall ends at `z=1.825 m`, exactly 1.8 m above the main
soffit.  The face at that elevation is internal fluid, not a pressure outlet.
Above it is a `0.30 x 0.30 x 1.20 m` external air box; only that box's four
sides and top are atmospheric patches.  This separates the physical 1.8 m
riser from the numerical plume extent and permits direct integration of
ejected water.

## Measurements

- PT1: the nearest valid cell to the downstream crown just upstream of the cap,
  represented by `(6.585, 0, 0.024) m`; the paper does not report its
  millimetre offset.  Gas-volume-weighted downstream-pocket pressure is also
  reported.
- PT2: the nearest valid cell above the main invert directly below the riser,
  `(3.47, 0, -0.024) m`.
- `Yint` is the lowest upward air-to-water `alpha.water=0.5` crossing and
  `Yfs` the highest water-to-air crossing (or the rim while its centreline is
  water-wet), reported from the main soffit (`z=0.025 m`).  This directional
  definition does not misidentify the initial free surface as an air-pocket
  nose when only one interface exists.
- `Ta` is the interpolated first crossing of `Yint=0.02 m`.  `vfs` and `vint`
  use the same maximum 0.6 s rolling climb-rate operation as the existing 1-D
  comparison, between `Ta` and either rim arrival or the maximum `Yint`.
- outlet flow and spray volume: the signed atmospheric-patch water flux is
  reported directly.  Cumulative water crossing the physical rim is obtained
  by external-domain water inventory plus signed cumulative far-field
  outflow; these are complementary stock and flux terms, not two independent
  ejection estimates.  Water is not numerically removed at the physical rim.
