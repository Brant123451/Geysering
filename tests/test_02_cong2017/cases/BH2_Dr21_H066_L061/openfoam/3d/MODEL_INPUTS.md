# B-H2 initial and boundary-condition register

This register distinguishes paper inputs from explicit numerical choices.  All
pressures used by `compressibleInterFoam` are absolute.  Coordinates and paper
evidence are defined in `PAPER_AUDIT.md`.

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
riser headspace, and external domain.  The water hydrostatic reduced pressure
is

```text
p_rgh,w = p_atm + rho_w g zfs
        = 101325 + 998*9.81*0.635
        = 107541.8913 Pa.
```

The pocket and atmospheric air start at `p=101325 Pa`; its `p_rgh` is
initialized consistently with hydrostatic air.  The two initial pressures are
allowed to jump across the closed valve plane.

The analytic initial downstream-pocket audit is:

```text
Vair = pi D^2 L0 / 4 = 1.197732199e-3 m3 = 1.197732 L
rhoair(101325 Pa, 296.15 K) = 1.191912085 kg/m3
mair,pocket = 1.427591483e-3 kg
```

Mesh-discrete volume and mass are recorded separately; they are not forced to
equal the analytic values by adjusting geometry or phase fraction.

## Patch conditions

| Patch | Physical meaning | `U` | `p_rgh` | `p` | `alpha.water` | `T` | `k/epsilon/alphat` |
|---|---|---|---|---|---|---|---|
| `inlet` | upstream constant-head tank at `x=0` | `pressureInletOutletVelocity` | `totalPressure`, reduced total `p0=107541.8913 Pa` | `calculated` | `inletOutlet`, inflow 1 | `inletOutlet`, 296.15 K | low-turbulence `inletOutlet` |
| `downstreamCap` | plastic closed end | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |
| `walls` | main, riser and external-domain floor | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |
| `atmosphere` | open sides/top of external air domain | `pressureInletOutletVelocity` | `totalPressure`, `p0=101325 Pa` | `calculated` | `inletOutlet`, inflow 0 | `inletOutlet`, 296.15 K | low-turbulence `inletOutlet` |
| `valveCouple0/1` | coupled portion of opening valve | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` | `cyclicACMI` |
| `valveWall0/1` | still-blocked portion of opening valve | `noSlip` | `fixedFluxPressure` | `calculated` | 90° `constantAlphaContactAngle` | `zeroGradient` | wall functions |

The upstream boundary specifies head, not velocity or discharge.  Reverse flow
there is water (`alpha.water=1`).  The external boundary specifies atmospheric
total pressure and air on inflow (`alpha.water=0`).  The only downstream end of
the horizontal main is a wall; it never acts as an outlet.

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
- `Yint` and `Yfs`: lower and upper `alpha.water=0.5` crossings along the
  riser centreline, reported from the main soffit (`z=0.025 m`).
- outlet flow and spray volume: atmospheric patch flux plus water volume in
  the external-domain cell zone.  Water is not numerically removed at the
  physical rim.
