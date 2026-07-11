# B-H6 paper-faithful 3-D OpenFOAM case

This directory models Cong, Chan & Lee (2017) Series-B run B-H6.  It is the
strict non-geyser counterpart of B-H1: the riser diameter is `0.041 m` instead
of `0.016 m`; geometry other than `Dr`, initial conditions, boundary
conditions, valve law, solver, discretisation, and post-processing definitions
are common.  The measured `NO GEYSER` label is a comparison result and is not
used to force the calculation.

Read `PAPER_AUDIT.md` first.  It is the gate for this model and resolves the
repository's conflicting dimensions from the primary paper.

## Physical and numerical domain

| Quantity | Value |
|---|---:|
| Circular main-pipe inside diameter | `D = 0.050 m` |
| Main-pipe length | `6.59 m` |
| Circular riser inside diameter | `Dr = 0.041 m` |
| Tee axis | `x = 3.47 m` |
| Selected ball-valve plane | `x = 5.98 m` |
| Downstream closed cap | `x = 6.59 m` |
| Physical riser rim | `1.8 m` above pipe soffit |
| External-domain top | `3.0 m` above pipe soffit |
| Upstream head / initial riser level | `H0 = 0.66 m` above pipe invert |
| Initial downstream pocket | `x = 5.98...6.59 m`, atmospheric |
| Temperature | `23 degC = 296.15 K` |

`make_mesh.py` forms one genuine three-dimensional fluid volume by fusing a
circular main pipe, a circular riser, and an external atmosphere box.  There
are no wedge, empty, thin-layer, or equivalent-rectangle patches.  The
physical riser ends at `z=1.825 m`; the box above it is external air, not an
artificially lengthened riser.

## Solver and material model

The solver is OpenFOAM v2512 `compressibleInterFoam`.  Air uses `perfectGas`
and therefore compresses and expands with its solved absolute pressure and
temperature.  Water uses the paper's constant `998 kg/m3` density; water
acoustic compressibility is not needed to represent the entrapped-gas spring.
Surface tension is `0.072 N/m`.  There is no pressure, velocity, or eruption
forcing and no outcome-dependent source.

The passive valve model is the only momentum sink.  A `10 mm` cell zone at the
measured valve plane represents a quarter-turn valve.  A Darcy sealing term
prevents numerical leakage while the smoothstep area fraction is below `0.02`;
it then vanishes.  During opening, the contraction loss follows
`K=((1-A)/A)^2`, with `A` bounded only while the valve is nominally closed.
The OpenFOAM-native Darcy--Forchheimer option is updated at 20 equal opening
stages, restarting from the immediately preceding field state; both terms are
zero after opening.  The same staging and law are used for the `0.10 s` and
`0.40 s` sensitivity runs; no coefficient is fitted to B-H6 observations.

## Initial and boundary conditions

Absolute atmospheric pressure is `101325 Pa`.  Main-pipe centreline is `z=0`,
so pipe invert is `z=-0.025 m`, initial free surface is `z=0.635 m`, and the
water-side reduced pressure is
`p_rgh = p_atm + rho_w g (0.635 m)`.
`p_rgh` is the solved pressure and exactly encodes that hydrostatic state.
Before decomposition, the `hydrostaticInitialize` function writes the
thermodynamic `p` field as `p_rgh + rho gh` in water and `p_atm` in air.  This
avoids using the first transient pressure correction as an initializer.

| Region / patch | `U` | `p` / `p_rgh` | `alpha.water` | `T` |
|---|---|---|---|---|
| Water-filled pipe upstream of valve | `0` | hydrostatic to `H0` | `1` | `296.15 K` |
| Initial pocket, valve to cap | `0` | atmospheric | `0` | `296.15 K` |
| Riser below initial level | `0` | hydrostatic to `H0` | `1` | `296.15 K` |
| Riser above initial level and external air | `0` | atmospheric | `0` | `296.15 K` |
| `reservoir` at `x=0` | pressure-driven | fixed `p_rgh` for `H0`; calculated `p` | inlet water `1`, outflow zero-gradient | inlet `296.15 K` |
| `walls` including closed cap | no-slip | `fixedFluxPressure`; calculated `p` | neutral `90 deg` constant contact angle | adiabatic zero-gradient |
| external `atmosphere` top/sides | pressure inlet/outlet | `prghPressure`, `p=101325 Pa` | incoming air `0` | incoming `296.15 K` |

The downstream cap remains a wall.  The physical riser opening is internal to
the fused riser/external-air volume; water is not deleted at the rim.

## Required campaign

OpenFOAM v2512, Gmsh, NumPy, and Matplotlib are supplied by
`.cursor/Dockerfile`.

```bash
chmod +x Allrun Allclean

# Mesh plus a short run through full valve opening
BH6_PROFILE=smoke ./Allrun

# One profile, with compact results outside the generated case
BH6_PROFILE=base \
BH6_RESULTS_DIR="$PWD/results/base" \
./Allrun

# Static, base/refined, and valve-time campaign
python3 run_campaign.py --results-dir results
```

Profiles are:

| Profile | End time | Mesh | Valve duration |
|---|---:|---|---:|
| `static` | `1 s` | base | held closed |
| `smoke` | `0.30 s` by default | base | `0.20 s` |
| `base` | `13 s` | base | `0.20 s` |
| `refined` | `13 s` | refined | `0.20 s` |
| `valve-fast` | `13 s` | base | `0.10 s` |
| `valve-slow` | `13 s` | base | `0.40 s` |

The post-processor writes only compact CSV/JSON/PNG.  It reports `PT1`,
`Yfs`, `Yint`, entrapped/apparatus air, far-field flow, external-water
inventory and cumulative expelled water; compares experiment, frozen 1-D, and
3-D without a time shift; and audits liquid volume and gas mass including open
boundary fluxes.

## Generated-file policy

Never commit `processor*`, `constant/polyMesh`, numerical time directories,
`postProcessing`, `.msh`, dynamic-code builds, logs, or frame sequences.
`Allclean` removes them.  Only this source case and compact files under
`results/` are intended for version control.
