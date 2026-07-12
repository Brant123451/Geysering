# B-H2 paper-faithful 3-D OpenFOAM case

This directory models Cong, Chan & Lee (2017) high-speed-camera Run B-H2
without using its reported geyser classification as an input.  The model uses
OpenFOAM v2512 `compressibleInterIsoFoam`: water has constant density, air uses
the ideal-gas equation of state, and geometric VOF reconstruction limits
interface smearing on the tetrahedral T junction.  There is no pressure,
momentum, or velocity source intended to create an eruption.

Read `PAPER_AUDIT.md` before changing any dimension.  The resolved geometry is:

- circular main: `D=0.050 m`, `x=0...6.59 m`;
- circular riser: `Dr=0.021 m`, centre `x=3.47 m`;
- physical riser wall: 1.8 m above the main soffit, rim `z=1.825 m`;
- selected ball valve: `x=5.98 m`;
- atmospheric pocket: `x=5.98...6.59 m`, `L0=0.61 m`;
- closed cap: `x=6.59 m`;
- initial riser free surface: `z=0.635 m` (`H0=0.66 m` from the main invert).

The physical riser opens into a separate `0.30 x 0.30 x 1.20 m` external air
domain.  This numerical far field is not described as part of the 1.8 m
experimental riser.  It lets ejected water remain in the solution instead of
being deleted at the rim.

## Valve representation

The production mesh is fragmented at the valve plane.  `createBaffles` creates
a `cyclicACMI` pair and duplicate wall patches there.  Its coupled effective
area increases linearly from zero to one over the paper's approximately
`0.2 s` manual opening.  This is an area-law idealization, not a fitted pressure
or velocity source.  The sensitivity run uses an instantaneous opening.  The
closed-valve test replaces the ACMI pair with two impermeable wall patches.

## Reproduce

The scripts create disposable cases below `runs/`; generated meshes, time
directories, logs, `processor*`, and `postProcessing` are ignored.

```bash
# Build the base mesh, initialize fields, and run all geometry/topology checks.
./Allrun prepare base baseline

# Closed-valve hydrostatic hold and one-step open-valve smoke test.
./Allrun closed base
./Allrun smoke base baseline

# 13 s event-window runs.
./Allrun solve base baseline
./Allrun solve refined baseline
./Allrun solve base instant

# Convert raw function-object output to compact CSV/JSON/PNG.
python3 postprocess.py
```

`baseline` means a 0.2 s linear effective-area opening; `instant` means fully
open at `t=0`.  Environment variables can reduce MPI ranks but not alter
physical parameters:

```bash
BH2_NP=4 ./Allrun solve base baseline
```

## Outputs and validation policy

`postprocess.py` writes only compact files below `results/`.  It reports:

- downstream crown pressure and downstream-pocket gas-weighted pressure;
- directional `Yfs` and `Yint` crossings from the riser centreline alpha
  profile, with `Ta`, `vfs`, and `vint` reduced using the same operational
  definitions as the existing 1-D comparison;
- inlet and atmospheric volume/mass flow;
- cumulative water volume entering the external domain;
- water and gas mass balances;
- mesh and valve-opening sensitivity;
- experiment, existing 1-D model, and 3-D metrics side by side, including
  scalar errors and classification matches.

The experiment reports only scalar B-H2 metrics (`Ta=7.84 s`,
`vfs=0.768 m/s`, `vint=1.022 m/s`, geyser).  It does not provide a digitized
B-H2 pressure or level history in this Case, so no time-series RMSE is
manufactured.  The existing 1-D model used a different 6.0 m/2.88 m geometry;
the comparison labels that difference and never shifts the 3-D time axis.

## Known, non-fitted numerical choices

`MODEL_INPUTS.md` records every patch condition and constitutive input.  The
neutral 90-degree contact angle, smooth no-slip walls, external-domain size,
and linear valve-area law are explicit uncertainty sources, not calibration
knobs.  The base/refined meshes resolve the 21 mm riser with approximately
6/8 cells across its diameter away from the locally finer T junction.  This is
a practical sensitivity pair, not a claim to resolve the reported
0.6--1.2 mm falling film pointwise.
