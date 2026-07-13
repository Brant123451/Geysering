# OpenFOAM 3-D Case A fidelity model

This is the geometry-faithful model for Vasconcelos and Wright (2011), Case A.
It replaces the planar pilot in `../2d`, which cannot preserve
the circular pipe/tower area ratio and terminates at the tower rim.

## Experiment mapping

| Quantity | Model value |
| --- | ---: |
| Horizontal acrylic pipe diameter | `0.094 m` |
| Upstream air chamber length | `0.546 m` |
| Valve-to-tower distance | `2.970 m` |
| Tower centre | `x = 3.516 m` |
| Closed downstream length | `0.490 m` |
| Circular tower diameter | `0.0571 m` |
| Tower length above pipe crown | `0.610 m` |
| Initial free surface above crown | `0.356 m` |
| Initial air gauge-pressure head | `0.305 m` |
| Pressure probe | `(1.616, -0.043, 0) m` (near invert) |

The Boolean CAD domain uses a circular main pipe and circular tower, giving
the physical area ratio

```text
A_tower / A_pipe = (0.0571 / 0.094)^2 = 0.368991625
```

The tower opens into a `0.30 m x 0.30 m` external atmosphere volume extending
`1.20 m` above the physical rim. Water detected in this volume is reported
separately as above-rim ejection. It is not removed at the rim by a pressure
boundary.

The audited initial absolute pressures are `104311.7 Pa` in the upstream air
chamber and `105271.3 Pa` reduced pressure in the initially water-filled
apparatus. The latter represents a hydrostatic free surface at absolute
`y = 0.403 m`. The atmosphere patch uses constant reduced pressure
`p_rgh = 101333 Pa`, corresponding to `101325 Pa` absolute pressure at the
physical rim. Both horizontal ends and the apparatus walls are closed,
no-slip walls.

The raw pressure probe is deliberately inside the fluid, `4 mm` above the
pipe invert. The paper's calibrated `H*` curves use the ventilation-tower
base/pipe-crown elevation, consistently with the repository's independently
validated small-tower case. Post-processing therefore subtracts the fixed
piezometric datum difference
`(0.047 - (-0.043))/0.610 = 0.147541` from the probe `H*`. The uncorrected
probe history is retained beside the crown-datum history in the compact CSV;
this is an elevation-datum conversion, not a fitted pressure offset.

## Run

The cloud image needs OpenFOAM v2512, Gmsh, NumPy, and Matplotlib, all declared
in `.cursor/Dockerfile`.

```bash
chmod +x Allrun Allrun.resume
./Allrun
```

The defaults use an `8 mm` nominal tetrahedron edge in the apparatus, a
`20 mm` atmosphere far field, at most six local MPI ranks, and an end time of
`9 s`. `Allrun` records both a standard `checkMesh` result and a stricter
all-topology/all-geometry audit. Override the mesh sizes for a grid study, or
request the solver's serial single-step setup check for a smoke test:

```bash
CASEA_CORE_SIZE=0.012 CASEA_PLUME_SIZE=0.030 CASEA_DRY_RUN=1 ./Allrun
```

Resume an interrupted decomposed run with `./Allrun.resume`.

The full run logs an exact domain integral of
`alpha.water*thermo:rho.water` every `0.005 s`. Post-processing converts those
samples to `outputs/openfoam_3d_water_mass.csv` and reports final and maximum
relative mass drift in `outputs/openfoam_3d_metrics.json`; raw logs and fields
remain untracked.

The paper measures the **top of the rounded air-water interface** from video and
documents that its initial rise is asymmetric and leaves a descending wall
film. A single centreline probe can therefore jump between the rounded nose,
film, and detached drops. After the solver finishes, `Allrun` samples
`areaAverage(alpha.water)` on horizontal tower sections at `10 mm` intervals.
The reported air-pocket nose is the bottom of the uppermost contiguous water
column at a section-averaged water fraction of `0.90`; the free surface uses
the conventional `0.50` crossing. The complete section averages, original
centreline estimate, and a `0.80–0.95` nose-threshold sensitivity sweep are
retained as compact CSV files.

Fig. 7 contains three separate interface trajectories from repeated manual
valve openings. The metrics retain the conservative RMSE against the complete
marker cloud and also separate the approximately parallel trajectories,
reporting a no-shift RMSE, climb velocity, and fitted catch time for each
repetition. This avoids treating simultaneous markers from different runs as
one impossible interface shape.

`constant/fvOptions` applies a coded correction equivalent to OpenFOAM's
temperature limiter directly to the solver's shared mixture-temperature field
at `250–350 K`. The experiment starts at `293.15 K`, and its pressure ratio
cannot physically approach either bound. This prevents an isolated interface
cell from passing an invalid temperature to the phase equations of state
during tower entry. The metrics record every actual limiter activation and the
unlimited extrema so this stabilization remains auditable.

## What “matched” means

This case matches the published dimensions, circular cross-sections, initial
water level, initial air pressure, closed ends, open atmosphere, gravity,
physical air compressibility, water bulk modulus, and reported probe location.
Case A is experimentally classified as non-geysering, but the external domain
allows the computation to contradict that classification rather than imposing
it.

Two experimental details are not available as resolved input data: the exact
time history of the manually opened valve (reported only as less than one
second) and measured wall wetting/roughness. The model therefore uses the
paper's instantaneous-opening idealization, smooth no-slip acrylic walls, and
no fitted contact-angle or turbulence parameters. These limitations must be
reported; a CFD result should not be described as an exact reproduction until
mesh convergence and uncertainty checks are also complete.
