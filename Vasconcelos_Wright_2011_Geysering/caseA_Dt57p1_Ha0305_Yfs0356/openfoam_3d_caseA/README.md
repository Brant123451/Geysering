# OpenFOAM 3-D Case A fidelity model

This is the geometry-faithful model for Vasconcelos and Wright (2011), Case A.
It replaces the planar pilot in `../openfoam_2d_caseA`, which cannot preserve
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
| Pressure probe | `(1.616, -0.043, 0) m` |

The Boolean CAD domain uses a circular main pipe and circular tower, giving
the physical area ratio

```text
A_tower / A_pipe = (0.0571 / 0.094)^2 = 0.368991625
```

The tower opens into a `0.30 m x 0.30 m` external atmosphere volume extending
`1.20 m` above the physical rim. Water detected in this volume is reported
separately as above-rim ejection. It is not removed at the rim by a pressure
boundary.

## Run

The cloud image needs OpenFOAM v2512, Gmsh, NumPy, and Matplotlib, all declared
in `.cursor/Dockerfile`.

```bash
chmod +x Allrun Allrun.resume
./Allrun
```

The defaults use an `8 mm` nominal tetrahedron edge in the apparatus, a
`20 mm` atmosphere far field, at most six local MPI ranks, and an end time of
`9 s`. Override them for a smoke run or grid study:

```bash
CASEA_CORE_SIZE=0.012 CASEA_PLUME_SIZE=0.030 CASEA_END_TIME=0.02 ./Allrun
```

Resume an interrupted decomposed run with `./Allrun.resume`.

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
