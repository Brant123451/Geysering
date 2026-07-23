# OpenFOAM 2-D Case A

Compressible air-water VOF pilot for Vasconcelos and Wright (2011), Case A.

## Geometry and initial state

- horizontal acrylic pipe: `D = 0.094 m`
- upstream air chamber: `0.546 m`
- water-filled middle pipe: `2.970 m`
- tower centre: `x = 3.516 m`
- downstream closed pipe: `0.490 m`
- tower: `Dt = 0.0571 m`, `L = 0.610 m`, open top
- initial water level: `Yfs,0 = 0.356 m` above the pipe crown
- initial chamber gauge pressure: `rho*g*Ha0`, `Ha0 = 0.305 m`
- transducer: `x = 1.616 m`, sampled near the pipe invert; the comparison
  converts its piezometric head to the paper's pipe-crown datum and retains
  the uncorrected probe series for audit

The valve is represented by an initially open internal block interface at
`x = 0.546 m`; the initial pressure and phase discontinuity therefore models
an instantaneous opening. The experiment reports a manual opening in less
than one second.

## Numerical model

- OpenFOAM v2512 `compressibleInterFoam`
- laminar, isothermal-start air-water VOF
- physical air ideal-gas compressibility
- water `rho0 = 998.2 kg/m3`, `c = sqrt(2.2e6) = 1483 m/s`
- fine in-plane cell size about `2 mm` (~106,381 cells; was ~4 mm / 26k)
- VOF: `interfaceCompression` + `MULESCorr`; Co budget `maxCo=0.20`, `maxAlphaCo=0.15`
- automatic Scotch decomposition, capped at six local MPI ranks

Run in Ubuntu WSL:

```bash
chmod +x Allrun Allrun.resume
./Allrun
```

Set `OPENFOAM_NP` to request fewer MPI ranks. `Allrun` also executes
`postprocess_compare.py` and writes the compact comparison artifacts to
`outputs/`.

Resume an interrupted run with `./Allrun.resume`.

## Important 2-D limitation

The vertical-plane geometry preserves every published length and diameter,
but a planar 2-D extrusion cannot preserve the circular pipe/tower area ratio
at the same time. Its area ratio is `Dt/D = 0.607`, whereas the physical
circular ratio is `(Dt/D)^2 = 0.369`. It also cannot resolve the annular
three-dimensional wall film, and its pressure outlet at the tower rim removes
any above-rim jet from the domain. Consequently this run is a morphology and
time-history diagnostic, not a geometry-exact substitute. Use `../3d` for the
circular T-junction and above-rim atmosphere.
