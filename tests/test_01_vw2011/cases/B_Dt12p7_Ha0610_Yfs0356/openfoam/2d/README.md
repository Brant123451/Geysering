# OpenFOAM 2-D Case B (VW2011 centre panel)

Paper-consistent planar pilot: lengths, `D=0.094 m`, `Dt=0.0127 m`,
`Ha0=0.610 m`, `Yfs0=0.356 m`, open top, chamber pressure 107298.3 Pa.

## Limitation

Planar extrusion cannot keep circular area ratio `(Dt/D)^2`. With paper `Dt`
as tower width, free surface reaches near the rim (`Yfs*~0.95`) but may not
fully spill. Use `../3d` for geometry-exact geyser reproduction.

```bash
./Allrun
```
