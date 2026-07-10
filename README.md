# Geysering OpenFOAM cloud runs

This repository contains reproducible OpenFOAM cases for comparison with
Vasconcelos and Wright (2011), Case A. The three-dimensional case is the
experiment-fidelity model; the older two-dimensional case is retained only as
a low-cost planar diagnostic.

## Cloud run

The Cursor Cloud Agent image is defined by `.cursor/Dockerfile` and installs
OpenFOAM v2512, Gmsh, and the Python post-processing dependencies.

```bash
cd Vasconcelos_Wright_2011_Geysering/caseA_Dt57p1_Ha0305_Yfs0356/openfoam_3d_caseA
chmod +x Allrun Allrun.resume
./Allrun
```

`Allrun` generates the mesh, initializes the two-phase fields, runs
`compressibleInterFoam`, and creates the experimental-comparison plots and
metrics under `outputs/`. The 3-D domain preserves the true circular area ratio
and includes an open atmosphere volume above the physical tower rim so a
geyser, if predicted, is represented rather than clipped.

Large field time directories, decomposed processor directories, logs, and
probe output are intentionally excluded from Git. Only compact comparison
artifacts should be force-added when a cloud run is complete.
