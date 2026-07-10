# Geysering OpenFOAM cloud runs

This repository contains the minimal reproducible OpenFOAM case used to compare
the two-dimensional Case A simulation with Vasconcelos and Wright (2011).

## Cloud run

The Cursor Cloud Agent image is defined by `.cursor/Dockerfile` and installs
OpenFOAM v2512 plus the Python post-processing dependencies.

```bash
cd Vasconcelos_Wright_2011_Geysering/caseA_Dt57p1_Ha0305_Yfs0356/openfoam_2d_caseA
chmod +x Allrun Allrun.resume
./Allrun
```

`Allrun` generates the mesh, initializes the two-phase fields, runs
`compressibleInterFoam`, and creates the experimental-comparison plots and
metrics under `outputs/`.

Large field time directories, decomposed processor directories, logs, and
probe output are intentionally excluded from Git. Only compact comparison
artifacts should be force-added when a cloud run is complete.
