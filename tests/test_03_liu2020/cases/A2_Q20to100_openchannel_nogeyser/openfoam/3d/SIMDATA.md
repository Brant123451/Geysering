# Refined A2 simulation data for local rendering

This directory retains the refined-mesh `interFoam` result products needed to
rebuild ParaView/VTK front elevations and probe-based motion figures locally.

## Included

| Path | Role |
|---|---|
| `VTK/` | foamToVTK dumps at t≈-12, 12, 13, 14 s (`purgeWrite=3`) |
| `-12/`, `12/`, `13/`, `14/` | reconstructed serial OpenFOAM fields |
| `constant/polyMesh/` | refined mesh |
| `postProcessing/` | continuous probes / wet-area / fluxes |
| `system/*ProbeLocations` | probe coordinates used by the run |
| `constant/triSurface/geometry.stl` | combined surface used for meshing checks |

## Not included

- `processor*` (same 4 times as the serial dumps; redundant)
- `log.interFoam.full` (~488 MB text; not needed for rendering)

## Local render

```bash
cd openfoam/3d
python3 render_front_water_air.py
```

Requires `pyvista`, `matplotlib`, `numpy`, and `Pillow`.

## Note on gitignore

Root `.gitignore` normally excludes `postProcessing/` and `polyMesh/`.
Those paths are force-added here for this Case only so local rendering does
not need a full re-solve. Large probe/VTK blobs use Git LFS.

## Sizes at upload

- `VTK/`: 73.0 MiB
- `-12/`: 10.0 MiB
- `12/`: 26.6 MiB
- `13/`: 26.6 MiB
- `14/`: 26.6 MiB
- `postProcessing/`: 157.0 MiB
- `constant/polyMesh/`: 14.1 MiB
