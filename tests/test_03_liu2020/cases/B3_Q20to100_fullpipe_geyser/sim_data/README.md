# B3 OpenFOAM simulation data (for local rendering)

This folder holds retained runtime data from the completed Liu2020 **Case B3**
3-D OpenFOAM validation on branch `cursor/b3-3d-openfoam-validation-9db2`.

## Layout

```
sim_data/
  baseline/          # ~105k tet production mesh, solved to 16.4 s
  refined/           # ~142k tet sensitivity mesh, solved to 6.5 s
```

Each bundle contains:

| path | contents |
|---|---|
| `constant/polyMesh/` | mesh |
| `constant/*` | g, thermo, turbulence (standalone open) |
| `0/`, `15.5/` … or `5.75/`…`6.5/` | reconstructed fields: `alpha.water`, `U`, `p`, `p_rgh` (+ `T`/`phi`/`rho` when present) |
| `postProcessing/` | full probe / flux / interface / extrema time series |
| `VTK/` | `foamToVTK` export for ParaView |
| `b3_3d.msh` | Gmsh source mesh |
| `mesh-metadata.txt`, `log.checkMesh` | mesh quality record |
| `log.compressibleInterIsoFoam.summary.txt` | head+tail of the huge solver log |
| `system_meta/` | `controlDict` + riser probe coordinates |

## Important limitation

`controlDict` used `purgeWrite 4` and `writeInterval 0.25`, so **only the last
four volume dumps** were retained on disk:

- baseline volume times: `0`, `15.5`, `15.75`, `16`, `16.25` (end state; geyser
  peak around solver `t≈4 s` is **not** in the volume dumps)
- refined volume times: `0`, `5.75`, `6`, `6.25`, `6.5`

The **complete** geyser motion is in `postProcessing/` (especially
`riserCentreline/0/alpha.water` and `probesPT/`), not in the sparse VTK times.

Full 1.5 GB / 0.6 GB solver logs were **not** uploaded; only summaries.

## Quick ParaView

1. Open `baseline/VTK/*.vtm` or the internal `.vtu` series.
2. Or open the OpenFOAM case structure under `baseline/` (has `constant/polyMesh`
   and time directories) with the ParaView OpenFOAM reader.
3. Colour by `alpha.water`; clip / slice on the x–z plane for a front view.

## Compact validation artefacts

CSV/PNG/JSON/MP4 already live in `../outputs/openfoam_3d_*`.
