# B-H1 refined 3-D — local ParaView / OpenFOAM render handoff

Completed run: `refined-open-full-tau0`  
Solver: `bh1CompressibleInterFoam` (OpenFOAM v2512)  
Mesh: refined (~2 079 770 cells)  
End time: `13 s` (writeInterval `0.1 s` → 131 field dumps `0` … `13`)  
Wall clock: ~793 653 s (~9.2 days on 4 MPI ranks)  
Execution time: ~527 621 s

## What is already in git (compact)

Under `outputs/openfoam3d/`:

| Artifact | Role |
|---|---|
| `refined-open-full-tau0-metrics.json` | Acceptance + experiment comparison |
| `refined-open-full-tau0-series.csv` | Time series (levels, PT1_proxy, masses, fluxes) |
| `refined-open-full-tau0-comparison.png` | Fig.9(a) / Fig.10(a) / 1-D overlay |
| `refined-open-full-tau0-postProcessing.tar.xz` | Full function-object archive (~7.5 MB) |
| `refined-open-full-tau0-VTK/` | Key-time ParaView packages (`alpha.water` only; see `MANIFEST.csv`) |
| `base-open-full-tau0{,p2,p5}-VTK/` | Base-mesh key-time `alpha.water` VTK |
| `postProcessing-archives/` | Function-object tarballs for all completed runs |
| `solver-logs/` | Compressed solver logs for full-event runs |
| `sensitivity-summary.{csv,json,png}` | base / refined / τ=0.2 / τ=0.5 |
| `DATA_INVENTORY.md` | Full list of what is (and is not) in git |
| `LOCAL_RENDER_HANDOFF.md` | This note |

Key metrics (refined, τ=0 instant open):

- `Ta = 8.195 s` (exp. 8.07 s, +1.5%)
- `vfs = 1.162 m/s` (exp. 0.924 m/s)
- `observed_3d_geyser = true`, classification match
- Temperature validity `292–305 K` (PASS)
- Water / gas / total mass-budget relative errors all ≪ 1% (PASS)
- Exact `waterRhoPhi` / `airRhoPhi` recorded

## Full field data (local only; gitignored)

Path on the cloud agent disk (not committed; ~37 GB):

```text
tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/3d/runs/refined-open-full-tau0/
  processor0..3/   # binary fields every 0.1 s
  postProcessing/  # also archived as tar.xz above
  log.bh1CompressibleInterFoam
  constant/polyMesh
  system/
```

Copy this entire run directory to the workstation before reconstructing.
Do **not** expect GitHub to hold the parallel dumps.

## Suggested event times for rendering

| Time [s] | Why |
|---:|---|
| 0.0 | Initial H0 / L0 inventory |
| 8.2 | Gas first enters riser (`Ta`) |
| 9.5 | Free surface reaches physical rim |
| 10.0–11.0 | Eruption / ejection window |
| 12.0–13.0 | Late plume / fall-back |

Fields of interest: `alpha.water`, `U`, `p`, `p_rgh`, `T`, `rho`.

## Quick ParaView path (from git, no 37 GB dump)

```bash
cd outputs/openfoam3d/refined-open-full-tau0-VTK
# example: free-surface at rim
tar -xJf t9p5_refined-open-full-tau0_88922.tar.xz
# open the extracted .vtm / internal.vtu in ParaView
```

Times packaged (field = `alpha.water` only, GitHub-size safe):
`0, 8, 8.2, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13`. See `MANIFEST.csv`.

Base-mesh key frames are under `base-open-full-tau0{,p2,p5}-VTK/`.
Full inventory: `DATA_INVENTORY.md`.

For velocity/pressure rendering, either re-export locally from the full run
directory or reconstruct selected times as below.

## Reconstruct + multi-field VTK (local, needs the 37 GB run)

```bash
cd tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/3d/runs/refined-open-full-tau0
source /usr/lib/openfoam/openfoam2512/etc/bashrc   # or your v2512 install

# Selected times only (recommended)
reconstructPar -time '0,8.2,9.5,10,11,12,13'

# Optional VTK export for ParaView without foam reader plugins
foamToVTK -time '0,8.2,9.5,10,11,12,13' -fields '(alpha.water U p T rho)'
```

Open either reconstructed OpenFOAM case times or `VTK/` in ParaView.

Clip / slice tips:

1. Slice on the riser mid-plane `z=0` (or `x=3.47 m` through the tee).
2. Contour `alpha.water = 0.5` for free-surface / gas-core morphology.
3. Glyph a sparse subsample of `U` in the riser only during 9.5–11 s.
4. Overlay `PT1_proxy` series from `refined-open-full-tau0-series.csv`
   (pressure morphology only — Fig.10(a) is Run B-1, not B-H1).

## Function-object archive without full fields

```bash
cd outputs/openfoam3d
tar -xJf refined-open-full-tau0-postProcessing.tar.xz
# yields postProcessing/{pt1Pt2,riserCentreline,atmosphereFluxes,...}
```

`riserCentreline` is the densest probe set for quick 1-D-style height plots
without reconstructing the volume mesh.

## Re-running postprocess only

If the run directory is present but metrics need regeneration:

```bash
cd openfoam/3d
python3 postprocess_openfoam.py \
  --run-dir runs/refined-open-full-tau0 \
  --mode full \
  --valve-duration 0 \
  --output-dir ../../outputs/openfoam3d \
  --experimental-levels ../../data/digitized/fig9a_levels.csv \
  --experimental-pressure ../../data/digitized/fig10a_pt1.csv \
  --one-d ../../outputs/caseA_model_series.csv
```

## Sensitivity context

All four full-event cases report `geyser=true` and `full_event.pass=true`.
`base-open-full-tau0` is a legacy instantaneous-open base-mesh record without
registered `waterRhoPhi`/`airRhoPhi`; valve (0.2/0.5 s) and refined runs carry
exact phase-mass fluxes. See `sensitivity-summary.json`.
