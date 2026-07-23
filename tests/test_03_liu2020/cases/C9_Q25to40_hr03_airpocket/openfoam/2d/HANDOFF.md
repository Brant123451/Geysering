# C9 2D fine-mesh handoff

## Goal
Reproduce Liu et al. (2020) Case C9 chronology on a **fine 2-D mid-plane mesh** when 3-D wall-time is prohibitive.

## Paths
```text
tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/2d/
```

- Generator: `prepare_case_2d.py`
- Case: `case/`
- Meta: `MESH_META.json`
- 3-D archive (phase1 only): `../3d/case/computed_data/`

## Mesh / physics
- **nCells ≈ 217004**, 4 ranks, `compressibleInterFoam` v2512
- Paper lengths/diameters/Q/hr0/EOS(305 m/s)/`kOmegaSST` matched
- Upstream slope flattened to `z=drop` for conformal blocks (documented)
- Inlet 2-D flow rate scaled to keep paper mean velocity

## Runtime status (UTC)
- Mesh: OK, 1 region, ~217004 cells
- `Allrun.initialize` finished at solver **0.25**
- `./Allrun.resume full` running toward **20.25**
- Latest tip ~**t=7.15** (~35%); checkpoints archived:
  - `case/computed_data/checkpoints/processor_T6.75.tar.xz`
  - `case/computed_data/checkpoints/processor_T7.tar.xz`
- Rate ~1.4–2.2 sim-s/wall-h → ETA ~**6–10 wall-h** remaining if host stays awake
- Known gap vs paper: hydrostatic PT2/PT3 init not yet matched (to revisit after full trajectory)

## Important
**Phase 2 and eight eruptions have not yet been reproduced.**

Archive further checkpoints every ~1 sim-s into `case/computed_data/checkpoints/` and `git lfs push`.

## Important
**Phase 2 and eight eruptions have not yet been reproduced.**

Archive checkpoints into `case/computed_data/` (or sibling) with LFS as they appear — do not rely on VM disk alone.
