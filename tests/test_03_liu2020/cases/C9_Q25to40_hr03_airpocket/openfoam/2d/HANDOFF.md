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
- Mesh: OK, 1 region
- `Allrun.initialize` finished at solver **0.25**
- `./Allrun.resume full` started toward **20.25**
- Init-tip rate ≈ **2.9 sim-s/wall-h** → ETA for 20.25 roughly **7–12 h** (up to ~20 h if stiff)

## Important
**Phase 2 and eight eruptions have not yet been reproduced.**

Archive checkpoints into `case/computed_data/` (or sibling) with LFS as they appear — do not rely on VM disk alone.
