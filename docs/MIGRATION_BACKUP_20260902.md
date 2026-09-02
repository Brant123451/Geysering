# Geysering migration backup — 2026-09-02

This repository snapshot is intended for moving the project to another computer.

## Remote snapshot

- Repository: `Brant123451/Geysering`
- Branch: `codex/geysering-migration-20260902`
- Base branch: `cursor/test1-caseb-3d-0614`

On Windows, enable Git long-path support before cloning because several retained OpenFOAM source paths exceed the legacy path limit:

```powershell
git config --global core.longpaths true
```

## Included in the Git snapshot

- Current English and Chinese manuscript sources and compiled main PDFs.
- Manuscript figures, plotting and figure-generation scripts.
- Current 1D model implementations, test scripts, OpenFOAM setup/source files, and project documentation.
- Compact CSV, JSON, NPZ, HTML, and image evidence needed for the active validation figures.
- Case A interactive comparison page and its rendered frame resources.
- Intentional removal of obsolete Case A comparison HTML variants.

## Large local data not suitable for the normal Git snapshot

The following categories remain local unless separately transferred or stored through Git LFS/release assets:

- Raw OpenFOAM time directories, `processor*` decompositions, VTK/VTU exports, solver logs, temporary renders, caches, and failed/sensitivity attempts.
- `output/remote_transfer/B3_2d_completed_server_20260816/b3_2d_72k_completed_root_fields_20260816.tar.zst` (about 1.48 GiB).
- `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case/computed_data/phase1_render/VTK.tar.xz` (about 454 MiB).
- The C9 reconstructed render-field archive (about 289 MiB) under the same campaign output tree.

These large raw products are not required to edit or compile the paper, but should be copied separately if the complete CFD archive must be preserved.
