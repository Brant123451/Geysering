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
- Current Case B aligned 1D/2D frames and formal connected-core metrics.
- Cong BH1/BH3/BH6 manuscript comparisons, canonical metrics, and the BH6 interactive viewer resources.
- Liu A2 comparison outputs plus compact B3 and C9 reports and manifests.
- Intentional removal of obsolete Case A comparison HTML variants.

## Large-data branch additions

The companion branch `codex/geysering-migration-large-data-20260902` stores the following through Git LFS:

- Completed B3 root fields (about 1.55 GB).
- C9 phase-1 VTK and reconstructed render-field archives (about 779 MB combined).
- Canonical Cong BH1 3D VTK time sequence (about 693 MB).
- Compact A2, B3, and C9 recovery/checkpoint archives (about 114 MB combined).

## Large local data intentionally not uploaded

The following regenerable or low-value categories remain local:

- Raw OpenFOAM time directories and `processor*` decompositions not represented by the archives above.
- Loose VTK/VTU exports, solver logs, temporary renders, caches, operating-system/bootstrap images, and failed or sensitivity attempts.

These excluded products are not required to edit or compile the paper. Copy them separately only if a complete forensic archive of every historical run is required.

At the final audit, 37,007 untracked files (17.28 GiB) remained only on the source computer: 13.03 GiB under `tests/`, 3.63 GiB under `output/`, 0.48 GiB under `tmp/`, 0.08 GiB of paper build/render intermediates, and 0.06 GiB of bundled dependencies. The retained local volume is dominated by loose CFD fields and historical intermediates rather than manuscript sources or selected formal result bundles.
