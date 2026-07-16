# Cloud Agent Bootstrap

Cloud Agents must start from branch:

`bootstrap/geysering-cases-20260711`

Each Agent works on exactly one Case directory under `tests/<test>/cases/<case>`.
Agents must not modify another Case, `paper/`, `.gitignore`, or the repository root.

## Test 3 / Case A2 assets

- Case root: `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser`
- Source paper: `references/liu2020.pdf`
- Paper parameter notes:
  `tests/test_03_liu2020/_shared/metadata/paper_reference/paper_parameters_Liu2020_JHE.md`
- Digitized experiment data:
  `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/data/digitized`
- Paper scans:
  `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/reference/paper_scans`
- Frozen 1-D model:
  `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/model`
- 3-D OpenFOAM pilot:
  `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d`

The OpenFOAM case is a pilot that requires independent verification against the
paper before its geometry, initial conditions, or boundary conditions are
treated as authoritative.

## Repository policy

Tracked assets include source code, configuration, README files, manifests,
papers, digitized data, compact CSV/JSON output, figures, reports, and final
PDFs. Generated OpenFOAM time directories, `processor*`, `postProcessing`,
meshes, logs, frame sequences, caches, and compiled legacy binaries are
excluded.

Every Agent must commit and push only to its own branch. Agents must never
force-push or merge `main`.
