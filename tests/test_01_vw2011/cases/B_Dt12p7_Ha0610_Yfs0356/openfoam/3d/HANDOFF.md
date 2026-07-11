# Cloud-agent handoff — Geysering Test 1 Case B 3D

## Repository state

* Pull request: `https://github.com/brant123451/geysering/pull/10`
* Branch: `cursor/test1-caseb-3d-0614`
* Base branch: `main`
* Case root:
  `tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356`
* CFD root: `openfoam/3d`
* Complete copy/paste prompt: `openfoam/3d/NEW_AGENT_PROMPT.md`

All source changes and compact base-mesh evidence are committed and pushed.
Large runtime artifacts are intentionally ignored, so a fresh cloud agent must
rerun solver stages.  Do not use or merge the old 2-D branch
`cursor/test1-caseb-2d-4ac2`, and do not modify the Case-A reference directory.

## Verified before handoff

1. The paper and Case-B definition were audited in `PAPER_AUDIT.md`.
2. Migrated Case-B Python paths pass their path checks.
3. OpenFOAM.com v2512 and Gmsh 4.12.1 generate a genuinely 3-D circular
   pipe/tower/exterior-atmosphere mesh.
4. The committed base mesh evidence contains:
   * 1,904,269 tetrahedra;
   * 12.1 nominal edges across the 12.7 mm tower;
   * standard `checkMesh`: `Mesh OK.`;
   * maximum non-orthogonality 70.26 degrees (two faces over 70 degrees);
   * maximum skewness 1.104;
   * minimum interpolation weight 0.0968;
   * minimum face-volume ratio 0.107.
5. The strict `checkMesh -allTopology -allGeometry` log has one documented
   OpenFOAM boundary-tetrahedron determinant diagnostic: 448 low-determinant
   cells versus 446 cells with only two internal faces.  All other strict
   checks pass.  The source records this as
   `accepted_boundary_tet_exception`, not as a strict `Mesh OK.` result.
6. `setFields` and hydrostatic `setExprFields` initialization complete.
7. A four-rank closed-valve startup reached and postprocessed 0.001 s without
   a solver crash.  Dynamic compilation of the phase-accounting function and
   valve source succeeded.  Commits `71d014d` and `2740253` make short-stage
   intervals and single-sample derivatives robust.  The committed smoke
   evidence is correctly marked incomplete: no overflow or gas entry, zero
   one-sample mass-balance error, and 0.00246 dimensionless free-surface drift.

## Still required

The scientific reproduction is **not complete**.  Continue in this order:

1. Run the full 1.0 s closed-valve hold and assess leakage, interface drift,
   pressure drift, and mass balance.
2. Run the opened-valve 0.5 s smoke case.
3. Run the 10.5 s base case through \(T^*\ge6\).
4. Run the refined grid and required timestep/valve/compressibility
   sensitivities.
5. Commit only compact CSV/JSON/plots; never commit meshes, time directories,
   `processor*`, `postProcessing`, logs, or dynamic-code output.
6. Update the PR without claiming a completed experiment until the acceptance
   fields in `outputs/openfoam_3d_metrics.json` pass.

The initial 0.001 s hold needed about 6.5 minutes on four ranks.  Its maximum
Courant number stayed near 0.3 and the timestep settled near \(1.4\times10^{-5}\)
s, so runtime cost and startup velocity should be reviewed before launching
the full window.

## Re-entry commands

```bash
git fetch origin cursor/test1-caseb-3d-0614
git switch cursor/test1-caseb-3d-0614
cd tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d

# Read these first.
less HANDOFF.md
less PAPER_AUDIT.md
less README.md

# Recreate stages from source.
./Allclean
CASEB_STAGE=mesh CASEB_MESH=base ./Allrun

./Allclean
CASEB_STAGE=hold CASEB_MESH=base OPENFOAM_NP=4 ./Allrun

./Allclean
CASEB_STAGE=smoke CASEB_MESH=base CASEB_END_TIME=0.5 \
  OPENFOAM_NP=4 ./Allrun
```

The environment requires OpenFOAM.com v2512, Gmsh with Python bindings,
NumPy, and Matplotlib.  Runtime recovery is available through
`./Allrun.resume` only when `processor*` state exists in the same VM.

## Prompt for a fresh cloud agent

For the complete task specification, copy the entire fenced block from
`NEW_AGENT_PROMPT.md`.  The shorter prompt below is only a routing summary.

> Continue Geysering Test 1 Case B as a genuinely 3-D OpenFOAM validation from
> PR #10 and branch `cursor/test1-caseb-3d-0614`. Read
> `openfoam/3d/HANDOFF.md`, `PAPER_AUDIT.md`, and `README.md` first. Preserve
> the paper audit and authenticity rules, do not use the old 2-D branch, and
> do not modify Case A. Start by verifying the short closed-valve diagnostic
> fix, then complete hold, smoke, base/refined full runs, sensitivities, and
> compact experiment/1-D/3-D comparisons. Never claim completion without
> committed runtime evidence and passing acceptance fields.

