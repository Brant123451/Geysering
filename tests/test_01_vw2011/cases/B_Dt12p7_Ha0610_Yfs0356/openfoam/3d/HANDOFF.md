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

The stock `compressibleInterFoam` startup evidence below exposed a sustained
free-surface capillary hotspot (`|U|max=3.839 m/s` at 0.006 s).  Conformal
meshing, curvature-alpha smoothing and a least-squares `nHat` gradient were
screened and rejected.  The source deck now contains a numerical candidate
using DLR-RY TwoPhaseFlow `compressibleInterFlow` at pinned commit
`de9826f9ffb24f4b635ac97fd388ebd560cfc174`, with
`isoAdvection + plicRDF + RDF`.  Its clean 0.006 s screen now passes, but this
candidate is not accepted as the baseline until the full 1.0 s closed-valve
hold passes.

The first full-hold attempt then exposed stock-deck corrector counts that made
each step perform about 18 pressure solves.  The source now materialises the
TwoPhaseFlow surface-tension-case pattern (one alpha correction, two alpha
subcycles, one outer loop, two pressure correctors and no non-orthogonal
corrector).  This lower-cost candidate passed its repeat 0.006 s screen in
587 s wall time.  Its peak velocity rose from 1.371 to 1.682 m/s relative to
the legacy-corrector RDF screen, but the 0.006 s value remained 66.5% below
stock `compressibleInterFoam`; alpha and mass bounds remained tight.  It is
therefore cleared for the full hold, not accepted as a baseline.

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
8. The pinned RDF candidate completed a clean four-rank 0.006 s screen:
   * solver exit code 0 and wall time 1118 s;
   * maximum written-time velocity 1.371 m/s at the initial tower free surface,
     versus 3.839 m/s at 0.006 s for the stock Gauss-linear diagnostic;
   * alpha range over the run
     \([-2.66\times10^{-9}, 1+3.40\times10^{-9}]\);
   * maximum gas and total balance errors \(3.74\times10^{-6}\%\) and
     \(1.53\times10^{-8}\%\), with no water above the rim.
   The first attempt exposed a teardown-only conflict when
   `fieldFunctionObjects` loaded stock `libgeometricVoF` beside TwoPhaseFlow
   `libVoF`; commit `9a59974` moved extrema into the existing accounting object,
   and the clean rerun exited normally.
9. The repeat 0.006 s screen with the materialised TwoPhaseFlow-reference
   correctors also exited normally:
   * wall time 587 s, 47.5% below the legacy-corrector RDF screen;
   * peak velocity 1.682 m/s and final velocity 1.285 m/s, versus
     3.839 m/s for stock `compressibleInterFoam` at 0.006 s;
   * alpha range
     \([-5.77\times10^{-9}, 1+3.52\times10^{-12}]\);
   * maximum gas and total balance errors \(3.16\times10^{-6}\%\) and
     \(1.80\times10^{-8}\%\), with no water above the rim.

## Still required

The scientific reproduction is **not complete**.  Continue in this order:

1. Run the full 1.0 s closed-valve hold and assess leakage, interface drift,
   pressure drift, alpha bounds and mass balance.
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

# Build the pinned geometric-VOF solver.
./build_twophaseflow.sh

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

