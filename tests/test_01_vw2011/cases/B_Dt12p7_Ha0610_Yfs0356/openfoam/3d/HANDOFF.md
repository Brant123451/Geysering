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
therefore a valid startup screen, not an accepted hold baseline.  The ensuing
full-hold attempt was rejected at 0.06 s: pressure drift had already exceeded
the immutable acceptance limit, and a growing velocity hotspot had moved to
the first cell upstream of the sharp penalty-valve zone.

The conformal two-sided baffle replacement passed its generated-mesh and
0.001 s startup checks, and it removed that valve-edge hotspot.  Its extended
RDF screen was nevertheless rejected at 0.04 s: the velocity maximum remained
at the initial tower free surface, while the transducer range reached
\(H^*=0.05658\), already above the 0.02 hold limit.  Alpha, phase/total
balances, rim-water inventory and gas-entry checks stayed clean.  This
separates the physically correct closed-valve topology from the remaining
free-surface curvature/pressure-balance defect.  The next numerical screen is
`plicRDF + fitParaboloid`, not another full RDF hold.  Its first 0.006 s run
reduced the written peak and final velocities to 1.187 and 0.796 m/s, but a
one-step startup Courant spike reached 1.957.  It must be repeated with
`interpolateNormal false`, as used by TwoPhaseFlow's static surface-tension
benchmarks, before any extension.

## Verified before handoff

1. The paper and Case-B definition were audited in `PAPER_AUDIT.md`.
2. Migrated Case-B Python paths pass their path checks.
3. OpenFOAM.com v2512 and Gmsh 4.12.1 generate a genuinely 3-D circular
   pipe/tower/exterior-atmosphere mesh.
4. The committed base mesh evidence contains:
   * 1,903,549 tetrahedra with a conformal 1,829-face valve plane;
   * 12.1 nominal edges across the 12.7 mm tower;
   * standard `checkMesh`: `Mesh OK.`;
   * maximum non-orthogonality 74.24 degrees (one face over 70 degrees);
   * maximum skewness 1.058;
   * minimum interpolation weight 0.0909;
   * minimum face-volume ratio 0.100.
5. The strict `checkMesh -allTopology -allGeometry` log has one documented
   OpenFOAM boundary-tetrahedron determinant diagnostic: 646 low-determinant
   cells versus 644 cells with only two internal faces.  All other strict
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
10. The connected-domain penalty-valve hold was intentionally stopped and
    postprocessed after it became impossible to pass:
    * duration 0.06 s and requested duration 1.0 s;
    * pressure peak-to-peak \(H^*=0.0570\), already above the 0.02 limit;
    * the same cell immediately upstream of `valveZone` grew from
      1.414 m/s at 0.04 s to 1.825 m/s at 0.06 s;
    * global Courant control reduced the timestep to \(2.79\times10^{-5}\) s,
      while interface and capillary controls were inactive;
    * phase/total balances remained below \(1.7\times10^{-5}\%\), alpha
      remained tightly bounded, and no rim water or gas entry occurred.
    This separates a local closed-valve pressure-support defect from the RDF
    free-surface screening result.
11. The conformal-baffle RDF screen was intentionally stopped and
    postprocessed once it also became impossible to pass:
    * requested duration 0.08 s and postprocessed duration 0.039996564 s;
    * pressure peak-to-peak \(H^*=0.05658\), above the 0.02 limit;
    * written velocity maxima remained at the initial tower free surface,
      decreasing from 1.757 m/s at 0.01 s to 1.362 m/s at 0.04 s;
    * no valve-edge hotspot or valve leakage was observed;
    * gas and total balance errors remained below
      \(6.5\times10^{-6}\%\), alpha remained tightly bounded, and no rim
      water or gas entry occurred.
    This validates the baffle topology but rejects `plicRDF + RDF` as the
    hold configuration.
12. The first conformal-baffle `fitParaboloid` screen exited normally at
    0.006 s:
    * written peak and final velocities were 1.187 and 0.796 m/s, 29.4% and
      38.1% below the reference-corrector RDF screen;
    * alpha stayed within
      \([-2.12\times10^{-22},1+4.20\times10^{-11}]\);
    * gas and total balance errors remained below
      \(3.9\times10^{-6}\%\), with no rim water or gas entry;
    * a one-step startup Courant spike reached 1.957 and interface Courant
      reached 0.676.
    The velocity reduction is promising, but the CFL regression prevents
    extension until the static-benchmark normal-interpolation setting is
    tested.

## Still required

The scientific reproduction is **not complete**.  Continue in this order:

1. Materialise and record `plicRDF`'s `interpolateNormal` control, repeat the
   conformal-baffle `fitParaboloid` screen through 0.006 s with interpolation
   disabled, and extend past the 0.04 s pressure-drift onset only if both the
   velocity and Courant metrics improve.  Only a candidate with \(H^*\)
   peak-to-peak at or below 0.02 may proceed to the full 1.0 s hold.  Opening
   runs retain the dissipative resistance and must be tested separately.
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

