# Geysering Test 3 Case C9 — Cloud Agent handoff

This file is the cross-account continuation record for the agent named
`Geysering-test3CaseC9` (the replacement agent may be named
`geysering test 3 caseC9`).

## Git handoff

- Repository: `https://github.com/brant123451/geysering`
- Pull request: `https://github.com/brant123451/geysering/pull/8`
- Working branch: `cursor/c9-openfoam-3d-bf97`
- Base branch: `main`
- Last complete smoke-evidence commit: `6d960ce`
- Use the current remote branch tip, which also contains this handoff update.

All source changes and small validation artifacts are committed to the PR.
OpenFOAM meshes, decomposed processor directories, time directories, logs, and
`postProcessing/` are intentionally ignored. Consequently, another Cloud Agent
cannot resume the original VM's solver checkpoint and must regenerate the mesh
and rerun the stages from committed source.

## Completed work

1. `PAPER_AUDIT.md` records the apparatus, C9 conditions, PT1–PT4 positions,
   phase chronology, Eq. (7)/(8), and every unresolved air-pocket/tailgate
   parameter with paper-page citations.
2. `case_parameters.json`, `make_geometry.py`, and `prepare_case.py` generate a
   full three-dimensional OpenFOAM v2512 case using
   `compressibleInterFoam`, perfect-gas air, weakly compressible water, VOF,
   gravity, surface tension, a resolved tailgate, and an atmospheric plume.
3. The STL domain is topologically closed. The 142,343-cell base mesh passes
   standard `checkMesh`; `checkMesh -allGeometry -allTopology` still reports
   2,228 concave cells.
4. The no-ramp initialization and the full paper-time 0–1.00 s smoke window
   completed. Required CSV, JSON, and PNG artifacts are in `outputs/`.
5. Conservation uses the actual `rhoPhi` mass-flux column and the conservative
   MULES `alphaPhi0.water` flux. It does not use unavailable
   `alphaRhoPhi.*` fields.
6. A phase-1 continuation was started and deliberately stopped for this
   account handoff. Local ignored data reached solver time 1.75 s (paper time
   about 1.50 s); the committed post-processing histories end at paper time
   1.504 s.

## Current evidence

The authoritative machine-readable record is
`outputs/openfoam_3d_metrics.json`.

- Initialized PT2: 2.853 kPa gauge; target 2.970 kPa.
- First PT2 peak: 10.818 kPa at 0.392 s; paper 10.690 kPa at 0.500 s.
- First mixture rim crossing: 0.640 s; paper 0.730 s.
- Total/gas conservation residual through 1.504 s:
  `6.13e-6` / `4.06e-4`.
- Operational main-pocket transfer: 0.620 s; paper arrival 6.46 s.
- Upstream gas retained at 1.504 s: 8.20% (1.441 g of 17.557 g).
- The 12 m/s limiter affected as many as 18,217 cells (12.8%).
- Phase 1 is incomplete, phase 2 has not run, and eight eruptions have not
  been reproduced.

The close first pressure peak does not validate phase 2. The baseline pocket
leaves the upstream zone much too early, and the velocity limiter materially
activates. These are unresolved model/numerical discrepancies, not results to
hide or tune against the eruption count.

## Reproduce and continue

OpenFOAM v2512 is expected at `/usr/lib/openfoam/openfoam2512`.

```bash
cd tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d
python3 -m pip install -r requirements.txt
python3 prepare_case.py
cd case
./Allrun.mesh
./Allrun.initialize
./Allrun.resume smoke
./Allrun.resume phase1
./Allrun.resume full
./Allrun.postprocess
```

Do not start at `phase1` after a fresh clone: ignored time directories are not
in Git. Complete `mesh`, `initialize`, and `smoke` first. Long solver commands
should run in a persistent session.

Before claiming a completed validation:

1. Address or explicitly qualify the strict-mesh concave-cell failure.
2. Run a control sensitivity that demonstrates whether the 12 m/s limiter
   changes the pressure peak, rim crossing, pocket transport, or eruption
   count.
3. Treat pocket position/volume as unreported priors. Do not tune them
   arbitrarily to eight eruptions. Any chronology-constrained case must be
   labelled calibration rather than independent validation.
4. Complete paper time 6.50 s before reporting Eq. (8) period/phase 1.
5. Complete paper time 20.00 s before reporting phase 2, final PT2/PT3/PT4, or
   total eruption count.
6. Commit and push updated artifacts, then update PR #8 without deleting the
   recorded failed/qualified baseline.

## Start from another Cursor account

There is no documented cross-account ownership transfer for an in-progress
Cloud Agent run. The new account should:

1. Have read/write access to this GitHub repository and authorize the Cursor
   GitHub integration for it.
2. Open PR #8 and start a **new** Cloud Agent on its current head branch
   `cursor/c9-openfoam-3d-bf97`.
3. Select the existing PR/current branch (API users set
   `workOnCurrentBranch: true`) so the agent updates PR #8 instead of creating
   a separate branch.
4. Recreate account/team-scoped secrets and environment settings; they are not
   transferred through Git.
5. Ensure the old and new agents do not write to the same branch concurrently.

Suggested prompt for the replacement agent:

> Continue `geysering test 3 caseC9` from PR #8 on branch
> `cursor/c9-openfoam-3d-bf97`. Read `openfoam/3d/HANDOFF.md`,
> `PAPER_AUDIT.md`, `README.md`, and `outputs/openfoam_3d_metrics.json` first.
> Reproduce the committed base stages from source because runtime checkpoints
> are ignored. Preserve evidence of the strict-mesh failure, velocity-limiter
> activation, and 0.620 s early pocket transfer. Do not fit unreported pocket
> parameters to the target or claim phase 2 until an actual 20 s run supports
> it. Commit, push, and update the existing PR.
