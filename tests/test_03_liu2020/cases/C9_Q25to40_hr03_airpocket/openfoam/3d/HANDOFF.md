# Geysering Test 3 Case C9 — Cloud Agent handoff

This file is the cross-account continuation record for the agent named
`Geysering-test3CaseC9` (the replacement agent may be named
`geysering test 3 caseC9`).

## Git handoff

- Repository: `https://github.com/brant123451/geysering`
- Pull request: `https://github.com/brant123451/geysering/pull/8`
- Working branch: `cursor/c9-openfoam-3d-bf97`
- Base branch: `main`
- Use the current remote branch tip; historical output artifacts are retained
  as failed/qualified provenance, not as the current production result.

All source changes and small validation artifacts are committed to the PR.
OpenFOAM meshes, decomposed processor directories, time directories, logs, and
`postProcessing/` are intentionally ignored. Consequently, another Cloud Agent
cannot resume the original VM's solver checkpoint and must regenerate the mesh
and rerun the stages from committed source.

## Completed work

1. `references/liu2020.pdf` was read directly. `PAPER_AUDIT.md` now records
   the apparatus, C9 conditions, PT1–PT4 positions, boundary approximations,
   Reynolds numbers, pressure-wave speed, phase chronology, Eq. (7)/(8), and
   every unresolved air-pocket/tailgate parameter with page citations.
2. The full reported geometry is reproduced. The default 481,874-cell
   `cartesianMesh` passes standard and strict `checkMesh` with zero concave
   cells; the 873,032-cell thin-layer-refined mesh also passes.
3. Production velocity clipping is disabled. The historical 12 m/s limiter
   changed the transient and affected 12.8% of the old mesh, so that trajectory
   is retained only as failed evidence.
4. The source initialization is hydrostatic and uses smooth constant-\(Q\)
   chamber/gate streamtubes. A no-clipping diagnostic showed that the early
   chamber-side gas signal is the paper's connected thin layer, while the
   thick body remained near its initial nose.
5. Direct inspection of OpenFOAM v2512 found a critical historical EOS error:
   `perfectFluid` uses \(\rho=\rho_0+p/(RT)\), but the old generator used
   `R=K/rho`. It produced an effective water-wave speed near 25.4 km/s.
6. The corrected source uses the paper's approximately 305 m/s acrylic-pipe
   wave speed as a rigid-wall effective modulus (92.86 MPa), preserves
   998.2 kg/m³ at 101325 Pa and 293.15 K, defaults to RANS
   \(k\)-\(\omega\) SST, and carries a conservative thick-body-only source
   tracer using the air-phase mass flux. Intrinsic-water 2.2 GPa and laminar
   flow remain sensitivities.
7. Resume scripts hash the initialized source schema and reject old
   checkpoints. A new corrected run must start from `Allrun.initialize`;
   historical fields cannot be resumed because they lack RANS and tracer
   fields.

## Current evidence

`outputs/openfoam_3d_metrics.json` remains the machine-readable record of the
failed historical limiter/EOS generation until corrected smoke data replace
it. Do not compare it as if it came from the current source.

The latest local diagnostic was stopped safely at solver 0.5568 s (paper
0.3068 s) after the EOS audit:

- initialized PT2: 3.242 kPa gauge; target 2.970 kPa;
- zero limiter activation; maximum 43.7 m/s was in low-density crown gas;
- total/gas conservation residuals: `1.7e-6` / `3.4e-5`;
- upstream gas-mass retention: 94.6%;
- thick-body morphology front near \(x=-0.92\) m, while thin-layer gas had
  reached the chamber-side deep probe.

This supports the diagnosis that the old 0.620 s metric was not a coherent
main-body arrival. It does not validate pressure chronology because the run
used the erroneous 25.4 km/s EOS and laminar closure.

A corrected-EOS/RANS smoke subsequently reached paper time 1.00 s. Its first
rim crossing was 0.655 s and its no-limiter pressure history was stable, but
the then-current tracer used mixture `rhoPhi/rho`. The tracer diluted into the
water phase (maximum 1 to 0.00285) and falsely labelled the paper's early crown
passage as a 0.71 s body arrival. It is rejected for pocket chronology. The
source now transports the tag with
`interpolate(rho.air)*(phi - alphaPhi0.water)`, reports 1% transfer as leakage,
and requires 20% transfer plus a sustained connection for operational bulk
arrival. This direct phase-flux form avoids the non-equivalent subtraction of
water mass flux from compressible mixture `rhoPhi`. A fresh initialization and
smoke are required to verify this change before phase 1. The v2512
compressible scalar branch does not apply its `bounded01` switch. The initial
clear/clamp workaround was therefore bounded but non-conservative: the
`maxCo=0.70` diagnostic was stopped at solver time 0.6401 s (paper time
0.3901 s) after losing 7.65% of its paper-time-zero physical tracer inventory.
It is rejected. Two later variable-density MULES versions were rejected as
well: a function-generated `alpha*rho` carrier had an invalid old-time level,
while an explicit reconstruction of both levels still violated MULES's
low-order positivity requirement and reached order `1e24` by solver time
0.0202 s. The current `boundedPhaseMassTransport` still projects the raw gas
mass flux onto discrete gas continuity. Several follow-on tracer forms were
tested and rejected after that projection:

- Intensity Sp on `fvm::ddt(alpha,rho,s)` vanishes when the projected
  continuity residual is already small, so it collapses to the previously
  rejected product-ddt inventory loss (~5.9% by ~0.52 s).
- Clipping `sigma` onto `[0,alpha*rho]` keeps recovered `s` in `[0,1]` but
  destroys inventory at ~0.5% per 0.01 s with essentially zero open-boundary
  tagged flux.
- Face velocity `phi/(alpha*rho)_f` explodes on thin-phase faces.
- Unclipped conservative
  `sigma := sigmaOld - dt*div(flux(phi,s))` preserves the finite-volume
  inventory construction, but recovered `s = sigma/(alpha*rho)` grew above 2
  by ~1e-4 s and is not an accepted trajectory.

Latest tracer iteration (branch tip): upwind intensity fluxes
`phi_air*s` with conservative donor-outflow positivity scaling; inventory
is `sigma`; recovered `s` may exceed 1 when `alpha*rho` shrinks. Gate on
integral `sigma` conservation (<1%), not pointwise `s∈[0,1]`. Rejected
paths (do not revive): clear/clamp; variable-density MULES on `s`; Sp(ddt);
thin-floor `phiVol`; unscaled intensity `phi·s`; post-update sigma clip to
`[0,αρ]`.

### Initialize conservation gate (passed)

Fresh initialize with the positivity-scaled tracer completed through solver
0.25 s. An accidental mid-run `controlDict.full` swap (now hardened in
`Allrun.initialize` / `Allrun.resume`) overran to a clean `writeNow` stop at
`0.3289420474` (paper ≈ 0.079 s). From
`postProcessing/matrixPocketBodyTracerMass`:

- `∫sigma = 1.67259665e-02` from 0.01 through 0.32 (relative change **0**);
- numerical mass-source residual ~`1e-16`;
- inlet/gate/atmosphere tagged fluxes all zero through 0.32 (pocket still
  in-domain).

Record: `case/results-init/initialize_conservation_gate.json` (gitignored
runtime artifact).

### Smoke (in progress on this VM)

Resumed from checkpoint `0.3289420474` toward smoke `endTime=1.25`
(paper 1.00 s) with 4-rank `compressibleInterFoam`, `maxCo=0.70`,
`maxAlphaCo=0.20`.

A first smoke attempt after that checkpoint showed a **one-time ~5%
`∫sigma` drop** (1.6726e-2 → 1.5893e-2) with zero open-boundary tagged
flux, then perfect flatness. Root cause: on restart the function object
rebuilt `sigma` from `alpha*rho*s` (`NO_READ`) while unresolved cells had
written `s=0`, destroying excess inventory. That trajectory was archived
under `results-smoke/rejected_restart_jump_*` and discarded.

Fix (current tip): `READ_IF_PRESENT` for `pocketBodyTracerSigma` and do not
reseed `sigma` from `α·ρ·s` on the first post-restart execute. Smoke was
restarted from the same checkpoint with the rebuilt library.

Multi-stage 30 min CPU watchdog monitors `log.smoke`; a
smoke→conservation-gate→phase1 chain is armed. Smoke inventory acceptance
still uses `∫sigma` + boundary flux + residual <1% after `End`.

The historical, now-rejected clear/clamp `maxCo=0.35` reference remained in
`[0,1]` through solver time 0.37 s and lost about 0.5% of its paper-time-zero
inventory by 0.31 s, but cut-cell Courant control reduced its step to roughly
\(3\times 10^{-5}\) s. The selected baseline is now `maxCo=0.70` with
`maxAlphaCo=0.20`, backed by the earlier same-physics 0.35/0.70 benchmark;
0.35 remains an explicit sensitivity.

Phase 1 is incomplete. **Phase 2 and eight eruptions have not yet been
reproduced.**

## Reproduce and continue

OpenFOAM v2512 plus `openfoam2512-source` and `openfoam2512-tools` are expected
at `/usr/lib/openfoam/openfoam2512`; initialization automatically builds the
local conservative tracer function object.

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

Do not start at `phase1` after a fresh clone or reuse a pre-EOS-fix local
checkpoint: ignored time directories are not in Git, and the resume script
rejects source-schema mismatches. Complete `mesh`, fresh `initialize`, and
corrected `smoke` first. Long solver commands should run in a persistent
session.

Before claiming a completed validation:

1. Verify the corrected 305 m/s EOS, RANS fields, and body tracer in a fresh
   serial/parallel initialization and restart smoke.
2. Benchmark safe Courant/MPI settings before committing compute to phase 1;
   do not reintroduce a velocity limiter for throughput.
3. Complete the thin-layer mesh, 305 m/s versus intrinsic-water, and
   RANS-versus-laminar sensitivities needed to qualify the production choice.
4. Treat pocket position/volume as unreported priors. Do not tune them
   arbitrarily to eight eruptions. Any chronology-constrained case must be
   labelled calibration rather than independent validation.
5. Complete paper time 6.50 s before reporting Eq. (8) period/phase 1.
6. Complete paper time 20.00 s before reporting phase 2, final PT2/PT3/PT4, or
   total eruption count.
7. Commit and push updated artifacts, then update PR #8 without deleting the
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

The complete copy/paste prompt is:

`openfoam/3d/CONTINUATION_PROMPT.md`

within this C9 case. Paste the **entire contents** into the replacement Agent;
do not use only an abbreviated “continue the PR” instruction. It includes the
original physical objective, paper targets, current failed/qualified evidence,
allowed file scope, all execution stages, sensitivity requirements, required
artifacts, anti-tuning rules, Git/PR requirements, and final-report checklist.
