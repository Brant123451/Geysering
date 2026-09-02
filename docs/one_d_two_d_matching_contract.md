# Geysering 1D--2D matching contract

This study uses the eight completed spatial calculations as a common diagnostic
reference for the one-dimensional model:

- Vasconcelos and Wright: Test A and Test B;
- Cong, Chan and Lee: B-H1, B-H3, and B-H6;
- Liu, Shao and Zhu: A2, B3, and C9.

The unfinished manuscript test section is outside the scope of this work.  No
paper claim or figure is changed until the matching baseline is frozen.

## Rules

1. Compare native ramp-start physical time.  A documented CFD settling period
   is removed analytically; no fitted time shift or event alignment is allowed.
2. Compare raw pressure at a common probe and datum.  Display offsets are never
   used in a metric.
3. Use the same observation operator on both dimensions: free surface,
   leading gas interface, gas arrival, physical rim passage, and pressure head.
4. Geometry and documented initial/boundary conditions vary by apparatus and
   case.  Physical and numerical closure coefficients do not vary by case.
5. A change is retained only after a full eight-case regression.  Improving one
   curve while degrading the branch map or conservation is not a global match.
6. The planar calculations are exploratory spatial references.  Experiments
   remain the independent validation evidence.

## Current baseline state

| Case | 2D data for matching | 1D baseline state | Immediate action |
|---|---|---|---|
| VW Test A | pressure and level histories complete | production semantics selected; unified rerun required | regenerate from one Campaign-1 source; keep sensitivity/spatial archives outside the baseline path |
| VW Test B | raw pressure and connected-core levels complete | declared case-local baseline | use as the initial junction/shaft reference |
| Cong B-H1 | Yfs, Yint, and PT1 histories to 14.85 s | unified complete-event run missing | wire the existing local valve into the persistent owner and add checkpointed output |
| Cong B-H3 | Yfs, Yint, and PT1 histories to 20 s | unified complete-event run missing | use the same persistent runner and re-extract the leading gas interface |
| Cong B-H6 | Yfs, Yint, and PT1 histories to 20 s | unified complete-event run missing | use the same observer and persistent runner as B-H1/B-H3 |
| Liu A2 | standardized pressure and riser histories complete | case-local zero-retuning baseline exported | merge entrainment and the mixture column into the common Campaign-3 source |
| Liu B3 | standardized pressure and base-connected VTP riser histories complete | case-local zero-retuning baseline exported | migrate unchanged onto the C9 superset before shared-physics changes |
| Liu C9 | audited event/pressure summaries to 20.25 s | all-water and sealed-pocket 20 s variants exported | repair the sealed-pocket long-horizon pressure runaway in the common source |

Direct zero-retuning 1D--2D comparisons now exist for Liu A2 and B3, and a
summary-level comparison exists for Liu C9.  These remain case-local baselines,
not one frozen global model.  Campaign-3 results and discrepancy metrics are
recorded in `docs/campaign3_1d_2d_baseline_v1.md`.

## Common metrics

The first global report will contain, where the source data permit:

- geyser/no-geyser branch;
- gas-arrival time at the shaft;
- free-surface and leading-interface trajectories;
- physical-rim arrival or overtop time;
- pressure-head history, peak, peak time, and collapse time;
- event ordering;
- liquid and gas balance diagnostics.

Curve errors are computed only over the common valid time window.  RMSE, signed
bias, and event-time error are reported separately so that a phase error cannot
be hidden by a low amplitude error.

## Execution order

1. Freeze one 1D entry point per campaign and hash all model sources.
2. Repair the observation interfaces: common leading-gas definition for Cong;
   Liu A2 probe-column and B3 VTP-connected-height extraction are complete;
   retain summary-only handling for Liu C9 until its raw probe archive is
   restored.
3. Run the unchanged 1D model once for all eight cases and publish the baseline
   discrepancy matrix.
4. Modify only shared physics, beginning with T-mouth exchange, local valve
   impedance, and shaft countercurrent/film closure.
5. Re-run all eight cases after every retained change and freeze the selected
   source hashes and outputs.

The machine-readable source map is
`tools/one_d_two_d_matching/case_registry.json`; run
`py -3.12 tools/one_d_two_d_matching/audit_registry.py --root .` to verify it.
