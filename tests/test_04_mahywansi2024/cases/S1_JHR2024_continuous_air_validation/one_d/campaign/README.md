# S1 1-D campaign protocol

This directory contains orchestration contracts only. It does not implement
horizontal-pipe physics, the vertical two-fluid solver, the two T nodes, the
exterior plume, or the canonical trajectory observer, and it cannot promote
any of those components to production status.

## Evidence boundary

- Published: the Stage-1 isolation valve is closed; after steady water flow is
  established, Stage 2 opens the continuous air boundary at 5700 Pa gauge.
- Derived from published dimensions and heads: the reference velocity, volume
  flow, mass flow, pressure difference, and 15.649 s ideal advection scale in
  `S1_CAMPAIGN_PROTOCOL.json`.
- Declared audit gate: at least 16 s Stage-1 coverage, the final 4 s stability
  window, its fixed pressure/velocity/flow thresholds, the 120 s fail-closed
  guard, the 0.1 s exact common grid, and the 25 s Stage-2 plan.
- Missing from the paper: a fixed Stage-1 settling time. The controller never
  describes 16 s, 3 s, or any other fixed duration as a published value.

The stability thresholds are copied numerically from the frozen 2-D
`STAGE1_STABILITY_GATE.json` before any formal 1-D trajectory. They are not
eruption targets and may not be edited after examining a 1-D response.

## Formal sequence

1. Create the source-aligned state once and run the closed-air Stage 1.
2. Sample accepted states at exact 0.1 s ceilings. Starting at 16 s, evaluate
   the full final 4 s using linear slope, half-window shift, detrended range,
   forward-flow, and inlet/outlet balance statistics.
3. A numerical pass is only `STABLE_CANDIDATE_REQUIRES_MANUAL_ACCEPTANCE`.
   A trusted callback must accept it before any checkpoint or progress marker
   is written.
4. Serialize that accepted state, verify deterministic decode/re-encode, save
   it with SHA-256, but continue in memory with the identical state object.
5. Switch only the boundary command to the published 5700 Pa gauge reservoir.
   This opening is Stage-2 `t=0`; the state clock remains continuous.
6. Advance the solver itself to every `t = 0.1 n` Stage-2 ceiling. A runner
   that returns a missed or interpolated ceiling fails the protocol.
7. At 25 s, the controller may write only `RUN_COMPLETE_UNVALIDATED`. Eruption
   classification and 1-D/2-D/paper validation remain separate downstream
   gates.

## Authorization layers

- `PREPRODUCTION_SMOKE_AUTHORIZED`: permits one 0.02 s closed-air smoke even
  when the operator is not production ready. It writes no campaign marker.
- `MODEL_IMPLEMENTATION_ACCEPTED` and `FORMAL_CAMPAIGN_AUTHORIZED`: both are
  required for a formal long run, and the concrete runner must independently
  report `production_ready == true`.
- `RUN_COMPLETE`, `RESULT_ACCEPTED`, and `ERUPTION_ACCEPTED` are forbidden to
  this controller.

Every authorization is a JSON object containing the registered schema, case
identifier, exact campaign-config SHA-256, and `"authorized": true`. The test
suite creates such files only under temporary test directories; this package
does not create real authorizations or real completion markers by itself.

## Integration boundary

`contracts.py` defines `ExactAdvanceRunner`, `StateCodec`, and
`ObservationBridge` protocols. The observation/runner work can be connected
only after its own implementation gate passes. In particular, the bridge must
provide native P1--P6 gauge pressures, physical velocity vectors at those
locations, and accepted gross inlet/outlet water rates; the campaign must not
infer these values from rendered interfaces or synthesize a reduced pressure.
