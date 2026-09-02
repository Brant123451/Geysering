# Existing Campaign 2 outlet-evidence audit

This inventory was completed before interpreting the unified rim-plane
series. It separates what the pre-existing artifacts could prove from what
was missing.

## B-H1 refined

- Source run: `/tmp/bh1-2d-study/h1_refined_co015`.
- Existing VTK: `VTK_FRAME_WORK/h1_refined_co015_209405/internal.vtu`, time
  `14.8 s`, with both cell/point `alpha.water` and `U` fields.
- Existing model metrics classified geysering with a superseded 98%-of-rim
  level definition. The one full-field VTK frame showed real liquid in the
  exterior, but one frame could not provide first physical crossing or a
  cumulative volume ledger.
- Unified result: all 298 stored fields from `0` through `14.85 s` were sampled
  and the physical-rim gate passed at `14.0 s`.

## B-H3 completed baseline

- Source run: `/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq`.
- Existing VTK directory: `VTK_OUTLET_AUDIT/`, containing full-field VTU data
  at `13.2`, `13.25`, and `17.65 s` (the VTM series file lists only the first
  two, while the `17.65 s` VTU remains present separately).
- Existing whole-width audit:
  `cases/BH3_Dr26_H066_L061/openfoam/2d/complete_event/end20/results/physical_outlet_crossing_audit.md`.
  It decoded the `13.25` and `17.65 s` cell/point fields directly. At
  `17.65 s`, maximum cell alpha above the rim was `2.3335812784e-6`, maximum
  positive cell `alpha*Uz` was `3.9791260065e-7 m/s`, and no above-rim cell had
  `alpha > 1e-5`. Those frames supported a dilute-trace diagnosis but were too
  sparse for a complete-event cumulative volume.
- Unified result: all 401 stored fields from `0` through `20 s` give maximum
  rim-plane alpha `7.8864150055e-6`, cumulative positive model volume
  `9.0458089742e-13 m3`, and no resolved crossing. This is consistent with the
  earlier whole-width field audit and remains a diagnosed H3 baseline miss.

## B-H6 completed event

- Source run: `/tmp/bh6-2d-study/paper_tau0p2_areaeq`.
- No pre-existing full-field VTK/VTU outlet series was present.
- Existing `postProcessing` data included centreline, plume probes, global
  water volume, and field extrema. Existing `openfoam_2d_metrics.json` used a
  98%-of-rim free-surface test; it did not contain rim-plane Q+ or cumulative
  passed volume. Therefore the requested physical outlet metrics were missing
  from the original post-processing and were not inferred from figures.
- Unified result: all 401 stored fields from `0` through `20 s` give zero
  rim-plane alpha, Q+, and cumulative passed volume, hence no resolved
  crossing.

## B-H3 refined, active

- Source run: `/tmp/bh3-2d-qualification/h3_refined_iso_riser20`.
- At the recorded snapshot it had source checkpoints and ordinary runtime
  post-processing but no completed unified physical-rim surface series.
- Because the solver was active and had no normal `End`, it was not sampled or
  assigned a final outlet classification. Requested outlet metrics remain
  explicitly missing in `h3_refined_checkpoint_status.json`.
