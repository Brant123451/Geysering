# S1 canonical trajectory observer/exporter

This package is an evidence layer, not a physical closure and not a result
acceptance tool. It observes immutable accepted `CoupledState` snapshots at
the exact unshifted Stage-2 grid `t=0,0.1,... s`, consumes every intervening
accepted `LedgerEntry`, and writes:

- `one_d_canonical_timeseries.csv`;
- `riser_twofluid_profiles.npz` with native independent
  `Aup/Qup/Adown/Qdown/Mg/Jg` information;
- `conservation_ledger.csv`; and
- `one_d_trajectory.metadata.json`.

`horizontal_slug_velocity_m_s` is the water-volume-weighted magnitude
`abs(Ql/Al)` in the published Figure-8 window while a bracketed full-bore
water slug intersects it. Gas-nose/front/edge velocity is forbidden as a
substitute. Published P1--P6 coordinates and the project aliases for P4--P6
are frozen in `../config/COMMON_OBSERVABLES.yaml`.

Formal mode calls `require_production_operator()` before building an artifact
and requires `operator.production_ready is True`. It also rejects any missing
canonical diagnostic. The synthetic role exists only for end-to-end contract
tests; unavailable values are blank/NaN and must be enumerated in metadata.
Neither role changes `production_ready` or writes `RESULT_ACCEPTED`,
`RUN_COMPLETE_UNVALIDATED`, or `ERUPTION_ACCEPTED`.

## Runtime interface still required

The current network runner returns the final state and ledgers but does not
yet expose a persistent accepted-state observer stream. A future production
driver must provide, without reconstructing rejected RK trials:

1. accepted states exactly at every common time, including the Stage-2
   opening checkpoint;
2. every accepted ledger entry between successive common samples;
3. native P1--P6 gauge pressures;
4. accepted supply/mouth gross phase fluxes;
5. accepted residuals and reaction impulse from both zero-storage T nodes;
   and
6. the native internal mouth-event state.

Until those interfaces exist and the physical operator itself is production
ready, this package cannot create a formal long-run artifact.
