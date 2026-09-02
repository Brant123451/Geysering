# Case-A finite-node/two-stream minimal main-loop patch

## Result to preserve

The requested late-time water entry is a counter-current state.  At the 2-D
mouth, gross upward and downward liquid rates are both large while their
difference is small.  The production state must therefore retain

\[
(A_\uparrow,Q_\uparrow,A_\downarrow,Q_\downarrow),\qquad
Q_{\rm net}=Q_\uparrow-Q_\downarrow.
\]

Adding an upward source to the legacy `Alr,Qlr` pair, or changing a rendered
polygon, cannot represent this state and is not an admissible patch.

## One owner at the T

After the port topology opens, the finite compressible node must be the sole
owner of all west, east, and vertical shared-face fluxes.  The following four
legacy operations in `vw2011_network_twofluid.py` must be removed from the
post-event branch in the same edit:

1. `G1[0] = junction_liquid_area * u_t_liquid` as a committed bottom flux;
2. `_countercurrent_flooding_liquid_flow(...)` applied to signed `q_net`;
3. `G1[0] = -material_return_flow` during the Taylor sweep; and
4. `_apply_finite_width_side_t_exchange(...)` as an additional horizontal
   side source.

Their characteristic, Taylor-film, and Wallis quantities may be retained as
inputs to the new node/mouth constitutive solve.  They may not move mass a
second time.

## Persistent state and topology event

Add a persistent `riser_twostream_state = None` beside `Alr,Qlr`.  On the
resolved topology event, not at a selected time, initialize it once:

```python
cell_bottom = zr - 0.5 * dz
swept_fraction = np.clip(
    (riser_material_front - cell_bottom) / dz,
    0.0,
    1.0,
)
mapping = map_taylor_breakthrough_to_twostream(
    Alr,
    Qlr,
    twostream_parameters,
    taylor_core_area_fraction=riser_gas_core_fraction,
    taylor_rise_velocity=material_front_velocity,
    swept_fraction=swept_fraction,
)
riser_twostream_state = mapping.state
```

The map preserves every cell's liquid area and net momentum.  It does not use
a requested hold-up, a 2-D frame, or elapsed time.

## Global stage order

The smallest conservative post-event stage is:

1. Build west, east, and vertical branch traces from the current global
   predictor state.
2. Evaluate the finite-node pressure and all six phase face flux components.
3. Form one `FiniteNodeQnetTransaction`; its vertical liquid-volume flux is
   the only `q_net` supplied to the mouth.
4. Call `stage_from_finite_node_ssprk2(...)` (or the matching global Euler
   residual) to obtain `Q_up,Q_down` without changing `q_net`.
5. Advance the persistent vertical two-stream state with those two bottom
   boundary fluxes.  The top boundary is atmospheric outflow only.
6. Apply the physical three-body gas/upward-liquid/downward-liquid drag and
   insert its equal-and-opposite impulse into `Jgrs`.
7. Commit every finite-node face component once to its adjacent horizontal or
   vertical branch residual.  Verify all keys from
   `required_commit_keys()` before accepting the stage.
8. Export only the exact totals for existing diagnostics:

   ```python
   Alr_new = np.asarray(riser_twostream_state.liquid_area)
   Qlr_new = np.asarray(riser_twostream_state.liquid_discharge)
   ```

9. Store all four directional fields for the next step; never reconstruct
   them later from `Alr,Qlr`.

The local `casea_compressible_node_ssprk2` currently freezes adjacent branch
traces between its two stages.  It is suitable for a component check but not
for the production network.  The main loop must either recompute all branch
traces at both global SSP-RK2 stages, or consistently use a globally
first-order Euler update under the strict node CFL.  Mixing a local RK2 node
with one-stage neighbouring branches is not conservative in time.

## Exact insertion anchors in the current main file

- State allocation: immediately after `Alr`, `Qlr`, `Mgr`, `Mgrs`, and `Jgrs`
  are created in `run_network`.
- Topology initialization: immediately after
  `material_front_reached_surface` becomes true and before the old Taylor
  projection is applied.
- Post-event residual: replace the single-liquid riser block beginning at
  `# ================= RISER update` through its `Alr_new,Qlr_new` transport
  and bottom side-source ownership.
- Gas reaction: replace the one-liquid
  `gas_advance.vertical_liquid_momentum_increment` insertion with the
  three-body impulses while retaining the conservative gas mass/tracer solve.
- State commit: at the existing assignment of
  `Alr,Qlr,Mgr,Mgrs,Jgrs = ...`, also persist the four directional liquid
  arrays.
- Diagnostics: add gross mouth rates, bottom 0.10-m inventory, both directional
  volume ledgers, gas/three-body momentum residual, and maximum resolved gas
  Mach number.

## Acceptance gate before rendering

Do not create a new HTML until a raw 9.2-s run satisfies all of the following:

- no legacy mouth owner is active after the topology event;
- finite-node gas and liquid inventories close at every accepted stage;
- upward-stream and downward-stream volume ledgers close independently;
- `Q_up - Q_down` equals the finite-node vertical `q_net` to roundoff;
- the gas/liquid drag impulse is equal and opposite;
- no donor limiter, geometry cap, density floor, or topology transfer is
  persistently active as a hidden boundary condition;
- bottom 0.10-m inventory, whole-riser inventory, gross-up, gross-down, net
  flow, and gas Mach are written and compared with the independent 2-D audit.

The independent raw 2-D control-volume audit gives the following validation
bands over 8.5--9.2 s.  They are output-only acceptance quantities and must
never be passed to the solver as controls:

- gross upward liquid rate, 5th--95th percentile: 0.0412--0.1140 L/s;
- gross downward magnitude, 5th--95th percentile: 0.0595--0.1216 L/s;
- signed net liquid rate: -0.0490--0.0314 L/s;
- liquid inventory in the bottom 0.10 m: 0.2047--0.2217 L;
- mean bottom liquid fraction: 0.799--0.866; and
- whole-riser equivalent liquid height: 0.10215--0.10428 m.

The key discriminator is simultaneous gross flow: a one-momentum state with
the same small net rate is not equivalent to this counter-current motion.

The executable fail-closed screen is:

```powershell
$env:PYTHONPATH='E:\Geysering\.codex_deps;E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\model'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\scripts\caseA_twostream_network_preflight.py' `
  --target-time 9.2 --expect-blocked
```

It currently stops after the exact port event/launch and writes a blocker
report instead of fabricating a continuation.  Once the atomic main-loop patch
is installed, `COMPLETE_CASEA_NETWORK_READY` may be set true only after the
raw-field acceptance gate above passes.

After that preflight reports `ready_for_9p2s_run`, the raw 9.2-s screening run
and independent acceptance audit are:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\scripts\caseA_make_frame_viewer.py' `
  --variant twostream_tnode_dx40_dz20_9p2s `
  --ds 0.04 --dz 0.02 --t-end 9.2 --phase-volume-cfl 0.25 `
  --output-interval 0.05 --no-render

& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\scripts\caseA_accept_1d_tjunction_against_2d.py' `
  'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\outputs\vertical_fields_twostream_tnode_dx40_dz20_9p2s.npz' `
  'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\outputs\solver_diagnostics_twostream_tnode_dx40_dz20_9p2s.json' `
  --output 'E:\Geysering\tests\test_01_vw2011\cases\A_Dt57p1_Ha0305_Yfs0356\outputs\acceptance_twostream_tnode_dx40_dz20_9p2s.json' `
  --strict
```

Rendering is deliberately excluded from this first run.  A failed raw-field
audit is not converted into an HTML candidate.
