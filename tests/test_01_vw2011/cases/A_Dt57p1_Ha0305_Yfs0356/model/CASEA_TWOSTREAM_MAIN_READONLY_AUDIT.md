# Case-A persistent two-stream main-loop integration audit

## Scope and audited source

This is a read-only integration audit.  It does not modify
`vw2011_network_twofluid.py` and it is not evidence that the production Case-A
network is ready for a new run.

The principal file audited here was
`vw2011_network_twofluid.py`, SHA-256
`92E811E2FB54734F51930B9AFB2BA57769E03366DE6328400B4405728F14720A`.
Line numbers below refer to that snapshot; the quoted anchor text is the
stable locator if later edits move a block.

The intended post-breakthrough state is persistent:

\[
U_r=(A_\uparrow,Q_\uparrow,A_\downarrow,Q_\downarrow),\qquad
A_{lr}=A_\uparrow+A_\downarrow,\quad
Q_{lr}=Q_\uparrow+Q_\downarrow .
\]

`Alr,Qlr` may remain compatibility views after the event, but they must never
again be the authoritative state from which the two directional streams are
reconstructed.  Reconstruction from the signed net discharge would erase the
counter-current state precisely when `Q_up` and `Q_down` nearly cancel.

## Audit conclusion

There is no safe one-line insertion.  The smallest conservative change is one
atomic ownership replacement spanning the Taylor event, the T-node boundary,
the riser residual, the gas-drag reaction, and the state commit.  A hybrid that
adds the new closure while retaining any old mass/momentum owner will either
move the same liquid twice or erase the new directional momentum.

Two horizontal-node topologies exist and are mutually exclusive:

1. **Explicit finite node.**  The node owns west, east, and vertical face
   fluxes.  No distributed side source, net or gross, may touch the horizontal
   cells.  Horizontal response is committed through the west/east shared-face
   fluxes.
2. **Distributed footprint.**  The gross-aware footprint removes `Q_up` and
   its local axial parcel momentum and deposits `Q_down` with zero axial
   momentum.  No explicit node inventory or west/east finite-node face commit
   may then be applied.

`stage_from_finite_node_ssprk2` selects the first topology, while
`apply_twochannel_horizontal_footprint` explicitly requires the second and
rejects the first.  Calling both is not a more complete model; it is a double
update.

The explicit-node route is the physically preferable final architecture, but
the current isolated node has no liquid-momentum inventory or gross mixing
residual.  Therefore the gross horizontal momentum reaction is still an
integration blocker, not something that can be supplied by also calling the
distributed footprint.

## Required variable lifetimes

| Variable | Creation | Lifetime and owner | Forbidden reconstruction/reset |
|---|---|---|---|
| `finite_node_state` | Exact west-front contact with the measured finite T volume | Persistent from the node-launch topology through the end of the run; advanced once per accepted global stage | Never re-created from `Alt[jx]`, `P_connected_j`, or a gassy/not-gassy threshold |
| `riser_twostream_state` | Rising edge of the Taylor-front/free-surface intersection | Persistent and authoritative after breakthrough | Never call `from_legacy_single_stream(Alr,Qlr)` on later steps; it destroys simultaneous gross flow |
| `twostream_active` | Set once when the conservative Taylor map succeeds | Latched topology flag; a later collapse requires a separate conservative topology event | Do not tie it to instantaneous `Mgrs` threshold or `material_front_tracked` |
| `mouth_plan` | Recomputed from current predictor traces at every accepted global RK/Euler stage | Ephemeral; contains `Q_up,Q_down` and the unchanged node-owned `q_net` | Never reuse a previous-step plan or fit it to a time/height |
| `gas_drag_result` | After the gas transport and two-stream liquid transport of the same stage | Ephemeral; its liquid state and returned gas momentum are committed together | Never apply only the liquid impulse or only the gas impulse |
| `Alr,Qlr` after breakthrough | Derived from `riser_twostream_state.liquid_area` and `.liquid_discharge` | Read-only compatibility views for pressure, gas geometry, output, and old diagnostics | No clipping, projection, diffusion, friction, or boundary overwrite may be applied to these totals and then treated as authoritative |
| `G1,G2` after breakthrough | At most diagnostic compatibility arrays | No longer own the riser bottom or interior liquid update | `G1[0]`, `G2[0]`, and `G1[-1]` cannot update mass, momentum, or balances |

The Taylor rise velocity used by the event map is also event data.  It must be
captured on the crossing step because the current loop resets
`material_front_velocity = 0.0` at line 4603 on every step.

## Atomic insertion anchors

### 1. Imports and persistent allocation

- Import anchor: after the coupled-gas imports at lines 42--47.
- Riser allocation anchor: immediately after `Alr`, `Qlr`, `Mgr`, `Mgrs`, and
  `Jgrs` at lines 3108--3116.
- Event-state anchor: beside `riser_material_front` and
  `riser_breakthrough` at lines 3147--3151.

Allocate the two-stream parameters once from the measured riser geometry and
set `riser_twostream_state = None`.  Do not silently use the parameter class's
zero-friction defaults as the production closure: the existing cell-wise
laminar/turbulent film stress does not map automatically to its constant
`wall_friction_up/down` fields.

An explicit finite node additionally needs a persistent state and an explicit
topology enum such as `SHOCK_FIT`, `NODE_WEST_LAUNCH`, and `NODE_POST_LAUNCH`.
The current Boolean `junction_topology_opened` at lines 3069--3073 cannot
distinguish the exact-zero-gas launch from a post-launch three-branch gas node.

### 2. Exact node launch and unique `q_net`

Current node ownership is spread over several places:

- lines 3915--3960 choose a zero-storage three-branch pressure or set the node
  pressure to the connected gas EOS;
- lines 3961--4045 compute a signed vertical characteristic;
- lines 4049--4061 clip that signed flow with the old CCFL routine;
- lines 4487--4488 commit it into `G1[0],G2[0]`;
- lines 4506--4585 modify it again with vertical and horizontal donor/void
  limiters;
- lines 4671--4683 may replace it with Taylor return; and
- lines 4794--4833 read the final value and apply a horizontal side source.

After the finite-node topology opens, all of those are mutually exclusive
with a node transaction.  The only admissible signed flux is

```text
q_net = finite_node_transaction.outward["vertical"].liquid_volume
```

and the same transaction must commit every west/east/vertical gas and liquid
face component to its adjacent branch residual.  `verify_atomic_branch_commit`
provides the twelve-key audit for this purpose.

The current local node SSP-RK2 helper is not a direct production drop-in:
`casea_compressible_node_ssprk2.PRODUCTION_READY` is false because its second
stage recomputes node pressure but freezes adjacent branch traces.  The main
loop must either recompute node and all three branch traces at both global
SSP-RK2 stages, or use one globally consistent Forward-Euler stage under the
strict node CFL.  A local RK2 node coupled to one-stage neighbours is not
temporally conservative.

The corrected mouth closure preserves this `q_net` and limits **total**
downward gross flow.  If it raises `NetFluxExceedsDownwardCapacity`, neither a
post-hoc clip nor a smaller `dt` solves the physical incompatibility.  The node
pressure/flux problem must be re-solved with the active downward-capacity
inequality as a complementarity condition.

### 3. Taylor breakthrough map

The geometric crossing is currently created by
`_advance_riser_taylor_front` at lines 4613--4632.  On the crossing step,
`material_front_reached_surface` becomes true, but `riser_breakthrough` is not
latched until lines 5190--5191.  The old topology projection at lines
5135--5188 therefore still runs on that same step.

The one-time two-stream map belongs immediately before the old projection
anchor (`if material_front_tracked ...` at line 5135), after the pre-event
single-stream transport and physical source updates have produced
`Alr_new,Qlr_new`.  The event predicate must be a rising edge:

```python
crossed_this_step = (
    riser_twostream_state is None
    and material_front_reached_surface
)
```

Use the actual axial cut geometry, not an all-cell switch:

```python
cell_bottom = zr - 0.5 * dz
swept_fraction = np.clip(
    (riser_material_front - cell_bottom) / dz,
    0.0,
    1.0,
)
mapping = map_taylor_breakthrough_to_twostream(
    Alr_new,
    Qlr_new,
    twostream_parameters,
    taylor_core_area_fraction=riser_gas_core_fraction,
    taylor_rise_velocity=event_taylor_velocity,
    swept_fraction=swept_fraction,
)
riser_twostream_state = mapping.state
```

The map preserves every cell's area and net momentum.  Once it succeeds, skip
`_project_riser_taylor_topology` and its gas-momentum overwrite.  A strictly
second-order implementation must split the step at the crossing time and run
the remainder with the post-event operator.  A conservative first-order
minimum may finish the crossing step with the pre-event operator, map once at
its end, and start two-stream evolution on the next step; it must disclose the
one-step event timing error.

### 4. Post-event riser residual

The single-stream block beginning at the anchor
`# ================= RISER update (Rusanov) =================` (line 4430)
through the `Alr_new,Qlr_new` transport at lines 4593--4594 must become a
pre-event-only branch.

For the post-event branch:

1. Build the current `Alr,Qlr` views from the persistent directional state.
2. Build the finite-node transaction from current-stage west/east/vertical
   traces.
3. Pass its vertical `q_net` to the corrected two-channel closure.
4. Use the returned gross rates as the two bottom boundary fluxes.
5. Advance `VerticalTwoStreamState` once with the common pressure faces,
   gravity, per-stream wall closure, interstream exchange, and the atmospheric
   top boundary.
6. Preserve the resulting four arrays for the next step.

The mouth closure is not a second mass source.  It only decomposes one signed
node flux.  Its `DirectionalMouthLosses` fields are currently an energy ledger,
not an applied pressure/momentum residual.  A production node solve must apply
the directional loss exactly once; merely recording the ledger does not close
the momentum problem.

### 5. Gross horizontal exchange

The current line-4825 call to `_apply_finite_width_side_t_exchange` knows only
`q_net`.  At `q_net = 0` it makes no horizontal momentum change even if both
gross streams are large.  `apply_twochannel_horizontal_footprint` fixes that
for the distributed-footprint topology by removing the horizontal parcel
momentum carried by `Q_up` and depositing `Q_down` normal to the axis.

It may replace lines 4820--4837 only if the integration deliberately retains
`HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT`.  In that case:

- apply it once after the ordinary horizontal west/east FV fluxes;
- do not also advance an explicit finite-node inventory;
- do not commit finite-node west/east face fluxes; and
- use its exact `-q_net dt` volume ledger as the sole horizontal T update.

For `HorizontalNodeTopology.EXPLICIT_FINITE_NODE`, lines 4820--4837 must be
disabled completely.  The finite node's west/east face fluxes are then the
only horizontal update.  Calling the gross footprint to “show more water”
would double-count the node/riser exchange.  The unresolved gross mixing
reaction must instead be represented in the node's own momentum/energy
closure; the present finite-node inventory model does not yet contain that
state.

### 6. Gas transport and equal-and-opposite momentum reaction

The gas mass/tracer solve at lines 4902--5030 can be retained using the exact
total liquid views from the directional state.  Its existing vertical
single-liquid drag must not also run.

The atomic post-event sequence is:

1. Advance gas transport with vertical interphase drag disabled.  In the
   present gas module that means using the existing
   `vertical_confined_interface_kinematics` bypass for the vertical branch;
   horizontal gas/liquid drag remains active.
2. Advance the two-stream liquid transport without its optional frozen-gas
   `GasMomentumCoupling` source.
3. Construct `PhysicalGasInterphaseState` from the transported gas mass,
   momentum, void area, and the resolved core/film interface geometry.
4. Call `implicit_physical_three_body_drag_exchange` once.
5. Commit both `drag_result.state` and `drag_result.gas_momentum` in the same
   accepted stage.

Consequently keep line 5031, the horizontal
`horizontal_liquid_momentum_increment`, but disable line 5032,
`Qlr_new += gas_advance.vertical_liquid_momentum_increment`.  Applying that
increment and then the three-body exchange would count vertical gas/liquid
drag twice.  Applying only the two liquid impulses without replacing/adjusting
`Jgrs_new` would create mixture momentum.

The Taylor projection's assignment
`Jgrs_new[swept_core] = Mgr_new[swept_core] * material_front_velocity` at
lines 5181--5188 must also be disabled after the event; otherwise it overwrites
the conservative gas reaction just committed.

### 7. State commit and diagnostics

At the current commit anchor, lines 5320--5322, store the four directional
arrays first, then derive `Alr,Qlr`.  Do not persist only the totals.

Replace all post-event top-flow accounting based on `G1[-1]` (lines
5272--5274 and 5285--5290) by the actual two-stream top boundary flux.  For the
Case-A atmospheric top, upward liquid may leave and no downward liquid
reservoir is imposed.  Record both gross bottom rates, their difference, both
directional volume ledgers, the three-body momentum residual, and topology
transfer activity.

## Legacy post-event paths that must be disabled

| Current anchor | Current operation | Why it cannot coexist with the new state |
|---|---|---|
| 4049--4061 | `_countercurrent_flooding_liquid_flow` clips signed `q_net` | Corrected Wallis/Nusselt closure limits total `Q_down`; signed clipping changes the node solution and applies CCFL twice |
| 4430--4594 | Legacy `G1,G2` single-liquid transport and bottom boundary | Second mass and momentum owner for the same riser |
| 4506--4585 | Legacy vertical/horizontal donor and void limiters mutate `G1[0]` | The finite node and two-channel donor inequalities must be solved before one shared face commit; post-clipping breaks atomic conservation |
| 4640--4683 | Taylor return replaces `G1[0],G2[0]` | After event it is not a boundary owner; before event it remains part of the pre-event topology only |
| 4820--4837 | Net-only finite-width horizontal side source | Double-counts an explicit node; in distributed mode it must be replaced, not supplemented, by the gross-aware operator |
| 4844--4860 | External shock-fit west/east correction from `q_up` | Must end at the exact node-launch handoff; cannot run alongside finite-node west/east commits |
| 5032 | Old single-liquid vertical gas-drag reaction | Double-counts the new three-body exchange |
| 5095--5103 | `_riser_liquid_friction_rate(Alr_new,Qlr_new,...)` | Acts on the collapsed net momentum and erases directional wall stresses; use per-stream friction once inside the two-stream operator |
| 5122--5130 | Vertical Smagorinsky diffusion on total `Qlr_new` | Mixes the two directional momenta after their conservative ledgers and can erase counter-current motion; any diffusion must be formulated per shared stream face |
| 5135--5188 | Repeated Taylor topology projection and gas drift overwrite | Recreates the area split every step and overwrites both persistent stream momentum and gas reaction |
| 5197--5204 | `_project_single_liquid_column` | Instantaneous tracer thresholds cannot collapse a persistent two-stream topology; use a separate conservative topology event if a true collapse occurs |
| 5208--5219 | Near-dry regularization and open-top zeroing of total `Qlr_new` | Alters totals without updating the two directional owners; positivity and top outflow belong inside their FV update |
| 5233 | `Alr_new = maximum(Alr_new,0)` as a state repair | A derived view must not be clipped independently; reject an inadmissible directional state instead |
| 5272--5290 | Balance/ejection from `G1[-1]` | `G1` is stale after ownership replacement; use the two-stream top flux ledger |

The horizontal friction and horizontal Smagorinsky blocks may remain because
they act on the horizontal branch, provided its selected node topology is
updated exactly once.  The gas mass/tracer FV solve may also remain.  The
riser acoustic bulk-viscosity pressure modification at lines 3690--3695 is a
single-continuum closure; it should be disabled or reformulated after
breakthrough rather than silently applied to both directional streams as a
common physical pressure.

## Minimal accepted-stage skeleton

```text
build all branch predictor traces
if pre-Taylor-breakthrough:
    advance existing pre-event Taylor topology only
    if material front crosses free surface:
        conservatively map totals -> persistent directional state once
else:
    solve sole T-node transaction, including active mouth complementarity
    decompose node-owned q_net -> Q_up,Q_down without changing q_net
    advance directional riser with those gross bottom fluxes
    advance gas transport with old vertical drag disabled
    exchange gas/up/down momentum once and commit all three reactions
    commit node west/east/vertical face fluxes exactly once
    derive Alr,Qlr compatibility views
validate every liquid, gas, and momentum ledger
accept the stage and persist node + four directional riser arrays together
```

If any node face, mouth inequality, stream packing bound, topology transfer,
or drag ledger is inadmissible, reject the whole stage.  Do not keep the gas
update while reverting the liquid update, and do not fall back to the old
`G1[0]` or footprint path.

## Blocking items before a physical 9.2-s run

1. The finite-node pressure solve must incorporate the corrected total-downward
   capacity as a complementarity condition; the mouth closure now correctly
   fails closed when a supplied negative `q_net` is incompatible.
2. Adjacent west/east/vertical traces must be recomputed at the same global
   RK stages as the node, or the complete coupled update must be consistently
   first-order.
3. The explicit finite node needs a physical gross horizontal liquid
   momentum/mixing reaction.  The current mouth loss fields are diagnostics,
   and adding the distributed footprint is forbidden in that topology.
4. The post-event stream-area evolution and per-stream wall/interstream
   closures remain declared missing in `casea_vertical_twostream_fv.py`.
5. Core/film interface perimeter and hydraulic diameter must come from the
   actual directional geometry before the three-body drag can be applied.
6. The exact top/free-surface boundary must be verified for both directional
   streams; total-`Qlr` open-top zeroing is not admissible.

Until these are closed, a run that merely imports `VerticalTwoStreamState`
would be an exploratory hybrid, not the requested natural simulation.  No
paper figure or HTML should be regenerated from such a hybrid.
