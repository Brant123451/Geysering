# Case A two-channel vertical-mouth integration draft

## Scope

This is an implementation draft, not a change to
`vw2011_network_twofluid.py`.  The proposed closure contains no event time,
target liquid height, target flux, or rendering-dependent condition.  The 2-D
data below are used only for post-simulation acceptance.

## What the current run says

The latest available candidate at the time of this audit was
`vertical_fields_natural_tjunction_taylor_ccfl_v3_dx160_cfl1_9p2s.npz` with
its matching solver diagnostics.  Its post-breakthrough state is:

| physical time (s) | 1-D equivalent riser liquid height (m) | first-riser-cell liquid fraction | applied net mouth flow (L/s) | unconstrained characteristic flow (L/s) |
|---:|---:|---:|---:|---:|
| 8.50 | 0.08399 | 0.2783 | -0.2028 | -3.6379 |
| 8.85 | 0.07464 | approximately zero | approximately zero | -0.7357 |
| 9.20 | 0.06168 | approximately zero | -0.0020 | -0.3947 |

The independently extracted 2-D reference retains about 0.102--0.104 m of
equivalent riser liquid over the same audit window, and its first tower row is
about 0.79--0.99 liquid.  Thus the principal failure is not absence of an
algebraic gross-flow label: the single signed characteristic first drains the
mouth cell, after which donor positivity clips the applied net flow to nearly
zero.

An offline evaluation of the target-free two-channel closure on the *existing*
1-D states gives a natural counter-current capacity of approximately 0.20 L/s
at 8.50 s and 0.17 L/s at 8.60 s.  Once the first cell becomes dry, the local
gravity-film capacity is exactly zero and so is the circulation.  Therefore
adding `Q_c` after the existing `G1[0]` calculation cannot repair the run.  The
net-flux owner and the distributed vertical state must be corrected first.

## Required ownership

The finite T node must be the only liquid-flux owner.  For each Runge--Kutta
stage it supplies one signed `q_net`, positive from the horizontal node into
the riser.  The two-channel closure then constructs

```text
Q_up   = max(q_net, 0)  + Q_c
Q_down = max(-q_net, 0) + Q_c
Q_up - Q_down = q_net
```

where `Q_c` is limited only by the instantaneous gravity-film, Wallis, finite
node donor, and riser-film donor capacities.

When this owner is enabled, all of the following legacy updates are disabled:

1. The characteristic vertical flow may be an incident boundary trace for the
   finite-node solve, but it must not independently update `G1[0]`.
2. The Taylor front remains a geometric shock fit.  Its swept-liquid return
   must not replace or augment `G1[0]`; film geometry enters the constitutive
   state instead.
3. The post-breakthrough Wallis limiter must not be applied to the signed
   `q_net`.  Wallis already limits the counter-current part `Q_c` inside the
   two-channel closure.
4. With an **explicit** finite node, its west/east face fluxes already connect
   the horizontal cells and no horizontal side source is applied.  In a
   transitional run that retains the current distributed T cell, the
   gross-aware footprint update replaces `_apply_finite_width_side_t_exchange`;
   the old net-only side source is not called in the same stage.  These two
   horizontal ownership topologies are mutually exclusive.

The coupled gas-network T flux is a gas-phase owner and remains active.  It is
not a duplicate of the liquid exchange.

## Minimal source-level hook points

The following anchors refer to function/block names so the draft remains valid
if line numbers move.

1. After the finite-node SSP--RK stage has returned its time-averaged vertical
   `StratifiedFlux`, pass its liquid rate as `q_net` to
   `stage_twochannel_mouth_coupling`.
2. Replace the assignment of `G1[0]` from
   `junction_vertical_node_flow` with `plan.exchange.q_net` exactly once.
3. Do not execute `_countercurrent_flooding_liquid_flow` on that signed flux.
4. Keep `_advance_riser_taylor_front` and the topology reconstruction, but
   remove the block that overwrites `G1[0]` with
   `-material_return_flow`.
5. If the explicit finite-node west/east fluxes are active, remove the
   horizontal side source entirely.  For a transitional distributed-node
   trial only, replace `_apply_finite_width_side_t_exchange` with
   `apply_twochannel_horizontal_footprint`.  This update removes the gross
   upward parcel and its horizontal momentum, deposits the gross downward
   parcel with zero horizontal momentum, and changes volume by only
   `-q_net*dt`.  The adapter rejects this operation when the explicit-node
   topology is selected.
6. Apply the vertical total-volume residual `+q_net` once.  Apply the two gross
   convective momentum residuals only to a two-liquid-momentum riser state.
7. Record `q_net`, `Q_up`, `Q_down`, `Q_c`, all four capacities, the momentum
   excess, and the non-negative loss powers.  Comparison with 2-D follows the
   completed run and never feeds back into the model.

The adapter rejects a stage if any legacy path is marked as already applied,
so accidental double counting fails loudly.

## Is a second vertical liquid momentum required?

Yes, if “simultaneous upward and downward flow” is to be part of the numerical
solution rather than a diagnostic label.

A single riser discharge stores only

```text
q_net = A_up*u_up + A_down*u_down,
```

while the convective momentum flux is

```text
M2 = A_up*u_up^2 + A_down*u_down^2.
```

Many counter-current states have the same `q_net` but different `M2`, kinetic
energy, wall loss, and donor direction.  In particular, `q_net=0` can mean
either stagnant liquid (`M2=0`) or vigorous equal-and-opposite streams
(`M2>0`).  No limiter or plotting interpolation can recover that missing
degree of freedom.

The minimum credible vertical state is therefore either:

* total liquid area and momentum plus downward-film area and momentum; or
* explicit upward/core and downward/film areas and momenta.

If film area is imposed algebraically from annular geometry, one additional
film momentum is mathematically sufficient.  For transient retention of the
approximately 0.10 m liquid inventory, however, a transported film inventory
is preferable: otherwise the film can disappear and reappear without a local
mass history.  Interfacial drag, gravity, wall shear, and phase transfer must
exchange equal-and-opposite momentum between the two liquid streams.  Their
areas must sum to the existing `Alr`, and their signed discharges must sum to
the existing `Qlr`.

Using the gross second moment directly as `G2[0]` while retaining only one
`Qlr` would be an unresolved Reynolds-stress forcing.  It may visibly support
more water, but without the relative-momentum state its energy and direction
cannot be audited; this draft intentionally rejects that shortcut.

## Safe staged path

1. **Audit-only hook:** compute and record gross flows, but update the current
   one-momentum model with `q_net` only.  This checks capacities and ownership;
   it is not expected to reproduce the 2-D counter-current hold-up.
2. **Finite-node ownership:** make the conservative finite node the sole source
   of `q_net`; update its adjacent horizontal branches through the returned
   west/east face fluxes, with no separate side source.  This should remove the
   current characteristic/drain/positivity contradiction, but gross streams
   remain diagnostic.
3. **Two-liquid-momentum riser:** add the film inventory and momentum, use the
   gross boundary residuals, and evolve the two vertical liquid streams with
   conservative interfacial exchange.  Only this stage can be accepted as a
   physical reproduction of simultaneous upward/downward liquid.

Required acceptance checks are exact combined liquid conservation, exact
`Q_up-Q_down=q_net`, non-negative directional dissipation, no duplicated mouth
owner, bounded gas speed, and post-run comparison of equivalent riser liquid
height and net/gross mouth fluxes against the independent 2-D extraction.
