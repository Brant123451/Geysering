# Case-A post-arrival T-junction integration contract

This note is a numerical-model integration contract.  It is not manuscript
evidence and it does not change the production loop.

## Review result

`casea_tjunction_shock_network.py` fixes the missing graph topology: after the
west gas front reaches the side T, the east and vertical fronts are independent
and share one node pressure.  It must **not** be inserted as another source term
into the current post-arrival loop.  The current loop already advances the same
liquid and gas inventories through:

- `external_horizontal_solver.step` and
  `apply_junction_liquid_fluxes`;
- the finite-volume horizontal and vertical liquid equations;
- `advance_lumped_pocket_vertical_network`;
- the vertical atmospheric gas boundary.

Running the graph advance in addition to those operators would give two owners
for gas mass, gas volume, vertical-front motion, and atmospheric outflow.  A
conservative result can have only one owner for each quantity.

The direct distributed handoff at about 6.5 s is also not an admissible route.
It changes the pressure law and numerical operator on an already evolved field,
and maps one lumped gas pressure into cellwise gas states.  The resulting
odd/even mode is an operator-transition error, not a physical wave.

## Required topology event

1. Let `x_T` be `external_horizontal_solver.junction_face_x`, not the centre of
   a nearby cell.
2. If a trial pre-arrival step crosses `x_T`, subdivide that step and locate the
   event so that `interface_x = x_T` to the nonlinear tolerance.
3. At the event, preserve the shock-fit `area` and `discharge` arrays bit for
   bit.  Do not remap them to the older distributed Rusanov state.
4. Stop advancing the old single `interface_x`.  It is a pre-arrival topology
   variable and must never move into the east dead leg.
5. Create an east branch front at distance zero from the T and a vertical front
   at distance zero from the T.  Their characteristic feet come from the first
   complete liquid states in their own branches.

The transition itself must change no liquid volume, gas mass, gas volume,
pressure, or momentum.  Only the topology metadata changes.

## State ownership

The production implementation must choose one of the following, not mix them.

### Reduced graph path (exploratory only)

The shock network owns the complete connected gas mass, all four gas volumes,
both moving fronts, and the atmospheric flux.  In that case the current lumped
pocket--vertical gas advance must be disabled.  This path does not retain the
resolved vertical two-fluid gas field and therefore is not yet the requested
production model.

### Resolved vertical two-fluid path (production target)

Refactor the graph core into a pure nonlinear node/branch-boundary solver.  The
resolved finite-volume fields remain the state owners:

- west horizontal liquid field: same MUSCL operator as before arrival;
- east liquid field plus one east fitted front;
- vertical liquid and gas fields: the existing resolved two-fluid operator;
- connected gas mass and atmospheric exchange: one gas-network ledger only.

The pure node solve may predict volumes during its pressure iteration, but it
must not also commit `gas_mass`, `west_gas_volume`, `vertical_front`, or top
mass transfer after the finite-volume operators commit those same quantities.
The accepted geometric volumes are recomputed from the accepted conserved
fields and checked against the predictor.

## Branch-flow signs and face replacement

The shock-network convention is that `q_w`, `q_e`, and `q_v` are positive from
the tee into the west, east, and vertical branches.  In the main loop's global
coordinates the corresponding liquid volume fluxes are

```text
west horizontal face (positive east):  F_w = -q_w
east horizontal face (positive east):  F_e = +q_e
vertical bottom face (positive up):     G_v = +q_v
```

For a zero-storage node, `q_w + q_e + q_v = 0`, equivalently
`F_w - F_e = q_v`.  With an explicit finite tee control volume,

```text
dV_g,tee/dt = q_w + q_e + q_v
dV_l,tee/dt = -(q_w + q_e + q_v).
```

The tee liquid storage must then be included in the apparatus liquid ledger.
Case A currently contains no sourced value for extra tee volume.  An arbitrary
`tee_total_volume` must not be introduced into a production run.  Until the
fitting geometry is sourced, use the zero-storage node limit.

`apply_junction_liquid_fluxes` expects horizontal fluxes positive east.  If it
is retained as a temporary adapter, its arguments would be

```python
west_flow = -q_w
east_flow = +q_e
```

not `mean_flow +/- q_v/2`.  However, replacing volume flux alone is not a
complete hyperbolic boundary condition.  The graph API must also return the
liquid momentum/pressure flux for each branch.  The provisional T-face area
**and momentum** fluxes must both be replaced.  An area-only correction of the
two adjacent cells can launch a grid-local momentum impulse even when volume is
exactly conserved.

After those face replacements, do not also call
`_apply_finite_width_side_t_exchange` for the same `q_v`: that would apply the
same internal transfer twice.

## Gas and volume ledger

At topology creation, define disjoint volumes:

- `V_w`: horizontal void strictly west of the T face;
- `V_e`: east-front gas volume strictly east of the T face;
- `V_v`: resolved vertical void below the current liquid surface;
- `V_T`: only fitting volume not already represented by the horizontal or
  vertical control volumes.

Do not include the same horizontal T cell in both `V_w` and `V_T`.  Do not use
the dry atmospheric headspace above the liquid surface as trapped-gas volume.
`branch_gas_masses` returned by the graph core are a diagnostic partition of
one EOS mass; they are not additional cell masses.

When creating the graph gas ledger, count each material inventory once.  In
particular, do not add both the lumped horizontal inventory and its mapped
`Mgt` display field.  If the graph owns atmospheric transfer, do not also add
`gas_advance.atmospheric_mass_exchange` from the old gas operator.

## Vertical opening event

The vertical gas becomes atmospheric when its material front reaches the
current resolved liquid surface `wtop`, not only at the 0.610 m riser rim.
`advance_tjunction_shock_network` now accepts
`vertical_liquid_surface_height=wtop`.  A step that crosses that moving surface
raises `StepSubdivisionRequired`; locate the crossing and then apply the top
Riemann flux.  Opening no longer fills the dry headspace to the rim in one step.

## Mandatory tests before a 9.5 s run

1. **Event identity:** pre/post topology transition has machine-precision
   equality of liquid volume, gas mass, gas volume, pressure, and momentum.
2. **No Nyquist injection:** a smooth event state has no growth of the
   `2*dx` odd/even amplitude in either `A_l` or `Q_l` after one coupled step.
3. **Exact face signs:** arbitrary signed `q_w/q_e/q_v` changes each branch by
   the same signed volume used in the node ledger.
4. **Tee storage:** zero-storage gives `q_w+q_e+q_v=0`; finite storage, if later
   sourced, conserves liquid only when explicit tee liquid storage is included.
5. **Momentum equilibrium:** a hydrostatic equal-pressure state remains at
   rest for one step; no adjacent-cell discharge impulse is generated.
6. **Single gas owner:** connected gas mass plus cumulative atmospheric mass is
   invariant to roundoff in a closed test and changes only by the one top flux
   in an open test.
7. **No legacy front:** after the event, the old straight-pipe `interface_x`
   cannot advance beyond `x_T`, and no mapper may infer east gas from
   `x < interface_x`.
8. **East branch freedom:** the east front advances, stalls, and recedes under
   signed pressure perturbations (already covered by the independent core).
9. **Moving-surface opening:** a front below 0.610 m opens at the supplied
   current liquid surface, and crossing requests event subdivision (covered by
   the independent core tests).
10. **Grid refinement:** the 6.5--7.1 s node response must converge under
    `dx, dz` refinement and the odd/even metric must decrease, not merely move
    to a smaller wavelength.

Only after these tests pass should a 9.5 s raw-field run be audited.  HTML or
rendering changes cannot validate this integration.
