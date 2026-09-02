# Case-A vertical two-stream finite-volume core

## Purpose and state ownership

`casea_vertical_twostream_fv.py` is an isolated riser operator for simultaneous
upward liquid in a core/water tongue and downward liquid in a wall-film
continuum. It owns four cell-centred conserved fields:

\[
U_r=(A_\uparrow,Q_\uparrow,A_\downarrow,Q_\downarrow),\qquad
Q_\uparrow\ge 0,\quad Q_\downarrow\le 0.
\]

The mapping to the current one-liquid fields is exact:

\[
A_{lr}=A_\uparrow+A_\downarrow,\qquad
Q_{lr}=Q_\uparrow+Q_\downarrow
      =Q_{\rm gross,up}-Q_{\rm gross,down}.
\]

The split state must persist between time steps. Reconstructing it from only
`Alr` and `Qlr` would discard the counter-current degree of freedom.

## Intended integration interface

1. Before the counter-current topology starts, use
   `VerticalTwoStreamState.from_legacy_single_stream(Alr, Qlr)`. At the
   Taylor breakthrough event, call `map_taylor_breakthrough_to_twostream(...)`
   with the current `Alr`, `Qlr`, Taylor-core area fraction, Taylor rise
   velocity, and swept-cell fractions. The film corridor follows the current
   Taylor geometry and the Davies--Taylor displacement balance; every cell's
   area and momentum remain unchanged by the map.
2. The finite T-node Riemann solve supplies gross bottom-face rates and donor
   speeds through `DirectionalBoundaryFlux`. One object carries both streams,
   so its `net_rate` is exactly `Q_up - Q_down`.
3. The gas/pressure solve supplies one common pressure at every riser face.
   Gas mass, mass-based momentum, gas area, interface perimeters, and hydraulic
   diameters are passed as `PhysicalGasInterphaseState`.
4. Call `advance_vertical_two_stream_fv(...)`. Use its single shared face flux
   at both neighbouring control volumes. If SSP--RK2 is used globally, repeat
   the T-node, pressure, and gas closures at both RK stages.
5. Store all four returned fields for the next step. Export
   `result.state.liquid_area` to the current `Alr` diagnostics and
   `result.state.liquid_discharge` to the current net-flow diagnostics.
6. After the transport/body-force stage, call
   `implicit_physical_three_body_drag_exchange(...)`. It solves gas, upward
   liquid, and downward liquid velocities simultaneously. The Fanning factor
   uses the same Reynolds-number law as the existing coupled gas network, and
   its coefficient is computed from gas density, interface perimeter, and
   hydraulic diameter. The three returned physical impulses sum to zero in
   each cell.
7. A stopped or reversed labelled stream is passed through
   `conservative_directional_topology_transfer(...)`. This transfers its area
   and momentum to the physically matching channel; it does not clamp velocity
   or discard momentum. The FV step invokes this transfer automatically.
8. Check both area residuals and the momentum residual in the returned ledger.

Once this operator owns the riser mouth, the legacy characteristic bottom
liquid flux, Taylor-sweep return flux, post-breakthrough CCFL mass limiter, and
any extra horizontal side-source flux must not update the same liquid again.
They may contribute constitutive information to the new T-node solve, but a
second inventory update would double count mass.

## Included physics and numerical guarantees

- Conservative donor-cell area and convective-momentum transport.
- A local draining-time limiter: a cell cannot donate more liquid than it
  contains, and the same limited face flux is used on both sides.
- Axial gravity and a supplied common-pressure gradient. A discrete
  hydrostatic pressure profile cancels gravity to roundoff.
- Darcy wall friction for each stream.
- Quadratic liquid--liquid momentum exchange integrated at fixed area; its two
  impulses are exactly equal and opposite.
- Physical gas--liquid drag based on gas inventory, Reynolds number, interface
  perimeter, and hydraulic diameter. The three-body implicit solve strictly
  conserves gas-plus-two-liquid momentum.
- Conservative topology transfer at zero velocity or reversal, including its
  area, momentum, and kinetic-energy ledger.
- Explicit rejection of cross-section over-packing; it is never hidden by
  clipping or prescribed animation data.

## Physical closures still required

This module is a tested core, not yet a complete Case-A riser model. The
following closures must be selected from physics or independent measurements
before production integration:

- the post-breakthrough evolution law for the core/film area partition (the
  event mapping itself is implemented);
- hydraulic diameters and wall wetted perimeters for the core and film;
- the liquid--liquid interfacial-drag coefficient;
- fully coupled gas void and gas-pressure evolution at both Runge--Kutta stages
  (the physical three-body drag source itself is implemented);
- the finite T-junction two-stream Riemann problem and its horizontal branch
  coupling;
- the upper free-surface/vent boundary and liquid-exit condition.

No closure in this module depends on elapsed time, a requested water depth, a
2-D frame, or a plotted curve. Therefore it cannot by itself guarantee the
amount of Case-A re-entry seen in 2-D; that quantity must emerge after the
missing junction, gas, and area-partition closures are supplied and validated.

## Minimum executable coupling sequence

The independent core can be connected without changing its API using the
following per-stage sequence:

```python
if taylor_breakthrough_event:
    two_stream = map_taylor_breakthrough_to_twostream(
        Alr, Qlr, params,
        taylor_core_area_fraction=current_core_fraction,
        taylor_rise_velocity=current_taylor_speed,
        swept_fraction=current_swept_fraction,
    ).state

mouth = solve_finite_t_node_gross_fluxes(...)  # Q_up and Q_down separately
fv = advance_vertical_two_stream_fv(
    two_stream,
    params,
    dt=stage_dt,
    pressure_faces=current_vertical_pressure_faces,
    boundaries=mouth,
)
drag = implicit_physical_three_body_drag_exchange(
    fv.state,
    params,
    current_physical_gas_trace,
    dt=stage_dt,
)
two_stream = drag.state
vertical_gas_momentum[:] = drag.gas_momentum
Alr[:] = two_stream.liquid_area
Qlr[:] = two_stream.liquid_discharge
```

This sequence is not yet a 9.2-s Case-A runner. The present main solver stores
only one riser liquid area and one riser liquid momentum, so it cannot retain
the two gross streams between stages. A valid screening run first needs a
sidecar/state adapter that persists all four fields and makes the two-stream
T-node the sole owner of the mouth flux. Reconstructing the split from `Alr`
and `Qlr` after every step would reduce the model back to one liquid velocity.
