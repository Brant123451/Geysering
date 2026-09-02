# Case-A vertical two-stream physical closures

`casea_vertical_twostream_closures.py` is an independent, main-loop-callable
closure layer around `casea_vertical_twostream_fv.py`. It does not read a
clock, a 2-D field, a requested water inventory, or a rendered frame.

## Event timing and persistent state

The persistent two-stream state starts when tunnel gas first becomes connected
to the vertical branch through the T mouth. It must **not** wait until the
Taylor core reaches the upper free surface: the upward core and the falling
film already coexist while the gas core rises through the liquid column.

At first T connectivity, embed the current legacy liquid state once with
`VerticalTwoStreamState.from_legacy_single_stream`. For every subsequent
global RK stage:

1. advance the material Taylor nose from the current kinematics;
2. compute the old and new geometric swept fractions in each riser cell;
3. call `advance_taylor_sweep_geometry`;
4. call `advance_post_event_core_film_stage`; and
5. retain all four returned fields for the next stage.

`advance_taylor_sweep_geometry` first calls
`extend_taylor_sweep_in_persistent_state`, which modifies only cells whose swept
fraction increased. For a newly swept fraction `dS`, it transfers no more than

```text
dA_f = dS (1 - alpha_core) A_r
```

from the current upward inventory into the wall-film inventory. The added film
receives the Davies--Taylor return velocity; the equal and opposite discharge
remains in the core. Cell liquid area and total axial momentum therefore close
to roundoff. Previously swept cells are copied exactly, and repeating the same
sweep is idempotent. The algorithm never reconstructs those cells from
`Alr,Qlr`, because doing so would erase the gross counter-current degree of
freedom.

The same combined call then opens the newly swept gas core. It removes

```text
dA_g = dS alpha_core A_r
```

first from the remaining upward liquid in those newly swept cells. If the
legacy/current cell is entirely downward-moving, the closure next removes
downward liquid only above the cumulative geometric film requirement
`S_new (1-alpha_core) A_r`. Consequently a full downward-moving cell can become
the required falling film plus gas void instead of reporting a false zero-core
event. Every removed parcel carries its instantaneous signed axial momentum.
Upward and downward parcels are deposited into their matching directional
streams in available void above the new front, nearest unswept cell first. No
interface impulse is invented. If the receiver cells do not have enough
capacity, the result returns directional and total liquid volume/kinematic
momentum as an explicit overflow inventory. The global stage must route that
inventory through the atmospheric rim or reject the stage. It may not discard
it. State plus overflow conserves volume and axial momentum to roundoff.

Do not call `extend_taylor_sweep_in_persistent_state` alone in the production
loop: it forms the film but, by construction, does not create gas void. The
combined API is the complete geometry event.

The upper-free-surface event changes only the top boundary: the pressure face
becomes atmospheric and liquid may leave but may not enter. It does not reset
the core/film partition.

## Geometry, pressure, and drag

`coaxial_core_film_geometry` computes an instantaneous core--gas--film annulus
from the conserved areas. Interface perimeters and the gas hydraulic diameter
are geometric consequences of those areas. `adapt_gas_void_and_pressure_faces`
then evaluates the isothermal EOS from gas mass and void volume. A finite void
without gas mass is rejected rather than silently filled with atmospheric gas.

`advance_post_event_core_film_stage` transports the two liquid inventories with
one shared face flux per interface, applies local conservative topology
transfer at reversal, rebuilds the gas/film geometry, and optionally applies
the implicit gas/upward-liquid/downward-liquid drag exchange. The returned gas
momentum already includes the equal-and-opposite drag reaction and must replace
the pre-drag vertical gas momentum.

## What the closure can and cannot guarantee

Keeping separate upward and downward inventories prevents vigorous gross
counter-current flow from collapsing into a nearly zero net liquid velocity.
It therefore makes a persistent, high-holdup bottom mixing region possible.
Its actual holdup is not prescribed here: it must emerge from the finite
T-node gross rates, gas pressure, donor inventories, gravity, wall friction,
and interfacial drag. The Case-A 2-D hold-up range is an acceptance test only,
not an input to these closures.
