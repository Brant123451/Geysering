# Campaign-2 finite bottom-pocket and lower-front closure

## Scope and topology state

This closure continues the already accepted, no-seed first bottom-gas entry.
It is common to B-H1, B-H3 and B-H6 and reads neither a case identifier nor an
experimental geyser label.  The only case-varying physical input remains the
riser diameter.

The persisted pair

```text
lower_material_front_cell = k
lower_material_front_orientation = gas_below_liquid_above
```

has one precise meaning: `k` is the first cell containing liquid.  It is either

- a cut cell with `0 < Al[k] < A`, lower-component gas below and liquid above;
  or
- the first full-liquid cell above an exactly grid-aligned gas pocket.

Cells below `k` are connected pure gas.  Starting at `k`, the liquid cells form
one connected plug.  A single liquid-below/top-gas surface terminates that
plug.  Two unresolved interfaces in one cell, disconnected pockets and an
annular gas-core/liquid-film topology remain inadmissible because
`(Al,Ql,Mg,Jg)` plus one orientation marker cannot distinguish them.

## Finite-storage T-mouth continuation

After the lower pocket exists, the physical riser bottom face belongs to gas.
The liquid plug above the lower front is not a third liquid characteristic at
the zero-volume T.  The horizontal liquid node is therefore the unique
two-branch solve

```text
q_i(p_l*) = A_i [J_i + p_l*/(rho_l c_l)],  i = west,east,
q_w + q_e = 0,
Q_l,bottom = 0.
```

A common shift of both liquid gauge-pressure references shifts `p_l*` by the
same amount and leaves the two flows unchanged.  The gas opening is separate.
The existing horizontal and riser gas traces give

```text
u_g* = [p_h - p_r + Z_h u_h + Z_r u_r] / (Z_h + Z_r),
p_g* = [Z_r p_h + Z_h p_r + Z_h Z_r (u_h-u_r)] / (Z_h+Z_r),
Qg = A_open u_g*,
mdot = rho_upwind Qg,
Pi_g = mdot u_g*.
```

`Qg`, `mdot`, `Pi_g` and `p_g*` remain independent transaction data.  The
vertical receiver never reconstructs `Qg` as `mdot/(p/RT)`.  It audits

```text
rho_d = mdot/Qg,
u_d = Pi_g/mdot,
A_open = Qg/u_d,
```

including their signs, finite values and physical mouth area.  Reverse flow is
valid: `Qg<0`, `mdot<0`, `u_d<0`, while `Pi_g=mdot*u_d>0` remains the signed
upward-momentum flux tensor component.

## Lower material-star solve

The lower interface uses the actual isothermal gas state below and the liquid
state above.  With upward positive velocity, impedances
`Zg=rho_g c_iso` and `Zl=rho_l c_l` give

```text
u_f = [p_g-p_l + Zg u_g + Zl u_l] / (Zg+Zl),
p_f = [Zl p_g + Zg p_l + Zg Zl (u_g-u_l)] / (Zg+Zl),
Q_f = A u_f.
```

The lower gas-pocket volume changes geometrically by `Q_f dt`.  Its bottom
volume exchange is `Qg dt`; consequently

```text
V_compression = (Qg-Q_f) dt
```

is the finite isothermal storage/compression volume, not a mass source.  Gas
mass changes only through `mdot`; pressure then follows the conserved mass and
geometric volume through the existing isothermal EOS.

## Conservative interface transport

Both ends of the incompressible liquid plug translate with `Q_f`.

- No liquid crosses the gas-owned bottom face.
- No gas crosses the lower material surface into the liquid plug.
- All liquid faces inside the plug carry the same `Q_f`.
- At the upper surface, gas mass and momentum are displaced using the actual
  donor density and velocity, exactly as in the existing upper-front closure.
- At a grid-aligned lower advance, the newly opened gas volume receives mass
  from the real lower gas donor; no film or seed is created.
- During retreat, the bottom pocket may shrink and disappear only when its
  positive gas inventory has left through the physical T.  Positive mass is
  never discarded to force a topology change.

For a cut lower cell, the common interface pressure contributes the explicit
pair

```text
I_l,f = +p_f A dt,
I_g,f = -p_f A dt,
I_l,f + I_g,f = 0.
```

The remaining lower/upper boundary pressure forces, gravity and wall forces
stay in their respective phase control volumes.  The isothermal model has no
gas-energy variable, so this closure audits mass, volume, phase momenta and
paired pressure impulse; it does not claim total-energy conservation.

## Event splitting, restart and rollback

The directional geometric limit is the minimum time to exhaust either

- the advancing lower cell's liquid or retreating lower cell's gas; and
- the corresponding gas or liquid inventory at the translated upper surface.

The shared driver applies its case-independent CFL `0.45` to acoustic and
ordinary transport limits.  A material-event limit itself is allowed to land
exactly on the face; multiplying it repeatedly by `0.45` would approach the
face asymptotically.  The marker then becomes the next first-liquid cell
(advance), the previous gas cell becomes a cut (grid-aligned retreat), or the
marker disappears after complete bottom-pocket evacuation.  Residual phase
momentum at an ulp-scale exhausted cell is transferred to the adjacent
connected component rather than erased.

After complete evacuation, the lower marker is absent but the translated
upper free surface can remain a cut cell.  A new first-entry transaction moves
the liquid plug at the actual net fill `Ql+Qg`, not at `Qg` alone.  The shared
step therefore takes the minimum of

```text
Al[0] dz / Qg
```

and the directional upper-surface phase inventory divided by `|Ql+Qg|`, in
addition to the acoustic and ordinary transport limits.  This prevents a
nominal driver step from crossing the residual upper cut before the
conservative receiver sees it.

All of this occurs inside the driver's existing atomic transaction.  A failed
geometry, donor, pressure, phase-area or budget check restores both owners,
clocks and ledgers byte-for-byte.

## Verified properties and remaining scope

The kernel tests cover intra-cell advance, exact face landing, multi-cell event
propagation, reversal and complete disappearance, restart determinism, pressure
reference translation, two donor densities with identical mass/momentum data,
overlarge-step rollback and phase-opening non-overlap.  The public production
driver now also covers all three riser diameters with two explicitly declared
constructed stress paths:

1. one continuous lower-front path crosses two vertical cell faces, persists
   its marker through a restart reconstruction, and leaves every local/global
   phase budget closed; and
2. one independent reverse checkpoint exhausts its finite bottom pocket
   exactly, reconstructs the saturated no-marker restart, and accepts a fresh
   no-seed gas entry only at the earlier residual upper-surface event.  A
   `nextafter`-larger step is rejected atomically at both events.

The second path is not joined to the first by an artificial momentum flip.  It
therefore verifies the disappearance and re-entry branches without pretending
that one naturally generated Campaign-2 trajectory has already reversed after
crossing two cells.

This closes the finite T-mouth storage and single lower-front propagation gap.
It does not supply the separately declared production gaps: finite-speed liquid
pressure relaxation, a regime-dependent interfacial geometry, a full interior
acoustic Riemann flux, or entrainment/breakup/coalescence transitions.

## Acceptance boundary

The closure is accepted at the constructed-state level.  All three shared
riser diameters pass the public-driver event, restart, rollback and budget
checks described above.  These tests use conserved fields and complete T-mouth
transactions; they contain no production gas seed, measured outcome, geyser
threshold or case-specific coefficient.  The related seven-file regression
run passes 119 tests, and the edited Python files pass Ruff and byte-code
compilation.

A prior fresh H3 trajectory advanced from `t=0` to exactly `0.200000 s` in
15,870 accepted transactions without an exception, beyond the former
`0.172 s` smoke boundary.  The lower marker was still absent and `Qg=0` there,
so that run did **not** exercise first entry or the finite-pocket branch.  The
remaining evidence gap is one continuous, naturally generated trajectory that
reaches first entry, crosses multiple lower-front cells, reverses, evacuates
the pocket and re-enters without a constructed checkpoint.  That qualification
must wait for the separately owned local-valve/MOC integration and a physical
bottom gas-entry event.  The constructed tests must not be reported as a
complete-event or geyser-classification result.
