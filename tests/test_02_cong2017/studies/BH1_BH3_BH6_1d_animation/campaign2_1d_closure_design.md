# Campaign-2 1D closure design and qualification gates

Status: phase-3C moving-shock/cut-cell closure design and implementation gate record. This
file is not simulation evidence and does not qualify any H1/H3/H6 result.

## Non-negotiable model contract

- H1/H3/H6 use one apparatus, material, valve, numerical and closure contract;
  only the riser diameter differs.
- The horizontal model is the Case-1 Tosan shock-fit/Saint-Venant/MOC core.
  Its audited source remains unchanged at SHA-256
  `90e84da9afa0ec8465d80f87fc701dfb8f0fad6f97350ea708074a50192b6119`.
- The vertical model owns conservative `Al/Ql/Mg/Jg` finite-volume state.
- Neither solver receives the experimental geyser label. A geyser is diagnosed
  only from positive liquid volume crossing the physical riser rim.
- No dry-film rendering rule, source term, threshold change or per-case
  coefficient may alter the simulated outcome.

## Case-1 local-valve extension

The physical valve is at `x=5.98 m`, or mirrored `x'=0.61 m`; with
`dx=0.01 m` it is exactly face 61. The active Campaign-2 path must replace the
current global hydraulic-time scaling with a fixed local passive valve using
the shared `phi/K/dp` law. Physical time, MOC propagation, gas-pocket evolution
and valve time must always advance by the same `dt`.

The original Case-1 source cannot receive an optional hook without changing
its hash. A version-pinned sibling extension is therefore required. With no
local face it must call the original `step` directly and be bitwise identical.
With the valve active it must return an immutable integrated transaction;
mutable ledgers are committed by the owner exactly once only after the whole
horizontal/reservoir/T/vertical/global-budget step succeeds.

Three numerical regions require distinct conservative closures:

1. `cut >= 62`: face 61 lies inside the free-surface FV region. Both SSPRK2
   stages must solve the valve from their own state/time. The two sides share
   one accepted volume flux but have distinct momentum fluxes whose difference
   is the valve-wall force.
2. `cut <= 59`: the face lies inside the pressurised MOC region. The MOC domain
   must be split at face 61 and the incoming `C+`/`C-` traces closed by the
   passive two-port valve; side elastic-volume changes are equal and opposite.
3. `cut in {60,61}`: the fixed valve touches the Tosan moving cut cell. A
   coupled fixed-valve/subcell-storage/moving-shock solve is required. Feeding
   only the FV boundary or only the MOC boundary loses one side of the volume
   transaction. This start-up region lasts about `0.097 s` in the unvalved
   baseline, when the valve still has material resistance, so it cannot be
   skipped.

The versioned sibling contains the phase-2 **generic clean-FV plumbing** used
by the phase-3A closure below. At a specified internal face, each SSPRK2 Euler stage exposes
that stage's reconstructed left/right traces and native Case-1 flux to an
immutable callback. An accepted override has one shared volume flux and two
momentum fluxes; their difference is the zero-volume face wall force. The
Case-1 donor factor is applied to the shared volume flux and both momentum
ports, while the original boundary treatment, hydrostatic/elastic area flux,
bed slope, Manning/Darcy friction, dry-front regularisation and interface
traction remain in the stage update. The second callback is recomputed from
the provisional SSPRK state at `t+dt`. `callback=None`, a callback returning
`None`, or an explicit exact-native (`K=0`) pair uses the original core update
bit-for-bit. Phase 3B extends that callback to exact-dry and directional
critical/supercritical traces without changing the generic stage machinery;
the pressurised MOC regime remains an explicit rejection. This infrastructure is
not H1/H3/H6 run evidence.

### Phase-3A clean free-surface valve closure

Phase 3A uses the native Case-1 central-upwind face as the lossless numerical
datum and adds only the passive valve jump. Let the reconstructed west/east
traces at face 61 be `(A_L,Q_L)` and `(A_R,Q_R)`, and let `(Q_0,F_0)` be the
native Case-1 mass and momentum fluxes from those same traces. Phase 3A first
qualified the two-wet/subcritical branch. A face touching the tracked shock or
an elastic/full-pipe trace is not regularised and remains rejected atomically.

For the circular Saint-Venant branch,

`F_p(A)=g I1(A)`, `c(A)^2=dF_p/dA=g A/T`, and

`dJ+ = du + c dA/A`, `dJ- = du - c dA/A`.

The corresponding circular characteristic primitive is

`Psi(A)=integral_0^A c(a)/a da = integral_0^h g/c(eta) d eta`.

This is not the full-liquid water-hammer invariant. Linearising the stationary
volume flux `Q=A u` on the incoming `J+` trace from the west and `J-` trace
from the east gives the positive subcritical flow impedances

`Z_L = rho c_L^2/[A_L(c_L-u_L)]`,

`Z_R = rho c_R^2/[A_R(c_R+u_R)]`.

Only in the small-velocity limit do these reduce to `rho c/A`; the acoustic
`rho a/A` formula is not used on this branch. The native face flux `Q_0` is
the exact `K=0` anchor. With `A_v` equal to the upwind reconstructed *wetted*
area (the 2D loss acts on local liquid velocity, not on full-bore superficial
velocity), `u_v=Q/A_v` and

`R=0.5 rho K/A_v^2`.

This area has a different role from two existing full-area values. In the 2D
`UEqn.H`, the loss is `0.5 rho K |U|U/L` using the local mixture/liquid
velocity; `referenceFlowArea` is used only to audit the resistance-zone length.
Likewise, `FixedInternalValveSpec.valve_flow_area_m2=pi D^2/4` is the nominal
apparatus/full-liquid area and remains the input to the separate full-liquid
water-hammer API. Neither is substituted for `A_v` on a partially wet
Saint-Venant face. Here `A_v` is named and recorded as
`upwind_wetted_area_m2` solely to map the shared FV flow `Q` to that local
liquid velocity.

The shared valve flux is the unique same-sign root of

`(Z_L+Z_R)(Q_0-Q) = R |Q| Q`.

Thus `|Q|<=|Q_0|`, `dp=R|Q|Q`, and `dp Q>=0`. The sine-squared opening is
already contained in `K=phi^-2-1`; `A_v` is not multiplied by `phi` again.
At exact `K=0`, the callback returns the original Case-1 flux object rather
than reconstructing an approximately equal value. For every positive `K`, the
stable quadratic root is used with no small-`K` threshold, so `Q->Q_0` and all
momentum corrections vanish continuously as `K->0`.

The characteristic pressure corrections are apportioned by impedance,

`delta p_L = [Z_L/(Z_L+Z_R)] dp`,

`delta p_R = -[Z_R/(Z_L+Z_R)] dp`.

Using the native momentum flux as datum, the common throat-advection datum is

`F_c = F_0 + (Q^2-Q_0^2)/A_v`.

The two numerical momentum ports are then constructed as

`F_L = F_c + A_v delta p_L/rho`,

`F_R = F_L - A_v dp/rho`.

Consequently their difference is the valve-wall force exactly by construction:

`rho(F_R-F_L) = -A_v dp = F_wall,on-liquid`.

The face owns no volume. The generic donor limiter must leave the physical
valve flux unscaled (`theta_v=1`); otherwise the completed Forchheimer solve
would no longer correspond to the accepted flux, so the substep is rejected
for a smaller-step retry instead of silently changing the law.

Each SSPRK2 stage reconstructs and solves independently at its own stage time.
For one accepted substep of length `dt`, the valve transaction uses the SSPRK2
quadrature weights, e.g.

`Delta V = 0.5 dt (Q_1+Q_2)`,

`I_wall = 0.5 dt (F_wall,1+F_wall,2)`, and

`E_diss = 0.5 dt (dp_1 Q_1+dp_2 Q_2)`.

The mirrored/physical coordinate reversal is applied only after these
integrals: signed through-volume and axial wall impulse change sign, while the
west/east momentum ports exchange. A requested step crossing `t=0.20 s` is
split exactly at that event before CFL subcycling. This phase does not provide
the initially dry gate/cut-cell closure or the pressurised MOC closure.

### Phase-3B directional critical, supercritical and wet/dry closure

Phase 3B retains `(Q_0,F_0)` as the Case-1 Riemann supply and counts only
characteristics that can reach a stationary face. Define `s=sign(Q_0)` and
directional velocity `v=s u`. The upstream trace is west for `s>0` and east for
`s<0`; the other trace is downstream.

- An exact-dry trace is `A=Q=c=0`. Case-1 reconstructed states at or below its
  existing dry-area classification are mapped to this zero state. No positive
  area or numerical film is introduced.
- A wet upstream trace with `-c<v<c` contributes its incoming impedance. A wet
  subcritical downstream trace contributes the opposing impedance. A dry
  downstream trace or a supercritical outflow trace with `v>=c` contributes no
  incoming condition.
- An upstream trace with `v>=c` sends both characteristics into the stationary
  face. A possible native-supply solution is therefore choked: `Q=Q_0`.
  Reducing `Q` locally would be an unphysical Froude clipping because no
  downstream characteristic can change that supplied state. This branch is
  admissible only when the downstream trace is exact-dry or supercritical
  outflow and the post-loss specific energy can support a downstream state.
- A dry native-flow donor, an upstream trace with `v<=-c`, or an opposing
  downstream supercritical stream with `v<=-c` has no single-valve Riemann
  resolution in this branch and is rejected atomically.

For two controllable traces, the phase-3A tangent closure is retained using the
sum of the two active impedances,

`Z_active (Q_0-Q) = R |Q|Q`.

For a one-sided upstream trace, the actual Campaign-2 loss is not assumed to be
a small perturbation near `Fr=1`. Let
`Phi'(A)=c(A)/A` and preserve the upstream incoming invariant exactly. With
`s=sign(Q_0)`, upstream trace `(A_0,q_0,u_0)` and candidate area `A`,

`u_char(A)=u_0-s[Phi(A)-Phi(A_0)]`,

`q_char(A)=A u_char(A)`.

The numerical continuation is anchored to the native Case-1 Riemann flux,

`Q_hat(A)=Q_0+[q_char(A)-q_char(A_0)]`.

This offset is essential: recomputing a new CU flux would reintroduce the HLL
area-jump diffusion, while replacing `Q_0` by the analytical critical capacity
would break exact `K=0` transparency. The one-sided root solves

`g[h(A)-h(A_0)] = 0.5 K [Q_hat(A)/A]^2`

on the same-sign branch before full area. It is monotone for the supported
subcritical upstream continuation. The solved `A` is the local upstream wetted
area used in `u_v=Q/A`; no valve-area fraction or nominal full area is applied
again. Failure to find the root before full area or flow reversal is an atomic
rejection.

For a supply-choked upstream trace,

`Q=Q_0`, `dp=R|Q_0|Q_0`.

Let `m=|Q_0|` and

`E_up=h(A_v)+m^2/(2 g A_v^2)`,

`H_L=|dp|/(rho g)=K m^2/(2 g A_v^2)`, and

`E_c(m)=min_(0<A<Af) [h(A)+m^2/(2 g A^2)]`.

The circular critical area recorded in the solution satisfies
`m^2 T(A_c)=g A_c^3`. Fixed-`Q` supply choking is accepted only when
`E_up-H_L>=E_c` to roundoff. If this condition fails, or if a subcritical
downstream characteristic feeds back to the valve, the state requires a
resolved upstream entropy shock/full nonlinear Riemann solve and is rejected
atomically in phase 3B.

If upstream is west, `delta p_L=0` and `delta p_R=-dp`; if upstream is east,
`delta p_R=0` and `delta p_L=dp`. In both cases the existing momentum-port
construction gives

`rho(F_R-F_L)=-A_v dp`, and `dp Q>=0`.

Thus the choked branch is conservative and passive even though its mass supply
cannot react instantaneously. Its stage record stores `E_up`, `H_L`, `A_c`,
`E_c` and the energy margin, while the immutable SSPRK transaction stores the
two momentum impulses, their wall-force difference, and
`rho g H_L |Q_0| dt` dissipation. At exact `K=0`, trace admissibility and
characteristic counting are bypassed and the native Case-1 flux is returned
bit-for-bit. For every `K>0` there is no Froude cap, outcome threshold,
artificial source, per-case coefficient or small-`K` switch.

The unscaled Case-1 state at `t=0.120 s`, face 61, has reconstructed
`Fr_L=-1.075275` and `Fr_R=-0.997526`, with `Q_0<0`. The east trace is the
subcritical upstream control; the west trace is supercritical downstream
outflow and supplies no incoming condition. It therefore uses the one-sided
east nonlinear `J-` continuation, not a Froude rejection or a choked
approximation. Its first-stage root is approximately
`A*=1.22260e-3 m2`, `Q*=-3.63022e-4 m3/s`; both SSPRK stages solve their own
updated trace independently.

The dry side of the clean-FV callback is supported only after face 61 is
unambiguously inside that regular region (`cut>=62`). The adjacent cut element
is instead closed by phase 3C below; pressurised MOC (`cut<=59`) remains the
explicit next gap.

### Phase-3C fixed-valve/moving-shock cut element

When `cut in {60,61}`, the Case-1 cell average cannot be passed to the regular
valve callback: it is a geometric mixture of a fitted free-surface subcell and
an elastic full-pipe subcell. Phase 3C reconstructs the incoming Case-1 feet
only from complete cells, solves the fixed valve and the moving Tosan interface
in one implicit cut element, and rebuilds the mixed cell from the solved shock
position. The cut average is never treated as a characteristic foot and no
film, source, Froude cap, result threshold or case-specific coefficient is
introduced.

For `cut=60`, a full-pipe subcell lies between the fitted interface and fixed
face 61. Let `Q` be the one valve flow, `H_L/H_R` its full-pipe heads and `w`
the reflected Case-1 entropy shock speed. The right incoming `C-`, the common
Forchheimer jump and the fitted mass/momentum jumps are solved together:

`u-u_f-g/a(H_R-H_f)+S_p=0`,

`rho g(H_L-H_R)=0.5 rho K |u|u`, `Q=A_f u`,

`Q-A_s u_s-w(A_f-A_s)=0`,

`Q^2/A_f+gA_f(H_L-D/2)-A_su_s^2-gI_1(A_s)-w(Q-A_su_s)=0`.

The free-surface trace `(A_s,u_s)` follows the same reflected Case-1 incoming
characteristic and source term used by the pinned negative-interface solve.
The two valve momentum ports are

`F_L=Q^2/A_f+gA_f(H_L-D/2)` and
`F_R=Q^2/A_f+gA_f(H_R-D/2)`.

For `cut=61`, a free-surface subcell lies between face 61 and the fitted shock.
The two positive areas `(A_L,A_R)`, shared `Q`, pressurised interface
`(u_p,H_p)` and shock speed `w` are solved from the west free-surface incoming
characteristic, east full-pipe incoming characteristic, fitted mass and
momentum jumps, passive mechanical-energy loss and stationary valve momentum
balance. With upwind wetted area `A_u`,

`dp=0.5 rho K |Q/A_u|(Q/A_u)`,

`E_L-E_R=dp/(rho g)`, and

`rho[(Q^2/A_R+gI_1(A_R))-(Q^2/A_L+gI_1(A_L))]=-dp A_u`.

Thus both orientations have one volume transaction, two momentum ports whose
difference is exactly the valve-wall force, and non-negative `dp Q`
dissipation. Each accepted substep solves stage 1 at `t`, builds a provisional
moving-cut state, reconstructs stage 2 at `t+dt`, and applies SSPRK2 half
weights. If the Case-1 donor limiter would alter the solved valve flow, or the
implicit cut residual has no admissible passive root, the candidate is rejected
without mutating the input and retried by binary physical-time subdivision.
This subdivision changes neither the 0.20 s opening law nor elapsed time.
The bounded nonlinear solve is accepted only by its dimensionless equation
residual; its deterministic seeds come from the current complete-cell traces
and pinned Case-1 interface solution. This numerical residual gate contains no
experimental outcome or case label.

At exact `K=0`, an entire shock-cut substep dispatches directly to the pinned
Case-1 core and is bitwise identical. A step ending at `t=0.20 s` may have a
positive-loss first stage and exact-native second stage; the outer event plan
lands exactly on `0.20 s` before continuing. The phase-3C regression starts
from the real Campaign-2 `t=0`, exercises both `cut=61` and `cut=60`, preserves
global liquid volume to binary64 roundoff, and verifies atomic donor/nonlinear
rollback. It does not implement or bypass the separate split-MOC valve needed
once `cut<=59`.

## Vertical sharp-interface reconstruction

For one monotone upper interface (liquid below, gas above), a cut-cell position
can be reconstructed from liquid volume,
`s=z_low+(Al/A) dz`. A downward liquid-column flux at a grid-aligned surface
must not pull liquid from the dry cell above. Faces through the bottom-connected
saturated prefix carry the common negative liquid flux, while the physical
interface face carries zero liquid flux. The highest full-liquid cell therefore
becomes a cut cell.

The new void is filled in the same step by a negative gas-mass flux from the
adjacent top-connected gas donor. `Mg` and `Jg` are removed from the donor and
added to the receiver by one shared star state; no atmospheric gas is created.
Liquid volume, gas mass, combined phase momentum and paired interface-pressure
impulse/work must close before the step is accepted. Interface CFL, donor
inventory, EOS positivity and topology uniqueness are atomic acceptance gates.

The implemented reverse transaction uses the same monotone component. For
positive liquid flow, a grid-aligned front selects the first pure-gas cell;
an existing cut front keeps that cut cell. The common saturated-column liquid
flux reaches its lower material face, while its upper material face carries
zero liquid flux. Gas in that interface cell is the authoritative donor:
`rho_g=Mg/(Ag dz)` and `u_g=Jg/Mg`, so `mdot_g=rho_g Q_l` and the convective
momentum flux is `mdot_g u_g` upward into the connected gas column. The lower
material face carries zero gas. A restart reconstructs every exactly positive
paired `Al/Ag/Mg` cut without a film or phase threshold; only an exhausted
phase within `16 ulp(A)` is canonicalized to exact zero/full geometry. During
that binary64-only pin, residual gas mass and momentum are transferred to the
adjacent connected gas cell, never created or discarded.

The four cell averages alone do not identify interface direction: the same
`(Al=A/2,Ql=0,Mg=m,Jg=0)` can represent liquid-below/gas-above or the reverse,
and the correct lower-face phase flux differs. Campaign 2 may infer the single
upper interface only while bottom-liquid/top-gas connectivity is monotone. A
restart-capable general implementation must persist orientation/connectivity.
Two interfaces in one cell, disconnected gas pockets, or gas-core/liquid-film
coflow require extra topology/component state or an explicit rejection.

First gas entry through the bottom T is a separate `gas below / liquid above`
interface. It must jointly solve gas mass inflow, interface speed/pressure and
liquid-column displacement. The current `open_area=min(existing void,...)`
rule has a zero-area deadlock and cannot be used. Gas and liquid fluxes cannot
be prescribed independently. The isothermal state has no total-energy
variable, so it may audit paired pressure work and thermostat heat, but must
not claim total-energy conservation.

### Minimum first-bottom-entry transaction extension

The present `TeeTransaction` is not sufficient to close that first entry
uniquely. It stores gas mass flux `mdot_g`, convective gas momentum flux
`Pi_g=mdot_g*u_g`, and interface pressure, so `u_g=Pi_g/mdot_g` is known. It
does **not** store the upwind donor density or gas volume flux. The positive
flow branch of `solve_gas_tee` uses the horizontal gas-pocket density, whereas
the receiving riser uses an isothermal EOS. Two donor states can have the same
interface pressure, `mdot_g` and `Pi_g` but different density; then

`Q_g=mdot_g/rho_d` and `A_open=Q_g/u_g`

are different. Reconstructing `rho_d` as `p/(R*T_riser)` inside the vertical
kernel would silently replace the horizontal donor state with the receiving
state. The transaction also stores liquid volume flux but no convective liquid
momentum flux; once the mouth becomes two-phase, using `rho_l*Q_l^2/A` assumes
an unrecorded liquid flow area.

The minimum independent conservative additions are therefore:

- `gas_volume_flow_to_riser_m3_s`, computed by the T gas Riemann solver from
  its actual upwind donor state;
- `liquid_normal_momentum_flow_N`, the convective liquid momentum flux with
  the common pressure term excluded.

No separate gas velocity or opening field is mathematically required: an
accepted transaction can audit `u_g=Pi_g/mdot_g`,
`rho_d=mdot_g/Q_g`, and `A_open=Q_g/u_g`, then require positive finite values
and `0<A_open<=A_r`. The inherited liquid and gas pressure members must map to
one common absolute pressure `p*` to binary64 roundoff. That one `p*` is
consumed once as the bottom pressure face; it is not added again to either
convective momentum flux.

For the first material-front step, with no gas above the new lower front,
the conservative ALE identities are

`A_g,0^(n+1) = (dt/dz) Q_g,0`,

`A_l,0^(n+1) = A_r - A_g,0^(n+1)`,

`Q_l,1 = Q_l,0 + Q_g,0`,

`M_g,0^(n+1) = dt mdot_g,0`, and

`J_g,0^(transport) = dt Pi_g,0`.

The same `Q_l,1` continues through the connected saturated liquid plug to its
upper material surface. These equations make the cellwise mixture-volume
residual `d[(Al+Ag)dz]/dt` and the gas mass residual identically zero; the new
liquid momentum member makes the boundary momentum ledger determined rather
than guessed. A release implementation must also record equal-and-opposite
internal pressure impulses `+/- p* A_r dt` and their zero pair residual.

Acceptance gates are exact positive mass/volume pairing, common-pressure
compatibility, donor density and opening admissibility, one-cell lower- and
upper-interface CFL, available upper void, and atomic rollback. The initial
receiver must have `Ag=Mg=Jg=0`; a tolerance-sized or pre-seeded void is not an
input to the closure. After acceptance, checkpoint state must persist the
lower-front cell and `gas-below/liquid-above` orientation because `Al/Ql/Mg/Jg`
alone cannot reconstruct it. Until those two transaction fields and restart
topology are produced by the common T solver/owner, the vertical kernel must
reject every strictly positive first gas exchange, including the smallest
positive binary64 value; it must not infer the missing fluxes locally.

## Required release tests

- Original Case-1 hash unchanged; no-valve extension path bitwise equal for
  multiple states, time steps and internal subcycles.
- Exact face/area/sign mapping; positive and reverse flow; SSPRK2 and MOC
  characteristic residuals; equal-and-opposite side volume; wall impulse and
  non-negative dissipation; `K=0` native transparency.
- Dry start with no pre-wetting; cut `60 -> 61 -> 62` transitions without lost
  or duplicate volume; event-aligned `t=0.20 s` step; limiter rejection/retry.
- Static upper interface; grid-aligned and partial-cell retreat; gas donor
  conservation; opposite/ambiguous topology rejection; interface-CFL retry.
- First bottom-gas entry without seed-void dependence; equal-pressure fixed
  point; paired interface impulse/work; donor/capacity rejection and rollback.
- One valve-ledger commit per accepted coupled physical step; injected failure
  after each owner restores state, clocks and every ledger byte-for-byte.
- Shared H1/H3/H6 contract differs only by `Dr`; complete-event runs conserve
  liquid/gas and classify physical rim crossing without outcome inputs.
