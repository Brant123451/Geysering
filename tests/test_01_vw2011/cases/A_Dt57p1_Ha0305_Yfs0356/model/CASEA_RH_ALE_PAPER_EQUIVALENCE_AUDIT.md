# Case A material-front RH/ALE equivalence audit

## Scope and evidence

This audit compares the following implementation chain:

- `casea_material_front_rh_adapter.py`
- `casea_gas_coupled_front.py`
- `casea_material_front_cutcell.py`
- the called `solve_front_rankine_hugoniot` /
  `tosan2021_horizontal_shockfit.solve_oriented_interface`

against the model in
`E:\Research\论文\model_algorithm_revised_20260803`, principally main-text
Eqs. (3), (12), (14), (15), (17), (20)--(21), (33)--(35), (39)--(40), and
Appendix-A Eqs. (A30)--(A40).

The audit changes scientific content, not presentation.  It does not alter a
time loop, an animation, an OpenFOAM case, or either manuscript figure.

## Verdict

The existing **ALE cut-cell operator is a valid conservative geometric
operator**, conditional on being supplied with correct physical traces.  The
existing **RH adapter is not equivalent to the paper closure**.  It is an old
Tosan-type two-equation interface with a linear gas-pressure characteristic.
Passing its own ALE residual checks establishes internal consistency of that
old closure; it does not establish consistency with the paper equations.

Consequently, the existing adapter must not be used as paper-model evidence,
and in particular must not be used as the vertical-front closure.

## Equation-by-equation mapping

| Paper requirement | Existing implementation | Finding |
|---|---|---|
| Elastic interface state `U_p=[A_p,Q_p]` with `A_p` unknown, Eqs. (12), (20) | `_assemble_traces` sets `PressurisedState(area=A_f, ...)` | **Not equivalent.** Elastic inventory and characteristic compatibility at the trace are lost. |
| Pressurised momentum `Q_p^2/A_p + a^2(A_p-A_f)` (small-deformation Eq. (12)) | Tosan solve uses `A_f u_p^2 + g A_f(H_p-D/2)` and the adapter again fixes `A_p=A_f` | **Not equivalent.** This is the old hydrostatic-reference form, not the paper trace variable. |
| Stratified momentum `Q_l^2/A_l + 0.5 Lambda_d A_l^2`, Eqs. (3), (12) | `g I_1(h)+g H_g A_f`; no density-ratio, inclination/top-width, or slip contribution | **Not equivalent.** It cannot reproduce the paper's inclination or IKH terms. |
| Material gas condition `u_g,Gamma=w`, Eq. (15) | Gas trace momentum is constructed with `u_g=w` | **Equivalent.** |
| Acoustic predictor `p_Gamma=p_1+rho_1 c_g(w-u_1)`, Eq. (40), on its positive-pressure domain | `right_going_boundary_velocity(p)=u_1+(p-p_1)/(rho_1 c_g)` and the outer scalar solve enforces equality with `w` | **Algebraically equivalent.** The existing code rejects an unbracketed/non-positive state rather than applying the paper's `P_min` floor. Rejection is retained in the strict audit core because a floor was prohibited for this Case-A repair. |
| Gas ALE entries `(0,p_Gamma A_g)`, Eqs. (28), (35) | Gas physical momentum uses `(p_Gamma-p_atm)A_g`, hence ALE impulse is gauge pressure times area | **Not equivalent to the stated paper flux.** The paper explicitly calls `p_Gamma A_g` an internal piston impulse. |
| Active-set ordering by `w` relative to `u_l +/- sqrt(Lambda_R A_R)` | No slow/middle/fast classification in the adapter/Tosan solve | **Missing.** |
| Instantaneous RH/characteristic closure, Eqs. (14), (17) | Tosan characteristic adds `g dt(S_f-S_0)` and uses `bed_slope` there | **Not equivalent.** Regular axial sources belong in branch updates, not in the zero-thickness RH jump. |
| No result cap for this repair | Tosan Newton admissibility requires `abs(u_p)<=2a`, `abs(w)<=2a`, bounds head, and clips free-surface depth before solving | **Violation.** The adapter docstring's “without a speed cap” statement is false through its transitive dependency. |
| Inclination through `zeta=cos(theta)/T`; axial gravity in regular sources | `bed_slope` changes only the time-discrete characteristic source; transverse `g I_1` remains horizontal | **Not equivalent and not vertical-safe.** |

## ALE cut-cell result

`ALEInterfaceFlux.from_traces` correctly evaluates `G=F-wU`, verifies the two
liquid ALE fluxes, verifies `j_g-w m_g=0`, and returns zero gas-mass flux.  The
inventory update and exact face-crossing remap do not contain a fill, area
clip, prescribed waveform, or front-speed rule.  Averaging the two liquid ALE
fluxes is harmless for an exact closure (the two values agree to roundoff), but
it cannot repair a physically wrong trace provider.

For the equilibrium state used by the old adapter regression test, the old
trace gives `A_p/A_f=1` and `w=0`.  Re-evaluating exactly that trace with the
paper momentum potential gives a dimensional momentum residual of
`-2.2388e-2 m^4/s^2` for a horizontal branch (the same order as the complete
pressure flux).  Its gas ALE impulse is `5.1723 N/rho-scale` in the old
gauge-pressure convention, whereas the paper absolute-pressure impulse is
`146.7822` in the same per-density area-pressure scale.  This example explains
why the old self-consistency tests pass while paper-equation consistency does
not.

## Independent strict replacement core

`casea_paper_material_front_rh.py` was added without connecting it to the main
loop.  It provides:

1. the full paper `Lambda_d`, including `cos(theta)/T` and the slip term;
2. fixed common-node pressure or the affine acoustic predictor of Eq. (40);
3. an unknown elastic `A_p,Gamma` reconstructed with the incoming waterhammer
   characteristic;
4. exact liquid mass and momentum RH balances;
5. enumeration of all real admissible roots of the reduced paper closure;
6. explicit slow/middle/fast labels when the adjacent state is hyperbolic;
7. exact cut-cell traces with gas ALE impulse `p_Gamma A_g`.

The root enumerator contains no prescribed speed, speed cap, state fill,
clipping, or HLL fallback.  It substitutes every polynomial root back into the
unmultiplied dimensional equations.  A multiple-root state is exposed to the
caller instead of silently selecting a trajectory.

Five independent tests cover a vertical (`cos(theta)=0`) exact solution,
polynomial recovery, liquid ALE identities, absolute gas piston impulse, the
linear gas characteristic, and a deliberately super-`2a` exact state showing
that no hidden Tosan speed cap remains.

## Vertical-branch applicability

There are two separate questions:

1. **Algebraic inclination:** the strict replacement evaluates the paper
   equations at `cos(theta)=0`, so the transverse hydrostatic/buoyancy term
   vanishes while gas-pressure and slip terms remain.  The old adapter does
   not do this and cannot be used for a vertical branch.
2. **Physical riser topology:** the paper stratified state assumes positive
   gas and liquid areas in the same axial cross-section.  A flat, full-bore
   water column with gas above it is instead an axial gas--liquid material
   interface, and bubbly/slug riser flow needs its own vertical two-fluid
   branch representation.  The strict RH kernel alone does not supply that
   topology, the T-junction balance, or the riser top boundary.

Therefore the new kernel is **inclination-correct but not, by itself, a
complete vertical-riser solver**.  It can be connected only after the vertical
branch state and T-node flux use the same conserved variables and pressure
reference.  Until then, the safe production status of the current vertical
front path is **not ready**.

## Integration gate

Before replacing the old provider in a Case-A run:

- use the inclination-aware four-variable branch flux on the stratified side;
- use the same absolute/gauge pressure convention at regular faces, the front
  impulse, and the `P_g dA_g/dx` source so no reference-pressure force is
  counted twice;
- pass the finite T-node pressure to the fixed-pressure form, or use the
  adjacent gas characteristic, but never solve two inconsistent pressures;
- reject `Lambda_d<=0` where a hyperbolic branch update is required rather
  than masking it with a wave-speed floor;
- expose multiple RH roots for physical active-set/entropy selection;
- verify gas mass, liquid volume, liquid momentum, and the T-node finite-volume
  ledger before generating any HTML or paper figure.

