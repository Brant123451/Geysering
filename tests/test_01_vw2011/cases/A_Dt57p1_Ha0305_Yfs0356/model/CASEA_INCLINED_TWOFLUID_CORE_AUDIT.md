# Case A inclined two-fluid branch audit

## Scope

This note records the equation-level audit behind
`casea_inclined_twofluid_branch.py`.  The change is scientific model
development only.  It does not alter the manuscript, figures, archived Case A
results, or any current driver.

## Equation source

The implemented state and flux are taken from
`E:\Research\论文\model_algorithm_revised_20260803\main_text_section_2_zh.tex`,
Eqs. (1)--(3), and the matching English source in
`main_text_current_algorithm.tex`:

- `U = [A_g rho_g, rho_g Q_g, A_l, Q_l]^T`;
- `F_g = [rho_g Q_g, rho_g Q_g^2/A_g + P_g A_g]^T`;
- `F_l = [Q_l, Q_l^2/A_l + 0.5 Lambda_d A_l^2]^T`;
- `Lambda_d = 2 g H_g/A_l + ((rho_l-rho_g)/rho_l) g zeta
  - (rho_g/rho_l)(u_g-u_l)^2/A_g`;
- `zeta = cos(theta)/[D sin(gamma/2)]`.

The axial-gravity signs and the decoupled liquid source follow Appendix A,
Eqs. (A5), (A20), and (A30).  With wall and interfacial friction omitted,

- `S_g = -rho_g A_g g sin(theta)`;
- `S_d = -(1-rho_g/rho_l) A_l g sin(theta)`.

The gas equation also retains the two nonconservative terms from Eq. (2),
`P_g dA_g/dx - A_g rho_g g cos(theta) dh_l/dx`, evaluated with one centered
cell derivative.  Optional wall/interfacial terms can be supplied only as
explicit already-decoupled momentum sources; none are hidden in the core.

The gas pressure is isothermal, `P_g=c_g^2 rho_g`.  The liquid restoring
potential uses the gauge head `H_g=(P_g-P_ref)/(rho_l g)`.  This is the
pressure-perturbation convention used by the current model analysis and by the
existing Case A horizontal operator; a spatially uniform atmospheric pressure
therefore does not create a liquid restoring force.

## Audit of `casea_mixed_branch_fv.py`

The existing operator must not be reused for the vertical branch as written:

1. Its module contract and `MixedBranchParameters` explicitly reject every
   nonzero `bed_slope`; it is horizontal-only.
2. Its liquid constitutive path imports `HorizontalLiquidParameters` and the
   horizontal circular-segment pressure-potential operator.  The lower-level
   operator still contains an area cap, void floor, density bounds, and a
   numerical celerity floor.  The current mixed-branch wrapper now audits and
   rejects their activation, but that does not turn the constitutive law into
   a vertical law.
3. Its ordinary gas faces use the existing positive-density gas Roe API while
   liquid faces use a separate Rusanov operator.  The new core instead exposes
   the four-equation block Rusanov flux stated in Eq. (30), with no alternate
   solver path.
4. It accepts distributed sources from its caller but does not derive the
   inclination-dependent axial-gravity or gas geometry terms internally.
5. Its moving-front traces come from the existing Case A RH adapter.  That
   front adapter was not derived or verified here for an inclined/vertical
   branch.

## Guarantees of the new core

- exact admissible domain `m_g>0` and `0<A_l<A_f`, with no state modification;
- exact circular-segment inversion for every strictly interior liquid area;
- exact horizontal and vertical trigonometric limits;
- Eq. (3) retained without a celerity floor;
- `Lambda_d<0` rejects the operation as loss of hyperbolicity;
- the neutral value `Lambda_d=0` remains exactly zero;
- one shared numerical flux per face and a componentwise stage ledger;
- a CFL violation or inadmissible candidate rejects the entire immutable
  Euler input tuple;
- no topology, wave shape, gas pocket, or result is prescribed.

## Is this sufficient for the Case A riser?

No.  It is sufficient only as an interior finite-volume operator for an
already-defined, hyperbolic stratified branch.  A production Case A riser still
needs all of the following:

1. an orientation-consistent moving pressurised--stratified front closure that
   supplies both traces and front speed;
2. a conservative three-branch T-junction boundary Riemann problem for gas and
   liquid fluxes;
3. the upper free-surface/vent boundary and its topology event;
4. verified wall and interfacial shear closures for the vertical topology;
5. a declared vertical phase topology.  At `theta=pi/2`, `zeta=0`, so a
   circular bottom-segment area alone does not distinguish annular, slug, or
   side-by-side arrangements;
6. a policy for the physically neutral or elliptic states exposed by
   `Lambda_d<=0`.  This core reports them and does not hide them by numerical
   regularization.

Accordingly, this module should not be connected to the Case A main loop until
the front and T-junction closures above are independently derived and tested.

## Tests

`test_casea_inclined_twofluid_branch.py` checks:

- Eq. (3) term by term at an arbitrary inclination;
- exact preservation of a uniform horizontal rest state;
- horizontal low-density reduction to the Saint--Venant celerity;
- the exact vertical axial-gravity source and zero transverse geometry;
- periodic conservation of all four integrated state components;
- one declared Eq. (30) face method with no fallback;
- absence of near-full area and tiny-density bounds;
- fail-fast loss of hyperbolicity, CFL rejection, and invalid-area rejection;
- exact retention of the neutral vertical celerity.
