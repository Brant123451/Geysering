# Case A distributed horizontal-liquid operator repair

Scope: numerical/scientific implementation only.  The Case-A time loop,
T-junction schedule, HTML, manuscript, and archived outputs are unchanged.

## Corrected consistency defects

1. The stratified momentum potential is evaluated as
   `Psi=C+0.5*Lambda(A)*A^2`.  Its Jacobian now uses the complete derivative
   `dPsi/dA=Lambda*A+0.5*A^2*dLambda/dA`, including gas-density, circular-width,
   liquid-velocity, and slip derivatives while `Q`, gas mass, and gas momentum
   are held fixed.
2. Negative `dPsi/dA` raises `LossOfHyperbolicity`.  It is never converted to a
   zero wave speed with `maximum(..., 0)`.
3. The mass-supported stratified potential is anchored to the finite-tension
   elastic crown potential at one shared transition area.  The momentum flux is
   therefore continuous when the crown closes; no empirical pressure impulse,
   wave amplitude, or fitted damping is introduced.
4. `ssprk2_stage_step` calls a stage RHS twice using the actual stage state and
   time.  Fluxes, physical sources, gas coupling, and boundary conditions can
   therefore be recomputed at both stages by the main solver.

## Regression evidence

`test_casea_horizontal_liquid_operator.py` checks:

- analytic `dLambda/dA` against centred finite differences;
- reported characteristic speed against the finite-difference flux-Jacobian
  spectrum;
- pressure-potential continuity across the stratified/elastic crown;
- explicit reporting of a non-hyperbolic slip state;
- non-growth of the `2*dx` checkerboard mode under periodic SSP-RK2 evolution,
  and two distinct stage-RHS evaluations.

The new module is intentionally not wired into the monolithic while loop in
this change.  Integration should replace the old pressure-potential and wave-
speed block inside `_decoupled_liquid_rusanov_flux`, then call the SSP-RK2 helper
from the owner of the coupled stage state.  A 9.5 s raw-field audit is required
before any 13 s HTML or manuscript evidence is regenerated.
