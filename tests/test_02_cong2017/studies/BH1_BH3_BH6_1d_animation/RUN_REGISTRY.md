# B-H1 / B-H3 / B-H6 run registry

## Previously accepted 1D archives — rejected for the current qualification

Directory: `case1_frozen_complete/model_1d/`

These archives are no longer accepted for the user's present requirement.
The source is a Campaign-2 full-network FV solver, not the hash-locked Case-1
horizontal core; it also used a 0.25 s valve law, passed 0.66 m as a crown
head instead of 0.61 m, and accumulated material positivity corrections in
H1/H3. Keep the files only as historical diagnostics.

- Source: the hash-locked Campaign-1 full-network core in
  `studies/criterion_map/model/cong2017_network_twofluid.py`.
- Frozen source SHA-256:
  `cea1ffbf6dc5dbae38ab98205f08dbba4544a3e8f951d56de3eb944f3cd9ca23`.
- Common geometry: pipe 6.59 m, tee x=3.47 m, valve x=5.98 m,
  downstream pocket length 0.61 m.
- Common conditions: D=0.05 m, H0=0.66 m, atmospheric initial pocket,
  frozen 0.25 s manual-valve closure (paper reports approximately 0.2 s).
- B-H1: 20 s, geyser; rim at 12.820 s; event tail complete.
- B-H3: 23 s, no geyser; maximum level 1.179 m; model miss; event tail complete.
- B-H6: 20 s, no geyser; maximum level 0.830 m; event tail complete.

The frozen core retains a 2% near-dry numerical film in the released reach.
Physical wet-front detection and rendering use 2.01% as the wet threshold, so
the regularisation film is not presented as physical water. Long-time pressure
spikes and cumulative positivity corrections are not accepted as quantitative
validation evidence.

## Current unified 1D qualification candidate

- Shared contract: `campaign2_shared_contract.py`; H1/H3/H6 differ only in
  `Dr=0.016/0.026/0.041 m`. Solver calls receive no experimental-outcome field.
- Paper/2D apparatus contract: `D=0.050 m`, length `6.590 m`, tee
  `x=3.470 m`, valve `x=5.980 m`, riser height `1.800 m`, initial head
  `0.660 m` from the invert (`0.610 m` from the crown), `T=296.15 K`, and
  a 0.20 s sine-squared effective-area law. The shared 1D material/ambient
  inputs now explicitly match the 2D contract (`rho_l=998.0 kg/m3`,
  `mu_l=0.001003 Pa s`, `K_l=2.2e9 Pa`, `p_atm=101325 Pa`, `g=9.81 m/s2`,
  `R_air=287.05 J/(kg K)`). The horizontal grid is exactly `dx=0.010 m`, so
  the mirrored valve and T are faces 61 and 312 (physical T face 347).
- Horizontal source: Case-1 Tosan shock-fit core SHA-256
  `90e84da9afa0ec8465d80f87fc701dfb8f0fad6f97350ea708074a50192b6119`.
  The study adapter now has conservative physical-coordinate side-T liquid
  transactions and an open-polytropic gas-mass transaction; gas cannot leave
  before the material pocket reaches the T. `campaign2_local_valve.py` is a
  separately tested passive Case-1-characteristic two-port implementation of
  the common local Forchheimer valve law; it is not integrated yet. The active
  adapter still uses the rejected global hydraulic-time scaling during valve
  opening, so no current 1D result may be described as local-valve qualified.
- Vertical source is the conservative `Al/Ql/Mg/Jg` 1D two-fluid FV kernel in
  `campaign2_vertical_twofluid_kernel.py`. Its transient common pressure is
  now reconstructed from conserved gas mass and resolved gas volume with the
  isothermal EOS `p=Mg*R*T/(Ag*dz)`; the mixture-hydrostatic pressure field is
  retained only for initialization and base-state diagnostics. T pressure is
  consumed once as a boundary-face force, and the atmospheric rim uses a
  linear outgoing gas characteristic. The kernel still declares the complete
  Campaign-2 closure unavailable rather than hiding its missing physics.
  Its 36 dedicated tests now also prove that the existing T transaction cannot
  uniquely determine first bottom-gas volume: it drops the horizontal upwind
  gas volume flux/density and carries no liquid convective momentum flux. Every
  strictly positive first gas exchange into an exactly saturated bottom cell
  is therefore rejected atomically, including `nextafter(0,+inf)` after a
  restart; a zero-gas transaction remains bitwise identical to the no-
  transaction path.
- Persistent owner driver: `run_campaign2_persistent_candidate.py`. The same
  T transaction is applied once and in equal amount to the Case-1 horizontal
  owner and the 1D vertical owner; only the reservoir and physical rim are
  external global-budget boundaries. A physical geyser is true only after a
  finite positive liquid volume is integrated through the top FV face.
- The local-valve and vertical-interface implementation gates are frozen in
  `campaign2_1d_closure_design.md`; that document is design provenance only,
  not qualification evidence.
- The adaptive persistent-driver test file currently passes 29 tests, including
  atomic over-limit rejection, exact target/event landing, and all three shared
  diameters. Independent 0.005 s smokes for `Dr=0.016/0.026/0.041 m` each
  completed 324 accepted steps and 324 T transactions; maximum liquid global
  residual was `1.56e-17 m3`, gas residual was zero, and no pre-event top
  liquid appeared. The former 0.05 s/250-step smoke record used `2e-4 s` as a
  fixed step and is superseded because it did not enforce the vertical acoustic
  CFL.
- H3 diagnostic with physical shared efficiencies `(1,1,1)`, delayed T-arrival
  handoff, and full top-outflow accounting ended at 13 s with maximum level
  `1.21446 m` at `10.5404 s` and exactly zero top liquid outflow. It remains a
  clear miss and is not a final result.

The persistent ownership and short-time budgets now pass, but this is not a
20 s qualification result. Atomic failure rollback is verified, and a
bottom-connected saturated liquid block now uses a conservative common-flux
and incompressible pressure projection. Both directions of its unique upper
free surface now use joint liquid/gas displacement: retreat draws real gas
downward from the top-connected donor, while advance/rewetting expels real gas
upward from the grid-aligned first gas cell or the existing cut cell. The
driver applies the same directional geometric CFL to `Al*dz` for retreat and
`Ag*dz` for advance. Exact-positive cut topology survives restart; only an
exhausted phase below `16 ulp(full_area)` is canonicalized, with residual gas
mass and momentum transferred conservatively to the adjacent gas cell. H3
advanced from zero through the former rewetting stop and reached exactly
`0.172 s` in 11495 accepted steps; this is a closure smoke milestone, not a
complete-event or classification result.
The common `2e-4 s` value is now only `max_dt`: every accepted physical step
uses shared `CFL=0.45` and the minimum of the current horizontal limit,
strictly occupied vertical gas-cell `dz/(|ug|+c_iso)` limits, occupied liquid-
cell `dz/(|ul|+28)` limits, the directional upper-interface inventory limit,
and the next target/event. The initial shared step
is `1.543397774503213e-5 s`. The full T-mouth two-phase Riemann/receiving-
capacity closure is still incomplete. First bottom-gas entry is no longer
described as needing an already resolved void: that would be the forbidden
seed-void workaround. The minimum missing transaction data are an authoritative
donor-side gas volume flux and a convective liquid normal-momentum flux; the
lower-front orientation must also persist in restart state. Their conservative
ALE equations and acceptance gates are recorded in
`campaign2_1d_closure_design.md`. Joint interface displacement, local-valve
integration, and full-event convergence must pass before the shared three-case
suite can be promoted.

## Complete-event OpenFOAM 2D status

- B-H1: completed refined qualification run, effective end 14.8529203 s;
  the common physical-rim audit classifies it as `GEYSER`, with the first full
  area/volume gate at 14.00 s.  This passes outcome classification only.  The
  independent result audit finds front arrival `8.50 s` versus `8.07 s`, but
  true ejection is about 46.6% late and the three trajectory speeds are
  76.5--99.7% low; it must not be described as transient quantitative
  validation.
- B-H3 baseline: completed normally at 20 s, but remained `NO GEYSER` and is
  therefore a diagnosed classification miss. Its maximum free-surface height
  was `1.501784 m` above the crown, `0.298216 m` below the physical rim.
  A whole-width VTU audit at 17.65 s found maximum exterior
  `alpha.water=2.3336e-6`, zero cells above `1e-5`, and only
  `2.0497e-12 m3/s` positive VOF-weighted flow after applying the 1 mm
  extrusion. This is numerical trace, not resolved ejection. The durable
  audit is in
  `cases/BH3_Dr26_H066_L061/openfoam/2d/complete_event/end20/results/physical_outlet_crossing_audit.md`.
- B-H6: completed normally at 20 s and accepted as `NO GEYSER`, matching the
  experiment.  The common physical-rim audit sampled 401 planes from 0 to
  20 s and found rim alpha, positive alpha-weighted flow and cumulative passed
  liquid volume all exactly zero.  Arrival is `8.04 s` versus `8.10 s`, and
  the near-hydrostatic pocket-pressure relation is supported, but the measured
  interface speeds are underpredicted by 42.8--70.7%; this is qualitative,
  not transient quantitative, agreement.  The archived metrics and run record
  are in
  `cases/BH6_Dr41_H066_L061/openfoam/2d/complete_event/end20/`.

The case-independent physical outlet reports are in
`2d_physical_outlet_audit/`.  The independent paper-parameter audit is in
`../campaign2_parameter_audit/`, and the H1/H6 result audit plus the frozen H3
metric gate is in `../campaign2_result_audit/`.  The parameter status is
`SUPPORTED_WITH_DECLARED_MODEL_CHOICES`: the shared paper scalars and axial
layout are supported, while the area-equivalent planar riser, sine-squared
Forchheimer valve path, 0.1% minimum valve aperture, laminar closure and
unreported constitutive properties are declared model choices.  In particular,
the first 4.027 ms are effectively near-closed at model resolution, not
mathematically sealed.

Do not rebuild the final synchronized HTML until the B-H3 extension and its
independent refined qualification run are classified. B-H6 no longer blocks
the final viewer.

## Dated interim progress viewer

- Artifact: `progress_latest/bh1_bh3_bh6_1d2d_progress_20260810.html`.
- Historical complete-event 1D endpoints included for progress display:
  B-H1 20 s, B-H3 23 s, B-H6 20 s.  They are rejected for the current
  Case-1-horizontal/two-fluid-vertical qualification and are not final 1D
  evidence.
- Safe OpenFOAM checkpoints included in this dated viewer: B-H1 14.80 s,
  B-H3 16.80 s, B-H6 17.40 s.
- B-H3 remains active; B-H6 has since ended normally at 20 s. This artifact is explicitly labelled
  as interim and must not be cited as final manuscript evidence.
- B-H1 uses its 13 s formal-run frame sequence plus sparse refined-run
  checkpoints after 13 s. This provenance split is recorded in
  `progress_latest/progress_viewer_manifest.json` and is acceptable only for
  progress inspection, not for a final quantitative comparison.

## H3 classification-recovery qualification run

- Directory:
  `cases/BH3_Dr26_H066_L061/openfoam/2d/qualification/h3_refined_iso_riser20/`.
- State: running from `t=0` toward 20 s in the independent scratch directory
  `/tmp/bh3-2d-qualification/h3_refined_iso_riser20`.
  Latest checked time: at least `3.008815299 s`; the complete `3.0 s`
  checkpoint is stored, and one six-rank solver is active,
  with no duplicate process, true fatal error, NaN, or normal End yet.
- Published geometry, materials, initial/boundary conditions and passive
  0.20 s valve law are unchanged.
- Numerical qualification changes: official OpenFOAM v2512 energy equation,
  isoAdvector interface transport, linear-upwind momentum convection,
  20 cells across the area-equivalent riser, 0.75 mm riser vertical spacing,
  and stricter Courant controls.
- Mesh/paper audit passed; total cells: 155100. This remains qualification
  evidence until the from-zero run ends normally and resolved water crosses
  the physical rim. No scripted outcome correction is permitted.

### Superseded H3 near-rim archive

- `cases/BH3_Dr26_H066_L061/final-from-zero/` is a genuine 115800-cell,
  from-zero isoAdvector run that ended normally at 13 s and passed its paper
  contract audit.
- Its postprocessor labelled the event `GEYSER` using the older 98%-of-rim
  threshold: maximum free-surface height 1.786715 m above the pipe crown and
  threshold-passage time 10.131788 s.
- It does **not** pass the present physical-crossing gate. The physical rim is
  1.800 m above the pipe crown, and plume probes at absolute elevations
  1.824, 1.900 and 2.100 m recorded only numerical trace values of order
  1e-8, not resolved liquid ejection. Retain this run as near-rim diagnostic
  evidence only; do not use its boolean `geysering` field as the final H3
  classification.

## Rejected or superseded 1D trials

- `case1_model_rerun/`: Case-1 Tosan horizontal core retained until tee
  arrival; B-H1 was incorrectly classified as no geyser.
- `case1_release_full_event/`: Case-1 release wave handed to the Campaign-2
  network after cap wetting; B-H1 still failed to geyser.
- `case1_full_network_complete/`: used a worktree model copy that had been
  modified by concurrent exploratory work; produced a false pre-arrival H1
  geyser and non-conservative liquid correction.
- `case1_frozen_dry_complete/`: forced the frozen core's 2% regularisation film
  to zero; produced rapid numerical filling and liquid creation in a 0.3 s gate
  test.

These directories are diagnostic evidence only and must not be used in the
paper or final viewer.
