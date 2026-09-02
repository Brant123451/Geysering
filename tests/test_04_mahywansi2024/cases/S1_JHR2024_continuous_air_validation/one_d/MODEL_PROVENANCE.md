# Model provenance and readiness audit

## Horizontal lineage

The committed, regression-frozen Case-1 horizontal seed is:

- `tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/model/tosan2021_horizontal_shockfit.py`
  (SHA-256 `90E84DA9AFA0EC8465D80F87FC701DFB8F0FAD6F97350EA708074A50192B6119`);
- `tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/model/casea_shockfit_network.py`
  (SHA-256 `1B24C90C3DC997F0E17CFBEF9A720DB92510C14B86E42DF3902EA2AE8E6061B3`).

The seed provides the Case-1 circular-pipe liquid state and pressure law,
elastic full-pipe branch, MUSCL/hydrostatic reconstruction, conservative
moving-interface balance, SSP-RK2 update and donor draining limiter. Its gas
representation is one closed, spatially uniform, finite pocket. The
valve-release shock fit and finite-pocket topology are not applicable to
Mahyawansi's continuously supplied side-air inlet.

The current S1 two-T component divides the horizontal main into the three
physical segments delimited by the two T faces.  Each segment calls the
hash-pinned seed's `_muscl_free_surface_face_states`,
`_central_upwind_flux`, and `_apply_donor_draining_limiter` directly.  At a T
endpoint the independently solved atomic node flux replaces only that endpoint
face; a resolved gas nose is likewise an explicit material-interface face.
The whole-network driver owns the same two-stage SSP-RK composition, so the
component returns one Case-1 forward-Euler spatial stage rather than nesting a
second time integrator.  Its evidence class is therefore
`hash-pinned Case1 spatial seed reused with two atomic T endpoints`, not the
former first-order-derived stencil.

The self-contained Case-1 application entry point is
`vw2011_network_twofluid.py`, but that file and its horizontal/gas dependencies
currently contain large uncommitted changes.  They are not silently imported
or described as a frozen production dependency here.  The exact claim is
reuse of the two immutable hashes listed above, not wholesale reuse of the
mutable Case-1 application loop or its apparatus-specific gas topology.

Table 1 gives different Fluent water-boundary types: the upstream pressure
inlet supplies total head, whereas the downstream pressure outlet supplies
static pressure/head. Both retain the outgoing Case-1 characteristic. The
inlet ghost closes `H + u^2/(2g)`; the outlet ghost closes `H` and never adds
kinetic head to the 0.584 m datum. That outlet head is the declared elastic
storage zero, and the 0.5842 m Stage-1 field is initialized as a small positive
elastic increment with a round-trip head test. The real six-port two-node
owner now evaluates this component together with the supply and riser and
returns one atomic stage rate. The trajectory remains non-production because
persistent atmospheric-plume, generalized topology and water-end phase
boundaries are still incomplete.

The current `casea_coupled_gas_network.py` is modified in the working tree and
is not a frozen dependency.  It may be consulted for equations and regression
ideas, but no accepted Case-3 result may depend silently on that mutable file.

## Vertical two-fluid lineage

The available Case-1 components implement useful pieces:

- persistent `VerticalTwoStreamState(A_up,Q_up,A_down,Q_down)`;
- shared-face finite-volume transport and conservation ledgers;
- coaxial core/film geometry and a common gas/liquid pressure;
- atmospheric donor outflow;
- dynamic net/circulation mouth states and equal-and-opposite three-body drag.

They are component-level evidence only.  Their own readiness flags state that
the complete Case-1 riser/network is not ready.  The last conservative Case-1
preflight was `blocked_fail_closed`, and a later full 1-D/2-D audit failed
vertical inventory, mouth-flux and gas-Mach gates.  Case-1's fixed Taylor-core
fraction, rise-speed floor and Wallis constants are apparatus-specific and
must not be transplanted to force the Mahyawansi eruption.

## Required new coupled owner

At every global predictor/corrector stage, one owner must advance:

```text
horizontal: A_l, Q_l, M_g, J_g
air branch: A_l, Q_l, M_g, J_g
vertical:   A_up, Q_up, A_down, Q_down, M_g, J_g
nodes:      common pressure, gas-area fraction, gross port packets, wall reaction
```

The two T nodes have zero material storage; their trial variables and port
packets are algebraic stage data, not persistent liquid or gas inventory.

The same predicted state supplies west/east/vertical traces.  One common-node
solve returns a three-face atomic packet containing gas mass, gas momentum,
liquid volume and liquid momentum fluxes.  That packet is committed once to
all adjacent cells.  The signed vertical flux is decomposed into gross upward
and downward liquid streams without reconstructing them from net flow.

Legacy characteristic mouth updates, Taylor-return mass sources,
CCFL-on-signed-net-flow and net-only horizontal side sources are mutually
exclusive with this owner and therefore forbidden.

## Promotion rule

The current status is architecture and acceptance definition only.  It cannot
be called a production model until unit conservation tests, source-aligned
initialization, a long stable trajectory, the eruption branch, horizontal
gas/slug transport and vertical two-fluid histories all pass against the
refined 2-D calculation without hidden tuning.

## Implemented, still non-production adapters

- `model/horizontal_case1_adapter.py` enforces both horizontal source hashes,
  exposes the pinned circular liquid geometry/physical flux and exact
  MUSCL/central-upwind/draining functions, maps elastic Case-1 pressure force
  to a physical T-port aperture, solves dynamic total-pressure ghosts, freezes
  the S1 geometry and rejects finite-pocket or valve-release initialization.
- `model/vertical_case1_adapter.py` enforces the reviewed local-component
  hashes/readiness flags, preserves the Case-1 signed downward stream only as
  a component view, and explicitly converts it to the S1 non-negative gross
  `Qdown` contract without reconstructing from net flow.
- `model/pressure_reservoir.py` implements a declared isothermal HLL
  translation of the published 5700 Pa pressure inlet. It permits physical
  pressure-driven inflow/backflow and does not claim to reproduce the
  unreported experimental valve or supply losses.
- `model/initialization.py` assembles the full-water horizontal pipe, exact
  z=0.5842 m riser cut cell, atmospheric riser gas and two distinct
  zero-storage algebraic nodes into a validated partial Stage-1 state.  The
  method-inferred water-filled supply branch remains a required state owner.
- `model/horizontal_distributed.py` is a conservation-tested gas-only branch
  component, but its own readiness gate rejects a source-aligned trajectory;
  it cannot replace the required initially water-filled two-phase branch.
- `model/vertical_twostream_solver.py` advances persistent directional liquid
  and gas component states using pinned Case-1 operators, but correctly
  rejects newly created gas voids until the global gas/void Riemann remap is
  supplied.
- `model/vertical_pressure_void_component.py` supplies that atomic local
  remap when the same RK-stage riser-bottom port carries an explicit gas
  parcel.  The parcel's mass and axial momentum seed only the bottom-connected
  new void and are removed from the later boundary transport to prevent
  double injection; a source-free isolated void still rejects.  When the
  source-aligned bottom cell is liquid-full, the parcel's EOS volume at the
  common node pressure opens a bottom cut cell through a conservative
  finite-volume piston flux and deposits exactly the displaced liquid volume
  at the resolved z=0.5842 m free-surface cut cell.  The piston rejects an
  internal gas gap or isolated liquid column instead of merging across it.
  A single liquid label may reverse by a per-cell area/momentum-conservative
  direction remap, whereas two existing labels separated by finite gas remain
  fail-closed.  The component momentum audit is assembled from pre-final
  FV/boundary/body/wall terms and rejects a missing bottom pressure traction
  or any unbudgeted perturbation.  The atmospheric rim additionally accepts a
  finite explicit exterior liquid parcel for one-stage re-entry and limits it
  by demand, area-speed capacity and parcel inventory.  No persistent
  exterior-plume owner yet stores prior outflow or advances parcel depletion,
  so repeated-cycle fallback and the complete component remain non-production.

These adapters/components pass structural and conservation tests. The
water-filled supply branch can displace water in its isolated component step,
but it and the vertical component still lack the common pure-trial adapter and
the global zero-storage-node/riser predictor-corrector that assembles one production
`JointStageRate`; consequently no production trajectory is provided.
