# One-dimensional coupled-model status

Updated: 2026-08-10

## Current classification

`PREREGISTERED_F0__REAL_SIX_PORT_OWNER_PASS__TOP_TOPOLOGY_OUTPUT_PENDING`
- no one-dimensional physical trajectory or 1-D/2-D comparison has been
accepted.

The result-independent baseline closure set is frozen in
`config/S1_1D_F0_closures.yaml` with status
`preregistered_implementation_pending`. This freezes choices; it is not an
authorization to run a physical trajectory.

Completed:

- source-aligned geometry, boundary and initial-phase matrix;
- explicit separation of one physical condition from three 2-D mesh levels;
- Case-1 horizontal and vertical component provenance audit;
- joint paper -> 2-D -> 1-D acceptance contract;
- fixed internal mouth-event flow/volume threshold and common 0.10 s time grid;
- immutable horizontal, supply-branch and vertical state ownership plus two
  zero-storage T-node port owners, with atomic commit and separate Px/Pz
  conservation ledgers;
- pinned Case-1 circular horizontal liquid geometry, conservative pressure
  law, MUSCL central-upwind face kernel and donor-draining limiter while
  blocking its finite-pocket and valve-release topology;
- pinned persistent Case-1-derived vertical state, explicit signed-flow
  conversion, and exact z=0.5842 m water/air cut-cell initialization;
- declared 5700 Pa isothermal pressure-reservoir Riemann boundary;
- source-aligned Stage-1 assembly in which the finite 0.1373 m supply branch is
  explicitly owned as 14 initially water-filled `Al/Ql/Mg/Jg` cells and is
  included in the liquid inventory;
- conservative two-phase supply-branch component: Stage 1 wall, Stage 2 pure
  air at 5700 Pa gauge, no seeded pocket and no massless void;
- preregistered smooth-pipe Darcy wall shear in the supply branch, using the
  same sign-preserving semi-implicit relaxation as the riser, with liquid,
  gas and total wall impulses recorded in the component momentum ledger;
- full-network SSP-RK2 atomic orchestration framework: horizontal main, supply
  branch, riser and both zero-storage T nodes are evaluated at both RK stages,
  only one final averaged packet can commit, and any predictor/stage/node/
  admissibility/conservation failure rolls the entire transaction back;
- one real same-RK-state six-port owner: four Case-1 main traces, the finite
  supply-bottom trace and the persistent six-state riser-bottom trace are sent
  together to the four-unknown two-node solver; the three accepted physical
  proposals are assembled into exactly one conservative `JointStageRate`;
- source-initial Stage-1 owner validation through both RK stages for one
  `1.0e-7 s` atomic microstep, including six-port/node/component/global
  conservation and one final commit; manufactured Newton failure leaves the
  state and ledger unchanged. This is not the blocked full `0.02 s` physical
  smoke or a trajectory result;
- topology-derived planar gas-nose ownership and same-stage elastic release,
  void opening, gas mass and liquid displacement when gas first crosses the
  air T into either full-water Case-1 main segment;
- per-component and per-port phase ledgers, node residual checks, and explicit
  vector T-node wall reactions in separate Px/Pz ledgers;
- a Case-1-derived horizontal F0 component with both physical T faces, four
  independent horizontal port traces, distributed `Al/Ql/Mg/Jg`, dynamic
  smooth-pipe wall/interfacial exchange, paired first gas entry and atomic
  whole-proposal rollback;
- full/elastic `Al >= Af` pressure-force mapping into every horizontal T trace,
  without clamping `Al > Af` to a full-area atmospheric/reference trace;
- bidirectional one-sided gas-nose propagation into an originally full-water
  neighbor with same-stage liquid displacement, gas mass, gas momentum and
  whole-proposal conservation/rollback; massless void creation remains
  fail-closed;
- three hash-pinned Case-1 liquid segments separated by the two physical T
  faces; only each segment endpoint is replaced by its atomic node flux, while
  the whole-network SSP-RK2 driver owns both Case-1 forward-Euler stages;
- source-semantic water ghosts at both Table-1 boundaries, retaining the
  outgoing Case-1 characteristic while closing upstream total head and
  downstream static head rather than treating both as one boundary type;
- tangent matching of the formal S1 horizontal pressure law with the frozen 2-D
  OpenFOAM `perfectFluid` tangent, `c=sqrt(3000*293.15)=937.789955 m/s`, while
  retaining the Case-1 circular geometry and conservative pressure law; this
  is not described as full thermodynamic EOS equivalence;
- formal initialization of the published 0.5842 m horizontal water level as a
  small positive elastic increment above the declared 0.584 m storage zero,
  with a round-trip head check in the Case-1 circular pressure law;
- a persistent vertical pressure/void component with independent
  `Aup/Qup/Adown/Qdown/Mg/Jg`, conservative void remapping, local dynamic wall
  friction, exact three-body recoil and bidirectional atmospheric gas Riemann
  exchange; a newly opened isolated bottom void is accepted only when the
  same immutable RK trial supplies an explicit gas parcel, and that parcel is
  remapped once rather than injected again by the subsequent transport step;
- the source z=0.5842 m cut-cell transition: a single liquid label may reverse
  through the pinned per-cell area/momentum-conservative direction remap, but
  two pre-existing liquid labels still cannot merge across finite gas; when a
  same-stage gas parcel reaches a liquid-full riser bottom, its node-pressure
  EOS volume is opened by a conservative bottom-to-free-surface piston flux,
  with internal gas gaps and isolated upper liquid columns rejected;
- a bounded one-stage atmospheric liquid re-entry transaction: an explicit
  finite exterior parcel is limited by the falling-liquid demand, area-speed
  capacity and available volume, and its consumed volume and momentum enter
  the same external ledgers; this is not yet a repeated-cycle exterior owner;
- an independent vertical momentum gate assembled before the final state from
  accepted top/bottom advection, common-pressure work, liquid/gas gravity,
  wall shear and internal recoil audits; the former practice of defining an
  external force from the final momentum difference has been removed, and a
  bottom port missing its pressure traction now fails closed;
- a declared planar-2-D capillary translation matching the frozen
  `zeroGradient` alpha walls: flat stationary source surface, topology-fixed
  semicircular moving caps with `|kappa|=2/D`, and no guessed contact angle;
- structural-zero 0.02 s transaction validation, which is only an orchestration
  test and is not a physical water-settling, gas-injection or eruption result;
- 32 targeted vertical adapter/two-stream/pressure-void tests passing after the
  cut-cell and finite-rim-parcel changes, 70 targeted horizontal/supply/joint/
  integration-gate tests passing after the Case-1 operator, wall-closure and
  pressure-boundary/physical-owner work, and the complete one-dimensional suite passing 186
  tests on 2026-08-10.

Not completed - all remain fail-closed:

- Table-1 `inletOutlet` phase re-entry if resolved gas ever reaches either
  water end cell; the current inlet-total/outlet-static characteristic
  implementation fails closed outside its water-end scope;
- a persistent exterior-plume owner for the atmospheric riser top, including
  atomic storage of prior rim outflow, ballistic/falling-state evolution and
  depletion by later re-entry; the implemented finite one-stage parcel cannot
  by itself close repeated cycles;
- generalized bottom-gas piston motion through arbitrary internal gas/liquid
  topology and simultaneous top spill; the source-aligned contiguous-column
  cut-cell path is implemented, while internal gaps remain fail-closed;
- circular-3-D capillarity remains fail-closed because the paper does not
  publish a contact angle; only the declared planar-2-D comparison mode is
  currently authorized at component level;
- physical 0.02 s smoke, Stage-1 settling, Stage-2 injection, long run and any
  eruption, gas-pocket/slug, pressure, vertical-phase or conservation
  comparison against refined 2-D.
- the end-to-end canonical observer/exporter for horizontal gas/slug motion,
  all six vertical profiles and P1/P2/P3 pressure histories. The names and
  file contracts are frozen, but no production state-to-output pipeline exists.

## Hard blockers inherited from the audit

The Case-1 frozen horizontal seed contains a finite lumped gas pocket rather
than a continuous pressure source with distributed gas momentum. Its verified
circular geometry, pressure law, MUSCL central-upwind and draining functions
are retained on three S1 segments; its finite-pocket topology is not.  The
self-contained `vw2011_network_twofluid.py` application and two horizontal/gas
dependencies currently contain large uncommitted changes, so this work claims
only the two immutable seed hashes recorded in `MODEL_PROVENANCE.md`, not a
silent import of that mutable application loop. The valve release, fixed
friction, fitted node inertance and Case-1 vertical empirical constants remain
blocked.

The joint atomic framework, real horizontal/supply/vertical proposals,
four-unknown algebraic two-node kernel and six-port `JointStageRate` owner are
integrated and pass a source-initial atomic microstep. The current operator
still fails the global production gate because the persistent repeated-cycle
exterior state, generalized phase topology, water-end gas re-entry and output
pipeline remain unresolved. Planar 2-D capillarity, the liquid EOS mapping,
the source cut-cell piston path and a finite one-stage returning parcel are
frozen. Existing historical, isolated-component, structural-zero or owner-
microstep output cannot be renamed as an S1 result.

The model may advance to a physical smoke only after these closures are
implemented from the preregistered F0 rules or independently documented
physics. They may not be adjusted after viewing eruption or 2-D comparison
results merely to force the expected branch.

## Next promotion gates

1. `MODEL_IMPLEMENTATION_ACCEPTED`: every F0 closure is implemented; the
   integrated two-T-node owner, capillarity, EOS tangent-matching audit,
   complete persistent atmospheric top boundary, generalized phase topology
   and canonical output sampler pass positivity, phase routing and
   conservation tests.
2. `RUN_COMPLETE_UNVALIDATED`: one common frozen F0 parameter set completes
   the planned Stage-1 and Stage-2 windows without numerical failure.
3. `ERUPTION_ACCEPTED`: the fixed internal mouth event passes and the refined
   2-D external plume event also passes.
4. `RESULT_ACCEPTED`: horizontal gas/slug motion, riser two-fluid motion,
   P1/P2/P3, eruption timing/duration and conservation all pass the frozen
   comparison.

The passing tests support component, contract and physical-owner integration
claims only. They do not satisfy any physical-result promotion gate.
