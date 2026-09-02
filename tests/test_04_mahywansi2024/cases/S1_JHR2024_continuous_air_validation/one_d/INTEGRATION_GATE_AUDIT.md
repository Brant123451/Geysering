# S1 1-D integration-gate audit

Updated: 2026-08-10

Scope: independent code/configuration audit of `one_d/`. One source-initial
`1.0e-7 s` two-RK physical-owner microstep was run only as an integration
gate. No Stage-1/Stage-2 trajectory or 1-D/2-D agreement is claimed.

## Gate result

| Gate | Result | Evidence and limitation |
|---|---|---|
| Case-1 horizontal lineage | component gate passed | The two declared Case-1 files match the frozen SHA-256 values. The S1 adapter loads the verified seed and directly calls its MUSCL reconstruction, central-upwind flux and donor-draining limiter on three segments separated by the two physical T faces. Case-1 finite-pocket and valve-release topology remain forbidden. |
| Source-aligned horizontal initial state | component gate passed | The common initializer writes the 0.5842 m state as a positive elastic increment above the declared 0.584 m storage reference and tests the Case-1 head round trip. |
| Table-1 water-boundary semantics | component gate passed | The pressure inlet uses total head with the outgoing characteristic; the pressure outlet uses static head with the outgoing characteristic. The outlet does not receive a kinetic-head term. |
| Distributed horizontal gas motion | component gate passed only | `Al/Ql/Mg/Jg`, first gas entry and one-sided gas-nose transfer are conservative in component tests. No accepted network trajectory or horizontal slug observer exists yet. |
| Vertical two-fluid state | component gate passed | The riser persistently owns `Aup/Qup/Adown/Qdown/Mg/Jg`; gross streams are not reconstructed from net flow. The source cut cell, same-stage bottom gas parcel and independent mass/momentum audits pass component tests. |
| Two T nodes | integration-owner gate passed | One owner constructs all six physical traces/acoustic scales/directional seeds/interface records from the same immutable RK state, calls the four-unknown simultaneous node kernel with the real horizontal/supply/vertical proposals and returns one `JointStageRate`. Global production remains false for independent top/topology/end/output gates. |
| Supply-branch F0 wall closure | component and joint-trial gate passed | The source-aligned water-filled branch applies the frozen smooth-pipe Darcy law to both phase plugs with sign-preserving semi-implicit relaxation. Its real pressure-reservoir/piston trial now supplies the accepted bottom packet and explicit external momentum ledger to the common owner. |
| Conservation | owner microstep gate passed; trajectory pending | Six-port node balances, per-component ledgers, separate Px/Pz ledgers, one two-RK final commit and Newton-failure atomic rollback pass. Only one `1.0e-7 s` source-initial microstep was executed; whole-trajectory conservation is untested. |
| 1-D/2-D output compatibility | contract only | Scalar names, unshifted 0.10 s grid, exact eruption-branch rule and all six riser profile fields are frozen. There is no production state-to-observable/export pipeline for gas/slug positions, P1/P2/P3 or riser profiles. |
| Production authorization | correctly blocked | Every current physical component and the node kernel keep `production_ready=false`; the default physical runner fails closed. |

## Verification performed

- full test suite: `186 passed`;
- Python compile check: passed for `model`, `alignment` and `tests`;
- YAML parse check: passed for `ACCEPTANCE.yaml` and all three configuration files;
- scalar acceptance fields are a subset of `COMMON_OBSERVABLES.yaml`;
- required vertical profiles explicitly contain all six prognostic riser states
  plus derived gas area and velocity;
- both horizontal Case-1 source hashes and the vertical Case-1 pin audit pass.

## Remaining hard blockers

1. Add persistent exterior-plume inventory for repeated rim outflow/re-entry;
   retain the current finite one-stage parcel only as a tested substep.
2. Resolve or explicitly keep fail-closed the generalized bottom-piston/top-
   spill topology and Table-1 water-end phase re-entry.
3. Add a pure canonical sampler/exporter for horizontal connected gas/slug
   motion, P1/P2/P3 and native-cell `Aup/Qup/Adown/Qdown/Mg/Jg` profiles.
4. Only after 1--3 pass: run the full physical 0.02 s smoke, Stage-1 settling, then
   one frozen Stage-2 trajectory. Compare it to the refined 2-D result first;
   do not change unpublished parameters to force eruption.

The minimum next implementation step is item 1. The real coupling owner is no
longer the blocker, but its microstep cannot be promoted into a trajectory
while persistent exterior material can leave/re-enter without a state owner.
