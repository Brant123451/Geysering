# Contract-only coupled 1-D core

Status: **non-production; physical closure intentionally absent**.

This package fixes the coupling invariants while the Case-1 horizontal
operator and riser two-fluid closures are adapted. The current horizontal
component calls the hash-pinned Case-1 MUSCL central-upwind and donor-draining
spatial functions on three segments separated by the two physical T nodes.
This is direct reuse of that reviewed spatial seed, but not a claim that the
mutable full `vw2011_network_twofluid.py` application loop is embedded:

- the horizontal owner stores `Al, Ql, Mg, Jg`;
- the riser owner stores persistent `Aup, Qup, Adown, Qdown, Mg, Jg`;
- one `AtomicFluxPacket` advances horizontal, supply and riser cells while
  applying the air-supply T-node packet at `x=-1.52 m`, the riser T-node packet
  at `x=0`, and external exchanges together; both nodes remain zero-storage
  algebraic junctions and neither has a second update path;
- every accepted packet has an explicit liquid-volume, gas-mass, and
  mixture-momentum ledger entry;
- gross upward and downward mouth streams remain distinct during eruption
  event integration; an event additionally requires water connectivity to the
  mouth, a continuous 0.10 s window, mean outflow of at least
  `3.2175924923e-5 m3/s`, and window volume of at least
  `3.2175924923e-6 m3`;
- 5700 Pa remains a continuous pressure boundary. It is not converted to a
  guessed gas flow while the valve/line/tank closure is missing.

`CoupledStepper` therefore raises `MissingPhysicalClosure` by default. Passing
the structural tests means only that these contracts are enforced; it is not
evidence of a stable solution, an eruption, or agreement with the paper.

The package also contains fail-closed component adapters:

- `horizontal_case1_adapter.py` verifies the pinned Case-1 source, exposes its
  circular-pipe geometry/flux, MUSCL reconstruction, central-upwind flux and
  donor limiter, supplies the conservative elastic T-port pressure-force
  mapping, the Table-1 total-head inlet characteristic ghost and static-head
  outlet characteristic ghost, and
  constructs the S1 full-water state;
- `vertical_case1_adapter.py` verifies locally ready Case-1 two-stream pieces,
  converts their signed downward convention explicitly, and constructs the
  exact S1 water/air initial column without enabling a trajectory;
- `vertical_pressure_void_component.py` keeps the six riser states persistent,
  admits a same-stage bottom gas parcel once, opens the source-aligned
  liquid-full bottom through a conservative piston remap to the z=0.5842 m
  cut cell, and audits liquid/gas/momentum budgets independently. It consumes
  only finite state-owned returning-rim parcels and can launch re-entry from a
  zero interior `Qdown` trace;
- `atmospheric_exterior_plume.py` persistently owns gross airborne and
  returning liquid volume/momentum, atomically relabels a falling lump at the
  rim, and has two-cycle, rollback and single-commit conservation tests. Its
  first-moment height is a declared zero-dimensional proxy, not a resolved
  external free surface or a paper-fitted eruption height;
- `pressure_reservoir.py` translates the published 5700 Pa boundary into a
  declared isothermal HLL reservoir flux without claiming to model the
  experimental valve;
- `supply_branch_twophase.py` now exposes its real lower trace and evaluates
  each common-node pressure trial through the existing finite pressure-
  reservoir/piston component, including the frozen Darcy wall ledger;
- `physical_joint_owner.py` builds all six same-state traces/acoustic scales,
  preserves the riser's gross directional seeds, calls the simultaneous
  four-unknown two-node solve, and returns one unique physical
  `JointStageRate`. It is validation-only while generalized bottom/top phase
  topology, water-end phase re-entry, canonical export and result gates remain
  open;
- `initialization.py` assembles both pipes and both zero-storage nodes into one
  validated Stage-1 state.
