# Coupled 1-D model for the Mahyawansi JHR 2024 validation case

This directory is independent of the OpenFOAM mesh-level directories.  It is
the only place where the new Case-3 one-dimensional model and its outputs may
be created.  Case 1 is a read-only source of tested operators and is never
modified by this work.

## Campaign identity

The current OpenFOAM campaign contains **one physical condition** (the
continuous-air experiment) evaluated on three numerical mesh levels:
`coarse`, `medium_refine`, and `refined`.  Those are three 2-D calculations,
not three experimental conditions.  One source-aligned 1-D calculation is
therefore compared at common physical times with all three meshes; it must not
be duplicated and presented as three independent physical cases.

## Mandatory model architecture

1. The horizontal branch must retain the applicable Case-1 circular-pipe
   liquid algorithm while extending the state to conservative distributed gas
   mass and momentum. The current two-T implementation splits the main into
   three segments and calls the hash-pinned Case-1 MUSCL central-upwind and
   donor-draining functions on every segment. Each T endpoint is replaced by
   its atomic node flux; a resolved gas nose is an explicit material-interface
   face. The Case-1 valve-release shock-fitting initial stage is not applicable
   to a continuous side-air pressure inlet and is not transplanted.
2. The 0.1373 m branch below the air-supply boundary is water-filled in the
   Stage-1 baseline and must own distributed `(Al,Ql,Mg,Jg)`.  At Stage 2 its
   top becomes the continuous 5700 Pa, pure-air pressure boundary.  It is not
   a prescribed finite gas pocket and it must not use the separate `0.06 m/s`
   finite-pocket initial condition from the source paper.
3. The riser owns persistent directional liquid states
   `(A_up, Q_up, A_down, Q_down)` plus resolved gas inventory and momentum.
   Reconstructing the two streams from their signed net flow after each step
   is forbidden because it erases counter-current motion.
4. Each T junction has one and only one mass/momentum owner.  The horizontal
   cells, supply/vertical gross streams, zero-storage node port packet and gas
   source must be advanced in one conservative transaction; a T node may not
   acquire material inventory and a second side-source update is forbidden.
5. Paper inputs are frozen before comparison.  Missing paper inputs are
   reported as missing or declared numerical choices and are never tuned to
   force an eruption or a target curve.

## Validation order

The evidence chain is strictly ordered:

1. Paper inputs and physical branch -> each 2-D mesh.
2. Refined 2-D event and internal histories -> 1-D model.
3. Coarse/medium/refined spread -> numerical uncertainty, not three physical
   calibrations.

A numerically stable result that does not erupt is a physics-alignment
failure.  A 1-D run cannot be accepted merely because it matches pressure if
the horizontal gas/slug transport or the riser eruption branch is wrong.

The mouth-above-rim external plume is inside the 2-D domain but outside the
1-D pipe network.  The 1-D model directly validates mouth wetting, liquid
outflow, eruption onset/duration, pressure, and internal phase motion.  Any
ballistic height inferred from mouth flux is labelled `derived_plume_proxy`
and is not described as a directly resolved two-fluid height.

## Current status

The source/acceptance contract is frozen in `config/S1_source_aligned.yaml`
and `ACCEPTANCE.yaml`. Pinned Case-1 component adapters, the declared 5700 Pa
pressure-reservoir boundary, the exact main-pipe/riser Stage-1 water/air
initialization, two-node atomic state contracts and comparison tools are
implemented and tested. The horizontal component now preserves elastic
Case-1 pressure force at T traces and transports a one-sided gas nose with
same-stage liquid displacement, gas mass and gas momentum. It also uses
the published total head at the Table-1 pressure inlet and the published
static head at the pressure outlet, each with the outgoing characteristic.
The formal initializer now stores the exact 0.5842 m state as a small
positive elastic increment above the declared 0.584 m storage zero and checks
its Case-1 head round trip. The riser now conservatively opens the source cut-cell
path for a same-stage Stage-2 bottom gas parcel and can budget one finite
returning-rim liquid parcel, while retaining all six directional two-fluid
states. The four-unknown two-node kernel is only a component-trial algebraic
gate, but it is now driven by one real same-RK-stage owner that builds all six
physical traces, calls the actual horizontal/supply/riser proposals and
returns one conservative `JointStageRate`. A source-initial two-RK microstep,
cross-T gas-nose transfer and Newton-failure rollback pass. This does not make
the model production-ready: persistent exterior-plume/re-entry ownership,
generalized topologies, water-end gas re-entry and canonical outputs remain
required. See `RUN_STATUS.md`.
