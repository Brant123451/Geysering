"""Fail-closed network adapter for the Case-A two-stream riser.

This module is deliberately outside :mod:`vw2011_network_twofluid`.  It
connects three already isolated pieces without changing their ownership:

* the finite compressible T node owns the signed mouth rate ``q_net``;
* the two-channel mouth closure decomposes that rate into simultaneous gross
  upward and downward liquid rates; and
* the vertical finite-volume operator persists the two liquid areas and
  momenta in the riser.

The adapter never changes ``q_net`` and never prescribes a water height.  It
also does not pretend that updating the vertical branch completes a network
step: west/east/vertical gas and liquid face fluxes from the finite-node
transaction still have to be committed atomically to all adjacent branch
cells by the global RK stage.  Until that owner replacement is made in the
main loop, ``COMPLETE_CASEA_NETWORK_READY`` remains false.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from casea_finite_node_qnet_owner import FiniteNodeQnetTransaction
from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (
    LegacyMouthPathActivity,
    TwoChannelMouthCouplingPlan,
    stage_from_finite_node_ssprk2,
)
from casea_vertical_twostream_fv import (
    DirectionalBoundaryFlux,
    PhysicalGasInterphaseState,
    PhysicalThreeBodyDragResult,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    VerticalTwoStreamStepResult,
    advance_vertical_two_stream_fv,
    implicit_physical_three_body_drag_exchange,
)


TWOSTREAM_NETWORK_ADAPTER_READY = True
COMPLETE_CASEA_NETWORK_READY = False
GLOBAL_INTEGRATION_BLOCKERS = (
    "recompute_west_east_vertical_branch_traces_at_both_ssprk_stages",
    "commit_all_finite_node_face_fluxes_to_adjacent_branch_cells_atomically",
    "replace_legacy_G1_taylor_ccfl_and_distributed_side_source_owners",
    "advance_gas_pressure_and_three_body_drag_in_the_same_global_rk_state",
)


class TwoStreamNetworkAdapterError(RuntimeError):
    """A local stage violates the ownership or conservation contract."""


@dataclass(frozen=True)
class CaseATwoStreamRiserStage:
    """One accepted vertical stage driven by one finite-node transaction."""

    state: VerticalTwoStreamState
    mouth: TwoChannelMouthCouplingPlan
    transport: VerticalTwoStreamStepResult
    physical_drag: PhysicalThreeBodyDragResult | None
    q_net_owner_value: float
    q_net_boundary_value: float
    q_net_residual: float
    riser_liquid_volume_before: float
    riser_liquid_volume_after: float
    expected_riser_volume_change: float
    riser_volume_residual: float
    global_branch_commit_pending: bool = True

    @property
    def gross_upward_rate(self) -> float:
        return self.mouth.exchange.upward_flow

    @property
    def gross_downward_rate(self) -> float:
        return self.mouth.exchange.downward_flow


def natural_open_top_boundary(
    state: VerticalTwoStreamState,
    *,
    dry_area_tolerance: float = 1.0e-14,
) -> DirectionalBoundaryFlux:
    """Use the last-cell upward characteristic as an unforced top outflow.

    Liquid may leave an atmospheric open top, but no liquid reservoir is
    imposed above it.  A falling film already inside the riser therefore has
    zero prescribed inflow at the top.  For Case A the surface remains below
    the top, so this closure normally evaluates to zero without a time switch.
    """

    tolerance = float(dry_area_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("dry-area tolerance must be finite and non-negative")
    area = float(state.upward_area[-1])
    rate = max(float(state.upward_discharge[-1]), 0.0)
    if area <= tolerance or rate == 0.0:
        return DirectionalBoundaryFlux()
    return DirectionalBoundaryFlux(
        upward_rate=rate,
        upward_speed=rate / area,
    )


def _riser_volume(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
) -> float:
    return parameters.cell_length * math.fsum(state.liquid_area)


def advance_casea_twostream_riser_from_finite_node(
    state: VerticalTwoStreamState,
    transaction: FiniteNodeQnetTransaction,
    parameters: VerticalTwoStreamParameters,
    *,
    pressure_faces: Iterable[float],
    phase: VerticalMouthPhaseState,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    losses: DirectionalMouthLosses,
    physical_gas: PhysicalGasInterphaseState | None = None,
    top_boundary: DirectionalBoundaryFlux | None = None,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> CaseATwoStreamRiserStage:
    """Advance the persistent two-stream riser using the node-owned ``q_net``.

    The finite-node result has already advanced its own inventory.  This
    function applies its vertical shared-face rate once to the riser.  The
    mouth closure creates gross counter-current rates without altering that
    signed rate.  Physical gas drag, when supplied, is an operator-split
    three-body exchange whose opposite gas impulse is returned to the caller;
    the caller must consume it in the gas-momentum stage.

    This local operation cannot certify a complete network commit.  Its result
    therefore remains marked ``global_branch_commit_pending`` until the
    global integrator has written every finite-node face flux to the west,
    east, and vertical neighbour residuals.
    """

    if state.cell_count != parameters.cell_count:
        raise ValueError("two-stream state and riser parameters disagree")
    dt = float(transaction.result.ledger.dt)
    q_net = float(transaction.q_net)
    result_q_net = float(transaction.result.vertical.liquid_area)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("finite-node transaction needs a positive finite dt")
    tolerance = 256.0 * math.ulp(max(abs(q_net), abs(result_q_net), 1.0))
    if not math.isclose(q_net, result_q_net, rel_tol=0.0, abs_tol=tolerance):
        raise TwoStreamNetworkAdapterError(
            "transaction q_net differs from its vertical shared-face flux"
        )
    mouth_area = float(state.liquid_area[0])
    if not math.isclose(
        float(phase.liquid_area),
        mouth_area,
        rel_tol=2.0e-10,
        abs_tol=max(parameters.packing_tolerance, 2.0e-14),
    ):
        raise TwoStreamNetworkAdapterError(
            "mouth phase state does not match the first two-stream riser cell"
        )
    if not math.isclose(
        geometry.full_area,
        parameters.full_area,
        rel_tol=2.0e-12,
        abs_tol=2.0e-14,
    ):
        raise TwoStreamNetworkAdapterError(
            "mouth and riser cross-sectional areas disagree"
        )

    volume_before = _riser_volume(state, parameters)
    plan = stage_from_finite_node_ssprk2(
        transaction.result,
        phase=phase,
        geometry=geometry,
        material=material,
        wallis=wallis,
        riser_liquid_donor_volume=volume_before,
        losses=losses,
        legacy_activity=legacy_activity,
    )
    bottom = DirectionalBoundaryFlux(
        upward_rate=plan.exchange.upward_flow,
        upward_speed=plan.exchange.upward_channel_velocity,
        downward_rate=plan.exchange.downward_flow,
        downward_speed=abs(plan.exchange.downward_channel_velocity),
    )
    top = (
        natural_open_top_boundary(
            state,
            dry_area_tolerance=parameters.dry_area_tolerance,
        )
        if top_boundary is None
        else top_boundary
    )
    transport = advance_vertical_two_stream_fv(
        state,
        parameters,
        dt=dt,
        pressure_faces=pressure_faces,
        boundaries=VerticalTwoStreamBoundaries(bottom=bottom, top=top),
    )
    drag_result = None
    final_state = transport.state
    if physical_gas is not None:
        drag_result = implicit_physical_three_body_drag_exchange(
            final_state,
            parameters,
            physical_gas,
            dt=dt,
        )
        final_state = drag_result.state

    volume_after = _riser_volume(final_state, parameters)
    expected_change = dt * (bottom.net_rate - top.net_rate)
    volume_residual = volume_after - volume_before - expected_change
    scale = max(volume_before, volume_after, abs(expected_change), 1.0e-15)
    conservation_tolerance = 2048.0 * math.ulp(scale)
    if abs(volume_residual) > conservation_tolerance:
        raise TwoStreamNetworkAdapterError(
            "two-stream riser volume ledger does not close"
        )
    q_residual = bottom.net_rate - q_net
    if abs(q_residual) > tolerance:
        raise TwoStreamNetworkAdapterError(
            "gross mouth decomposition changed the finite-node q_net"
        )
    if drag_result is not None:
        drag_scale = max(
            *(abs(value) for value in drag_result.gas_impulse),
            *(abs(value) for value in drag_result.upward_liquid_impulse),
            *(abs(value) for value in drag_result.downward_liquid_impulse),
            1.0e-15,
        )
        drag_tolerance = max(
            2048.0 * math.ulp(drag_scale),
            1.0e-12 * drag_scale,
            1.0e-18,
        )
        if any(
            abs(value) > drag_tolerance
            for value in drag_result.cell_momentum_residual
        ):
            raise TwoStreamNetworkAdapterError(
                "three-body gas/liquid momentum exchange does not close"
            )
    return CaseATwoStreamRiserStage(
        state=final_state,
        mouth=plan,
        transport=transport,
        physical_drag=drag_result,
        q_net_owner_value=q_net,
        q_net_boundary_value=bottom.net_rate,
        q_net_residual=q_residual,
        riser_liquid_volume_before=volume_before,
        riser_liquid_volume_after=volume_after,
        expected_riser_volume_change=expected_change,
        riser_volume_residual=volume_residual,
    )


__all__ = [
    "COMPLETE_CASEA_NETWORK_READY",
    "GLOBAL_INTEGRATION_BLOCKERS",
    "TWOSTREAM_NETWORK_ADAPTER_READY",
    "CaseATwoStreamRiserStage",
    "TwoStreamNetworkAdapterError",
    "advance_casea_twostream_riser_from_finite_node",
    "natural_open_top_boundary",
]
