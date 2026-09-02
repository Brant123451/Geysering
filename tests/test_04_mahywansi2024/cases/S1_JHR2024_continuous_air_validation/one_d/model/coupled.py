"""Atomic commit machinery and fail-closed coupled-step interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

from .conservation import ConservationLedger, LedgerEntry
from .errors import AtomicCommitError, ContractViolation, MissingPhysicalClosure
from .flux import AtomicFluxPacket, TNodePortResidual, state_token
from .state import (
    CoupledGeometry,
    CoupledState,
    ExteriorPlumeState,
    HorizontalState,
    SupplyBranchState,
    TNodeState,
    VerticalState,
)


def _add(left: tuple[float, ...], right: tuple[float, ...], name: str) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ContractViolation(f"{name} state/delta cell counts differ")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _apply_packet(state: CoupledState, packet: AtomicFluxPacket) -> CoupledState:
    h = state.horizontal
    dh = packet.horizontal
    supply = state.supply_branch
    d_supply = packet.supply_branch
    v = state.vertical
    dv = packet.vertical
    air_node = state.air_supply_node
    d_air_node = packet.air_supply_node
    riser_node = state.riser_node
    d_riser_node = packet.riser_node
    plume = state.exterior_plume
    d_plume = packet.exterior_plume
    return CoupledState(
        time_s=state.time_s + packet.dt_s,
        horizontal=HorizontalState(
            Al=_add(h.Al, dh.Al, "horizontal Al"),
            Ql=_add(h.Ql, dh.Ql, "horizontal Ql"),
            Mg=_add(h.Mg, dh.Mg, "horizontal Mg"),
            Jg=_add(h.Jg, dh.Jg, "horizontal Jg"),
        ),
        supply_branch=SupplyBranchState(
            Al=_add(supply.Al, d_supply.Al, "supply-branch Al"),
            Ql=_add(supply.Ql, d_supply.Ql, "supply-branch Ql"),
            Mg=_add(supply.Mg, d_supply.Mg, "supply-branch Mg"),
            Jg=_add(supply.Jg, d_supply.Jg, "supply-branch Jg"),
        ),
        vertical=VerticalState(
            Aup=_add(v.Aup, dv.Aup, "vertical Aup"),
            Qup=_add(v.Qup, dv.Qup, "vertical Qup"),
            Adown=_add(v.Adown, dv.Adown, "vertical Adown"),
            Qdown=_add(v.Qdown, dv.Qdown, "vertical Qdown"),
            Mg=_add(v.Mg, dv.Mg, "vertical Mg"),
            Jg=_add(v.Jg, dv.Jg, "vertical Jg"),
        ),
        exterior_plume=ExteriorPlumeState(
            airborne_liquid_volume_m3=(
                plume.airborne_liquid_volume_m3
                + d_plume.airborne_liquid_volume_m3
            ),
            airborne_vertical_momentum_kg_m_s=(
                plume.airborne_vertical_momentum_kg_m_s
                + d_plume.airborne_vertical_momentum_kg_m_s
            ),
            airborne_liquid_first_moment_m4=(
                plume.airborne_liquid_first_moment_m4
                + d_plume.airborne_liquid_first_moment_m4
            ),
            returning_liquid_volume_m3=(
                plume.returning_liquid_volume_m3
                + d_plume.returning_liquid_volume_m3
            ),
            returning_downward_momentum_kg_m_s=(
                plume.returning_downward_momentum_kg_m_s
                + d_plume.returning_downward_momentum_kg_m_s
            ),
        ),
        air_supply_node=TNodeState(
            liquid_volume=air_node.liquid_volume + d_air_node.liquid_volume,
            gas_mass=air_node.gas_mass + d_air_node.gas_mass,
            liquid_momentum=air_node.liquid_momentum + d_air_node.liquid_momentum,
            gas_momentum=air_node.gas_momentum + d_air_node.gas_momentum,
        ),
        riser_node=TNodeState(
            liquid_volume=riser_node.liquid_volume + d_riser_node.liquid_volume,
            gas_mass=riser_node.gas_mass + d_riser_node.gas_mass,
            liquid_momentum=riser_node.liquid_momentum + d_riser_node.liquid_momentum,
            gas_momentum=riser_node.gas_momentum + d_riser_node.gas_momentum,
        ),
    )


@dataclass(slots=True)
class AtomicCommitter:
    """Validate and commit one whole coupled packet or nothing at all."""

    geometry: CoupledGeometry
    ledger: ConservationLedger = field(default_factory=ConservationLedger)
    node_liquid_residual_tolerance_m3_s: float = 1.0e-12
    node_gas_residual_tolerance_kg_s: float = 1.0e-12
    node_momentum_residual_tolerance_N: float = 1.0e-9
    _committed_transaction_ids: set[str] = field(default_factory=set, init=False)
    _committed_base_state_tokens: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        for name in (
            "node_liquid_residual_tolerance_m3_s",
            "node_gas_residual_tolerance_kg_s",
            "node_momentum_residual_tolerance_N",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractViolation(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def _validate_node_ports(self, name: str, residual: TNodePortResidual) -> None:
        checks = (
            (
                "liquid",
                residual.liquid_volume_m3_s,
                self.node_liquid_residual_tolerance_m3_s,
                "m3/s",
            ),
            (
                "gas",
                residual.gas_mass_kg_s,
                self.node_gas_residual_tolerance_kg_s,
                "kg/s",
            ),
            (
                "Px",
                residual.mixture_momentum_x_N,
                self.node_momentum_residual_tolerance_N,
                "N",
            ),
            (
                "Pz",
                residual.mixture_momentum_z_N,
                self.node_momentum_residual_tolerance_N,
                "N",
            ),
        )
        failed = [
            f"{label}={value:.6e} {unit}"
            for label, value, tolerance, unit in checks
            if abs(value) > tolerance
        ]
        if failed:
            raise AtomicCommitError(
                f"{name} zero-storage port balance failed: " + ", ".join(failed)
            )

    def commit(
        self, state: CoupledState, packet: AtomicFluxPacket
    ) -> tuple[CoupledState, LedgerEntry]:
        if packet.transaction_id in self._committed_transaction_ids:
            raise AtomicCommitError(f"transaction {packet.transaction_id!r} was already committed")
        if packet.base_state_token in self._committed_base_state_tokens:
            raise AtomicCommitError(
                "this base state already has an accepted atomic successor; "
                "a second T-node update would create duplicate ownership"
            )
        if packet.base_state_token != state_token(state):
            raise AtomicCommitError("packet was built from a different or stale coupled state")

        if not packet.air_supply_node.is_zero or not packet.riser_node.is_zero:
            raise AtomicCommitError(
                "T nodes are zero-storage algebraic junctions; node inventory deltas are forbidden"
            )
        self._validate_node_ports("air-supply T node", packet.air_supply_node_ports)
        self._validate_node_ports("riser T node", packet.riser_node_ports)

        # All proposal work and all checks happen on immutable temporary state.
        # No ledger or transaction marker is changed until every check succeeds.
        proposed = _apply_packet(state, packet)
        self.geometry.validate_state(proposed)
        entry = self.ledger.evaluate(
            transaction_id=packet.transaction_id,
            before_state=state,
            after_state=proposed,
            geometry=self.geometry,
            dt_s=packet.dt_s,
            boundary=packet.boundary,
        )
        self.ledger.append(entry)
        self._committed_transaction_ids.add(packet.transaction_id)
        self._committed_base_state_tokens.add(packet.base_state_token)
        return proposed, entry


class AtomicFluxClosure(Protocol):
    """Future physical closure that must produce exactly one atomic packet."""

    def build_packet(
        self, state: CoupledState, geometry: CoupledGeometry, dt_s: float
    ) -> AtomicFluxPacket:
        ...


@dataclass(slots=True)
class CoupledStepper:
    """Thin driver that cannot run until evidence-backed closures are supplied."""

    committer: AtomicCommitter
    closure: AtomicFluxClosure | None = None

    def advance(self, state: CoupledState, dt_s: float) -> tuple[CoupledState, LedgerEntry]:
        if self.closure is None:
            raise MissingPhysicalClosure(
                "no coupled horizontal/two-T-node/vertical physical closure is installed; "
                "contract-only skeleton refuses to generate a trajectory"
            )
        packet = self.closure.build_packet(state, self.committer.geometry, dt_s)
        return self.committer.commit(state, packet)
