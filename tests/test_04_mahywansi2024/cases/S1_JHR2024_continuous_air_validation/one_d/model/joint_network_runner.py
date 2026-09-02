"""Atomic two-stage runner for the complete S1 one-dimensional topology.

This module owns *orchestration*, not an unpublished physical closure.  A
single stage evaluation must return rates for the Case-1-derived horizontal
main, the finite water-initial supply branch, the persistent two-stream riser,
and both zero-storage T junctions at once.  The runner evaluates that same
whole-network operator at both SSP-RK2 stages and commits only the final
averaged packet.  A failed predictor, second stage, node balance, admissibility
check, or conservation audit therefore leaves both the state and the global
ledger untouched.

The current components now feed one real six-port physical owner and the
simultaneous four-unknown two-T-node solver.  That owner can return one unique
conservative stage rate and exercise a source-initial microstep, but
``CurrentS1PhysicalJointOperator`` remains globally non-production while the
declared exterior/topology/end-boundary gates are open.  The separate
``StructuralZeroJointOperator`` remains only an orchestration test; it is not
a water-flow settling result, eruption result, or production trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, Protocol, runtime_checkable

from .accepted_observation_diagnostics import (
    InstantaneousGaugePressures,
    NativeIntervalDiagnostics,
    average_interval_diagnostics,
    build_instantaneous_gauge_pressures,
    interval_diagnostics_from_stage_rate,
)
from .conservation import ConservationLedger, ConservationSnapshot, LedgerEntry
from .coupled import AtomicCommitter, _apply_packet
from .errors import (
    AtomicCommitError,
    ConservationError,
    ContractViolation,
    MissingPhysicalClosure,
)
from .flux import (
    AtomicFluxPacket,
    BoundaryExchange,
    ExteriorPlumeDelta,
    HorizontalDelta,
    SupplyBranchDelta,
    TNodePortResidual,
    VerticalDelta,
    state_token,
)
from .initialization import S1InitialAssembly, build_s1_initial_assembly
from .port_contracts import ComponentStageProposal, TNodeTrial
from .state import (
    CoupledGeometry,
    CoupledState,
    ExteriorPlumeState,
    HorizontalState,
    SupplyBranchState,
    VerticalState,
)


PhysicalStage = Literal["stage1_closed", "stage2_pressure_reservoir"]

AIR_NODE_PORTS = frozenset(("main_left", "main_right", "supply_bottom"))
RISER_NODE_PORTS = frozenset(("main_left", "main_right", "riser_bottom"))
PUBLISHED_STAGE2_GAUGE_PRESSURE_PA = 5700.0


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ContractViolation(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class StageBoundaryContract:
    """Frozen interpretation of the two experimental/numerical stages."""

    stage: PhysicalStage
    air_source_open: bool
    supply_top_kind: Literal["wall", "pressure_reservoir"]
    gas_gauge_pressure_Pa: float | None
    evidence_status: str


def stage_boundary_contract(stage: PhysicalStage) -> StageBoundaryContract:
    if stage == "stage1_closed":
        return StageBoundaryContract(
            stage=stage,
            air_source_open=False,
            supply_top_kind="wall",
            gas_gauge_pressure_Pa=None,
            evidence_status="published_stage1_closed_isolation_valve",
        )
    if stage == "stage2_pressure_reservoir":
        return StageBoundaryContract(
            stage=stage,
            air_source_open=True,
            supply_top_kind="pressure_reservoir",
            gas_gauge_pressure_Pa=PUBLISHED_STAGE2_GAUGE_PRESSURE_PA,
            evidence_status="published_Table1_5700_Pa_gauge_pressure",
        )
    raise ContractViolation(f"unsupported S1 physical stage: {stage!r}")


@dataclass(frozen=True, slots=True)
class GrossComponentPortFlux:
    """Gross material rates and signed Cartesian momentum sent through a port.

    Material directions are named from the component's point of view.  Thus a
    zero-storage node has ``sum(into_component - out_of_component) == 0``.
    Momentum is a signed Cartesian rate delivered from the node to the
    component; pressure traction can remain non-zero when all advective rates
    are zero.  Gross rates and their donor speeds are never reconstructed from
    a net value.
    """

    name: str
    liquid_into_component_m3_s: float = 0.0
    liquid_out_of_component_m3_s: float = 0.0
    gas_into_component_kg_s: float = 0.0
    gas_out_of_component_kg_s: float = 0.0
    liquid_into_speed_m_s: float = 0.0
    liquid_out_speed_m_s: float = 0.0
    gas_into_speed_m_s: float = 0.0
    gas_out_speed_m_s: float = 0.0
    mixture_momentum_to_component_x_N: float = 0.0
    mixture_momentum_to_component_z_N: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractViolation("node port name must be non-empty")
        pairs = (
            (
                "liquid into",
                self.liquid_into_component_m3_s,
                self.liquid_into_speed_m_s,
            ),
            (
                "liquid out",
                self.liquid_out_of_component_m3_s,
                self.liquid_out_speed_m_s,
            ),
            ("gas into", self.gas_into_component_kg_s, self.gas_into_speed_m_s),
            ("gas out", self.gas_out_of_component_kg_s, self.gas_out_speed_m_s),
        )
        for label, raw_rate, raw_speed in pairs:
            rate = _nonnegative(f"{self.name} {label} rate", raw_rate)
            speed = _nonnegative(f"{self.name} {label} speed", raw_speed)
            if rate == 0.0 and speed != 0.0:
                raise ContractViolation(f"{self.name} {label} speed has no gross rate")
            if rate > 0.0 and speed == 0.0:
                raise ContractViolation(f"{self.name} {label} gross rate needs donor speed")
        for name in (
            "mixture_momentum_to_component_x_N",
            "mixture_momentum_to_component_z_N",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @property
    def liquid_net_into_component_m3_s(self) -> float:
        return self.liquid_into_component_m3_s - self.liquid_out_of_component_m3_s

    @property
    def gas_net_into_component_kg_s(self) -> float:
        return self.gas_into_component_kg_s - self.gas_out_of_component_kg_s


@dataclass(frozen=True, slots=True)
class ZeroStorageTNodeSolution:
    """One simultaneous three-port solution for a zero-volume T node."""

    name: str
    ports: tuple[GrossComponentPortFlux, ...]
    wall_reaction_on_fluid_x_N: float = 0.0
    wall_reaction_on_fluid_z_N: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractViolation("T-node solution name must be non-empty")
        if len(self.ports) != 3:
            raise ContractViolation("each S1 T-node solution must own exactly three ports")
        names = tuple(port.name for port in self.ports)
        if len(set(names)) != len(names):
            raise ContractViolation("T-node port names must be unique")
        for field_name in (
            "wall_reaction_on_fluid_x_N",
            "wall_reaction_on_fluid_z_N",
        ):
            object.__setattr__(self, field_name, _finite(field_name, getattr(self, field_name)))

    @property
    def port_names(self) -> frozenset[str]:
        return frozenset(port.name for port in self.ports)

    @property
    def residual(self) -> TNodePortResidual:
        return TNodePortResidual(
            liquid_volume_m3_s=sum(
                port.liquid_net_into_component_m3_s for port in self.ports
            ),
            gas_mass_kg_s=sum(port.gas_net_into_component_kg_s for port in self.ports),
            mixture_momentum_x_N=(
                sum(port.mixture_momentum_to_component_x_N for port in self.ports)
                - self.wall_reaction_on_fluid_x_N
            ),
            mixture_momentum_z_N=(
                sum(port.mixture_momentum_to_component_z_N for port in self.ports)
                - self.wall_reaction_on_fluid_z_N
            ),
        )


@dataclass(frozen=True, slots=True)
class JointStageRate:
    """Whole-network conservative rate returned by one pure RK evaluation."""

    physical_stage: PhysicalStage
    horizontal: HorizontalDelta
    supply_branch: SupplyBranchDelta
    vertical: VerticalDelta
    air_supply_node: ZeroStorageTNodeSolution
    riser_node: ZeroStorageTNodeSolution
    exterior_plume: ExteriorPlumeDelta = ExteriorPlumeDelta()
    horizontal_external: BoundaryExchange = BoundaryExchange()
    supply_external: BoundaryExchange = BoundaryExchange()
    vertical_external: BoundaryExchange = BoundaryExchange()
    exterior_plume_exchange: BoundaryExchange = BoundaryExchange()
    air_supply_node_common_absolute_pressure_Pa: float | None = None
    riser_node_common_absolute_pressure_Pa: float | None = None
    evidence_status: str = ""

    def __post_init__(self) -> None:
        stage_boundary_contract(self.physical_stage)
        if not self.evidence_status.strip():
            raise ContractViolation("joint stage rate evidence_status must be non-empty")
        pressures = (
            self.air_supply_node_common_absolute_pressure_Pa,
            self.riser_node_common_absolute_pressure_Pa,
        )
        if (pressures[0] is None) != (pressures[1] is None):
            raise ContractViolation(
                "joint stage must carry both native node pressures or neither"
            )
        for name, value in (
            (
                "air-supply node common absolute pressure",
                self.air_supply_node_common_absolute_pressure_Pa,
            ),
            (
                "riser node common absolute pressure",
                self.riser_node_common_absolute_pressure_Pa,
            ),
        ):
            if value is not None and _finite(name, value) <= 0.0:
                raise ContractViolation(f"{name} must be positive")


@runtime_checkable
class JointRKStageOperator(Protocol):
    """Pure simultaneous operator evaluated at both SSP-RK2 stages."""

    @property
    def production_ready(self) -> bool:
        ...

    @property
    def validation_only(self) -> bool:
        ...

    def evaluate(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> JointStageRate:
        ...


@runtime_checkable
class HorizontalTwoTeeStageComponent(Protocol):
    """Missing horizontal residual/flux interface seen by both T-node solves."""

    source_aligned_trajectory_ready: bool
    production_ready: bool

    def propose_joint_stage(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        *,
        air_node_trial: TNodeTrial,
        riser_node_trial: TNodeTrial,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> ComponentStageProposal:
        ...


@runtime_checkable
class VerticalPressureVoidStageComponent(Protocol):
    """Missing coupled liquid-pressure/gas-void/rim-Riemann riser interface."""

    production_ready: bool

    def propose_joint_stage(
        self,
        state: VerticalState,
        geometry: CoupledGeometry,
        *,
        riser_node_trial: TNodeTrial,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> ComponentStageProposal:
        ...


@runtime_checkable
class SimultaneousTwoTNodeFluxSolver(Protocol):
    """Missing common nonlinear solve for both zero-storage T junctions."""

    production_ready: bool

    def evaluate_joint_stage(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
        horizontal_component: HorizontalTwoTeeStageComponent,
        supply_branch_component: object,
        vertical_component: VerticalPressureVoidStageComponent,
    ) -> JointStageRate:
        ...


@dataclass(frozen=True, slots=True)
class JointStepDiagnostics:
    rk1_air_node: ZeroStorageTNodeSolution
    rk1_riser_node: ZeroStorageTNodeSolution
    rk2_air_node: ZeroStorageTNodeSolution
    rk2_riser_node: ZeroStorageTNodeSolution
    validation_only: bool
    production_ready: bool
    pressure_before: InstantaneousGaugePressures | None = None
    pressure_after: InstantaneousGaugePressures | None = None
    rk1_native_interval: NativeIntervalDiagnostics | None = None
    rk2_native_interval: NativeIntervalDiagnostics | None = None
    accepted_native_interval: NativeIntervalDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class JointStepResult:
    state: CoupledState
    ledger: LedgerEntry
    diagnostics: JointStepDiagnostics


@dataclass(frozen=True, slots=True)
class JointRunResult:
    state: CoupledState
    entries: tuple[LedgerEntry, ...]
    physical_stage: PhysicalStage
    validation_only: bool
    production_ready: bool
    status: str


@dataclass(frozen=True, slots=True)
class AcceptedStepContext:
    """Post-commit callback packet for one accepted Stage-2 transaction.

    The two states and actual ``dt_s`` are exact.  ``ledger_entries`` contains
    every global ledger entry generated by this accepted transaction (one for
    the current atomic owner).  Pressures are instantaneous accepted-state
    diagnostics; gross fluxes and node reactions are accepted SSP-RK2 interval
    averages.  This object can only be constructed after the commit succeeds.
    """

    before_state: CoupledState
    after_state: CoupledState
    actual_dt_s: float
    ledger_entries: tuple[LedgerEntry, ...]
    physical_stage: PhysicalStage
    stage2_time_start_s: float | None
    stage2_time_end_s: float | None
    diagnostics: JointStepDiagnostics

    def __post_init__(self) -> None:
        dt = _finite("accepted callback actual dt", self.actual_dt_s)
        if dt <= 0.0:
            raise ContractViolation("accepted callback dt must be positive")
        object.__setattr__(self, "actual_dt_s", dt)
        if self.physical_stage == "stage2_pressure_reservoir":
            if self.stage2_time_start_s is None or self.stage2_time_end_s is None:
                raise ContractViolation(
                    "Stage-2 accepted callback must carry elapsed physical time"
                )
            start = _finite("Stage-2 callback start", self.stage2_time_start_s)
            end = _finite("Stage-2 callback end", self.stage2_time_end_s)
            if start < 0.0 or end <= start or not math.isclose(
                end - start, dt, rel_tol=0.0, abs_tol=2.0e-12
            ):
                raise ContractViolation("accepted callback Stage-2 time interval is invalid")
            object.__setattr__(self, "stage2_time_start_s", start)
            object.__setattr__(self, "stage2_time_end_s", end)
        elif self.stage2_time_start_s is not None or self.stage2_time_end_s is not None:
            raise ContractViolation(
                "Stage-1 accepted callback cannot claim Stage-2 elapsed time"
            )
        entries = tuple(self.ledger_entries)
        if len(entries) != 1 or entries[0].time_start_s != self.before_state.time_s:
            raise ContractViolation(
                "accepted callback must carry the complete one-entry atomic ledger interval"
            )
        if not math.isclose(
            entries[0].time_end_s,
            self.after_state.time_s,
            rel_tol=0.0,
            abs_tol=1.0e-13,
        ):
            raise ContractViolation("accepted callback ledger does not end at after_state")
        object.__setattr__(self, "ledger_entries", entries)
        required = (
            self.diagnostics.pressure_before,
            self.diagnostics.pressure_after,
            self.diagnostics.rk1_native_interval,
            self.diagnostics.rk2_native_interval,
            self.diagnostics.accepted_native_interval,
        )
        if any(value is None for value in required):
            raise MissingPhysicalClosure(
                "accepted callback cannot carry a missing formal native diagnostic"
            )


AcceptedStepCallback = Callable[[AcceptedStepContext], None]


def _scaled_sum(
    left: tuple[float, ...],
    right: tuple[float, ...],
    scale: float,
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ContractViolation("RK rate vectors have different cell counts")
    return tuple(scale * (a + b) for a, b in zip(left, right, strict=True))


def _state_plus_rate(
    state: CoupledState,
    rate: JointStageRate,
    dt_s: float,
) -> CoupledState:
    dt = _finite("stage dt_s", dt_s)
    if dt <= 0.0:
        raise ContractViolation("stage dt_s must be positive")

    def add(values: tuple[float, ...], rates: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) != len(rates):
            raise ContractViolation("state and stage-rate cell counts differ")
        return tuple(value + dt * derivative for value, derivative in zip(values, rates, strict=True))

    h = state.horizontal
    s = state.supply_branch
    v = state.vertical
    plume = state.exterior_plume
    return CoupledState(
        time_s=state.time_s + dt,
        horizontal=HorizontalState(
            Al=add(h.Al, rate.horizontal.Al),
            Ql=add(h.Ql, rate.horizontal.Ql),
            Mg=add(h.Mg, rate.horizontal.Mg),
            Jg=add(h.Jg, rate.horizontal.Jg),
        ),
        supply_branch=SupplyBranchState(
            Al=add(s.Al, rate.supply_branch.Al),
            Ql=add(s.Ql, rate.supply_branch.Ql),
            Mg=add(s.Mg, rate.supply_branch.Mg),
            Jg=add(s.Jg, rate.supply_branch.Jg),
        ),
        vertical=VerticalState(
            Aup=add(v.Aup, rate.vertical.Aup),
            Qup=add(v.Qup, rate.vertical.Qup),
            Adown=add(v.Adown, rate.vertical.Adown),
            Qdown=add(v.Qdown, rate.vertical.Qdown),
            Mg=add(v.Mg, rate.vertical.Mg),
            Jg=add(v.Jg, rate.vertical.Jg),
        ),
        exterior_plume=ExteriorPlumeState(
            airborne_liquid_volume_m3=(
                plume.airborne_liquid_volume_m3
                + dt * rate.exterior_plume.airborne_liquid_volume_m3
            ),
            airborne_vertical_momentum_kg_m_s=(
                plume.airborne_vertical_momentum_kg_m_s
                + dt * rate.exterior_plume.airborne_vertical_momentum_kg_m_s
            ),
            airborne_liquid_first_moment_m4=(
                plume.airborne_liquid_first_moment_m4
                + dt * rate.exterior_plume.airborne_liquid_first_moment_m4
            ),
            returning_liquid_volume_m3=(
                plume.returning_liquid_volume_m3
                + dt * rate.exterior_plume.returning_liquid_volume_m3
            ),
            returning_downward_momentum_kg_m_s=(
                plume.returning_downward_momentum_kg_m_s
                + dt * rate.exterior_plume.returning_downward_momentum_kg_m_s
            ),
        ),
        air_supply_node=state.air_supply_node,
        riser_node=state.riser_node,
    )


_BOUNDARY_FIELDS = (
    "liquid_inflow_m3_s",
    "liquid_outflow_m3_s",
    "gas_inflow_kg_s",
    "gas_outflow_kg_s",
    "momentum_x_in_N",
    "momentum_x_out_N",
    "momentum_z_in_N",
    "momentum_z_out_N",
    "external_force_x_N",
    "external_force_z_N",
)


def _sum_boundaries(*values: BoundaryExchange) -> BoundaryExchange:
    return BoundaryExchange(
        **{
            name: sum(getattr(value, name) for value in values)
            for name in _BOUNDARY_FIELDS
        }
    )


def _boundary_with_node_reactions(rate: JointStageRate) -> BoundaryExchange:
    vertical = rate.vertical_external
    plume = rate.exterior_plume_exchange
    tolerance = 2.0e-11
    if not math.isclose(
        vertical.liquid_outflow_m3_s,
        plume.liquid_inflow_m3_s,
        rel_tol=2.0e-11,
        abs_tol=1.0e-16,
    ) or not math.isclose(
        vertical.liquid_inflow_m3_s,
        plume.liquid_outflow_m3_s,
        rel_tol=2.0e-11,
        abs_tol=1.0e-16,
    ):
        raise ConservationError(
            "vertical/exterior rim liquid gross rates do not pair internally"
        )
    if any(
        value != 0.0
        for value in (
            plume.gas_inflow_kg_s,
            plume.gas_outflow_kg_s,
            plume.momentum_x_in_N,
            plume.momentum_x_out_N,
            plume.momentum_z_out_N,
        )
    ):
        raise ConservationError("exterior liquid owner emitted an unsupported exchange")
    remaining_vertical_momentum_out = (
        vertical.momentum_z_out_N - plume.momentum_z_in_N
    )
    if remaining_vertical_momentum_out < -tolerance:
        raise ConservationError(
            "exterior liquid momentum exceeds the vertical rim momentum packet"
        )
    if abs(remaining_vertical_momentum_out) <= tolerance:
        remaining_vertical_momentum_out = 0.0
    # The paired rim liquid exchange is internal to the enlarged 1-D state and
    # must not pollute the global gross-boundary record.  Only atmospheric gas
    # advection plus the independently budgeted vertical/plume forces remain.
    atmospheric_remainder = BoundaryExchange(
        gas_inflow_kg_s=vertical.gas_inflow_kg_s,
        gas_outflow_kg_s=vertical.gas_outflow_kg_s,
        momentum_x_in_N=vertical.momentum_x_in_N,
        momentum_x_out_N=vertical.momentum_x_out_N,
        momentum_z_in_N=vertical.momentum_z_in_N,
        momentum_z_out_N=remaining_vertical_momentum_out,
        external_force_x_N=(
            vertical.external_force_x_N + plume.external_force_x_N
        ),
        external_force_z_N=(
            vertical.external_force_z_N + plume.external_force_z_N
        ),
    )
    source = _sum_boundaries(
        rate.horizontal_external,
        rate.supply_external,
        atmospheric_remainder,
    )
    return replace(
        source,
        external_force_x_N=(
            source.external_force_x_N
            + rate.air_supply_node.wall_reaction_on_fluid_x_N
            + rate.riser_node.wall_reaction_on_fluid_x_N
        ),
        external_force_z_N=(
            source.external_force_z_N
            + rate.air_supply_node.wall_reaction_on_fluid_z_N
            + rate.riser_node.wall_reaction_on_fluid_z_N
        ),
    )


def _average_boundary(left: BoundaryExchange, right: BoundaryExchange) -> BoundaryExchange:
    return BoundaryExchange(
        **{
            name: 0.5 * (getattr(left, name) + getattr(right, name))
            for name in _BOUNDARY_FIELDS
        }
    )


def _worst_residual(
    left: TNodePortResidual, right: TNodePortResidual
) -> TNodePortResidual:
    def pick(name: str) -> float:
        a = getattr(left, name)
        b = getattr(right, name)
        return a if abs(a) >= abs(b) else b

    return TNodePortResidual(
        liquid_volume_m3_s=pick("liquid_volume_m3_s"),
        gas_mass_kg_s=pick("gas_mass_kg_s"),
        mixture_momentum_x_N=pick("mixture_momentum_x_N"),
        mixture_momentum_z_N=pick("mixture_momentum_z_N"),
    )


@dataclass(frozen=True, slots=True)
class _ComponentInventory:
    liquid_volume_m3: float
    gas_mass_kg: float
    momentum_x_kg_m_s: float
    momentum_z_kg_m_s: float


def _component_inventories(
    state: CoupledState, geometry: CoupledGeometry
) -> dict[str, _ComponentInventory]:
    rho = geometry.liquid_density_kg_m3
    horizontal = _ComponentInventory(
        liquid_volume_m3=sum(
            area * dx
            for area, dx in zip(
                state.horizontal.Al, geometry.horizontal_dx_m, strict=True
            )
        ),
        gas_mass_kg=sum(
            mass * dx
            for mass, dx in zip(
                state.horizontal.Mg, geometry.horizontal_dx_m, strict=True
            )
        ),
        momentum_x_kg_m_s=sum(
            (rho * discharge + gas_momentum) * dx
            for discharge, gas_momentum, dx in zip(
                state.horizontal.Ql,
                state.horizontal.Jg,
                geometry.horizontal_dx_m,
                strict=True,
            )
        ),
        momentum_z_kg_m_s=0.0,
    )
    supply = _ComponentInventory(
        liquid_volume_m3=sum(
            area * dz
            for area, dz in zip(
                state.supply_branch.Al,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        ),
        gas_mass_kg=sum(
            mass * dz
            for mass, dz in zip(
                state.supply_branch.Mg,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        ),
        momentum_x_kg_m_s=0.0,
        momentum_z_kg_m_s=sum(
            (rho * discharge + gas_momentum) * dz
            for discharge, gas_momentum, dz in zip(
                state.supply_branch.Ql,
                state.supply_branch.Jg,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        ),
    )
    vertical = _ComponentInventory(
        liquid_volume_m3=sum(
            (up + down) * dz
            for up, down, dz in zip(
                state.vertical.Aup,
                state.vertical.Adown,
                geometry.vertical_dz_m,
                strict=True,
            )
        ),
        gas_mass_kg=sum(
            mass * dz
            for mass, dz in zip(
                state.vertical.Mg, geometry.vertical_dz_m, strict=True
            )
        ),
        momentum_x_kg_m_s=0.0,
        momentum_z_kg_m_s=sum(
            (rho * (up - down) + gas_momentum) * dz
            for up, down, gas_momentum, dz in zip(
                state.vertical.Qup,
                state.vertical.Qdown,
                state.vertical.Jg,
                geometry.vertical_dz_m,
                strict=True,
            )
        ),
    )
    exterior_plume = _ComponentInventory(
        liquid_volume_m3=state.exterior_plume.liquid_volume_m3,
        gas_mass_kg=0.0,
        momentum_x_kg_m_s=0.0,
        momentum_z_kg_m_s=state.exterior_plume.vertical_momentum_kg_m_s,
    )
    return {
        "horizontal": horizontal,
        "supply": supply,
        "vertical": vertical,
        "exterior_plume": exterior_plume,
    }


def _named_port(
    solution: ZeroStorageTNodeSolution, name: str
) -> GrossComponentPortFlux:
    for port in solution.ports:
        if port.name == name:
            return port
    raise ContractViolation(f"T-node solution has no port named {name!r}")


class S1JointNetworkRunner:
    """SSP-RK2 whole-network driver with one final atomic commit."""

    def __init__(
        self,
        geometry: CoupledGeometry,
        operator: JointRKStageOperator,
        *,
        committer: AtomicCommitter | None = None,
    ) -> None:
        if not isinstance(operator, JointRKStageOperator):
            raise ContractViolation("operator does not implement JointRKStageOperator")
        self.geometry = geometry
        self.operator = operator
        self.committer = AtomicCommitter(geometry) if committer is None else committer
        if self.committer.geometry != geometry:
            raise ContractViolation("runner and committer geometries differ")

    def _validate_node(
        self,
        node: ZeroStorageTNodeSolution,
        *,
        expected_ports: frozenset[str],
        label: str,
        rk_stage: int,
    ) -> None:
        if node.port_names != expected_ports:
            raise AtomicCommitError(
                f"RK{rk_stage} {label} ports {sorted(node.port_names)!r} do not match "
                f"the frozen topology {sorted(expected_ports)!r}"
            )
        residual = node.residual
        checks = (
            (
                "liquid",
                residual.liquid_volume_m3_s,
                self.committer.node_liquid_residual_tolerance_m3_s,
            ),
            (
                "gas",
                residual.gas_mass_kg_s,
                self.committer.node_gas_residual_tolerance_kg_s,
            ),
            (
                "Px",
                residual.mixture_momentum_x_N,
                self.committer.node_momentum_residual_tolerance_N,
            ),
            (
                "Pz",
                residual.mixture_momentum_z_N,
                self.committer.node_momentum_residual_tolerance_N,
            ),
        )
        failed = [f"{name}={value:.6e}" for name, value, tolerance in checks if abs(value) > tolerance]
        if failed:
            raise AtomicCommitError(
                f"RK{rk_stage} {label} zero-storage balance failed: " + ", ".join(failed)
            )

    def _validate_component_ledgers(
        self,
        before_state: CoupledState,
        after_state: CoupledState,
        rate: JointStageRate,
        *,
        dt_s: float,
        rk_stage: int,
    ) -> None:
        """Match each component inventory change to its own ports and boundary.

        Global conservation alone cannot detect a supply-to-riser transfer
        accidentally applied to a horizontal cell.  These three local ledgers
        make the port ownership explicit before the final atomic packet exists.
        """

        before = _component_inventories(before_state, self.geometry)
        after = _component_inventories(after_state, self.geometry)
        component_ports = {
            "horizontal": (
                _named_port(rate.air_supply_node, "main_left"),
                _named_port(rate.air_supply_node, "main_right"),
                _named_port(rate.riser_node, "main_left"),
                _named_port(rate.riser_node, "main_right"),
            ),
            "supply": (_named_port(rate.air_supply_node, "supply_bottom"),),
            "vertical": (_named_port(rate.riser_node, "riser_bottom"),),
            "exterior_plume": (),
        }
        externals = {
            "horizontal": rate.horizontal_external,
            "supply": rate.supply_external,
            "vertical": rate.vertical_external,
            "exterior_plume": rate.exterior_plume_exchange,
        }

        absolute = self.committer.ledger.absolute_tolerance
        relative = self.committer.ledger.relative_tolerance

        def accepted(residual: float, start: float, end: float, expected: float) -> bool:
            scale = max(abs(start), abs(end), abs(expected), 1.0)
            return abs(residual) <= absolute + relative * scale

        for name in ("horizontal", "supply", "vertical", "exterior_plume"):
            ports = component_ports[name]
            external = externals[name]
            start = before[name]
            end = after[name]
            expected = _ComponentInventory(
                liquid_volume_m3=dt_s
                * (
                    external.liquid_volume_net_rate
                    + sum(port.liquid_net_into_component_m3_s for port in ports)
                ),
                gas_mass_kg=dt_s
                * (
                    external.gas_mass_net_rate
                    + sum(port.gas_net_into_component_kg_s for port in ports)
                ),
                momentum_x_kg_m_s=dt_s
                * (
                    external.mixture_momentum_x_net_rate
                    + sum(
                        port.mixture_momentum_to_component_x_N for port in ports
                    )
                ),
                momentum_z_kg_m_s=dt_s
                * (
                    external.mixture_momentum_z_net_rate
                    + sum(
                        port.mixture_momentum_to_component_z_N for port in ports
                    )
                ),
            )
            fields = (
                "liquid_volume_m3",
                "gas_mass_kg",
                "momentum_x_kg_m_s",
                "momentum_z_kg_m_s",
            )
            residuals = {
                field: getattr(end, field) - getattr(start, field) - getattr(expected, field)
                for field in fields
            }
            failed = [
                f"{field}={residuals[field]:.6e}"
                for field in fields
                if not math.isfinite(residuals[field])
                or not accepted(
                    residuals[field],
                    getattr(start, field),
                    getattr(end, field),
                    getattr(expected, field),
                )
            ]
            if failed:
                raise ConservationError(
                    f"RK{rk_stage} {name} component/port ledger failed: "
                    + ", ".join(failed)
                )

    def _evaluate_stage(
        self,
        state: CoupledState,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> tuple[JointStageRate, CoupledState, BoundaryExchange]:
        rate = self.operator.evaluate(
            state,
            self.geometry,
            physical_stage=physical_stage,
            rk_stage=rk_stage,
            dt_s=dt_s,
        )
        if not isinstance(rate, JointStageRate):
            raise ContractViolation("joint operator returned a non-JointStageRate value")
        if rate.physical_stage != physical_stage:
            raise ContractViolation("joint operator changed the requested physical stage")
        self._validate_node(
            rate.air_supply_node,
            expected_ports=AIR_NODE_PORTS,
            label="air-supply T node",
            rk_stage=rk_stage,
        )
        self._validate_node(
            rate.riser_node,
            expected_ports=RISER_NODE_PORTS,
            label="riser T node",
            rk_stage=rk_stage,
        )
        proposed = _state_plus_rate(state, rate, dt_s)
        self.geometry.validate_state(proposed)
        self._validate_component_ledgers(
            state,
            proposed,
            rate,
            dt_s=dt_s,
            rk_stage=rk_stage,
        )
        boundary = _boundary_with_node_reactions(rate)

        # Each RK derivative must conserve on its own.  This prevents opposite
        # mistakes at RK1/RK2 from disappearing in the final average.
        stage_ledger = ConservationLedger(
            absolute_tolerance=self.committer.ledger.absolute_tolerance,
            relative_tolerance=self.committer.ledger.relative_tolerance,
        )
        stage_ledger.evaluate(
            transaction_id=f"rk{rk_stage}-precommit-audit",
            before_state=state,
            after_state=proposed,
            geometry=self.geometry,
            dt_s=dt_s,
            boundary=boundary,
        )
        return rate, proposed, boundary

    def _prepare_atomic_state(
        self,
        state: CoupledState,
        *,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> CoupledState:
        """Apply a pure conservative event relabel before either RK stage."""

        prepare = getattr(self.operator, "prepare_atomic_state", None)
        if not callable(prepare):
            return state
        prepared = prepare(
            state,
            self.geometry,
            physical_stage=physical_stage,
            dt_s=dt_s,
        )
        if not isinstance(prepared, CoupledState):
            raise ContractViolation("joint operator returned a non-CoupledState preparation")
        if (
            prepared.time_s != state.time_s
            or prepared.horizontal != state.horizontal
            or prepared.supply_branch != state.supply_branch
            or prepared.vertical != state.vertical
            or prepared.air_supply_node != state.air_supply_node
            or prepared.riser_node != state.riser_node
        ):
            raise AtomicCommitError(
                "pre-RK event preparation may only relabel exterior plume state"
            )
        if not math.isclose(
            prepared.exterior_plume.liquid_volume_m3,
            state.exterior_plume.liquid_volume_m3,
            rel_tol=0.0,
            abs_tol=self.committer.ledger.absolute_tolerance,
        ) or not math.isclose(
            prepared.exterior_plume.vertical_momentum_kg_m_s,
            state.exterior_plume.vertical_momentum_kg_m_s,
            rel_tol=0.0,
            abs_tol=self.committer.ledger.absolute_tolerance,
        ):
            raise AtomicCommitError(
                "pre-RK exterior event relabel changed material or momentum"
            )
        self.geometry.validate_state(prepared)
        return prepared

    def advance_one(
        self,
        state: CoupledState,
        *,
        dt_s: float,
        physical_stage: PhysicalStage,
        transaction_id: str,
        require_production: bool = True,
        accepted_step_callback: AcceptedStepCallback | None = None,
        stage2_origin_absolute_s: float | None = None,
        require_native_diagnostics: bool = False,
    ) -> JointStepResult:
        stage_boundary_contract(physical_stage)
        dt = _finite("joint dt_s", dt_s)
        if dt <= 0.0:
            raise ContractViolation("joint dt_s must be positive")
        self.geometry.validate_state(state)
        if require_production and not self.operator.production_ready:
            raise MissingPhysicalClosure(
                "joint S1 trajectory requested from a non-production operator"
            )
        native_required = require_native_diagnostics or accepted_step_callback is not None
        origin: float | None = None
        if native_required and physical_stage == "stage2_pressure_reservoir":
            if stage2_origin_absolute_s is None:
                raise ContractViolation(
                    "Stage-2 accepted diagnostics require the absolute opening time"
                )
            origin = _finite("Stage-2 absolute origin", stage2_origin_absolute_s)
            if origin < 0.0 or state.time_s < origin - 1.0e-12:
                raise ContractViolation("Stage-2 origin is after the accepted-step base state")
        elif physical_stage == "stage1_closed" and stage2_origin_absolute_s is not None:
            raise ContractViolation("Stage-1 callback cannot declare a Stage-2 origin")

        # No mutation or ledger append occurs during the pure event relabel or
        # either RK stage audit.  The relabel is included in the sole final
        # packet, so a later rejection rolls it back with the physical rates.
        prepared = self._prepare_atomic_state(
            state, physical_stage=physical_stage, dt_s=dt
        )
        rk1, predictor, boundary1 = self._evaluate_stage(
            prepared,
            physical_stage=physical_stage,
            rk_stage=1,
            dt_s=dt,
        )
        rk2, _, boundary2 = self._evaluate_stage(
            predictor,
            physical_stage=physical_stage,
            rk_stage=2,
            dt_s=dt,
        )

        packet = AtomicFluxPacket(
            transaction_id=transaction_id,
            base_state_token=state_token(state),
            dt_s=dt,
            horizontal=HorizontalDelta(
                Al=_scaled_sum(rk1.horizontal.Al, rk2.horizontal.Al, 0.5 * dt),
                Ql=_scaled_sum(rk1.horizontal.Ql, rk2.horizontal.Ql, 0.5 * dt),
                Mg=_scaled_sum(rk1.horizontal.Mg, rk2.horizontal.Mg, 0.5 * dt),
                Jg=_scaled_sum(rk1.horizontal.Jg, rk2.horizontal.Jg, 0.5 * dt),
            ),
            supply_branch=SupplyBranchDelta(
                Al=_scaled_sum(
                    rk1.supply_branch.Al, rk2.supply_branch.Al, 0.5 * dt
                ),
                Ql=_scaled_sum(
                    rk1.supply_branch.Ql, rk2.supply_branch.Ql, 0.5 * dt
                ),
                Mg=_scaled_sum(
                    rk1.supply_branch.Mg, rk2.supply_branch.Mg, 0.5 * dt
                ),
                Jg=_scaled_sum(
                    rk1.supply_branch.Jg, rk2.supply_branch.Jg, 0.5 * dt
                ),
            ),
            vertical=VerticalDelta(
                Aup=_scaled_sum(rk1.vertical.Aup, rk2.vertical.Aup, 0.5 * dt),
                Qup=_scaled_sum(rk1.vertical.Qup, rk2.vertical.Qup, 0.5 * dt),
                Adown=_scaled_sum(
                    rk1.vertical.Adown, rk2.vertical.Adown, 0.5 * dt
                ),
                Qdown=_scaled_sum(
                    rk1.vertical.Qdown, rk2.vertical.Qdown, 0.5 * dt
                ),
                Mg=_scaled_sum(rk1.vertical.Mg, rk2.vertical.Mg, 0.5 * dt),
                Jg=_scaled_sum(rk1.vertical.Jg, rk2.vertical.Jg, 0.5 * dt),
            ),
            exterior_plume=ExteriorPlumeDelta(
                airborne_liquid_volume_m3=(
                    prepared.exterior_plume.airborne_liquid_volume_m3
                    - state.exterior_plume.airborne_liquid_volume_m3
                    +
                    0.5
                    * dt
                    * (
                        rk1.exterior_plume.airborne_liquid_volume_m3
                        + rk2.exterior_plume.airborne_liquid_volume_m3
                    )
                ),
                airborne_vertical_momentum_kg_m_s=(
                    prepared.exterior_plume.airborne_vertical_momentum_kg_m_s
                    - state.exterior_plume.airborne_vertical_momentum_kg_m_s
                    +
                    0.5
                    * dt
                    * (
                        rk1.exterior_plume.airborne_vertical_momentum_kg_m_s
                        + rk2.exterior_plume.airborne_vertical_momentum_kg_m_s
                    )
                ),
                airborne_liquid_first_moment_m4=(
                    prepared.exterior_plume.airborne_liquid_first_moment_m4
                    - state.exterior_plume.airborne_liquid_first_moment_m4
                    +
                    0.5
                    * dt
                    * (
                        rk1.exterior_plume.airborne_liquid_first_moment_m4
                        + rk2.exterior_plume.airborne_liquid_first_moment_m4
                    )
                ),
                returning_liquid_volume_m3=(
                    prepared.exterior_plume.returning_liquid_volume_m3
                    - state.exterior_plume.returning_liquid_volume_m3
                    +
                    0.5
                    * dt
                    * (
                        rk1.exterior_plume.returning_liquid_volume_m3
                        + rk2.exterior_plume.returning_liquid_volume_m3
                    )
                ),
                returning_downward_momentum_kg_m_s=(
                    prepared.exterior_plume.returning_downward_momentum_kg_m_s
                    - state.exterior_plume.returning_downward_momentum_kg_m_s
                    +
                    0.5
                    * dt
                    * (
                        rk1.exterior_plume.returning_downward_momentum_kg_m_s
                        + rk2.exterior_plume.returning_downward_momentum_kg_m_s
                    )
                ),
            ),
            air_supply_node_ports=_worst_residual(
                rk1.air_supply_node.residual, rk2.air_supply_node.residual
            ),
            riser_node_ports=_worst_residual(
                rk1.riser_node.residual, rk2.riser_node.residual
            ),
            boundary=_average_boundary(boundary1, boundary2),
        )

        pressure_before: InstantaneousGaugePressures | None = None
        pressure_after: InstantaneousGaugePressures | None = None
        rk1_native: NativeIntervalDiagnostics | None = None
        rk2_native: NativeIntervalDiagnostics | None = None
        accepted_native: NativeIntervalDiagnostics | None = None
        candidate_after: CoupledState | None = None
        if native_required:
            if (
                rk1.riser_node_common_absolute_pressure_Pa is None
                or rk2.riser_node_common_absolute_pressure_Pa is None
            ):
                raise MissingPhysicalClosure(
                    "physical RK stage omitted its native zero-storage-node pressures"
                )
            # RK1 is evaluated at the exact before-state.  Its node pressure is
            # therefore the native instantaneous P1 datum at that state.
            pressure_before = build_instantaneous_gauge_pressures(
                state,
                self.geometry,
                horizontal_component=getattr(self.operator, "horizontal_component", None),
                vertical_component=getattr(self.operator, "vertical_component", None),
                riser_node_common_absolute_pressure_Pa=(
                    rk1.riser_node_common_absolute_pressure_Pa
                ),
            )
            rk1_native = interval_diagnostics_from_stage_rate(
                rk1, vertical_area_m2=self.geometry.vertical_area_m2
            )
            rk2_native = interval_diagnostics_from_stage_rate(
                rk2, vertical_area_m2=self.geometry.vertical_area_m2
            )
            accepted_native = average_interval_diagnostics(rk1_native, rk2_native)

            # Preview is immutable and uses the exact packet later presented
            # to the committer.  The pure diagnostic node solve is deliberately
            # completed before commit, so a missing P1--P6 quantity leaves the
            # state and append-only ledger untouched.
            candidate_after = _apply_packet(state, packet)
            self.geometry.validate_state(candidate_after)
            diagnostic_node_pressures = getattr(
                self.operator, "diagnostic_node_pressures", None
            )
            if not callable(diagnostic_node_pressures):
                raise MissingPhysicalClosure(
                    "operator has no pure post-state diagnostic node solve"
                )
            post_air_pressure, post_riser_pressure = diagnostic_node_pressures(
                candidate_after,
                self.geometry,
                physical_stage=physical_stage,
                diagnostic_dt_s=dt,
            )
            # Both values are required even though only riser-T maps to P1;
            # this prevents a partial diagnostic node solve from passing.
            if min(
                _finite("post-state air-supply node pressure", post_air_pressure),
                _finite("post-state riser node pressure", post_riser_pressure),
            ) <= 0.0:
                raise MissingPhysicalClosure(
                    "pure post-state diagnostic node pressure is non-positive"
                )
            pressure_after = build_instantaneous_gauge_pressures(
                candidate_after,
                self.geometry,
                horizontal_component=getattr(self.operator, "horizontal_component", None),
                vertical_component=getattr(self.operator, "vertical_component", None),
                riser_node_common_absolute_pressure_Pa=post_riser_pressure,
            )

        advanced, entry = self.committer.commit(state, packet)
        if candidate_after is not None and advanced != candidate_after:
            raise AtomicCommitError(
                "committed state differs from the prevalidated diagnostic candidate"
            )
        diagnostics = JointStepDiagnostics(
            rk1_air_node=rk1.air_supply_node,
            rk1_riser_node=rk1.riser_node,
            rk2_air_node=rk2.air_supply_node,
            rk2_riser_node=rk2.riser_node,
            validation_only=self.operator.validation_only,
            production_ready=self.operator.production_ready,
            pressure_before=pressure_before,
            pressure_after=pressure_after,
            rk1_native_interval=rk1_native,
            rk2_native_interval=rk2_native,
            accepted_native_interval=accepted_native,
        )
        result = JointStepResult(
            state=advanced,
            ledger=entry,
            diagnostics=diagnostics,
        )
        if accepted_step_callback is not None:
            context = AcceptedStepContext(
                before_state=state,
                after_state=advanced,
                actual_dt_s=dt,
                ledger_entries=(entry,),
                physical_stage=physical_stage,
                stage2_time_start_s=(
                    None if origin is None else state.time_s - origin
                ),
                stage2_time_end_s=(
                    None if origin is None else advanced.time_s - origin
                ),
                diagnostics=diagnostics,
            )
            # This is intentionally the final operation: no rejected RK stage,
            # diagnostic failure or atomic-commit failure can invoke callback.
            accepted_step_callback(context)
        return result

    def advance(
        self,
        state: CoupledState,
        *,
        duration_s: float,
        maximum_dt_s: float,
        physical_stage: PhysicalStage,
        transaction_prefix: str,
        require_production: bool = True,
        maximum_steps: int = 1_000_000,
        accepted_step_callback: AcceptedStepCallback | None = None,
        stage2_origin_absolute_s: float | None = None,
        common_output_interval_s: float | None = None,
        require_native_diagnostics: bool = False,
    ) -> JointRunResult:
        duration = _finite("joint duration_s", duration_s)
        maximum_dt = _finite("joint maximum_dt_s", maximum_dt_s)
        if duration <= 0.0 or maximum_dt <= 0.0:
            raise ContractViolation("joint duration and maximum dt must be positive")
        if not transaction_prefix.strip():
            raise ContractViolation("transaction_prefix must be non-empty")
        native_required = require_native_diagnostics or accepted_step_callback is not None
        common_interval: float | None = None
        origin: float | None = None
        if native_required and physical_stage == "stage2_pressure_reservoir":
            if stage2_origin_absolute_s is None:
                raise ContractViolation("Stage-2 callback run has no absolute opening time")
            origin = _finite("Stage-2 absolute origin", stage2_origin_absolute_s)
            requested_common = (
                0.10
                if common_output_interval_s is None
                else _finite("common output interval", common_output_interval_s)
            )
            if not math.isclose(requested_common, 0.10, rel_tol=0.0, abs_tol=1.0e-15):
                raise ContractViolation("formal S1 common output interval must be 0.10 s")
            common_interval = requested_common
        elif physical_stage == "stage1_closed":
            if stage2_origin_absolute_s is not None or common_output_interval_s is not None:
                raise ContractViolation(
                    "Stage-1 callback cannot declare a Stage-2 origin/common grid"
                )
        target = state.time_s + duration
        current = state
        entries: list[LedgerEntry] = []
        for step in range(maximum_steps):
            if current.time_s >= target - 1.0e-14 * max(1.0, target):
                return JointRunResult(
                    state=current,
                    entries=tuple(entries),
                    physical_stage=physical_stage,
                    validation_only=self.operator.validation_only,
                    production_ready=self.operator.production_ready,
                    status=(
                        "production_joint_trajectory"
                        if self.operator.production_ready
                        else "physical_joint_owner_validation_only"
                        if bool(
                            getattr(self.operator, "integration_owner_ready", False)
                        )
                        else "structural_atomic_validation_only"
                    ),
                )
            dt = min(maximum_dt, target - current.time_s)
            stable = getattr(self.operator, "stable_timestep_s", None)
            if callable(stable):
                dt = min(
                    dt,
                    _finite(
                        "joint operator stable timestep",
                        stable(current, self.geometry, physical_stage=physical_stage),
                    ),
                )
            if common_interval is not None:
                assert origin is not None
                elapsed = current.time_s - origin
                if elapsed < -1.0e-12:
                    raise ContractViolation("Stage-2 run begins before its declared origin")
                # Integer construction avoids cumulative 0.1 additions.  If a
                # stable step would cross a common time, the solver itself is
                # stopped exactly there; no phase-interface state is sampled by
                # interpolation.
                index = math.floor(
                    (max(elapsed, 0.0) + 2.0e-13) / common_interval
                ) + 1
                next_common_absolute = origin + index * common_interval
                distance = next_common_absolute - current.time_s
                if distance > 2.0e-13:
                    dt = min(dt, distance)
            if dt <= 0.0:
                raise ContractViolation("joint operator returned a non-positive stable timestep")
            transaction_id = (
                f"{transaction_prefix}-{step:08d}-t{current.time_s:.17g}"
            )
            result = self.advance_one(
                current,
                dt_s=dt,
                physical_stage=physical_stage,
                transaction_id=transaction_id,
                require_production=require_production,
                accepted_step_callback=accepted_step_callback,
                stage2_origin_absolute_s=origin,
                require_native_diagnostics=native_required,
            )
            current = result.state
            entries.append(result.ledger)
        raise ContractViolation("maximum joint steps reached before requested duration")


def _zero_ports(names: frozenset[str]) -> tuple[GrossComponentPortFlux, ...]:
    return tuple(GrossComponentPortFlux(name=name) for name in sorted(names))


@dataclass(frozen=True, slots=True)
class StructuralZeroJointOperator:
    """Static full-topology transaction operator for contract tests only."""

    production_ready: bool = False
    validation_only: bool = True

    def stable_timestep_s(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
    ) -> float:
        del state, geometry
        stage_boundary_contract(physical_stage)
        return 1.0e-3

    def evaluate(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> JointStageRate:
        del geometry, dt_s
        if rk_stage not in (1, 2):
            raise ContractViolation("SSP-RK2 stage index must be 1 or 2")
        stage_boundary_contract(physical_stage)
        return JointStageRate(
            physical_stage=physical_stage,
            horizontal=HorizontalDelta.zeros(state.horizontal.cell_count),
            supply_branch=SupplyBranchDelta.zeros(state.supply_branch.cell_count),
            vertical=VerticalDelta.zeros(state.vertical.cell_count),
            air_supply_node=ZeroStorageTNodeSolution(
                name="air_supply_T", ports=_zero_ports(AIR_NODE_PORTS)
            ),
            riser_node=ZeroStorageTNodeSolution(
                name="riser_T", ports=_zero_ports(RISER_NODE_PORTS)
            ),
            evidence_status="structural_zero_flux_atomic_path_only",
        )


@dataclass(frozen=True, slots=True)
class ClosureCapability:
    name: str
    ready: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PhysicalReadinessReport:
    capabilities: tuple[ClosureCapability, ...]

    @property
    def production_ready(self) -> bool:
        return bool(self.capabilities) and all(item.ready for item in self.capabilities)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            f"{item.name}: {item.detail}" for item in self.capabilities if not item.ready
        )


class CurrentS1PhysicalJointOperator:
    """Non-production wrapper around the real six-port physical stage owner."""

    def __init__(
        self,
        *,
        horizontal_component: object,
        supply_branch_component: object,
        vertical_component: object,
        two_tnode_solver: object | None,
        joint_stage_owner: object | None = None,
    ) -> None:
        self.horizontal_component = horizontal_component
        self.supply_branch_component = supply_branch_component
        self.vertical_component = vertical_component
        self.two_tnode_solver = two_tnode_solver
        self.joint_stage_owner = joint_stage_owner

    @staticmethod
    def _flag(component: object, name: str) -> bool:
        return bool(getattr(component, name, False))

    @property
    def validation_only(self) -> bool:
        return not self.production_ready

    @property
    def integration_owner_ready(self) -> bool:
        owner = self.joint_stage_owner
        return bool(
            owner is not None
            and self._flag(owner, "integration_owner_ready")
            and callable(getattr(owner, "evaluate", None))
            and callable(getattr(owner, "stable_timestep_s", None))
            and self._flag(self.two_tnode_solver, "algebraic_gate_ready")
            and self._flag(self.horizontal_component, "joint_trial_ready")
            and self._flag(self.supply_branch_component, "joint_trial_ready")
            and self._flag(self.vertical_component, "joint_trial_ready")
        )

    @property
    def readiness(self) -> PhysicalReadinessReport:
        horizontal_ready = (
            isinstance(self.horizontal_component, HorizontalTwoTeeStageComponent)
            and self._flag(
                self.horizontal_component, "source_aligned_trajectory_ready"
            )
            and self._flag(self.horizontal_component, "production_ready")
        )
        supply_ready = self._flag(
            self.supply_branch_component, "production_ready"
        ) and callable(
            getattr(self.supply_branch_component, "propose_atomic_step", None)
        )
        vertical_ready = (
            isinstance(self.vertical_component, VerticalPressureVoidStageComponent)
            and self._flag(self.vertical_component, "production_ready")
        )
        node_ready = (
            self._flag(self.two_tnode_solver, "algebraic_gate_ready")
            and self.integration_owner_ready
        )
        return PhysicalReadinessReport(
            capabilities=(
                ClosureCapability(
                    "Case1 horizontal Al/Ql/Mg/Jg with both tees",
                    horizontal_ready,
                    (
                        "the component-level Case1 operator and physical joint owner are "
                        "present, but water-end phase re-entry and result acceptance remain closed"
                        if not horizontal_ready
                        else "ready"
                    ),
                ),
                ClosureCapability(
                    "water-initial supply Al/Ql/Mg/Jg",
                    supply_ready,
                    (
                        "the water-initial two-phase component now has a common pure T-trial "
                        "adapter, but unpublished apparatus valve/line-loss and global gates "
                        "still prohibit production"
                        if not supply_ready
                        else "ready"
                    ),
                ),
                ClosureCapability(
                    "persistent riser Aup/Qup/Adown/Qdown/Mg/Jg",
                    vertical_ready,
                    (
                        "the six-state pressure/void component and persistent exterior "
                        "storage/re-entry owner are present, but generalized bottom/top "
                        "phase topology remains fail-closed"
                        if not vertical_ready
                        else "ready"
                    ),
                ),
                ClosureCapability(
                    "simultaneous two-zero-storage-T-node solve",
                    node_ready,
                    (
                        "the four-unknown algebraic kernel or real six-port JointStageRate "
                        "owner is not fully integrated"
                        if not node_ready
                        else "ready"
                    ),
                ),
            )
        )

    @property
    def production_ready(self) -> bool:
        return self.readiness.production_ready

    def assert_ready(self) -> None:
        if not self.production_ready:
            raise MissingPhysicalClosure(
                "source-aligned 0.02 s joint physical smoke is blocked; "
                + " | ".join(self.readiness.blockers)
            )

    def assert_integration_owner_ready(self) -> None:
        if not self.integration_owner_ready:
            raise MissingPhysicalClosure(
                "real six-port physical JointStageRate owner is not ready"
            )

    def prepare_atomic_state(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> CoupledState:
        self.assert_integration_owner_ready()
        assert self.joint_stage_owner is not None
        return self.joint_stage_owner.prepare_atomic_state(
            state,
            geometry,
            physical_stage=physical_stage,
            dt_s=dt_s,
        )

    def stable_timestep_s(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
    ) -> float:
        self.assert_integration_owner_ready()
        assert self.joint_stage_owner is not None
        return self.joint_stage_owner.stable_timestep_s(
            state, geometry, physical_stage=physical_stage
        )

    def diagnostic_node_pressures(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        diagnostic_dt_s: float,
    ) -> tuple[float, float]:
        """Delegate the pure post-state node solve; never commit its proposals."""

        self.assert_integration_owner_ready()
        assert self.joint_stage_owner is not None
        method = getattr(self.joint_stage_owner, "diagnostic_node_pressures", None)
        if not callable(method):
            raise MissingPhysicalClosure(
                "physical joint owner has no pure diagnostic node-pressure solve"
            )
        result = method(
            state,
            geometry,
            physical_stage=physical_stage,
            diagnostic_dt_s=diagnostic_dt_s,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ContractViolation("diagnostic node-pressure solve returned wrong packet")
        return (_finite("air diagnostic pressure", result[0]), _finite("riser diagnostic pressure", result[1]))

    def evaluate(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> JointStageRate:
        self.assert_integration_owner_ready()
        assert self.joint_stage_owner is not None
        result = self.joint_stage_owner.evaluate(
            state,
            geometry,
            physical_stage=physical_stage,
            rk_stage=rk_stage,
            dt_s=dt_s,
        )
        if not isinstance(result, JointStageRate):
            raise ContractViolation("physical six-port owner returned wrong rate type")
        return result


def build_current_physical_operator() -> CurrentS1PhysicalJointOperator:
    """Build the actual current components while preserving the closed gate."""

    from .atmospheric_exterior_plume import F0AtmosphericExteriorPlumeOwner
    from .horizontal_case1_adapter import build_s1_2d_eos_aligned_horizontal_adapter
    from .horizontal_two_tee_component import (
        F0HorizontalTwoTeeStageComponent,
        PLANAR_2D_CAPILLARY_MODE,
    )
    from .simultaneous_two_tnode_solver import F0SimultaneousTwoTNodeSolver
    from .supply_branch_twophase import SupplyBranchTwoPhaseSolver
    from .vertical_pressure_void_component import F0VerticalPressureVoidStageComponent
    from .physical_joint_owner import F0PhysicalTwoTNodeStageOwner

    horizontal = F0HorizontalTwoTeeStageComponent(
        build_s1_2d_eos_aligned_horizontal_adapter(),
        capillary_geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        liquid_eos_reconciled_with_2d=True,
    )
    supply = SupplyBranchTwoPhaseSolver()
    # The source-aligned comparison mode is planar 2-D.  Select that declared
    # component geometry explicitly; no contact angle is invented.
    from .vertical_pressure_void_component import F0VerticalCapillaryOwner

    vertical = F0VerticalPressureVoidStageComponent(
        capillary_owner=F0VerticalCapillaryOwner(
            mode=PLANAR_2D_CAPILLARY_MODE
        )
    )
    node_solver = F0SimultaneousTwoTNodeSolver()
    exterior_plume = F0AtmosphericExteriorPlumeOwner()
    owner = F0PhysicalTwoTNodeStageOwner(
        horizontal_component=horizontal,
        supply_branch_component=supply,
        vertical_component=vertical,
        two_tnode_solver=node_solver,
        exterior_plume_owner=exterior_plume,
    )
    return CurrentS1PhysicalJointOperator(
        horizontal_component=horizontal,
        supply_branch_component=supply,
        vertical_component=vertical,
        two_tnode_solver=node_solver,
        joint_stage_owner=owner,
    )


def run_structural_source_initial_atomic_check(
    *,
    duration_s: float = 0.02,
    dt_s: float = 1.0e-3,
    assembly: S1InitialAssembly | None = None,
) -> JointRunResult:
    """Exercise the full atomic path; this is explicitly not a physical smoke."""

    source = build_s1_initial_assembly() if assembly is None else assembly
    runner = S1JointNetworkRunner(source.geometry, StructuralZeroJointOperator())
    return runner.advance(
        source.state,
        duration_s=duration_s,
        maximum_dt_s=dt_s,
        physical_stage="stage1_closed",
        transaction_prefix="s1-structural-zero-smoke",
        require_production=False,
    )


def assert_source_initial_physical_smoke_ready() -> None:
    """Fail closed until the real common two-node physical operator exists."""

    build_current_physical_operator().assert_ready()


__all__ = [
    "AcceptedStepCallback",
    "AcceptedStepContext",
    "AIR_NODE_PORTS",
    "ClosureCapability",
    "CurrentS1PhysicalJointOperator",
    "GrossComponentPortFlux",
    "HorizontalTwoTeeStageComponent",
    "JointRKStageOperator",
    "JointRunResult",
    "JointStageRate",
    "JointStepDiagnostics",
    "JointStepResult",
    "PUBLISHED_STAGE2_GAUGE_PRESSURE_PA",
    "PhysicalReadinessReport",
    "PhysicalStage",
    "RISER_NODE_PORTS",
    "S1JointNetworkRunner",
    "SimultaneousTwoTNodeFluxSolver",
    "StageBoundaryContract",
    "StructuralZeroJointOperator",
    "VerticalPressureVoidStageComponent",
    "ZeroStorageTNodeSolution",
    "assert_source_initial_physical_smoke_ready",
    "build_current_physical_operator",
    "run_structural_source_initial_atomic_check",
    "stage_boundary_contract",
]
