"""Physical six-port owner for one immutable S1 whole-network RK stage.

The owner is deliberately narrower than a production trajectory.  It builds
all four horizontal, one supply-bottom and one riser-bottom traces from the
same :class:`~model.state.CoupledState`, sends them together to the existing
four-unknown simultaneous two-T-node solver, and converts the three accepted
component proposals into exactly one :class:`~model.joint_network_runner.JointStageRate`.

No result target, seeded eruption or synthetic zero component is used.  The
horizontal proposal remains the hash-pinned Case-1 spatial operator and the
riser proposal remains the persistent six-state Case-1 two-fluid component.
The same atomic stage also owns a persistent atmospheric exterior liquid lump
for prior rim outflow and finite later re-entry.  The owner remains
non-production while generalized topology, water-end phase re-entry and the
canonical output gates are unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from .atmospheric_exterior_plume import F0AtmosphericExteriorPlumeOwner
from .errors import ContractViolation, MissingPhysicalClosure
from .flux import HorizontalDelta, SupplyBranchDelta, VerticalDelta
from .joint_network_runner import JointStageRate, PhysicalStage
from .port_contracts import (
    CapillaryInterfaceOwnership,
    PortKey,
    PortTraceState,
)
from .simultaneous_two_tnode_solver import (
    F0SimultaneousTwoTNodeSolver,
    PortAcousticScale,
    PortDirectionalSeed,
    SolvedJointNodeStage,
)
from .state import CoupledGeometry, CoupledState


_NODE_OWNER = {
    "air_supply_T": "air_supply_t_node",
    "riser_T": "riser_t_node",
}


@dataclass(frozen=True, slots=True)
class PhysicalJointStageInputs:
    """Complete immutable input set sent to one simultaneous node solve."""

    traces: tuple[PortTraceState, ...]
    acoustic_scales: tuple[PortAcousticScale, ...]
    directional_seeds: tuple[PortDirectionalSeed, ...]
    interfaces: tuple[CapillaryInterfaceOwnership, ...]

    @property
    def port_keys(self) -> frozenset[PortKey]:
        return frozenset(trace.key for trace in self.traces)


class F0PhysicalTwoTNodeStageOwner:
    """Assemble one real six-port/two-node F0 stage without committing it."""

    integration_owner_ready = True
    production_ready = False
    validation_only = True
    cfl_safety = 0.25
    evidence_status = (
        "S1-1D-F0_real_six_port_two_T_owner__same_RK_state__"
        "persistent_exterior_liquid__no_result_tuning_no_prescribed_eruption"
    )
    remaining_production_blockers = (
        "generalized bottom-piston/top-spill topology",
        "Table-1 water-end phase re-entry after horizontal gas arrival",
        "canonical trajectory observer/exporter and result acceptance",
    )

    def __init__(
        self,
        *,
        horizontal_component: object,
        supply_branch_component: object,
        vertical_component: object,
        two_tnode_solver: F0SimultaneousTwoTNodeSolver,
        exterior_plume_owner: F0AtmosphericExteriorPlumeOwner,
    ) -> None:
        self.horizontal_component = horizontal_component
        self.supply_branch_component = supply_branch_component
        self.vertical_component = vertical_component
        self.two_tnode_solver = two_tnode_solver
        self.exterior_plume_owner = exterior_plume_owner
        expected = {
            getattr(horizontal_component, "component_id", None),
            getattr(supply_branch_component, "component_id", None),
            getattr(vertical_component, "component_id", None),
        }
        if expected != {
            "horizontal_main",
            "air_supply_branch",
            "vertical_riser",
        }:
            raise ContractViolation(
                "physical joint owner requires the three frozen S1 components"
            )
        for component, methods in (
            (horizontal_component, ("port_traces", "evaluate_trial")),
            (supply_branch_component, ("port_trace", "evaluate_trial")),
            (vertical_component, ("port_trace", "evaluate_trial")),
        ):
            if any(not callable(getattr(component, name, None)) for name in methods):
                raise MissingPhysicalClosure(
                    f"{component!r} lacks a physical trace/trial adapter"
                )
        if not getattr(two_tnode_solver, "algebraic_gate_ready", False):
            raise MissingPhysicalClosure("simultaneous two-node algebraic gate is not ready")
        if not getattr(exterior_plume_owner, "persistent_cycle_ready", False):
            raise MissingPhysicalClosure(
                "persistent atmospheric exterior-plume owner is not ready"
            )
        if not callable(getattr(vertical_component, "with_liquid_fallback", None)):
            raise MissingPhysicalClosure(
                "vertical component cannot consume a stage-owned exterior parcel"
            )

    @staticmethod
    def _port_interface_records(
        traces: tuple[PortTraceState, ...],
    ) -> dict[PortKey, CapillaryInterfaceOwnership]:
        """Create only topology-required gas-nose records at a mixed T node.

        A record is attached to each water-receiving or mixed port when a
        different port at the same zero-storage node presents finite gas.  Its
        planar ``+2/D`` gas-nose sign is frozen by the preregistered F0 model;
        no comparison result or fitted contact angle enters this decision.
        """

        records: dict[PortKey, CapillaryInterfaceOwnership] = {}
        for node_name in ("air_supply_T", "riser_T"):
            node = tuple(trace for trace in traces if trace.key.node_name == node_name)
            if len(node) != 3:
                raise ContractViolation(f"{node_name} does not expose exactly three traces")
            scale = max(trace.full_area_m2 for trace in node)
            tolerance = 1.0e-12 * scale
            has_gas = any(trace.gas_area_m2 > tolerance for trace in node)
            has_liquid = any(trace.liquid_area_m2 > tolerance for trace in node)
            if not (has_gas and has_liquid):
                continue
            for trace in node:
                if trace.liquid_area_m2 <= tolerance:
                    continue
                diameter = math.sqrt(4.0 * trace.full_area_m2 / math.pi)
                curvature = 2.0 / diameter
                jump = 0.072 * curvature
                records[trace.key] = CapillaryInterfaceOwnership(
                    interface_id=(
                        f"{node_name}-{trace.key.port_name}-topology-gas-nose"
                    ),
                    owner=_NODE_OWNER[node_name],
                    surface_tension_N_m=0.072,
                    geometry_mode="planar_2d_zeroGradient_walls",
                    curvature_1_m=curvature,
                    contact_angle_deg=None,
                    pressure_jump_gas_minus_liquid_Pa=jump,
                    evidence_status=(
                        "declared_planar_semicircular_gas_nose__"
                        "cross_T_topology_derived_not_tuned"
                    ),
                )
        return records

    def _raw_traces(
        self, state: CoupledState, geometry: CoupledGeometry
    ) -> tuple[PortTraceState, ...]:
        horizontal = self.horizontal_component.port_traces(
            state.horizontal, geometry
        )
        supply = self.supply_branch_component.port_trace(
            state.supply_branch, geometry
        )
        vertical = self.vertical_component.port_trace(state.vertical, geometry)
        traces = tuple(horizontal) + (supply, vertical)
        if len(traces) != 6 or len({trace.key for trace in traces}) != 6:
            raise ContractViolation("physical joint owner did not build six unique ports")
        return traces

    def _resolved_traces(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        interfaces: Mapping[PortKey, CapillaryInterfaceOwnership],
    ) -> tuple[PortTraceState, ...]:
        horizontal_interfaces = {
            key: value
            for key, value in interfaces.items()
            if key.port_name in ("main_left", "main_right")
        }
        horizontal = self.horizontal_component.port_traces(
            state.horizontal,
            geometry,
            interfaces_by_port=horizontal_interfaces,
        )
        supply_key = PortKey("air_supply_T", "supply_bottom")
        riser_key = PortKey("riser_T", "riser_bottom")
        supply = self.supply_branch_component.port_trace(
            state.supply_branch,
            geometry,
            interface=interfaces.get(supply_key),
        )
        vertical = self.vertical_component.port_trace(
            state.vertical,
            geometry,
            interface=interfaces.get(riser_key),
        )
        return tuple(horizontal) + (supply, vertical)

    def _acoustic_scales(
        self, traces: tuple[PortTraceState, ...]
    ) -> tuple[PortAcousticScale, ...]:
        result: list[PortAcousticScale] = []
        adapter = self.horizontal_component.adapter
        for trace in traces:
            if trace.component_id in ("horizontal_main", "air_supply_branch"):
                acoustic_area = (
                    trace.liquid_area_m2
                    if trace.liquid_area_m2 > 1.0e-14 * trace.full_area_m2
                    else trace.full_area_m2
                )
                liquid_sound = adapter.celerity_m_s(acoustic_area)
            else:
                # The vertical reduction uses the same frozen OpenFOAM
                # perfectFluid tangent as the Case-1-derived main adapter.
                liquid_sound = adapter.wave_speed_m_s
            gas_sound = math.sqrt(
                trace.gas_absolute_pressure_Pa / trace.gas_density_kg_m3
            )
            result.append(
                PortAcousticScale(
                    key=trace.key,
                    liquid_sound_speed_m_s=liquid_sound,
                    gas_sound_speed_m_s=gas_sound,
                )
            )
        return tuple(result)

    @staticmethod
    def _directional_seeds(state: CoupledState) -> tuple[PortDirectionalSeed, ...]:
        """Preserve the riser's independent gross up/down and signed gas flow."""

        key = PortKey("riser_T", "riser_bottom")
        up_rate = state.vertical.Qup[0]
        down_rate = state.vertical.Qdown[0]
        up_area = state.vertical.Aup[0]
        down_area = state.vertical.Adown[0]
        if up_rate > 0.0 and up_area <= 0.0:
            raise ContractViolation("riser upward bottom rate has no donor area")
        if down_rate > 0.0 and down_area <= 0.0:
            raise ContractViolation("riser downward bottom rate has no donor area")
        gas_momentum = state.vertical.Jg[0]
        gas_mass = state.vertical.Mg[0]
        if gas_momentum != 0.0 and gas_mass <= 0.0:
            raise ContractViolation("riser bottom gas momentum has no gas mass")
        gas_speed = 0.0 if gas_mass <= 0.0 else abs(gas_momentum / gas_mass)
        return (
            PortDirectionalSeed(
                key=key,
                liquid_into_node_m3_s=down_rate,
                liquid_out_of_node_m3_s=up_rate,
                gas_into_node_kg_s=max(-gas_momentum, 0.0),
                gas_out_of_node_kg_s=max(gas_momentum, 0.0),
                liquid_into_node_speed_m_s=(
                    down_rate / down_area if down_rate > 0.0 else 0.0
                ),
                liquid_out_of_node_speed_m_s=(
                    up_rate / up_area if up_rate > 0.0 else 0.0
                ),
                gas_into_node_speed_m_s=(
                    gas_speed if gas_momentum < 0.0 else 0.0
                ),
                gas_out_of_node_speed_m_s=(
                    gas_speed if gas_momentum > 0.0 else 0.0
                ),
            ),
        )

    def build_physical_inputs(
        self, state: CoupledState, geometry: CoupledGeometry
    ) -> PhysicalJointStageInputs:
        """Build all six physical inputs from one immutable state snapshot."""

        geometry.validate_state(state)
        raw = self._raw_traces(state, geometry)
        by_port = self._port_interface_records(raw)
        traces = self._resolved_traces(state, geometry, by_port)
        return PhysicalJointStageInputs(
            traces=traces,
            acoustic_scales=self._acoustic_scales(traces),
            directional_seeds=self._directional_seeds(state),
            interfaces=tuple(by_port[key] for key in sorted(by_port)),
        )

    def prepare_atomic_state(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> CoupledState:
        """Apply only a conservative zero-time exterior rim-event relabel."""

        del dt_s
        if physical_stage not in ("stage1_closed", "stage2_pressure_reservoir"):
            raise ContractViolation("unsupported physical stage")
        geometry.validate_state(state)
        plume = self.exterior_plume_owner.prepare_atomic_state(
            state.exterior_plume, geometry
        )
        if plume == state.exterior_plume:
            return state
        prepared = replace(state, exterior_plume=plume)
        geometry.validate_state(prepared)
        return prepared

    def stable_timestep_s(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
    ) -> float:
        """Result-independent CFL ceiling for the three physical components."""

        if physical_stage not in ("stage1_closed", "stage2_pressure_reservoir"):
            raise ContractViolation("unsupported physical stage")
        geometry.validate_state(state)
        horizontal_dt = self.horizontal_component.stable_timestep_s(
            state.horizontal, geometry
        )
        supply_dt = self.supply_branch_component.stable_timestep_s(
            state.supply_branch
        )
        liquid_sound = self.horizontal_component.adapter.wave_speed_m_s
        gas_rt = (
            self.vertical_component.atmospheric_top.gas_constant_J_kg_K
            * self.vertical_component.atmospheric_top.temperature_K
        )
        maximum_speed = max(liquid_sound, math.sqrt(gas_rt))
        for up, qup, down, qdown, mass, momentum in zip(
            state.vertical.Aup,
            state.vertical.Qup,
            state.vertical.Adown,
            state.vertical.Qdown,
            state.vertical.Mg,
            state.vertical.Jg,
            strict=True,
        ):
            if up > 0.0:
                maximum_speed = max(maximum_speed, qup / up + liquid_sound)
            if down > 0.0:
                maximum_speed = max(maximum_speed, qdown / down + liquid_sound)
            if mass > 0.0:
                maximum_speed = max(
                    maximum_speed, abs(momentum / mass) + math.sqrt(gas_rt)
                )
        vertical_dt = (
            self.cfl_safety * min(geometry.vertical_dz_m) / maximum_speed
        )
        exterior_dt = self.exterior_plume_owner.stable_timestep_s(
            state.exterior_plume, geometry
        )
        result = min(horizontal_dt, supply_dt, vertical_dt, exterior_dt)
        if not math.isfinite(result) or result <= 0.0:
            raise ContractViolation("physical joint owner returned an invalid timestep")
        return result

    def evaluate(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> JointStageRate:
        """Return one unique physical rate or fail before any shared commit."""

        if rk_stage not in (1, 2):
            raise ContractViolation("physical joint owner requires RK stage 1 or 2")
        solved, stage_vertical = self._solve_node_stage(
            state,
            geometry,
            physical_stage=physical_stage,
            rk_stage=rk_stage,
            dt_s=dt_s,
        )
        proposals = {
            proposal.component_id: proposal
            for proposal in solved.component_proposals
        }
        if set(proposals) != {
            "horizontal_main",
            "air_supply_branch",
            "vertical_riser",
        }:
            raise ContractViolation("physical node solve did not return all three proposals")
        horizontal = proposals["horizontal_main"].delta
        supply = proposals["air_supply_branch"].delta
        vertical = proposals["vertical_riser"].delta
        if not isinstance(horizontal, HorizontalDelta):
            raise ContractViolation("physical horizontal proposal has wrong delta type")
        if not isinstance(supply, SupplyBranchDelta):
            raise ContractViolation("physical supply proposal has wrong delta type")
        if not isinstance(vertical, VerticalDelta):
            raise ContractViolation("physical vertical proposal has wrong delta type")
        replay = stage_vertical.evaluate_joint_stage(
            state.vertical,
            geometry,
            riser_node_trial=solved.riser_trial,
            physical_stage=physical_stage,
            dt_s=dt_s,
        )
        if replay.proposal != proposals["vertical_riser"]:
            raise ContractViolation(
                "accepted vertical trial replay changed the physical proposal"
            )
        if replay.diagnostics is None:
            raise ContractViolation(
                "accepted vertical trial replay returned no rim diagnostics"
            )
        exterior = self.exterior_plume_owner.evaluate_stage(
            state.exterior_plume,
            geometry,
            top_liquid=replay.diagnostics.top_liquid,
            dt_s=dt_s,
        )
        return JointStageRate(
            physical_stage=physical_stage,
            horizontal=horizontal,
            supply_branch=supply,
            vertical=vertical,
            exterior_plume=exterior.delta,
            air_supply_node=solved.air_node,
            riser_node=solved.riser_node,
            horizontal_external=proposals["horizontal_main"].external_exchange,
            supply_external=proposals["air_supply_branch"].external_exchange,
            vertical_external=proposals["vertical_riser"].external_exchange,
            exterior_plume_exchange=exterior.component_exchange,
            air_supply_node_common_absolute_pressure_Pa=(
                solved.air_trial.common_absolute_pressure_Pa
            ),
            riser_node_common_absolute_pressure_Pa=(
                solved.riser_trial.common_absolute_pressure_Pa
            ),
            evidence_status=self.evidence_status,
        )

    def _solve_node_stage(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
    ) -> tuple[SolvedJointNodeStage, object]:
        """Pure common node solve shared by physics and state diagnostics."""

        inputs = self.build_physical_inputs(state, geometry)
        fallback = self.exterior_plume_owner.finite_reentry_fallback(
            state.exterior_plume, geometry, dt_s=dt_s
        )
        stage_vertical = self.vertical_component.with_liquid_fallback(fallback)
        solved = self.two_tnode_solver.solve_pure_stage(
            state,
            geometry,
            physical_stage=physical_stage,
            rk_stage=rk_stage,
            dt_s=dt_s,
            traces=inputs.traces,
            acoustic_scales=inputs.acoustic_scales,
            component_operators=(
                self.horizontal_component,
                self.supply_branch_component,
                stage_vertical,
            ),
            directional_seeds=inputs.directional_seeds,
            interfaces=inputs.interfaces,
        )
        return solved, stage_vertical

    def diagnostic_node_pressures(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        diagnostic_dt_s: float,
    ) -> tuple[float, float]:
        """Return instantaneous post-state node pressures without a commit.

        ``diagnostic_dt_s`` is the actual accepted interval and is passed to
        the same capacity-aware algebraic closure.  Component proposals and
        boundary exchanges produced while closing that algebraic problem are
        discarded: this method neither advances state nor appends a ledger.
        """

        solved, _ = self._solve_node_stage(
            state,
            geometry,
            physical_stage=physical_stage,
            rk_stage=1,
            dt_s=diagnostic_dt_s,
        )
        return (
            solved.air_trial.common_absolute_pressure_Pa,
            solved.riser_trial.common_absolute_pressure_Pa,
        )


__all__ = [
    "F0PhysicalTwoTNodeStageOwner",
    "PhysicalJointStageInputs",
]
