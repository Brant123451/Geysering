"""Pure simultaneous closure for the two zero-storage S1 T junctions.

The closure owns neither component state nor a conservation ledger.  For one
immutable whole-network RK stage it solves the four algebraic unknowns

``(p_air, alpha_g_air, p_riser, alpha_g_riser)``

from zero liquid-volume and zero gas-mass storage at both nodes.  Every
residual evaluation constructs both :class:`TNodeTrial` objects with the same
base token, physical stage, RK index and ``dt`` and evaluates all supplied
component proposal operators before the nonlinear iterate may be accepted.

Port rates are the unfitted linear-acoustic characteristic translation of the
immutable component traces.  A component-to-node rate uses the component
phase area and density; a node-to-component rate uses the solved node phase
area and the isothermal node gas density.  Optional gross directional seeds
preserve the riser's independent up/down streams instead of reconstructing
them from a net discharge.  They add no node volume or inertia.

The pressure brackets and convergence scales are not user knobs.  They are
recomputed from the trial absolute pressures, densities, frozen wave speeds,
port areas and machine precision.  Failure of a component capacity gate,
non-positive pressure, an out-of-range phase fraction or nonlinear
convergence fails the complete two-node group without a state update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .errors import ContractViolation
from .flux import state_token
from .joint_network_runner import (
    GrossComponentPortFlux,
    ZeroStorageTNodeSolution,
)
from .port_contracts import (
    AIR_NODE_PORT_NAMES,
    RISER_NODE_PORT_NAMES,
    CapillaryInterfaceOwnership,
    ComponentStageProposal,
    GrossNodePortFlux,
    NodeName,
    PhysicalStage,
    PortKey,
    PortTraceState,
    TNodeTrial,
    evaluate_component_trial_pure,
    validate_trial_set,
)
from .state import CoupledGeometry, CoupledState


_EPS = np.finfo(float).eps
_SQRT_EPS = math.sqrt(_EPS)
_MAX_NEWTON_ITERATIONS = 96
_MAX_LINE_SEARCH_BISECTIONS = 52
_ATOMIC_NODE_LIQUID_TOLERANCE_M3_S = 5.0e-13
_ATOMIC_NODE_GAS_TOLERANCE_KG_S = 5.0e-13


class JointNodeSolveFailure(ContractViolation):
    """Fail-closed refusal of one complete two-node stage evaluation."""


@dataclass(frozen=True, slots=True)
class PortAcousticScale:
    """Frozen phase characteristic speeds for one immutable port trace."""

    key: PortKey
    liquid_sound_speed_m_s: float
    gas_sound_speed_m_s: float

    def __post_init__(self) -> None:
        for name in ("liquid_sound_speed_m_s", "gas_sound_speed_m_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractViolation(f"{self.key.label} {name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PortDirectionalSeed:
    """Persistent gross streams presented to the pressure correction.

    The riser bottom may simultaneously carry ``Qup`` away from the node and
    ``Qdown`` into it.  These gross streams must survive the node solve.  A
    zero seed is the ordinary single-characteristic port.
    """

    key: PortKey
    liquid_into_node_m3_s: float = 0.0
    liquid_out_of_node_m3_s: float = 0.0
    gas_into_node_kg_s: float = 0.0
    gas_out_of_node_kg_s: float = 0.0
    liquid_into_node_speed_m_s: float = 0.0
    liquid_out_of_node_speed_m_s: float = 0.0
    gas_into_node_speed_m_s: float = 0.0
    gas_out_of_node_speed_m_s: float = 0.0

    def __post_init__(self) -> None:
        pairs = (
            ("liquid into", self.liquid_into_node_m3_s, self.liquid_into_node_speed_m_s),
            ("liquid out", self.liquid_out_of_node_m3_s, self.liquid_out_of_node_speed_m_s),
            ("gas into", self.gas_into_node_kg_s, self.gas_into_node_speed_m_s),
            ("gas out", self.gas_out_of_node_kg_s, self.gas_out_of_node_speed_m_s),
        )
        for label, raw_rate, raw_speed in pairs:
            rate = float(raw_rate)
            speed = float(raw_speed)
            if not math.isfinite(rate) or rate < 0.0:
                raise ContractViolation(f"{self.key.label} seed {label} rate is invalid")
            if not math.isfinite(speed) or speed < 0.0:
                raise ContractViolation(f"{self.key.label} seed {label} speed is invalid")
            if (rate == 0.0) != (speed == 0.0):
                raise ContractViolation(
                    f"{self.key.label} seed {label} rate/speed must vanish together"
                )


@dataclass(frozen=True, slots=True)
class NodeNumericalScale:
    node_name: NodeName
    pressure_lower_Pa: float
    pressure_upper_Pa: float
    pressure_scale_Pa: float
    liquid_rate_scale_m3_s: float
    gas_rate_scale_kg_s: float
    pressure_tolerance_Pa: float
    liquid_residual_tolerance_m3_s: float
    gas_residual_tolerance_kg_s: float
    area_fraction_tolerance: float
    derivation: str = (
        "machine_sqrt_eps_times_scales_from_absolute_p_rho_c_dt_and_port_area"
    )


@dataclass(frozen=True, slots=True)
class JointNodeSolveDiagnostics:
    iterations: int
    normalized_residual_inf: float
    air_scale: NodeNumericalScale
    riser_scale: NodeNumericalScale
    component_evaluations: int
    evidence_status: str = "S1-1D-F0_result_independent_two_node_algebraic_gate"


@dataclass(frozen=True, slots=True)
class SolvedJointNodeStage:
    air_trial: TNodeTrial
    riser_trial: TNodeTrial
    component_proposals: tuple[ComponentStageProposal, ...]
    air_node: ZeroStorageTNodeSolution
    riser_node: ZeroStorageTNodeSolution
    diagnostics: JointNodeSolveDiagnostics


class ComponentTrialOperator(Protocol):
    component_id: str

    def evaluate_trial(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> ComponentStageProposal:
        ...


def _expected_ports(node_name: NodeName) -> frozenset[str]:
    return AIR_NODE_PORT_NAMES if node_name == "air_supply_T" else RISER_NODE_PORT_NAMES


def _weighted_speed(rate_a: float, speed_a: float, rate_b: float, speed_b: float) -> float:
    total = rate_a + rate_b
    if total == 0.0:
        return 0.0
    return (rate_a * speed_a + rate_b * speed_b) / total


def _axis_projection(trace: PortTraceState) -> float:
    # S1 ports are axis-aligned.  The stored axial velocity is +x in the main
    # and +z in both vertical components.
    if abs(trace.normal_into_node_x) > 0.5:
        return trace.normal_into_node_x
    return trace.normal_into_node_z


class F0SimultaneousTwoTNodeSolver:
    """Result-independent, side-effect-free four-unknown node solver."""

    # The algebraic closure is now consumed by the real six-port physical
    # owner.  It still cannot advertise global production readiness because
    # the explicitly listed generalized topology/end-boundary and result gates
    # remain unresolved. Persistent exterior storage/re-entry is now owned by
    # the whole-network physical stage owner, not by this algebraic kernel.
    algebraic_gate_ready = True
    physical_owner_compatible = True
    production_ready = False
    validation_only = True
    closure_set_id = "S1-1D-F0"
    upstream_production_blockers = (
        "general bottom-piston/top-spill topology and water-end phase re-entry remain fail-closed",
        "canonical trajectory observer/exporter and result acceptance remain missing",
    )

    @staticmethod
    def _scale(
        node_name: NodeName,
        traces: Sequence[PortTraceState],
        acoustic: Mapping[PortKey, PortAcousticScale],
        dt_s: float,
    ) -> NodeNumericalScale:
        # dt is part of the frozen stage scale and is deliberately validated,
        # although a zero-storage rate tolerance must not shrink with dt.
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ContractViolation("joint node dt must be positive")
        pressures: list[float] = []
        acoustic_pressures: list[float] = []
        liquid_rates: list[float] = []
        gas_rates: list[float] = []
        for trace in traces:
            scale = acoustic[trace.key]
            pressures.extend(
                (trace.liquid_absolute_pressure_Pa, trace.gas_absolute_pressure_Pa)
            )
            acoustic_pressures.extend(
                (
                    trace.liquid_density_kg_m3 * scale.liquid_sound_speed_m_s**2,
                    trace.gas_density_kg_m3 * scale.gas_sound_speed_m_s**2,
                )
            )
            liquid_rates.append(trace.full_area_m2 * scale.liquid_sound_speed_m_s)
            gas_rates.append(
                trace.full_area_m2
                * trace.gas_density_kg_m3
                * scale.gas_sound_speed_m_s
            )
        p_scale = max((*pressures, *acoustic_pressures))
        p_low = _EPS * p_scale
        p_high = max(pressures) + p_scale
        q_scale = sum(liquid_rates)
        m_scale = sum(gas_rates)
        return NodeNumericalScale(
            node_name=node_name,
            pressure_lower_Pa=p_low,
            pressure_upper_Pa=p_high,
            pressure_scale_Pa=p_scale,
            liquid_rate_scale_m3_s=q_scale,
            gas_rate_scale_kg_s=m_scale,
            pressure_tolerance_Pa=_SQRT_EPS * p_scale,
            liquid_residual_tolerance_m3_s=min(
                _SQRT_EPS * q_scale,
                _ATOMIC_NODE_LIQUID_TOLERANCE_M3_S,
            ),
            gas_residual_tolerance_kg_s=min(
                _SQRT_EPS * m_scale,
                _ATOMIC_NODE_GAS_TOLERANCE_KG_S,
            ),
            area_fraction_tolerance=_SQRT_EPS,
        )

    @staticmethod
    def _validate_inputs(
        traces: tuple[PortTraceState, ...],
        acoustic: Mapping[PortKey, PortAcousticScale],
        seeds: Mapping[PortKey, PortDirectionalSeed],
    ) -> dict[NodeName, tuple[PortTraceState, ...]]:
        keys = [trace.key for trace in traces]
        if len(set(keys)) != len(keys):
            raise ContractViolation("joint node traces contain duplicate ports")
        expected = {
            PortKey(node, name)
            for node in ("air_supply_T", "riser_T")
            for name in _expected_ports(node)
        }
        if set(keys) != expected:
            raise ContractViolation("joint node solve requires all six frozen T ports")
        if set(acoustic) != expected:
            raise ContractViolation("one frozen acoustic scale is required for every T port")
        if any(item.key != key for key, item in acoustic.items()):
            raise ContractViolation("acoustic-scale mapping keys are inconsistent")
        if any(item.key != key for key, item in seeds.items()):
            raise ContractViolation("directional-seed mapping keys are inconsistent")
        if any(key not in expected for key in seeds):
            raise ContractViolation("directional seed belongs to an unknown T port")
        return {
            node: tuple(trace for trace in traces if trace.key.node_name == node)
            for node in ("air_supply_T", "riser_T")
        }

    @staticmethod
    def _phase_rate(
        *,
        base_into: float,
        base_out: float,
        base_into_speed: float,
        base_out_speed: float,
        characteristic_velocity_into_node_m_s: float,
        trace_area_m2: float,
        node_area_m2: float,
        trace_density_kg_m3: float | None,
        node_density_kg_m3: float | None,
    ) -> tuple[float, float, float, float]:
        velocity = characteristic_velocity_into_node_m_s
        if velocity >= 0.0:
            density = 1.0 if trace_density_kg_m3 is None else trace_density_kg_m3
            correction = trace_area_m2 * velocity * density
            into = base_into + correction
            out = base_out
            into_speed = _weighted_speed(
                base_into, base_into_speed, correction, velocity
            )
            out_speed = base_out_speed
        else:
            speed = -velocity
            density = 1.0 if node_density_kg_m3 is None else node_density_kg_m3
            correction = node_area_m2 * speed * density
            into = base_into
            out = base_out + correction
            into_speed = base_into_speed
            out_speed = _weighted_speed(base_out, base_out_speed, correction, speed)
        return into, out, into_speed, out_speed

    def _port_flux(
        self,
        trace: PortTraceState,
        acoustic: PortAcousticScale,
        seed: PortDirectionalSeed,
        *,
        common_pressure_Pa: float,
        node_gas_area_fraction: float,
    ) -> GrossNodePortFlux:
        if common_pressure_Pa <= 0.0 or not math.isfinite(common_pressure_Pa):
            raise JointNodeSolveFailure("node trial pressure became non-positive")
        if not 0.0 <= node_gas_area_fraction <= 1.0:
            raise JointNodeSolveFailure("node gas area fraction left [0,1]")
        projection = _axis_projection(trace)
        node_gas_area = node_gas_area_fraction * trace.full_area_m2
        node_liquid_area = trace.full_area_m2 - node_gas_area

        liquid_seeded = (
            seed.liquid_into_node_m3_s + seed.liquid_out_of_node_m3_s > 0.0
        )
        gas_seeded = seed.gas_into_node_kg_s + seed.gas_out_of_node_kg_s > 0.0
        liquid_u = (
            (0.0 if liquid_seeded else trace.liquid_axial_velocity_m_s * projection)
            + (trace.liquid_absolute_pressure_Pa - common_pressure_Pa)
            / (trace.liquid_density_kg_m3 * acoustic.liquid_sound_speed_m_s)
        )
        gas_u = (
            (0.0 if gas_seeded else trace.gas_axial_velocity_m_s * projection)
            + (trace.gas_absolute_pressure_Pa - common_pressure_Pa)
            / (trace.gas_density_kg_m3 * acoustic.gas_sound_speed_m_s)
        )
        specific_rt = trace.gas_absolute_pressure_Pa / trace.gas_density_kg_m3
        node_gas_density = common_pressure_Pa / specific_rt

        li, lo, lis, los = self._phase_rate(
            base_into=seed.liquid_into_node_m3_s,
            base_out=seed.liquid_out_of_node_m3_s,
            base_into_speed=seed.liquid_into_node_speed_m_s,
            base_out_speed=seed.liquid_out_of_node_speed_m_s,
            characteristic_velocity_into_node_m_s=liquid_u,
            trace_area_m2=trace.liquid_area_m2,
            node_area_m2=node_liquid_area,
            trace_density_kg_m3=None,
            node_density_kg_m3=None,
        )
        gi, go, gis, gos = self._phase_rate(
            base_into=seed.gas_into_node_kg_s,
            base_out=seed.gas_out_of_node_kg_s,
            base_into_speed=seed.gas_into_node_speed_m_s,
            base_out_speed=seed.gas_out_of_node_speed_m_s,
            characteristic_velocity_into_node_m_s=gas_u,
            trace_area_m2=trace.gas_area_m2,
            node_area_m2=node_gas_area,
            trace_density_kg_m3=trace.gas_density_kg_m3,
            node_density_kg_m3=node_gas_density,
        )
        advective = (
            trace.liquid_density_kg_m3 * (li * lis + lo * los)
            + gi * gis
            + go * gos
        )
        pressure = common_pressure_Pa * trace.full_area_m2
        return GrossNodePortFlux(
            key=trace.key,
            liquid_into_node_m3_s=li,
            liquid_out_of_node_m3_s=lo,
            gas_into_node_kg_s=gi,
            gas_out_of_node_kg_s=go,
            liquid_into_node_speed_m_s=lis,
            liquid_out_of_node_speed_m_s=los,
            gas_into_node_speed_m_s=gis,
            gas_out_of_node_speed_m_s=gos,
            advective_momentum_to_node_x_N=trace.normal_into_node_x * advective,
            advective_momentum_to_node_z_N=trace.normal_into_node_z * advective,
            pressure_traction_to_node_x_N=trace.normal_into_node_x * pressure,
            pressure_traction_to_node_z_N=trace.normal_into_node_z * pressure,
        )

    @staticmethod
    def _component_trials(
        operator: ComponentTrialOperator, trials: tuple[TNodeTrial, ...]
    ) -> tuple[TNodeTrial, ...]:
        component_id = operator.component_id
        owned = tuple(
            trial
            for trial in trials
            if any(trace.component_id == component_id for trace in trial.port_traces)
        )
        if not owned:
            raise ContractViolation(f"component {component_id!r} owns no frozen T port")
        return owned

    def _evaluate_components(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
        operators: tuple[ComponentTrialOperator, ...],
    ) -> tuple[ComponentStageProposal, ...]:
        proposals: list[ComponentStageProposal] = []
        for operator in operators:
            owned = self._component_trials(operator, trials)
            proposal = evaluate_component_trial_pure(operator, state, geometry, owned)
            if not proposal.committable:
                rejection = proposal.capacity_reject
                detail = "component capacity rejection"
                if rejection is not None:
                    detail = f"{rejection.component_id}:{rejection.reason_code}: {rejection.detail}"
                raise JointNodeSolveFailure(detail)
            proposals.append(proposal)
        return tuple(proposals)

    @staticmethod
    def _accepted_fluxes(
        proposals: tuple[ComponentStageProposal, ...],
    ) -> dict[PortKey, GrossNodePortFlux]:
        fluxes = [
            flux for proposal in proposals for flux in proposal.accepted_gross_fluxes
        ]
        result = {flux.key: flux for flux in fluxes}
        if len(result) != len(fluxes):
            raise JointNodeSolveFailure(
                "more than one component accepted the same frozen T port"
            )
        expected = {
            PortKey(node, name)
            for node in ("air_supply_T", "riser_T")
            for name in _expected_ports(node)
        }
        if set(result) != expected:
            raise JointNodeSolveFailure(
                "component proposals do not cover all six frozen T ports"
            )
        return result

    @staticmethod
    def _node_solution(
        node_name: NodeName,
        accepted_fluxes: Mapping[PortKey, GrossNodePortFlux],
    ) -> ZeroStorageTNodeSolution:
        flux_tuple = tuple(
            accepted_fluxes[PortKey(node_name, name)]
            for name in sorted(_expected_ports(node_name))
        )
        ports = tuple(
            GrossComponentPortFlux(
                name=flux.key.port_name,
                liquid_into_component_m3_s=flux.liquid_out_of_node_m3_s,
                liquid_out_of_component_m3_s=flux.liquid_into_node_m3_s,
                gas_into_component_kg_s=flux.gas_out_of_node_kg_s,
                gas_out_of_component_kg_s=flux.gas_into_node_kg_s,
                liquid_into_speed_m_s=flux.liquid_out_of_node_speed_m_s,
                liquid_out_speed_m_s=flux.liquid_into_node_speed_m_s,
                gas_into_speed_m_s=flux.gas_out_of_node_speed_m_s,
                gas_out_speed_m_s=flux.gas_into_node_speed_m_s,
                mixture_momentum_to_component_x_N=-flux.mixture_momentum_to_node_x_N,
                mixture_momentum_to_component_z_N=-flux.mixture_momentum_to_node_z_N,
            )
            for flux in flux_tuple
        )
        reaction_x = sum(port.mixture_momentum_to_component_x_N for port in ports)
        reaction_z = sum(port.mixture_momentum_to_component_z_N for port in ports)
        return ZeroStorageTNodeSolution(
            name=node_name,
            ports=ports,
            wall_reaction_on_fluid_x_N=reaction_x,
            wall_reaction_on_fluid_z_N=reaction_z,
        )

    def solve_pure_stage(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        *,
        physical_stage: PhysicalStage,
        rk_stage: int,
        dt_s: float,
        traces: Iterable[PortTraceState],
        acoustic_scales: Iterable[PortAcousticScale],
        component_operators: Iterable[ComponentTrialOperator],
        directional_seeds: Iterable[PortDirectionalSeed] = (),
        interfaces: Iterable[CapillaryInterfaceOwnership] = (),
    ) -> SolvedJointNodeStage:
        """Solve one immutable RK-stage trial set or fail without side effects."""

        geometry.validate_state(state)
        base_token = state_token(state)
        trace_tuple = tuple(traces)
        acoustic = {item.key: item for item in acoustic_scales}
        seeds = {item.key: item for item in directional_seeds}
        grouped = self._validate_inputs(trace_tuple, acoustic, seeds)
        operators = tuple(component_operators)
        if {operator.component_id for operator in operators} != {
            "horizontal_main",
            "air_supply_branch",
            "vertical_riser",
        }:
            raise ContractViolation("joint node gate requires exactly the three S1 components")
        if len({operator.component_id for operator in operators}) != len(operators):
            raise ContractViolation("joint node gate received a duplicate component operator")

        interface_tuple = tuple(interfaces)
        interface_by_id = {item.interface_id: item for item in interface_tuple}
        if len(interface_by_id) != len(interface_tuple):
            raise ContractViolation("network capillary interface ownership is duplicated")
        referenced = {
            trace.interface_id for trace in trace_tuple if trace.interface_id is not None
        }
        if referenced != set(interface_by_id):
            raise ContractViolation("joint traces and capillary owner records disagree")

        air_scale = self._scale("air_supply_T", grouped["air_supply_T"], acoustic, dt_s)
        riser_scale = self._scale("riser_T", grouped["riser_T"], acoustic, dt_s)
        scales = {"air_supply_T": air_scale, "riser_T": riser_scale}

        initial: list[float] = []
        alpha_locked: dict[NodeName, float | None] = {}
        for node in ("air_supply_T", "riser_T"):
            node_traces = grouped[node]
            weights: list[float] = []
            pressure_values: list[float] = []
            for trace in node_traces:
                if trace.liquid_area_m2 > 0.0:
                    weights.append(trace.liquid_area_m2)
                    pressure_values.append(trace.liquid_absolute_pressure_Pa)
                if trace.gas_area_m2 > 0.0:
                    weights.append(trace.gas_area_m2)
                    pressure_values.append(trace.gas_absolute_pressure_Pa)
            p0 = (
                sum(w * p for w, p in zip(weights, pressure_values, strict=True))
                / sum(weights)
            )
            gas_area = sum(trace.gas_area_m2 for trace in node_traces)
            full_area = sum(trace.full_area_m2 for trace in node_traces)
            alpha0 = gas_area / full_area
            if gas_area <= _EPS * full_area:
                alpha_locked[node] = 0.0
                alpha0 = 0.0
            elif sum(trace.liquid_area_m2 for trace in node_traces) <= _EPS * full_area:
                alpha_locked[node] = 1.0
                alpha0 = 1.0
            else:
                alpha_locked[node] = None
            initial.extend((math.log(p0 / scales[node].pressure_scale_Pa), alpha0))

        evaluations = 0

        def build_and_evaluate(vector: np.ndarray, iteration: int):
            nonlocal evaluations
            trials: list[TNodeTrial] = []
            for offset, node in ((0, "air_supply_T"), (2, "riser_T")):
                scale = scales[node]
                pressure = math.exp(float(vector[offset])) * scale.pressure_scale_Pa
                alpha = float(vector[offset + 1])
                locked = alpha_locked[node]
                if locked is not None:
                    alpha = locked
                if not (
                    scale.pressure_lower_Pa <= pressure <= scale.pressure_upper_Pa
                ):
                    raise JointNodeSolveFailure(f"{node} pressure left preregistered bracket")
                if not 0.0 <= alpha <= 1.0:
                    raise JointNodeSolveFailure(f"{node} gas area fraction left [0,1]")
                node_traces = grouped[node]
                fluxes = tuple(
                    self._port_flux(
                        trace,
                        acoustic[trace.key],
                        seeds.get(trace.key, PortDirectionalSeed(trace.key)),
                        common_pressure_Pa=pressure,
                        node_gas_area_fraction=alpha,
                    )
                    for trace in node_traces
                )
                node_interface_ids = tuple(
                    dict.fromkeys(
                        trace.interface_id
                        for trace in node_traces
                        if trace.interface_id is not None
                    )
                )
                node_interfaces = tuple(
                    interface_by_id[interface_id]
                    for interface_id in node_interface_ids
                )
                trial = TNodeTrial(
                    trial_id=(
                        f"joint-rk{rk_stage}-iter{iteration:03d}-{node}-"
                        f"p{pressure:.17g}-a{alpha:.17g}"
                    ),
                    base_state_token=base_token,
                    node_name=node,
                    physical_stage=physical_stage,
                    rk_stage=rk_stage,
                    dt_s=dt_s,
                    common_absolute_pressure_Pa=pressure,
                    node_gas_area_fraction=alpha,
                    port_traces=node_traces,
                    gross_fluxes=fluxes,
                    interfaces=node_interfaces,
                )
                trials.append(trial)
            trial_tuple = tuple(trials)
            validate_trial_set(trial_tuple)
            proposals = self._evaluate_components(
                state, geometry, trial_tuple, operators
            )
            accepted = self._accepted_fluxes(proposals)
            residual = []
            for node in ("air_supply_T", "riser_T"):
                scale = scales[node]
                node_fluxes = tuple(
                    accepted[PortKey(node, name)]
                    for name in _expected_ports(node)
                )
                residual.extend(
                    (
                        sum(flux.liquid_net_into_node_m3_s for flux in node_fluxes)
                        / scale.liquid_rate_scale_m3_s,
                        sum(flux.gas_net_into_node_kg_s for flux in node_fluxes)
                        / scale.gas_rate_scale_kg_s,
                    )
                )
            evaluations += 1
            return np.asarray(residual, dtype=float), trial_tuple, proposals

        x = np.asarray(initial, dtype=float)
        last_norm = math.inf
        final_trials: tuple[TNodeTrial, ...] | None = None
        final_proposals: tuple[ComponentStageProposal, ...] | None = None
        for iteration in range(_MAX_NEWTON_ITERATIONS + 1):
            residual, trials_now, proposals_now = build_and_evaluate(x, iteration)
            if not np.all(np.isfinite(residual)):
                raise JointNodeSolveFailure("joint node residual became non-finite")
            norm = float(np.linalg.norm(residual, ord=np.inf))
            last_norm = norm
            convergence_limits = np.asarray(
                (
                    air_scale.liquid_residual_tolerance_m3_s
                    / air_scale.liquid_rate_scale_m3_s,
                    air_scale.gas_residual_tolerance_kg_s
                    / air_scale.gas_rate_scale_kg_s,
                    riser_scale.liquid_residual_tolerance_m3_s
                    / riser_scale.liquid_rate_scale_m3_s,
                    riser_scale.gas_residual_tolerance_kg_s
                    / riser_scale.gas_rate_scale_kg_s,
                ),
                dtype=float,
            )
            if np.all(np.abs(residual) <= convergence_limits):
                final_trials = trials_now
                final_proposals = proposals_now
                break
            if iteration == _MAX_NEWTON_ITERATIONS:
                break

            jacobian = np.empty((4, 4), dtype=float)
            for column in range(4):
                node = "air_supply_T" if column < 2 else "riser_T"
                if column % 2 == 1 and alpha_locked[node] is not None:
                    jacobian[:, column] = 0.0
                    continue
                step = _SQRT_EPS * max(1.0, abs(float(x[column])))
                candidate = x.copy()
                candidate[column] += step
                if column % 2 == 1 and candidate[column] > 1.0:
                    candidate[column] = x[column] - step
                    step = candidate[column] - x[column]
                trial_residual, _, _ = build_and_evaluate(
                    candidate, iteration
                )
                jacobian[:, column] = (trial_residual - residual) / step
            try:
                direction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            except np.linalg.LinAlgError as exc:
                raise JointNodeSolveFailure("joint node Jacobian solve failed") from exc
            if not np.all(np.isfinite(direction)):
                raise JointNodeSolveFailure("joint node Newton direction is non-finite")

            accepted = False
            factor = 1.0
            for _ in range(_MAX_LINE_SEARCH_BISECTIONS):
                candidate = x + factor * direction
                for offset, node in ((0, "air_supply_T"), (2, "riser_T")):
                    scale = scales[node]
                    candidate[offset] = min(
                        math.log(scale.pressure_upper_Pa / scale.pressure_scale_Pa),
                        max(
                            math.log(scale.pressure_lower_Pa / scale.pressure_scale_Pa),
                            candidate[offset],
                        ),
                    )
                    if alpha_locked[node] is None:
                        candidate[offset + 1] = min(1.0, max(0.0, candidate[offset + 1]))
                    else:
                        candidate[offset + 1] = alpha_locked[node]
                candidate_residual, _, _ = build_and_evaluate(candidate, iteration)
                candidate_norm = float(np.linalg.norm(candidate_residual, ord=np.inf))
                if candidate_norm < norm:
                    x = candidate
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                raise JointNodeSolveFailure(
                    "joint two-node nonlinear solve did not find a residual-decreasing step"
                )

        if final_trials is None or final_proposals is None:
            raise JointNodeSolveFailure(
                f"joint two-node nonlinear solve did not converge; normalized residual={last_norm:.6e}"
            )
        air_trial, riser_trial = final_trials
        final_accepted = self._accepted_fluxes(final_proposals)
        air_node = self._node_solution("air_supply_T", final_accepted)
        riser_node = self._node_solution("riser_T", final_accepted)
        # Momentum is closed only by the explicitly recorded rigid-node wall
        # reaction; it is not used as fictitious node inertia.
        if any(
            abs(value) > _SQRT_EPS
            for node in (air_node, riser_node)
            for value in (
                node.residual.mixture_momentum_x_N,
                node.residual.mixture_momentum_z_N,
            )
        ):
            raise JointNodeSolveFailure("explicit T-node wall reaction bookkeeping failed")
        return SolvedJointNodeStage(
            air_trial=air_trial,
            riser_trial=riser_trial,
            component_proposals=final_proposals,
            air_node=air_node,
            riser_node=riser_node,
            diagnostics=JointNodeSolveDiagnostics(
                iterations=iteration,
                normalized_residual_inf=last_norm,
                air_scale=air_scale,
                riser_scale=riser_scale,
                component_evaluations=evaluations,
            ),
        )


__all__ = [
    "ComponentTrialOperator",
    "F0SimultaneousTwoTNodeSolver",
    "JointNodeSolveDiagnostics",
    "JointNodeSolveFailure",
    "NodeNumericalScale",
    "PortAcousticScale",
    "PortDirectionalSeed",
    "SolvedJointNodeStage",
]
