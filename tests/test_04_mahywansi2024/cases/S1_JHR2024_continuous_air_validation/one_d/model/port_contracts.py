"""Pure port trial/response contracts for the preregistered S1-1D-F0 model.

The classes in this module carry no physical solver.  They define the data a
future nonlinear two-T-node iteration is allowed to exchange with component
operators.  A trial contains immutable phase traces, phase pressures, gross
port flux guesses and one explicit owner for every capillary interface.  A
component response is either an accepted conservative delta or a capacity
rejection; a rejected response can never contain a committable delta.

No committer or conservation ledger is part of the evaluation protocol.  The
only state link is a deterministic base-state token, so repeated nonlinear
trial evaluations remain proposals rather than hidden updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from .errors import ContractViolation
from .flux import (
    BoundaryExchange,
    HorizontalDelta,
    SupplyBranchDelta,
    VerticalDelta,
    state_token,
)
from .state import CoupledGeometry, CoupledState


F0_CLOSURE_SET_ID = "S1-1D-F0"
F0_SURFACE_TENSION_N_M = 0.072
F0_CAPILLARY_PRODUCTION_STATUS = (
    "planar_2d_interface_records_integrated__global_production_blockers_remain"
)

PhysicalStage = Literal["stage1_closed", "stage2_pressure_reservoir"]
CapillaryGeometryMode = Literal[
    "planar_2d_zeroGradient_walls",
    "circular_3d_pipe",
]
NodeName = Literal["air_supply_T", "riser_T"]
ComponentId = Literal["horizontal_main", "air_supply_branch", "vertical_riser"]
InterfaceOwner = Literal[
    "horizontal_main",
    "air_supply_branch",
    "vertical_riser",
    "air_supply_t_node",
    "riser_t_node",
]
CapacityReason = Literal[
    "cfl",
    "phase_capacity",
    "pressure_bracket",
    "void_mass_pairing",
    "missing_closure",
    "nonfinite_trial",
]
ProposalStatus = Literal["accepted", "capacity_rejected"]
ComponentDelta: TypeAlias = HorizontalDelta | SupplyBranchDelta | VerticalDelta


AIR_NODE_PORT_NAMES = frozenset(("main_left", "main_right", "supply_bottom"))
RISER_NODE_PORT_NAMES = frozenset(("main_left", "main_right", "riser_bottom"))
_COMPONENT_IDS = frozenset(
    ("horizontal_main", "air_supply_branch", "vertical_riser")
)
_INTERFACE_OWNERS = frozenset(
    (
        "horizontal_main",
        "air_supply_branch",
        "vertical_riser",
        "air_supply_t_node",
        "riser_t_node",
    )
)
_CAPACITY_REASONS = frozenset(
    (
        "cfl",
        "phase_capacity",
        "pressure_bracket",
        "void_mass_pairing",
        "missing_closure",
        "nonfinite_trial",
    )
)


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ContractViolation(f"{name} must be positive")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ContractViolation(f"{name} must be non-negative")
    return result


def _expected_ports(node_name: NodeName) -> frozenset[str]:
    if node_name == "air_supply_T":
        return AIR_NODE_PORT_NAMES
    if node_name == "riser_T":
        return RISER_NODE_PORT_NAMES
    raise ContractViolation(f"unsupported S1 T-node name: {node_name!r}")


def _expected_component(key: "PortKey") -> ComponentId:
    if key.port_name in ("main_left", "main_right"):
        return "horizontal_main"
    if key.node_name == "air_supply_T" and key.port_name == "supply_bottom":
        return "air_supply_branch"
    if key.node_name == "riser_T" and key.port_name == "riser_bottom":
        return "vertical_riser"
    raise ContractViolation(f"no component owns frozen port {key.label}")


@dataclass(frozen=True, slots=True, order=True)
class PortKey:
    """Globally qualified T-node port name."""

    node_name: NodeName
    port_name: str

    def __post_init__(self) -> None:
        expected = _expected_ports(self.node_name)
        if self.port_name not in expected:
            raise ContractViolation(
                f"{self.node_name} port {self.port_name!r} is outside the frozen topology"
            )

    @property
    def label(self) -> str:
        return f"{self.node_name}:{self.port_name}"


@dataclass(frozen=True, slots=True)
class CapillaryInterfaceOwnership:
    """Exactly one declared owner and pressure-jump record for one interface.

    The F0 surface tension is frozen from the OpenFOAM translation.  The
    source-aligned comparison mode is explicitly planar two-dimensional:
    OpenFOAM's ``zeroGradient`` alpha wall does not prescribe a contact angle,
    so a declared local curvature and its ``sigma*kappa`` jump are sufficient.
    A circular three-dimensional meniscus still requires an explicit contact
    angle.  An unresolved record remains legal for contract iteration but is
    never production-ready.
    """

    interface_id: str
    owner: InterfaceOwner
    surface_tension_N_m: float = F0_SURFACE_TENSION_N_M
    geometry_mode: CapillaryGeometryMode | None = None
    curvature_1_m: float | None = None
    contact_angle_deg: float | None = None
    pressure_jump_gas_minus_liquid_Pa: float | None = None
    evidence_status: str = F0_CAPILLARY_PRODUCTION_STATUS

    def __post_init__(self) -> None:
        if not self.interface_id.strip():
            raise ContractViolation("capillary interface_id must be non-empty")
        if self.owner not in _INTERFACE_OWNERS:
            raise ContractViolation(f"unsupported capillary interface owner: {self.owner!r}")
        if self.geometry_mode not in (
            None,
            "planar_2d_zeroGradient_walls",
            "circular_3d_pipe",
        ):
            raise ContractViolation(
                f"unsupported capillary geometry mode: {self.geometry_mode!r}"
            )
        sigma = _positive("surface_tension_N_m", self.surface_tension_N_m)
        if not math.isclose(
            sigma, F0_SURFACE_TENSION_N_M, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ContractViolation(
                "S1-1D-F0 surface tension must remain 0.072 N/m"
            )
        object.__setattr__(self, "surface_tension_N_m", sigma)
        if not self.evidence_status.strip():
            raise ContractViolation("capillary evidence_status must be non-empty")

        curvature = self.curvature_1_m
        jump = self.pressure_jump_gas_minus_liquid_Pa
        if curvature is None or jump is None:
            if curvature is not None or jump is not None:
                raise ContractViolation(
                    "capillary curvature and pressure jump must be supplied together"
                )
        else:
            curvature_value = _finite("curvature_1_m", curvature)
            jump_value = _finite("capillary pressure jump", jump)
            expected = sigma * curvature_value
            if not math.isclose(
                jump_value,
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ContractViolation(
                    "capillary jump must satisfy delta_p_sigma = sigma*kappa"
                )
            object.__setattr__(self, "curvature_1_m", curvature_value)
            object.__setattr__(
                self, "pressure_jump_gas_minus_liquid_Pa", jump_value
            )

        if self.contact_angle_deg is not None:
            angle = _finite("contact_angle_deg", self.contact_angle_deg)
            if not 0.0 <= angle <= 180.0:
                raise ContractViolation("contact_angle_deg must lie in [0, 180]")
            object.__setattr__(self, "contact_angle_deg", angle)
        if (
            self.geometry_mode == "planar_2d_zeroGradient_walls"
            and self.contact_angle_deg is not None
        ):
            raise ContractViolation(
                "planar zeroGradient translation must not invent a contact angle"
            )
        if (
            self.geometry_mode == "circular_3d_pipe"
            and curvature is not None
            and self.contact_angle_deg is None
        ):
            raise ContractViolation(
                "circular 3-D capillarity requires an explicit contact angle"
            )

    @property
    def geometrically_resolved(self) -> bool:
        return (
            self.curvature_1_m is not None
            and self.contact_angle_deg is not None
            and self.pressure_jump_gas_minus_liquid_Pa is not None
        )

    @property
    def production_ready(self) -> bool:
        jump_ready = (
            self.curvature_1_m is not None
            and self.pressure_jump_gas_minus_liquid_Pa is not None
        )
        if self.geometry_mode == "planar_2d_zeroGradient_walls":
            return jump_ready and self.contact_angle_deg is None
        if self.geometry_mode == "circular_3d_pipe":
            return jump_ready and self.contact_angle_deg is not None
        return False


@dataclass(frozen=True, slots=True)
class PortTraceState:
    """Immutable component trace presented to a zero-storage node trial.

    Both phase pressures and densities are absolute/positive even if one phase
    has zero geometric area.  The absent-phase values are Riemann ghost data;
    phase presence is decided only by the two complementary areas.
    """

    key: PortKey
    component_id: ComponentId
    normal_into_node_x: float
    normal_into_node_z: float
    full_area_m2: float
    liquid_area_m2: float
    gas_area_m2: float
    liquid_density_kg_m3: float
    gas_density_kg_m3: float
    liquid_absolute_pressure_Pa: float
    gas_absolute_pressure_Pa: float
    liquid_axial_velocity_m_s: float = 0.0
    gas_axial_velocity_m_s: float = 0.0
    interface_id: str | None = None
    evidence_status: str = "S1-1D-F0_port_trace_contract"

    def __post_init__(self) -> None:
        if self.component_id not in _COMPONENT_IDS:
            raise ContractViolation(f"unsupported component_id: {self.component_id!r}")
        if self.component_id != _expected_component(self.key):
            raise ContractViolation(
                f"{self.key.label} belongs to {_expected_component(self.key)}, "
                f"not {self.component_id}"
            )
        nx = _finite("normal_into_node_x", self.normal_into_node_x)
        nz = _finite("normal_into_node_z", self.normal_into_node_z)
        if not math.isclose(math.hypot(nx, nz), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ContractViolation("port normal into the node must be a unit vector")
        object.__setattr__(self, "normal_into_node_x", nx)
        object.__setattr__(self, "normal_into_node_z", nz)

        full = _positive("full_area_m2", self.full_area_m2)
        liquid = _nonnegative("liquid_area_m2", self.liquid_area_m2)
        gas = _nonnegative("gas_area_m2", self.gas_area_m2)
        if liquid > full or gas > full or not math.isclose(
            liquid + gas, full, rel_tol=1.0e-12, abs_tol=1.0e-14
        ):
            raise ContractViolation(
                "port liquid and gas areas must be complementary inside the full area"
            )
        object.__setattr__(self, "full_area_m2", full)
        object.__setattr__(self, "liquid_area_m2", liquid)
        object.__setattr__(self, "gas_area_m2", gas)

        for name in (
            "liquid_density_kg_m3",
            "gas_density_kg_m3",
            "liquid_absolute_pressure_Pa",
            "gas_absolute_pressure_Pa",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("liquid_axial_velocity_m_s", "gas_axial_velocity_m_s"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.interface_id is not None and not self.interface_id.strip():
            raise ContractViolation("trace interface_id must be non-empty when supplied")
        if not self.evidence_status.strip():
            raise ContractViolation("port trace evidence_status must be non-empty")

    @property
    def gas_area_fraction(self) -> float:
        return self.gas_area_m2 / self.full_area_m2

    @property
    def phase_pressure_jump_gas_minus_liquid_Pa(self) -> float:
        return self.gas_absolute_pressure_Pa - self.liquid_absolute_pressure_Pa


@dataclass(frozen=True, slots=True)
class GrossNodePortFlux:
    """Gross phase rates using the F0 convention: positive *into the node*."""

    key: PortKey
    liquid_into_node_m3_s: float = 0.0
    liquid_out_of_node_m3_s: float = 0.0
    gas_into_node_kg_s: float = 0.0
    gas_out_of_node_kg_s: float = 0.0
    liquid_into_node_speed_m_s: float = 0.0
    liquid_out_of_node_speed_m_s: float = 0.0
    gas_into_node_speed_m_s: float = 0.0
    gas_out_of_node_speed_m_s: float = 0.0
    advective_momentum_to_node_x_N: float = 0.0
    advective_momentum_to_node_z_N: float = 0.0
    pressure_traction_to_node_x_N: float = 0.0
    pressure_traction_to_node_z_N: float = 0.0

    def __post_init__(self) -> None:
        pairs = (
            (
                "liquid into node",
                self.liquid_into_node_m3_s,
                self.liquid_into_node_speed_m_s,
            ),
            (
                "liquid out of node",
                self.liquid_out_of_node_m3_s,
                self.liquid_out_of_node_speed_m_s,
            ),
            (
                "gas into node",
                self.gas_into_node_kg_s,
                self.gas_into_node_speed_m_s,
            ),
            (
                "gas out of node",
                self.gas_out_of_node_kg_s,
                self.gas_out_of_node_speed_m_s,
            ),
        )
        for label, raw_rate, raw_speed in pairs:
            rate = _nonnegative(f"{self.key.label} {label} rate", raw_rate)
            speed = _nonnegative(f"{self.key.label} {label} speed", raw_speed)
            if rate == 0.0 and speed != 0.0:
                raise ContractViolation(f"{self.key.label} {label} speed has no rate")
            if rate > 0.0 and speed == 0.0:
                raise ContractViolation(f"{self.key.label} {label} rate needs donor speed")
        for name in (
            "advective_momentum_to_node_x_N",
            "advective_momentum_to_node_z_N",
            "pressure_traction_to_node_x_N",
            "pressure_traction_to_node_z_N",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @property
    def liquid_net_into_node_m3_s(self) -> float:
        return self.liquid_into_node_m3_s - self.liquid_out_of_node_m3_s

    @property
    def gas_net_into_node_kg_s(self) -> float:
        return self.gas_into_node_kg_s - self.gas_out_of_node_kg_s

    @property
    def mixture_momentum_to_node_x_N(self) -> float:
        return self.advective_momentum_to_node_x_N + self.pressure_traction_to_node_x_N

    @property
    def mixture_momentum_to_node_z_N(self) -> float:
        return self.advective_momentum_to_node_z_N + self.pressure_traction_to_node_z_N


@dataclass(frozen=True, slots=True)
class TNodeTrial:
    """One immutable nonlinear trial for one F0 zero-storage T node."""

    trial_id: str
    base_state_token: str
    node_name: NodeName
    physical_stage: PhysicalStage
    rk_stage: int
    dt_s: float
    common_absolute_pressure_Pa: float
    node_gas_area_fraction: float
    port_traces: tuple[PortTraceState, ...]
    gross_fluxes: tuple[GrossNodePortFlux, ...]
    interfaces: tuple[CapillaryInterfaceOwnership, ...] = ()

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.base_state_token.strip():
            raise ContractViolation("T-node trial id and base-state token are required")
        if self.physical_stage not in (
            "stage1_closed",
            "stage2_pressure_reservoir",
        ):
            raise ContractViolation("unsupported F0 physical stage")
        if self.rk_stage not in (1, 2):
            raise ContractViolation("F0 node trial RK stage must be 1 or 2")
        object.__setattr__(self, "dt_s", _positive("T-node trial dt_s", self.dt_s))
        object.__setattr__(
            self,
            "common_absolute_pressure_Pa",
            _positive(
                "common_absolute_pressure_Pa", self.common_absolute_pressure_Pa
            ),
        )
        fraction = _finite("node_gas_area_fraction", self.node_gas_area_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ContractViolation("node_gas_area_fraction must lie in [0, 1]")
        object.__setattr__(self, "node_gas_area_fraction", fraction)

        expected = {PortKey(self.node_name, name) for name in _expected_ports(self.node_name)}
        trace_keys = [trace.key for trace in self.port_traces]
        flux_keys = [flux.key for flux in self.gross_fluxes]
        if len(set(trace_keys)) != len(trace_keys) or set(trace_keys) != expected:
            raise ContractViolation("T-node trial must contain one trace for each frozen port")
        if len(set(flux_keys)) != len(flux_keys) or set(flux_keys) != expected:
            raise ContractViolation("T-node trial must contain one gross flux for each port")

        interface_ids = [interface.interface_id for interface in self.interfaces]
        if len(set(interface_ids)) != len(interface_ids):
            raise ContractViolation(
                "one T-node trial cannot assign more than one owner to an interface"
            )
        interface_map = {
            interface.interface_id: interface for interface in self.interfaces
        }
        referenced = {
            trace.interface_id
            for trace in self.port_traces
            if trace.interface_id is not None
        }
        if referenced != set(interface_map):
            raise ContractViolation(
                "every referenced capillary interface must have exactly one owner record"
            )
        for trace in self.port_traces:
            if trace.interface_id is None:
                continue
            interface = interface_map[trace.interface_id]
            jump = interface.pressure_jump_gas_minus_liquid_Pa
            if jump is not None and not math.isclose(
                trace.phase_pressure_jump_gas_minus_liquid_Pa,
                jump,
                rel_tol=1.0e-12,
                abs_tol=1.0e-10,
            ):
                raise ContractViolation(
                    "port phase pressures disagree with their capillary owner record"
                )

    @property
    def trial_token(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    @property
    def capillarity_production_ready(self) -> bool:
        return all(interface.production_ready for interface in self.interfaces)


def validate_trial_set(trials: tuple[TNodeTrial, ...]) -> None:
    """Validate cross-node consistency and globally unique interface ownership."""

    if not trials:
        raise ContractViolation("at least one T-node trial is required")
    if len({trial.trial_token for trial in trials}) != len(trials):
        raise ContractViolation("duplicate T-node trials are forbidden")
    first = trials[0]
    for trial in trials[1:]:
        if (
            trial.base_state_token != first.base_state_token
            or trial.physical_stage != first.physical_stage
            or trial.rk_stage != first.rk_stage
            or not math.isclose(trial.dt_s, first.dt_s, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ContractViolation(
                "joint component trials must share state, stage, RK index and dt"
            )
    owners: dict[str, InterfaceOwner] = {}
    for trial in trials:
        for interface in trial.interfaces:
            previous = owners.get(interface.interface_id)
            if previous is not None:
                raise ContractViolation(
                    f"interface {interface.interface_id!r} appears in more than one trial; "
                    "one network interface must have one owner record"
                )
            owners[interface.interface_id] = interface.owner


@dataclass(frozen=True, slots=True)
class CapacityReject:
    """Non-committable component refusal returned to the nonlinear solve."""

    component_id: ComponentId
    reason_code: CapacityReason
    detail: str
    requested_dt_s: float
    retryable: bool
    maximum_admissible_dt_s: float | None = None

    def __post_init__(self) -> None:
        if self.component_id not in _COMPONENT_IDS:
            raise ContractViolation(f"unsupported component_id: {self.component_id!r}")
        if self.reason_code not in _CAPACITY_REASONS:
            raise ContractViolation(f"unsupported capacity reason: {self.reason_code!r}")
        if not self.detail.strip():
            raise ContractViolation("capacity rejection detail must be non-empty")
        requested = _positive("requested_dt_s", self.requested_dt_s)
        object.__setattr__(self, "requested_dt_s", requested)
        if self.maximum_admissible_dt_s is not None:
            maximum = _positive(
                "maximum_admissible_dt_s", self.maximum_admissible_dt_s
            )
            if maximum > requested:
                raise ContractViolation(
                    "maximum admissible dt cannot exceed the rejected requested dt"
                )
            object.__setattr__(self, "maximum_admissible_dt_s", maximum)


_DELTA_BY_COMPONENT = {
    "horizontal_main": HorizontalDelta,
    "air_supply_branch": SupplyBranchDelta,
    "vertical_riser": VerticalDelta,
}


@dataclass(frozen=True, slots=True)
class ComponentStageProposal:
    """Pure response to one or more node trials, accepted or rejected."""

    component_id: ComponentId
    base_state_token: str
    trial_tokens: tuple[str, ...]
    status: ProposalStatus
    delta: ComponentDelta | None
    accepted_gross_fluxes: tuple[GrossNodePortFlux, ...] = ()
    external_exchange: BoundaryExchange = BoundaryExchange()
    capacity_reject: CapacityReject | None = None
    evidence_status: str = "S1-1D-F0_pure_component_trial_response"

    def __post_init__(self) -> None:
        if self.component_id not in _COMPONENT_IDS:
            raise ContractViolation(f"unsupported component_id: {self.component_id!r}")
        if not self.base_state_token.strip() or not self.trial_tokens:
            raise ContractViolation(
                "component proposal requires a base-state token and trial token(s)"
            )
        if any(not token.strip() for token in self.trial_tokens):
            raise ContractViolation("component trial tokens must be non-empty")
        if len(set(self.trial_tokens)) != len(self.trial_tokens):
            raise ContractViolation("component trial tokens must be unique")
        if not self.evidence_status.strip():
            raise ContractViolation("component proposal evidence_status must be non-empty")
        flux_keys = [flux.key for flux in self.accepted_gross_fluxes]
        if len(set(flux_keys)) != len(flux_keys):
            raise ContractViolation("accepted gross port flux keys must be unique")

        if self.status == "accepted":
            if self.delta is None or self.capacity_reject is not None:
                raise ContractViolation(
                    "accepted proposal requires a delta and cannot contain a rejection"
                )
            expected = _DELTA_BY_COMPONENT[self.component_id]
            if not isinstance(self.delta, expected):
                raise ContractViolation(
                    f"{self.component_id} proposal carries the wrong delta type"
                )
        elif self.status == "capacity_rejected":
            if self.delta is not None:
                raise ContractViolation(
                    "capacity-rejected proposal cannot contain a committable delta"
                )
            if self.accepted_gross_fluxes:
                raise ContractViolation(
                    "capacity-rejected proposal cannot contain accepted port fluxes"
                )
            if self.external_exchange != BoundaryExchange():
                raise ContractViolation(
                    "capacity-rejected proposal cannot contain external exchange"
                )
            if (
                self.capacity_reject is None
                or self.capacity_reject.component_id != self.component_id
            ):
                raise ContractViolation(
                    "capacity-rejected proposal requires a matching rejection record"
                )
        else:
            raise ContractViolation(f"unsupported component proposal status: {self.status!r}")

    @property
    def committable(self) -> bool:
        return self.status == "accepted" and self.delta is not None

    def validate_against_trials(self, trials: tuple[TNodeTrial, ...]) -> None:
        validate_trial_set(trials)
        if self.base_state_token != trials[0].base_state_token:
            raise ContractViolation("component proposal was built from a stale state")
        expected_tokens = {trial.trial_token for trial in trials}
        if set(self.trial_tokens) != expected_tokens:
            raise ContractViolation("component proposal does not own the supplied trial set")
        allowed_keys = {
            trace.key for trial in trials for trace in trial.port_traces
        }
        if any(flux.key not in allowed_keys for flux in self.accepted_gross_fluxes):
            raise ContractViolation("component proposal returned a flux for an unknown port")
        if self.committable:
            expected_component_keys = {
                trace.key
                for trial in trials
                for trace in trial.port_traces
                if trace.component_id == self.component_id
            }
            if {flux.key for flux in self.accepted_gross_fluxes} != expected_component_keys:
                raise ContractViolation(
                    "accepted proposal must return one gross flux for every owned trial port"
                )
        if self.capacity_reject is not None and not math.isclose(
            self.capacity_reject.requested_dt_s,
            trials[0].dt_s,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ContractViolation("capacity rejection dt differs from the node trial dt")

    @classmethod
    def accepted(
        cls,
        *,
        component_id: ComponentId,
        base_state_token: str,
        trials: tuple[TNodeTrial, ...],
        delta: ComponentDelta,
        accepted_gross_fluxes: tuple[GrossNodePortFlux, ...] = (),
        external_exchange: BoundaryExchange = BoundaryExchange(),
        evidence_status: str = "S1-1D-F0_pure_component_trial_response",
    ) -> "ComponentStageProposal":
        validate_trial_set(trials)
        proposal = cls(
            component_id=component_id,
            base_state_token=base_state_token,
            trial_tokens=tuple(trial.trial_token for trial in trials),
            status="accepted",
            delta=delta,
            accepted_gross_fluxes=accepted_gross_fluxes,
            external_exchange=external_exchange,
            evidence_status=evidence_status,
        )
        proposal.validate_against_trials(trials)
        return proposal

    @classmethod
    def rejected(
        cls,
        *,
        component_id: ComponentId,
        base_state_token: str,
        trials: tuple[TNodeTrial, ...],
        rejection: CapacityReject,
        evidence_status: str = "S1-1D-F0_pure_component_trial_response",
    ) -> "ComponentStageProposal":
        validate_trial_set(trials)
        proposal = cls(
            component_id=component_id,
            base_state_token=base_state_token,
            trial_tokens=tuple(trial.trial_token for trial in trials),
            status="capacity_rejected",
            delta=None,
            capacity_reject=rejection,
            evidence_status=evidence_status,
        )
        proposal.validate_against_trials(trials)
        return proposal


@runtime_checkable
class PureComponentTrialOperator(Protocol):
    """Side-effect-free nonlinear component evaluator; no ledger is passed."""

    component_id: ComponentId

    def evaluate_trial(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> ComponentStageProposal:
        ...


def evaluate_component_trial_pure(
    operator: PureComponentTrialOperator,
    state: CoupledState,
    geometry: CoupledGeometry,
    trials: tuple[TNodeTrial, ...],
) -> ComponentStageProposal:
    """Evaluate and verify token purity without exposing a committer or ledger."""

    validate_trial_set(trials)
    geometry.validate_state(state)
    before = state_token(state)
    if before != trials[0].base_state_token:
        raise ContractViolation("trial set was built from a different coupled state")
    proposal = operator.evaluate_trial(state, geometry, trials)
    after = state_token(state)
    if after != before:
        raise ContractViolation("component trial evaluation mutated the coupled state")
    if proposal.component_id != operator.component_id:
        raise ContractViolation("component operator returned another component's proposal")
    proposal.validate_against_trials(trials)
    return proposal


__all__ = [
    "AIR_NODE_PORT_NAMES",
    "CapacityReject",
    "CapillaryGeometryMode",
    "CapillaryInterfaceOwnership",
    "ComponentDelta",
    "ComponentId",
    "ComponentStageProposal",
    "F0_CAPILLARY_PRODUCTION_STATUS",
    "F0_CLOSURE_SET_ID",
    "F0_SURFACE_TENSION_N_M",
    "GrossNodePortFlux",
    "InterfaceOwner",
    "NodeName",
    "PhysicalStage",
    "PortKey",
    "PortTraceState",
    "PureComponentTrialOperator",
    "RISER_NODE_PORT_NAMES",
    "TNodeTrial",
    "evaluate_component_trial_pure",
    "validate_trial_set",
]
