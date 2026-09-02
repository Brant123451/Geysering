"""Real S1 runner adapters for the preregistered campaign controller.

This module is deliberately a thin integration layer.  It does not alter a
physical closure or project a state.  The exact runner delegates every step to
``S1JointNetworkRunner``; the codec preserves every IEEE-754 value exactly;
and the Stage-1 observer performs only pure native diagnostic evaluations.

The current physical owner remains non-production.  It can therefore be used
only through the explicit ``preproduction_validation_only`` scope.  A formal
adapter reports ``production_ready=False`` and independently rejects source
initialisation/advance until the underlying operator passes its separate
implementation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from campaign.contracts import (
    BoundaryCommand,
    Stage1BoundaryFlows,
    Stage1Observation,
)
from campaign.orchestrator import CampaignProtocolError
from model.accepted_observation_diagnostics import (
    PUBLISHED_PRESSURE_COORDINATES,
    build_instantaneous_gauge_pressures,
)
from model.errors import ContractViolation, MissingPhysicalClosure
from model.flux import state_token
from model.initialization import S1InitialAssembly, build_s1_initial_assembly
from model.joint_network_runner import (
    AcceptedStepContext,
    CurrentS1PhysicalJointOperator,
    S1JointNetworkRunner,
    build_current_physical_operator,
)
from model.state import (
    CoupledGeometry,
    CoupledState,
    ExteriorPlumeState,
    HorizontalState,
    SupplyBranchState,
    TNodeState,
    VerticalState,
)


RuntimeScope = Literal[
    "preproduction_validation_only",
    "formal_campaign",
]

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CONTRACT_PATH = (
    _MODULE_DIR.parent / "config" / "S1_source_aligned.yaml"
).resolve()
PREPRODUCTION_SMOKE_DURATION_S = 0.02
COMMON_EVENT_CEILING_S = 0.10
_CODEC_SCHEMA = "s1_coupled_state_ieee754_hex_v1"
_SOURCE_CONTRACT_SHA256 = (
    "803944f2250b0d9b05b129337e7b6b0e572123331c8442bfe77d0d2dd5f98904"
)
_SOURCE_GEOMETRY_SHA256 = (
    "5713d87e2fb9514f13a5687c08c790265ab40aa8bfbe496424d6a6f0f9aa04be"
)
_SOURCE_STATE_TOKEN = (
    "816c9df27ca3da01de9eb24547599e0b318ce8d5459de167779a4c2e04d38bcb"
)
_CURRENT_FACTORY_SEAL = object()


class S1CampaignRuntimeError(CampaignProtocolError):
    """The concrete runner adapter violated a source or campaign gate."""


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise S1CampaignRuntimeError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise S1CampaignRuntimeError(f"{name} must be positive")
    return result


def _hex(value: float) -> str:
    result = _finite("checkpoint value", value)
    return result.hex()


def _vector_hex(values: tuple[float, ...]) -> list[str]:
    return [_hex(value) for value in values]


def _decode_hex(name: str, value: Any) -> float:
    if not isinstance(value, str):
        raise S1CampaignRuntimeError(f"checkpoint {name} must be an IEEE-754 hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise S1CampaignRuntimeError(f"checkpoint {name} is not a float hex string") from exc
    return _finite(f"checkpoint {name}", result)


def _decode_vector(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise S1CampaignRuntimeError(f"checkpoint {name} must be a non-empty array")
    return tuple(_decode_hex(f"{name}[{index}]", item) for index, item in enumerate(value))


def _no_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise S1CampaignRuntimeError(f"checkpoint contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _geometry_fingerprint(geometry: CoupledGeometry) -> str:
    payload = {
        "horizontal_dx_m": _vector_hex(geometry.horizontal_dx_m),
        "vertical_dz_m": _vector_hex(geometry.vertical_dz_m),
        "supply_branch_dz_m": _vector_hex(geometry.supply_branch_dz_m),
        "horizontal_area_m2": _hex(geometry.horizontal_area_m2),
        "vertical_area_m2": _hex(geometry.vertical_area_m2),
        "supply_branch_area_m2": _hex(geometry.supply_branch_area_m2),  # type: ignore[arg-type]
        "liquid_density_kg_m3": _hex(geometry.liquid_density_kg_m3),
        "horizontal_elastic_overarea_fraction": _hex(
            geometry.horizontal_elastic_overarea_fraction
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_stage2_tick(origin: float, target: float) -> int:
    """Return the exact canonical 0.1 s tick or fail without time snapping."""

    origin_value = _finite("Stage-2 origin", origin)
    target_value = _finite("Stage-2 target", target)
    elapsed = target_value - origin_value
    index = round(elapsed / COMMON_EVENT_CEILING_S)
    if index < 1:
        raise S1CampaignRuntimeError("Stage-2 target must follow its opening time")
    canonical = origin_value + index * COMMON_EVENT_CEILING_S
    if target_value != canonical:
        raise S1CampaignRuntimeError(
            "formal Stage-2 target is not the canonical unshifted 0.10 s tick; time snapping is forbidden"
        )
    return index


def _matches_controlled_current_factory(operator: object) -> bool:
    """Verify the exact concrete topology assembled by the controlled factory."""

    if type(operator) is not CurrentS1PhysicalJointOperator:
        return False
    from model.atmospheric_exterior_plume import F0AtmosphericExteriorPlumeOwner
    from model.horizontal_two_tee_component import F0HorizontalTwoTeeStageComponent
    from model.physical_joint_owner import F0PhysicalTwoTNodeStageOwner
    from model.simultaneous_two_tnode_solver import F0SimultaneousTwoTNodeSolver
    from model.supply_branch_twophase import SupplyBranchTwoPhaseSolver
    from model.vertical_pressure_void_component import (
        F0VerticalCapillaryOwner,
        F0VerticalPressureVoidStageComponent,
    )

    owner = operator.joint_stage_owner
    horizontal = operator.horizontal_component
    supply = operator.supply_branch_component
    vertical = operator.vertical_component
    nodes = operator.two_tnode_solver
    return bool(
        type(horizontal) is F0HorizontalTwoTeeStageComponent
        and type(supply) is SupplyBranchTwoPhaseSolver
        and type(vertical) is F0VerticalPressureVoidStageComponent
        and type(getattr(vertical, "capillary_owner", None)) is F0VerticalCapillaryOwner
        and type(nodes) is F0SimultaneousTwoTNodeSolver
        and type(owner) is F0PhysicalTwoTNodeStageOwner
        and owner.horizontal_component is horizontal
        and owner.supply_branch_component is supply
        and owner.vertical_component is vertical
        and owner.two_tnode_solver is nodes
        and type(owner.exterior_plume_owner) is F0AtmosphericExteriorPlumeOwner
    )


def _require_keys(name: str, value: Any, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise S1CampaignRuntimeError(f"checkpoint {name} must be an object")
    actual = set(value)
    if actual != expected:
        raise S1CampaignRuntimeError(
            f"checkpoint {name} fields differ: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return value


class S1CoupledStateCodec:
    """Deterministic, lossless codec for one immutable ``CoupledState``.

    Decimal JSON floats are intentionally avoided.  Every value is stored as
    ``float.hex()`` text and decoded with ``float.fromhex()``, so checkpoint
    round trips preserve signed zero and all finite binary values exactly.
    """

    def __init__(self, geometry: CoupledGeometry) -> None:
        if not isinstance(geometry, CoupledGeometry):
            raise S1CampaignRuntimeError("state codec requires CoupledGeometry")
        self.geometry = geometry
        self.geometry_sha256 = _geometry_fingerprint(geometry)

    @property
    def codec_id(self) -> str:
        return f"{_CODEC_SCHEMA}:{self.geometry_sha256}"

    def time_s(self, state: Any) -> float:
        if not isinstance(state, CoupledState):
            raise S1CampaignRuntimeError("state codec requires CoupledState")
        self.geometry.validate_state(state)
        return state.time_s

    def encode(self, state: Any) -> bytes:
        if not isinstance(state, CoupledState):
            raise S1CampaignRuntimeError("state codec requires CoupledState")
        self.geometry.validate_state(state)
        payload = {
            "schema_version": _CODEC_SCHEMA,
            "geometry_sha256": self.geometry_sha256,
            "time_s": _hex(state.time_s),
            "horizontal": {
                name: _vector_hex(getattr(state.horizontal, name))
                for name in HorizontalState.__dataclass_fields__
            },
            "supply_branch": {
                name: _vector_hex(getattr(state.supply_branch, name))
                for name in SupplyBranchState.__dataclass_fields__
            },
            "vertical": {
                name: _vector_hex(getattr(state.vertical, name))
                for name in VerticalState.__dataclass_fields__
            },
            "exterior_plume": {
                name: _hex(getattr(state.exterior_plume, name))
                for name in ExteriorPlumeState.__dataclass_fields__
            },
            "air_supply_node": {
                name: _hex(getattr(state.air_supply_node, name))
                for name in TNodeState.__dataclass_fields__
            },
            "riser_node": {
                name: _hex(getattr(state.riser_node, name))
                for name in TNodeState.__dataclass_fields__
            },
        }
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")

    def decode(self, payload: bytes) -> CoupledState:
        if not isinstance(payload, bytes) or not payload:
            raise S1CampaignRuntimeError("checkpoint payload must be non-empty bytes")
        try:
            raw = json.loads(
                payload.decode("ascii"),
                object_pairs_hook=_no_duplicate_json_object,
            )
        except S1CampaignRuntimeError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise S1CampaignRuntimeError("checkpoint payload is not canonical ASCII JSON") from exc
        root = _require_keys(
            "root",
            raw,
            {
                "schema_version",
                "geometry_sha256",
                "time_s",
                "horizontal",
                "supply_branch",
                "vertical",
                "exterior_plume",
                "air_supply_node",
                "riser_node",
            },
        )
        if root["schema_version"] != _CODEC_SCHEMA:
            raise S1CampaignRuntimeError("checkpoint schema is stale or unsupported")
        if root["geometry_sha256"] != self.geometry_sha256:
            raise S1CampaignRuntimeError("checkpoint geometry fingerprint does not match this runner")
        horizontal = _require_keys(
            "horizontal", root["horizontal"], set(HorizontalState.__dataclass_fields__)
        )
        supply = _require_keys(
            "supply_branch", root["supply_branch"], set(SupplyBranchState.__dataclass_fields__)
        )
        vertical = _require_keys(
            "vertical",
            root["vertical"],
            set(VerticalState.__dataclass_fields__),
        )
        plume = _require_keys(
            "exterior_plume",
            root["exterior_plume"],
            set(ExteriorPlumeState.__dataclass_fields__),
        )
        air_node = _require_keys(
            "air_supply_node",
            root["air_supply_node"],
            set(TNodeState.__dataclass_fields__),
        )
        riser_node = _require_keys(
            "riser_node",
            root["riser_node"],
            set(TNodeState.__dataclass_fields__),
        )
        try:
            state = CoupledState(
                time_s=_decode_hex("time_s", root["time_s"]),
                horizontal=HorizontalState(
                    **{
                        name: _decode_vector(f"horizontal.{name}", horizontal[name])
                        for name in HorizontalState.__dataclass_fields__
                    }
                ),
                supply_branch=SupplyBranchState(
                    **{
                        name: _decode_vector(f"supply_branch.{name}", supply[name])
                        for name in SupplyBranchState.__dataclass_fields__
                    }
                ),
                vertical=VerticalState(
                    **{
                        name: _decode_vector(f"vertical.{name}", vertical[name])
                        for name in VerticalState.__dataclass_fields__
                    }
                ),
                exterior_plume=ExteriorPlumeState(
                    **{
                        name: _decode_hex(f"exterior_plume.{name}", plume[name])
                        for name in ExteriorPlumeState.__dataclass_fields__
                    }
                ),
                air_supply_node=TNodeState(
                    **{
                        name: _decode_hex(f"air_supply_node.{name}", air_node[name])
                        for name in TNodeState.__dataclass_fields__
                    }
                ),
                riser_node=TNodeState(
                    **{
                        name: _decode_hex(f"riser_node.{name}", riser_node[name])
                        for name in TNodeState.__dataclass_fields__
                    }
                ),
            )
            self.geometry.validate_state(state)
        except ContractViolation as exc:
            raise S1CampaignRuntimeError("decoded checkpoint violates the frozen geometry") from exc
        if self.encode(state) != payload:
            raise S1CampaignRuntimeError(
                "checkpoint is not the deterministic canonical encoding of its decoded state"
            )
        return state


class S1CampaignExactAdvanceAdapter:
    """Map campaign event ceilings onto accepted atomic S1 runner steps."""

    def __init__(
        self,
        *,
        assembly: S1InitialAssembly,
        runner: S1JointNetworkRunner,
        scope: RuntimeScope,
        maximum_dt_s: float,
        source_contract_path: Path = DEFAULT_SOURCE_CONTRACT_PATH,
        maximum_steps_per_advance: int = 1_000_000,
        _factory_seal: object | None = None,
    ) -> None:
        if scope not in ("preproduction_validation_only", "formal_campaign"):
            raise S1CampaignRuntimeError(f"unsupported runtime scope {scope!r}")
        if runner.geometry != assembly.geometry:
            raise S1CampaignRuntimeError("runtime assembly and runner geometries differ")
        assembly.geometry.validate_state(assembly.state)
        if (
            _geometry_fingerprint(assembly.geometry) != _SOURCE_GEOMETRY_SHA256
            or state_token(assembly.state) != _SOURCE_STATE_TOKEN
            or assembly.stage != "stage1_closed_air_pressure_driven_water_settling"
            or assembly.air_source_open is not False
        ):
            raise S1CampaignRuntimeError(
                "runtime assembly does not match the frozen source geometry, state and closed-air stage"
            )
        self.assembly = assembly
        self.runner = runner
        self.scope = scope
        self.maximum_dt_s = _positive("maximum_dt_s", maximum_dt_s)
        self.maximum_steps_per_advance = int(maximum_steps_per_advance)
        if self.maximum_steps_per_advance <= 0:
            raise S1CampaignRuntimeError("maximum_steps_per_advance must be positive")
        self.source_contract_path = Path(source_contract_path).resolve()
        self._factory_seal = _factory_seal
        self._source_initialization_count = 0
        self._last_returned_state: CoupledState | None = None
        self._last_returned_token: str | None = None
        self._last_accepted_context: AcceptedStepContext | None = None
        self._accepted_context_count = 0
        self._advance_serial = 0
        self._has_stage1_advance = False
        self._stage2_origin_absolute_s: float | None = None
        self._terminal_fault: str | None = None

    @property
    def production_ready(self) -> bool:
        operator = self.runner.operator
        return bool(
            self.scope == "formal_campaign"
            and self._factory_seal is _CURRENT_FACTORY_SEAL
            and _matches_controlled_current_factory(operator)
            and operator.production_ready is True
            and operator.validation_only is False
            and operator.integration_owner_ready is True
        )

    @property
    def validation_only(self) -> bool:
        return not self.production_ready

    @property
    def source_initialization_count(self) -> int:
        return self._source_initialization_count

    @property
    def accepted_context_count(self) -> int:
        return self._accepted_context_count

    @property
    def last_accepted_context(self) -> AcceptedStepContext | None:
        return self._last_accepted_context

    @property
    def stage2_origin_absolute_s(self) -> float | None:
        return self._stage2_origin_absolute_s

    @property
    def stage1_advance_recorded(self) -> bool:
        return self._has_stage1_advance

    @property
    def terminal_fault(self) -> str | None:
        return self._terminal_fault

    @property
    def last_accepted_state(self) -> CoupledState | None:
        return self._last_returned_state

    def _assert_not_faulted(self) -> None:
        if self._terminal_fault is not None:
            raise S1CampaignRuntimeError(
                "campaign runtime is terminally faulted after an accepted/failed advance: "
                + self._terminal_fault
            )

    def _assert_scope_ready(self) -> None:
        self._assert_not_faulted()
        if self.scope == "formal_campaign" and not self.production_ready:
            raise S1CampaignRuntimeError(
                "formal S1 campaign requires the controlled factory-sealed concrete "
                "CurrentS1PhysicalJointOperator topology to be integration-owner-ready, "
                "non-validation and production_ready"
            )

    def source_initial_state(self, source_contract_path: Path) -> CoupledState:
        self._assert_scope_ready()
        supplied = Path(source_contract_path).resolve()
        if supplied != self.source_contract_path or not supplied.is_file():
            raise S1CampaignRuntimeError(
                "campaign source contract is not the frozen S1_source_aligned.yaml"
            )
        actual_source_sha = hashlib.sha256(supplied.read_bytes()).hexdigest()
        if actual_source_sha != _SOURCE_CONTRACT_SHA256:
            raise S1CampaignRuntimeError(
                "frozen S1 source contract content hash changed; runtime assembly provenance is invalid"
            )
        if self._source_initialization_count != 0:
            raise S1CampaignRuntimeError(
                "source-aligned initial state may be constructed exactly once per campaign runtime"
            )
        self._source_initialization_count = 1
        self._last_returned_state = self.assembly.state
        self._last_returned_token = state_token(self.assembly.state)
        return self.assembly.state

    def _capture_callback(
        self,
        downstream: Callable[[Any], None] | None,
    ) -> Callable[[AcceptedStepContext], None]:
        def capture(context: AcceptedStepContext) -> None:
            if not isinstance(context, AcceptedStepContext):
                raise S1CampaignRuntimeError("joint runner emitted the wrong callback packet")
            if not context.ledger_entries:
                raise S1CampaignRuntimeError("accepted callback omitted its conservation ledger")
            if context.diagnostics.pressure_after is None:
                raise MissingPhysicalClosure("accepted callback omitted native P1-P6")
            if self._last_returned_state is not context.before_state:
                raise S1CampaignRuntimeError(
                    "accepted callback predecessor differs from the runtime-owned state"
                )
            if self._last_returned_token != state_token(context.before_state):
                raise S1CampaignRuntimeError(
                    "accepted callback predecessor token changed in place"
                )
            self._last_returned_state = context.after_state
            self._last_returned_token = state_token(context.after_state)
            self._last_accepted_context = context
            self._accepted_context_count += 1
            if downstream is not None:
                try:
                    downstream(context)
                except BaseException as exc:
                    self._terminal_fault = (
                        "downstream accepted-step callback raised after atomic commit: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

        return capture

    def _validate_transition(
        self,
        state: CoupledState,
        target: float,
        boundary: BoundaryCommand,
    ) -> tuple[str, float | None]:
        self._assert_scope_ready()
        if self._source_initialization_count != 1 or self._last_returned_state is None:
            raise S1CampaignRuntimeError("advance requested before the one source initialisation")
        if state is not self._last_returned_state:
            raise S1CampaignRuntimeError(
                "advance input is not the identical last accepted in-memory state; reinitialisation is forbidden"
            )
        if self._last_returned_token != state_token(state):
            raise S1CampaignRuntimeError(
                "last accepted state was altered in place after ownership was recorded"
            )
        if not isinstance(boundary, BoundaryCommand):
            raise S1CampaignRuntimeError("advance requires a validated BoundaryCommand")
        self.runner.geometry.validate_state(state)
        target_value = _finite("target_absolute_time_s", target)
        if target_value <= state.time_s:
            raise S1CampaignRuntimeError("exact target must be after the accepted state")

        if self.scope == "preproduction_validation_only":
            if boundary.physical_stage != "stage1_closed":
                raise S1CampaignRuntimeError("validation-only smoke cannot open the air source")
            if target_value > PREPRODUCTION_SMOKE_DURATION_S + 2.0e-12:
                raise S1CampaignRuntimeError(
                    "validation-only runtime cannot advance beyond the frozen 0.02 s smoke horizon"
                )
            return "stage1_closed", None

        if boundary.physical_stage == "stage1_closed":
            if self._stage2_origin_absolute_s is not None:
                raise S1CampaignRuntimeError("formal campaign cannot return to Stage 1 after opening air")
            return "stage1_closed", None

        if not self._has_stage1_advance:
            raise S1CampaignRuntimeError("Stage 2 cannot open before a Stage-1 accepted advance")
        origin = (
            state.time_s
            if self._stage2_origin_absolute_s is None
            else self._stage2_origin_absolute_s
        )
        _canonical_stage2_tick(origin, target_value)
        if self._stage2_origin_absolute_s is None:
            self._stage2_origin_absolute_s = origin
        return "stage2_pressure_reservoir", origin

    def advance_exact(
        self,
        state: Any,
        *,
        target_absolute_time_s: float,
        boundary: BoundaryCommand,
        accepted_step_callback: Callable[[Any], None] | None,
    ) -> CoupledState:
        if not isinstance(state, CoupledState):
            raise S1CampaignRuntimeError("exact runner requires CoupledState")
        physical_stage, stage2_origin = self._validate_transition(
            state, target_absolute_time_s, boundary
        )
        target = float(target_absolute_time_s)
        serial = self._advance_serial
        self._advance_serial += 1
        callback_count_before = self._accepted_context_count
        ledger_count_before = len(self.runner.committer.ledger.entries)
        try:
            result = self.runner.advance(
                state,
                duration_s=target - state.time_s,
                maximum_dt_s=self.maximum_dt_s,
                physical_stage=physical_stage,  # type: ignore[arg-type]
                transaction_prefix=f"s1-campaign-{serial:06d}-{physical_stage}",
                require_production=self.scope == "formal_campaign",
                maximum_steps=self.maximum_steps_per_advance,
                accepted_step_callback=self._capture_callback(accepted_step_callback),
                stage2_origin_absolute_s=stage2_origin,
                common_output_interval_s=(
                    COMMON_EVENT_CEILING_S if stage2_origin is not None else None
                ),
                require_native_diagnostics=True,
            )
        except BaseException as exc:
            if self._terminal_fault is None:
                self._terminal_fault = (
                    "joint exact advance raised: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise

        callback_count = self._accepted_context_count - callback_count_before
        ledger_entries = tuple(self.runner.committer.ledger.entries[ledger_count_before:])
        if (
            not result.entries
            or callback_count != len(result.entries)
            or ledger_entries != result.entries
            or self._last_returned_state is not result.state
            or self._last_returned_token != state_token(result.state)
            or result.state.time_s <= state.time_s
        ):
            self._terminal_fault = (
                "joint exact advance returned without one-to-one accepted callback, "
                "ledger, successor-state and strict-time ownership"
            )
            raise S1CampaignRuntimeError(self._terminal_fault)
        if result.state.time_s != target:
            self._terminal_fault = (
                "joint runner missed the exact campaign event ceiling; "
                "time tolerance or interpolation is forbidden"
            )
            raise S1CampaignRuntimeError(self._terminal_fault)
        self._last_returned_state = result.state
        self._last_returned_token = state_token(result.state)
        if physical_stage == "stage1_closed":
            self._has_stage1_advance = True
        return result.state


def _linear_sample(
    coordinates: tuple[float, ...],
    values: tuple[float, ...],
    target: float,
    *,
    label: str,
) -> float:
    if len(coordinates) != len(values) or not coordinates:
        raise MissingPhysicalClosure(f"{label} native samples are incomplete")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ContractViolation(f"{label} coordinates are not strictly increasing")
    tolerance = 5.0e-13 * max(1.0, abs(target))
    if target < coordinates[0] - tolerance or target > coordinates[-1] + tolerance:
        raise MissingPhysicalClosure(f"{label} lies outside native support")
    if target <= coordinates[0]:
        return values[0]
    if target >= coordinates[-1]:
        return values[-1]
    for index, (left, right) in enumerate(zip(coordinates, coordinates[1:])):
        if left - tolerance <= target <= right + tolerance:
            fraction = (target - left) / (right - left)
            return values[index] + fraction * (values[index + 1] - values[index])
    raise MissingPhysicalClosure(f"{label} has no native bracket")


class S1Stage1ObservationBridge:
    """Read native P1--P6, velocity vectors and water-end flows without commit."""

    def __init__(
        self,
        *,
        runner: S1JointNetworkRunner,
        diagnostic_dt_s: float = 1.0e-7,
    ) -> None:
        self.runner = runner
        self.geometry = runner.geometry
        self.operator = runner.operator
        self.diagnostic_dt_s = _positive("Stage-1 diagnostic dt", diagnostic_dt_s)

    def _diagnostic_dt(self, state: CoupledState) -> float:
        stable = getattr(self.operator, "stable_timestep_s", None)
        if not callable(stable):
            return self.diagnostic_dt_s
        return min(
            self.diagnostic_dt_s,
            _positive(
                "Stage-1 native stable diagnostic dt",
                stable(state, self.geometry, physical_stage="stage1_closed"),
            ),
        )

    def _horizontal_velocity_vectors(
        self, state: CoupledState
    ) -> dict[str, tuple[float, float, float]]:
        component = getattr(self.operator, "horizontal_component", None)
        adapter = getattr(component, "adapter", None)
        grid = getattr(adapter, "grid", None)
        if component is None or grid is None:
            raise MissingPhysicalClosure("horizontal native grid is unavailable")
        centres = tuple(
            float(grid.x_left_m) + (index + 0.5) * float(grid.dx_m)
            for index in range(state.horizontal.cell_count)
        )
        speed = tuple(
            discharge / area
            if area > 0.0
            else math.nan
            for area, discharge in zip(
                state.horizontal.Al, state.horizontal.Ql, strict=True
            )
        )
        if not all(math.isfinite(value) for value in speed):
            raise MissingPhysicalClosure("Stage-1 horizontal water velocity is unavailable")
        segments = (
            (0, int(component.air_face)),
            (int(component.air_face), int(component.riser_face)),
            (int(component.riser_face), len(centres)),
        )
        vectors: dict[str, tuple[float, float, float]] = {}
        for name in ("P4", "P5", "P6"):
            target = float(PUBLISHED_PRESSURE_COORDINATES[name][0])
            segment = next(
                (
                    (start, end)
                    for start, end in segments
                    if centres[start] <= target <= centres[end - 1]
                ),
                None,
            )
            if segment is None:
                raise MissingPhysicalClosure(f"{name} has no segment-local velocity bracket")
            start, end = segment
            value = _linear_sample(
                centres[start:end],
                speed[start:end],
                target,
                label=f"{name} horizontal water velocity",
            )
            vectors[name] = (value, 0.0, 0.0)
        return vectors

    def _vertical_velocity_vectors(
        self, state: CoupledState
    ) -> dict[str, tuple[float, float, float]]:
        component = getattr(self.operator, "vertical_component", None)
        port_trace = getattr(component, "port_trace", None)
        if not callable(port_trace):
            raise MissingPhysicalClosure("six-state riser port trace is unavailable")
        # The trace call remains a fail-closed check of the physical riser
        # bottom state, but its scalar liquid velocity is a net value.  A net
        # value can cancel two large countercurrent streams and falsely make a
        # Stage-1 stability channel appear stationary.  The one-vector
        # campaign contract therefore uses the signed velocity of the larger-
        # magnitude persistent liquid population at each location.  This is a
        # conservative non-cancelling diagnostic; it does not combine, relabel
        # or modify Aup/Qup/Adown/Qdown.
        port_trace(state.vertical, self.geometry)

        def representative(up: float, qup: float, down: float, qdown: float) -> float:
            populations: list[float] = []
            if up > 0.0:
                populations.append(qup / up)
            if down > 0.0:
                populations.append(-qdown / down)
            if not populations:
                return math.nan
            return max(populations, key=lambda value: (abs(value), value))

        centres: list[float] = []
        z = 0.0
        for width in self.geometry.vertical_dz_m:
            centres.append(z + 0.5 * width)
            z += width
        velocity = [
            representative(up, qup, down, qdown)
            for up, qup, down, qdown in zip(
                state.vertical.Aup,
                state.vertical.Qup,
                state.vertical.Adown,
                state.vertical.Qdown,
                strict=True,
            )
        ]
        vectors = {"P1": (0.0, 0.0, velocity[0])}
        for name in ("P2", "P3"):
            target = float(PUBLISHED_PRESSURE_COORDINATES[name][1])
            value = _linear_sample(
                tuple(centres),
                tuple(velocity),
                target,
                label=f"{name} riser net liquid velocity",
            )
            if not math.isfinite(value):
                raise MissingPhysicalClosure(f"{name} lies in a gas-only riser cell")
            vectors[name] = (0.0, 0.0, value)
        return vectors

    def observe_stage1(
        self,
        state: Any,
        *,
        stage1_time_s: float,
        boundary: BoundaryCommand,
    ) -> Stage1Observation:
        if not isinstance(state, CoupledState):
            raise S1CampaignRuntimeError("Stage-1 observation requires CoupledState")
        if (
            not isinstance(boundary, BoundaryCommand)
            or boundary.physical_stage != "stage1_closed"
        ):
            raise S1CampaignRuntimeError("Stage-1 observer requires the closed-wall command")
        requested_time = _finite("Stage-1 observation time", stage1_time_s)
        if not math.isclose(
            state.time_s,
            requested_time,
            rel_tol=0.0,
            abs_tol=2.0e-12 * max(1.0, abs(requested_time)),
        ):
            raise S1CampaignRuntimeError("Stage-1 observer was given a shifted state time")
        self.geometry.validate_state(state)
        token_before = state_token(state)
        ledger_before = tuple(self.runner.committer.ledger.entries)
        dt = self._diagnostic_dt(state)
        rate = self.operator.evaluate(
            state,
            self.geometry,
            physical_stage="stage1_closed",
            rk_stage=1,
            dt_s=dt,
        )
        air_pressure = rate.air_supply_node_common_absolute_pressure_Pa
        riser_pressure = rate.riser_node_common_absolute_pressure_Pa
        if air_pressure is None or riser_pressure is None:
            raise MissingPhysicalClosure("Stage-1 native rate omitted a zero-storage-T pressure")
        if min(float(air_pressure), float(riser_pressure)) <= 0.0:
            raise MissingPhysicalClosure("Stage-1 native node pressure is non-positive")
        pressure = build_instantaneous_gauge_pressures(
            state,
            self.geometry,
            horizontal_component=getattr(self.operator, "horizontal_component", None),
            vertical_component=getattr(self.operator, "vertical_component", None),
            riser_node_common_absolute_pressure_Pa=riser_pressure,
        )
        velocities = self._horizontal_velocity_vectors(state)
        velocities.update(self._vertical_velocity_vectors(state))
        qin = rate.horizontal_external.liquid_inflow_m3_s
        qout = rate.horizontal_external.liquid_outflow_m3_s
        rho = self.geometry.liquid_density_kg_m3
        if state_token(state) != token_before:
            raise S1CampaignRuntimeError("Stage-1 observation mutated the accepted state")
        if tuple(self.runner.committer.ledger.entries) != ledger_before:
            raise S1CampaignRuntimeError("Stage-1 observation appended a conservation ledger entry")
        return Stage1Observation(
            stage1_time_s=requested_time,
            gauge_pressures_pa={
                name: float(getattr(pressure, name))
                for name in ("P1", "P2", "P3", "P4", "P5", "P6")
            },
            velocity_vectors_m_s=velocities,
            boundary_flows=Stage1BoundaryFlows(
                qin_m3_s=qin,
                qout_m3_s=qout,
                mdot_in_kg_s=rho * qin,
                mdot_out_kg_s=rho * qout,
            ),
        )


@dataclass(frozen=True, slots=True)
class S1CampaignRuntimeBundle:
    assembly: S1InitialAssembly
    exact_runner: S1CampaignExactAdvanceAdapter
    codec: S1CoupledStateCodec
    stage1_observation_bridge: S1Stage1ObservationBridge


def build_current_s1_campaign_runtime(
    *,
    scope: RuntimeScope,
    maximum_dt_s: float,
    diagnostic_dt_s: float = 1.0e-7,
) -> S1CampaignRuntimeBundle:
    """Build one source state, one real owner, and all campaign adapters."""

    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    exact = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=runner,
        scope=scope,
        maximum_dt_s=maximum_dt_s,
        _factory_seal=_CURRENT_FACTORY_SEAL,
    )
    return S1CampaignRuntimeBundle(
        assembly=assembly,
        exact_runner=exact,
        codec=S1CoupledStateCodec(assembly.geometry),
        stage1_observation_bridge=S1Stage1ObservationBridge(
            runner=runner,
            diagnostic_dt_s=diagnostic_dt_s,
        ),
    )


__all__ = [
    "COMMON_EVENT_CEILING_S",
    "DEFAULT_SOURCE_CONTRACT_PATH",
    "PREPRODUCTION_SMOKE_DURATION_S",
    "RuntimeScope",
    "S1CampaignExactAdvanceAdapter",
    "S1CampaignRuntimeBundle",
    "S1CampaignRuntimeError",
    "S1CoupledStateCodec",
    "S1Stage1ObservationBridge",
    "build_current_s1_campaign_runtime",
]
