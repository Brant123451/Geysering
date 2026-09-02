"""Narrow integration protocols for the S1 campaign controller.

The concrete observation bridge and production runner are intentionally not
imported here.  They may satisfy these protocols after their independent
physical and diagnostic gates pass; this module cannot make them production
ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable


PhysicalStage = Literal["stage1_closed", "stage2_pressure_reservoir"]


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_mapping(name: str, values: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        label = str(key).strip()
        if not label or label in result:
            raise ValueError(f"{name} contains an empty or duplicate channel")
        result[label] = _finite(f"{name}[{label}]", value)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class BoundaryCommand:
    """One source-aligned boundary command, separate from the physical state."""

    physical_stage: PhysicalStage
    air_port_mode: Literal["closed_wall", "isothermal_pressure_reservoir"]
    air_gauge_pressure_pa: float | None
    evidence_status: str

    def __post_init__(self) -> None:
        if self.physical_stage == "stage1_closed":
            if self.air_port_mode != "closed_wall" or self.air_gauge_pressure_pa is not None:
                raise ValueError("Stage 1 must use a closed air wall and no pressure value")
        elif self.physical_stage == "stage2_pressure_reservoir":
            if self.air_port_mode != "isothermal_pressure_reservoir":
                raise ValueError("Stage 2 must use the pressure-reservoir air boundary")
            pressure = _finite("Stage-2 air gauge pressure", self.air_gauge_pressure_pa)  # type: ignore[arg-type]
            if pressure != 5700.0:
                raise ValueError("the published Stage-2 air gauge pressure is exactly 5700 Pa")
            object.__setattr__(self, "air_gauge_pressure_pa", pressure)
        else:  # pragma: no cover - retained as a runtime fail-closed guard
            raise ValueError(f"unsupported physical stage {self.physical_stage!r}")
        if not self.evidence_status.strip():
            raise ValueError("boundary evidence_status must be non-empty")


@dataclass(frozen=True, slots=True)
class Stage1BoundaryFlows:
    """Positive-forward water boundary rates used by the settling gate."""

    qin_m3_s: float
    qout_m3_s: float
    mdot_in_kg_s: float
    mdot_out_kg_s: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class Stage1Observation:
    """Native accepted-state quantities for the physical Stage-1 gate.

    A future bridge must map each 1-D axial velocity into a physical 3-vector
    at the published P1--P6 locations.  The controller does not fabricate a
    reduced-pressure field or infer velocity from a rendered interface.
    """

    stage1_time_s: float
    gauge_pressures_pa: Mapping[str, float]
    velocity_vectors_m_s: Mapping[str, tuple[float, float, float]]
    boundary_flows: Stage1BoundaryFlows

    def __post_init__(self) -> None:
        time_s = _finite("Stage-1 observation time", self.stage1_time_s)
        if time_s < 0.0:
            raise ValueError("Stage-1 observation time must be non-negative")
        object.__setattr__(self, "stage1_time_s", time_s)
        object.__setattr__(
            self,
            "gauge_pressures_pa",
            _finite_mapping("gauge_pressures_pa", self.gauge_pressures_pa),
        )
        if not isinstance(self.velocity_vectors_m_s, Mapping) or not self.velocity_vectors_m_s:
            raise ValueError("velocity_vectors_m_s must be a non-empty mapping")
        velocity: dict[str, tuple[float, float, float]] = {}
        for key, raw in self.velocity_vectors_m_s.items():
            label = str(key).strip()
            if not label or label in velocity or len(raw) != 3:
                raise ValueError("velocity channel names must be unique and vectors must have 3 components")
            velocity[label] = tuple(
                _finite(f"velocity_vectors_m_s[{label}]", value) for value in raw
            )  # type: ignore[assignment]
        object.__setattr__(
            self,
            "velocity_vectors_m_s",
            MappingProxyType(dict(sorted(velocity.items()))),
        )
        if not isinstance(self.boundary_flows, Stage1BoundaryFlows):
            raise ValueError("boundary_flows must be a Stage1BoundaryFlows packet")


@dataclass(frozen=True, slots=True)
class AcceptedCommonState:
    """A real accepted state at an exact, uninterpolated campaign ceiling."""

    physical_stage: PhysicalStage
    stage_time_s: float
    absolute_time_s: float
    state: Any

    def __post_init__(self) -> None:
        stage_time = _finite("common stage time", self.stage_time_s)
        absolute_time = _finite("common absolute time", self.absolute_time_s)
        if stage_time < 0.0 or absolute_time < 0.0:
            raise ValueError("common times must be non-negative")
        object.__setattr__(self, "stage_time_s", stage_time)
        object.__setattr__(self, "absolute_time_s", absolute_time)


@dataclass(frozen=True, slots=True)
class Stage1AcceptanceCandidate:
    """Stable automatic candidate presented to a trusted acceptance callback."""

    state: Any
    report: Any
    campaign_config_sha256: str


AcceptedStepCallback = Callable[[Any], None]
CommonStateCallback = Callable[[AcceptedCommonState], None]
Stage1AcceptanceCallback = Callable[[Stage1AcceptanceCandidate], bool]


@runtime_checkable
class StateCodec(Protocol):
    """Deterministic checkpoint codec supplied by the eventual runner adapter."""

    @property
    def codec_id(self) -> str: ...

    def time_s(self, state: Any) -> float: ...

    def encode(self, state: Any) -> bytes: ...

    def decode(self, payload: bytes) -> Any: ...


@runtime_checkable
class ExactAdvanceRunner(Protocol):
    """Runner contract that must cap an accepted step at every requested time."""

    @property
    def production_ready(self) -> bool: ...

    def source_initial_state(self, source_contract_path: Path) -> Any: ...

    def advance_exact(
        self,
        state: Any,
        *,
        target_absolute_time_s: float,
        boundary: BoundaryCommand,
        accepted_step_callback: AcceptedStepCallback | None,
    ) -> Any:
        """Return a committed state exactly at ``target_absolute_time_s``.

        The implementation must shorten its physical step before crossing the
        target.  Returning an interpolated state violates this protocol.
        """


@runtime_checkable
class ObservationBridge(Protocol):
    """Read-only bridge from one accepted state to native Stage-1 observables."""

    def observe_stage1(
        self,
        state: Any,
        *,
        stage1_time_s: float,
        boundary: BoundaryCommand,
    ) -> Stage1Observation: ...


__all__ = [
    "AcceptedCommonState",
    "AcceptedStepCallback",
    "BoundaryCommand",
    "CommonStateCallback",
    "ExactAdvanceRunner",
    "ObservationBridge",
    "PhysicalStage",
    "Stage1AcceptanceCallback",
    "Stage1AcceptanceCandidate",
    "Stage1BoundaryFlows",
    "Stage1Observation",
    "StateCodec",
]
