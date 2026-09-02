"""Frozen internal-mouth eruption event classifier for the 1-D model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .grid import COMMON_DT_S, AlignmentError, validate_strict_common_grid


ERUPTION_PERSISTENCE_S = 0.10
ERUPTION_MEAN_FLOW_M3_S = 3.2175924923e-5
ERUPTION_VOLUME_M3 = 3.2175924923e-6
_NUMERIC_REL_TOL = 1.0e-12


@dataclass(frozen=True)
class MouthWindowDecision:
    start_s: float
    end_s: float
    duration_s: float
    mean_liquid_outflow_m3_s: float
    cumulative_liquid_outflow_m3: float
    outward_flow_positive: bool
    water_path_connected: bool
    topology_admissible: bool
    conservation_admissible: bool
    mean_threshold_met: bool
    volume_threshold_met: bool
    qualifies: bool


@dataclass(frozen=True)
class EruptionEpisode:
    onset_s: float
    end_s: float
    duration_s: float
    cumulative_liquid_outflow_m3: float
    mean_liquid_outflow_m3_s: float
    qualifying_window_count: int


@dataclass(frozen=True)
class InternalMouthEventDecision:
    """Serializable event result with no unpublished acceptance tolerance."""

    schema_version: int
    event_name: str
    classification: str
    eruption_detected: bool
    first_onset_s: float | None
    event_count: int
    comparison_dt_s: float
    time_shift_applied_s: float
    thresholds: dict[str, float]
    windows: tuple[MouthWindowDecision, ...]
    episodes: tuple[EruptionEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_flow(values: Sequence[float], expected_length: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != expected_length:
        raise AlignmentError("mouth flow and time arrays have different lengths")
    if not all(math.isfinite(value) for value in converted):
        raise AlignmentError("mouth liquid outflow contains NaN or infinity")
    return converted


def _bool_mask(
    values: Sequence[bool], expected_length: int, label: str
) -> tuple[bool, ...]:
    if len(values) != expected_length:
        raise AlignmentError(f"{label} and time arrays have different lengths")
    if any(not isinstance(value, bool) for value in values):
        raise AlignmentError(f"{label} must contain explicit booleans")
    return tuple(values)


def _meets(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=_NUMERIC_REL_TOL, abs_tol=0.0
    )


def classify_internal_mouth_event(
    time_s: Sequence[float],
    mouth_liquid_outflow_m3_s: Sequence[float],
    *,
    water_path_connected: Sequence[bool],
    topology_admissible: Sequence[bool],
    conservation_admissible: Sequence[bool],
) -> InternalMouthEventDecision:
    """Classify the frozen 1-D internal-mouth event on 0.10 s windows.

    Flow is trapezoid-integrated over every adjacent pair of grid samples.
    Both endpoints must retain a connected water path, positive outward flow,
    admissible topology, and admissible conservation ledgers.  Consequently an
    isolated one-sample spike surrounded by zero flow cannot form an event.
    """

    times = validate_strict_common_grid(time_s)
    if len(times) < 2:
        raise AlignmentError("event classification requires at least one 0.10 s window")
    flow = _finite_flow(mouth_liquid_outflow_m3_s, len(times))
    connected = _bool_mask(water_path_connected, len(times), "water_path_connected")
    topology = _bool_mask(topology_admissible, len(times), "topology_admissible")
    conservation = _bool_mask(
        conservation_admissible, len(times), "conservation_admissible"
    )

    windows: list[MouthWindowDecision] = []
    for index in range(len(times) - 1):
        duration = times[index + 1] - times[index]
        if not math.isclose(
            duration, ERUPTION_PERSISTENCE_S, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise AlignmentError("eruption windows must be exactly 0.10 s")
        mean_flow = 0.5 * (flow[index] + flow[index + 1])
        volume = mean_flow * duration
        positive = flow[index] > 0.0 and flow[index + 1] > 0.0
        is_connected = connected[index] and connected[index + 1]
        is_topological = topology[index] and topology[index + 1]
        is_conservative = conservation[index] and conservation[index + 1]
        mean_met = _meets(mean_flow, ERUPTION_MEAN_FLOW_M3_S)
        volume_met = _meets(volume, ERUPTION_VOLUME_M3)
        qualifies = all(
            (
                positive,
                is_connected,
                is_topological,
                is_conservative,
                mean_met,
                volume_met,
            )
        )
        windows.append(
            MouthWindowDecision(
                start_s=times[index],
                end_s=times[index + 1],
                duration_s=duration,
                mean_liquid_outflow_m3_s=mean_flow,
                cumulative_liquid_outflow_m3=volume,
                outward_flow_positive=positive,
                water_path_connected=is_connected,
                topology_admissible=is_topological,
                conservation_admissible=is_conservative,
                mean_threshold_met=mean_met,
                volume_threshold_met=volume_met,
                qualifies=qualifies,
            )
        )

    episodes: list[EruptionEpisode] = []
    start_index: int | None = None
    for index in range(len(windows) + 1):
        active = index < len(windows) and windows[index].qualifies
        if active and start_index is None:
            start_index = index
        if not active and start_index is not None:
            selected = windows[start_index:index]
            total_duration = sum(window.duration_s for window in selected)
            total_volume = sum(
                window.cumulative_liquid_outflow_m3 for window in selected
            )
            episodes.append(
                EruptionEpisode(
                    onset_s=selected[0].start_s,
                    end_s=selected[-1].end_s,
                    duration_s=total_duration,
                    cumulative_liquid_outflow_m3=total_volume,
                    mean_liquid_outflow_m3_s=total_volume / total_duration,
                    qualifying_window_count=len(selected),
                )
            )
            start_index = None

    detected = bool(episodes)
    return InternalMouthEventDecision(
        schema_version=1,
        event_name="internal_mouth_event",
        classification="EVENT_DETECTED" if detected else "NO_EVENT",
        eruption_detected=detected,
        first_onset_s=episodes[0].onset_s if detected else None,
        event_count=len(episodes),
        comparison_dt_s=COMMON_DT_S,
        time_shift_applied_s=0.0,
        thresholds={
            "minimum_mean_liquid_outflow_m3_s": ERUPTION_MEAN_FLOW_M3_S,
            "minimum_cumulative_liquid_outflow_m3": ERUPTION_VOLUME_M3,
            "persistence_s": ERUPTION_PERSISTENCE_S,
        },
        windows=tuple(windows),
        episodes=tuple(episodes),
    )
