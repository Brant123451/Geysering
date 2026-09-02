"""Riser-mouth liquid-outflow integration and eruption event persistence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from .errors import ContractViolation


@dataclass(frozen=True, slots=True)
class TopOutflowEventSnapshot:
    time_s: float
    gross_liquid_outflow_m3: float
    gross_liquid_inflow_m3: float
    net_liquid_outflow_m3: float
    active_persistence_s: float
    window_liquid_outflow_m3: float
    window_mean_liquid_outflow_m3_s: float
    connected_water_to_mouth: bool
    event_accepted: bool
    event_onset_s: float | None
    acceptance_time_s: float | None


@dataclass(slots=True)
class TopOutflowEventIntegrator:
    """Integrate gross mouth streams without reducing them to signed net flow."""

    required_persistence_s: float = 0.10
    minimum_mean_outflow_m3_s: float = 3.2175924923e-5
    minimum_cumulative_outflow_m3: float = 3.2175924923e-6
    time_s: float = 0.0
    gross_liquid_outflow_m3: float = 0.0
    gross_liquid_inflow_m3: float = 0.0
    active_persistence_s: float = 0.0
    candidate_onset_s: float | None = None
    event_accepted: bool = False
    event_onset_s: float | None = None
    acceptance_time_s: float | None = None
    connected_water_to_mouth: bool = False
    _window_segments: deque[tuple[float, float]] = field(
        default_factory=deque, repr=False
    )
    _window_duration_s: float = field(default=0.0, repr=False)
    _window_volume_m3: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.required_persistence_s <= 0.0:
            raise ContractViolation("required_persistence_s must be positive")
        if self.minimum_mean_outflow_m3_s <= 0.0:
            raise ContractViolation("minimum_mean_outflow_m3_s must be positive")
        if self.minimum_cumulative_outflow_m3 <= 0.0:
            raise ContractViolation("minimum_cumulative_outflow_m3 must be positive")

    def _reset_window(self) -> None:
        self._window_segments.clear()
        self._window_duration_s = 0.0
        self._window_volume_m3 = 0.0

    def _append_to_window(self, dt_s: float, outflow_m3_s: float) -> None:
        self._window_segments.append((dt_s, outflow_m3_s))
        self._window_duration_s += dt_s
        self._window_volume_m3 += dt_s * outflow_m3_s

        excess = self._window_duration_s - self.required_persistence_s
        while excess > 1.0e-15 and self._window_segments:
            segment_dt, segment_q = self._window_segments[0]
            removed_dt = min(excess, segment_dt)
            self._window_duration_s -= removed_dt
            self._window_volume_m3 -= removed_dt * segment_q
            excess -= removed_dt
            if removed_dt + 1.0e-15 >= segment_dt:
                self._window_segments.popleft()
            else:
                self._window_segments[0] = (segment_dt - removed_dt, segment_q)
        if abs(self._window_duration_s - self.required_persistence_s) <= 1.0e-15:
            self._window_duration_s = self.required_persistence_s

    def advance(
        self,
        dt_s: float,
        q_up_out_m3_s: float,
        q_down_in_m3_s: float = 0.0,
        *,
        connected_water_to_mouth: bool,
    ) -> TopOutflowEventSnapshot:
        values = (dt_s, q_up_out_m3_s, q_down_in_m3_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ContractViolation("mouth event inputs must be finite")
        if dt_s <= 0.0 or q_up_out_m3_s < 0.0 or q_down_in_m3_s < 0.0:
            raise ContractViolation("dt must be positive and gross discharges non-negative")
        if not isinstance(connected_water_to_mouth, bool):
            raise ContractViolation("connected_water_to_mouth must be an explicit boolean")

        self.gross_liquid_outflow_m3 += q_up_out_m3_s * dt_s
        self.gross_liquid_inflow_m3 += q_down_in_m3_s * dt_s
        self.connected_water_to_mouth = connected_water_to_mouth
        active = self.connected_water_to_mouth and q_up_out_m3_s > 0.0
        if active:
            if self.active_persistence_s == 0.0:
                self.candidate_onset_s = self.time_s
            self.active_persistence_s += dt_s
            self._append_to_window(dt_s, q_up_out_m3_s)
            window_full = (
                self._window_duration_s + 1.0e-15 >= self.required_persistence_s
            )
            window_mean = (
                self._window_volume_m3 / self._window_duration_s
                if self._window_duration_s > 0.0
                else 0.0
            )
            if (
                not self.event_accepted
                and window_full
                and window_mean + 1.0e-15 >= self.minimum_mean_outflow_m3_s
                and self._window_volume_m3 + 1.0e-15
                >= self.minimum_cumulative_outflow_m3
            ):
                self.event_accepted = True
                self.event_onset_s = self.time_s + dt_s - self.required_persistence_s
                self.acceptance_time_s = self.time_s + dt_s
        else:
            self.active_persistence_s = 0.0
            self.candidate_onset_s = None
            self._reset_window()
        self.time_s += dt_s
        return self.snapshot()

    def snapshot(self) -> TopOutflowEventSnapshot:
        return TopOutflowEventSnapshot(
            time_s=self.time_s,
            gross_liquid_outflow_m3=self.gross_liquid_outflow_m3,
            gross_liquid_inflow_m3=self.gross_liquid_inflow_m3,
            net_liquid_outflow_m3=self.gross_liquid_outflow_m3
            - self.gross_liquid_inflow_m3,
            active_persistence_s=self.active_persistence_s,
            window_liquid_outflow_m3=self._window_volume_m3,
            window_mean_liquid_outflow_m3_s=(
                self._window_volume_m3 / self._window_duration_s
                if self._window_duration_s > 0.0
                else 0.0
            ),
            connected_water_to_mouth=self.connected_water_to_mouth,
            event_accepted=self.event_accepted,
            event_onset_s=self.event_onset_s,
            acceptance_time_s=self.acceptance_time_s,
        )
