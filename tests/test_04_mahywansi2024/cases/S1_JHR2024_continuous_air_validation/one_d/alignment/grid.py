"""Strict common-time grid construction and gap-safe interpolation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import math
from typing import Sequence


COMMON_DT_S = 0.10
_TIME_ABS_TOL_S = 1.0e-9


class AlignmentError(ValueError):
    """Base class for a rejected alignment operation."""


class TimeShiftNotAllowedError(AlignmentError):
    """Raised when a caller requests a non-zero physical-time shift."""


class AlignmentGapError(AlignmentError):
    """Raised rather than interpolating across a missing solver interval."""


class AlignmentCoverageError(AlignmentError):
    """Raised when the requested grid lies outside native-series coverage."""


@dataclass(frozen=True)
class AlignedSeries:
    """One scalar series sampled on the frozen physical-time grid."""

    name: str
    time_s: tuple[float, ...]
    values: tuple[float, ...]
    comparison_dt_s: float = COMMON_DT_S
    time_shift_applied_s: float = 0.0
    interpolation: str = "linear_with_gap_rejection"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_float_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise AlignmentError(f"{label} must not be empty")
    if not all(math.isfinite(value) for value in converted):
        raise AlignmentError(f"{label} contains NaN or infinity")
    return converted


def validate_strict_common_grid(
    time_s: Sequence[float],
    *,
    origin_s: float = 0.0,
    dt_s: float = COMMON_DT_S,
) -> tuple[float, ...]:
    """Validate an unshifted, contiguous grid of exact ``dt_s`` ticks.

    A comparison window may begin after time zero, but every time must be an
    integer tick from the Stage-2 origin and no tick may be missing.
    """

    if not math.isfinite(origin_s):
        raise AlignmentError("origin_s must be finite")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise AlignmentError("dt_s must be positive and finite")

    times = _finite_float_tuple(time_s, "time_s")
    ticks: list[int] = []
    for time in times:
        raw_tick = (time - origin_s) / dt_s
        tick = round(raw_tick)
        expected = origin_s + tick * dt_s
        if abs(time - expected) > _TIME_ABS_TOL_S:
            raise AlignmentError(
                f"time {time:.17g} s is not on the {dt_s:g} s grid "
                f"from origin {origin_s:g} s"
            )
        if tick < 0:
            raise AlignmentError("comparison time precedes the Stage-2 origin")
        ticks.append(tick)

    for previous, current in zip(ticks, ticks[1:]):
        if current != previous + 1:
            raise AlignmentGapError(
                "common grid must be strictly increasing and contiguous; "
                f"encountered ticks {previous} and {current}"
            )
    return times


def make_common_grid(
    start_s: float,
    end_s: float,
    *,
    origin_s: float = 0.0,
    dt_s: float = COMMON_DT_S,
) -> tuple[float, ...]:
    """Construct an inclusive grid without rounding an off-grid endpoint."""

    endpoints = validate_strict_common_grid(
        (start_s,), origin_s=origin_s, dt_s=dt_s
    ) + validate_strict_common_grid((end_s,), origin_s=origin_s, dt_s=dt_s)
    if endpoints[1] < endpoints[0]:
        raise AlignmentError("end_s must not precede start_s")
    start_tick = round((endpoints[0] - origin_s) / dt_s)
    end_tick = round((endpoints[1] - origin_s) / dt_s)
    return tuple(origin_s + tick * dt_s for tick in range(start_tick, end_tick + 1))


def resample_to_common_grid(
    name: str,
    source_time_s: Sequence[float],
    source_values: Sequence[float],
    target_time_s: Sequence[float],
    *,
    time_shift_s: float = 0.0,
    max_source_gap_s: float = COMMON_DT_S,
) -> AlignedSeries:
    """Linearly sample native output while refusing time shifts and gaps.

    Exact native samples are copied.  Interpolation is permitted only when the
    two bracketing native samples are no farther apart than
    ``max_source_gap_s`` (0.10 s by default).  A rejected point raises an
    exception; it is never silently dropped or bridged.
    """

    if abs(float(time_shift_s)) > _TIME_ABS_TOL_S:
        raise TimeShiftNotAllowedError(
            "physical-time shifting is forbidden for 1-D/2-D comparison"
        )
    if not math.isfinite(max_source_gap_s) or max_source_gap_s <= 0.0:
        raise AlignmentError("max_source_gap_s must be positive and finite")

    native_time = _finite_float_tuple(source_time_s, "source_time_s")
    native_values = _finite_float_tuple(source_values, "source_values")
    if len(native_time) != len(native_values):
        raise AlignmentError("source_time_s and source_values lengths differ")
    if len(native_time) < 2:
        raise AlignmentCoverageError("at least two native samples are required")
    for left, right in zip(native_time, native_time[1:]):
        if right <= left:
            raise AlignmentError("source_time_s must be strictly increasing")

    target = validate_strict_common_grid(target_time_s)
    aligned: list[float] = []
    for requested_time in target:
        right_index = bisect_left(native_time, requested_time)

        exact_index: int | None = None
        if right_index < len(native_time):
            if abs(native_time[right_index] - requested_time) <= _TIME_ABS_TOL_S:
                exact_index = right_index
        if exact_index is None and right_index > 0:
            if abs(native_time[right_index - 1] - requested_time) <= _TIME_ABS_TOL_S:
                exact_index = right_index - 1
        if exact_index is not None:
            aligned.append(native_values[exact_index])
            continue

        if right_index == 0 or right_index == len(native_time):
            raise AlignmentCoverageError(
                f"target time {requested_time:g} s is outside native coverage"
            )

        left_index = right_index - 1
        left_time = native_time[left_index]
        right_time = native_time[right_index]
        source_gap = right_time - left_time
        if source_gap > max_source_gap_s + _TIME_ABS_TOL_S:
            raise AlignmentGapError(
                f"refusing to interpolate {name!r} at {requested_time:g} s "
                f"across native gap [{left_time:g}, {right_time:g}] s"
            )

        fraction = (requested_time - left_time) / source_gap
        value = native_values[left_index] + fraction * (
            native_values[right_index] - native_values[left_index]
        )
        aligned.append(value)

    return AlignedSeries(name=name, time_s=target, values=tuple(aligned))
