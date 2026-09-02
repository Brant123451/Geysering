"""Preregistered Stage-1 physical-stability statistics for the S1 1-D run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .config import Stage1Config
from .contracts import Stage1Observation


class StabilityInputError(ValueError):
    """Accepted-state observations cannot support the frozen stability gate."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StabilityInputError("cannot take the mean of an empty series")
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class LinearScalarStatistics:
    mean: float
    slope_per_s: float
    half_window_mean_shift: float
    detrended_peak_to_peak: float


@dataclass(frozen=True, slots=True)
class VelocityVectorStatistics:
    component_slopes_m_s2: tuple[float, float, float]
    slope_norm_m_s2: float
    half_window_mean_vector_change_m_s: float
    maximum_detrended_residual_vector_magnitude_m_s: float


@dataclass(frozen=True, slots=True)
class FlowStatistics:
    linear: LinearScalarStatistics
    normalization_scale: float
    relative_half_window_mean_change: float
    relative_detrended_peak_to_peak: float


@dataclass(frozen=True, slots=True)
class StabilityCheck:
    check_id: str
    measured: float
    comparison: str
    threshold: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Stage1StabilityReport:
    decision: str
    latest_stage1_time_s: float | None
    terminal_window_start_s: float | None
    sample_count: int
    source_scales: Mapping[str, float]
    pressure_statistics: Mapping[str, LinearScalarStatistics]
    velocity_statistics: Mapping[str, VelocityVectorStatistics]
    flow_statistics: Mapping[str, FlowStatistics]
    balance_statistics: Mapping[str, Mapping[str, float]]
    checks: tuple[StabilityCheck, ...]
    automatic_acceptance: bool = False
    thresholds_preregistered: bool = True

    @property
    def stable_candidate(self) -> bool:
        return self.decision == "STABLE_CANDIDATE_REQUIRES_MANUAL_ACCEPTANCE"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _linear_scalar_metrics(
    times: Sequence[float], values: Sequence[float]
) -> LinearScalarStatistics:
    if len(times) != len(values) or len(times) < 2:
        raise StabilityInputError("linear metrics require paired samples")
    mean_t = _mean(times)
    mean_y = _mean(values)
    variance_t = sum((time - mean_t) ** 2 for time in times)
    if variance_t <= 0.0:
        raise StabilityInputError("linear metrics require distinct times")
    slope = sum(
        (time - mean_t) * (value - mean_y)
        for time, value in zip(times, values, strict=True)
    ) / variance_t
    intercept = mean_y - slope * mean_t
    residuals = [
        value - (intercept + slope * time)
        for time, value in zip(times, values, strict=True)
    ]
    midpoint = 0.5 * (times[0] + times[-1])
    first = [value for time, value in zip(times, values, strict=True) if time <= midpoint]
    second = [value for time, value in zip(times, values, strict=True) if time > midpoint]
    if not first or not second:
        raise StabilityInputError("terminal samples do not populate both half windows")
    return LinearScalarStatistics(
        mean=mean_y,
        slope_per_s=slope,
        half_window_mean_shift=abs(_mean(second) - _mean(first)),
        detrended_peak_to_peak=max(residuals) - min(residuals),
    )


def _vector_metrics(
    times: Sequence[float], vectors: Sequence[tuple[float, float, float]]
) -> VelocityVectorStatistics:
    components = [[vector[index] for vector in vectors] for index in range(3)]
    scalar = [_linear_scalar_metrics(times, component) for component in components]
    slopes = tuple(item.slope_per_s for item in scalar)
    midpoint = 0.5 * (times[0] + times[-1])
    first = [vector for time, vector in zip(times, vectors, strict=True) if time <= midpoint]
    second = [vector for time, vector in zip(times, vectors, strict=True) if time > midpoint]
    first_mean = tuple(_mean([row[index] for row in first]) for index in range(3))
    second_mean = tuple(_mean([row[index] for row in second]) for index in range(3))
    mean_t = _mean(times)
    component_means = tuple(_mean(component) for component in components)
    intercepts = tuple(
        mean - slope * mean_t for mean, slope in zip(component_means, slopes, strict=True)
    )
    residual_magnitudes: list[float] = []
    for time, vector in zip(times, vectors, strict=True):
        residual = tuple(
            value - (intercept + slope * time)
            for value, intercept, slope in zip(vector, intercepts, slopes, strict=True)
        )
        residual_magnitudes.append(math.sqrt(sum(value * value for value in residual)))
    return VelocityVectorStatistics(
        component_slopes_m_s2=slopes,  # type: ignore[arg-type]
        slope_norm_m_s2=math.sqrt(sum(value * value for value in slopes)),
        half_window_mean_vector_change_m_s=math.sqrt(
            sum(
                (second_value - first_value) ** 2
                for first_value, second_value in zip(first_mean, second_mean, strict=True)
            )
        ),
        maximum_detrended_residual_vector_magnitude_m_s=max(residual_magnitudes),
    )


def _percentile_nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise StabilityInputError("cannot take a percentile of an empty series")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _check(
    checks: list[StabilityCheck],
    check_id: str,
    measured: float,
    comparison: str,
    threshold: float,
) -> None:
    if comparison == "<=":
        passed = measured <= threshold + 1.0e-15
    elif comparison == ">=":
        passed = measured + 1.0e-15 >= threshold
    else:  # pragma: no cover - all comparisons are frozen below
        raise StabilityInputError(f"unsupported comparison {comparison!r}")
    checks.append(
        StabilityCheck(
            check_id=check_id,
            measured=measured,
            comparison=comparison,
            threshold=threshold,
            passed=passed,
        )
    )


def _empty_report(config: Stage1Config, samples: int) -> Stage1StabilityReport:
    return Stage1StabilityReport(
        decision="INCONCLUSIVE",
        latest_stage1_time_s=None,
        terminal_window_start_s=None,
        sample_count=samples,
        source_scales={
            "driving_pressure_difference_pa": config.scales.driving_pressure_difference_pa,
            "ideal_head_velocity_m_s": config.scales.ideal_head_velocity_m_s,
            "reference_volume_flow_m3_s": config.scales.reference_volume_flow_m3_s,
            "reference_mass_flow_kg_s": config.scales.reference_mass_flow_kg_s,
            "ideal_advective_time_s": config.scales.ideal_advective_time_s,
        },
        pressure_statistics={},
        velocity_statistics={},
        flow_statistics={},
        balance_statistics={},
        checks=(),
    )


def evaluate_stage1_stability(
    observations: Iterable[Stage1Observation], config: Stage1Config
) -> Stage1StabilityReport:
    """Evaluate the final 4 s with the frozen 2-D-equivalent statistics.

    Passing is only an automatic candidate.  This function cannot authorize a
    state, write a marker, or change any threshold.
    """

    samples = sorted(tuple(observations), key=lambda item: item.stage1_time_s)
    if not samples:
        return _empty_report(config, 0)
    if len({item.stage1_time_s for item in samples}) != len(samples):
        raise StabilityInputError("duplicate Stage-1 observation time")
    latest = samples[-1].stage1_time_s
    if latest + 1.0e-12 < config.minimum_physical_time_s:
        report = _empty_report(config, len(samples))
        return Stage1StabilityReport(
            **{
                **report.to_json_dict(),
                "latest_stage1_time_s": latest,
            }
        )

    window_start = latest - config.terminal_window_s
    terminal = [
        item
        for item in samples
        if window_start - 1.0e-10 <= item.stage1_time_s <= latest + 1.0e-10
    ]
    expected_count = int(round(config.terminal_window_s / config.sample_interval_s)) + 1
    if len(terminal) != expected_count:
        raise StabilityInputError(
            f"terminal window requires {expected_count} exact samples, found {len(terminal)}"
        )
    for index, sample in enumerate(terminal):
        expected = window_start + index * config.sample_interval_s
        if not math.isclose(sample.stage1_time_s, expected, rel_tol=0.0, abs_tol=2.0e-10):
            raise StabilityInputError("Stage-1 terminal samples are not on the exact 0.1 s grid")

    expected_pressure = set(config.required_pressure_channels)
    expected_velocity = set(config.required_velocity_channels)
    for sample in terminal:
        if set(sample.gauge_pressures_pa) != expected_pressure:
            raise StabilityInputError("Stage-1 pressure-channel set drifted from P1--P6")
        if set(sample.velocity_vectors_m_s) != expected_velocity:
            raise StabilityInputError("Stage-1 velocity-channel set drifted from P1--P6")

    times = [item.stage1_time_s for item in terminal]
    checks: list[StabilityCheck] = []
    pressure_statistics: dict[str, LinearScalarStatistics] = {}
    for channel in config.required_pressure_channels:
        metrics = _linear_scalar_metrics(
            times, [item.gauge_pressures_pa[channel] for item in terminal]
        )
        pressure_statistics[channel] = metrics
        _check(
            checks,
            f"pressure.{channel}.absolute_slope_pa_s",
            abs(metrics.slope_per_s),
            "<=",
            config.pressure.maximum_absolute_slope_pa_s,
        )
        _check(
            checks,
            f"pressure.{channel}.half_window_mean_shift_pa",
            metrics.half_window_mean_shift,
            "<=",
            config.pressure.maximum_half_window_mean_shift_pa,
        )
        _check(
            checks,
            f"pressure.{channel}.detrended_peak_to_peak_pa",
            metrics.detrended_peak_to_peak,
            "<=",
            config.pressure.maximum_detrended_peak_to_peak_pa,
        )

    velocity_statistics: dict[str, VelocityVectorStatistics] = {}
    for channel in config.required_velocity_channels:
        metrics = _vector_metrics(
            times, [item.velocity_vectors_m_s[channel] for item in terminal]
        )
        velocity_statistics[channel] = metrics
        _check(
            checks,
            f"velocity.{channel}.slope_norm_m_s2",
            metrics.slope_norm_m_s2,
            "<=",
            config.velocity.maximum_slope_norm_m_s2,
        )
        _check(
            checks,
            f"velocity.{channel}.half_window_mean_vector_change_m_s",
            metrics.half_window_mean_vector_change_m_s,
            "<=",
            config.velocity.maximum_half_window_mean_vector_change_m_s,
        )
        _check(
            checks,
            f"velocity.{channel}.maximum_detrended_residual_vector_magnitude_m_s",
            metrics.maximum_detrended_residual_vector_magnitude_m_s,
            "<=",
            config.velocity.maximum_detrended_residual_vector_magnitude_m_s,
        )

    flow_values = {
        "qin_m3_s": [item.boundary_flows.qin_m3_s for item in terminal],
        "qout_m3_s": [item.boundary_flows.qout_m3_s for item in terminal],
        "mdot_in_kg_s": [item.boundary_flows.mdot_in_kg_s for item in terminal],
        "mdot_out_kg_s": [item.boundary_flows.mdot_out_kg_s for item in terminal],
    }
    flow_statistics: dict[str, FlowStatistics] = {}
    for channel, values in flow_values.items():
        scalar = _linear_scalar_metrics(times, values)
        floor = (
            config.boundary_flow.volume_flow_denominator_floor_m3_s
            if "m3_s" in channel
            else config.boundary_flow.mass_flow_denominator_floor_kg_s
        )
        scale = max(abs(scalar.mean), floor)
        metrics = FlowStatistics(
            linear=scalar,
            normalization_scale=scale,
            relative_half_window_mean_change=scalar.half_window_mean_shift / scale,
            relative_detrended_peak_to_peak=scalar.detrended_peak_to_peak / scale,
        )
        flow_statistics[channel] = metrics
        _check(checks, f"flow.{channel}.mean_forward", scalar.mean, ">=", floor)
        _check(
            checks,
            f"flow.{channel}.relative_half_window_mean_change",
            metrics.relative_half_window_mean_change,
            "<=",
            config.boundary_flow.maximum_half_window_relative_mean_change,
        )
        _check(
            checks,
            f"flow.{channel}.relative_detrended_peak_to_peak",
            metrics.relative_detrended_peak_to_peak,
            "<=",
            config.boundary_flow.maximum_detrended_peak_to_peak_fraction,
        )

    q_floor = config.boundary_flow.volume_flow_denominator_floor_m3_s
    m_floor = config.boundary_flow.mass_flow_denominator_floor_kg_s
    volume_imbalance: list[float] = []
    mass_imbalance: list[float] = []
    for item in terminal:
        q_scale = max(
            0.5 * (abs(item.boundary_flows.qin_m3_s) + abs(item.boundary_flows.qout_m3_s)),
            q_floor,
        )
        m_scale = max(
            0.5 * (abs(item.boundary_flows.mdot_in_kg_s) + abs(item.boundary_flows.mdot_out_kg_s)),
            m_floor,
        )
        volume_imbalance.append(
            abs(item.boundary_flows.qin_m3_s - item.boundary_flows.qout_m3_s) / q_scale
        )
        mass_imbalance.append(
            abs(item.boundary_flows.mdot_in_kg_s - item.boundary_flows.mdot_out_kg_s) / m_scale
        )
    balance_statistics: dict[str, Mapping[str, float]] = {}
    for name, values in (
        ("volume_flow", volume_imbalance),
        ("mass_flow", mass_imbalance),
    ):
        metrics = {
            "mean_relative_imbalance": _mean(values),
            "p95_instantaneous_relative_imbalance": _percentile_nearest_rank(values, 0.95),
        }
        balance_statistics[name] = metrics
        _check(
            checks,
            f"balance.{name}.mean_relative_imbalance",
            metrics["mean_relative_imbalance"],
            "<=",
            config.balance.maximum_mean_relative_imbalance,
        )
        _check(
            checks,
            f"balance.{name}.p95_instantaneous_relative_imbalance",
            metrics["p95_instantaneous_relative_imbalance"],
            "<=",
            config.balance.maximum_p95_instantaneous_relative_imbalance,
        )

    passed = all(item.passed for item in checks)
    return Stage1StabilityReport(
        decision=(
            "STABLE_CANDIDATE_REQUIRES_MANUAL_ACCEPTANCE" if passed else "UNSTABLE"
        ),
        latest_stage1_time_s=latest,
        terminal_window_start_s=window_start,
        sample_count=len(terminal),
        source_scales={
            "driving_pressure_difference_pa": config.scales.driving_pressure_difference_pa,
            "ideal_head_velocity_m_s": config.scales.ideal_head_velocity_m_s,
            "reference_volume_flow_m3_s": config.scales.reference_volume_flow_m3_s,
            "reference_mass_flow_kg_s": config.scales.reference_mass_flow_kg_s,
            "ideal_advective_time_s": config.scales.ideal_advective_time_s,
        },
        pressure_statistics=pressure_statistics,
        velocity_statistics=velocity_statistics,
        flow_statistics=flow_statistics,
        balance_statistics=balance_statistics,
        checks=tuple(checks),
    )


__all__ = [
    "FlowStatistics",
    "LinearScalarStatistics",
    "StabilityCheck",
    "StabilityInputError",
    "Stage1StabilityReport",
    "VelocityVectorStatistics",
    "evaluate_stage1_stability",
]
