"""Unshifted waveform-error and diagnostic phase calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .grid import COMMON_DT_S, AlignmentError, validate_strict_common_grid


@dataclass(frozen=True)
class WaveformMetrics:
    """Metrics only: deliberately no automatic pass/fail tolerance."""

    sample_count: int
    comparison_dt_s: float
    rmse: float
    mean_bias_candidate_minus_reference: float
    zero_lag_correlation: float | None
    diagnostic_phase_lag_s: float | None
    diagnostic_phase_correlation: float | None
    phase_lag_convention: str
    phase_min_overlap_samples: int | None
    time_shift_applied_before_error_metrics_s: float
    automatic_acceptance_applied: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_values(
    values: Sequence[float], expected_length: int, label: str
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != expected_length:
        raise AlignmentError(f"{label} and time arrays have different lengths")
    if not all(math.isfinite(value) for value in converted):
        raise AlignmentError(f"{label} contains NaN or infinity")
    return converted


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = tuple(value - left_mean for value in left)
    right_delta = tuple(value - right_mean for value in right)
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _phase_diagnostic(
    reference: tuple[float, ...], candidate: tuple[float, ...]
) -> tuple[float | None, float | None, int | None]:
    sample_count = len(reference)
    if sample_count < 3:
        return None, None, None

    # Requiring at least half the record (and at least three points) prevents a
    # two-point edge match from masquerading as a phase estimate.  This is a
    # reported diagnostic method, not a physical acceptance tolerance.
    min_overlap = max(3, (sample_count + 1) // 2)
    max_lag = sample_count - min_overlap
    best_lag: int | None = None
    best_correlation: float | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            ref_slice = reference[:-lag]
            candidate_slice = candidate[lag:]
        elif lag < 0:
            ref_slice = reference[-lag:]
            candidate_slice = candidate[:lag]
        else:
            ref_slice = reference
            candidate_slice = candidate
        correlation = _correlation(ref_slice, candidate_slice)
        if correlation is None:
            continue
        if best_correlation is None or correlation > best_correlation + 1.0e-15:
            best_lag = lag
            best_correlation = correlation
        elif math.isclose(correlation, best_correlation, abs_tol=1.0e-15):
            assert best_lag is not None
            if (abs(lag), lag) < (abs(best_lag), best_lag):
                best_lag = lag
                best_correlation = correlation

    if best_lag is None:
        return None, None, min_overlap
    return best_lag * COMMON_DT_S, best_correlation, min_overlap


def compute_waveform_metrics(
    time_s: Sequence[float],
    reference: Sequence[float],
    candidate: Sequence[float],
) -> WaveformMetrics:
    """Compute errors at identical physical times without applying phase shift.

    The diagnostic cross-correlation lag is reported separately.  Positive lag
    means the candidate occurs later than the reference; it is never applied
    before calculating RMSE or bias.
    """

    times = validate_strict_common_grid(time_s)
    reference_values = _finite_values(reference, len(times), "reference")
    candidate_values = _finite_values(candidate, len(times), "candidate")
    error = tuple(
        candidate_value - reference_value
        for reference_value, candidate_value in zip(
            reference_values, candidate_values
        )
    )
    rmse = math.sqrt(sum(value * value for value in error) / len(error))
    bias = sum(error) / len(error)
    phase_lag, phase_correlation, min_overlap = _phase_diagnostic(
        reference_values, candidate_values
    )
    return WaveformMetrics(
        sample_count=len(times),
        comparison_dt_s=COMMON_DT_S,
        rmse=rmse,
        mean_bias_candidate_minus_reference=bias,
        zero_lag_correlation=_correlation(reference_values, candidate_values),
        diagnostic_phase_lag_s=phase_lag,
        diagnostic_phase_correlation=phase_correlation,
        phase_lag_convention="positive means candidate lags reference",
        phase_min_overlap_samples=min_overlap,
        time_shift_applied_before_error_metrics_s=0.0,
        automatic_acceptance_applied=False,
    )
