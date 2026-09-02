"""Read-only acceptance audit for the Case-A 1-D T-junction response.

The audit consumes a saved 1-D NPZ field archive, its solver-diagnostics
JSON, the frozen raw-field 2-D junction baseline, and the independently
extracted 2-D mouth/hold-up reference.  It never imports the solver and never
prescribes a state, flux, time, or target value back to a simulation.  All
acceptance gates live in this post-processing module.

The 2-D reference flux is the 0.25 s Savitzky--Golay derivative of the mapped
riser liquid inventory.  The 1-D junction flux is therefore smoothed over the
same duration before comparison.  In addition to similarity distances, the
audit checks conservation ledgers and a gas-speed sanity guard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent

RISER_DIAMETER_M = 0.0571
RISER_HEIGHT_M = 0.610
RISER_AREA_M2 = math.pi * RISER_DIAMETER_M**2 / 4.0
AUDIT_START_S = 8.5
AUDIT_END_S = 9.2
SMOOTHING_WIDTH_S = 0.25

# Isothermal gas acoustic scale used by the Case-A network.  The default
# guard rejects the near-sonic rarefied-cell failure mode while allowing the
# resolved blow-down in the currently accepted conservative gas solver.
GAS_CONSTANT_J_KG_K = 287.05
GAS_TEMPERATURE_K = 293.0
GAS_SOUND_SPEED_M_S = math.sqrt(GAS_CONSTANT_J_KG_K * GAS_TEMPERATURE_K)


@dataclass(frozen=True)
class AuditGates:
    """Frozen evaluation-only gates; none are imported by the solver."""

    riser_height_mae_limit_m: float = 0.5 * RISER_DIAMETER_M
    riser_height_mean_ratio_min: float = 0.50
    riser_height_mean_ratio_max: float = 1.50
    junction_flux_rmse_limit_m3_s: float = 5.0e-5
    minimum_flux_correlation: float = 0.0
    bidirectional_flux_threshold_m3_s: float = 5.0e-6
    gross_exchange_integral_ratio_min: float = 0.50
    gross_exchange_integral_ratio_max: float = 1.50
    gross_flow_sign_tolerance_m3_s: float = 1.0e-12
    gross_activity_threshold_m3_s: float = 5.0e-6
    minimum_simultaneous_gross_fraction: float = 0.80
    gross_net_closure_absolute_limit_m3_s: float = 1.0e-10
    gross_net_closure_relative_limit: float = 1.0e-6
    bottom_inventory_mae_limit_m3: float = 5.0e-5
    bottom_inventory_mean_ratio_min: float = 0.70
    bottom_inventory_mean_ratio_max: float = 1.30
    gas_speed_mach_limit: float = 0.50
    ledger_relative_error_limit: float = 1.0e-10
    gas_step_error_limit_kg: float = 1.0e-10


def _require_keys(mapping: Any, keys: tuple[str, ...], source: Path) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{source} is missing required keys: {missing}")


def _cell_widths(z: np.ndarray) -> np.ndarray:
    if z.ndim != 1 or z.size < 1 or not np.all(np.isfinite(z)):
        raise ValueError("z must be a finite one-dimensional array")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if z.size == 1:
        return np.asarray([RISER_HEIGHT_M], dtype=float)
    edges = np.r_[0.0, 0.5 * (z[:-1] + z[1:]), RISER_HEIGHT_M]
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("riser centres are inconsistent with the 0.610 m bore")
    return widths


def _odd_window_count(width: float, time: np.ndarray, degree: int) -> int:
    spacing = float(np.median(np.diff(time)))
    if spacing <= 0.0:
        raise ValueError("time must be strictly increasing")
    count = max(degree + 2, int(round(width / spacing)))
    if count % 2 == 0:
        count += 1
    maximum = time.size if time.size % 2 else time.size - 1
    return max(degree + 1, min(count, maximum))


def _local_polynomial(
    time: np.ndarray,
    values: np.ndarray,
    *,
    width_s: float,
    degree: int = 3,
    derivative: int = 0,
) -> np.ndarray:
    """Savitzky--Golay-equivalent local polynomial on a near-uniform grid."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or time.size < degree + 1:
        raise ValueError("local polynomial inputs have incompatible shapes")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    count = _odd_window_count(width_s, time, degree)
    half = count // 2
    result = np.empty_like(values)
    factor = math.factorial(derivative)
    for index, centre in enumerate(time):
        start = max(0, min(index - half, time.size - count))
        stop = start + count
        local_time = time[start:stop] - centre
        coefficients = np.polynomial.polynomial.polyfit(
            local_time, values[start:stop], degree
        )
        result[index] = factor * coefficients[derivative]
    return result


def _correlation(model: np.ndarray, reference: np.ndarray) -> float | None:
    model_std = float(np.std(model))
    reference_std = float(np.std(reference))
    if model_std <= 1.0e-15 or reference_std <= 1.0e-15:
        return None
    return float(np.corrcoef(model, reference)[0, 1])


def _ratio(value: float, reference: float) -> float | None:
    if abs(reference) <= 1.0e-30:
        return None
    return float(value / reference)


def _series_distance(model: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(model, dtype=float) - np.asarray(reference, dtype=float)
    reference_rms = float(np.sqrt(np.mean(np.asarray(reference) ** 2)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "model_mean": float(np.mean(model)),
        "reference_mean": float(np.mean(reference)),
        "mean_ratio": _ratio(float(np.mean(model)), float(np.mean(reference))),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "reference_rms": reference_rms,
        "normalized_rmse_by_reference_rms": _ratio(rmse, reference_rms),
        "correlation": _correlation(model, reference),
        "model_minimum": float(np.min(model)),
        "reference_minimum": float(np.min(reference)),
        "minimum_ratio": _ratio(float(np.min(model)), float(np.min(reference))),
        "model_maximum": float(np.max(model)),
        "reference_maximum": float(np.max(reference)),
        "maximum_ratio": _ratio(float(np.max(model)), float(np.max(reference))),
    }


def _ledger(values: np.ndarray, unit: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("conservation ledger must be a finite 1-D series")
    initial = float(values[0])
    absolute_error = np.abs(values - initial)
    scale = max(abs(initial), np.finfo(float).tiny)
    return {
        "unit": unit,
        "initial": initial,
        "final": float(values[-1]),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "maximum_relative_error": float(np.max(absolute_error) / scale),
        "relative_span": float(np.ptp(values) / scale),
    }


def _trapezoidal_integral(time: np.ndarray, values: np.ndarray) -> float:
    """Integrate a finite time series without depending on NumPy aliases."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or time.size < 2:
        raise ValueError("trapezoidal integral inputs have incompatible shapes")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("trapezoidal integral inputs must be finite")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("trapezoidal integral time must be strictly increasing")
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def _diagnostic_series(
    diagnostics: dict[str, Any],
    key: str,
    diag_time: np.ndarray,
    source: Path,
) -> np.ndarray:
    if key not in diagnostics:
        raise KeyError(f"{source} is missing required key: {key}")
    values = np.asarray(diagnostics[key], dtype=float)
    if values.shape != diag_time.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"diagnostic series {key!r} has an invalid time axis")
    return values


def _saved_gas_speed(
    mass: np.ndarray,
    momentum: np.ndarray,
    mask: np.ndarray,
) -> float:
    mass_window = np.asarray(mass, dtype=float)[mask]
    momentum_window = np.asarray(momentum, dtype=float)[mask]
    if mass_window.shape != momentum_window.shape:
        raise ValueError("gas mass and momentum fields have inconsistent shapes")
    mass_floor = max(float(np.max(mass_window)) * 1.0e-9, 1.0e-14)
    resolved = mass_window > mass_floor
    if not np.any(resolved):
        return 0.0
    speed = np.divide(
        momentum_window,
        mass_window,
        out=np.zeros_like(momentum_window),
        where=resolved,
    )
    return float(np.max(np.abs(speed[resolved])))


def _check(
    name: str,
    passed: bool,
    observed: Any,
    criterion: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "criterion": criterion,
    }


def audit(
    fields_path: Path,
    diagnostics_path: Path,
    baseline_path: Path,
    *,
    mouth_reference_path: Path | None = None,
    gates: AuditGates = AuditGates(),
    start_s: float = AUDIT_START_S,
    end_s: float = AUDIT_END_S,
) -> dict[str, Any]:
    """Return a structured audit without modifying any input or solver file."""

    if end_s <= start_s:
        raise ValueError("audit end time must be after its start time")
    with np.load(fields_path, allow_pickle=False) as fields:
        required_fields = (
            "time",
            "z",
            "alpha_l",
            "horizontal_gas_mass",
            "horizontal_gas_momentum",
            "vertical_gas_mass",
            "vertical_gas_momentum",
        )
        _require_keys(fields, required_fields, fields_path)
        time = np.asarray(fields["time"], dtype=float)
        z = np.asarray(fields["z"], dtype=float)
        alpha_l = np.asarray(fields["alpha_l"], dtype=float)
        horizontal_gas_mass = np.asarray(fields["horizontal_gas_mass"], dtype=float)
        horizontal_gas_momentum = np.asarray(
            fields["horizontal_gas_momentum"], dtype=float
        )
        vertical_gas_mass = np.asarray(fields["vertical_gas_mass"], dtype=float)
        vertical_gas_momentum = np.asarray(
            fields["vertical_gas_momentum"], dtype=float
        )

    if time.ndim != 1 or time.size < 5 or np.any(np.diff(time) <= 0.0):
        raise ValueError("1-D archive time must be a strictly increasing 1-D series")
    if alpha_l.shape[0] != time.size or alpha_l.shape[1] != z.size:
        raise ValueError("vertical liquid field dimensions are inconsistent")
    if not np.all(np.isfinite(alpha_l)):
        raise ValueError("vertical liquid field contains non-finite values")
    if time[0] > start_s or time[-1] < end_s:
        raise ValueError(
            f"1-D archive does not cover the audit window [{start_s}, {end_s}] s"
        )

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    required_diagnostics = (
        "t",
        "junction_vertical_liquid_flux",
        "junction_gross_upward_liquid_flux",
        "junction_gross_downward_liquid_flux",
        "twostream_bottom_0p1m_inventory",
        "total_liquid_including_escape",
        "total_gas_mass_including_atmosphere",
        "horizontal_gas_mass_error",
    )
    _require_keys(diagnostics, required_diagnostics, diagnostics_path)
    diag_time = np.asarray(diagnostics["t"], dtype=float)
    junction_flux = np.asarray(
        diagnostics["junction_vertical_liquid_flux"], dtype=float
    )
    if junction_flux.shape != diag_time.shape or np.any(np.diff(diag_time) <= 0.0):
        raise ValueError("diagnostic junction flux has an invalid time axis")
    if diag_time[0] > start_s or diag_time[-1] < end_s:
        raise ValueError("diagnostics do not cover the audit window")
    gross_upward_flux = _diagnostic_series(
        diagnostics,
        "junction_gross_upward_liquid_flux",
        diag_time,
        diagnostics_path,
    )
    gross_downward_flux = _diagnostic_series(
        diagnostics,
        "junction_gross_downward_liquid_flux",
        diag_time,
        diagnostics_path,
    )
    bottom_inventory = _diagnostic_series(
        diagnostics,
        "twostream_bottom_0p1m_inventory",
        diag_time,
        diagnostics_path,
    )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    trace = [
        row
        for row in baseline["trace"]
        if start_s - 1.0e-10 <= float(row["time_s"]) <= end_s + 1.0e-10
    ]
    if len(trace) < 5:
        raise ValueError("2-D baseline has insufficient samples in the audit window")
    reference_time = np.asarray([float(row["time_s"]) for row in trace])
    reference_height = np.asarray(
        [float(row["riser_equivalent_liquid_height_m"]) for row in trace]
    )
    reference_flux = np.asarray(
        [float(row["volume_derivative_flux_m3_s"]) for row in trace]
    )

    if mouth_reference_path is None:
        mouth_reference_path = (
            CASE / "outputs/caseA_openfoam2d_mouth_hold_up_reference_7p5_9p2s.json"
        )
    mouth_reference = json.loads(mouth_reference_path.read_text(encoding="utf-8"))
    _require_keys(mouth_reference, ("trace",), mouth_reference_path)
    mouth_trace = [
        row
        for row in mouth_reference["trace"]
        if start_s - 1.0e-10 <= float(row["time_s"]) <= end_s + 1.0e-10
    ]
    if len(mouth_trace) < 5:
        raise ValueError(
            "2-D mouth/hold-up reference has insufficient samples in the audit window"
        )
    mouth_required = (
        "time_s",
        "mouth_gross_up_m3_s",
        "mouth_gross_down_m3_s",
        "bottom_0p10m_equivalent_liquid_volume_m3",
    )
    for row in mouth_trace:
        _require_keys(row, mouth_required, mouth_reference_path)
    mouth_reference_time = np.asarray(
        [float(row["time_s"]) for row in mouth_trace], dtype=float
    )
    if np.any(np.diff(mouth_reference_time) <= 0.0):
        raise ValueError("2-D mouth/hold-up reference time must be strictly increasing")
    reference_gross_upward = np.asarray(
        [float(row["mouth_gross_up_m3_s"]) for row in mouth_trace], dtype=float
    )
    reference_gross_downward_signed = np.asarray(
        [float(row["mouth_gross_down_m3_s"]) for row in mouth_trace], dtype=float
    )
    reference_bottom_inventory = np.asarray(
        [
            float(row["bottom_0p10m_equivalent_liquid_volume_m3"])
            for row in mouth_trace
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(reference_gross_upward))
        or not np.all(np.isfinite(reference_gross_downward_signed))
        or not np.all(np.isfinite(reference_bottom_inventory))
    ):
        raise ValueError("2-D mouth/hold-up reference contains non-finite values")
    sign_tolerance = gates.gross_flow_sign_tolerance_m3_s
    if np.any(reference_gross_upward < -sign_tolerance):
        raise ValueError("2-D gross-up reference violates its positive sign convention")
    if np.any(reference_gross_downward_signed > sign_tolerance):
        raise ValueError("2-D gross-down reference violates its negative sign convention")
    if np.any(reference_bottom_inventory < 0.0):
        raise ValueError("2-D bottom inventory must be non-negative")
    # The raw 2-D extractor stores downward flow as a signed negative flux.
    # The 1-D two-stream diagnostics store its positive magnitude.
    reference_gross_downward = -reference_gross_downward_signed

    widths = _cell_widths(z)
    riser_height = np.sum(np.clip(alpha_l, 0.0, 1.0) * widths[None, :], axis=1)
    model_height = np.interp(reference_time, time, riser_height)

    junction_flux_smoothed = _local_polynomial(
        diag_time, junction_flux, width_s=SMOOTHING_WIDTH_S
    )
    model_flux = np.interp(reference_time, diag_time, junction_flux_smoothed)
    inventory_flux = _local_polynomial(
        time,
        RISER_AREA_M2 * riser_height,
        width_s=SMOOTHING_WIDTH_S,
        derivative=1,
    )
    model_inventory_flux = np.interp(reference_time, time, inventory_flux)

    height_distance = _series_distance(model_height, reference_height)
    flux_distance = _series_distance(model_flux, reference_flux)
    flux_inventory_closure = _series_distance(model_flux, model_inventory_flux)

    model_gross_upward = np.interp(
        mouth_reference_time, diag_time, gross_upward_flux
    )
    model_gross_downward = np.interp(
        mouth_reference_time, diag_time, gross_downward_flux
    )
    model_bottom_inventory = np.interp(
        mouth_reference_time, diag_time, bottom_inventory
    )
    gross_upward_distance = _series_distance(
        model_gross_upward, reference_gross_upward
    )
    gross_downward_distance = _series_distance(
        model_gross_downward, reference_gross_downward
    )
    bottom_inventory_distance = _series_distance(
        model_bottom_inventory, reference_bottom_inventory
    )
    model_gross_upward_integral = _trapezoidal_integral(
        mouth_reference_time, model_gross_upward
    )
    model_gross_downward_integral = _trapezoidal_integral(
        mouth_reference_time, model_gross_downward
    )
    reference_gross_upward_integral = _trapezoidal_integral(
        mouth_reference_time, reference_gross_upward
    )
    reference_gross_downward_integral = _trapezoidal_integral(
        mouth_reference_time, reference_gross_downward
    )
    gross_upward_integral_ratio = _ratio(
        model_gross_upward_integral, reference_gross_upward_integral
    )
    gross_downward_integral_ratio = _ratio(
        model_gross_downward_integral, reference_gross_downward_integral
    )

    native_gross_window = (
        (diag_time >= start_s - 1.0e-10) & (diag_time <= end_s + 1.0e-10)
    )
    native_upward = gross_upward_flux[native_gross_window]
    native_downward = gross_downward_flux[native_gross_window]
    native_net = junction_flux[native_gross_window]
    gross_net_residual = native_upward - native_downward - native_net
    gross_net_absolute_error = float(np.max(np.abs(gross_net_residual)))
    gross_net_scale = max(
        float(np.sqrt(np.mean(native_upward**2 + native_downward**2))),
        float(np.sqrt(np.mean(native_net**2))),
        np.finfo(float).tiny,
    )
    gross_net_relative_error = gross_net_absolute_error / gross_net_scale
    simultaneous_gross_fraction = float(
        np.mean(
            (model_gross_upward >= gates.gross_activity_threshold_m3_s)
            & (model_gross_downward >= gates.gross_activity_threshold_m3_s)
        )
    )
    reference_simultaneous_gross_fraction = float(
        np.mean(
            (reference_gross_upward >= gates.gross_activity_threshold_m3_s)
            & (reference_gross_downward >= gates.gross_activity_threshold_m3_s)
        )
    )

    field_window = (time >= start_s - 1.0e-10) & (time <= end_s + 1.0e-10)
    saved_horizontal_speed = _saved_gas_speed(
        horizontal_gas_mass, horizontal_gas_momentum, field_window
    )
    saved_vertical_speed = _saved_gas_speed(
        vertical_gas_mass, vertical_gas_momentum, field_window
    )
    diagnostic_speeds: dict[str, float] = {}
    diagnostic_window = (
        (diag_time >= start_s - 1.0e-10) & (diag_time <= end_s + 1.0e-10)
    )
    for key in ("horizontal_gas_maximum_velocity", "coupled_gas_maximum_velocity"):
        if key in diagnostics:
            values = np.asarray(diagnostics[key], dtype=float)
            if values.shape == diag_time.shape:
                diagnostic_speeds[key] = float(np.max(np.abs(values[diagnostic_window])))
    selected_gas_speed = max(
        [saved_horizontal_speed, saved_vertical_speed, *diagnostic_speeds.values()]
    )
    gas_mach = selected_gas_speed / GAS_SOUND_SPEED_M_S

    liquid_ledger = _ledger(
        np.asarray(diagnostics["total_liquid_including_escape"], dtype=float),
        "m3",
    )
    gas_ledger = _ledger(
        np.asarray(diagnostics["total_gas_mass_including_atmosphere"], dtype=float),
        "kg",
    )
    gas_step_error = float(
        np.max(np.abs(np.asarray(diagnostics["horizontal_gas_mass_error"], dtype=float)))
    )

    correlation = flux_distance["correlation"]
    height_ratio = height_distance["mean_ratio"]
    bottom_inventory_ratio = bottom_inventory_distance["mean_ratio"]
    checks = [
        _check(
            "riser_hold_up_mae",
            height_distance["mae"] <= gates.riser_height_mae_limit_m,
            height_distance["mae"],
            f"<= {gates.riser_height_mae_limit_m:.8g} m (half the riser diameter)",
        ),
        _check(
            "riser_hold_up_mean_ratio",
            height_ratio is not None
            and gates.riser_height_mean_ratio_min
            <= height_ratio
            <= gates.riser_height_mean_ratio_max,
            height_ratio,
            (
                f"between {gates.riser_height_mean_ratio_min:g} and "
                f"{gates.riser_height_mean_ratio_max:g}"
            ),
        ),
        _check(
            "junction_net_flux_rmse",
            flux_distance["rmse"] <= gates.junction_flux_rmse_limit_m3_s,
            flux_distance["rmse"],
            f"<= {gates.junction_flux_rmse_limit_m3_s:.8g} m3/s",
        ),
        _check(
            "junction_net_flux_correlation",
            correlation is not None and correlation >= gates.minimum_flux_correlation,
            correlation,
            f">= {gates.minimum_flux_correlation:g}",
        ),
        _check(
            "junction_bidirectional_exchange",
            float(np.max(model_flux)) >= gates.bidirectional_flux_threshold_m3_s
            and float(np.min(model_flux)) <= -gates.bidirectional_flux_threshold_m3_s,
            {
                "minimum_m3_s": float(np.min(model_flux)),
                "maximum_m3_s": float(np.max(model_flux)),
            },
            (
                "both signs exceed "
                f"{gates.bidirectional_flux_threshold_m3_s:.8g} m3/s"
            ),
        ),
        _check(
            "junction_gross_flux_sign_convention",
            float(np.min(native_upward)) >= -sign_tolerance
            and float(np.min(native_downward)) >= -sign_tolerance,
            {
                "minimum_upward_m3_s": float(np.min(native_upward)),
                "minimum_downward_magnitude_m3_s": float(np.min(native_downward)),
            },
            (
                "1-D gross upward and gross downward-magnitude diagnostics are "
                f">= -{sign_tolerance:.8g} m3/s"
            ),
        ),
        _check(
            "junction_gross_upward_integral_ratio",
            gross_upward_integral_ratio is not None
            and gates.gross_exchange_integral_ratio_min
            <= gross_upward_integral_ratio
            <= gates.gross_exchange_integral_ratio_max,
            gross_upward_integral_ratio,
            (
                f"between {gates.gross_exchange_integral_ratio_min:g} and "
                f"{gates.gross_exchange_integral_ratio_max:g} of the 2-D integral"
            ),
        ),
        _check(
            "junction_gross_downward_integral_ratio",
            gross_downward_integral_ratio is not None
            and gates.gross_exchange_integral_ratio_min
            <= gross_downward_integral_ratio
            <= gates.gross_exchange_integral_ratio_max,
            gross_downward_integral_ratio,
            (
                f"between {gates.gross_exchange_integral_ratio_min:g} and "
                f"{gates.gross_exchange_integral_ratio_max:g} of the 2-D magnitude integral"
            ),
        ),
        _check(
            "junction_simultaneous_gross_exchange",
            simultaneous_gross_fraction >= gates.minimum_simultaneous_gross_fraction,
            {
                "model_fraction": simultaneous_gross_fraction,
                "reference_2d_fraction": reference_simultaneous_gross_fraction,
            },
            (
                f"model fraction >= {gates.minimum_simultaneous_gross_fraction:g} "
                f"with each gross stream >= {gates.gross_activity_threshold_m3_s:.8g} m3/s"
            ),
        ),
        _check(
            "junction_gross_net_decomposition_closure",
            gross_net_absolute_error
            <= gates.gross_net_closure_absolute_limit_m3_s
            and gross_net_relative_error <= gates.gross_net_closure_relative_limit,
            {
                "maximum_absolute_error_m3_s": gross_net_absolute_error,
                "relative_error": gross_net_relative_error,
            },
            (
                "max |Q_up - Q_down - Q_net| <= "
                f"{gates.gross_net_closure_absolute_limit_m3_s:.8g} m3/s and "
                f"relative error <= {gates.gross_net_closure_relative_limit:.8g}"
            ),
        ),
        _check(
            "bottom_0p1m_inventory_mae",
            bottom_inventory_distance["mae"]
            <= gates.bottom_inventory_mae_limit_m3,
            bottom_inventory_distance["mae"],
            f"<= {gates.bottom_inventory_mae_limit_m3:.8g} m3",
        ),
        _check(
            "bottom_0p1m_inventory_mean_ratio",
            bottom_inventory_ratio is not None
            and gates.bottom_inventory_mean_ratio_min
            <= bottom_inventory_ratio
            <= gates.bottom_inventory_mean_ratio_max,
            bottom_inventory_ratio,
            (
                f"between {gates.bottom_inventory_mean_ratio_min:g} and "
                f"{gates.bottom_inventory_mean_ratio_max:g}"
            ),
        ),
        _check(
            "resolved_gas_speed_guard",
            gas_mach <= gates.gas_speed_mach_limit,
            {"maximum_m_s": selected_gas_speed, "mach": gas_mach},
            f"Mach <= {gates.gas_speed_mach_limit:g}",
        ),
        _check(
            "liquid_conservation_ledger",
            liquid_ledger["maximum_relative_error"]
            <= gates.ledger_relative_error_limit,
            liquid_ledger["maximum_relative_error"],
            f"<= {gates.ledger_relative_error_limit:.8g}",
        ),
        _check(
            "gas_conservation_ledger",
            gas_ledger["maximum_relative_error"] <= gates.ledger_relative_error_limit,
            gas_ledger["maximum_relative_error"],
            f"<= {gates.ledger_relative_error_limit:.8g}",
        ),
        _check(
            "horizontal_gas_step_balance",
            gas_step_error <= gates.gas_step_error_limit_kg,
            gas_step_error,
            f"<= {gates.gas_step_error_limit_kg:.8g} kg",
        ),
    ]
    failed = [check["name"] for check in checks if check["status"] == "FAIL"]

    height_scale = RISER_DIAMETER_M
    flux_scale = max(float(np.sqrt(np.mean(reference_flux**2))), 1.0e-30)
    combined_distance = math.sqrt(
        0.5
        * (
            (float(height_distance["rmse"]) / height_scale) ** 2
            + (float(flux_distance["rmse"]) / flux_scale) ** 2
        )
    )
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "provenance": {
            "one_dimensional_fields": str(fields_path.resolve()),
            "one_dimensional_diagnostics": str(diagnostics_path.resolve()),
            "two_dimensional_baseline": str(baseline_path.resolve()),
            "two_dimensional_mouth_hold_up_reference": str(
                mouth_reference_path.resolve()
            ),
            "window_s": [start_s, end_s],
            "comparison_uses_raw_saved_fields": True,
            "rendered_images_used": False,
            "solver_imported": False,
            "result_prescription": False,
            "note": (
                "All gates are post-processing only and are never imported by "
                "the Case-A solver."
            ),
        },
        "gates": asdict(gates),
        "checks": checks,
        "distance": {
            "combined_dimensionless": combined_distance,
            "riser_height_rmse_by_diameter": float(height_distance["rmse"])
            / RISER_DIAMETER_M,
            "junction_flux_rmse_by_2d_rms": float(flux_distance["rmse"])
            / flux_scale,
        },
        "riser_hold_up": {
            "unit": "m",
            "comparison": height_distance,
            "model_at_reference_times": model_height.tolist(),
            "reference_2d": reference_height.tolist(),
        },
        "junction_net_liquid_flux": {
            "unit": "m3/s",
            "positive_direction": "upward from horizontal pipe into riser",
            "smoothing_width_s": SMOOTHING_WIDTH_S,
            "comparison": flux_distance,
            "junction_vs_riser_inventory_derivative": flux_inventory_closure,
            "model_at_reference_times": model_flux.tolist(),
            "model_riser_inventory_derivative": model_inventory_flux.tolist(),
            "reference_2d": reference_flux.tolist(),
        },
        "junction_gross_liquid_exchange": {
            "unit": "m3/s",
            "one_dimensional_sign_convention": (
                "both upward and downward entries are non-negative magnitudes"
            ),
            "two_dimensional_source_sign_convention": (
                "upward is positive and downward is negative in the raw reference"
            ),
            "comparison_downward_uses_positive_magnitude": True,
            "upward": {
                "comparison": gross_upward_distance,
                "model_at_reference_times": model_gross_upward.tolist(),
                "reference_2d": reference_gross_upward.tolist(),
                "model_integral_m3": model_gross_upward_integral,
                "reference_2d_integral_m3": reference_gross_upward_integral,
                "integral_ratio": gross_upward_integral_ratio,
            },
            "downward_magnitude": {
                "comparison": gross_downward_distance,
                "model_at_reference_times": model_gross_downward.tolist(),
                "reference_2d": reference_gross_downward.tolist(),
                "model_integral_m3": model_gross_downward_integral,
                "reference_2d_integral_m3": reference_gross_downward_integral,
                "integral_ratio": gross_downward_integral_ratio,
            },
            "simultaneous_exchange_fraction": {
                "model": simultaneous_gross_fraction,
                "reference_2d": reference_simultaneous_gross_fraction,
                "activity_threshold_m3_s": gates.gross_activity_threshold_m3_s,
            },
            "net_decomposition_closure": {
                "definition": "Q_up - Q_down_magnitude - Q_net",
                "maximum_absolute_error_m3_s": gross_net_absolute_error,
                "relative_error": gross_net_relative_error,
            },
        },
        "bottom_0p1m_liquid_inventory": {
            "unit": "m3",
            "comparison": bottom_inventory_distance,
            "model_at_reference_times": model_bottom_inventory.tolist(),
            "reference_2d": reference_bottom_inventory.tolist(),
        },
        "gas_speed": {
            "sound_speed_m_s": GAS_SOUND_SPEED_M_S,
            "saved_field_horizontal_maximum_m_s": saved_horizontal_speed,
            "saved_field_vertical_maximum_m_s": saved_vertical_speed,
            "diagnostic_substep_maxima_m_s": diagnostic_speeds,
            "selected_maximum_m_s": selected_gas_speed,
            "selected_mach": gas_mach,
        },
        "conservation": {
            "liquid": liquid_ledger,
            "gas": gas_ledger,
            "maximum_horizontal_gas_step_error_kg": gas_step_error,
        },
        "reference_times_s": reference_time.tolist(),
        "mouth_reference_times_s": mouth_reference_time.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=CASE / "outputs/caseA_openfoam2d_junction_wave_baseline_7p5_10p5s.json",
    )
    parser.add_argument(
        "--mouth-reference",
        type=Path,
        default=(
            CASE / "outputs/caseA_openfoam2d_mouth_hold_up_reference_7p5_9p2s.json"
        ),
        help=(
            "frozen raw-field 2-D mouth gross-exchange and bottom-0.10-m "
            "hold-up reference"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 2 when any acceptance gate fails",
    )
    args = parser.parse_args()
    result = audit(
        args.fields,
        args.diagnostics,
        args.baseline,
        mouth_reference_path=args.mouth_reference,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.strict and result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
