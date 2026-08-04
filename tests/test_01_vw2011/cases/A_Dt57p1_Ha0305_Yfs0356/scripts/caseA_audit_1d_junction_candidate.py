"""Audit a Case-A 1-D candidate against the fixed 2-D junction metrics.

The script is diagnostic only.  It reconstructs circular-pipe liquid depth,
uses shape-preserving interpolation to the 2-D sampling scale, and applies the
same spatial decomposition as ``caseA_extract_2d_junction_baseline.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


D = 0.094
L_TUNNEL = 0.546 + 2.970 + 0.490
TOWER_X = 3.516
TOWER_LEFT = TOWER_X - 0.5 * 0.0571
WAVE_X_MIN = 2.45
WAVE_X_MAX = TOWER_LEFT


def _odd_window(target_length: float, spacing: float, n: int, minimum: int = 5) -> int:
    raw = max(minimum, int(round(target_length / spacing)))
    if raw % 2 == 0:
        raw += 1
    maximum = n if n % 2 else n - 1
    return min(raw, maximum)


def _active_component(x: np.ndarray, envelope: np.ndarray, threshold: float) -> dict:
    active = envelope >= threshold
    padded = np.r_[False, active, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1) - 1
    if starts.size == 0:
        return {"largest_length_m": 0.0, "total_length_m": 0.0, "largest_bounds_m": []}
    dx = float(np.median(np.diff(x)))
    lengths = (stops - starts + 1) * dx
    selected = int(np.argmax(lengths))
    return {
        "largest_length_m": float(lengths[selected]),
        "total_length_m": float(np.sum(lengths)),
        "largest_bounds_m": [
            float(x[starts[selected]] - 0.5 * dx),
            float(x[stops[selected]] + 0.5 * dx),
        ],
    }


def _circular_depth_fraction(area_fraction: np.ndarray) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, 20001)
    area = (theta - np.sin(theta)) / (2.0 * math.pi)
    depth = 0.5 * (1.0 - np.cos(0.5 * theta))
    return np.interp(np.clip(area_fraction, 0.0, 1.0), area, depth)


def _wave_metrics(x_cell: np.ndarray, depth_cell: np.ndarray) -> dict:
    use = (x_cell >= WAVE_X_MIN) & (x_cell <= WAVE_X_MAX)
    x_source = x_cell[use]
    depth_source = depth_cell[use]
    sample_dx = 0.005
    x = np.arange(x_source[0], x_source[-1] + 0.25 * sample_dx, sample_dx)
    depth = PchipInterpolator(x_source, depth_source, extrapolate=False)(x)
    small_window = _odd_window(0.025, sample_dx, len(x), minimum=5)
    trend_window = _odd_window(0.30, sample_dx, len(x), minimum=9)
    envelope_window = _odd_window(0.08, sample_dx, len(x), minimum=5)
    resolved = savgol_filter(depth, small_window, 2, mode="interp")
    trend = savgol_filter(resolved, trend_window, 2, mode="interp")
    residual = resolved - trend
    kernel = np.ones(envelope_window) / envelope_window
    envelope = np.sqrt(np.convolve(residual * residual, kernel, mode="same"))
    return {
        "residual_a90_m": float(np.percentile(residual, 95) - np.percentile(residual, 5)),
        "residual_peak_to_peak_m": float(np.ptp(residual)),
        "residual_rms_m": float(np.sqrt(np.mean(residual * residual))),
        "envelope_max_m": float(np.max(envelope)),
        "active_fixed_2mm": _active_component(x, envelope, 0.002),
    }


def _reversal_times(time: np.ndarray, flow: np.ndarray, threshold: float) -> list[float]:
    state = 0
    reversals: list[float] = []
    for t, q in zip(time, flow):
        sign = 1 if q > threshold else -1 if q < -threshold else 0
        if sign == 0:
            continue
        if state and sign != state:
            reversals.append(float(t))
        state = sign
    return reversals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fields = np.load(args.fields)
    time = np.asarray(fields["time"], dtype=float)
    alpha = np.asarray(fields["horizontal_alpha_l"], dtype=float)
    nt = alpha.shape[1]
    dx = L_TUNNEL / nt
    x_cell = (np.arange(nt) + 0.5) * dx
    depth = D * _circular_depth_fraction(alpha)

    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    diag_time = np.asarray(diagnostics["t"], dtype=float)
    flow = np.asarray(diagnostics["junction_vertical_liquid_flux"], dtype=float)
    window = (diag_time >= 8.0) & (diag_time <= 10.0)
    q = flow[window]
    qt = diag_time[window]
    diag_dt = float(np.median(np.diff(diag_time)))
    q_window = _odd_window(0.25, diag_dt, len(flow), minimum=5)
    flow_smoothed = savgol_filter(
        flow, q_window, 3, mode="interp"
    )
    qs = flow_smoothed[window]

    riser_alpha = np.asarray(fields["alpha_l"], dtype=float)
    riser_dz = 0.610 / riser_alpha.shape[1]
    riser_equivalent_height = np.sum(riser_alpha, axis=1) * riser_dz

    snapshots = {}
    for target in (8.0, 9.0, 9.35, 10.0):
        index = int(np.argmin(np.abs(time - target)))
        snapshots[f"{time[index]:.2f}"] = _wave_metrics(x_cell, depth[index])

    liquid_inventory = np.asarray(
        diagnostics["total_liquid_including_escape"], dtype=float
    )
    gas_balance = np.asarray(diagnostics["horizontal_gas_mass_error"], dtype=float)
    result = {
        "fields": str(args.fields),
        "diagnostics": str(args.diagnostics),
        "junction_flux_8_to_10_s": {
            "minimum_L_s": float(np.min(q) * 1000.0),
            "maximum_L_s": float(np.max(q) * 1000.0),
            "rms_L_s": float(np.sqrt(np.mean(q * q)) * 1000.0),
            "mean_L_s": float(np.mean(q) * 1000.0),
            "reversal_times_0p005_L_s": _reversal_times(qt, q, 5.0e-6),
        },
        "junction_flux_8_to_10_s_0p25_s_savgol": {
            "minimum_L_s": float(np.min(qs) * 1000.0),
            "maximum_L_s": float(np.max(qs) * 1000.0),
            "rms_L_s": float(np.sqrt(np.mean(qs * qs)) * 1000.0),
            "mean_L_s": float(np.mean(qs) * 1000.0),
            "reversal_times_0p005_L_s": _reversal_times(qt, qs, 5.0e-6),
        },
        "wave_snapshots": snapshots,
        "riser_equivalent_liquid_height_snapshots_m": {
            f"{target:.2f}": float(
                riser_equivalent_height[int(np.argmin(np.abs(time - target)))]
            )
            for target in (7.5, 7.75, 8.0, 8.25, 9.0, 9.35, 10.0)
        },
        "riser_top_6p5_to_10_s": {
            "minimum_m": float(
                np.min(np.asarray(diagnostics["wtop"])[
                    (diag_time >= 6.5) & (diag_time <= 10.0)
                ])
            ),
            "maximum_m": float(
                np.max(np.asarray(diagnostics["wtop"])[
                    (diag_time >= 6.5) & (diag_time <= 10.0)
                ])
            ),
        },
        "conservation": {
            "liquid_inventory_range_m3": float(np.ptp(liquid_inventory)),
            "maximum_gas_step_error_kg": float(np.max(np.abs(gas_balance))),
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
