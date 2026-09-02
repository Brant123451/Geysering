"""Audit Case-A 1-D riser emptying against the raw-field 2-D reference.

This is an offline, read-only diagnostic.  It does not import the Case-A
solver and none of its reference intervals or phase boundaries are available
to the simulation.  The phase split is based on saved breakthrough and flux
diagnostics, while all volume contributions are measured from the saved
``alpha_l`` field itself.
"""

from __future__ import annotations

import argparse
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
BOTTOM_ZONE_HEIGHT_M = 0.10


def _cell_edges(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 1 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be a strictly increasing one-dimensional array")
    if z.size == 1:
        return np.asarray([0.0, RISER_HEIGHT_M])
    edges = np.r_[0.0, 0.5 * (z[:-1] + z[1:]), RISER_HEIGHT_M]
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("z is incompatible with the 0.610 m riser")
    return edges


def _distance(model: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    model = np.asarray(model, dtype=float)
    reference = np.asarray(reference, dtype=float)
    residual = model - reference
    model_mean = float(np.mean(model))
    reference_mean = float(np.mean(reference))
    return {
        "model_mean": model_mean,
        "reference_mean": reference_mean,
        "mean_ratio": model_mean / reference_mean if reference_mean else None,
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
    }


def _integral(time: np.ndarray, values: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        return 0.0
    return float(np.trapezoid(values[start : stop + 1], time[start : stop + 1]))


def _phase(
    name: str,
    time: np.ndarray,
    inventory: np.ndarray,
    flux: np.ndarray,
    start: int,
    stop: int,
    total_loss: float,
) -> dict[str, Any]:
    inventory_change = float(inventory[stop] - inventory[start])
    flux_integral = _integral(time, flux, start, stop)
    loss = -inventory_change
    return {
        "name": name,
        "index_bounds_inclusive": [int(start), int(stop)],
        "time_window_s": [float(time[start]), float(time[stop])],
        "inventory_start_m3": float(inventory[start]),
        "inventory_end_m3": float(inventory[stop]),
        "inventory_change_m3": inventory_change,
        "inventory_loss_m3": loss,
        "share_of_total_inventory_loss": loss / total_loss if total_loss > 0.0 else None,
        "integrated_saved_net_flux_m3": flux_integral,
        "flux_minus_inventory_change_m3": flux_integral - inventory_change,
    }


def audit(
    fields_path: Path,
    diagnostics_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    with np.load(fields_path, allow_pickle=False) as fields:
        time = np.asarray(fields["time"], dtype=float)
        z = np.asarray(fields["z"], dtype=float)
        alpha = np.clip(np.asarray(fields["alpha_l"], dtype=float), 0.0, 1.0)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diag_time = np.asarray(diagnostics["t"], dtype=float)
    flux = np.asarray(diagnostics["junction_vertical_liquid_flux"], dtype=float)
    breakthrough = np.asarray(diagnostics["riser_breakthrough"], dtype=float)
    taylor_flux = np.asarray(
        diagnostics["junction_taylor_return_liquid_flux"], dtype=float
    )
    if not np.allclose(time, diag_time, rtol=0.0, atol=1.0e-9):
        raise ValueError("field and diagnostic time axes differ")
    if alpha.shape != (time.size, z.size):
        raise ValueError("alpha_l has incompatible dimensions")

    edges = _cell_edges(z)
    widths = np.diff(edges)
    bottom_overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], BOTTOM_ZONE_HEIGHT_M)
        - np.maximum(edges[:-1], 0.0),
    )
    whole_height = alpha @ widths
    bottom_height = alpha @ bottom_overlap
    whole_volume = RISER_AREA_M2 * whole_height
    bottom_volume = RISER_AREA_M2 * bottom_height

    breakthrough_indices = np.flatnonzero(breakthrough > 0.5)
    if breakthrough_indices.size == 0:
        raise ValueError("saved diagnostics contain no riser breakthrough")
    breakthrough_index = int(breakthrough_indices[0])
    taylor_indices = np.flatnonzero(taylor_flux > 1.0e-12)
    taylor_start_index = int(taylor_indices[0]) if taylor_indices.size else None

    # Define the breakthrough-associated episode from the first saved
    # breakthrough state through the sustained half-peak decay of downward
    # flux.  This avoids selecting the phase end by eye.  A window sensitivity
    # is also reported below because no unique mathematical event cutoff exists.
    downward = np.maximum(-flux, 0.0)
    peak_relative = int(np.argmax(downward[breakthrough_index:]))
    peak_index = breakthrough_index + peak_relative
    half_peak = 0.5 * downward[peak_index]
    pulse_end_index = time.size - 1
    for index in range(peak_index + 1, time.size):
        if downward[index] <= half_peak and np.all(
            downward[index:] <= half_peak + 1.0e-14
        ):
            pulse_end_index = index
            break

    total_loss = float(whole_volume[0] - whole_volume[-1])
    phases = [
        _phase(
            "prebreak_emptying",
            time,
            whole_volume,
            flux,
            0,
            breakthrough_index,
            total_loss,
        ),
        _phase(
            "breakthrough_associated_transient",
            time,
            whole_volume,
            flux,
            breakthrough_index,
            pulse_end_index,
            total_loss,
        ),
        _phase(
            "postbreak_tail_net_drainage",
            time,
            whole_volume,
            flux,
            pulse_end_index,
            time.size - 1,
            total_loss,
        ),
    ]
    # The phases above share their boundary states; inventory changes telescope
    # exactly, while trapezoidal flux integrals should not be summed without
    # accounting for the shared boundary intervals.

    prebreak_detail: dict[str, Any] = {}
    if taylor_start_index is not None:
        prebreak_detail = {
            "before_saved_taylor_return": _phase(
                "before_saved_taylor_return",
                time,
                whole_volume,
                flux,
                0,
                taylor_start_index,
                total_loss,
            ),
            "saved_taylor_return_to_breakthrough": _phase(
                "saved_taylor_return_to_breakthrough",
                time,
                whole_volume,
                flux,
                taylor_start_index,
                breakthrough_index,
                total_loss,
            ),
        }

    pulse_sensitivity: dict[str, Any] = {}
    for duration in (0.10, 0.35, 0.50):
        stop = int(
            min(
                time.size - 1,
                np.searchsorted(
                    time, time[breakthrough_index] + duration, side="right"
                )
                - 1,
            )
        )
        pulse_sensitivity[f"{duration:.2f}_s_after_first_breakthrough"] = _phase(
            f"breakthrough_window_{duration:.2f}_s",
            time,
            whole_volume,
            flux,
            breakthrough_index,
            stop,
            total_loss,
        )

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_trace = reference["trace"]
    reference_time = np.asarray([row["time_s"] for row in reference_trace], dtype=float)
    reference_whole_height = np.asarray(
        [row["whole_riser_equivalent_liquid_height_m"] for row in reference_trace],
        dtype=float,
    )
    reference_bottom_volume = np.asarray(
        [row["bottom_0p10m_equivalent_liquid_volume_m3"] for row in reference_trace],
        dtype=float,
    )
    reference_net = np.asarray(
        [row["mouth_net_m3_s"] for row in reference_trace], dtype=float
    )
    reference_mouth_fraction = np.asarray(
        [row["mouth_water_fraction"] for row in reference_trace], dtype=float
    )
    reference_gross_up = np.asarray(
        [row["mouth_gross_up_m3_s"] for row in reference_trace], dtype=float
    )
    reference_gross_down = np.asarray(
        [row["mouth_gross_down_m3_s"] for row in reference_trace], dtype=float
    )
    model_whole_height = np.interp(reference_time, time, whole_height)
    model_bottom_volume = np.interp(reference_time, time, bottom_volume)
    model_net = np.interp(reference_time, time, flux)
    model_mouth_fraction = np.interp(reference_time, time, alpha[:, 0])

    late = reference_time >= 8.50 - 1.0e-10
    late_whole_distance = _distance(
        model_whole_height[late], reference_whole_height[late]
    )
    late_bottom_distance = _distance(
        model_bottom_volume[late], reference_bottom_volume[late]
    )
    late_mouth_fraction_distance = _distance(
        model_mouth_fraction[late], reference_mouth_fraction[late]
    )
    whole_ratio = late_whole_distance["mean_ratio"]
    bottom_ratio = late_bottom_distance["mean_ratio"]
    mouth_fraction_ratio = late_mouth_fraction_distance["mean_ratio"]
    whole_height_mae = float(late_whole_distance["mae"])
    bottom_height_mae = float(late_bottom_distance["mae"]) / RISER_AREA_M2
    checks = [
        {
            "name": "late_whole_riser_mean_ratio",
            "status": (
                "PASS"
                if whole_ratio is not None and 0.5 <= whole_ratio <= 1.5
                else "FAIL"
            ),
            "observed": whole_ratio,
            "criterion": "between 0.5 and 1.5 (offline morphology screen)",
        },
        {
            "name": "late_whole_riser_height_mae",
            "status": "PASS" if whole_height_mae <= 0.5 * RISER_DIAMETER_M else "FAIL",
            "observed_m": whole_height_mae,
            "criterion": f"<= {0.5 * RISER_DIAMETER_M:.8g} m",
        },
        {
            "name": "late_bottom_0p10m_inventory_mean_ratio",
            "status": (
                "PASS"
                if bottom_ratio is not None and 0.5 <= bottom_ratio <= 1.5
                else "FAIL"
            ),
            "observed": bottom_ratio,
            "criterion": "between 0.5 and 1.5 (offline morphology screen)",
        },
        {
            "name": "late_bottom_0p10m_equivalent_height_mae",
            "status": "PASS" if bottom_height_mae <= 0.5 * RISER_DIAMETER_M else "FAIL",
            "observed_m": bottom_height_mae,
            "criterion": f"<= {0.5 * RISER_DIAMETER_M:.8g} m",
        },
        {
            "name": "late_mouth_cross_section_water_fraction_mean_ratio",
            "status": (
                "PASS"
                if mouth_fraction_ratio is not None
                and 0.5 <= mouth_fraction_ratio <= 1.5
                else "FAIL"
            ),
            "observed": mouth_fraction_ratio,
            "criterion": "between 0.5 and 1.5 (offline morphology screen)",
        },
    ]
    failed_checks = [row["name"] for row in checks if row["status"] == "FAIL"]
    selected_rows = []
    for target in (7.50, 7.75, 8.00, 8.50, 8.85, 9.20):
        index = int(np.argmin(np.abs(reference_time - target)))
        selected_rows.append(
            {
                "time_s": float(reference_time[index]),
                "model_whole_riser_height_m": float(model_whole_height[index]),
                "reference_whole_riser_height_m": float(reference_whole_height[index]),
                "model_bottom_0p10m_volume_m3": float(model_bottom_volume[index]),
                "reference_bottom_0p10m_volume_m3": float(
                    reference_bottom_volume[index]
                ),
                "model_net_m3_s": float(model_net[index]),
                "reference_net_m3_s": float(reference_net[index]),
                "model_mouth_water_fraction": float(model_mouth_fraction[index]),
                "reference_mouth_water_fraction": float(
                    reference_mouth_fraction[index]
                ),
                "reference_gross_up_m3_s": float(reference_gross_up[index]),
                "reference_gross_down_m3_s": float(reference_gross_down[index]),
            }
        )

    reference_window_start = int(np.searchsorted(time, reference_time[0]))
    reference_window_stop = int(np.argmin(np.abs(time - reference_time[-1])))
    model_reference_window_inventory_change = float(
        whole_volume[reference_window_stop] - whole_volume[reference_window_start]
    )
    model_reference_window_net = _integral(
        time, flux, reference_window_start, reference_window_stop
    )
    reference_integrated = reference["integrated_exchange"]

    return {
        "status": "FAIL" if failed_checks else "PASS",
        "failed_checks": failed_checks,
        "checks": checks,
        "provenance": {
            "one_dimensional_fields": str(fields_path.resolve()),
            "one_dimensional_diagnostics": str(diagnostics_path.resolve()),
            "two_dimensional_raw_field_reference": str(reference_path.resolve()),
            "solver_imported": False,
            "rendered_images_used": False,
            "result_prescription": False,
            "scientific_content": "diagnostic evidence only",
        },
        "model_riser_inventory": {
            "initial_equivalent_height_m": float(whole_height[0]),
            "final_equivalent_height_m": float(whole_height[-1]),
            "initial_volume_m3": float(whole_volume[0]),
            "final_volume_m3": float(whole_volume[-1]),
            "total_loss_m3": total_loss,
            "final_bottom_0p10m_equivalent_height_m": float(bottom_height[-1]),
            "final_bottom_0p10m_volume_m3": float(bottom_volume[-1]),
        },
        "phase_definition": {
            "first_breakthrough_time_s": float(time[breakthrough_index]),
            "first_breakthrough_index": breakthrough_index,
            "postbreak_downward_flux_peak_time_s": float(time[peak_index]),
            "postbreak_downward_flux_peak_m3_s": float(downward[peak_index]),
            "breakthrough_episode_end_rule": (
                "first sample after the postbreak downward-flux peak at which the "
                "downward magnitude is at or below half peak and remains so"
            ),
            "breakthrough_episode_end_time_s": float(time[pulse_end_index]),
        },
        "phase_inventory_decomposition": phases,
        "prebreak_detail": prebreak_detail,
        "breakthrough_window_sensitivity": pulse_sensitivity,
        "comparison_to_2d": {
            "7p5_to_9p2": {
                "whole_riser_height": _distance(
                    model_whole_height, reference_whole_height
                ),
                "bottom_0p10m_liquid_volume": _distance(
                    model_bottom_volume, reference_bottom_volume
                ),
                "net_mouth_flux_unsmoothed": _distance(model_net, reference_net),
                "model_inventory_change_m3": model_reference_window_inventory_change,
                "model_integrated_saved_net_flux_m3": model_reference_window_net,
                "reference_inventory_change_m3": float(
                    reference_integrated["whole_riser_inventory_change_m3"]
                ),
                "reference_integrated_net_flux_m3": float(
                    reference_integrated["net_m3"]
                ),
            },
            "8p5_to_9p2": {
                "whole_riser_height": late_whole_distance,
                "bottom_0p10m_liquid_volume": late_bottom_distance,
                "mouth_cross_section_water_fraction": (
                    late_mouth_fraction_distance
                ),
                "net_mouth_flux_unsmoothed": _distance(
                    model_net[late], reference_net[late]
                ),
            },
            "selected_times": selected_rows,
        },
        "offline_acceptance_intervals": {
            "source": (
                "Observed P05-P95 envelopes of raw 2-D fields over 8.5-9.2 s; "
                "not solver coefficients. Time-aligned comparisons remain primary."
            ),
            **reference["offline_acceptance_reference"][
                "late_window_observed_p05_p95"
            ],
            "single_signed_flux_model_limitation": (
                "A one-flux 1-D node cannot represent simultaneous gross upward and "
                "gross downward 2-D exchange. Gross intervals are capability "
                "diagnostics, not hard pass/fail gates for that node formulation."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            CASE
            / "outputs/caseA_openfoam2d_mouth_hold_up_reference_7p5_9p2s.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.fields, args.diagnostics, args.reference)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
