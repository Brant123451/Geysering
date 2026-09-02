"""Audit one raw Case-A 1-D field archive against raw 2-D acceptance data.

This script reads conservative solver fields rather than rendered images.  It
compares the vertical liquid-equivalent height and horizontal gas-thickness
statistics at the four frozen OpenFOAM times.  It deliberately reports errors
and ratios without fitting an acceptance threshold to the 2-D result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
MODEL = CASE / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from vw2011_network_twofluid import _depth_frac  # noqa: E402


PIPE_LENGTH = 4.006
PIPE_DIAMETER = 0.094
WINDOWS = {
    "whole_pipe": (0.0, PIPE_LENGTH),
    "t_left_window": (2.4, 3.48745),
    "t_neighbourhood": (3.0, 3.65),
    "far_west": (0.0, 2.4),
}
METRICS = (
    "gas_thickness_mean_m",
    "gas_thickness_max_m",
    "gas_thickness_peak_to_peak_m",
    "gas_thickness_total_variation_m",
    "gas_thickness_second_difference_rms_m",
)


def _metrics(values: np.ndarray) -> dict[str, float]:
    second = np.diff(values, n=2)
    return {
        "gas_thickness_mean_m": float(np.mean(values)),
        "gas_thickness_max_m": float(np.max(values)),
        "gas_thickness_peak_to_peak_m": float(np.ptp(values)),
        "gas_thickness_total_variation_m": float(
            np.sum(np.abs(np.diff(values)), dtype=np.float64)
        ),
        "gas_thickness_second_difference_rms_m": float(
            np.sqrt(np.mean(second * second)) if second.size else 0.0
        ),
    }


def _comparison(model: float, reference: float) -> dict[str, float | None]:
    return {
        "model": float(model),
        "reference_2d": float(reference),
        "signed_error": float(model - reference),
        "absolute_error": float(abs(model - reference)),
        "model_to_2d_ratio": (
            None if reference == 0.0 else float(model / reference)
        ),
    }


def audit(fields_path: Path, reference_path: Path) -> dict[str, object]:
    with np.load(fields_path) as fields:
        time = np.asarray(fields["time"], dtype=float)
        z = np.asarray(fields["z"], dtype=float)
        vertical_liquid = np.asarray(fields["alpha_l"], dtype=float)
        horizontal_liquid = np.asarray(
            fields["horizontal_alpha_l"], dtype=float
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    if time.ndim != 1 or time.size < 2:
        raise ValueError("1-D archive must contain at least two output times")
    if vertical_liquid.shape[0] != time.size:
        raise ValueError("vertical field time dimension is inconsistent")
    if horizontal_liquid.shape[0] != time.size:
        raise ValueError("horizontal field time dimension is inconsistent")
    if z.size < 2:
        raise ValueError("vertical grid must contain at least two centres")

    dz = float(np.median(np.diff(z)))
    count = horizontal_liquid.shape[1]
    dx = PIPE_LENGTH / count
    x = (np.arange(count, dtype=float) + 0.5) * dx
    frames: list[dict[str, object]] = []
    for frozen in reference["frames"]:
        requested_time = float(frozen["time_s"])
        index = int(np.argmin(np.abs(time - requested_time)))
        alpha_l = np.clip(horizontal_liquid[index], 0.0, 1.0)
        gas_thickness = PIPE_DIAMETER * (
            1.0 - _depth_frac(alpha_l)
        )
        reference_x = np.asarray(frozen["horizontal"]["x_m"], dtype=float)
        reference_gas_native = np.asarray(
            frozen["horizontal"]["equivalent_gas_thickness_m"],
            dtype=float,
        )
        reference_gas_matched = np.interp(
            x, reference_x, reference_gas_native
        )
        vertical_height = float(
            np.sum(np.clip(vertical_liquid[index], 0.0, None)) * dz
        )
        vertical_reference = float(
            frozen["vertical"]["equivalent_liquid_height_above_crown_m"]
        )
        window_result: dict[str, object] = {}
        for name, (left, right) in WINDOWS.items():
            mask = (x >= left) & (x <= right)
            model_metrics = _metrics(gas_thickness[mask])
            reference_metrics = _metrics(reference_gas_matched[mask])
            reference_native_metrics = frozen["horizontal"][name]
            window_result[name] = {
                key: {
                    **_comparison(model_metrics[key], reference_metrics[key]),
                    "reference_2d_native_grid": float(
                        reference_native_metrics[key]
                    ),
                }
                for key in METRICS
            }
        frames.append(
            {
                "requested_time_s": requested_time,
                "model_time_s": float(time[index]),
                "time_offset_s": float(time[index] - requested_time),
                "vertical_liquid_equivalent_height_m": _comparison(
                    vertical_height, vertical_reference
                ),
                "horizontal": window_result,
            }
        )
    return {
        "provenance": {
            "one_dimensional_fields": str(fields_path.resolve()),
            "two_dimensional_reference": str(reference_path.resolve()),
            "one_dimensional_results_rendered": False,
            "metrics_computed_from_raw_fields": True,
            "two_dimensional_profile_interpolated_to_1d_centres": True,
            "acceptance_threshold_fitted": False,
            "horizontal_diameter_m": PIPE_DIAMETER,
            "horizontal_length_m": PIPE_LENGTH,
        },
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=CASE / "outputs" / "acceptance_reference_2d.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.fields, args.reference)
    output = args.output or args.fields.with_suffix(".acceptance.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
