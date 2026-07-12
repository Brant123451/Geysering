#!/usr/bin/env python3
"""Combine base/refined and valve-time runs into one compact study summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
REQUIRED = (
    "base_topen0p20",
    "refined_topen0p20",
    "base_topen0p10",
    "base_topen0p30",
)
GRID_PROBE_SPACING_M = 0.02
MASS_BALANCE_RELATIVE_LIMIT = 0.005
MIXED_VOLUME_FRACTION_LIMIT = 0.02
ALPHA_BOUND_TOLERANCE = 1.0e-5
COURANT_LIMIT = 1.0
FIELDS = (
    "classification_3d",
    "Ta_3d_s",
    "vfs_3d_m_per_s",
    "vint_3d_m_per_s",
    "Yfs_max_above_crown_m",
    "external_water_max_m3",
    "ejected_water_positive_flux_m3",
    "pocket_pressure_peak_gauge_head_m",
    "pocket_volume_min_m3",
    "riser_interface_mixed_volume_max_m3",
    "max_water_volume_balance_relative",
    "max_gas_mass_balance_relative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CASE_ROOT / "outputs" / "openfoam3d",
    )
    return parser.parse_args()


def relative_change(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (b - a) / max(abs(a), 1.0e-30)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    runs: dict[str, dict] = {}
    for path in sorted(input_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text())
        runs[data["label"]] = data

    rows = []
    for label, data in sorted(runs.items()):
        row = {"label": label}
        row.update({field: data.get(field) for field in FIELDS})
        row["cells"] = data.get("mesh", {}).get("checkMesh", {}).get("cells")
        row["checkMesh_passed"] = data.get("mesh", {}).get("checkMesh", {}).get(
            "passed"
        )
        rows.append(row)

    csv_path = input_dir / "study_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "cells", "checkMesh_passed", *FIELDS),
        )
        writer.writeheader()
        writer.writerows(rows)

    missing = [label for label in REQUIRED if label not in runs]
    base = runs.get("base_topen0p20")
    refined = runs.get("refined_topen0p20")
    grid_changes = {}
    if base and refined:
        for field in (
            "Ta_3d_s",
            "vfs_3d_m_per_s",
            "vint_3d_m_per_s",
            "Yfs_max_above_crown_m",
            "external_water_max_m3",
            "pocket_pressure_peak_gauge_head_m",
            "pocket_volume_min_m3",
        ):
            grid_changes[field] = relative_change(
                base.get(field), refined.get(field)
            )

    required_available = not missing
    required_runs = (
        [runs[label] for label in REQUIRED] if required_available else []
    )

    grid_classification_margin = {}
    grid_margin_passed = False
    if base and refined:
        base_yfs = base.get("Yfs_max_above_crown_m")
        refined_yfs = refined.get("Yfs_max_above_crown_m")
        rim = base.get("Yfs_rim_above_crown_m")
        if base_yfs is not None and refined_yfs is not None and rim is not None:
            grid_delta = abs(refined_yfs - base_yfs)
            clearance = rim - max(base_yfs, refined_yfs)
            required_clearance = 2.0 * grid_delta + GRID_PROBE_SPACING_M
            grid_margin_passed = clearance > required_clearance
            grid_classification_margin = {
                "base_max_Yfs_above_crown_m": base_yfs,
                "refined_max_Yfs_above_crown_m": refined_yfs,
                "absolute_grid_delta_m": grid_delta,
                "minimum_rim_clearance_m": clearance,
                "required_clearance_m": required_clearance,
                "passed": grid_margin_passed,
                "criterion": (
                    "rim clearance > 2*|refined-base| + one 0.02 m "
                    "centreline-probe interval"
                ),
            }

    source_schemes = (HERE / "system" / "fvSchemes").read_text()
    turbulence = (HERE / "constant" / "turbulenceProperties").read_text()
    source_model_checks = {
        "bounded_vof_vanleer": bool(
            re.search(
                r"div\(phi,alpha\)\s+Gauss\s+vanLeer\s*;",
                source_schemes,
            )
        ),
        "second_order_momentum_advection": bool(
            re.search(
                r"div\(rhoPhi,U\)\s+Gauss\s+linearUpwind\s+grad\(U\)\s*;",
                source_schemes,
            )
        ),
        "no_rans_dissipation": bool(
            re.search(r"simulationType\s+laminar\s*;", turbulence)
        ),
        "no_artificial_fvoptions_source": not (
            HERE / "constant" / "fvOptions"
        ).exists(),
    }

    all_meshes_passed = required_available and all(
        run.get("mesh", {}).get("checkMesh", {}).get("passed") is True
        and run.get("mesh", {}).get("checkMesh_acmi", {}).get("passed") is True
        for run in required_runs
    )
    unblocked_external_domain = required_available and all(
        run.get("mesh", {}).get("atmosphere_top_z_m") == 3.05
        and run.get("mesh", {})
        .get("boundary_surface_counts", {})
        .get("atmosphere")
        == 5
        for run in required_runs
    )
    event_windows_complete = required_available and all(
        run.get("event_window_complete") is True for run in required_runs
    )
    all_required_no_geyser = required_available and all(
        run.get("classification_3d") == "NO_GEYSER"
        for run in required_runs
    )
    mass_conservation_passed = required_available and all(
        abs(run.get("max_water_volume_balance_relative", math.inf))
        <= MASS_BALANCE_RELATIVE_LIMIT
        and abs(run.get("max_gas_mass_balance_relative", math.inf))
        <= MASS_BALANCE_RELATIVE_LIMIT
        for run in required_runs
    )
    interface_sharpness_passed = required_available and all(
        run.get("riser_interface_mixed_volume_max_m3", math.inf)
        <= MIXED_VOLUME_FRACTION_LIMIT
        * max(run.get("pocket_volume_initial_m3", 0.0), 0.0)
        for run in required_runs
    )
    numerical_health_passed = required_available and all(
        (health := run.get("numerical_health", {})).get("log_available")
        is True
        and health.get("fatal_error_detected") is False
        and 250.0 <= health.get("temperature_min_K", -math.inf)
        and health.get("temperature_max_K", math.inf) <= 500.0
        and health.get("alpha_water_min", -math.inf)
        >= -ALPHA_BOUND_TOLERANCE
        and health.get("alpha_water_max", math.inf)
        <= 1.0 + ALPHA_BOUND_TOLERANCE
        and health.get("courant_max", math.inf) <= COURANT_LIMIT
        and health.get("interface_courant_max", math.inf) <= COURANT_LIMIT
        for run in required_runs
    )
    valve_release_present = required_available and all(
        run.get("valve_volume_flux_peak_abs_m3_s", 0.0) > 1.0e-5
        and run.get("pocket_volume_min_m3", math.inf)
        < 0.95 * run.get("pocket_volume_initial_m3", 0.0)
        for run in required_runs
    )
    air_arrival_present = bool(base and refined) and all(
        run.get("Ta_3d_s") is not None for run in (base, refined)
    )

    confirmation_checks = {
        "required_event_windows_complete": event_windows_complete,
        "all_required_mesh_checks_passed": all_meshes_passed,
        "unblocked_five-face_external_atmosphere_present": (
            unblocked_external_domain
        ),
        "all_required_runs_no_geyser": all_required_no_geyser,
        "grid_uncertainty_cannot_reach_rim": grid_margin_passed,
        "water_and_gas_balance_within_0p5_percent": (
            mass_conservation_passed
        ),
        "mixed_interface_below_2_percent_initial_pocket_volume": (
            interface_sharpness_passed
        ),
        "temperature_alpha_and_courant_health_passed": (
            numerical_health_passed
        ),
        "valve_release_and_pocket_response_present": valve_release_present,
        "air_arrival_present_on_base_and_refined_meshes": air_arrival_present,
        "bounded_vof_and_higher_order_momentum_without_rans_or_sources": all(
            source_model_checks.values()
        ),
    }
    confirmation_passed = required_available and all(
        confirmation_checks.values()
    )

    summary = {
        "case": "B-H4",
        "status": "complete" if not missing else "incomplete",
        "physical_no_geyser_confirmation": (
            "PASS"
            if confirmation_passed
            else ("NOT_CONFIRMED" if required_available else "INCOMPLETE")
        ),
        "missing_runs": missing,
        "experiment": {
            "classification": "NO_GEYSER",
            "Ta_s": 8.14,
            "vfs_m_per_s": 0.207,
            "vint_m_per_s": 0.418,
        },
        "runs": runs,
        "base_to_refined_relative_change": grid_changes,
        "predeclared_false_no_geyser_limits": {
            "mass_balance_relative": MASS_BALANCE_RELATIVE_LIMIT,
            "mixed_volume_fraction_of_initial_pocket": (
                MIXED_VOLUME_FRACTION_LIMIT
            ),
            "alpha_bound_tolerance": ALPHA_BOUND_TOLERANCE,
            "maximum_courant": COURANT_LIMIT,
            "temperature_range_K": [250.0, 500.0],
        },
        "source_model_checks": source_model_checks,
        "grid_classification_margin": grid_classification_margin,
        "false_no_geyser_checks": confirmation_checks,
    }
    json_path = input_dir / "study_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
        labels = [row["label"] for row in rows]
        x = np.arange(len(rows))
        yfs = [row["Yfs_max_above_crown_m"] for row in rows]
        pressure = [row["pocket_pressure_peak_gauge_head_m"] for row in rows]
        mixing = [
            1e6 * row["riser_interface_mixed_volume_max_m3"]
            if row["riser_interface_mixed_volume_max_m3"] is not None
            else np.nan
            for row in rows
        ]
        axes[0].bar(x, yfs)
        axes[0].axhline(1.8, color="k", linestyle=":", label="riser rim")
        axes[0].set_ylabel("max Yfs above crown [m]")
        axes[0].legend()
        axes[1].bar(x, pressure)
        axes[1].axhline(0.66, color="k", linestyle=":", label="H0")
        axes[1].set_ylabel("peak pocket head [m]")
        axes[1].legend()
        axes[2].bar(x, mixing)
        axes[2].set_ylabel("max mixed-interface volume [mL]")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
        fig.suptitle("Cong 2017 B-H4: grid and valve-opening sensitivity")
        fig.savefig(input_dir / "study_summary.png", dpi=150)
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
