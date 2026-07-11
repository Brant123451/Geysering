#!/usr/bin/env python3
"""Combine base/refined and valve-time runs into one compact study summary."""

from __future__ import annotations

import argparse
import csv
import json
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

    summary = {
        "case": "B-H4",
        "status": "complete" if not missing else "incomplete",
        "missing_runs": missing,
        "experiment": {
            "classification": "NO_GEYSER",
            "Ta_s": 8.14,
            "vfs_m_per_s": 0.207,
            "vint_m_per_s": 0.418,
        },
        "runs": runs,
        "base_to_refined_relative_change": grid_changes,
        "false_no_geyser_checks": {
            "all_checkMesh_passed": bool(rows)
            and all(row["checkMesh_passed"] for row in rows),
            "all_required_runs_no_geyser": not missing
            and all(
                runs[label].get("classification_3d") == "NO_GEYSER"
                for label in REQUIRED
            ),
            "external_domain_present": all(
                run.get("mesh", {}).get("atmosphere_top_z_m") == 3.05
                for run in runs.values()
            ),
            "grid_metrics_quantified": bool(grid_changes),
            "mass_and_interface_metrics_reported": all(
                run.get("max_water_volume_balance_relative") is not None
                and run.get("max_gas_mass_balance_relative") is not None
                and run.get("riser_interface_mixed_volume_max_m3") is not None
                for run in runs.values()
            ),
        },
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
