#!/usr/bin/env python3
"""Aggregate compact run metrics and flag classification changes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    metrics = []
    for path in sorted(args.outputs.glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("case") == "B-H3":
            metrics.append(data)

    rows = []
    for data in metrics:
        conservation = data["conservation"]
        ejection = data["ejection"]
        numerical_controls = data.get("numerical_controls", {})
        closed_hold = data.get("closed_hold", {})
        rows.append(
            {
                "run_id": data["run_id"],
                "run_mode": data["run_mode"],
                "valve_opening": data["valve_opening"],
                "surface_tension_n_m": numerical_controls.get(
                    "surface_tension_n_per_m"
                ),
                "end_time_s": data["simulated_end_time_s"],
                "full_13s": int(data["full_13s_window_completed"]),
                "geyser": int(data["geysering"]),
                "Ta_s": data["Ta_3d_s"],
                "Yfs_max_m": data["Yfs_max_3d_m"],
                "Yint_max_m": data["Yint_max_3d_m"],
                "vfs_m_s": data["vfs_3d_m_per_s"],
                "vint_m_s": data["vint_3d_m_per_s"],
                "rim_ejected_l": ejection["cumulative_positive_rim_water_volume_l"],
                "gas_mass_residual_fraction": conservation[
                    "max_abs_global_gas_mass_residual_fraction"
                ],
                "total_mass_residual_fraction": conservation[
                    "max_abs_total_mass_residual_fraction"
                ],
                "closed_hold_pass": closed_hold.get("pass"),
            }
        )

    fields = [
        "run_id",
        "run_mode",
        "valve_opening",
        "surface_tension_n_m",
        "end_time_s",
        "full_13s",
        "geyser",
        "Ta_s",
        "Yfs_max_m",
        "Yint_max_m",
        "vfs_m_s",
        "vint_m_s",
        "rim_ejected_l",
        "gas_mass_residual_fraction",
        "total_mass_residual_fraction",
        "closed_hold_pass",
    ]
    with (args.outputs / "sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    full = [row for row in rows if row["full_13s"]]
    classifications = sorted({row["geyser"] for row in full})
    required = {
        "base_nominal",
        "refined_nominal",
        "dt_fine",
        "valve_0p2",
        "valve_0p5",
        "interface_diffuse",
        "interface_sharp",
        "sigma_zero",
    }
    completed = {row["run_id"] for row in full}
    summary = {
        "schema_version": 1,
        "case": "B-H3",
        "required_full_variants": sorted(required),
        "completed_full_variants": sorted(completed),
        "missing_full_variants": sorted(required - completed),
        "all_required_full_variants_complete": required <= completed,
        "classification_values_across_completed_full_variants": classifications,
        "classification_changed_by_numerics": len(classifications) > 1,
        "closed_hold_passed": any(
            row["closed_hold_pass"] is True for row in rows
        ),
        "experimental_classification": "GEYSER",
        "note": "No result is tuned to the experimental label.",
    }
    (args.outputs / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if rows:
        plot_rows = [row for row in rows if row["run_mode"] == "event"]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        names = [row["run_id"] for row in plot_rows]
        axes[0].bar(names, [row["Yfs_max_m"] for row in plot_rows])
        axes[0].axhline(1.85, color="k", linestyle="--", label="physical rim")
        axes[0].set_ylabel("Yfs,max (m)")
        axes[0].legend()
        axes[1].bar(
            names,
            [row["gas_mass_residual_fraction"] for row in plot_rows],
        )
        axes[1].set_yscale("log")
        axes[1].set_ylabel("gas mass residual fraction")
        axes[1].tick_params(axis="x", rotation=40)
        fig.tight_layout()
        fig.savefig(args.outputs / "sensitivity_summary.png", dpi=160)
        plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
