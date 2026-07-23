#!/usr/bin/env python3
"""Summarize the base/refined and valve-duration BH1 event runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = (
    "base-open-full-tau0",
    "refined-open-full-tau0",
    "base-open-full-tau0p2",
    "base-open-full-tau0p5",
)
FIELDS = (
    "Ta_gas_enters_riser_s",
    "t_free_surface_at_rim_s",
    "vfs_first_passage_m_per_s",
    "vint_first_passage_m_per_s",
    "PT1_peak_over_H0",
    "exterior_water_max_m3",
    "ejected_water_cumulative_positive_m3",
)


def relative(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return None
    return (a - b) / b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Instantaneous-open base mesh was recorded before waterRhoPhi/airRhoPhi
    # registration; keep it for mesh-sensitivity context, but require exact
    # phase fluxes on every later accepted event (valve + refined).
    LEGACY_WITHOUT_EXACT_PHASE_FLUX = {"base-open-full-tau0"}

    records = {}
    for run in RUNS:
        path = args.output_dir / f"{run}-metrics.json"
        if not path.exists():
            raise SystemExit(f"Missing completed sensitivity result: {path}")
        records[run] = json.loads(path.read_text(encoding="utf-8"))
        if not records[run].get("full_event", {}).get("pass", False):
            raise SystemExit(f"Sensitivity result failed full-event acceptance: {path}")
        exact = records[run].get("conservation", {}).get(
            "exact_phase_mass_fluxes_recorded", False
        )
        if not exact and run not in LEGACY_WITHOUT_EXACT_PHASE_FLUX:
            raise SystemExit(f"Sensitivity result lacks exact phase mass fluxes: {path}")

    csv_path = args.output_dir / "sensitivity-summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "mesh",
                "valve_duration_s",
                "geyser",
                "full_event_pass",
                "exact_phase_mass_fluxes",
                *FIELDS,
                "water_mass_budget_relative_error",
                "gas_mass_budget_relative_error",
                "total_mass_budget_relative_error",
            ]
        )
        for run in RUNS:
            item = records[run]
            writer.writerow(
                [
                    run,
                    "refined" if run.startswith("refined") else "base",
                    item["valve_duration_s"],
                    int(item["observed_3d_geyser"]),
                    int(item["full_event"]["pass"]),
                    int(bool(item.get("conservation", {}).get("exact_phase_mass_fluxes_recorded"))),
                    *[item.get(field) for field in FIELDS],
                    item["conservation"]["water_budget_relative_error"],
                    item["conservation"]["gas_budget_relative_error"],
                    item["conservation"]["total_budget_relative_error"],
                ]
            )

    base = records["base-open-full-tau0"]
    refined = records["refined-open-full-tau0"]
    summary = {
        "mesh_sensitivity_refined_minus_base_over_base": {
            field: relative(refined.get(field), base.get(field)) for field in FIELDS
        },
        "valve_sensitivity": {
            duration: {
                field: relative(records[run].get(field), base.get(field))
                for field in FIELDS
            }
            for duration, run in (
                ("0.2_s", "base-open-full-tau0p2"),
                ("0.5_s", "base-open-full-tau0p5"),
            )
        },
        "interpretation": (
            "The known GEYSER label was not used as a forcing or tuning "
            "criterion. Null event times mean the threshold was not crossed. "
            "base-open-full-tau0 is a legacy instantaneous-open base-mesh "
            "record without registered waterRhoPhi/airRhoPhi; mesh "
            "sensitivity uses it only for timing/peak morphology context. "
            "Valve and refined events carry exact phase-mass fluxes."
        ),
        "legacy_runs_without_exact_phase_mass_fluxes": sorted(
            LEGACY_WITHOUT_EXACT_PHASE_FLUX
        ),
    }
    (args.output_dir / "sensitivity-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    positions = range(len(RUNS))
    ta = [records[run].get("Ta_gas_enters_riser_s") or math.nan for run in RUNS]
    rim = [records[run].get("t_free_surface_at_rim_s") or math.nan for run in RUNS]
    axis.plot(positions, ta, "o-", label="air enters riser")
    axis.plot(positions, rim, "s-", label="free surface reaches rim")
    axis.axhline(8.07, color="0.35", linestyle=":", label="experimental Ta")
    axis.set_xticks(list(positions), ["base\n0 s", "refined\n0 s", "base\n0.2 s", "base\n0.5 s"])
    axis.set_ylabel("event time [s]")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(args.output_dir / "sensitivity-summary.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
