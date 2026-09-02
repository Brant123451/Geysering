#!/usr/bin/env python3
"""Compare archived H1 1D, paper-layout OpenFOAM 2D, and Cong 2017 data.

The curves are plotted on their native time axes. This script does not
time-shift either model to improve agreement with the experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
H0 = 0.66


def read_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or ()}


def read_levels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    free_surface: list[tuple[float, float]] = []
    gas_front: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            point = (float(row["t_s"]), float(row["Y_m"]))
            (free_surface if row["kind"] == "fs" else gas_front).append(point)
    return np.asarray(sorted(free_surface)), np.asarray(sorted(gas_front))


def scalar_row(name: str, units: str, paper: object, one_d: object, two_d: object) -> dict[str, object]:
    return {
        "metric": name,
        "units": units,
        "experiment": paper,
        "model_1d": one_d,
        "openfoam_2d": two_d,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.results_dir if args.results_dir.is_absolute() else HERE / args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    matched_1d = CASE_ROOT / "outputs" / "paper_layout_1d"
    legacy_1d = CASE_ROOT / "outputs"
    one_d_root = matched_1d if (matched_1d / "caseA_model_series.csv").exists() else legacy_1d
    one_d_series_path = one_d_root / "caseA_model_series.csv"
    one_d_metrics_path = one_d_root / "caseA_comparison_metrics.json"
    two_d_series_path = out / "openfoam_2d_riser_series.csv"
    two_d_pt1_path = out / "openfoam_2d_pt1_series.csv"
    two_d_metrics_path = out / "openfoam_2d_metrics.json"
    required = [one_d_series_path, one_d_metrics_path, two_d_series_path, two_d_metrics_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison input(s): " + ", ".join(missing))

    one = read_named_csv(one_d_series_path)
    two = read_named_csv(two_d_series_path)
    one_metrics = json.loads(one_d_metrics_path.read_text(encoding="utf-8"))
    two_metrics = json.loads(two_d_metrics_path.read_text(encoding="utf-8"))
    one_model = one_metrics["model"]
    two_model = two_metrics["model"]
    paper = two_metrics["experiment"]

    fs_exp, int_exp = read_levels(CASE_ROOT / "data" / "digitized" / "fig9a_levels.csv")
    pt1_exp = read_named_csv(CASE_ROOT / "data" / "digitized" / "fig10a_pt1.csv")
    two_pt1 = read_named_csv(two_d_pt1_path) if two_d_pt1_path.exists() else {}

    rows = [
        scalar_row("gas arrival Ta", "s", paper["Ta_s"], one_model["Ta_gas_enters_riser_s"], two_model["Ta_s"]),
        scalar_row(
            "free surface reaches 98% rim",
            "s",
            one_metrics["paper"].get("t_free_surface_at_rim_s"),
            one_model["t_free_surface_at_rim_s"],
            two_model["t_free_surface_at_98pct_rim_s"],
        ),
        scalar_row("free-surface rise speed", "m/s", paper["vfs_m_s"], one_model["v_fs_event_mean_mps"], two_model["vfs_m_s"]),
        scalar_row("gas-front rise speed", "m/s", paper["vint_m_s"], one_model["v_int_event_fit_mps"], two_model["vint_m_s"]),
        scalar_row(
            "pressure surge H/H0",
            "-",
            one_metrics["paper"].get("PT1_geyser_surge_over_H0"),
            one_model["pocket_surge_over_H0"],
            None if two_model["PT1_max_head_m_water"] is None else two_model["PT1_max_head_m_water"] / H0,
        ),
        scalar_row("geyser classification", "boolean", True, one_model["geyser"], two_model["geysering"]),
    ]

    payload = {
        "schema_version": 1,
        "case": "Cong 2017 B-H1",
        "comparison_policy": {
            "native_time_axes": True,
            "time_shift_applied": False,
            "outcome_fitting_applied": False,
            "height_datum": "riser entrance / main-pipe crown",
        },
        "geometry": {
            "experiment_and_2d": {"tee_x_m": 3.47, "valve_x_m": 5.98, "pipe_end_x_m": 6.59},
            "selected_1d": {
                "tee_x_m": one_metrics["case"]["tee_x"],
                "valve_x_m": one_metrics["case"]["valve_x"],
                "source": str(one_d_root),
            },
        },
        "metrics": rows,
        "status_2d": two_metrics["status"],
        "limitations": [
            "The preferred 1D input is the separately preserved Fig. 1(b) geometry-matched run; the earlier kinematics-derived 1D archive is not overwritten.",
            "The 2D riser width is area-equivalent (Dr^2/D=5.12 mm); a planar model cannot reproduce the 3D annular wall film.",
            "Digitized Fig. 10(a) is Run B-1, not an exact B-H1 pressure replicate; it is shown only as a morphology/envelope reference.",
        ],
    }
    (out / "h1_1d_2d_experiment_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (out / "h1_1d_2d_experiment_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "units", "experiment", "model_1d", "openfoam_2d"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 11.2), sharex=False)
    ax = axes[0]
    ax.plot(one["t_s"], one["Yfs_m"], color="#2563eb", lw=1.5, label="1D free surface")
    ax.plot(two["t_s"], two["Yfs_m_above_crown"], color="#dc2626", lw=1.5, label="2D free surface")
    if fs_exp.size:
        ax.plot(fs_exp[:, 0], fs_exp[:, 1], "s", ms=4, mfc="none", mec="#111827", label="experiment Fig. 9(a)")
    ax.axhline(1.8, color="0.4", ls=":", lw=1.0, label="riser rim")
    ax.set(xlim=(0, 13), ylim=(0, 1.95), ylabel="$Y_{fs}$ above crown (m)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1]
    ax.plot(one["t_s"], one["Yint_m"], color="#2563eb", lw=1.5, label="1D gas front")
    ax.plot(two["t_s"], two["Yint_m_above_crown"], color="#dc2626", lw=1.5, label="2D gas front")
    if int_exp.size:
        ax.plot(int_exp[:, 0], int_exp[:, 1], "o", ms=4, mfc="none", mec="#111827", label="experiment Fig. 9(a)")
    ax.set(xlim=(0, 13), ylim=(0, 1.95), ylabel="$Y_{int}$ above crown (m)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[2]
    ax.fill_between(
        pt1_exp["t_s"],
        pt1_exp["HoverH0_min"],
        pt1_exp["HoverH0_max"],
        color="0.75",
        alpha=0.45,
        label="experiment Fig. 10(a) B-1 envelope",
    )
    ax.plot(pt1_exp["t_s"], pt1_exp["HoverH0_med"], color="0.25", lw=1.0)
    ax.plot(one["t_s"], one["pocket_head_m"] / H0, color="#2563eb", lw=1.3, label="1D pocket head")
    if two_pt1:
        ax.plot(two_pt1["t_s"], two_pt1["head_m_water"] / H0, color="#dc2626", lw=1.3, label="2D PT1 head")
    ax.set(xlim=(0, 13), xlabel="Time after valve opening (s)", ylabel="Gauge head $H/H_0$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    fig.suptitle(
        "Cong 2017 B-H1: archived 1D vs paper-layout OpenFOAM 2D\n"
        "(native time axes; no alignment or fitted shift)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "h1_1d_2d_experiment_comparison.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
