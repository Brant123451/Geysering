#!/usr/bin/env python3
"""Create compact BH3 pressure/interface/flux/conservation deliverables."""
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


# OpenFOAM perfectGas uses the universal gas constant divided by molWeight.
R_AIR = 8314.46261815324 / 28.966
RHO_WATER = 998.0
GRAVITY = 9.81
RIM_Z = 1.850
INITIAL_FS = 0.660
POCKET_TARGET = math.pi * 0.050**2 * 0.610 / 4.0
PAPER_RISER_WATER_TARGET = math.pi * 0.026**2 * 0.660 / 4.0
NONOVERLAP_RISER_WATER_TARGET = math.pi * 0.026**2 * 0.610 / 4.0
AIR_MASS_TARGET = 101325.0 * POCKET_TARGET / (R_AIR * 296.15)
RISER_Z = np.arange(0.060, 1.840 + 0.0001, 0.010)
PLUME_Z = np.arange(1.860, 2.980 + 0.0001, 0.020)
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument("--reference-root", type=Path, default=Path("../.."))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-mode", choices=("event", "closed"), required=True)
    parser.add_argument("--valve-opening", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


def numeric_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in NUMBER.findall(line)]
        if values:
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No numeric data in {path}")
    width = max(len(row) for row in rows)
    if any(len(row) != width for row in rows):
        raise RuntimeError(f"Inconsistent row widths in {path}")
    return np.asarray(rows, dtype=float)


def function_file(case: Path, name: str, filename: str) -> Path:
    paths = sorted(
        (case / "postProcessing" / name).glob(f"*/{filename}"),
        key=lambda path: float(path.parent.name),
    )
    if not paths:
        raise FileNotFoundError(f"Missing {name}/{filename}")
    return paths[-1]


def read_function(case: Path, name: str, filename: str) -> np.ndarray:
    return numeric_rows(function_file(case, name, filename))


def read_probe(case: Path, name: str, field: str) -> np.ndarray:
    return read_function(case, name, field)


def interpolate(table: np.ndarray, time: np.ndarray, column: int = -1) -> np.ndarray:
    return np.interp(time, table[:, 0], table[:, column], left=np.nan, right=np.nan)


def cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(time)
    if len(time) > 1:
        dt = np.diff(time)
        result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * dt)
    return result


def vector_probe_magnitudes(table: np.ndarray, n_probes: int) -> np.ndarray:
    values = table[:, 1:]
    if values.shape[1] != 3 * n_probes:
        raise RuntimeError(
            f"Expected {3*n_probes} vector components, got {values.shape[1]}"
        )
    return np.linalg.norm(values.reshape(len(table), n_probes, 3), axis=2)


def interface_series(
    riser_alpha: np.ndarray, plume_alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = riser_alpha[:, 0]
    riser = riser_alpha[:, 1:]
    if riser.shape[1] != len(RISER_Z):
        raise RuntimeError(f"Expected {len(RISER_Z)} riser probes, got {riser.shape[1]}")
    plume = np.column_stack(
        [
            np.interp(time, plume_alpha[:, 0], plume_alpha[:, col])
            for col in range(1, plume_alpha.shape[1])
        ]
    )
    if plume.shape[1] != len(PLUME_Z):
        raise RuntimeError(f"Expected {len(PLUME_Z)} plume probes, got {plume.shape[1]}")

    y_fs = np.full(len(time), np.nan)
    y_int = np.full(len(time), np.nan)
    above_rim = np.zeros(len(time), dtype=bool)
    for row in range(len(time)):
        wet_riser = np.flatnonzero(riser[row] >= 0.5)
        wet_plume = np.flatnonzero(plume[row] >= 0.05)
        if wet_plume.size:
            y_fs[row] = PLUME_Z[wet_plume[-1]]
            above_rim[row] = True
        elif wet_riser.size:
            y_fs[row] = RISER_Z[wet_riser[-1]]

        if riser[row, 0] < 0.5:
            gas = riser[row] < 0.5
            first_water = np.flatnonzero(~gas)
            stop = int(first_water[0]) if first_water.size else len(gas)
            if stop:
                y_int[row] = RISER_Z[stop - 1]
    return time, y_fs, y_int, above_rim


def first_time(time: np.ndarray, condition: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return float(time[indices[0]]) if indices.size else None


def trajectory_slope(
    time: np.ndarray,
    level: np.ndarray,
    low: float,
    high: float,
) -> float | None:
    mask = np.isfinite(level) & (level >= low) & (level <= high)
    if np.count_nonzero(mask) < 4:
        return None
    slope, _ = np.polyfit(time[mask], level[mask], 1)
    return float(slope)


def scalar_from_file(table: np.ndarray) -> np.ndarray:
    if table.shape[1] < 2:
        raise RuntimeError("Function table has no value column")
    return table[:, -1]


def load_reference(case_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    with (case_root / "data" / "series_b_measurement.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        measured_row = next(csv.DictReader(stream))
    with (case_root / "outputs" / "series_b_model_summary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        model_row = next(csv.DictReader(stream))
    return measured_row, model_row


def safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    case = args.case.resolve()
    output = (case / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / args.run_id
    reference_root = args.reference_root.resolve()

    pressure = read_probe(case, "pressureProbes", "p")
    riser_alpha = read_probe(case, "riserCentreline", "alpha.water")
    plume_alpha = read_probe(case, "plumeCentreline", "alpha.water")
    riser_u = read_probe(case, "riserCentreline", "U")

    time, y_fs, y_int, above_rim = interface_series(riser_alpha, plume_alpha)
    pt1 = np.interp(time, pressure[:, 0], pressure[:, 1])
    pt2 = np.interp(time, pressure[:, 0], pressure[:, 2])
    max_probe_speed = np.max(
        vector_probe_magnitudes(riser_u, len(RISER_Z)), axis=1
    )

    water_volume_table = read_function(case, "waterVolume", "volFieldValue.dat")
    total_mass_table = read_function(case, "totalMass", "volFieldValue.dat")
    air_pt_table = read_function(case, "globalAirPT", "volFieldValue.dat")
    pocket_volume_table = read_function(
        case, "initialPocketVolume", "volFieldValue.dat"
    )
    riser_volume_table = read_function(
        case, "initialRiserWaterVolume", "volFieldValue.dat"
    )
    initial_pocket_t0_table = read_function(
        case, "initialPocketVolumeAtT0", "volFieldValue.dat"
    )
    initial_riser_t0_table = read_function(
        case, "initialRiserWaterVolumeAtT0", "volFieldValue.dat"
    )
    internal_air_pt_table = read_function(case, "internalAirPT", "volFieldValue.dat")
    all_speed_table = read_function(case, "allSpeedMaximum", "volFieldValue.dat")
    water_speed_table = read_function(
        case, "waterSpeedMaximum", "volFieldValue.dat"
    )
    gas_speed_table = read_function(case, "gasSpeedMaximum", "volFieldValue.dat")
    inlet_water_flux_table = read_function(
        case, "inletWaterFlux", "surfaceFieldValue.dat"
    )
    atmosphere_water_flux_table = read_function(
        case, "atmosphereWaterFlux", "surfaceFieldValue.dat"
    )
    atmosphere_air_pt_flux_table = read_function(
        case, "atmosphereAirPTFlux", "surfaceFieldValue.dat"
    )
    inlet_mass_flux_table = read_function(
        case, "inletMassFlux", "surfaceFieldValue.dat"
    )
    atmosphere_mass_flux_table = read_function(
        case, "atmosphereMassFlux", "surfaceFieldValue.dat"
    )
    rim_water_flux_table = read_function(
        case, "rimWaterFlux", "surfaceFieldValue.dat"
    )

    water_volume = interpolate(water_volume_table, time)
    total_mass = interpolate(total_mass_table, time)
    air_mass = interpolate(air_pt_table, time) / R_AIR
    internal_air_mass = interpolate(internal_air_pt_table, time) / R_AIR
    all_domain_speed = interpolate(all_speed_table, time)
    water_weighted_speed = interpolate(water_speed_table, time)
    gas_weighted_speed = interpolate(gas_speed_table, time)
    q_inlet_water = interpolate(inlet_water_flux_table, time)
    q_atmos_water = interpolate(atmosphere_water_flux_table, time)
    q_atmos_air_mass = interpolate(atmosphere_air_pt_flux_table, time) / R_AIR
    q_inlet_mass = interpolate(inlet_mass_flux_table, time)
    q_atmos_mass = interpolate(atmosphere_mass_flux_table, time)
    q_rim_water = interpolate(rim_water_flux_table, time)

    finite = np.isfinite(
        q_inlet_water
        + q_atmos_water
        + q_atmos_air_mass
        + q_inlet_mass
        + q_atmos_mass
    )
    if not np.all(finite):
        raise RuntimeError("Flux function objects did not span the probe interval")

    water_boundary_integral = cumulative_trapezoid(
        time, q_inlet_water + q_atmos_water
    )
    gas_boundary_integral = cumulative_trapezoid(time, q_atmos_air_mass)
    total_boundary_integral = cumulative_trapezoid(
        time, q_inlet_mass + q_atmos_mass
    )
    water_residual = water_volume - water_volume[0] + water_boundary_integral
    gas_residual = air_mass - air_mass[0] + gas_boundary_integral
    total_mass_residual = total_mass - total_mass[0] + total_boundary_integral
    ejected_volume = cumulative_trapezoid(time, np.maximum(q_rim_water, 0.0))
    escaped_water_volume = cumulative_trapezoid(
        time, np.maximum(q_atmos_water, 0.0)
    )

    ta = first_time(time, np.isfinite(y_int))
    t_rim = first_time(time, above_rim)
    geyser = bool(np.any(above_rim))
    yfs_max = float(np.nanmax(y_fs))
    yint_max = float(np.nanmax(y_int)) if np.any(np.isfinite(y_int)) else None
    vfs = trajectory_slope(time, y_fs, INITIAL_FS + 0.03, min(RIM_Z, yfs_max) - 0.03)
    vint = (
        trajectory_slope(time, y_int, 0.10, min(RIM_Z, yint_max) - 0.03)
        if yint_max is not None
        else None
    )

    measured, model_1d = load_reference(reference_root)
    initial_pocket_mesh = float(scalar_from_file(initial_pocket_t0_table)[0])
    initial_riser_mesh = float(scalar_from_file(initial_riser_t0_table)[0])
    fs_drift = float(np.nanmax(np.abs(y_fs - y_fs[0])))
    pocket_drift = float(
        np.nanmax(
            np.abs(
                scalar_from_file(pocket_volume_table)
                / scalar_from_file(pocket_volume_table)[0]
                - 1.0
            )
        )
    )
    closed_hold_pass = (
        bool(
            fs_drift <= 0.01
            and pocket_drift <= 0.01
            and np.nanmax(water_weighted_speed) <= 0.02
        )
        if args.run_mode == "closed"
        else None
    )

    metrics = {
        "schema_version": 1,
        "case": "B-H3",
        "run_id": args.run_id,
        "run_mode": args.run_mode,
        "valve_opening": args.valve_opening,
        "simulated_end_time_s": float(time[-1]),
        "full_13s_window_completed": bool(time[-1] >= 12.99),
        "geometry_model": "3-D circular pipe, circular riser, Boolean tee, external air",
        "solver": "compressibleInterFoam",
        "geysering": geyser,
        "geyser_detection": "alpha.water >= 0.05 at centreline probes above z=1.85 m",
        "Ta_3d_s": ta,
        "t_rim_3d_s": t_rim,
        "Yfs_max_3d_m": yfs_max,
        "Yint_max_3d_m": yint_max,
        "vfs_3d_m_per_s": vfs,
        "vint_3d_m_per_s": vint,
        "maximum_sampled_speed_m_per_s": float(np.nanmax(max_probe_speed)),
        "maximum_all_domain_speed_m_per_s": float(
            np.nanmax(all_domain_speed)
        ),
        "maximum_water_weighted_speed_m_per_s": float(
            np.nanmax(water_weighted_speed)
        ),
        "maximum_gas_weighted_speed_m_per_s": float(
            np.nanmax(gas_weighted_speed)
        ),
        "initial_volume_audit": {
            "pocket_target_m3": POCKET_TARGET,
            "pocket_mesh_m3": initial_pocket_mesh,
            "pocket_relative_error": initial_pocket_mesh / POCKET_TARGET - 1.0,
            "paper_riser_water_target_m3": PAPER_RISER_WATER_TARGET,
            "paper_vair_over_vw_target": (
                POCKET_TARGET / PAPER_RISER_WATER_TARGET
            ),
            "mesh_pocket_over_paper_vw": (
                initial_pocket_mesh / PAPER_RISER_WATER_TARGET
            ),
            "riser_nonoverlap_water_target_m3": (
                NONOVERLAP_RISER_WATER_TARGET
            ),
            "riser_nonoverlap_water_mesh_m3": initial_riser_mesh,
            "riser_nonoverlap_relative_error": (
                initial_riser_mesh / NONOVERLAP_RISER_WATER_TARGET - 1.0
            ),
            "pocket_ideal_gas_mass_target_kg": AIR_MASS_TARGET,
            "pocket_ideal_gas_mass_from_mesh_volume_kg": (
                101325.0 * initial_pocket_mesh / (R_AIR * 296.15)
            ),
        },
        "conservation": {
            "max_abs_water_volume_residual_m3": float(
                np.nanmax(np.abs(water_residual))
            ),
            "max_abs_water_volume_residual_fraction": float(
                np.nanmax(np.abs(water_residual)) / max(water_volume[0], 1e-30)
            ),
            "max_abs_global_gas_mass_residual_kg": float(
                np.nanmax(np.abs(gas_residual))
            ),
            "max_abs_global_gas_mass_residual_fraction": float(
                np.nanmax(np.abs(gas_residual)) / max(air_mass[0], 1e-30)
            ),
            "max_abs_total_mass_residual_kg": float(
                np.nanmax(np.abs(total_mass_residual))
            ),
            "max_abs_total_mass_residual_fraction": float(
                np.nanmax(np.abs(total_mass_residual)) / max(total_mass[0], 1e-30)
            ),
            "internal_air_mass_initial_kg": float(internal_air_mass[0]),
            "internal_air_mass_final_kg": float(internal_air_mass[-1]),
            "total_mass_boundary_flux": (
                "direct sum(rhoPhi) on inlet and atmosphere"
            ),
            "residual_reference": (
                "first common runtime sample; t=0 volumes are audited separately"
            ),
        },
        "ejection": {
            "cumulative_positive_rim_water_volume_m3": float(ejected_volume[-1]),
            "cumulative_positive_rim_water_volume_l": float(
                1000.0 * ejected_volume[-1]
            ),
            "cumulative_atmosphere_escape_water_volume_m3": float(
                escaped_water_volume[-1]
            ),
        },
        "closed_hold": {
            "applicable": args.run_mode == "closed",
            "free_surface_max_drift_m": fs_drift,
            "initial_pocket_zone_max_relative_volume_drift": pocket_drift,
            "pass": closed_hold_pass,
            "criteria": {
                "free_surface_drift_m": 0.01,
                "pocket_relative_volume_drift": 0.01,
                "water_weighted_speed_m_per_s": 0.02,
            },
            "all_domain_and_gas_speed_are_reported_but_not_gates": (
                "Low-density gas-side CSF velocity is monitored separately; "
                "the hold gate uses phase-weighted water speed plus interface "
                "and pocket-volume drift."
            ),
        },
        "experiment": {
            "geyser": bool(int(measured["geyser_meas"])),
            "Ta_s": safe_float(measured["Ta_meas_s"]),
            "vfs_m_per_s": safe_float(measured["vfs_meas"]),
            "vint_m_per_s": safe_float(measured["vint_meas"]),
        },
        "existing_1d": {
            "geyser": bool(int(model_1d["geyser_model"])),
            "Ta_s": safe_float(model_1d["Ta_model_s"]),
            "vfs_m_per_s": safe_float(model_1d["v_fs_model"]),
            "vint_m_per_s": safe_float(model_1d["v_int_model"]),
            "Yfs_max_m": safe_float(model_1d["Yfs_max_m"]),
        },
        "classification_changed_by_numerics": None,
        "honesty_note": (
            "Experimental classification is a comparison target only. "
            "No source term or fitted event trigger is used."
        ),
    }
    prefix.with_name(prefix.name + "_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    timeseries_path = prefix.with_name(prefix.name + "_timeseries.csv")
    with timeseries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "time_s",
                "pt1_abs_pa",
                "pt2_abs_pa",
                "pt1_gauge_head_m",
                "pt2_gauge_head_m",
                "Yfs_m",
                "Yint_m",
                "global_air_mass_kg",
                "internal_air_mass_kg",
                "water_volume_m3",
                "total_mass_kg",
                "all_domain_speed_max_m_per_s",
                "water_weighted_speed_max_m_per_s",
                "gas_weighted_speed_max_m_per_s",
                "inlet_total_mass_flow_kg_s",
                "atmosphere_total_mass_flow_kg_s",
                "rim_water_flow_m3_s",
                "atmosphere_water_flow_m3_s",
                "cumulative_rim_ejected_m3",
                "water_volume_residual_m3",
                "gas_mass_residual_kg",
                "total_mass_residual_kg",
            ]
        )
        for row in zip(
            time,
            pt1,
            pt2,
            (pt1 - 101325.0) / (RHO_WATER * GRAVITY),
            (pt2 - 101325.0) / (RHO_WATER * GRAVITY),
            y_fs,
            y_int,
            air_mass,
            internal_air_mass,
            water_volume,
            total_mass,
            all_domain_speed,
            water_weighted_speed,
            gas_weighted_speed,
            q_inlet_mass,
            q_atmos_mass,
            q_rim_water,
            q_atmos_water,
            ejected_volume,
            water_residual,
            gas_residual,
            total_mass_residual,
        ):
            writer.writerow(row)

    comparison_path = prefix.with_name(prefix.name + "_comparison.csv")
    comparison_rows = (
        ("geyser", int(measured["geyser_meas"]), int(model_1d["geyser_model"]), int(geyser), "boolean"),
        ("Ta", measured["Ta_meas_s"], model_1d["Ta_model_s"], ta, "s"),
        ("vfs", measured["vfs_meas"], model_1d["v_fs_model"], vfs, "m/s"),
        ("vint", measured["vint_meas"], model_1d["v_int_model"], vint, "m/s"),
        ("Yfs_max", "", model_1d["Yfs_max_m"], yfs_max, "m"),
    )
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "experiment", "existing_1d", "openfoam_3d", "unit"])
        writer.writerows(comparison_rows)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(time, (pt1 - 101325) / (RHO_WATER * GRAVITY), label="PT1")
    axes[0].plot(time, (pt2 - 101325) / (RHO_WATER * GRAVITY), label="PT2")
    axes[0].set_ylabel("gauge head (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, y_fs, label="3-D Yfs")
    axes[1].plot(time, y_int, label="3-D Yint")
    axes[1].axhline(RIM_Z, color="k", linestyle="--", label="physical rim")
    axes[1].axvline(float(measured["Ta_meas_s"]), color="tab:red", linestyle=":", label="exp Ta")
    axes[1].set_ylabel("elevation (m)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)

    axes[2].plot(time, gas_residual * 1e6, label="gas mass residual (mg)")
    axes[2].plot(time, water_residual * 1e6, label="water residual (mL)")
    axes[2].set_ylabel("conservation residual")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Cong2017 B-H3 — {args.run_id}")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_summary.png"), dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=False))
    if args.run_mode == "closed" and closed_hold_pass is not True:
        raise SystemExit("Closed-hold acceptance criteria failed")


if __name__ == "__main__":
    main()
