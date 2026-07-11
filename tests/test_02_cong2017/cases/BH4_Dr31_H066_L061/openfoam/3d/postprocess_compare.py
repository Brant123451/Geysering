#!/usr/bin/env python3
"""Reduce one B-H4 OpenFOAM run to compact, auditable CSV/JSON/PNG outputs."""

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
PATM = 101325.0
RHO_WATER = 998.0
GRAVITY = 9.81
PIPE_CROWN_Z = 0.050
RISER_RIM_Z = 1.850
RISER_Z = np.arange(0.060, 3.040 + 1.0e-9, 0.020)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=HERE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "outputs" / "openfoam3d",
    )
    parser.add_argument("--label", default="base_topen0p20")
    parser.add_argument("--mesh-metadata", type=Path, default=None)
    return parser.parse_args()


def numeric_time(path: Path) -> float:
    try:
        return float(path.parent.name)
    except ValueError:
        return -1.0


def data_files(case_dir: Path, function_name: str, filename: str) -> list[Path]:
    files = list(
        (case_dir / "postProcessing" / function_name).glob(f"*/{filename}")
    )
    return sorted(files, key=numeric_time)


def read_numeric_rows(files: list[Path]) -> np.ndarray:
    rows: dict[float, list[float]] = {}
    number = re.compile(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    )
    for path in files:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(token) for token in number.findall(line)]
            if len(values) >= 2:
                rows[values[0]] = values
    if not rows:
        names = ", ".join(str(path) for path in files) or "<none>"
        raise FileNotFoundError(f"No numeric OpenFOAM rows found in {names}")
    width = min(len(row) for row in rows.values())
    return np.asarray([rows[t][:width] for t in sorted(rows)], dtype=float)


def read_function_table(
    case_dir: Path, function_name: str, filename: str
) -> np.ndarray:
    return read_numeric_rows(data_files(case_dir, function_name, filename))


def interp_column(table: np.ndarray | None, times: np.ndarray, column: int) -> np.ndarray:
    if table is None or table.shape[1] <= column:
        return np.full_like(times, np.nan)
    return np.interp(times, table[:, 0], table[:, column])


def first_finite_time(times: np.ndarray, values: np.ndarray) -> float | None:
    valid = np.flatnonzero(np.isfinite(values))
    return None if not valid.size else float(times[valid[0]])


def crossing_time(
    times: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    for index in range(1, len(times)):
        y0, y1 = values[index - 1], values[index]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 < threshold <= y1:
            fraction = (threshold - y0) / max(y1 - y0, 1.0e-14)
            return float(times[index - 1] + fraction * (times[index] - times[index - 1]))
    return None


def interface_levels(
    alpha: np.ndarray, probe_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    yfs = np.full(alpha.shape[0], np.nan)
    yint = np.full(alpha.shape[0], np.nan)
    for row_index, profile in enumerate(alpha):
        water = np.flatnonzero(profile >= 0.5)
        if not water.size:
            continue
        top_index = int(water[-1])
        yfs[row_index] = probe_z[top_index]
        gas_below_surface = np.flatnonzero(
            (profile[: top_index + 1] < 0.5)
            & (probe_z[: top_index + 1] > PIPE_CROWN_Z)
        )
        if gas_below_surface.size:
            yint[row_index] = probe_z[int(gas_below_surface[-1])]
    return yfs, yint


def fit_velocity(
    times: np.ndarray,
    values: np.ndarray,
    start: float | None,
    stop: float | None,
) -> float | None:
    if start is None:
        return None
    upper = times[-1] if stop is None else stop
    mask = (
        (times >= start)
        & (times <= upper)
        & np.isfinite(values)
    )
    if np.count_nonzero(mask) < 4:
        return None
    return float(np.polyfit(times[mask], values[mask], 1)[0])


def cumulative_trapezoid(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(times)
    if len(times) > 1:
        dt = np.diff(times)
        result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * dt)
    return result


def safe_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def read_experiment() -> dict[str, str]:
    path = CASE_ROOT / "data" / "series_b_measurement.csv"
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def read_one_dimensional() -> dict[str, str]:
    path = CASE_ROOT / "outputs" / "series_b_model_summary.csv"
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_probe = read_function_table(
        case_dir, "riserCentreline", "alpha.water"
    )
    time = alpha_probe[:, 0]
    alpha = alpha_probe[:, 1:]
    probe_count = min(alpha.shape[1], len(RISER_Z))
    alpha = alpha[:, :probe_count]
    probe_z = RISER_Z[:probe_count]
    yfs_z, yint_z = interface_levels(alpha, probe_z)

    ta = first_finite_time(time, yint_z)
    t_break = None
    if ta is not None:
        after = np.flatnonzero(
            (time > ta + 0.05)
            & np.isfinite(yint_z)
            & np.isfinite(yfs_z)
            & ((yfs_z - yint_z) <= 0.04)
        )
        if after.size:
            t_break = float(time[after[0]])
    t_rim = crossing_time(time, yfs_z, RISER_RIM_Z - 0.01)
    fit_stop = t_rim if t_rim is not None else t_break
    vfs = fit_velocity(time, yfs_z, ta, fit_stop)
    vint = fit_velocity(time, yint_z, ta, fit_stop)

    def optional_table(function_name: str, filename: str) -> np.ndarray | None:
        files = data_files(case_dir, function_name, filename)
        return None if not files else read_numeric_rows(files)

    pocket_pressure_table = optional_table(
        "pocketPressure", "volFieldValue.dat"
    )
    pt2_pressure_table = optional_table("pt2Pressure", "volFieldValue.dat")
    pocket_volume_table = optional_table(
        "pocketGasVolume", "volFieldValue.dat"
    )
    pocket_mass_table = optional_table(
        "pocketGasMass", "volFieldValue.dat"
    )
    external_water_table = optional_table(
        "externalWaterVolume", "volFieldValue.dat"
    )
    mixed_table = optional_table(
        "interfaceMixedVolume", "volFieldValue.dat"
    )
    phase_volume_table = optional_table("phaseVolumes", "volFieldValue.dat")
    phase_mass_table = optional_table("phaseMasses", "volFieldValue.dat")
    atmosphere_water_flux_table = optional_table(
        "atmosphereWaterVolumeFlux", "surfaceFieldValue.dat"
    )
    reservoir_water_flux_table = optional_table(
        "reservoirWaterVolumeFlux", "surfaceFieldValue.dat"
    )
    atmosphere_gas_flux_table = optional_table(
        "atmosphereGasMassFlux", "surfaceFieldValue.dat"
    )

    pocket_pressure = interp_column(pocket_pressure_table, time, 1)
    pt2_pressure = interp_column(pt2_pressure_table, time, 1)
    pocket_volume = interp_column(pocket_volume_table, time, 1)
    pocket_mass = interp_column(pocket_mass_table, time, 1)
    external_water = interp_column(external_water_table, time, 1)
    mixed_volume = interp_column(mixed_table, time, 1)
    water_volume = interp_column(phase_volume_table, time, 1)
    gas_volume = interp_column(phase_volume_table, time, 2)
    water_mass = interp_column(phase_mass_table, time, 1)
    gas_mass = interp_column(phase_mass_table, time, 2)
    atmosphere_water_flux = interp_column(
        atmosphere_water_flux_table, time, 1
    )
    reservoir_water_flux = interp_column(
        reservoir_water_flux_table, time, 1
    )
    atmosphere_gas_flux = interp_column(
        atmosphere_gas_flux_table, time, 1
    )

    water_net_out = cumulative_trapezoid(
        time, np.nan_to_num(atmosphere_water_flux + reservoir_water_flux)
    )
    ejected_water = cumulative_trapezoid(
        time, np.maximum(np.nan_to_num(atmosphere_water_flux), 0)
    )
    gas_net_out = cumulative_trapezoid(
        time, np.nan_to_num(atmosphere_gas_flux)
    )
    water_balance = water_volume - water_volume[0] + water_net_out
    gas_balance = gas_mass - gas_mass[0] + gas_net_out

    yfs_above_crown = yfs_z - PIPE_CROWN_Z
    yint_above_crown = yint_z - PIPE_CROWN_Z
    max_yfs = float(np.nanmax(yfs_z))
    geyser = bool(max_yfs >= RISER_RIM_Z - 0.01)
    pocket_head = (pocket_pressure - PATM) / (RHO_WATER * GRAVITY)
    pt2_head = (pt2_pressure - PATM) / (RHO_WATER * GRAVITY)

    experiment = read_experiment()
    one_d = read_one_dimensional()
    mesh_metadata_path = args.mesh_metadata
    if mesh_metadata_path is None:
        mesh_metadata_path = case_dir / "mesh_metadata.json"
    mesh_metadata = (
        json.loads(mesh_metadata_path.read_text())
        if mesh_metadata_path.exists()
        else {}
    )

    initial_gas_mass = gas_mass[0] if np.isfinite(gas_mass[0]) else np.nan
    initial_water_volume = (
        water_volume[0] if np.isfinite(water_volume[0]) else np.nan
    )
    max_water_balance = float(np.nanmax(np.abs(water_balance)))
    max_gas_balance = float(np.nanmax(np.abs(gas_balance)))
    metrics = {
        "case": "B-H4",
        "label": args.label,
        "solver": "compressibleInterFoam",
        "simulated_end_time_s": float(time[-1]),
        "classification_3d": "GEYSER" if geyser else "NO_GEYSER",
        "classification_experiment": "NO_GEYSER",
        "classification_match": not geyser,
        "Ta_3d_s": safe_float(ta),
        "Ta_experiment_s": float(experiment["Ta_meas_s"]),
        "vfs_3d_m_per_s": safe_float(vfs),
        "vfs_experiment_m_per_s": float(experiment["vfs_meas"]),
        "vint_3d_m_per_s": safe_float(vint),
        "vint_experiment_m_per_s": float(experiment["vint_meas"]),
        "Yfs_max_above_crown_m": max_yfs - PIPE_CROWN_Z,
        "Yfs_initial_above_crown_m": safe_float(float(yfs_above_crown[0])),
        "Yfs_max_drift_from_initial_m": safe_float(
            float(np.nanmax(np.abs(yfs_above_crown - yfs_above_crown[0])))
        ),
        "Yfs_rim_above_crown_m": RISER_RIM_Z - PIPE_CROWN_Z,
        "external_water_max_m3": safe_float(float(np.nanmax(external_water))),
        "ejected_water_positive_flux_m3": safe_float(float(ejected_water[-1])),
        "pocket_pressure_peak_gauge_head_m": safe_float(
            float(np.nanmax(pocket_head))
        ),
        "pt2_pressure_peak_gauge_head_m": safe_float(float(np.nanmax(pt2_head))),
        "pocket_volume_min_m3": safe_float(float(np.nanmin(pocket_volume))),
        "pocket_volume_initial_m3": safe_float(float(pocket_volume[0])),
        "pocket_volume_final_m3": safe_float(float(pocket_volume[-1])),
        "pocket_volume_initial_relative_error": safe_float(
            float(pocket_volume[0] / 0.0011977321991811086 - 1)
        ),
        "pocket_volume_relative_change": safe_float(
            float(pocket_volume[-1] / max(abs(pocket_volume[0]), 1e-30) - 1)
        ),
        "pocket_air_mass_initial_kg": safe_float(float(pocket_mass[0])),
        "pocket_air_mass_initial_relative_error": safe_float(
            float(pocket_mass[0] / 0.0014276016764529437 - 1)
        ),
        "riser_interface_mixed_volume_max_m3": safe_float(
            float(np.nanmax(mixed_volume))
        ),
        "initial_water_volume_m3": safe_float(float(initial_water_volume)),
        "initial_gas_mass_domain_kg": safe_float(float(initial_gas_mass)),
        "max_water_volume_balance_error_m3": max_water_balance,
        "max_water_volume_balance_relative": safe_float(
            max_water_balance / max(abs(initial_water_volume), 1e-30)
        ),
        "max_gas_mass_balance_error_kg": max_gas_balance,
        "max_gas_mass_balance_relative": safe_float(
            max_gas_balance / max(abs(initial_gas_mass), 1e-30)
        ),
        "existing_1d": one_d,
        "mesh": mesh_metadata,
        "notes": [
            "PT1 is a gas-volume-weighted pocket pressure proxy because the paper does not report its exact x.",
            "Yfs and Yint use alpha.water=0.5 on the three-dimensional riser centreline.",
            "Gas balance includes the atmosphere gas-mass flux; water balance includes reservoir and atmosphere volume fluxes.",
        ],
    }

    csv_path = output_dir / f"{args.label}_timeseries.csv"
    fieldnames = [
        "time_s",
        "Yfs_m_above_crown",
        "Yint_m_above_crown",
        "pocket_gauge_head_m",
        "pt2_gauge_head_m",
        "pocket_gas_volume_m3",
        "pocket_gas_mass_kg",
        "external_water_volume_m3",
        "atmosphere_water_flux_m3_s",
        "cumulative_ejected_water_m3",
        "water_balance_error_m3",
        "gas_balance_error_kg",
        "riser_interface_mixed_volume_m3",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for index, t_value in enumerate(time):
            writer.writerow(
                [
                    f"{t_value:.8g}",
                    f"{yfs_above_crown[index]:.8g}",
                    f"{yint_above_crown[index]:.8g}",
                    f"{pocket_head[index]:.8g}",
                    f"{pt2_head[index]:.8g}",
                    f"{pocket_volume[index]:.8g}",
                    f"{pocket_mass[index]:.8g}",
                    f"{external_water[index]:.8g}",
                    f"{atmosphere_water_flux[index]:.8g}",
                    f"{ejected_water[index]:.8g}",
                    f"{water_balance[index]:.8g}",
                    f"{gas_balance[index]:.8g}",
                    f"{mixed_volume[index]:.8g}",
                ]
            )

    metrics_path = output_dir / f"{args.label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(time, yfs_above_crown, label="3D Yfs")
    axes[0, 0].plot(time, yint_above_crown, label="3D Yint")
    axes[0, 0].axhline(1.8, color="k", linestyle=":", label="physical rim")
    axes[0, 0].set_ylabel("height above pipe crown [m]")
    axes[0, 0].legend()

    axes[0, 1].plot(time, pocket_head, label="PT1 pocket proxy")
    axes[0, 1].plot(time, pt2_head, label="PT2")
    axes[0, 1].axhline(0.66, color="k", linestyle=":", label="H0")
    axes[0, 1].set_ylabel("gauge pressure head [m water]")
    axes[0, 1].legend()

    axes[1, 0].plot(time, 1e6 * pocket_volume, label="pocket gas")
    axes[1, 0].plot(time, 1e6 * external_water, label="water above rim")
    axes[1, 0].plot(time, 1e6 * ejected_water, label="cumulative water out")
    axes[1, 0].set_ylabel("volume [mL]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].legend()

    axes[1, 1].plot(
        time,
        water_balance / max(abs(initial_water_volume), 1e-30),
        label="water volume balance",
    )
    axes[1, 1].plot(
        time,
        gas_balance / max(abs(initial_gas_mass), 1e-30),
        label="gas mass balance",
    )
    axes[1, 1].plot(
        time,
        mixed_volume / max(np.nanmax(mixed_volume), 1e-30),
        label="normalised interface mixing",
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("relative / normalised")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].legend()

    fig.suptitle(
        f"Cong 2017 B-H4 — {args.label}: "
        f"{metrics['classification_3d']} (experiment NO_GEYSER)"
    )
    figure_path = output_dir / f"{args.label}_comparison.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"wrote {csv_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
