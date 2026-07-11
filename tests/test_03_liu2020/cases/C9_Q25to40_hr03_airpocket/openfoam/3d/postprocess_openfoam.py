#!/usr/bin/env python3
"""Convert actual C9 OpenFOAM function-object data into required artifacts.

Missing or short simulations remain explicitly incomplete.  This script never
substitutes the existing one-dimensional result for three-dimensional data.
"""
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
C9_ROOT = HERE.parents[1]
OUTPUTS = C9_ROOT / "outputs"
DATA = C9_ROOT / "data" / "digitized"
P_ATM = 101325.0
RHO_W = 998.2
RAMP_OFFSET = 0.25
RIM_HEIGHT = 1.22
CHAMBER_VOLUME = 0.30 * 0.30 * 0.45
UPSTREAM_VOLUME = math.pi * 0.20**2 / 4.0 * 5.80

PAPER = {
    "P1m_kPa": 10.69,
    "t_P1m_s": 0.50,
    "t_first_top_s": 0.73,
    "T_osc_s": 1.45,
    "air_pocket_arrival_s": 6.46,
    "geyser_count": 8,
    "PT2_final_kPa": 8.79,
    "PT3_final_kPa": 12.76,
    "PT4_final_kPa": 9.25,
}


def numeric_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir():
            try:
                float(child.name)
            except ValueError:
                continue
            out.append(child)
    return sorted(out, key=lambda path: float(path.name))


def parse_probe_scalar(post: Path, name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    rows: dict[float, list[float]] = {}
    for directory in numeric_dirs(post / name):
        path = directory / field
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            try:
                time = float(values[0])
                row = [float(value) for value in values[1:]]
            except (ValueError, IndexError):
                continue
            rows[time] = row
    if not rows:
        return np.empty(0), np.empty((0, 0))
    times = np.array(sorted(rows), dtype=float)
    width = min(len(rows[time]) for time in times)
    values = np.asarray([rows[time][:width] for time in times], dtype=float)
    return times, values


def parse_function(post: Path, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: dict[float, list[float]] = {}
    headers: list[str] = []
    for directory in numeric_dirs(post / name):
        candidates = sorted(directory.glob("*.dat"))
        if not candidates:
            candidates = [path for path in directory.iterdir() if path.is_file()]
        for path in candidates:
            for line in path.read_text(errors="replace").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    headers.append(stripped)
                    continue
                parts = stripped.replace("(", " ").replace(")", " ").split()
                try:
                    time = float(parts[0])
                    values = [float(value) for value in parts[1:]]
                except (ValueError, IndexError):
                    continue
                if values:
                    rows[time] = values
    if not rows:
        return np.empty(0), np.empty((0, 0)), headers
    times = np.asarray(sorted(rows), dtype=float)
    width = min(len(rows[time]) for time in times)
    values = np.asarray([rows[time][:width] for time in times], dtype=float)
    return times, values, headers


def first_column(post: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    times, values, _ = parse_function(post, name)
    return times, values[:, 0] if values.size else np.empty(0)


def interp_series(source_t, source_y, target_t):
    if len(source_t) == 0 or len(target_t) == 0:
        return np.full(len(target_t), np.nan)
    order = np.argsort(source_t)
    return np.interp(target_t, source_t[order], source_y[order], left=np.nan, right=np.nan)


def cumulative_trapezoid(times, values):
    result = np.zeros(len(times))
    if len(times) > 1:
        result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    return result


def local_peaks(times, values, start=0.0, end=6.5, separation=0.4):
    mask = np.isfinite(values) & (times >= start) & (times <= end)
    t = times[mask]
    y = values[mask]
    if len(y) < 5:
        return []
    window = max(1, min(21, len(y) // 30))
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(y, kernel, mode="same")
    else:
        smooth = y
    candidates = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]))[0] + 1
    candidates = sorted(candidates, key=lambda index: smooth[index], reverse=True)
    chosen = []
    for index in candidates:
        if all(abs(t[index] - t[other]) >= separation for other in chosen):
            chosen.append(index)
    return sorted([(float(t[index]), float(y[index])) for index in chosen])


def contiguous_events(times, active, minimum_duration=0.015):
    events = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(active) - 1):
            stop = index if flag and index == len(active) - 1 else index - 1
            if times[stop] - times[start] >= minimum_duration:
                events.append((start, stop))
            start = None
    return events


def read_experiment(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / f"fig9_{name}.csv"
    if not path.exists():
        return np.empty(0), np.empty(0)
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data["t_s"], float), np.asarray(data["p_med_kPa"], float)


def parse_mesh_quality(case: Path) -> dict:
    path = case / "log.checkMesh"
    result = {
        "checkMesh_run": False,
        "checkMesh_passed": False,
        "cells": None,
        "max_non_orthogonality": None,
        "max_skewness": None,
        "max_aspect_ratio": None,
        "min_volume_m3": None,
    }
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    result["checkMesh_run"] = True
    result["checkMesh_passed"] = "Mesh OK." in text
    patterns = {
        "cells": r"cells:\s+(\d+)",
        "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([-+0-9.eE]+)",
        "max_skewness": r"Max skewness\s*=\s*([-+0-9.eE]+)",
        "max_aspect_ratio": r"Max aspect ratio\s*=\s*([-+0-9.eE]+)",
        "min_volume_m3": r"Min volume\s*=\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = int(match.group(1)) if key == "cells" else float(match.group(1))
    return result


def relative_error(value, target):
    if value is None or not np.isfinite(value):
        return None
    return 100.0 * (value - target) / target


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    global OUTPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=HERE / "case")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    case = args.case.resolve()
    if args.output_dir is not None:
        OUTPUTS = args.output_dir.resolve()
    post = case / "postProcessing"
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    metadata_path = case / "generated_case.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    offset = float(metadata.get("paper_time_offset_s", RAMP_OFFSET))

    p_time_solver, p_values = parse_probe_scalar(post, "probesPT", "p")
    p_time = p_time_solver - offset
    if p_values.shape[1] >= 4:
        pressure = (p_values[:, :4] - P_ATM) / 1000.0
    else:
        pressure = np.full((len(p_time), 4), np.nan)
    write_csv(
        OUTPUTS / "openfoam_3d_PT1_PT2_PT3_PT4.csv",
        ["time_s", "PT1_kPa", "PT2_kPa", "PT3_kPa", "PT4_kPa"],
        ((t, *row) for t, row in zip(p_time, pressure)),
    )

    alpha_time_solver, alpha_values = parse_probe_scalar(post, "riserCentreline", "alpha.water")
    alpha_time = alpha_time_solver - offset
    if alpha_values.size:
        z = np.linspace(0.46, 2.67, alpha_values.shape[1])
        h50, h10, integral = [], [], []
        for row in alpha_values:
            wet50 = z[row >= 0.50]
            wet10 = z[row >= 0.10]
            h50.append(max(0.0, (wet50.max() if len(wet50) else 0.45) - 0.45))
            h10.append(max(0.0, (wet10.max() if len(wet10) else 0.45) - 0.45))
            inside = z <= 1.67
            integral.append(float(np.trapz(np.clip(row[inside], 0, 1), z[inside])))
        h50 = np.asarray(h50)
        h10 = np.asarray(h10)
        integral = np.asarray(integral)
    else:
        h50 = h10 = integral = np.empty(0)
    write_csv(
        OUTPUTS / "openfoam_3d_riser_height.csv",
        ["time_s", "alpha50_height_from_riser_bottom_m", "alpha10_mixture_height_m", "riser_water_holdup_m"],
        zip(alpha_time, h50, h10, integral),
    )

    uv_t_solver, upstream_water = first_column(post, "upstreamWaterVolume")
    uv_t = uv_t_solver - offset
    upstream_air_volume = UPSTREAM_VOLUME - upstream_water if len(upstream_water) else np.empty(0)
    um_t, upstream_mass = first_column(post, "upstreamMass")
    uwm_t, upstream_water_mass = first_column(post, "upstreamWaterMass")
    upstream_air_mass = (
        interp_series(um_t, upstream_mass, uv_t_solver)
        - interp_series(uwm_t, upstream_water_mass, uv_t_solver)
        if len(uv_t_solver)
        else np.empty(0)
    )
    cv_t, chamber_water = first_column(post, "chamberWaterVolume")
    chamber_air = CHAMBER_VOLUME - interp_series(cv_t, chamber_water, uv_t_solver)

    arrival_time = None
    if len(uv_t):
        initial_air = float(upstream_air_volume[0])
        threshold = max(0.001, 0.05 * max(initial_air, 0.0))
        candidates = np.where((uv_t >= 3.99) & (chamber_air > threshold))[0]
        if len(candidates):
            arrival_time = float(uv_t[candidates[0]])
    write_csv(
        OUTPUTS / "openfoam_3d_air_pocket.csv",
        ["time_s", "upstream_air_volume_m3", "upstream_air_mass_kg", "chamber_air_volume_m3"],
        zip(uv_t, upstream_air_volume, upstream_air_mass, chamber_air),
    )

    # Eruption events follow the paper's definition: mixture crossing the rim.
    event_rows = []
    if len(alpha_time):
        active = h10 >= RIM_HEIGHT
        for event_id, (start, stop) in enumerate(contiguous_events(alpha_time, active), start=1):
            segment = slice(start, stop + 1)
            peak_index = start + int(np.nanargmax(h10[segment]))
            event_time = float(alpha_time[peak_index])
            window = (p_time >= alpha_time[start] - 0.05) & (p_time <= alpha_time[stop] + 0.05)
            p_peaks = [
                float(np.nanmax(pressure[window, column])) if np.any(window) else math.nan
                for column in range(4)
            ]
            air_volume = (
                float(interp_series(uv_t, upstream_air_volume, np.array([event_time]))[0])
                if len(uv_t)
                else math.nan
            )
            maximum_height = float(np.nanmax(h10[segment]))
            kind = "geyser" if maximum_height > RIM_HEIGHT + 0.02 else "overflow"
            event_rows.append(
                [
                    event_id,
                    event_time,
                    *p_peaks,
                    maximum_height,
                    kind,
                    air_volume,
                ]
            )
    write_csv(
        OUTPUTS / "openfoam_3d_event_table.csv",
        [
            "event_id",
            "event_time_s",
            "PT1_peak_kPa",
            "PT2_peak_kPa",
            "PT3_peak_kPa",
            "PT4_peak_kPa",
            "maximum_riser_height_m",
            "overflow_or_geyser",
            "air_volume_m3",
        ],
        event_rows,
    )

    # Total and gas conservation from volume and flux function objects.
    mass_t, total_mass = first_column(post, "totalMass")
    wm_t, water_mass = first_column(post, "waterMass")
    gas_mass = total_mass - interp_series(wm_t, water_mass, mass_t) if len(mass_t) else np.empty(0)
    boundary_names = ("inletFlux", "gateFlux", "atmosphereFlux")
    water_flux_names = ("inletWaterMassFlux", "gateWaterMassFlux", "atmosphereWaterMassFlux")
    total_flux = np.zeros(len(mass_t))
    water_flux = np.zeros(len(mass_t))
    flux_complete = bool(len(mass_t))
    for name in boundary_names:
        ft, values = first_column(post, name)
        if not len(ft):
            flux_complete = False
        total_flux += interp_series(ft, values, mass_t)
    for name in water_flux_names:
        ft, values = first_column(post, name)
        if not len(ft):
            flux_complete = False
        water_flux += interp_series(ft, values, mass_t)
    if len(mass_t) and flux_complete:
        total_residual = total_mass - total_mass[0] + cumulative_trapezoid(mass_t, total_flux)
        gas_flux = total_flux - water_flux
        gas_residual = gas_mass - gas_mass[0] + cumulative_trapezoid(mass_t, gas_flux)
        mass_error = float(np.nanmax(np.abs(total_residual)) / max(abs(total_mass[0]), 1e-12))
        gas_error = float(np.nanmax(np.abs(gas_residual)) / max(abs(gas_mass[0]), 1e-12))
    else:
        mass_error = None
        gas_error = None

    sim_end = float(np.nanmax(p_time)) if len(p_time) else None
    p1m = t_p1m = None
    if len(p_time):
        window = (p_time >= 0.0) & (p_time <= min(1.5, p_time.max()))
        if np.any(window):
            indices = np.where(window)[0]
            index = indices[int(np.nanargmax(pressure[window, 1]))]
            p1m = float(pressure[index, 1])
            t_p1m = float(p_time[index])
    first_top = None
    if len(alpha_time):
        indices = np.where((alpha_time >= 0) & (h10 >= RIM_HEIGHT))[0]
        if len(indices):
            first_top = float(alpha_time[indices[0]])
    peaks = local_peaks(p_time, pressure[:, 1] if len(p_time) else np.empty(0))
    period = None
    if len(peaks) >= 2:
        period = float(peaks[1][0] - peaks[0][0])

    finals = [None, None, None]
    if sim_end is not None and sim_end >= 19.0:
        final_window = p_time >= 19.0
        finals = [float(np.nanmean(pressure[final_window, index])) for index in (1, 2, 3)]

    phase1_complete = sim_end is not None and sim_end >= 6.5
    phase2_complete = sim_end is not None and sim_end >= 19.0
    phase2_events = sum(
        1 for row in event_rows if arrival_time is not None and float(row[1]) >= arrival_time
    )
    metrics = {
        "case": "Liu2020 C9 three-dimensional compressible VOF",
        "status": (
            "complete_phase2"
            if phase2_complete
            else "complete_phase1_only"
            if phase1_complete
            else "smoke_only"
            if sim_end is not None
            else "not_run"
        ),
        "solver": metadata.get("application", "unknown"),
        "simulation_end_paper_time_s": sim_end,
        "paper_time_offset_s": offset,
        "phase_1": {
            "window_complete": phase1_complete,
            "reproduced": bool(phase1_complete and event_rows and p1m is not None),
            "P1m_kPa": p1m,
            "P1m_time_s": t_p1m,
            "first_riser_top_s": first_top,
            "oscillation_period_s": period,
        },
        "phase_2": {
            "window_complete": phase2_complete,
            "reproduced": bool(phase2_complete and phase2_events >= 2),
            "events_after_simulated_pocket_arrival": phase2_events,
            "note": "A phase-1 match is not counted as phase-2 reproduction.",
        },
        "simulated_geyser_count": len(event_rows),
        "experimental_geyser_count": PAPER["geyser_count"],
        "simulated_air_pocket_arrival_s": arrival_time,
        "experimental_air_pocket_arrival_s": PAPER["air_pocket_arrival_s"],
        "air_pocket_arrival_error_percent": relative_error(arrival_time, PAPER["air_pocket_arrival_s"]),
        "major_pressure_peak_error_percent": relative_error(p1m, PAPER["P1m_kPa"]),
        "major_pressure_peak_time_error_percent": relative_error(t_p1m, PAPER["t_P1m_s"]),
        "first_top_time_error_percent": relative_error(first_top, PAPER["t_first_top_s"]),
        "oscillation_period_error_percent": relative_error(period, PAPER["T_osc_s"]),
        "final_pressure_kPa": {"PT2": finals[0], "PT3": finals[1], "PT4": finals[2]},
        "final_pressure_error_percent": {
            "PT2": relative_error(finals[0], PAPER["PT2_final_kPa"]),
            "PT3": relative_error(finals[1], PAPER["PT3_final_kPa"]),
            "PT4": relative_error(finals[2], PAPER["PT4_final_kPa"]),
        },
        "mass_conservation_relative_error": mass_error,
        "gas_mass_conservation_relative_error": gas_error,
        "gas_mass_method": (
            "mixture mass minus alpha-weighted water mass; boundary gas flux is total minus "
            "alpha-weighted water mass flux"
        ),
        "initial_air_volume_m3": float(upstream_air_volume[0]) if len(upstream_air_volume) else None,
        "initial_air_mass_kg": float(upstream_air_mass[0]) if len(upstream_air_mass) else None,
        "mesh": parse_mesh_quality(case),
        "source_parameter_status": {
            "air_pocket_size": "uncertain sensitivity parameter, not reported by paper",
            "tailgate_opening": "derived boundary parameter, not reported by paper",
        },
        "paper_targets": PAPER,
    }
    with (OUTPUTS / "openfoam_3d_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")

    # Pressure comparison.
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for index, (axis, name) in enumerate(zip(axes, ("PT1", "PT2", "PT3", "PT4"))):
        exp_t, exp_p = read_experiment(name)
        axis.plot(exp_t, exp_p, color="#9ca3af", lw=1.0, label="experiment Fig. 9")
        if len(p_time):
            axis.plot(p_time, pressure[:, index], color="#0f4c81", lw=1.2, label="3-D OpenFOAM")
        axis.axvline(PAPER["air_pocket_arrival_s"], color="#a855f7", ls="--", lw=0.8)
        axis.set_ylabel(f"{name} [kPa]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("paper time [s]")
    axes[-1].set_xlim(-1, 20)
    figure.suptitle(f"C9 pressure comparison — status: {metrics['status']}")
    figure.tight_layout()
    figure.savefig(OUTPUTS / "openfoam_3d_pressure_comparison.png", dpi=180)
    plt.close(figure)

    # Riser/plume comparison against textual chronology.
    figure, axis = plt.subplots(figsize=(10, 5))
    if len(alpha_time):
        axis.plot(alpha_time, h10, label="3-D mixture front (alpha.water ≥ 0.1)", color="#0f766e")
        axis.plot(alpha_time, h50, label="3-D water front (alpha.water ≥ 0.5)", color="#2563eb")
    axis.axhline(RIM_HEIGHT, color="black", ls="--", lw=1, label="physical riser rim")
    axis.axvline(PAPER["t_first_top_s"], color="#dc2626", ls=":", label="experiment first top 0.73 s")
    axis.axvline(PAPER["air_pocket_arrival_s"], color="#a855f7", ls="--", label="experiment pocket 6.46 s")
    axis.set(xlabel="paper time [s]", ylabel="height from riser bottom [m]", xlim=(-0.5, 20))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUTS / "openfoam_3d_riser_comparison.png", dpi=180)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(10, 5))
    if len(uv_t):
        left.plot(uv_t, upstream_air_volume * 1000.0, color="#7c3aed", label="upstream air volume")
    left.set(xlabel="paper time [s]", ylabel="upstream air volume [L]", xlim=(-0.5, 20))
    left.grid(alpha=0.25)
    right = left.twinx()
    if len(uv_t):
        right.plot(uv_t, upstream_air_mass * 1000.0, color="#ea580c", label="upstream air mass")
    right.set_ylabel("upstream air mass [g]")
    left.axvline(PAPER["air_pocket_arrival_s"], color="black", ls="--", lw=0.8)
    handles = left.get_lines() + right.get_lines()
    left.legend(handles, [line.get_label() for line in handles], frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUTS / "openfoam_3d_air_pocket_evolution.png", dpi=180)
    plt.close(figure)

    sensitivity = OUTPUTS / "openfoam_3d_mesh_sensitivity.csv"
    if not sensitivity.exists():
        mesh = metrics["mesh"]
        write_csv(
            sensitivity,
            [
                "variant",
                "status",
                "cells",
                "maxCo",
                "maxDeltaT_s",
                "pocket_profile",
                "gate_area_m2",
                "contact_angle_deg",
                "cAlpha",
                "P1m_kPa",
                "first_top_s",
                "geyser_count",
                "air_arrival_s",
                "mass_error",
                "gas_mass_error",
            ],
            [
                [
                    metadata.get("mesh_profile", "base"),
                    metrics["status"],
                    mesh.get("cells"),
                    metadata.get("maxCo"),
                    metadata.get("maxDeltaT"),
                    metadata.get("pocket_profile"),
                    metadata.get("gate_area_m2"),
                    metadata.get("contact_angle_deg"),
                    metadata.get("interface_compression"),
                    p1m,
                    first_top,
                    len(event_rows),
                    arrival_time,
                    mass_error,
                    gas_error,
                ]
            ],
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
