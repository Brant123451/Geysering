#!/usr/bin/env python3
"""Create compact B-H6 3-D validation and conservation artifacts."""

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


P_ATM = 101325.0
RHO_W = 998.0
G = 9.81
H0 = 0.66
PIPE_D = 0.050
RISER_D = 0.041
L0 = 0.61
TEE_X = 3.47
VALVE_X = 5.98
RIM_Y = 1.80  # physical riser height above horizontal-pipe soffit
RIM_Z = 1.825
R_AIR = 8314.46261815324 / 28.965
TEMPERATURE = 296.15
RISER_Z = np.arange(0.035, 1.815 + 1e-12, 0.020)
PLUME_Z = np.arange(1.835, 3.015 + 1e-12, 0.020)
AUDIT_COLUMNS = (
    "time_s",
    "water_volume_m3",
    "water_mass_kg",
    "air_volume_m3",
    "air_mass_kg",
    "apparatus_air_volume_m3",
    "apparatus_air_mass_kg",
    "downstream_air_volume_m3",
    "downstream_air_mass_kg",
    "external_water_volume_m3",
    "reservoir_water_flux_out_m3_s",
    "atmosphere_water_flux_out_m3_s",
    "reservoir_air_mass_flux_out_kg_s",
    "atmosphere_air_mass_flux_out_kg_s",
    "max_speed_m_s",
    "min_pressure_Pa",
    "max_pressure_Pa",
)
INITIAL_AUDIT_COLUMNS = (
    "time_s",
    "water_volume_m3",
    "water_mass_kg",
    "air_volume_m3",
    "air_mass_kg",
    "apparatus_air_volume_m3",
    "apparatus_air_mass_kg",
    "downstream_air_volume_m3",
    "downstream_air_mass_kg",
    "external_water_volume_m3",
    "min_pressure_Pa",
    "max_pressure_Pa",
)
AUDIT_INDEX = {name: index for index, name in enumerate(AUDIT_COLUMNS)}
INITIAL_AUDIT_INDEX = {
    name: index for index, name in enumerate(INITIAL_AUDIT_COLUMNS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path.cwd())
    parser.add_argument("--reference-case-root", type=Path)
    parser.add_argument("--profile", default="custom")
    parser.add_argument("--opening-start", type=float, default=0.0)
    parser.add_argument("--opening-duration", type=float, default=0.2)
    parser.add_argument("--requested-end-time", type=float)
    parser.add_argument("--max-co", type=float)
    parser.add_argument("--max-alpha-co", type=float)
    parser.add_argument("--max-delta-t", type=float)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def numeric_directories(root: Path) -> list[Path]:
    paths = []
    for path in root.iterdir():
        if path.is_dir():
            try:
                float(path.name)
            except ValueError:
                continue
            paths.append(path)
    return sorted(paths, key=lambda path: float(path.name))


def probe_files(case: Path, name: str, field: str) -> list[Path]:
    root = case / "postProcessing" / name
    if not root.exists():
        raise FileNotFoundError(root)
    return [
        time_dir / field
        for time_dir in numeric_directories(root)
        if (time_dir / field).exists()
    ]


def read_scalar_probe(case: Path, name: str, field: str) -> np.ndarray:
    chunks = []
    for path in probe_files(case, name, field):
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size:
            chunks.append(data)
    if not chunks:
        raise RuntimeError(f"No probe data for {name}/{field}")
    data = np.vstack(chunks)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    _, reverse_indices = np.unique(data[::-1, 0], return_index=True)
    keep = np.sort(len(data) - 1 - reverse_indices)
    return data[keep]


def crossing(
    position0: float,
    position1: float,
    alpha0: float,
    alpha1: float,
    target: float = 0.5,
) -> float:
    if abs(alpha1 - alpha0) < 1e-12:
        return 0.5 * (position0 + position1)
    fraction = np.clip((target - alpha0) / (alpha1 - alpha0), 0.0, 1.0)
    return position0 + fraction * (position1 - position0)


def extract_riser_levels(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yint = np.zeros(alpha.shape[0])
    yfs = np.full(alpha.shape[0], np.nan)
    # Figure 7 interface trajectories are measured from the riser entrance at
    # the pipe soffit. H0 itself is separately defined from the pipe invert.
    elevations = RISER_Z - 0.025
    for row, profile in enumerate(alpha):
        wet = profile >= 0.5
        if not np.any(wet):
            yint[row] = np.nan
            continue
        # The air pocket eventually acquires a water-filled tail.  Select the
        # uppermost contiguous water column, which is the column between the
        # bubble nose and free surface; selecting the first wet cell would
        # mistake tail water for Yint after that topology change.
        last = int(len(wet) - 1 - np.argmax(wet[::-1]))
        first = last
        while first > 0 and wet[first - 1]:
            first -= 1
        if first == 0:
            lower = 0.0
        else:
            lower = crossing(
                elevations[first - 1],
                elevations[first],
                profile[first - 1],
                profile[first],
            )
        if last == len(wet) - 1:
            upper = elevations[-1]
        else:
            upper = crossing(
                elevations[last],
                elevations[last + 1],
                profile[last],
                profile[last + 1],
            )
        yint[row] = max(lower, 0.0)
        yfs[row] = upper
    return yint, yfs


def first_time(
    time: np.ndarray,
    condition: np.ndarray,
    after: float | None = None,
) -> float:
    mask = condition & np.isfinite(time)
    if after is not None:
        mask &= time >= after
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if indices.size else float("nan")


def moving_average(
    time: np.ndarray, values: np.ndarray, width_s: float
) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    half = width_s / 2.0
    for index, current in enumerate(time):
        mask = (
            (time >= current - half)
            & (time <= current + half)
            & np.isfinite(values)
        )
        if np.any(mask):
            result[index] = float(np.mean(values[mask]))
    return result


def interpolation_rmse(
    model_time: np.ndarray,
    model_value: np.ndarray,
    observed_time: np.ndarray,
    observed_value: np.ndarray,
) -> float:
    finite_model = np.isfinite(model_time) & np.isfinite(model_value)
    finite_observed = np.isfinite(observed_time) & np.isfinite(observed_value)
    if np.count_nonzero(finite_model) < 2:
        return float("nan")
    lo = np.min(model_time[finite_model])
    hi = np.max(model_time[finite_model])
    mask = finite_observed & (observed_time >= lo) & (observed_time <= hi)
    if not np.any(mask):
        return float("nan")
    predicted = np.interp(
        observed_time[mask],
        model_time[finite_model],
        model_value[finite_model],
    )
    return float(np.sqrt(np.mean((predicted - observed_value[mask]) ** 2)))


def interpolate_without_extrapolation(
    target_time: np.ndarray,
    source_time: np.ndarray,
    source_value: np.ndarray,
) -> np.ndarray:
    result = np.full(target_time.shape, np.nan, dtype=float)
    finite = np.isfinite(source_time) & np.isfinite(source_value)
    if np.count_nonzero(finite) < 2:
        return result
    source_time = source_time[finite]
    source_value = source_value[finite]
    inside = (target_time >= source_time[0]) & (target_time <= source_time[-1])
    result[inside] = np.interp(
        target_time[inside],
        source_time,
        source_value,
    )
    return result


def max_climb_rate(
    time: np.ndarray,
    level: np.ndarray,
    start: float,
    stop: float,
    width_s: float = 0.6,
) -> float:
    best = float("nan")
    for index, current in enumerate(time):
        if current < start or current + width_s > stop:
            continue
        mask = (
            (time >= current)
            & (time <= current + width_s)
            & np.isfinite(level)
        )
        if np.count_nonzero(mask) >= 3:
            slope = float(np.polyfit(time[mask], level[mask], 1)[0])
            best = slope if not np.isfinite(best) else max(best, slope)
    return best


def average_climb_rate(
    time: np.ndarray,
    level: np.ndarray,
    start: float,
    stop: float,
) -> float:
    mask = (
        (time >= start)
        & (time <= stop)
        & np.isfinite(time)
        & np.isfinite(level)
    )
    if np.count_nonzero(mask) < 3 or stop <= start:
        return float("nan")
    return float(np.polyfit(time[mask], level[mask], 1)[0])


def parse_tagged_rows(
    log_path: Path,
    tag: str,
    columns: tuple[str, ...],
) -> np.ndarray:
    rows = []
    pattern = re.compile(rf"{re.escape(tag)}\s+(.+)$")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values = [float(value) for value in match.group(1).split()]
        if len(values) == len(columns):
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No {tag} records in {log_path}")
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    _, reverse_indices = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - reverse_indices)]


def parse_audit(log_path: Path) -> np.ndarray:
    return parse_tagged_rows(log_path, "BH6_AUDIT", AUDIT_COLUMNS)


def parse_initial_audit(log_path: Path) -> np.ndarray:
    return parse_tagged_rows(
        log_path,
        "BH6_INITIAL_AUDIT",
        INITIAL_AUDIT_COLUMNS,
    )


def cumulative_trapezoid(time: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(time)
    if len(time) > 1:
        result[1:] = np.cumsum(
            0.5 * (value[1:] + value[:-1]) * np.diff(time)
        )
    return result


def parse_check_mesh(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    def integer(pattern: str) -> int | None:
        value = number(pattern)
        return int(value) if value is not None else None

    cells_match = re.search(r"\bcells:\s+(\d+)", text)
    return {
        "mesh_ok": "Mesh OK." in text,
        "failed_checks": integer(r"Failed\s+(\d+)\s+mesh checks"),
        "cells": int(cells_match.group(1)) if cells_match else None,
        "hexahedra": integer(r"\bhexahedra:\s+(\d+)"),
        "prisms": integer(r"\bprisms:\s+(\d+)"),
        "tetrahedra": integer(r"\btetrahedra:\s+(\d+)"),
        "polyhedra": integer(r"\bpolyhedra:\s+(\d+)"),
        "regions": integer(r"Number of regions:\s+(\d+)"),
        "duplicate_baffle_faces": integer(
            r"identical duplicate faces \(baffle faces\):\s+(\d+)"
        ),
        "max_aspect_ratio": number(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)"),
        "max_non_orthogonality_deg": number(
            r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)"
        ),
        "severely_non_orthogonal_faces": integer(
            r"severely non-orthogonal \(> 70 degrees\) faces:\s+(\d+)"
        ),
        "max_skewness": number(r"Max skewness\s*=\s*([0-9.eE+-]+)"),
        "underdetermined_cells": integer(
            r"Cells with small determinant .* number of cells:\s+(\d+)"
        ),
        "concave_cells": integer(
            r"Concave cells .* number of cells:\s+(\d+)"
        ),
        "low_weight_faces": integer(
            r"Faces with small interpolation weight .* number of faces:\s+(\d+)"
        ),
        "all_geometry_and_topology": True,
    }


def write_csv(path: Path, header: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def load_named_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    case = args.case.resolve()
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    reference_case = (
        args.reference_case_root.resolve()
        if args.reference_case_root
        else case.parents[1]
    )

    pt1 = read_scalar_probe(case, "PT1", "p")
    pt2 = read_scalar_probe(case, "PT2", "p")
    riser = read_scalar_probe(case, "riserCentreline", "alpha.water")
    plume = read_scalar_probe(case, "plumeCentreline", "alpha.water")
    audit = parse_audit(case / "log.compressibleInterFoam")
    initial = parse_initial_audit(case / "log.hydrostaticInitialize")[-1]
    initial_record = np.zeros(len(AUDIT_COLUMNS), dtype=float)
    for name in INITIAL_AUDIT_COLUMNS:
        initial_record[AUDIT_INDEX[name]] = initial[INITIAL_AUDIT_INDEX[name]]
    audit = audit[
        audit[:, AUDIT_INDEX["time_s"]]
        > initial_record[AUDIT_INDEX["time_s"]] + 1e-12
    ]
    audit = np.vstack((initial_record, audit))
    if riser.shape[1] - 1 != len(RISER_Z):
        raise RuntimeError(
            f"Expected {len(RISER_Z)} riser probes, got {riser.shape[1]-1}"
        )
    if plume.shape[1] - 1 != len(PLUME_Z):
        raise RuntimeError(
            f"Expected {len(PLUME_Z)} plume probes, got {plume.shape[1]-1}"
        )

    pt1_time = pt1[:, 0]
    pt1_head = (pt1[:, 1] - P_ATM) / (RHO_W * G * H0)
    pt1_smooth = moving_average(pt1_time, pt1_head, 0.10)
    pt2_time = pt2[:, 0]
    pt2_head = (pt2[:, 1] - P_ATM) / (RHO_W * G * H0)
    riser_time = riser[:, 0]
    yint, yfs = extract_riser_levels(riser[:, 1:])
    plume_time = plume[:, 0]
    highest_water = np.full(len(plume_time), np.nan)
    for row, profile in enumerate(plume[:, 1:]):
        wet = np.flatnonzero(profile >= 0.05)
        if wet.size:
            highest_water[row] = PLUME_Z[wet[-1]] - RIM_Z

    coverage_end_s = {
        "PT1": float(pt1_time[-1]),
        "PT2": float(pt2_time[-1]),
        "riserCentreline": float(riser_time[-1]),
        "plumeCentreline": float(plume_time[-1]),
        "conservationAudit": float(
            audit[-1, AUDIT_INDEX["time_s"]]
        ),
    }
    simulation_end_s = min(coverage_end_s.values())
    requested_end_s = (
        float(args.requested_end_time)
        if args.requested_end_time is not None
        else simulation_end_s
    )
    completion_tolerance_s = max(1e-8, 1e-6 * requested_end_s)
    run_completed = (
        simulation_end_s >= requested_end_s - completion_tolerance_s
    )
    if not run_completed:
        raise RuntimeError(
            "Incomplete result coverage: "
            f"requested {requested_end_s:g} s, common coverage ends at "
            f"{simulation_end_s:g} s; streams={coverage_end_s}"
        )
    valve_event_started = (
        args.opening_start <= simulation_end_s + completion_tolerance_s
    )

    ta = first_time(riser_time, yint >= 0.02)
    catch = first_time(
        riser_time,
        (yint >= 0.10) & ((yfs - yint) <= 0.05),
        after=ta if np.isfinite(ta) else None,
    )
    centreline_water_above_rim = bool(np.any(np.isfinite(highest_water)))
    yfs_max = float(np.nanmax(yfs)) if np.any(np.isfinite(yfs)) else float("nan")
    event_stop = catch if np.isfinite(catch) else float(riser_time[-1])
    vfs = (
        average_climb_rate(riser_time, yfs, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    vint = (
        average_climb_rate(riser_time, yint, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    vfs_max_0p6 = (
        max_climb_rate(riser_time, yfs, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    vint_max_0p6 = (
        max_climb_rate(riser_time, yint, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    ynet = np.full(yfs.shape, np.nan)
    finite_surface = np.isfinite(riser_time) & np.isfinite(yfs)
    if np.isfinite(ta) and np.count_nonzero(finite_surface) >= 2:
        yfs_at_arrival = float(
            np.interp(ta, riser_time[finite_surface], yfs[finite_surface])
        )
        finite_net = np.isfinite(yfs) & np.isfinite(yint) & (riser_time >= ta)
        ynet[finite_net] = yfs_at_arrival - (
            yfs[finite_net] - yint[finite_net]
        )
        vnet = average_climb_rate(riser_time, ynet, ta, event_stop)
    else:
        vnet = float("nan")
    taylor_velocity = 0.345 * math.sqrt(G * RISER_D)
    arrival_elapsed = ta - args.opening_start if np.isfinite(ta) else float("nan")
    horizontal_front_velocity = (
        (VALVE_X - TEE_X) / arrival_elapsed
        if np.isfinite(arrival_elapsed) and arrival_elapsed > 0.0
        else float("nan")
    )
    horizontal_front_froude = (
        horizontal_front_velocity / math.sqrt(G * PIPE_D)
        if np.isfinite(horizontal_front_velocity)
        else float("nan")
    )
    pressure_peak = (
        float(np.nanmax(pt1_smooth[pt1_time >= ta]))
        if np.isfinite(ta) and np.any(pt1_time >= ta)
        else float("nan")
    )

    audit_time = audit[:, AUDIT_INDEX["time_s"]]
    water_volume = audit[:, AUDIT_INDEX["water_volume_m3"]]
    air_mass = audit[:, AUDIT_INDEX["air_mass_kg"]]
    apparatus_air_volume = audit[:, AUDIT_INDEX["apparatus_air_volume_m3"]]
    apparatus_air_mass = audit[:, AUDIT_INDEX["apparatus_air_mass_kg"]]
    downstream_air_volume = audit[:, AUDIT_INDEX["downstream_air_volume_m3"]]
    downstream_air_mass = audit[:, AUDIT_INDEX["downstream_air_mass_kg"]]
    external_water_volume = audit[:, AUDIT_INDEX["external_water_volume_m3"]]
    water_flux = (
        audit[:, AUDIT_INDEX["reservoir_water_flux_out_m3_s"]]
        + audit[:, AUDIT_INDEX["atmosphere_water_flux_out_m3_s"]]
    )
    gas_mass_flux = (
        audit[:, AUDIT_INDEX["reservoir_air_mass_flux_out_kg_s"]]
        + audit[:, AUDIT_INDEX["atmosphere_air_mass_flux_out_kg_s"]]
    )
    cumulative_water_out = cumulative_trapezoid(audit_time, water_flux)
    cumulative_gas_out = cumulative_trapezoid(audit_time, gas_mass_flux)
    water_residual = water_volume - water_volume[0] + cumulative_water_out
    gas_residual = air_mass - air_mass[0] + cumulative_gas_out
    cumulative_far_water_out = cumulative_trapezoid(
        audit_time,
        np.maximum(
            audit[:, AUDIT_INDEX["atmosphere_water_flux_out_m3_s"]],
            0.0,
        ),
    )
    expelled_water = external_water_volume + cumulative_far_water_out
    ejection_volume_threshold = 1.0e-9
    water_above_rim = bool(
        np.nanmax(expelled_water) >= ejection_volume_threshold
    )
    estimated_rim_flow = np.gradient(expelled_water, audit_time)

    balance_rows = np.column_stack(
        (
            audit,
            cumulative_water_out,
            water_residual,
            cumulative_gas_out,
            gas_residual,
            cumulative_far_water_out,
            expelled_water,
            estimated_rim_flow,
        )
    )
    balance_header = AUDIT_COLUMNS + (
        "cumulative_water_out_m3",
        "water_volume_balance_residual_m3",
        "cumulative_air_mass_out_kg",
        "air_mass_balance_residual_kg",
        "cumulative_farfield_water_out_m3",
        "expelled_water_volume_m3",
        "estimated_riser_outflow_m3_s",
    )
    write_csv(results / "mass_balance.csv", balance_header, balance_rows)

    common_time = riser_time
    series_rows = np.column_stack(
        (
            common_time,
            yfs,
            yint,
            ynet,
            interpolate_without_extrapolation(
                common_time, pt1_time, pt1_head
            ),
            interpolate_without_extrapolation(
                common_time, pt1_time, pt1_smooth
            ),
            interpolate_without_extrapolation(
                common_time, pt2_time, pt2_head
            ),
            interpolate_without_extrapolation(
                common_time, audit_time, apparatus_air_volume
            ),
            interpolate_without_extrapolation(
                common_time, audit_time, apparatus_air_mass
            ),
            interpolate_without_extrapolation(
                common_time, audit_time, downstream_air_volume
            ),
            interpolate_without_extrapolation(
                common_time, audit_time, downstream_air_mass
            ),
            interpolate_without_extrapolation(
                common_time, audit_time, expelled_water
            ),
            interpolate_without_extrapolation(
                common_time, plume_time, highest_water
            ),
        )
    )
    write_csv(
        results / "series.csv",
        (
            "time_s",
            "Yfs_3d_m",
            "Yint_3d_m",
            "Ynet_3d_m",
            "PT1_H_over_H0_raw",
            "PT1_H_over_H0_smooth_0p10s",
            "PT2_H_over_H0_raw",
            "apparatus_air_volume_m3",
            "apparatus_air_mass_kg",
            "downstream_air_volume_m3",
            "downstream_air_mass_kg",
            "expelled_water_volume_m3",
            "sampled_water_height_above_rim_m",
        ),
        series_rows,
    )

    levels_exp = load_named_csv(
        reference_case / "data" / "digitized" / "fig7a_levels.csv"
    )
    pressure_exp = load_named_csv(
        reference_case / "data" / "digitized" / "fig10b_pt1.csv"
    )
    one_d = load_named_csv(reference_case / "outputs" / "caseB_model_series.csv")
    # The frozen 1-D model reports height above the pipe invert.  Figure 7 and
    # the 3-D probes report distance above the physical riser entrance at the
    # pipe soffit, so remove one pipe diameter before plotting the trajectories.
    one_d_yfs_entrance = np.maximum(one_d["Yfs_m"] - PIPE_D, 0.0)
    one_d_yint_entrance = np.maximum(one_d["Yint_m"] - PIPE_D, 0.0)
    fs_mask = levels_exp["kind"] == "fs"
    int_mask = levels_exp["kind"] == "int"
    level_validation_covered = bool(
        valve_event_started
        and riser_time[-1] >= float(np.nanmax(levels_exp["t_s"])) - 1.0e-6
    )
    pressure_proxy_covered = bool(
        valve_event_started
        and pt1_time[-1] >= float(np.nanmax(pressure_exp["t_s"])) - 1.0e-6
    )
    if level_validation_covered:
        fs_rmse = interpolation_rmse(
            riser_time,
            yfs,
            levels_exp["t_s"][fs_mask],
            levels_exp["Y_m"][fs_mask],
        )
        int_rmse = interpolation_rmse(
            riser_time,
            yint,
            levels_exp["t_s"][int_mask],
            levels_exp["Y_m"][int_mask],
        )
    else:
        fs_rmse = float("nan")
        int_rmse = float("nan")
    if pressure_proxy_covered:
        pressure_rmse = interpolation_rmse(
            pt1_time,
            pt1_smooth,
            pressure_exp["t_s"],
            pressure_exp["HoverH0_med"],
        )
    else:
        pressure_rmse = float("nan")

    geometry = json.loads(
        (case / "geometry_audit.runtime.json").read_text(encoding="utf-8")
    )
    pre_baffle_mesh = parse_check_mesh(case / "log.checkMesh.preBaffle")
    valve_baffle_mesh = parse_check_mesh(case / "log.checkMesh")
    mesh_generation = {
        "geometry": geometry.get("geometry"),
        "element_types_3d": geometry.get("element_types_3d"),
        "element_counts_3d": geometry.get("element_counts_3d"),
        "riser_sweep": geometry.get("riser_sweep"),
        "mesh_sizes_m": geometry.get("mesh_sizes_m"),
    }
    mesh_audit = {
        "generation": mesh_generation,
        "pre_baffle": pre_baffle_mesh,
        "with_valve_baffle": valve_baffle_mesh,
    }
    (results / "check_mesh_audit.json").write_text(
        json.dumps(mesh_audit, indent=2) + "\n",
        encoding="utf-8",
    )
    nominal_pocket_volume = math.pi * PIPE_D**2 * L0 / 4.0
    nominal_pocket_mass = (
        P_ATM * nominal_pocket_volume / (R_AIR * TEMPERATURE)
    )
    paper_nominal_riser_water_volume = math.pi * RISER_D**2 * H0 / 4.0
    nominal_water_volume = (
        math.pi * PIPE_D**2 * VALVE_X / 4.0
        + math.pi * RISER_D**2 * (H0 - PIPE_D) / 4.0
    )
    numerical_pocket_volume = float(
        initial[INITIAL_AUDIT_INDEX["downstream_air_volume_m3"]]
    )
    numerical_pocket_mass = float(
        initial[INITIAL_AUDIT_INDEX["downstream_air_mass_kg"]]
    )
    initial_audit = {
        "numerical_sample_time_s": float(
            initial[INITIAL_AUDIT_INDEX["time_s"]]
        ),
        "analytical_initial_water_volume_m3": nominal_water_volume,
        "numerical_initial_water_volume_m3": float(
            initial[INITIAL_AUDIT_INDEX["water_volume_m3"]]
        ),
        "analytical_initial_pocket_volume_m3": nominal_pocket_volume,
        "paper_nominal_riser_water_volume_m3": paper_nominal_riser_water_volume,
        "analytical_Vair_over_Vw": (
            nominal_pocket_volume / paper_nominal_riser_water_volume
        ),
        "numerical_initial_downstream_air_volume_m3": numerical_pocket_volume,
        "pocket_volume_relative_mesh_error": (
            numerical_pocket_volume / nominal_pocket_volume - 1.0
        ),
        "pocket_volume_mesh_error_within_one_percent": bool(
            abs(numerical_pocket_volume / nominal_pocket_volume - 1.0) <= 0.01
        ),
        "analytical_initial_pocket_air_mass_kg": nominal_pocket_mass,
        "numerical_initial_pocket_air_mass_kg": numerical_pocket_mass,
        "whole_domain_initial_air_mass_kg": float(
            initial[INITIAL_AUDIT_INDEX["air_mass_kg"]]
        ),
        "apparatus_initial_air_mass_kg": float(
            initial[INITIAL_AUDIT_INDEX["apparatus_air_mass_kg"]]
        ),
        "note": (
            "The whole-domain gas mass includes the external atmosphere. "
            "Pocket mass is integrated in the valve-to-cap region at t=0."
        ),
    }
    (results / "initial_volume_mass_audit.json").write_text(
        json.dumps(initial_audit, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    water_scale = max(abs(water_volume[0]), 1e-30)
    gas_scale = max(abs(air_mass[0]), 1e-30)
    pocket_volume_scale = max(abs(downstream_air_volume[0]), 1e-30)
    max_speed = float(np.nanmax(audit[:, AUDIT_INDEX["max_speed_m_s"]]))
    water_volume_change = float(water_volume[-1] - water_volume[0])
    downstream_air_volume_change = float(
        downstream_air_volume[-1] - downstream_air_volume[0]
    )
    closed_hold = (
        args.opening_start > requested_end_s + completion_tolerance_s
    )
    static_limits = {
        "max_speed_m_s": 0.025,
        "water_volume_relative_change": 1.0e-6,
        "downstream_air_volume_relative_change": 1.0e-6,
    }
    static_hold_pass = (
        bool(
            run_completed
            and max_speed <= static_limits["max_speed_m_s"]
            and abs(water_volume_change) / water_scale
            <= static_limits["water_volume_relative_change"]
            and abs(downstream_air_volume_change) / pocket_volume_scale
            <= static_limits["downstream_air_volume_relative_change"]
        )
        if closed_hold
        else None
    )
    mesh_summary = {
        **valve_baffle_mesh,
        **geometry.get("mesh_sizes_m", {}),
        "element_types_3d": geometry.get("element_types_3d"),
        "element_counts_3d": geometry.get("element_counts_3d"),
        "riser_sweep": geometry.get("riser_sweep"),
        "pre_baffle": pre_baffle_mesh,
        "with_valve_baffle": valve_baffle_mesh,
    }
    metrics = {
        "case": "BH6_Dr41_H066_L061",
        "profile": args.profile,
        "requested_end_s": requested_end_s,
        "simulation_end_s": simulation_end_s,
        "run_completed": run_completed,
        "data_coverage_end_s": coverage_end_s,
        "solver": "compressibleInterFoam",
        "time_stepping": {
            "maxCo": args.max_co,
            "maxAlphaCo": args.max_alpha_co,
            "maxDeltaT_s": args.max_delta_t,
        },
        "geometry": {
            "type": "true 3-D circular pipe, circular riser, T-junction, external air",
            "main_pipe_diameter_m": PIPE_D,
            "riser_diameter_m": RISER_D,
            "area_ratio": (RISER_D / PIPE_D) ** 2,
            "pipe_length_m": 6.59,
            "tee_x_m": TEE_X,
            "valve_x_m": VALVE_X,
            "physical_riser_height_m": 1.8,
            "external_top_above_soffit_m": 3.0,
        },
        "mesh": mesh_summary,
        "valve": {
            "opening_start_s": args.opening_start,
            "opening_duration_s": args.opening_duration,
            "representation": "variable-area cyclicACMI baffle",
            "table_samples": 40,
            "passive_loss_only": True,
        },
        "events": {
            "vertical_datum": "riser entrance at pipe soffit",
            "Ta_3d_s": ta,
            "Ta_from_valve_start_3d_s": arrival_elapsed,
            "horizontal_front_velocity_3d_m_s": horizontal_front_velocity,
            "horizontal_front_Uf_over_sqrt_gD_3d": horizontal_front_froude,
            "interface_catch_3d_s": catch,
            "Yfs_max_3d_m": yfs_max,
            "vfs_average_rise_3d_m_s": vfs,
            "vint_average_rise_3d_m_s": vint,
            "vnet_average_rise_3d_m_s": vnet,
            "vTaylor_m_s": taylor_velocity,
            "vnet_exceeds_vTaylor": bool(
                np.isfinite(vnet) and vnet > taylor_velocity
            ),
            "vfs_max_sustained_0p6s_m_s": vfs_max_0p6,
            "vint_max_sustained_0p6s_m_s": vint_max_0p6,
            "water_above_rim": water_above_rim,
            "centreline_water_above_rim": centreline_water_above_rim,
            "ejection_detection_volume_threshold_m3": ejection_volume_threshold,
            "max_sampled_water_height_above_rim_m": (
                float(np.nanmax(highest_water))
                if centreline_water_above_rim
                else 0.0
            ),
            "geyser_3d": water_above_rim,
            "geyser_definition": (
                "water ejection or splash through the physical riser rim; "
                "detected from full external-domain water inventory"
            ),
            "max_expelled_water_volume_m3": float(np.nanmax(expelled_water)),
        },
        "pressure": {
            "post_arrival_peak_H_over_H0": pressure_peak,
            "PT2_min_H_over_H0": float(np.nanmin(pt2_head)),
            "PT2_max_H_over_H0": float(np.nanmax(pt2_head)),
            "same_condition_repeat_proxy": (
                "Run B-32 Fig.10(b), not B-H6; same nominal Dr/H0/L0"
            ),
            "proxy_post_arrival_slow_peak_H_over_H0": 1.4,
            "proxy_RMSE_H_over_H0_no_time_shift": pressure_rmse,
        },
        "experiment": {
            "run": "B-H6",
            "Ta_s": 8.10,
            "horizontal_front_Uf_over_sqrt_gD": 0.443,
            "vfs_m_s": 0.246,
            "vint_m_s": 0.476,
            "vnet_m_s": 0.235,
            "vTaylor_m_s": 0.219,
            "Dr_over_D": 0.82,
            "Vair_over_Vw": 1.37,
            "interface_catch_s_approximate": "10.5-10.9",
            "Yfs_initial_m": 0.58,
            "Yfs_peak_m": 1.21,
            "geyser": False,
        },
        "one_d": {
            "source": "outputs/caseB_model_series.csv",
            "geometry_difference": (
                "Frozen 1-D uses 6.0 m / tee x=2.88 m; "
                "paper-audited 3-D uses 6.59 m / tee x=3.47 m. "
                "Its native vertical datum is the pipe invert; plotted "
                "trajectories are shifted by D=0.05 m to the riser entrance."
            ),
            "Yfs_max_native_above_invert_m": float(np.nanmax(one_d["Yfs_m"])),
            "Yfs_max_above_riser_entrance_m": float(
                np.nanmax(one_d_yfs_entrance)
            ),
            "geyser": bool(
                np.nanmax(one_d_yfs_entrance) >= 0.98 * RIM_Y
            ),
        },
        "comparison": {
            "B_H6_level_validation_covered": level_validation_covered,
            "B32_pressure_proxy_covered": pressure_proxy_covered,
            "Yfs_RMSE_m_no_time_shift": fs_rmse,
            "Yint_RMSE_m_no_time_shift": int_rmse,
            "B32_proxy_PT1_RMSE_H_over_H0_no_time_shift": pressure_rmse,
            "Ta_error_s": ta - 8.10 if np.isfinite(ta) else float("nan"),
            "Uf_over_sqrt_gD_error": (
                horizontal_front_froude - 0.443
                if np.isfinite(horizontal_front_froude)
                else float("nan")
            ),
            "vfs_average_error_m_s": (
                vfs - 0.246 if np.isfinite(vfs) else float("nan")
            ),
            "vint_average_error_m_s": (
                vint - 0.476 if np.isfinite(vint) else float("nan")
            ),
            "vnet_average_error_m_s": (
                vnet - 0.235 if np.isfinite(vnet) else float("nan")
            ),
            "no_event_time_shift": True,
            "no_outcome_tuning": True,
        },
        "conservation": {
            "liquid_volume_final_residual_m3": float(water_residual[-1]),
            "liquid_volume_relative_residual": float(
                water_residual[-1] / water_scale
            ),
            "liquid_volume_max_relative_residual": float(
                np.nanmax(np.abs(water_residual)) / water_scale
            ),
            "gas_mass_final_residual_kg": float(gas_residual[-1]),
            "gas_mass_relative_residual": float(gas_residual[-1] / gas_scale),
            "gas_mass_max_relative_residual": float(
                np.nanmax(np.abs(gas_residual)) / gas_scale
            ),
            "open_boundary_fluxes_included": True,
        },
        "static_diagnostics": {
            "applicable": closed_hold,
            "pass": static_hold_pass,
            "acceptance_limits": static_limits,
            "max_speed_m_s": max_speed,
            "water_volume_change_m3": water_volume_change,
            "water_volume_relative_change": water_volume_change / water_scale,
            "downstream_air_volume_change_m3": downstream_air_volume_change,
            "downstream_air_volume_relative_change": (
                downstream_air_volume_change / pocket_volume_scale
            ),
        },
        "limitations": [
            "Paper gives PT1 topology but no numerical cap offset.",
            "Neutral 90 degree acrylic contact angle is not measured or calibrated.",
            "No BH1 3-D OpenFOAM source exists in this repository for a file-level pairing audit.",
            "Fig.10(b) pressure is B-32, a same-condition repeat, not the B-H6 camera run.",
        ],
    }
    (results / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True)
    axes[0].scatter(
        levels_exp["t_s"][fs_mask],
        levels_exp["Y_m"][fs_mask],
        s=16,
        facecolors="none",
        edgecolors="#b22222",
        label="experiment B-H6 $Y_{fs}$",
    )
    axes[0].scatter(
        levels_exp["t_s"][int_mask],
        levels_exp["Y_m"][int_mask],
        s=16,
        facecolors="none",
        edgecolors="#1f4e79",
        label="experiment B-H6 $Y_{int}$",
    )
    axes[0].plot(
        one_d["t_s"],
        one_d_yfs_entrance,
        color="0.55",
        lw=1,
        label="1-D $Y_{fs}$ (datum-aligned)",
    )
    axes[0].plot(
        one_d["t_s"],
        one_d_yint_entrance,
        color="0.55",
        lw=1,
        ls="--",
        label="1-D $Y_{int}$ (datum-aligned)",
    )
    axes[0].plot(riser_time, yfs, color="#b22222", lw=1.4, label="3-D $Y_{fs}$")
    axes[0].plot(
        riser_time, yint, color="#1f4e79", lw=1.4, ls="--", label="3-D $Y_{int}$"
    )
    axes[0].axhline(RIM_Y, color="k", lw=0.8, ls=":", label="physical rim")
    axes[0].set_ylabel("level above riser entrance [m]")
    axes[0].legend(frameon=False, fontsize=7, ncol=3)

    axes[1].fill_between(
        pressure_exp["t_s"],
        pressure_exp["HoverH0_min"],
        pressure_exp["HoverH0_max"],
        color="0.88",
        label="B-32 digitised band",
    )
    axes[1].plot(
        pressure_exp["t_s"],
        pressure_exp["HoverH0_med"],
        color="0.25",
        lw=1,
        label="B-32 median",
    )
    axes[1].plot(pt1_time, pt1_smooth, color="#e66101", lw=1.4, label="3-D PT1")
    axes[1].set_ylabel("$H/H_0$")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        audit_time,
        1e3 * apparatus_air_volume,
        label="air below physical rim",
        color="#4c78a8",
    )
    axes[2].plot(
        audit_time,
        1e3 * downstream_air_volume,
        label="air in initial pocket region",
        color="#72b7b2",
    )
    axes[2].plot(
        audit_time,
        1e6 * expelled_water,
        label="expelled water",
        color="#b22222",
    )
    axes[2].set_ylabel("air [L] / expelled water [mL]")
    axes[2].set_xlabel("time after valve motion begins [s]")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xlim(0, max(13.0, float(riser_time[-1])))
    fig.tight_layout()
    fig.savefig(results / "experiment_1d_3d_comparison.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
