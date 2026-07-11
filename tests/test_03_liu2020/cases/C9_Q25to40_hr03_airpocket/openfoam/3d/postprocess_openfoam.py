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
    "initial_PT2_kPa": 2.97,
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


def parse_probe_scalar_with_locations(
    post: Path, name: str, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a scalar probe history without silently remapping probe columns."""
    rows: dict[float, list[float]] = {}
    locations: np.ndarray | None = None
    probe_pattern = re.compile(
        r"^#\s*Probe\s+(\d+)\s+\(\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
    )
    for directory in numeric_dirs(post / name):
        path = directory / field
        if not path.exists():
            continue
        file_locations: dict[int, tuple[float, float, float]] = {}
        file_rows: list[tuple[float, list[float]]] = []
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = probe_pattern.match(stripped)
            if match:
                file_locations[int(match.group(1))] = tuple(
                    float(match.group(index)) for index in range(2, 5)
                )
                continue
            if stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            try:
                time = float(values[0])
                row = [float(value) for value in values[1:]]
            except (ValueError, IndexError):
                continue
            file_rows.append((time, row))
        if file_locations:
            indices = sorted(file_locations)
            if indices != list(range(len(indices))):
                raise ValueError(f"{path}: non-contiguous probe indices {indices}")
            current_locations = np.asarray(
                [file_locations[index] for index in indices], dtype=float
            )
            if locations is None:
                locations = current_locations
            elif locations.shape != current_locations.shape or not np.allclose(
                # OpenFOAM may change probe-header precision between stage
                # restarts (for example 2.54946 versus 2.549455).
                locations,
                current_locations,
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(f"{path}: probe locations changed across restarts")
        expected_width = len(locations) if locations is not None else None
        for time, row in file_rows:
            if expected_width is None:
                expected_width = len(row)
            if len(row) != expected_width:
                raise ValueError(
                    f"{path}: time {time:g} has {len(row)} values; "
                    f"expected {expected_width}"
                )
            rows[time] = row
    if not rows:
        return np.empty(0), np.empty((0, 0)), np.empty((0, 3))
    times = np.array(sorted(rows), dtype=float)
    widths = {len(rows[time]) for time in times}
    if len(widths) != 1:
        raise ValueError(f"{post / name}: inconsistent scalar probe row widths")
    values = np.asarray([rows[time] for time in times], dtype=float)
    if locations is None:
        locations = np.empty((0, 3))
    elif len(locations) != values.shape[1]:
        raise ValueError(
            f"{post / name}: {len(locations)} probe locations but "
            f"{values.shape[1]} data columns"
        )
    return times, values, locations


def parse_probe_scalar(post: Path, name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    times, values, _ = parse_probe_scalar_with_locations(post, name, field)
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


def function_region_volume(post: Path, name: str, fallback: float) -> float:
    _, _, headers = parse_function(post, name)
    for header in headers:
        match = re.search(r"#\s*Volume\s*:\s*([-+0-9.eE]+)", header)
        if match:
            return float(match.group(1).rstrip("."))
    return fallback


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


def contiguous_events(times, active, minimum_duration=0.015, maximum_gap=None):
    events = []
    start = None
    for index, flag in enumerate(active):
        if (
            start is not None
            and maximum_gap is not None
            and index > 0
            and times[index] - times[index - 1] > maximum_gap
        ):
            stop = index - 1
            if times[stop] - times[start] >= minimum_duration:
                events.append((start, stop))
            start = None
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(active) - 1):
            stop = index if flag and index == len(active) - 1 else index - 1
            if times[stop] - times[start] >= minimum_duration:
                events.append((start, stop))
            start = None
    return events


def first_sustained_time(times, active, minimum_duration, maximum_gap=None):
    """Return the start of the first sustained true interval."""
    events = contiguous_events(
        times,
        active,
        minimum_duration=minimum_duration,
        maximum_gap=maximum_gap,
    )
    return float(times[events[0][0]]) if events else None


def dominant_gas_component(
    x: np.ndarray, alpha_water: np.ndarray, threshold: float = 0.50
) -> tuple[float, float, float]:
    """Return front, span, and furthest gas for the dominant connected component.

    The thin crown layer can thicken or shed bubbles downstream of the main
    pocket.  Taking the furthest gas-bearing probe therefore does not identify
    the main pocket.  On the deeper probe line, the longest contiguous
    gas-dominant component is the operational main body.
    """
    valid_gas = np.isfinite(alpha_water) & (alpha_water <= threshold)
    gas_indices = np.flatnonzero(valid_gas)
    if not len(gas_indices):
        return math.nan, 0.0, math.nan

    split_at = np.where(np.diff(gas_indices) > 1)[0] + 1
    components = np.split(gas_indices, split_at)
    dominant = max(
        components,
        key=lambda indices: (
            float(x[indices[-1]] - x[indices[0]]),
            len(indices),
            float(x[indices[-1]]),
        ),
    )
    front = float(x[dominant[-1]])
    span = float(x[dominant[-1]] - x[dominant[0]])
    furthest = float(x[gas_indices[-1]])
    return front, span, furthest


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
        "strict_check_run": False,
        "all_geometry_passed": False,
        "concave_cells": None,
        "concave_faces": None,
        "max_concave_face_angle_deg": None,
        "cells": None,
        "max_non_orthogonality": None,
        "max_skewness": None,
        "max_aspect_ratio": None,
        "min_volume_m3": None,
    }
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    strict_path = case / "log.checkMesh.all"
    strict_text = strict_path.read_text(errors="replace") if strict_path.exists() else text
    result["checkMesh_run"] = True
    result["checkMesh_passed"] = "Mesh OK." in text
    result["strict_check_run"] = strict_path.exists()
    result["all_geometry_passed"] = "Mesh OK." in strict_text
    patterns = {
        "cells": r"cells:\s+(\d+)",
        "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([-+0-9.eE]+)",
        "max_skewness": r"Max skewness\s*=\s*([-+0-9.eE]+)",
        "max_aspect_ratio": r"Max aspect ratio\s*=\s*([-+0-9.eE]+)",
        "min_volume_m3": r"Min volume\s*=\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, strict_text)
        if match:
            result[key] = (
                int(match.group(1))
                if key == "cells"
                else float(match.group(1).rstrip("."))
            )
    concave = re.search(r"Concave cells .* number of cells:\s*(\d+)", strict_text)
    if concave:
        result["concave_cells"] = int(concave.group(1))
    elif "Concave cell check OK." in strict_text:
        result["concave_cells"] = 0
    concave_faces = re.search(
        rf"There are\s+(\d+)\s+faces with concave angles.*?"
        rf"Max concave angle\s*=\s*({r'[-+0-9.eE]+'})",
        strict_text,
    )
    if concave_faces:
        result["concave_faces"] = int(concave_faces.group(1))
        result["max_concave_face_angle_deg"] = float(concave_faces.group(2))
    return result


def parse_numerics(
    case: Path,
    paper_time_offset: float = RAMP_OFFSET,
    maximum_solver_time: float | None = None,
) -> dict:
    """Summarize stability controls within the available field-history window."""
    result = {
        "logs": [],
        "max_courant_number": None,
        "max_interface_courant_number": None,
        "minimum_delta_t_s": None,
        "maximum_velocity_m_s": None,
        "maximum_limited_cells": 0,
        "maximum_limited_cell_percent": 0.0,
        "maximum_limited_faces": 0,
        "maximum_limited_face_percent": 0.0,
        "maximum_limited_stage": None,
        "maximum_limited_solver_time_s": None,
        "maximum_limited_paper_time_s": None,
        "velocity_limiter_activated": False,
        "limiter_by_stage": {},
    }
    number = r"[-+0-9.eE]+"
    time_line = re.compile(rf"^Time\s*=\s*({number})\s*$")
    courant = re.compile(rf"Courant Number mean:\s*{number}\s+max:\s*({number})")
    alpha_courant = re.compile(
        rf"Interface Courant Number mean:\s*{number}\s+max:\s*({number})"
    )
    delta_t = re.compile(rf"deltaT\s*=\s*({number})")
    velocity = re.compile(rf"max\(mag\(U\)\)\s*=\s*({number})")
    limited = re.compile(
        r"limitVelocity\s+\S+\s+Limited\s+(\d+)\s+\(([-+0-9.eE]+)%\)"
        r"\s+of cells,\s+(\d+)\s+\(([-+0-9.eE]+)%\)\s+of faces"
    )
    max_co = max_alpha_co = max_velocity = None
    min_delta_t = None
    for stage in ("initialize", "smoke", "phase1", "full"):
        path = case / f"log.{stage}"
        if not path.exists():
            continue
        lines = path.read_text(errors="replace").splitlines()
        logged_times = [
            float(match.group(1))
            for line in lines
            if (match := time_line.match(line.strip()))
        ]
        if maximum_solver_time is not None and not any(
            time <= maximum_solver_time + 1e-9 for time in logged_times
        ):
            continue
        result["logs"].append(path.name)
        current_time = None
        within_history = maximum_solver_time is None or (
            bool(logged_times) and logged_times[0] <= maximum_solver_time + 1e-9
        )
        stage_result = {
            "first_activation_solver_time_s": None,
            "first_activation_paper_time_s": None,
            "maximum_limited_cells": 0,
            "maximum_limited_cell_percent": 0.0,
            "maximum_limited_faces": 0,
            "maximum_limited_face_percent": 0.0,
            "maximum_activation_solver_time_s": None,
            "maximum_activation_paper_time_s": None,
        }
        for line in lines:
            match = time_line.match(line.strip())
            if match:
                current_time = float(match.group(1))
                within_history = (
                    maximum_solver_time is None
                    or current_time <= maximum_solver_time + 1e-9
                )
            if not within_history:
                continue
            match = courant.search(line)
            if match:
                value = float(match.group(1))
                max_co = value if max_co is None else max(max_co, value)
            match = alpha_courant.search(line)
            if match:
                value = float(match.group(1))
                max_alpha_co = value if max_alpha_co is None else max(max_alpha_co, value)
            match = delta_t.search(line)
            if match:
                value = float(match.group(1))
                min_delta_t = value if min_delta_t is None else min(min_delta_t, value)
            match = velocity.search(line)
            if match:
                value = float(match.group(1))
                max_velocity = value if max_velocity is None else max(max_velocity, value)
            match = limited.search(line)
            if match:
                cells = int(match.group(1))
                cell_percent = float(match.group(2))
                faces = int(match.group(3))
                face_percent = float(match.group(4))
                if cells > 0 and stage_result["first_activation_solver_time_s"] is None:
                    stage_result["first_activation_solver_time_s"] = current_time
                    stage_result["first_activation_paper_time_s"] = (
                        current_time - paper_time_offset
                        if current_time is not None
                        else None
                    )
                if cells > stage_result["maximum_limited_cells"]:
                    stage_result["maximum_limited_cells"] = cells
                    stage_result["maximum_limited_cell_percent"] = cell_percent
                    stage_result["maximum_activation_solver_time_s"] = current_time
                    stage_result["maximum_activation_paper_time_s"] = (
                        current_time - paper_time_offset
                        if current_time is not None
                        else None
                    )
                stage_result["maximum_limited_faces"] = max(
                    stage_result["maximum_limited_faces"], faces
                )
                stage_result["maximum_limited_face_percent"] = max(
                    stage_result["maximum_limited_face_percent"], face_percent
                )
                if cells > result["maximum_limited_cells"]:
                    result["maximum_limited_cells"] = cells
                    result["maximum_limited_cell_percent"] = cell_percent
                    result["maximum_limited_stage"] = stage
                    result["maximum_limited_solver_time_s"] = current_time
                    result["maximum_limited_paper_time_s"] = (
                        current_time - paper_time_offset
                        if current_time is not None
                        else None
                    )
                result["maximum_limited_faces"] = max(
                    result["maximum_limited_faces"], faces
                )
                result["maximum_limited_face_percent"] = max(
                    result["maximum_limited_face_percent"], face_percent
                )
        result["limiter_by_stage"][stage] = stage_result
    result["max_courant_number"] = max_co
    result["max_interface_courant_number"] = max_alpha_co
    result["minimum_delta_t_s"] = min_delta_t
    result["maximum_velocity_m_s"] = max_velocity
    result["velocity_limiter_activated"] = result["maximum_limited_cells"] > 0
    return result


def relative_error(value, target):
    if value is None or not np.isfinite(value):
        return None
    return 100.0 * (value - target) / target


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
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

    alpha_time_solver, alpha_values, riser_locations = parse_probe_scalar_with_locations(
        post, "riserCentreline", "alpha.water"
    )
    alpha_time = alpha_time_solver - offset
    if alpha_values.size:
        if not len(riser_locations):
            raise ValueError("riserCentreline probe output has no coordinate header")
        z = riser_locations[:, 2]
        h50, h10, integral = [], [], []
        for row in alpha_values:
            wet50 = z[row >= 0.50]
            wet10 = z[row >= 0.10]
            h50.append(max(0.0, (wet50.max() if len(wet50) else 0.45) - 0.45))
            h10.append(max(0.0, (wet10.max() if len(wet10) else 0.45) - 0.45))
            inside = z <= 1.67
            integral.append(float(np.trapezoid(np.clip(row[inside], 0, 1), z[inside])))
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

    deep_t_solver, deep_alpha, deep_locations = parse_probe_scalar_with_locations(
        post, "upstreamCrownDeep", "alpha.water"
    )
    deep_t = deep_t_solver - offset
    if deep_alpha.size:
        if not len(deep_locations):
            raise ValueError("upstreamCrownDeep probe output has no coordinate header")
        deep_x = deep_locations[:, 0]
        component_diagnostics = [
            dominant_gas_component(deep_x, row) for row in deep_alpha
        ]
        main_body_front = np.asarray(
            [diagnostic[0] for diagnostic in component_diagnostics], dtype=float
        )
        main_body_span = np.asarray(
            [diagnostic[1] for diagnostic in component_diagnostics], dtype=float
        )
        furthest_deep_gas = np.asarray(
            [diagnostic[2] for diagnostic in component_diagnostics], dtype=float
        )
        pocket = metadata.get("pocket", {})
        prior_body_length = float(pocket.get("body_nose_x_m", -1.0)) - float(
            pocket.get("tail_x_m", -4.8)
        )
        minimum_body_span = max(0.50, 0.25 * prior_body_length)
        probe_spacing = float(np.median(np.diff(deep_x))) if len(deep_x) > 1 else 0.0
        connected_body_at_chamber = (
            np.isfinite(main_body_front)
            & (main_body_front >= deep_x[-1] - 0.5 * probe_spacing)
            & (main_body_span >= minimum_body_span)
            & (deep_t >= 0.0)
        )
        arrival_time = first_sustained_time(
            deep_t,
            connected_body_at_chamber,
            minimum_duration=0.05,
            maximum_gap=0.025,
        )
    else:
        main_body_front = np.empty(0)
        main_body_span = np.empty(0)
        furthest_deep_gas = np.empty(0)
        minimum_body_span = None
        arrival_time = None
    probe_depth = metadata.get("main_body_probe_depth_below_crown_m")
    arrival_definition = (
        "first nonnegative paper time at which the longest contiguous "
        "alpha.air >= 0.50 component on the deep crown-probe line reaches "
        "x=-0.05 m for at least 0.05 s, while retaining a streamwise span of "
        f"at least {minimum_body_span if minimum_body_span is not None else 'unknown'} m; "
        "the line is "
        f"{probe_depth if probe_depth is not None else 'unknown'} m below the "
        "upstream-pipe crown; the probe is 4 mm below the selected initial "
        "thin-layer thickness. This is a line-sampled morphology proxy, not "
        "a three-dimensional component or source-identity tracker; a connected "
        "deep gas finger can still trigger it"
    )

    uv_t_solver, upstream_water = first_column(post, "upstreamWaterVolume")
    uv_t = uv_t_solver - offset
    upstream_zone_volume = function_region_volume(
        post, "upstreamWaterVolume", UPSTREAM_VOLUME
    )
    upstream_air_volume = (
        upstream_zone_volume - upstream_water if len(upstream_water) else np.empty(0)
    )
    um_t, upstream_mass = first_column(post, "upstreamMass")
    uwm_t, upstream_water_mass = first_column(post, "upstreamWaterMass")
    ugm_t, upstream_gas_mass_direct = first_column(post, "upstreamGasMass")
    if len(uv_t_solver) and len(ugm_t):
        upstream_air_mass = interp_series(ugm_t, upstream_gas_mass_direct, uv_t_solver)
    elif len(uv_t_solver):
        upstream_air_mass = interp_series(um_t, upstream_mass, uv_t_solver) - interp_series(
            uwm_t, upstream_water_mass, uv_t_solver
        )
    else:
        upstream_air_mass = np.empty(0)
    cv_t, chamber_water = first_column(post, "chamberWaterVolume")
    chamber_zone_volume = function_region_volume(
        post, "chamberWaterVolume", CHAMBER_VOLUME
    )
    chamber_air = chamber_zone_volume - interp_series(cv_t, chamber_water, uv_t_solver)

    gas_transfer_20pct = None
    transfer_baseline_time = None
    gas_transfer_definition = (
        "first nonnegative paper time with at least 20% upstream gas-mass loss "
        "and chamber gas-volume gain of at least max(1 L, 10% upstream volume), "
        "both relative to the first inventory sample at or after paper t=0; "
        "this is a substantial-transfer milestone, not gas-transfer onset or "
        "coherent main-pocket arrival"
    )
    if len(uv_t) and len(upstream_air_mass):
        baseline_candidates = np.where(uv_t >= -1e-9)[0]
        if len(baseline_candidates):
            baseline_index = int(baseline_candidates[0])
            transfer_baseline_time = float(uv_t[baseline_index])
            baseline_air = float(upstream_air_volume[baseline_index])
            baseline_air_mass = float(upstream_air_mass[baseline_index])
            baseline_chamber_air = float(chamber_air[baseline_index])
            chamber_gain_threshold = max(0.001, 0.10 * max(baseline_air, 0.0))
            candidates = np.where(
                (np.arange(len(uv_t)) >= baseline_index)
                & (upstream_air_mass <= 0.80 * baseline_air_mass)
                & (chamber_air - baseline_chamber_air >= chamber_gain_threshold)
            )[0]
            if len(candidates):
                gas_transfer_20pct = float(uv_t[candidates[0]])
    body_front_at_inventory_times = (
        interp_series(deep_t_solver, main_body_front, uv_t_solver)
        if len(deep_t_solver)
        else np.full(len(uv_t_solver), np.nan)
    )
    body_span_at_inventory_times = (
        interp_series(deep_t_solver, main_body_span, uv_t_solver)
        if len(deep_t_solver)
        else np.full(len(uv_t_solver), np.nan)
    )
    furthest_gas_at_inventory_times = (
        interp_series(deep_t_solver, furthest_deep_gas, uv_t_solver)
        if len(deep_t_solver)
        else np.full(len(uv_t_solver), np.nan)
    )
    write_csv(
        OUTPUTS / "openfoam_3d_air_pocket.csv",
        [
            "time_s",
            "upstream_air_volume_m3",
            "upstream_air_mass_kg",
            "chamber_air_volume_m3",
            "main_body_front_x_m",
            "main_body_component_span_m",
            "furthest_deep_gas_x_m",
        ],
        zip(
            uv_t,
            upstream_air_volume,
            upstream_air_mass,
            chamber_air,
            body_front_at_inventory_times,
            body_span_at_inventory_times,
            furthest_gas_at_inventory_times,
        ),
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
    geyser_rows = [row for row in event_rows if row[7] == "geyser"]

    # Total and gas conservation from volume and flux function objects.
    mass_t, total_mass = first_column(post, "totalMass")
    wm_t, water_mass = first_column(post, "waterMass")
    gm_t, gas_mass_direct = first_column(post, "gasMass")
    gas_mass = (
        interp_series(gm_t, gas_mass_direct, mass_t)
        if len(gm_t)
        else total_mass - interp_series(wm_t, water_mass, mass_t)
        if len(mass_t)
        else np.empty(0)
    )
    boundary_names = ("inletFlux", "gateFlux", "atmosphereFlux")
    water_flux_names = ("inletWaterMassFlux", "gateWaterMassFlux", "atmosphereWaterMassFlux")
    total_flux = np.zeros(len(mass_t))
    water_flux = np.zeros(len(mass_t))
    flux_complete = bool(len(mass_t))
    for name in boundary_names:
        ft, values, _ = parse_function(post, name)
        if not len(ft) or values.shape[1] < 2:
            flux_complete = False
            continue
        # Each boundary function writes (phi, rhoPhi); conservation requires
        # the second, mass-flux column.
        total_flux += interp_series(ft, values[:, 1], mass_t)
    for name in water_flux_names:
        ft, values = first_column(post, name)
        if not len(ft):
            flux_complete = False
            continue
        water_flux += interp_series(ft, values, mass_t)
    if len(mass_t) and flux_complete:
        # compressibleInterFoam constructs rhoPhi as the sum of phase mass
        # fluxes.  alphaPhi0.water is its conservative MULES water-volume
        # flux; density-weighting it gives the water contribution, leaving the
        # gas contribution by exact difference from rhoPhi.
        gas_flux = total_flux - water_flux
        total_residual = total_mass - total_mass[0] + cumulative_trapezoid(mass_t, total_flux)
        gas_residual = gas_mass - gas_mass[0] + cumulative_trapezoid(mass_t, gas_flux)
        mass_error = float(np.nanmax(np.abs(total_residual)) / max(abs(total_mass[0]), 1e-12))
        gas_residual_abs_max = float(np.nanmax(np.abs(gas_residual)))
        gas_error = float(gas_residual_abs_max / max(abs(gas_mass[0]), 1e-12))
        pocket_mass_reference = (
            float(upstream_air_mass[0]) if len(upstream_air_mass) else math.nan
        )
        gas_error_pocket_scale = (
            float(gas_residual_abs_max / abs(pocket_mass_reference))
            if np.isfinite(pocket_mass_reference)
            and abs(pocket_mass_reference) > 1e-12
            else None
        )
    else:
        mass_error = None
        gas_error = None
        gas_residual_abs_max = None
        gas_error_pocket_scale = None

    sim_end = float(np.nanmax(p_time)) if len(p_time) else None
    p1m = t_p1m = None
    if len(p_time) and sim_end is not None and sim_end > 0.0:
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
    if sim_end is not None and sim_end >= 6.5 and len(peaks) >= 3:
        period = float(np.median(np.diff([peak[0] for peak in peaks])))

    finals = [None, None, None]
    if sim_end is not None and sim_end >= 19.999:
        final_window = p_time >= 19.0
        finals = [float(np.nanmean(pressure[final_window, index])) for index in (1, 2, 3)]

    sim_start = float(np.nanmin(p_time)) if len(p_time) else None
    initialization_complete = (
        sim_start is not None
        and sim_start <= -0.24
        and sim_end is not None
        and sim_end >= 0.0
    )
    smoke_complete = sim_end is not None and sim_end >= 1.0
    phase1_complete = sim_end is not None and sim_end >= 6.5
    phase2_complete = sim_end is not None and sim_end >= 19.999
    initial_pt2 = None
    if initialization_complete and len(p_time):
        initial_index = int(np.nanargmin(np.abs(p_time)))
        if abs(float(p_time[initial_index])) <= 0.011:
            initial_pt2 = float(pressure[initial_index, 1])
    phase1_geysers = sum(
        1 for row in geyser_rows if float(row[1]) < PAPER["air_pocket_arrival_s"]
    )
    phase2_geysers = sum(
        1 for row in geyser_rows if arrival_time is not None and float(row[1]) >= arrival_time
    )
    pre_ramp_air_mass_change = None
    if len(uv_t) > 1 and np.isfinite(upstream_air_mass[0]):
        before_ramp = np.where(uv_t <= 0.0)[0]
        if len(before_ramp):
            pre_ramp_air_mass_change = float(
                (upstream_air_mass[before_ramp[-1]] - upstream_air_mass[0])
                / max(abs(upstream_air_mass[0]), 1e-12)
            )
    metrics = {
        "case": "Liu2020 C9 three-dimensional compressible VOF",
        "status": (
            "complete_phase2"
            if phase2_complete
            else "complete_phase1_only"
            if phase1_complete
            else "smoke_complete"
            if smoke_complete
            else "partial_smoke"
            if sim_end is not None and sim_end > 0.0
            else "initialization_only"
            if initialization_complete
            else "not_run"
        ),
        "solver": metadata.get("application", "unknown"),
        "mesh_generator": metadata.get("mesh_generator", "unknown"),
        "simulation_end_paper_time_s": sim_end,
        "paper_time_offset_s": offset,
        "initialization": {
            "window_complete": initialization_complete,
            "PT2_gauge_kPa": initial_pt2,
            "PT2_error_percent": relative_error(
                initial_pt2, PAPER["initial_PT2_kPa"]
            ),
        },
        "smoke": {
            "window_complete": smoke_complete,
        },
        "phase_1": {
            "window_complete": phase1_complete,
            "reproduced": bool(phase1_complete and phase1_geysers >= 2 and p1m is not None),
            "P1m_kPa": p1m,
            "P1m_time_s": t_p1m,
            "first_riser_top_s": first_top,
            "oscillation_period_s": period,
            "geyser_count_before_experimental_phase2_boundary": phase1_geysers,
        },
        "phase_2": {
            "window_complete": phase2_complete,
            "reproduced": bool(
                phase2_complete and len(geyser_rows) == 8 and phase2_geysers == 6
            ),
            "geysers_after_simulated_pocket_arrival": phase2_geysers,
            "note": "A phase-1 match is not counted as phase-2 reproduction.",
        },
        "simulated_rim_crossing_count": len(event_rows),
        "simulated_geyser_count": len(geyser_rows),
        "experimental_geyser_count": PAPER["geyser_count"],
        "simulated_air_pocket_arrival_s": arrival_time,
        "air_pocket_arrival_definition": arrival_definition,
        "air_pocket_arrival_proxy_plane_x_m": (
            float(deep_x[-1]) if deep_alpha.size else None
        ),
        "air_pocket_arrival_proxy_offset_from_chamber_m": (
            float(-deep_x[-1]) if deep_alpha.size else None
        ),
        "main_body_probe_depth_below_crown_m": probe_depth,
        "minimum_main_body_component_span_m": minimum_body_span,
        "simulated_gas_transfer_20pct_s": gas_transfer_20pct,
        "gas_transfer_20pct_definition": gas_transfer_definition,
        "gas_transfer_baseline_paper_time_s": transfer_baseline_time,
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
        "gas_mass_conservation_absolute_error_kg": gas_residual_abs_max,
        "gas_mass_conservation_error_per_initial_pocket_mass": gas_error_pocket_scale,
        "gas_mass_method": (
            "alpha.air-weighted thermo:rho.air inventory; boundary gas flux is "
            "rhoPhi minus density-weighted conservative alphaPhi0.water"
        ),
        "pre_ramp_upstream_air_mass_change_relative": pre_ramp_air_mass_change,
        "initial_air_volume_m3": float(upstream_air_volume[0]) if len(upstream_air_volume) else None,
        "initial_air_mass_kg": float(upstream_air_mass[0]) if len(upstream_air_mass) else None,
        "paper_time_zero_upstream_air_volume_m3": (
            float(upstream_air_volume[np.where(uv_t >= -1e-9)[0][0]])
            if len(uv_t) and len(np.where(uv_t >= -1e-9)[0])
            else None
        ),
        "paper_time_zero_upstream_air_mass_kg": (
            float(upstream_air_mass[np.where(uv_t >= -1e-9)[0][0]])
            if len(uv_t) and len(np.where(uv_t >= -1e-9)[0])
            else None
        ),
        "end_upstream_air_volume_m3": (
            float(upstream_air_volume[-1]) if len(upstream_air_volume) else None
        ),
        "end_upstream_air_mass_kg": (
            float(upstream_air_mass[-1]) if len(upstream_air_mass) else None
        ),
        "upstream_air_mass_retained_fraction": (
            float(upstream_air_mass[-1] / upstream_air_mass[0])
            if len(upstream_air_mass) and abs(upstream_air_mass[0]) > 1e-12
            else None
        ),
        "end_main_body_front_x_m": (
            float(main_body_front[-1])
            if len(main_body_front) and np.isfinite(main_body_front[-1])
            else None
        ),
        "end_main_body_component_span_m": (
            float(main_body_span[-1])
            if len(main_body_span) and np.isfinite(main_body_span[-1])
            else None
        ),
        "end_furthest_deep_gas_x_m": (
            float(furthest_deep_gas[-1])
            if len(furthest_deep_gas) and np.isfinite(furthest_deep_gas[-1])
            else None
        ),
        "mesh": parse_mesh_quality(case),
        "numerics": parse_numerics(
            case,
            offset,
            maximum_solver_time=sim_end + offset if sim_end is not None else None,
        ),
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

    figure, (left, front_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if len(uv_t):
        left.plot(uv_t, upstream_air_volume * 1000.0, color="#7c3aed", label="upstream air volume")
    left.set(ylabel="upstream air volume [L]", xlim=(-0.5, 20))
    left.grid(alpha=0.25)
    right = left.twinx()
    if len(uv_t):
        right.plot(uv_t, upstream_air_mass * 1000.0, color="#ea580c", label="upstream air mass")
    right.set_ylabel("upstream air mass [g]")
    left.axvline(PAPER["air_pocket_arrival_s"], color="black", ls="--", lw=0.8)
    handles = left.get_lines() + right.get_lines()
    left.legend(handles, [line.get_label() for line in handles], frameon=False, fontsize=8)
    if len(deep_t):
        front_axis.plot(deep_t, main_body_front, color="#0369a1", label="deep-air front")
    front_axis.axhline(0.0, color="black", ls="--", lw=0.8, label="chamber wall")
    front_axis.axvline(
        PAPER["air_pocket_arrival_s"],
        color="#a855f7",
        ls="--",
        lw=0.8,
        label="experiment 6.46 s",
    )
    front_axis.set(
        xlabel="paper time [s]",
        ylabel="main-body front x [m]",
        xlim=(-0.5, 20),
        ylim=(-5.8, 0.2),
    )
    front_axis.grid(alpha=0.25)
    front_axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUTS / "openfoam_3d_air_pocket_evolution.png", dpi=180)
    plt.close(figure)

    sensitivity = OUTPUTS / "openfoam_3d_mesh_sensitivity.csv"
    mesh = metrics["mesh"]
    write_csv(
        sensitivity,
        [
            "variant",
            "status",
            "solver",
            "interface_solver",
            "mesh_generator",
            "thin_layer_cell_size_m",
            "thin_layer_target_cells",
            "cells",
            "strict_check_passed",
            "maxCo",
            "maxDeltaT_s",
            "velocity_limit_m_s",
            "limiter_activated",
            "maximum_limited_cells",
            "maximum_limited_cell_percent",
            "pocket_profile",
            "gate_area_m2",
            "contact_angle_deg",
            "cAlpha",
            "air_Cp_J_kg_K",
            "water_bulk_modulus_Pa",
            "P1m_kPa",
            "first_top_s",
            "geyser_count",
            "air_arrival_s",
            "gas_transfer_20pct_s",
            "mass_error",
            "gas_mass_error",
        ],
        [
            [
                metadata.get("mesh_profile", "base"),
                metrics["status"],
                metadata.get("application"),
                metadata.get("interface_solver"),
                metadata.get("mesh_generator"),
                metadata.get("cartesian_thin_layer_cell_size_m"),
                metadata.get("thin_layer_target_cells"),
                mesh.get("cells"),
                mesh.get("all_geometry_passed"),
                metadata.get("maxCo"),
                metadata.get("maxDeltaT"),
                metadata.get("velocity_limit_m_s"),
                metrics["numerics"].get("velocity_limiter_activated"),
                metrics["numerics"].get("maximum_limited_cells"),
                metrics["numerics"].get("maximum_limited_cell_percent"),
                metadata.get("pocket_profile"),
                metadata.get("gate_area_m2"),
                metadata.get("contact_angle_deg"),
                metadata.get("interface_compression"),
                metadata.get("air_Cp_J_kg_K"),
                metadata.get("water_bulk_modulus_Pa"),
                p1m,
                first_top,
                len(geyser_rows),
                arrival_time,
                gas_transfer_20pct,
                mass_error,
                gas_error,
            ]
        ],
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
