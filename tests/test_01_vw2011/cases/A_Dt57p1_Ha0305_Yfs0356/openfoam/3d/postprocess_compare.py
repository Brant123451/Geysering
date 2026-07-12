"""Post-process the 3-D Case A probes and explicitly detect above-rim water."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_A = HERE.parents[1]
OUT = HERE / "outputs"
TOWER_RIM_Y = 0.657
PLUME_Y = np.arange(0.672, 1.853, 0.020)
WATER_MASS_PATTERN = re.compile(
    r"^CASEA_WATER_MASS_KG\s+"
    r"(?P<time>[-+0-9.eE]+)\s+(?P<mass>[-+0-9.eE]+)\s*$"
)


def load_common_postprocessor():
    source = HERE.parent / "2d" / "postprocess_compare.py"
    spec = importlib.util.spec_from_file_location("case_a_common_postprocess", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared postprocessor from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.CASE_A = CASE_A
    module.OUT = OUT
    return module


def parse_water_mass(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples: dict[float, float] = {}
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = WATER_MASS_PATTERN.match(line)
            if match:
                samples[float(match["time"])] = float(match["mass"])
    if not samples:
        raise RuntimeError(f"No CASEA_WATER_MASS_KG samples found in {log_path}")
    time = np.asarray(sorted(samples), dtype=float)
    mass = np.asarray([samples[value] for value in time], dtype=float)
    return time, mass


def parse_solver_provenance(log_path: Path) -> dict[str, object]:
    version = None
    ranks = None
    segments: list[tuple[float, float]] = []
    current_execution = None
    current_clock = None
    in_solver_segment = False
    completed_parallel_run = False
    lower_corrections = 0
    lower_limited_cells = 0
    minimum_unlimited_temperature = None
    upper_corrections = 0
    upper_limited_cells = 0
    maximum_unlimited_temperature = None

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            version_match = re.search(r"OPENFOAM=(\d+)\s+version=(\d+)", line)
            if version_match:
                version = version_match.group(2)
            rank_match = re.match(r"nProcs\s*:\s*(\d+)", line)
            if rank_match:
                ranks = int(rank_match.group(1))
            if line.startswith("Exec   :") and "compressibleInterFoam" in line:
                if in_solver_segment and current_execution is not None:
                    segments.append((current_execution, current_clock or 0.0))
                in_solver_segment = True
                current_execution = None
                current_clock = None
                continue
            timing_match = re.match(
                r"ExecutionTime = ([-+0-9.eE]+) s\s+"
                r"ClockTime = ([-+0-9.eE]+) s",
                line,
            )
            if in_solver_segment and timing_match:
                current_execution = float(timing_match.group(1))
                current_clock = float(timing_match.group(2))
            if in_solver_segment and line.strip() == "Finalising parallel run":
                if current_execution is not None:
                    segments.append((current_execution, current_clock or 0.0))
                completed_parallel_run = True
                in_solver_segment = False
                current_execution = None
                current_clock = None

            limiter_match = re.search(
                r"CASEA_TEMPERATURE_LIMITER lower_cells (\d+) "
                r"upper_cells (\d+) Tmin ([-+0-9.eE]+) "
                r"Tmax ([-+0-9.eE]+)",
                line,
            )
            if limiter_match:
                lower_limited = int(limiter_match.group(1))
                upper_limited = int(limiter_match.group(2))
                unlimited_minimum = float(limiter_match.group(3))
                unlimited_maximum = float(limiter_match.group(4))
                lower_corrections += int(lower_limited > 0)
                lower_limited_cells += lower_limited
                minimum_unlimited_temperature = (
                    unlimited_minimum
                    if minimum_unlimited_temperature is None
                    else min(minimum_unlimited_temperature, unlimited_minimum)
                )
                upper_corrections += int(upper_limited > 0)
                upper_limited_cells += upper_limited
                maximum_unlimited_temperature = (
                    unlimited_maximum
                    if maximum_unlimited_temperature is None
                    else max(maximum_unlimited_temperature, unlimited_maximum)
                )

    if in_solver_segment and current_execution is not None:
        segments.append((current_execution, current_clock or 0.0))

    return {
        "openfoam_version": version,
        "mpi_ranks": ranks,
        "solver_execution_time_s": float(sum(value[0] for value in segments)),
        "solver_clock_time_s": float(sum(value[1] for value in segments)),
        "solver_segments": len(segments),
        "completed_parallel_run": completed_parallel_run,
        "temperature_limiter": {
            "bounds_K": [250.0, 350.0],
            "lower_corrections_with_limiting": lower_corrections,
            "lower_total_cell_corrections": lower_limited_cells,
            "minimum_unlimited_temperature_K": minimum_unlimited_temperature,
            "upper_corrections_with_limiting": upper_corrections,
            "upper_total_cell_corrections": upper_limited_cells,
            "maximum_unlimited_temperature_K": maximum_unlimited_temperature,
        },
    }


def parse_mesh_provenance() -> dict[str, object]:
    check_text = (HERE / "log.checkMesh").read_text(
        encoding="utf-8", errors="replace"
    )
    strict_path = HERE / "log.checkMesh.strict"
    strict_text = (
        strict_path.read_text(encoding="utf-8", errors="replace")
        if strict_path.exists()
        else ""
    )
    gmsh_text = (HERE / "log.gmsh").read_text(encoding="utf-8", errors="replace")

    def match_float(pattern: str, text: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1).rstrip(".,;")) if match else None

    def match_int(pattern: str, text: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    gmsh_values: dict[str, str] = {}
    for line in gmsh_text.splitlines():
        if "=" in line and not line.startswith("Info"):
            key, value = line.split("=", 1)
            gmsh_values[key.strip()] = value.strip()

    mesh_volume = match_float(r"Total volume = ([-+0-9.eE]+)", check_text)
    cad_volume = (
        float(gmsh_values["fluid_volume_m3"])
        if "fluid_volume_m3" in gmsh_values
        else None
    )
    volume_error = (
        (mesh_volume - cad_volume) / cad_volume
        if mesh_volume is not None and cad_volume
        else None
    )

    return {
        "cells": match_int(r"cells:\s+(\d+)", check_text),
        "cell_type": "tetrahedra",
        "core_size_m": (
            float(gmsh_values["core_size_m"])
            if "core_size_m" in gmsh_values
            else None
        ),
        "plume_size_m": (
            float(gmsh_values["plume_size_m"])
            if "plume_size_m" in gmsh_values
            else None
        ),
        "gmsh_version": gmsh_values.get("gmsh_version"),
        "standard_checkMesh_ok": "Mesh OK." in check_text,
        "strict_checkMesh_ok": (
            "Mesh OK." in strict_text if strict_path.exists() else None
        ),
        "strict_failed_checks": (
            match_int(r"Failed\s+(\d+)\s+mesh checks", strict_text)
            if strict_path.exists()
            else None
        ),
        "max_non_orthogonality": match_float(
            r"Mesh non-orthogonality Max:\s*([-+0-9.eE]+)", check_text
        ),
        "mean_non_orthogonality": match_float(
            r"Mesh non-orthogonality Max:\s*[-+0-9.eE]+\s+average:\s*"
            r"([-+0-9.eE]+)",
            check_text,
        ),
        "max_skewness": match_float(r"Max skewness = ([-+0-9.eE]+)", check_text),
        "strict_underdetermined_cells": match_int(
            r"Cells with small determinant .* number of cells:\s*(\d+)",
            strict_text,
        )
        or 0,
        "cad_fluid_volume_m3": cad_volume,
        "openfoam_mesh_volume_m3": mesh_volume,
        "relative_mesh_volume_error": volume_error,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common_postprocessor()
    common.main()

    for suffix in (
        "series.csv",
        "levels.csv",
        "pressure_comparison.png",
        "pressure_comparison.pdf",
        "levels_comparison.png",
        "levels_comparison.pdf",
    ):
        (OUT / f"openfoam_2d_{suffix}").replace(OUT / f"openfoam_3d_{suffix}")

    plume = common.read_probe("plumeCentreline", "alpha.water")
    time = plume[:, 0]
    alpha = plume[:, 1:]
    if alpha.shape[1] != len(PLUME_Y):
        raise RuntimeError(
            f"Expected {len(PLUME_Y)} plume probes, found {alpha.shape[1]}"
        )

    highest_water_y = np.full(len(time), np.nan)
    for row, profile in enumerate(alpha):
        wet = np.where(profile >= 0.05)[0]
        if wet.size:
            highest_water_y[row] = PLUME_Y[wet[-1]]

    water_above_rim = bool(np.any(np.isfinite(highest_water_y)))
    if water_above_rim:
        max_height = float(np.nanmax(highest_water_y) - TOWER_RIM_Y)
    else:
        max_height = 0.0

    with (OUT / "openfoam_3d_plume.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "highest_water_y_m", "height_above_rim_m"])
        for sample_time, elevation in zip(time, highest_water_y):
            height = elevation - TOWER_RIM_Y if np.isfinite(elevation) else np.nan
            writer.writerow([sample_time, elevation, height])

    mass_time, water_mass = parse_water_mass(HERE / "log.compressibleInterFoam")
    reference_mass = float(water_mass[0])
    relative_mass_drift = (water_mass - reference_mass) / reference_mass
    with (OUT / "openfoam_3d_water_mass.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "water_mass_kg", "relative_drift"])
        writer.writerows(zip(mass_time, water_mass, relative_mass_drift))

    metrics_path = OUT / "openfoam_2d_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pressure_target = metrics["comparison_targets"]["pressure_plateau_Hstar"]
    free_surface_target = metrics["comparison_targets"]["free_surface_max_Ystar"]
    catch_target = metrics["comparison_targets"]["interface_catch_Tstar"]
    liftoff_repetitions = metrics["comparison_targets"][
        "interface_liftoff_Tstar_repetitions"
    ]
    liftoff = metrics["interface_liftoff_Tstar"]
    liftoff_range_error = max(
        min(liftoff_repetitions) - liftoff,
        0.0,
        liftoff - max(liftoff_repetitions),
    )
    metrics.update(
        {
            "geometry_model": "3-D circular pipe and circular tower",
            "circular_tower_to_pipe_area_ratio": (0.0571 / 0.094) ** 2,
            "plume_domain_height_above_rim_m": 1.2,
            "water_detected_above_rim": water_above_rim,
            "max_sampled_water_height_above_rim_m": max_height,
            "geysering": water_above_rim,
            "geyser_detection": (
                "alpha.water >= 0.05 on the external-atmosphere centreline; "
                "20 mm vertical sampling"
            ),
            "water_mass_initial_sample_kg": reference_mass,
            "water_mass_final_kg": float(water_mass[-1]),
            "water_mass_final_relative_drift": float(relative_mass_drift[-1]),
            "water_mass_max_abs_relative_drift": float(
                np.max(np.abs(relative_mass_drift))
            ),
            "water_mass_sampling": (
                "Integral of alpha.water*thermo:rho.water over all cells; "
                "first sample at the first 0.005 s write"
            ),
            "experimental_errors": {
                "pressure_plateau_Hstar_signed": (
                    metrics["pressure_plateau_Hstar_mean_T1to7"]
                    - pressure_target
                ),
                "pressure_plateau_percent_signed": 100.0
                * (
                    metrics["pressure_plateau_Hstar_mean_T1to7"]
                    - pressure_target
                )
                / pressure_target,
                "pressure_RMSE_Hstar_no_shift": metrics[
                    "pressure_RMSE_Hstar_no_shift"
                ],
                "free_surface_max_Ystar_signed": (
                    metrics["free_surface_max_Ystar"] - free_surface_target
                ),
                "free_surface_RMSE_Ystar_no_shift": metrics[
                    "free_surface_RMSE_Ystar_no_shift"
                ],
                "interface_RMSE_Ystar_no_shift": metrics[
                    "interface_RMSE_Ystar_no_shift"
                ],
                "interface_liftoff_distance_outside_repetition_range_Tstar": (
                    liftoff_range_error
                ),
                "interface_catch_signed_Tstar": (
                    metrics["interface_catch_Tstar"] - catch_target
                ),
            },
            "mesh": parse_mesh_provenance(),
            "solver": parse_solver_provenance(HERE / "log.compressibleInterFoam"),
            "caveat": (
                "Circular 3-D apparatus with an external atmosphere. No event-time "
                "shift or fitted wall/turbulence parameters were applied."
            ),
        }
    )
    metrics_path.replace(OUT / "openfoam_3d_metrics.json")
    metrics = json_safe(metrics)
    (OUT / "openfoam_3d_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
