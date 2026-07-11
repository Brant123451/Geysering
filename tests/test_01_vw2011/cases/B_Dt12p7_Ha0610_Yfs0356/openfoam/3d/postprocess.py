#!/usr/bin/env python3
"""Create compact Case-B CFD evidence and three-way comparison plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
OUTPUTS = CASE_ROOT / "outputs"
DIGITIZED = CASE_ROOT / "data" / "digitized"
RUNTIME = HERE / "outputs" / "runtime"

G = 9.81
DT = 0.0127
D = 0.094
L = 0.610
RHO_W = 998.2
P_ATM = 101325.0
TIME_SCALE = L / math.sqrt(G * DT)
CROWN_Y = D / 2.0
RIM_Y = CROWN_Y + L
TWOPHASEFLOW_COMMIT = "de9826f9ffb24f4b635ac97fd388ebd560cfc174"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FLOAT_RE = re.compile(
    rf"(?i)(?:{NUMBER_PATTERN}|nan)"
)


def configuration_id(manifest: dict) -> str:
    controls = {
        key: manifest.get(key)
        for key in (
            "stage",
            "mesh_preset",
            "valve_mode",
            "valve_open_time_s",
            "valve_seal_speed_m_per_s",
            "initial_air_head_m",
            "gas_equation_of_state",
            "max_co",
            "max_alpha_co",
            "max_capillary_num",
            "max_delta_t_s",
            "c_alpha",
            "solver",
            "two_phase_flow_commit",
            "advection_scheme",
            "reconstruction_scheme",
            "curvature_model",
            "n_alpha_bounds",
            "n_alpha_corr",
            "n_alpha_subcycles",
            "n_outer_correctors",
            "n_pressure_correctors",
            "n_non_orthogonal_correctors",
            "alpha_clip",
            "alpha_smooth_curvature_iterations",
            "end_time_s",
        )
    }
    digest = hashlib.sha256(
        json.dumps(controls, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"{manifest.get('mesh_preset', 'mesh')}-{digest}"


def is_baseline_numerics(manifest: dict) -> bool:
    return (
        manifest.get("solver") == "compressibleInterFlow"
        and manifest.get("two_phase_flow_commit") == TWOPHASEFLOW_COMMIT
        and manifest.get("advection_scheme") == "isoAdvection"
        and manifest.get("reconstruction_scheme") == "plicRDF"
        and manifest.get("curvature_model") == "RDF"
        and math.isclose(float(manifest.get("max_capillary_num", -1)), 1.0)
        and math.isclose(float(manifest.get("c_alpha", -1)), 1.0)
        and int(manifest.get("n_alpha_bounds", -1)) == 5
        and int(manifest.get("n_alpha_corr", -1)) == 1
        and int(manifest.get("n_alpha_subcycles", -1)) == 2
        and int(manifest.get("n_outer_correctors", -1)) == 1
        and int(manifest.get("n_pressure_correctors", -1)) == 2
        and int(manifest.get("n_non_orthogonal_correctors", -1)) == 0
        and manifest.get("alpha_clip") is False
    )


def is_baseline_full_physics(manifest: dict) -> bool:
    return (
        manifest.get("stage") == "full"
        and manifest.get("valve_mode") == "opening"
        and math.isclose(float(manifest.get("valve_open_time_s", -1)), 0.25)
        and math.isclose(
            float(manifest.get("valve_seal_speed_m_per_s", -1)), 1.0
        )
        and math.isclose(float(manifest.get("initial_air_head_m", -1)), 0.610)
        and manifest.get("gas_equation_of_state") == "perfectGas"
        and math.isclose(float(manifest.get("max_co", -1)), 0.30)
        and math.isclose(float(manifest.get("max_alpha_co", -1)), 0.20)
        and math.isclose(float(manifest.get("max_delta_t_s", -1)), 0.00025)
        and is_baseline_numerics(manifest)
        and float(manifest.get("end_time_s", 0)) >= 6 * TIME_SCALE
    )


def is_baseline_full(manifest: dict) -> bool:
    """Return whether this is the canonical base-mesh full run."""
    return (
        manifest.get("mesh_preset") == "base"
        and is_baseline_full_physics(manifest)
    )


def is_canonical_hold(manifest: dict) -> bool:
    """Return whether a hold belongs to the canonical acceptance deck."""
    return (
        manifest.get("stage") == "hold"
        and manifest.get("mesh_preset") == "base"
        and manifest.get("valve_mode") == "closed"
        and math.isclose(float(manifest.get("valve_open_time_s", -1)), 0.25)
        and math.isclose(
            float(manifest.get("valve_seal_speed_m_per_s", -1)), 1.0
        )
        and math.isclose(float(manifest.get("initial_air_head_m", -1)), 0.610)
        and manifest.get("gas_equation_of_state") == "perfectGas"
        and math.isclose(float(manifest.get("max_co", -1)), 0.30)
        and math.isclose(float(manifest.get("max_alpha_co", -1)), 0.20)
        and math.isclose(float(manifest.get("max_delta_t_s", -1)), 0.00025)
        and is_baseline_numerics(manifest)
    )


def should_update_hold_evidence(existing: dict, candidate: dict) -> bool:
    """Keep the strongest canonical hold instead of the most recent short run."""
    old = existing.get("hold_test", {})
    new = candidate.get("hold_test", {})
    if not old:
        return True
    old_passed = bool(old.get("passed"))
    new_passed = bool(new.get("passed"))
    if old_passed != new_passed:
        return new_passed
    old_duration = float(old.get("duration_s", -1))
    new_duration = float(new.get("duration_s", -1))
    return new_duration >= old_duration


def mesh_pair_compatible(first: dict, second: dict) -> bool:
    ignored = {"mesh_preset"}
    keys = {
        "stage",
        "valve_mode",
        "valve_open_time_s",
        "valve_seal_speed_m_per_s",
        "initial_air_head_m",
        "gas_equation_of_state",
        "max_co",
        "max_alpha_co",
        "max_capillary_num",
        "max_delta_t_s",
        "c_alpha",
        "solver",
        "two_phase_flow_commit",
        "advection_scheme",
        "reconstruction_scheme",
        "curvature_model",
        "n_alpha_bounds",
        "n_alpha_corr",
        "n_alpha_subcycles",
        "n_outer_correctors",
        "n_pressure_correctors",
        "n_non_orthogonal_correctors",
        "alpha_clip",
        "alpha_smooth_curvature_iterations",
        "end_time_s",
    } - ignored
    return all(first.get(key) == second.get(key) for key in keys)


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def first_crossing(x, y, threshold, above=True, after=-math.inf):
    for xi, yi in zip(x, y):
        if xi < after or not np.isfinite(yi):
            continue
        if (yi >= threshold) if above else (yi <= threshold):
            return float(xi)
    return None


def parse_mesh() -> dict:
    standard_text = (HERE / "log.checkMesh").read_text(errors="replace")
    strict_path = HERE / "log.checkMesh.strict"
    strict_text = (
        strict_path.read_text(errors="replace")
        if strict_path.is_file()
        else standard_text
    )

    def number(pattern, cast=float, source=strict_text):
        match = re.search(pattern, source, flags=re.I)
        return cast(match.group(1)) if match else None

    def passed(source: str) -> bool:
        return "Mesh OK." in source and not re.search(
            r"Failed\s+\d+\s+mesh checks", source, flags=re.I
        )

    standard_mesh_ok = passed(standard_text)
    strict_mesh_ok = passed(strict_text)
    strict_failed_checks = number(r"Failed\s+(\d+)\s+mesh checks", int)
    two_internal_face_cells = number(
        r"Writing\s+(\d+)\s+cells with two non-boundary faces", int
    )
    underdetermined_cells = number(
        r"Cells with small determinant[^:]*:\s*(\d+)", int
    )
    structural_excess = (
        underdetermined_cells - two_internal_face_cells
        if underdetermined_cells is not None
        and two_internal_face_cells is not None
        else None
    )
    strict_tet_boundary_exception = bool(
        standard_mesh_ok
        and not strict_mesh_ok
        and strict_failed_checks == 1
        and re.search(r"Cells with small determinant", strict_text, flags=re.I)
        and structural_excess is not None
        and 0 <= structural_excess <= 5
        and "Cell volumes OK." in strict_text
        and "Face pyramids OK." in strict_text
        and "Face tets OK." in strict_text
        and "Face interpolation weight check OK." in strict_text
        and "Face volume ratio check OK." in strict_text
        and not re.search(r"highly skew faces detected", strict_text, flags=re.I)
        and not re.search(r"Concave cells .* found", strict_text, flags=re.I)
    )
    mesh_ok = standard_mesh_ok and (
        strict_mesh_ok or strict_tet_boundary_exception
    )
    strict_audit_status = (
        "pass"
        if strict_mesh_ok
        else (
            "accepted_boundary_tet_exception"
            if strict_tet_boundary_exception
            else "fail"
        )
    )
    highly_skew_faces = number(r",\s*(\d+)\s+highly skew faces", int)
    if highly_skew_faces is None and re.search(
        r"Max skewness\s*=.*\bOK\.", strict_text, flags=re.I
    ):
        highly_skew_faces = 0
    low_weight_faces = number(
        r"Faces with small interpolation weight[^:]*:\s*(\d+)", int
    )
    if (
        low_weight_faces is None
        and "Face interpolation weight check OK." in strict_text
    ):
        low_weight_faces = 0
    low_volume_ratio_faces = number(
        r"Faces with small volume ratio[^:]*:\s*(\d+)", int
    )
    if (
        low_volume_ratio_faces is None
        and "Face volume ratio check OK." in strict_text
    ):
        low_volume_ratio_faces = 0
    return {
        "cells": number(r"\bcells:\s+(\d+)", int),
        "max_aspect_ratio": number(
            rf"Max aspect ratio\s*=\s*({NUMBER_PATTERN})"
        ),
        "max_non_orthogonality_deg": number(
            rf"Mesh non-orthogonality Max:\s*({NUMBER_PATTERN})"
        ),
        "severe_non_orthogonal_faces": number(
            r"severely non-orthogonal[^:]*faces:\s*(\d+)", int
        ),
        "max_skewness": number(
            rf"Max skewness\s*=\s*({NUMBER_PATTERN})"
        ),
        "highly_skew_faces": highly_skew_faces,
        "minimum_cell_volume_m3": number(
            rf"Min volume\s*=\s*({NUMBER_PATTERN})"
        ),
        "minimum_cell_determinant": number(
            rf"Cell determinant[^:]*:\s*minimum:\s*({NUMBER_PATTERN})"
        ),
        "underdetermined_cells": underdetermined_cells,
        "minimum_face_weight": number(
            rf"Face interpolation weight\s*:\s*minimum:\s*({NUMBER_PATTERN})"
        ),
        "low_weight_faces": low_weight_faces,
        "minimum_volume_ratio": number(
            rf"Face volume ratio\s*:\s*minimum:\s*({NUMBER_PATTERN})"
        ),
        "low_volume_ratio_faces": low_volume_ratio_faces,
        "two_internal_face_cells": two_internal_face_cells,
        "boundary_tet_determinant_excess_cells": structural_excess,
        "failed_checks": 0 if strict_mesh_ok else strict_failed_checks,
        "standard_mesh_ok": standard_mesh_ok,
        "strict_mesh_ok": strict_mesh_ok,
        "strict_tet_boundary_exception": strict_tet_boundary_exception,
        "strict_audit_status": strict_audit_status,
        "mesh_ok": mesh_ok,
    }


def update_mesh_csv(mesh: dict, manifest: dict, run_metrics: dict | None = None) -> None:
    path = OUTPUTS / "openfoam_3d_mesh_sensitivity.csv"
    fields = [
        "configuration_id",
        "mesh",
        "stage",
        "solver",
        "two_phase_flow_commit",
        "gas_eos",
        "initial_air_head_m",
        "valve_open_time_s",
        "gmsh_version",
        "algorithm",
        "optimizer",
        "box_transition_thickness_m",
        "cells",
        "nominal_cells_across_tower",
        "max_non_orthogonality_deg",
        "severe_non_orthogonal_faces",
        "max_skewness",
        "highly_skew_faces",
        "minimum_cell_volume_m3",
        "minimum_cell_determinant",
        "underdetermined_cells",
        "minimum_face_weight",
        "low_weight_faces",
        "minimum_volume_ratio",
        "low_volume_ratio_faces",
        "two_internal_face_cells",
        "boundary_tet_determinant_excess_cells",
        "failed_checks",
        "standard_mesh_ok",
        "strict_mesh_ok",
        "strict_tet_boundary_exception",
        "strict_audit_status",
        "mesh_ok",
        "maxCo",
        "maxAlphaCo",
        "maxCapillaryNum",
        "cAlpha",
        "advectionScheme",
        "reconstructionScheme",
        "curvatureModel",
        "nAlphaBounds",
        "nAlphaCorr",
        "nAlphaSubCycles",
        "nOuterCorrectors",
        "nCorrectors",
        "nNonOrthogonalCorrectors",
        "alphaClip",
        "alphaSmoothCurvature",
        "end_Tstar",
        "geyser",
        "Hstar_plateau",
        "gas_entry_Tstar",
        "free_surface_top_Tstar",
        "max_geyser_height_m",
    ]
    metadata = read_json(
        RUNTIME / f"mesh_{manifest.get('mesh_preset', 'base')}.json", {}
    )
    row = {
        "configuration_id": configuration_id(manifest),
        "mesh": manifest.get("mesh_preset", "base"),
        "stage": manifest.get("stage", "mesh"),
        "solver": manifest.get("solver"),
        "two_phase_flow_commit": manifest.get("two_phase_flow_commit"),
        "gas_eos": manifest.get("gas_equation_of_state"),
        "initial_air_head_m": manifest.get("initial_air_head_m"),
        "valve_open_time_s": manifest.get("valve_open_time_s"),
        "gmsh_version": metadata.get("gmsh_version"),
        "algorithm": metadata.get("algorithm"),
        "optimizer": metadata.get("optimizer"),
        "box_transition_thickness_m": metadata.get(
            "box_transition_thickness_m"
        ),
        "cells": mesh.get("cells"),
        "nominal_cells_across_tower": metadata.get(
            "nominal_cells_across_tower"
        ),
        "max_non_orthogonality_deg": mesh.get("max_non_orthogonality_deg"),
        "severe_non_orthogonal_faces": mesh.get(
            "severe_non_orthogonal_faces"
        ),
        "max_skewness": mesh.get("max_skewness"),
        "highly_skew_faces": mesh.get("highly_skew_faces"),
        "minimum_cell_volume_m3": mesh.get("minimum_cell_volume_m3"),
        "minimum_cell_determinant": mesh.get("minimum_cell_determinant"),
        "underdetermined_cells": mesh.get("underdetermined_cells"),
        "minimum_face_weight": mesh.get("minimum_face_weight"),
        "low_weight_faces": mesh.get("low_weight_faces"),
        "minimum_volume_ratio": mesh.get("minimum_volume_ratio"),
        "low_volume_ratio_faces": mesh.get("low_volume_ratio_faces"),
        "two_internal_face_cells": mesh.get("two_internal_face_cells"),
        "boundary_tet_determinant_excess_cells": mesh.get(
            "boundary_tet_determinant_excess_cells"
        ),
        "failed_checks": mesh.get("failed_checks"),
        "standard_mesh_ok": mesh.get("standard_mesh_ok"),
        "strict_mesh_ok": mesh.get("strict_mesh_ok"),
        "strict_tet_boundary_exception": mesh.get(
            "strict_tet_boundary_exception"
        ),
        "strict_audit_status": mesh.get("strict_audit_status"),
        "mesh_ok": mesh.get("mesh_ok"),
        "maxCo": manifest.get("max_co"),
        "maxAlphaCo": manifest.get("max_alpha_co"),
        "maxCapillaryNum": manifest.get("max_capillary_num"),
        "cAlpha": manifest.get("c_alpha"),
        "advectionScheme": manifest.get("advection_scheme"),
        "reconstructionScheme": manifest.get("reconstruction_scheme"),
        "curvatureModel": manifest.get("curvature_model"),
        "nAlphaBounds": manifest.get("n_alpha_bounds"),
        "nAlphaCorr": manifest.get("n_alpha_corr"),
        "nAlphaSubCycles": manifest.get("n_alpha_subcycles"),
        "nOuterCorrectors": manifest.get("n_outer_correctors"),
        "nCorrectors": manifest.get("n_pressure_correctors"),
        "nNonOrthogonalCorrectors": manifest.get(
            "n_non_orthogonal_correctors"
        ),
        "alphaClip": manifest.get("alpha_clip"),
        "alphaSmoothCurvature": manifest.get(
            "alpha_smooth_curvature_iterations"
        ),
        "end_Tstar": (run_metrics or {}).get("end_Tstar"),
        "geyser": (run_metrics or {}).get("geyser"),
        "Hstar_plateau": (run_metrics or {}).get("Hstar_plateau"),
        "gas_entry_Tstar": (run_metrics or {}).get("gas_entry_Tstar"),
        "free_surface_top_Tstar": (run_metrics or {}).get(
            "free_surface_top_Tstar"
        ),
        "max_geyser_height_m": (run_metrics or {}).get("max_geyser_height_m"),
    }
    existing = []
    if path.is_file():
        with path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    key = str(row["configuration_id"])
    existing = [
        old
        for old in existing
        if old.get("configuration_id") != key
    ]
    existing.append({name: row.get(name) for name in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)


def update_sensitivity_csv(metrics: dict) -> None:
    """Upsert one full-run row for cross-parameter sensitivity analysis."""
    manifest = metrics.get("run_configuration", {})
    if manifest.get("stage") != "full":
        return
    run = metrics.get("openfoam_3d", {})
    path = OUTPUTS / "openfoam_3d_sensitivity.csv"
    fields = [
        "configuration_id",
        "mesh",
        "solver",
        "two_phase_flow_commit",
        "gas_eos",
        "initial_air_head_m",
        "valve_mode",
        "valve_open_time_s",
        "valve_seal_speed_m_per_s",
        "maxCo",
        "maxAlphaCo",
        "maxCapillaryNum",
        "maxDeltaT_s",
        "cAlpha",
        "advectionScheme",
        "reconstructionScheme",
        "curvatureModel",
        "nAlphaBounds",
        "nAlphaCorr",
        "nAlphaSubCycles",
        "nOuterCorrectors",
        "nCorrectors",
        "nNonOrthogonalCorrectors",
        "alphaClip",
        "alphaSmoothCurvature",
        "requested_end_time_s",
        "end_Tstar",
        "completion_status",
        "geyser",
        "Hstar_plateau",
        "Yfs_star_plateau",
        "gas_entry_Tstar",
        "interface_0p85L_Tstar",
        "free_surface_top_Tstar",
        "pressure_drop_Tstar",
        "Vint_star",
        "Vfs_star",
        "max_geyser_height_m",
        "overflow_volume_m3",
        "liquid_mass_error_pct_max_abs",
        "gas_mass_error_pct_max_abs",
        "total_mass_error_pct_max_abs",
    ]
    row = {
        "configuration_id": configuration_id(manifest),
        "mesh": manifest.get("mesh_preset"),
        "solver": manifest.get("solver"),
        "two_phase_flow_commit": manifest.get("two_phase_flow_commit"),
        "gas_eos": manifest.get("gas_equation_of_state"),
        "initial_air_head_m": manifest.get("initial_air_head_m"),
        "valve_mode": manifest.get("valve_mode"),
        "valve_open_time_s": manifest.get("valve_open_time_s"),
        "valve_seal_speed_m_per_s": manifest.get(
            "valve_seal_speed_m_per_s"
        ),
        "maxCo": manifest.get("max_co"),
        "maxAlphaCo": manifest.get("max_alpha_co"),
        "maxCapillaryNum": manifest.get("max_capillary_num"),
        "maxDeltaT_s": manifest.get("max_delta_t_s"),
        "cAlpha": manifest.get("c_alpha"),
        "advectionScheme": manifest.get("advection_scheme"),
        "reconstructionScheme": manifest.get("reconstruction_scheme"),
        "curvatureModel": manifest.get("curvature_model"),
        "nAlphaBounds": manifest.get("n_alpha_bounds"),
        "nAlphaCorr": manifest.get("n_alpha_corr"),
        "nAlphaSubCycles": manifest.get("n_alpha_subcycles"),
        "nOuterCorrectors": manifest.get("n_outer_correctors"),
        "nCorrectors": manifest.get("n_pressure_correctors"),
        "nNonOrthogonalCorrectors": manifest.get(
            "n_non_orthogonal_correctors"
        ),
        "alphaClip": manifest.get("alpha_clip"),
        "alphaSmoothCurvature": manifest.get(
            "alpha_smooth_curvature_iterations"
        ),
        "requested_end_time_s": manifest.get("end_time_s"),
        **{name: run.get(name) for name in fields if name in run},
    }
    existing = []
    if path.is_file():
        with path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    key = str(row["configuration_id"])
    existing = [
        old for old in existing if old.get("configuration_id") != key
    ]
    existing.append({name: row.get(name) for name in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)


def numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = []
        for token in FLOAT_RE.findall(line):
            value = float(token)
            values.append(math.nan if abs(value) > 1e100 else value)
        if values:
            rows.append(values)
    return rows


def read_probe(function_name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    collected: dict[float, list[float]] = {}
    for path in sorted((HERE / "postProcessing" / function_name).glob(f"*/{field}")):
        for row in numeric_rows(path):
            if len(row) >= 2:
                collected[row[0]] = row[1:]
    if not collected:
        raise FileNotFoundError(f"No {function_name}/{field} probe data")
    times = np.asarray(sorted(collected))
    width = max(len(collected[time]) for time in times)
    values = np.full((len(times), width), np.nan)
    for index, time in enumerate(times):
        row = collected[time]
        values[index, : len(row)] = row
    return times, values


def interval_levels(profile: np.ndarray, heights: np.ndarray) -> tuple[float, float]:
    wet = np.isfinite(profile) & (profile >= 0.5)
    indices = np.flatnonzero(wet)
    if not len(indices):
        return math.nan, math.nan
    splits = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    principal = max(splits, key=len)
    dz = float(np.median(np.diff(heights)))
    lower = max(CROWN_Y, heights[principal[0]] - 0.5 * dz)
    upper = min(RIM_Y, heights[principal[-1]] + 0.5 * dz)
    yint = (lower - CROWN_Y) / L
    yfs = (upper - CROWN_Y) / L
    if yint < 0.02:
        yint = 0.0
    return yint, yfs


def extract_levels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, alpha = read_probe("towerProfiles", "alpha.water")
    n_heights = 121
    n_lines = 5
    if alpha.shape[1] != n_lines * n_heights:
        raise ValueError(
            f"Expected {n_lines*n_heights} tower probes, got {alpha.shape[1]}"
        )
    if np.any(~np.isfinite(alpha)):
        raise ValueError("One or more tower probes are outside the mesh")
    alpha = alpha.reshape(
        (len(times), n_lines, n_heights)
    )
    mean_profile = np.nanmean(alpha, axis=1)
    heights = np.linspace(0.052, 0.652, n_heights)
    levels = np.asarray(
        [interval_levels(profile, heights) for profile in mean_profile]
    )
    return times, levels[:, 0], levels[:, 1]


def parse_solver_diagnostics(text: str) -> dict:
    def values(pattern: str) -> list[float]:
        return [
            float(value)
            for value in re.findall(pattern, text, flags=re.I | re.M)
        ]

    courant = values(
        rf"^Courant Number mean:\s*{NUMBER_PATTERN}\s+max:\s*({NUMBER_PATTERN})"
    )
    interface_courant = values(
        rf"^Interface Courant Number mean:\s*{NUMBER_PATTERN}\s+max:\s*"
        rf"({NUMBER_PATTERN})"
    )
    capillary = values(rf"^Capillary Number:\s*({NUMBER_PATTERN})")
    delta_t = values(rf"^deltaT\s*=\s*({NUMBER_PATTERN})")
    alpha_min = values(
        rf"Min\(alpha\.water\)\s*=\s*({NUMBER_PATTERN})"
    )
    alpha_max = values(
        rf"Max\(alpha\.water\)\s*=\s*({NUMBER_PATTERN})(?!\s*-\s*1)"
    )
    alpha_max_minus_one = values(
        rf"Max\(alpha\.water\)\s*-\s*1\s*=\s*({NUMBER_PATTERN})"
    )
    alpha_max_all = alpha_max + [
        1.0 + overshoot for overshoot in alpha_max_minus_one
    ]
    execution_time = values(
        rf"^ExecutionTime\s*=\s*({NUMBER_PATTERN})\s+s"
    )
    clock_time = values(
        rf"^ExecutionTime\s*=\s*{NUMBER_PATTERN}\s+s\s+ClockTime\s*=\s*"
        rf"({NUMBER_PATTERN})\s+s"
    )

    velocity_samples: list[tuple[float, list[float]]] = []
    velocity_pattern = re.compile(
        rf"max\(mag\(U\)\)\s*=\s*({NUMBER_PATTERN}).*?"
        rf"at location\s*\(\s*({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+"
        rf"({NUMBER_PATTERN})\s*\)",
        flags=re.I,
    )
    for match in velocity_pattern.finditer(text):
        velocity_samples.append(
            (
                float(match.group(1)),
                [float(match.group(i)) for i in range(2, 5)],
            )
        )

    result = {
        "max_courant_number": max(courant) if courant else None,
        "max_interface_courant_number": (
            max(interface_courant) if interface_courant else None
        ),
        "max_capillary_number": max(capillary) if capillary else None,
        "minimum_alpha_water": min(alpha_min) if alpha_min else None,
        "maximum_alpha_water": max(alpha_max_all) if alpha_max_all else None,
        "final_delta_t_s": delta_t[-1] if delta_t else None,
        "execution_time_s": execution_time[-1] if execution_time else None,
        "clock_time_s": clock_time[-1] if clock_time else None,
    }
    if velocity_samples:
        maximum_velocity = max(velocity_samples, key=lambda sample: sample[0])
        result["max_velocity_m_per_s"] = maximum_velocity[0]
        result["max_velocity_location_m"] = maximum_velocity[1]
        result["last_written_velocity_max_m_per_s"] = velocity_samples[-1][0]
    else:
        result["max_velocity_m_per_s"] = None
        result["max_velocity_location_m"] = None
        result["last_written_velocity_max_m_per_s"] = None
    return result


def parse_accounting() -> dict[str, np.ndarray]:
    text = (HERE / "log.compressibleInterFlow").read_text(errors="replace")
    values: dict[float, list[float]] = {}
    for line in text.splitlines():
        if "CASEB_ACCOUNTING" not in line:
            continue
        tail = line.split("CASEB_ACCOUNTING", 1)[1]
        numbers = [float(value) for value in FLOAT_RE.findall(tail)]
        if len(numbers) >= 10:
            values[numbers[0]] = numbers[1:10]
    if not values:
        raise ValueError("No CASEB_ACCOUNTING records in solver log")
    time = np.asarray(sorted(values))
    data = np.asarray([values[t] for t in time])
    keys = [
        "liquid_mass",
        "gas_mass",
        "liquid_volume",
        "water_above_rim",
        "geyser_height",
        "liquid_flux",
        "gas_flux",
        "total_mass",
        "total_mass_flux",
    ]
    result = {"time": time}
    result.update({key: data[:, i] for i, key in enumerate(keys)})
    for phase in ("liquid", "gas"):
        flux = result[f"{phase}_flux"]
        cumulative = np.zeros_like(time)
        if len(time) > 1:
            cumulative[1:] = np.cumsum(
                0.5 * (flux[1:] + flux[:-1]) * np.diff(time)
            )
        initial = result[f"{phase}_mass"][0]
        result[f"{phase}_cumulative_out"] = cumulative
        result[f"{phase}_balance_error_pct"] = (
            result[f"{phase}_mass"] + cumulative - initial
        ) / initial * 100.0
    total_cumulative = np.zeros_like(time)
    if len(time) > 1:
        total_cumulative[1:] = np.cumsum(
            0.5
            * (result["total_mass_flux"][1:] + result["total_mass_flux"][:-1])
            * np.diff(time)
        )
    result["total_cumulative_out"] = total_cumulative
    result["total_balance_error_pct"] = (
        result["total_mass"] + total_cumulative - result["total_mass"][0]
    ) / result["total_mass"][0] * 100.0
    return result


def smooth(time: np.ndarray, values: np.ndarray, window_s: float = 0.05):
    result = np.full_like(values, np.nan)
    for i, centre in enumerate(time):
        mask = np.abs(time - centre) <= window_s / 2.0
        if np.any(np.isfinite(values[mask])):
            result[i] = np.nanmean(values[mask])
    return result


def slope(x, y, mask) -> float | None:
    finite = mask & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return None
    return float(np.polyfit(x[finite], y[finite], 1)[0])


def write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in zip(*columns):
            writer.writerow(
                ["" if not np.isfinite(value) else f"{value:.9g}" for value in row]
            )


def experiment_data():
    pressure = np.genfromtxt(
        DIGITIZED / "fig6_caseB_Hstar_band.csv", delimiter=",", names=True
    )
    levels = np.genfromtxt(
        DIGITIZED / "fig8_caseB_levels.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return pressure, levels


def frozen_1d():
    return np.genfromtxt(
        OUTPUTS / "caseB_model_series.csv", delimiter=",", names=True
    )


def percent_error(value, target):
    if value is None or not np.isfinite(value):
        return None
    return 100.0 * (value - target) / target


def compare_mesh_metrics(base: dict, refined: dict) -> dict:
    differences = {}
    for key in (
        "Hstar_plateau",
        "Yfs_star_plateau",
        "gas_entry_Tstar",
        "interface_0p85L_Tstar",
        "free_surface_top_Tstar",
        "pressure_drop_Tstar",
        "Vint_star",
        "Vfs_star",
        "max_geyser_height_m",
        "overflow_volume_m3",
    ):
        base_value = base.get("openfoam_3d", {}).get(key)
        refined_value = refined.get("openfoam_3d", {}).get(key)
        if (
            isinstance(base_value, (int, float))
            and isinstance(refined_value, (int, float))
            and np.isfinite(base_value)
            and np.isfinite(refined_value)
        ):
            differences[key] = {
                "base": base_value,
                "refined": refined_value,
                "refined_minus_base": refined_value - base_value,
                "relative_to_refined_percent": (
                    100.0 * (base_value - refined_value) / refined_value
                    if refined_value != 0
                    else None
                ),
            }
    return {
        "available": True,
        "compatible_non_mesh_controls": True,
        "base_cells": base.get("mesh", {}).get("cells"),
        "refined_cells": refined.get("mesh", {}).get("cells"),
        "differences": differences,
    }


def apply_acceptance(metrics: dict, hold_passed: bool) -> None:
    """Recompute acceptance after hold or mesh-pair evidence changes."""
    run = metrics["openfoam_3d"]
    conservation_passed = (
        run["liquid_mass_error_pct_max_abs"] <= 1.0
        and run["gas_mass_error_pct_max_abs"] <= 1.0
    )
    acceptance = {
        "all_streams_reach_Tstar_6": run["end_Tstar"] >= 6.0,
        "geyser_resolved": bool(run["geyser"]),
        "phase_balance_within_1_percent": conservation_passed,
        "plume_not_domain_censored": not bool(
            run["geyser_height_censored_by_domain"]
        ),
        "closed_valve_hold_passed": bool(hold_passed),
        "compatible_base_refined_pair": bool(
            metrics["mesh_sensitivity"].get("available")
        ),
    }
    run["acceptance"] = acceptance
    run["completion_status"] = (
        "complete"
        if is_baseline_full(metrics["run_configuration"])
        and all(acceptance.values())
        else "incomplete"
    )


def postprocess(manifest: dict, mesh: dict) -> dict:
    baseline_physics = is_baseline_full_physics(manifest)
    canonical_baseline = is_baseline_full(manifest)
    run_outputs = OUTPUTS if canonical_baseline else RUNTIME
    run_outputs.mkdir(parents=True, exist_ok=True)
    pressure_time, pressure_values = read_probe("transducer", "p")
    if pressure_values.shape[1] != 1 or np.any(~np.isfinite(pressure_values)):
        raise ValueError("Transducer probe must contain exactly one finite value")
    pressure_pa = pressure_values[:, 0]
    hstar = (pressure_pa - P_ATM) / (RHO_W * G * L)
    hstar_smooth = smooth(pressure_time, hstar)
    pressure_tstar = pressure_time / TIME_SCALE

    level_time, yint, yfs = extract_levels()
    level_tstar = level_time / TIME_SCALE
    accounting = parse_accounting()
    accounting_tstar = accounting["time"] / TIME_SCALE
    solver_text = (HERE / "log.compressibleInterFlow").read_text(
        errors="replace"
    )
    numerical_diagnostics = parse_solver_diagnostics(solver_text)

    exp_pressure, exp_levels = experiment_data()
    one_d = frozen_1d()
    exp_h = np.interp(
        pressure_tstar,
        exp_pressure["Tstar"],
        exp_pressure["Hstar_med"],
        left=np.nan,
        right=np.nan,
    )
    legacy_one_d_h = np.interp(
        pressure_tstar,
        one_d["Tstar"],
        one_d["transducer_Hstar"] - D / L,
        left=np.nan,
        right=np.nan,
    )
    write_csv(
        run_outputs / "openfoam_3d_pressure_series.csv",
        [
            "time_s",
            "Tstar",
            "Hstar_direct_gauge",
            "Hstar_smooth_0p05s",
            "experiment_Hstar_median",
            "frozen_1d_Hstar_legacy_crown",
        ],
        [
            pressure_time,
            pressure_tstar,
            hstar,
            hstar_smooth,
            exp_h,
            legacy_one_d_h,
        ],
    )
    write_csv(
        run_outputs / "openfoam_3d_levels_series.csv",
        ["time_s", "Tstar", "Yfs_star", "Yint_star"],
        [level_time, level_tstar, yfs, yint],
    )
    vint_inst = (
        np.gradient(yint, level_tstar, edge_order=1)
        if len(level_tstar) >= 2
        else np.full_like(yint, np.nan)
    )
    write_csv(
        run_outputs / "openfoam_3d_gas_interface_series.csv",
        ["time_s", "Tstar", "Yint_star", "Vint_star_local"],
        [level_time, level_tstar, yint, vint_inst],
    )

    gas_entry = first_crossing(level_tstar, yint, 0.06)
    interface_080 = first_crossing(level_tstar, yint, 0.80)
    interface_085 = first_crossing(level_tstar, yint, 0.85)
    surface_top = first_crossing(level_tstar, yfs, 0.98)
    pressure_drop = first_crossing(
        pressure_tstar, hstar_smooth, 0.30, above=False, after=2.0
    )
    plateau_mask = (
        (pressure_tstar >= 1.0)
        & (pressure_tstar < 4.0)
        & np.isfinite(hstar_smooth)
    )
    h_plateau = (
        float(np.nanmedian(hstar_smooth[plateau_mask]))
        if np.any(plateau_mask)
        else None
    )
    before_entry = level_tstar < (gas_entry if gas_entry is not None else 3.65)
    fs_plateau_mask = before_entry & (level_tstar >= 3.0)
    yfs_plateau = (
        float(np.nanmedian(yfs[fs_plateau_mask]))
        if np.any(fs_plateau_mask)
        else None
    )
    vint = slope(
        level_tstar,
        yint,
        (yint >= 0.1)
        & (yint <= 0.8)
        & (level_tstar >= (gas_entry or 0.0))
        & (level_tstar <= (interface_080 or level_tstar[-1])),
    )
    vfs = slope(
        level_tstar,
        yfs,
        (level_tstar >= (gas_entry or 3.65))
        & (level_tstar <= (surface_top or level_tstar[-1]))
        & (yfs >= 0.82)
        & (yfs <= 0.98),
    )
    max_above = float(np.nanmax(accounting["water_above_rim"]))
    max_height = float(np.nanmax(accounting["geyser_height"]))
    ejected_threshold = math.pi * (DT / 2) ** 2 * 0.001
    sustained_ejection = (
        np.count_nonzero(accounting["water_above_rim"] >= ejected_threshold) >= 3
    )
    interface_at_surface_top = (
        float(np.interp(surface_top, level_tstar, yint))
        if surface_top is not None
        else None
    )
    event_order_ok = (
        surface_top is not None
        and interface_at_surface_top is not None
        and interface_at_surface_top < 0.96
    )
    geyser = bool(sustained_ejection and event_order_ok)
    stream_end_tstars = {
        "pressure": float(pressure_tstar[-1]),
        "levels": float(level_tstar[-1]),
        "accounting": float(accounting_tstar[-1]),
    }
    end_tstar = min(stream_end_tstars.values())
    plume_censored = max_height >= 1.15

    run = {
        "completion_status": "incomplete",
        "end_Tstar": end_tstar,
        "stream_end_Tstar": stream_end_tstars,
        "geyser": geyser,
        "geyser_criterion": {
            "free_surface_before_gas_breakthrough": event_order_ok,
            "Yint_star_at_free_surface_top": interface_at_surface_top,
            "minimum_sustained_above_rim_volume_m3": ejected_threshold,
            "samples_above_threshold": int(
                np.count_nonzero(
                    accounting["water_above_rim"] >= ejected_threshold
                )
            ),
        },
        "gas_entry_Tstar": gas_entry,
        "interface_0p85L_Tstar": interface_085,
        "free_surface_top_Tstar": surface_top,
        "pressure_drop_Tstar": pressure_drop,
        "Hstar_plateau": h_plateau,
        "Yfs_star_plateau": yfs_plateau,
        "Vint_star": vint,
        "Vfs_star": vfs,
        "max_geyser_height_m": max_height,
        "geyser_height_censored_by_domain": plume_censored,
        "overflow_volume_m3": max_above,
        "overflow_volume_method": "maximum liquid inventory above physical rim",
        "numerical_diagnostics": numerical_diagnostics,
        "liquid_mass_error_pct_final": float(
            accounting["liquid_balance_error_pct"][-1]
        ),
        "liquid_mass_error_pct_max_abs": float(
            np.nanmax(np.abs(accounting["liquid_balance_error_pct"]))
        ),
        "gas_mass_error_pct_final": float(accounting["gas_balance_error_pct"][-1]),
        "gas_mass_error_pct_max_abs": float(
            np.nanmax(np.abs(accounting["gas_balance_error_pct"]))
        ),
        "total_mass_error_pct_final": float(
            accounting["total_balance_error_pct"][-1]
        ),
        "total_mass_error_pct_max_abs": float(
            np.nanmax(np.abs(accounting["total_balance_error_pct"]))
        ),
        "mass_balance_method": {
            "total": "solver rho and rhoPhi (signed atmosphere flux)",
            "phases": (
                "registered thermo:rho phase densities and geometric "
                "alphaPhi.water boundary flux; transported alpha is not clipped"
            ),
        },
    }
    targets = {
        "geyser": True,
        "Hstar_plateau": 0.7575566751,
        "Hstar_plateau_window_Tstar": [1.0, 4.0],
        "Yfs_star_plateau": 0.8225,
        "gas_entry_Tstar_range": [3.648, 3.742],
        "free_surface_top_Tstar_range": [3.900, 3.981],
        "pressure_drop_Tstar": 4.048,
        "interface_0p85L_Tstar_legacy_csv": 3.846875,
        "interface_0p85L_Tstar_audited_range": [4.091, 4.169],
        "Vint_star_table2_diameter_average": 1.43,
        "Vfs_star_table2_diameter_average": 0.44,
    }
    errors = {
        "Hstar_plateau_percent": percent_error(h_plateau, targets["Hstar_plateau"]),
        "Yfs_star_plateau_percent": percent_error(
            yfs_plateau, targets["Yfs_star_plateau"]
        ),
        "pressure_drop_Tstar_percent": percent_error(
            pressure_drop, targets["pressure_drop_Tstar"]
        ),
        "Vint_star_percent": percent_error(
            vint, targets["Vint_star_table2_diameter_average"]
        ),
        "Vfs_star_percent": percent_error(
            vfs, targets["Vfs_star_table2_diameter_average"]
        ),
    }

    # Pressure: experiment band, frozen 1-D, and direct 3-D gauge head.
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.fill_between(
        exp_pressure["Tstar"],
        exp_pressure["Hstar_min"],
        exp_pressure["Hstar_max"],
        color="0.82",
        label="VW Fig.6 digitized repeat envelope",
    )
    ax.plot(
        exp_pressure["Tstar"],
        exp_pressure["Hstar_med"],
        color="0.2",
        lw=1.2,
        label="VW Fig.6 raster median",
    )
    ax.plot(
        one_d["Tstar"],
        one_d["transducer_Hstar"] - D / L,
        "--",
        color="#d97706",
        label="frozen 1-D (legacy −D/L datum)",
    )
    ax.plot(
        pressure_tstar,
        hstar_smooth,
        color="#c62828",
        lw=1.8,
        label="OpenFOAM 3-D (direct gauge head)",
    )
    ax.set(xlim=(0, 6), ylim=(0, 1.5), xlabel="$T^*$", ylabel="$H^*$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_outputs / "openfoam_3d_pressure_comparison.png", dpi=180)
    plt.close(fig)

    fs_exp = exp_levels[exp_levels["kind"] == "fs"]
    int_exp = exp_levels[exp_levels["kind"] == "int"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(
        fs_exp["Tstar"], fs_exp["Ystar"], "^", mfc="none", color="#1f77b4",
        label="VW Fig.8 free surface",
    )
    ax.plot(
        int_exp["Tstar"], int_exp["Ystar"], "o", mfc="none", color="0.15",
        label="VW Fig.8 gas interface",
    )
    ax.plot(one_d["Tstar"], one_d["Yfs_star"], "--", color="#d97706",
            label="frozen 1-D free surface")
    ax.plot(one_d["Tstar"], one_d["Yint_star"], ":", color="#d97706",
            label="frozen 1-D gas interface")
    ax.plot(level_tstar, yfs, color="#1565c0", lw=1.8,
            label="OpenFOAM 3-D free surface")
    ax.plot(level_tstar, yint, color="#c62828", lw=1.8,
            label="OpenFOAM 3-D gas interface")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set(xlim=(0, 6), ylim=(0, 1.08), xlabel="$T^*$", ylabel="$Y^*$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(run_outputs / "openfoam_3d_levels_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(
        accounting_tstar,
        accounting["geyser_height"],
        color="#c62828",
        lw=1.8,
        label="OpenFOAM 3-D water above rim (αw≥0.05)",
    )
    ax.axhline(0, color="0.25", lw=0.8, label="physical rim / 1-D cap")
    ax.text(
        0.02,
        0.95,
        "VW Fig.8 confirms spilling but does not report external height",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.set(xlabel="$T^*$", ylabel="maximum height above rim [m]", xlim=(0, 6))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_outputs / "openfoam_3d_geyser_height.png", dpi=180)
    plt.close(fig)

    metrics = {
        "case": {
            "tower_diameter_m": DT,
            "main_pipe_diameter_m": D,
            "tower_height_m": L,
            "initial_air_head_m": manifest.get("initial_air_head_m"),
            "initial_tower_level_m": 0.356,
            "time_scale_s": TIME_SCALE,
        },
        "solver": {
            "name": manifest.get("solver", "compressibleInterFlow"),
            "openfoam_version": "v2512",
            "two_phase_flow_commit": manifest.get("two_phase_flow_commit"),
            "advection_scheme": manifest.get("advection_scheme"),
            "reconstruction_scheme": manifest.get("reconstruction_scheme"),
            "curvature_model": manifest.get("curvature_model"),
            "gas_equation_of_state": manifest.get(
                "gas_equation_of_state", "unknown"
            ),
            "water_equation_of_state": (
                "perfectFluid rho=998.153943+p/(7504.690432*T)"
            ),
        },
        "run_configuration": manifest,
        "mesh": mesh,
        "experiment": targets,
        "openfoam_3d": run,
        "experimental_error": errors,
        "audit_notes": [
            "Table 2 velocities are diameter-level averages, not Case-B-only values.",
            "The legacy T*=3.846875 interface marker is probably misclassified; see PAPER_AUDIT.md.",
            "Pressure tap vertical datum is unresolved; 3-D uses direct gauge pressure without −D/L correction.",
        ],
    }
    preset = manifest.get("mesh_preset", "base")
    other_preset = "refined" if preset == "base" else "base"
    other_path = OUTPUTS / f"openfoam_3d_metrics_{other_preset}.json"
    paired_other = None
    if baseline_physics and other_path.is_file():
        other = read_json(other_path, {})
        compatible = mesh_pair_compatible(
            manifest, other.get("run_configuration", {})
        )
        if compatible:
            base = metrics if preset == "base" else other
            refined = metrics if preset == "refined" else other
            mesh_comparison = compare_mesh_metrics(base, refined)
            metrics["mesh_sensitivity"] = mesh_comparison
            other["mesh_sensitivity"] = mesh_comparison
            paired_other = other
        else:
            metrics["mesh_sensitivity"] = {
                "available": False,
                "compatible_non_mesh_controls": False,
                "reason": "The candidate mesh pair uses different non-mesh controls",
            }
    else:
        metrics["mesh_sensitivity"] = {
            "available": False,
            "reason": (
                f"No compatible baseline full {other_preset} metrics file is available"
                if baseline_physics
                else "This run is not the declared baseline configuration"
            ),
        }

    hold_path = OUTPUTS / "openfoam_3d_hold_metrics.json"
    existing_hold = read_json(hold_path, {})
    write_canonical_hold = False
    if manifest.get("stage") == "hold":
        initial_yfs = 0.356 / L
        hold = {
            "duration_s": float(
                min(
                    pressure_time[-1],
                    level_time[-1],
                    accounting["time"][-1],
                )
            ),
            "requested_duration_s": float(manifest.get("end_time_s", 0)),
            "Hstar_peak_to_peak": float(np.nanmax(hstar) - np.nanmin(hstar)),
            "Yfs_star_max_drift": float(np.nanmax(np.abs(yfs - initial_yfs))),
            "max_water_above_rim_m3": max_above,
            "gas_entry_detected": gas_entry is not None,
            "liquid_mass_error_pct_max_abs": run[
                "liquid_mass_error_pct_max_abs"
            ],
            "gas_mass_error_pct_max_abs": run["gas_mass_error_pct_max_abs"],
        }
        hold["passed"] = bool(
            hold["duration_s"] >= 0.9
            and hold["Hstar_peak_to_peak"] <= 0.02
            and hold["Yfs_star_max_drift"] <= 0.02
            and hold["max_water_above_rim_m3"] < 1e-10
            and not hold["gas_entry_detected"]
            and hold["liquid_mass_error_pct_max_abs"] <= 1.0
            and hold["gas_mass_error_pct_max_abs"] <= 1.0
        )
        metrics["hold_test"] = hold
        write_canonical_hold = (
            is_canonical_hold(manifest)
            and should_update_hold_evidence(existing_hold, metrics)
        )

    hold_passed = bool(
        (
            metrics
            if write_canonical_hold
            else existing_hold
        ).get("hold_test", {}).get("passed")
    )
    apply_acceptance(metrics, hold_passed)
    if paired_other is not None:
        apply_acceptance(paired_other, hold_passed)

    RUNTIME.mkdir(parents=True, exist_ok=True)
    config_id = configuration_id(manifest)
    snapshot = RUNTIME / f"metrics_{config_id}.json"
    snapshot.write_text(json.dumps(metrics, indent=2) + "\n")
    if manifest.get("stage") == "full":
        (OUTPUTS / f"openfoam_3d_metrics_{config_id}.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
    if baseline_physics:
        (OUTPUTS / f"openfoam_3d_metrics_{preset}.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
    if paired_other is not None:
        other_manifest = paired_other.get("run_configuration", {})
        other_config_id = configuration_id(other_manifest)
        other_path.write_text(json.dumps(paired_other, indent=2) + "\n")
        (OUTPUTS / f"openfoam_3d_metrics_{other_config_id}.json").write_text(
            json.dumps(paired_other, indent=2) + "\n"
        )
    canonical_metrics = (
        metrics
        if canonical_baseline
        else (
            paired_other
            if paired_other is not None
            and paired_other.get("run_configuration", {}).get("mesh_preset")
            == "base"
            else None
        )
    )
    if canonical_metrics is not None:
        (OUTPUTS / "openfoam_3d_metrics.json").write_text(
            json.dumps(canonical_metrics, indent=2) + "\n"
        )
    if write_canonical_hold:
        hold_path.write_text(json.dumps(metrics, indent=2) + "\n")
    update_mesh_csv(mesh, manifest, run)
    update_sensitivity_csv(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-only", action="store_true")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = read_json(RUNTIME / "run_manifest.json", {"stage": "mesh"})
    mesh = parse_mesh()
    if not mesh["mesh_ok"]:
        raise SystemExit("mesh did not meet the documented acceptance criteria")
    if args.mesh_only:
        update_mesh_csv(mesh, manifest)
        print(json.dumps({"mesh": mesh, "run_configuration": manifest}, indent=2))
        return
    metrics = postprocess(manifest, mesh)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
