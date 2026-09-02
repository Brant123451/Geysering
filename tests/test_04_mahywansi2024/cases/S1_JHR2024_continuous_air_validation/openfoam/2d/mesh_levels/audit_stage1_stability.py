#!/usr/bin/env python3
"""Read-only Stage-1 physical-stability audit for the three Case-3 meshes.

The numerical gates are loaded from STAGE1_STABILITY_GATE.json.  This script
does not run OpenFOAM and never creates, removes, or edits STAGE1_* markers.
Its only project output is one stage1_stability_audit.json per selected level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


BASE = Path(__file__).resolve().parent
GATE_PATH = BASE / "STAGE1_STABILITY_GATE.json"
LEVELS = {
    "coarse": {
        "case": BASE / "coarse",
        "launcher": BASE / "coarse" / "run_stage1_segment.sh",
        "template": BASE / "coarse" / "system" / "controlDict.stage1-smoke",
    },
    "medium_refine": {
        "case": BASE / "medium_refine" / "case",
        "launcher": BASE / "medium_refine" / "case" / "run_pipeline.sh",
        "template": BASE / "medium_refine" / "case" / "system" / "controlDict.stage1",
    },
    "refined": {
        "case": BASE / "refined" / "case",
        "launcher": BASE / "refined" / "case" / "run_pipeline.sh",
        "template": BASE / "refined" / "case" / "system" / "controlDict.stage1",
    },
}

FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FLOAT_RE = re.compile(FLOAT_TOKEN)
TIME_NAME_RE = re.compile(rf"^{FLOAT_TOKEN}$")


class AuditInputError(RuntimeError):
    """A required audit input is absent, malformed, or internally inconsistent."""


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise AuditInputError(f"{label} is not finite: {value!r}")
    return value


def _as_float(text: str, label: str) -> float:
    try:
        return _finite(float(text), label)
    except ValueError as exc:
        raise AuditInputError(f"cannot parse {label}: {text!r}") from exc


def _numeric_name(path: Path) -> float | None:
    if not path.is_dir() or not TIME_NAME_RE.fullmatch(path.name):
        return None
    try:
        return _finite(float(path.name), f"time directory {path.name}")
    except AuditInputError:
        return None


def _extract_braced(text: str, opening_brace: int) -> str:
    if opening_brace < 0 or opening_brace >= len(text) or text[opening_brace] != "{":
        raise AuditInputError("invalid opening brace location")
    depth = 0
    for index in range(opening_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index]
    raise AuditInputError("unterminated dictionary block")


def _extract_named_block(text: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", text)
    if not match:
        raise AuditInputError(f"dictionary block {name!r} is missing")
    opening = text.find("{", match.start())
    return _extract_braced(text, opening)


def _parse_scalar_list_from_patch(block: str, patch: str) -> list[float]:
    uniform = re.search(rf"\bvalue\s+uniform\s+({FLOAT_TOKEN})\s*;", block)
    if uniform:
        return [_as_float(uniform.group(1), f"{patch} uniform value")]

    nonuniform = re.search(
        r"\bvalue\s+nonuniform\s+List\s*<\s*scalar\s*>\s+"
        r"(\d+)\s*\((.*?)\)\s*;",
        block,
        flags=re.DOTALL,
    )
    if not nonuniform:
        raise AuditInputError(f"{patch}: scalar boundary value is missing or unsupported")
    expected = int(nonuniform.group(1))
    values = [
        _as_float(token, f"{patch} nonuniform value")
        for token in FLOAT_RE.findall(nonuniform.group(2))
    ]
    if len(values) != expected:
        raise AuditInputError(
            f"{patch}: declared {expected} boundary values but parsed {len(values)}"
        )
    return values


def read_scalar_patch_values(
    path: Path, wanted_patches: Sequence[str]
) -> tuple[tuple[int, ...], dict[str, list[float]]]:
    """Stream an ASCII OpenFOAM field and return selected scalar patch values.

    The water patches occur before the potentially large wall patch.  Streaming
    avoids retaining the internal field or unrelated boundary values in memory.
    """

    dimensions: tuple[int, ...] | None = None
    found_boundary = False
    waiting_boundary_brace = False
    pending_patch: str | None = None
    active_patch: str | None = None
    active_lines: list[str] = []
    active_depth = 0
    active_opened = False
    blocks: dict[str, str] = {}
    wanted = set(wanted_patches)

    try:
        handle = path.open("r", encoding="utf-8", errors="strict")
    except OSError as exc:
        raise AuditInputError(f"cannot read {path}: {exc}") from exc

    with handle:
        for raw_line in handle:
            line = re.sub(r"//.*$", "", raw_line)
            if dimensions is None:
                match = re.search(r"\bdimensions\s*\[([^]]+)\]\s*;", line)
                if match:
                    tokens = FLOAT_RE.findall(match.group(1))
                    numeric = [_as_float(token, f"{path.name} dimension") for token in tokens]
                    if any(abs(value - round(value)) > 1e-12 for value in numeric):
                        raise AuditInputError(f"{path}: non-integral dimensions {numeric}")
                    dimensions = tuple(int(round(value)) for value in numeric)

            if not found_boundary:
                if re.search(r"\bboundaryField\b", line):
                    waiting_boundary_brace = True
                    if "{" in line.split("boundaryField", 1)[1]:
                        found_boundary = True
                        waiting_boundary_brace = False
                elif waiting_boundary_brace and "{" in line:
                    found_boundary = True
                    waiting_boundary_brace = False
                continue

            if active_patch is not None:
                active_lines.append(line)
                active_depth += line.count("{") - line.count("}")
                active_opened = active_opened or "{" in line
                if active_opened and active_depth == 0:
                    blocks[active_patch] = "".join(active_lines)
                    active_patch = None
                    active_lines = []
                    active_opened = False
                    if wanted.issubset(blocks):
                        break
                continue

            stripped = line.strip()
            if pending_patch is not None:
                if not stripped:
                    continue
                if "{" not in line:
                    raise AuditInputError(f"{path}: expected opening brace for {pending_patch}")
                active_patch = pending_patch
                pending_patch = None
                active_lines = [line]
                active_depth = line.count("{") - line.count("}")
                active_opened = True
                if active_depth == 0:
                    blocks[active_patch] = "".join(active_lines)
                    active_patch = None
                    active_lines = []
                    active_opened = False
                continue

            candidate = stripped.split()[0] if stripped else ""
            if candidate in wanted:
                tail = stripped[len(candidate) :].strip()
                if tail.startswith("{"):
                    active_patch = candidate
                    active_lines = [line]
                    active_depth = line.count("{") - line.count("}")
                    active_opened = True
                    if active_depth == 0:
                        blocks[active_patch] = "".join(active_lines)
                        active_patch = None
                        active_lines = []
                        active_opened = False
                else:
                    pending_patch = candidate

    if dimensions is None:
        raise AuditInputError(f"{path}: dimensions entry is missing")
    missing = sorted(wanted - set(blocks))
    if missing:
        raise AuditInputError(f"{path}: missing boundary patches {missing}")
    values = {name: _parse_scalar_list_from_patch(blocks[name], name) for name in wanted}
    return dimensions, values


def parse_flux_snapshot(time_value: float, directory: Path) -> dict[str, float]:
    phi_dimensions, phi = read_scalar_patch_values(
        directory / "phi", ("waterInlet", "waterOutlet")
    )
    rho_dimensions, rho = read_scalar_patch_values(
        directory / "rho", ("waterInlet", "waterOutlet")
    )
    if phi_dimensions != (0, 3, -1, 0, 0, 0, 0):
        raise AuditInputError(
            f"{directory / 'phi'}: expected volume-flux dimensions, got {phi_dimensions}"
        )
    if rho_dimensions != (1, -3, 0, 0, 0, 0, 0):
        raise AuditInputError(
            f"{directory / 'rho'}: expected density dimensions, got {rho_dimensions}"
        )

    def expand_density(patch: str) -> list[float]:
        flux_values = phi[patch]
        density_values = rho[patch]
        if len(density_values) == 1:
            return density_values * len(flux_values)
        if len(density_values) != len(flux_values):
            raise AuditInputError(
                f"{directory}: {patch} rho/phi face-count mismatch "
                f"({len(density_values)} != {len(flux_values)})"
            )
        return density_values

    inlet_rho = expand_density("waterInlet")
    outlet_rho = expand_density("waterOutlet")
    inlet_phi = phi["waterInlet"]
    outlet_phi = phi["waterOutlet"]
    return {
        "time_s": time_value,
        "qin_m3_per_s": -sum(inlet_phi),
        "qout_m3_per_s": sum(outlet_phi),
        "mdot_in_kg_per_s": -sum(a * b for a, b in zip(inlet_rho, inlet_phi)),
        "mdot_out_kg_per_s": sum(a * b for a, b in zip(outlet_rho, outlet_phi)),
    }


def parse_probe_file(path: Path, vector: bool) -> tuple[int, list[tuple[float, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError as exc:
        raise AuditInputError(f"cannot read {path}: {exc}") from exc

    probe_indices: list[int] = []
    samples: list[tuple[float, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        probe = re.match(r"\s*#\s*Probe\s+(\d+)\b", line)
        if probe:
            probe_indices.append(int(probe.group(1)))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        time_match = re.match(rf"^({FLOAT_TOKEN})\s+(.*)$", stripped)
        if not time_match:
            raise AuditInputError(f"{path}:{line_number}: malformed probe row")
        time_value = _as_float(time_match.group(1), f"{path}:{line_number} time")
        payload = time_match.group(2)
        if vector:
            groups = re.findall(r"\(([^()]*)\)", payload)
            row: list[list[float]] = []
            for group in groups:
                values = [
                    _as_float(token, f"{path}:{line_number} vector")
                    for token in FLOAT_RE.findall(group)
                ]
                if len(values) != 3:
                    raise AuditInputError(
                        f"{path}:{line_number}: vector must have three components"
                    )
                row.append(values)
        else:
            row = [
                _as_float(token, f"{path}:{line_number} scalar")
                for token in FLOAT_RE.findall(payload)
            ]
        samples.append((time_value, row))

    if not probe_indices:
        raise AuditInputError(f"{path}: no '# Probe N' headers")
    probe_count = max(probe_indices) + 1
    if sorted(set(probe_indices)) != list(range(probe_count)):
        raise AuditInputError(f"{path}: non-contiguous probe indices {probe_indices}")
    for time_value, row in samples:
        if len(row) != probe_count:
            raise AuditInputError(
                f"{path}: time {time_value} has {len(row)} values for {probe_count} probes"
            )
    return probe_count, samples


def merge_probe_segments(
    case_dir: Path, field: str, vector: bool
) -> tuple[int, list[tuple[float, Any]], list[str]]:
    root = case_dir / "postProcessing" / "probesJHR"
    if not root.is_dir():
        raise AuditInputError(f"missing probes directory: {root}")
    segments: list[tuple[float, Path]] = []
    for child in root.iterdir():
        value = _numeric_name(child)
        if value is not None and (child / field).is_file():
            segments.append((value, child))
    segments.sort(key=lambda item: (item[0], item[1].name))
    if not segments:
        raise AuditInputError(f"no numeric probesJHR segments contain {field}")

    merged: dict[float, tuple[float, Any]] = {}
    expected_count: int | None = None
    for _, segment in segments:
        count, samples = parse_probe_file(segment / field, vector=vector)
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise AuditInputError(
                f"{field}: probe count changes from {expected_count} to {count}"
            )
        for time_value, row in samples:
            # A later restart segment intentionally supersedes an identical time.
            merged[round(time_value, 12)] = (time_value, row)
    return (
        int(expected_count),
        sorted(merged.values(), key=lambda item: item[0]),
        [segment.name for _, segment in segments],
    )


def list_flux_time_directories(case_dir: Path) -> list[tuple[float, Path]]:
    by_time: dict[float, Path] = {}
    if not case_dir.is_dir():
        return []
    for child in case_dir.iterdir():
        value = _numeric_name(child)
        if value is None or not (child / "phi").is_file() or not (child / "rho").is_file():
            continue
        previous = by_time.get(value)
        if previous is None or child.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            by_time[value] = child
    return sorted(by_time.items(), key=lambda item: item[0])


def _entry_float(text: str, entry: str, label: str) -> float:
    match = re.search(rf"\b{re.escape(entry)}\s+({FLOAT_TOKEN})\s*;", text)
    if not match:
        raise AuditInputError(f"{label}: missing {entry}")
    return _as_float(match.group(1), f"{label} {entry}")


def inspect_output_capability(meta: dict[str, Path], gate: dict[str, Any]) -> dict[str, Any]:
    launcher = meta["launcher"].read_text(encoding="utf-8")
    template = meta["template"].read_text(encoding="utf-8")
    write_matches = re.findall(
        rf"-entry\s+writeInterval\s+-set\s+({FLOAT_TOKEN})", launcher
    )
    purge_matches = re.findall(r"-entry\s+purgeWrite\s+-set\s+(\d+)", launcher)
    if not write_matches or not purge_matches:
        raise AuditInputError(
            f"{meta['launcher']}: Stage-1 writeInterval/purgeWrite override is missing"
        )
    effective_write = _as_float(write_matches[-1], "runtime writeInterval")
    effective_purge = int(purge_matches[-1])
    max_delta_t = _entry_float(template, "maxDeltaT", str(meta["template"]))
    probes_block = _extract_named_block(template, "probesJHR")
    probe_interval_steps = _entry_float(
        probes_block, "writeInterval", f"{meta['template']} probesJHR"
    )
    probe_spacing = max_delta_t * probe_interval_steps
    fields_match = re.search(r"\bfields\s*\(([^)]*)\)\s*;", probes_block)
    fields = fields_match.group(1).split() if fields_match else []
    coverage = gate["coverage"]
    terminal_window = float(coverage["terminal_window_s"])
    expected_flux = math.floor(terminal_window / effective_write + 1e-9) + 1
    expected_probes = math.floor(terminal_window / probe_spacing + 1e-9) + 1
    checks = {
        "runtime_write_interval_at_most_declared_max": effective_write
        <= float(coverage["maximum_saved_field_interval_s"]) + 1e-12,
        "runtime_purge_write_disabled": effective_purge == 0,
        "nominal_probe_spacing_at_most_declared_max": probe_spacing
        <= float(coverage["declared_probe_max_spacing_s"]) + 1e-12,
        "expected_flux_samples_meet_minimum": expected_flux
        >= int(coverage["minimum_flux_snapshots"]),
        "expected_probe_samples_meet_minimum": expected_probes
        >= int(coverage["minimum_probe_samples"]),
        "required_probe_fields_present": set(
            coverage["required_pressure_fields"] + [coverage["required_velocity_field"]]
        ).issubset(fields),
    }
    return {
        "launcher": str(meta["launcher"]),
        "control_template": str(meta["template"]),
        "runtime_saved_field_interval_s": effective_write,
        "runtime_purge_write": effective_purge,
        "template_max_delta_t_s": max_delta_t,
        "probe_interval_time_steps": int(probe_interval_steps),
        "nominal_max_probe_spacing_s": probe_spacing,
        "probe_fields": fields,
        "expected_terminal_window_flux_samples": expected_flux,
        "expected_terminal_window_probe_samples": expected_probes,
        "checks": checks,
        "sufficient_for_registered_quasi_steady_audit": all(checks.values()),
        "scope_note": "This cadence is sufficient for terminal quasi-steady drift, fluctuation, and balance gates; it is not claimed to resolve fast Stage-2 eruption transients.",
    }


def _series_coverage(
    times: Sequence[float],
    start: float,
    end: float,
    minimum_count: int,
    maximum_gap: float,
    endpoint_tolerance: float,
) -> dict[str, Any]:
    selected = sorted(time for time in times if start - 1e-10 <= time <= end + 1e-10)
    gaps = [second - first for first, second in zip(selected, selected[1:])]
    checks = {
        "minimum_count": len(selected) >= minimum_count,
        "window_start_reached": bool(selected) and selected[0] <= start + endpoint_tolerance,
        "window_end_reached": bool(selected) and selected[-1] >= end - endpoint_tolerance,
        "maximum_gap": bool(selected) and (not gaps or max(gaps) <= maximum_gap + 1e-12),
    }
    return {
        "count": len(selected),
        "first_time_s": selected[0] if selected else None,
        "last_time_s": selected[-1] if selected else None,
        "span_s": selected[-1] - selected[0] if len(selected) >= 2 else 0.0,
        "maximum_gap_s": max(gaps) if gaps else None,
        "checks": checks,
        "passed": all(checks.values()),
        "selected_times_s": selected,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AuditInputError("mean of an empty series")
    return sum(values) / len(values)


def _linear_scalar_metrics(times: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    if len(times) != len(values) or len(times) < 2:
        raise AuditInputError("linear metrics require at least two paired samples")
    mean_t = _mean(times)
    mean_y = _mean(values)
    variance_t = sum((time - mean_t) ** 2 for time in times)
    if variance_t <= 0:
        raise AuditInputError("linear metrics require distinct times")
    slope = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values)) / variance_t
    intercept = mean_y - slope * mean_t
    residuals = [
        value - (intercept + slope * time) for time, value in zip(times, values)
    ]
    midpoint = (times[0] + times[-1]) / 2.0
    first_half = [value for time, value in zip(times, values) if time <= midpoint]
    second_half = [value for time, value in zip(times, values) if time > midpoint]
    if not first_half or not second_half:
        raise AuditInputError("terminal series does not populate both half windows")
    return {
        "mean": mean_y,
        "slope_per_s": slope,
        "half_window_mean_shift": abs(_mean(second_half) - _mean(first_half)),
        "detrended_peak_to_peak": max(residuals) - min(residuals),
    }


def _vector_metrics(times: Sequence[float], vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    if any(len(vector) != 3 for vector in vectors):
        raise AuditInputError("velocity vector does not have three components")
    components = [[vector[index] for vector in vectors] for index in range(3)]
    scalar = [_linear_scalar_metrics(times, component) for component in components]
    slopes = [item["slope_per_s"] for item in scalar]
    midpoint = (times[0] + times[-1]) / 2.0
    first_vectors = [vector for time, vector in zip(times, vectors) if time <= midpoint]
    second_vectors = [vector for time, vector in zip(times, vectors) if time > midpoint]
    first_mean = [_mean([row[index] for row in first_vectors]) for index in range(3)]
    second_mean = [_mean([row[index] for row in second_vectors]) for index in range(3)]

    mean_t = _mean(times)
    component_means = [_mean(component) for component in components]
    intercepts = [mean - slope * mean_t for mean, slope in zip(component_means, slopes)]
    residual_magnitudes = []
    for time, vector in zip(times, vectors):
        residual = [
            value - (intercept + slope * time)
            for value, intercept, slope in zip(vector, intercepts, slopes)
        ]
        residual_magnitudes.append(math.sqrt(sum(value * value for value in residual)))
    return {
        "component_slopes_m_per_s2": slopes,
        "slope_norm_m_per_s2": math.sqrt(sum(value * value for value in slopes)),
        "half_window_mean_vector_change_m_per_s": math.sqrt(
            sum((second - first) ** 2 for first, second in zip(first_mean, second_mean))
        ),
        "maximum_detrended_residual_vector_magnitude_m_per_s": max(
            residual_magnitudes
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise AuditInputError("percentile of an empty series")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    measured: float,
    comparison: str,
    threshold: float,
) -> None:
    if comparison == "<=":
        passed = measured <= threshold + 1e-15
    elif comparison == ">=":
        passed = measured + 1e-15 >= threshold
    else:
        raise ValueError(f"unsupported comparison {comparison}")
    checks.append(
        {
            "id": check_id,
            "measured": measured,
            "comparison": comparison,
            "threshold": threshold,
            "passed": passed,
        }
    )


def calculate_metrics(
    gate: dict[str, Any],
    window_start: float,
    latest: float,
    probe_data: dict[str, list[tuple[float, Any]]],
    flux_snapshots: list[dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thresholds = gate["thresholds"]
    checks: list[dict[str, Any]] = []
    pressure_output: dict[str, Any] = {}
    for field in gate["coverage"]["required_pressure_fields"]:
        selected = [
            (time, values)
            for time, values in probe_data[field]
            if window_start - 1e-10 <= time <= latest + 1e-10
        ]
        times = [item[0] for item in selected]
        count = len(selected[0][1])
        per_probe = []
        for probe in range(count):
            values = [item[1][probe] for item in selected]
            metrics = _linear_scalar_metrics(times, values)
            per_probe.append({"probe": probe, **metrics})
            _check(
                checks,
                f"{field}.probe{probe}.absolute_slope_pa_per_s",
                abs(metrics["slope_per_s"]),
                "<=",
                float(thresholds["pressure"]["maximum_absolute_slope_pa_per_s"]),
            )
            _check(
                checks,
                f"{field}.probe{probe}.half_window_mean_shift_pa",
                metrics["half_window_mean_shift"],
                "<=",
                float(thresholds["pressure"]["maximum_half_window_mean_shift_pa"]),
            )
            _check(
                checks,
                f"{field}.probe{probe}.detrended_peak_to_peak_pa",
                metrics["detrended_peak_to_peak"],
                "<=",
                float(thresholds["pressure"]["maximum_detrended_peak_to_peak_pa"]),
            )
        pressure_output[field] = per_probe

    selected_u = [
        (time, values)
        for time, values in probe_data[gate["coverage"]["required_velocity_field"]]
        if window_start - 1e-10 <= time <= latest + 1e-10
    ]
    u_times = [item[0] for item in selected_u]
    velocity_output = []
    for probe in range(len(selected_u[0][1])):
        vectors = [item[1][probe] for item in selected_u]
        metrics = _vector_metrics(u_times, vectors)
        velocity_output.append({"probe": probe, **metrics})
        _check(
            checks,
            f"U.probe{probe}.slope_norm_m_per_s2",
            metrics["slope_norm_m_per_s2"],
            "<=",
            float(thresholds["velocity"]["maximum_slope_norm_m_per_s2"]),
        )
        _check(
            checks,
            f"U.probe{probe}.half_window_mean_vector_change_m_per_s",
            metrics["half_window_mean_vector_change_m_per_s"],
            "<=",
            float(
                thresholds["velocity"]["maximum_half_window_mean_vector_change_m_per_s"]
            ),
        )
        _check(
            checks,
            f"U.probe{probe}.maximum_detrended_residual_vector_magnitude_m_per_s",
            metrics["maximum_detrended_residual_vector_magnitude_m_per_s"],
            "<=",
            float(
                thresholds["velocity"]
                ["maximum_detrended_residual_vector_magnitude_m_per_s"]
            ),
        )

    flux_snapshots = sorted(flux_snapshots, key=lambda item: item["time_s"])
    flux_times = [item["time_s"] for item in flux_snapshots]
    boundary_output: dict[str, Any] = {}
    flow_specs = (
        ("qin_m3_per_s", "volume_flow_denominator_floor_m3_per_s"),
        ("qout_m3_per_s", "volume_flow_denominator_floor_m3_per_s"),
        ("mdot_in_kg_per_s", "mass_flow_denominator_floor_kg_per_s"),
        ("mdot_out_kg_per_s", "mass_flow_denominator_floor_kg_per_s"),
    )
    for field, floor_key in flow_specs:
        values = [item[field] for item in flux_snapshots]
        raw = _linear_scalar_metrics(flux_times, values)
        denominator = max(abs(raw["mean"]), float(thresholds["boundary_flow"][floor_key]))
        metrics = {
            **raw,
            "relative_half_window_mean_change": raw["half_window_mean_shift"]
            / denominator,
            "relative_detrended_peak_to_peak": raw["detrended_peak_to_peak"]
            / denominator,
        }
        boundary_output[field] = metrics
        _check(
            checks,
            f"{field}.mean_forward_flow",
            raw["mean"],
            ">=",
            float(thresholds["boundary_flow"][floor_key]),
        )
        _check(
            checks,
            f"{field}.relative_half_window_mean_change",
            metrics["relative_half_window_mean_change"],
            "<=",
            float(
                thresholds["boundary_flow"]["maximum_half_window_relative_mean_change"]
            ),
        )
        _check(
            checks,
            f"{field}.relative_detrended_peak_to_peak",
            metrics["relative_detrended_peak_to_peak"],
            "<=",
            float(
                thresholds["boundary_flow"]["maximum_detrended_peak_to_peak_fraction"]
            ),
        )

    q_floor = float(
        thresholds["boundary_flow"]["volume_flow_denominator_floor_m3_per_s"]
    )
    m_floor = float(
        thresholds["boundary_flow"]["mass_flow_denominator_floor_kg_per_s"]
    )
    volume_imbalance = []
    mass_imbalance = []
    for item in flux_snapshots:
        q_scale = max(
            0.5 * (abs(item["qin_m3_per_s"]) + abs(item["qout_m3_per_s"])),
            q_floor,
        )
        m_scale = max(
            0.5
            * (abs(item["mdot_in_kg_per_s"]) + abs(item["mdot_out_kg_per_s"])),
            m_floor,
        )
        volume_imbalance.append(
            abs(item["qin_m3_per_s"] - item["qout_m3_per_s"]) / q_scale
        )
        mass_imbalance.append(
            abs(item["mdot_in_kg_per_s"] - item["mdot_out_kg_per_s"]) / m_scale
        )

    balance_output = {}
    for name, values in (
        ("volume_flow", volume_imbalance),
        ("mass_flow", mass_imbalance),
    ):
        metrics = {
            "mean_relative_imbalance": _mean(values),
            "p95_instantaneous_relative_imbalance": _percentile(values, 0.95),
        }
        balance_output[name] = metrics
        _check(
            checks,
            f"{name}.mean_relative_imbalance",
            metrics["mean_relative_imbalance"],
            "<=",
            float(thresholds["balance"]["maximum_mean_relative_imbalance"]),
        )
        _check(
            checks,
            f"{name}.p95_instantaneous_relative_imbalance",
            metrics["p95_instantaneous_relative_imbalance"],
            "<=",
            float(
                thresholds["balance"]["maximum_p95_instantaneous_relative_imbalance"]
            ),
        )

    return (
        {
            "pressure_probes": pressure_output,
            "velocity_probes": velocity_output,
            "boundary_flows": boundary_output,
            "inlet_outlet_balance": balance_output,
            "flux_snapshot_series": flux_snapshots,
        },
        checks,
    )


def recommended_next_endpoint(
    gate: dict[str, Any], latest: float | None, stable: bool
) -> float | None:
    if stable:
        return None
    policy = gate["decision_policy"]
    minimum_time = float(gate["coverage"]["minimum_stage1_time_s"])
    extension = float(policy["minimum_extension_s_when_inconclusive_or_unstable"])
    rounding = float(policy["recommended_endpoint_rounding_s"])
    raw = minimum_time if latest is None else max(minimum_time, latest + extension)
    return math.ceil((raw - 1e-12) / rounding) * rounding


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_level(level: str, gate: dict[str, Any], gate_sha256: str) -> dict[str, Any]:
    meta = LEVELS[level]
    case_dir = meta["case"]
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "schema_version": "case3_stage1_stability_audit_v1",
        "generated_at_local": generated,
        "level": level,
        "case_directory": str(case_dir),
        "gate_file": str(GATE_PATH),
        "gate_sha256": gate_sha256,
        "read_only_scope": {
            "openfoam_started_by_auditor": False,
            "stage1_markers_touched_by_auditor": False,
            "allowed_output": str(case_dir / "stage1_stability_audit.json"),
        },
        "status": "INCONCLUSIVE",
        "automatic_gates_passed": False,
        "reasons": [],
        "metrics": None,
        "gate_checks": [],
    }

    errors: list[str] = []
    try:
        capability = inspect_output_capability(meta, gate)
    except (OSError, AuditInputError) as exc:
        capability = {"sufficient_for_registered_quasi_steady_audit": False, "error": str(exc)}
        errors.append(str(exc))
    result["output_capability"] = capability

    flux_dirs = list_flux_time_directories(case_dir)
    latest = flux_dirs[-1][0] if flux_dirs else None
    coverage_gate = gate["coverage"]
    terminal_window = float(coverage_gate["terminal_window_s"])
    window_start = latest - terminal_window if latest is not None else None
    minimum_time_passed = latest is not None and latest + 1e-12 >= float(
        coverage_gate["minimum_stage1_time_s"]
    )

    flux_coverage = _series_coverage(
        [item[0] for item in flux_dirs],
        window_start if window_start is not None else -terminal_window,
        latest if latest is not None else 0.0,
        int(coverage_gate["minimum_flux_snapshots"]),
        float(coverage_gate["maximum_flux_gap_s"]),
        float(coverage_gate["flux_window_endpoint_tolerance_s"]),
    )

    probe_data: dict[str, list[tuple[float, Any]]] = {}
    probe_coverage: dict[str, Any] = {}
    probe_segments: dict[str, list[str]] = {}
    required_fields = coverage_gate["required_pressure_fields"] + [
        coverage_gate["required_velocity_field"]
    ]
    for field in required_fields:
        try:
            count, samples, segments = merge_probe_segments(
                case_dir, field, vector=field == coverage_gate["required_velocity_field"]
            )
            probe_data[field] = samples
            probe_segments[field] = segments
            report = _series_coverage(
                [item[0] for item in samples],
                window_start if window_start is not None else -terminal_window,
                latest if latest is not None else 0.0,
                int(coverage_gate["minimum_probe_samples"]),
                float(coverage_gate["maximum_probe_gap_s"]),
                float(coverage_gate["probe_window_endpoint_tolerance_s"]),
            )
            report["probe_count"] = count
            report["required_probe_count_passed"] = count == int(
                coverage_gate["required_probe_count"]
            )
            report["passed"] = report["passed"] and report[
                "required_probe_count_passed"
            ]
            probe_coverage[field] = report
        except AuditInputError as exc:
            probe_coverage[field] = {"passed": False, "error": str(exc)}
            errors.append(str(exc))

    coverage_checks = {
        "minimum_physical_time_reached": minimum_time_passed,
        "flux_terminal_window_complete": flux_coverage["passed"],
        "all_probe_terminal_windows_complete": bool(probe_coverage)
        and all(item.get("passed", False) for item in probe_coverage.values()),
        "output_capability_sufficient": bool(
            capability.get("sufficient_for_registered_quasi_steady_audit", False)
        ),
    }
    coverage_passed = all(coverage_checks.values()) and not errors
    result["coverage"] = {
        "latest_saved_time_s": latest,
        "required_minimum_stage1_time_s": float(
            coverage_gate["minimum_stage1_time_s"]
        ),
        "terminal_window_requested_s": (
            [window_start, latest] if latest is not None else None
        ),
        "flux": {key: value for key, value in flux_coverage.items() if key != "selected_times_s"},
        "probes": probe_coverage,
        "probe_segments_read": probe_segments,
        "checks": coverage_checks,
        "passed": coverage_passed,
    }

    if not coverage_passed:
        if not minimum_time_passed:
            result["reasons"].append(
                "The declared minimum physical coverage of "
                f"{float(coverage_gate['minimum_stage1_time_s']):g} s has not been "
                "reached; no stability metrics are evaluated."
            )
        for name, passed in coverage_checks.items():
            if not passed and name != "minimum_physical_time_reached":
                result["reasons"].append(f"Coverage/input gate failed: {name}")
        result["reasons"].extend(errors)
        result["recommended_next_segment_end_s"] = recommended_next_endpoint(
            gate, latest, stable=False
        )
        result["decision_note"] = (
            "INCONCLUSIVE is not evidence of steady or unsteady flow. A segment endpoint is never an automatic steady-state acceptance."
        )
        return result

    try:
        selected_flux_dirs = [
            (time, directory)
            for time, directory in flux_dirs
            if window_start - 1e-10 <= time <= latest + 1e-10
        ]
        flux_snapshots = [
            parse_flux_snapshot(time, directory) for time, directory in selected_flux_dirs
        ]
        metrics, checks = calculate_metrics(
            gate, window_start, latest, probe_data, flux_snapshots
        )
        result["metrics"] = metrics
        result["gate_checks"] = checks
        stable = all(check["passed"] for check in checks)
        result["automatic_gates_passed"] = stable
        if stable:
            result["status"] = "STABLE_CANDIDATE_REQUIRES_MANUAL_ACCEPTANCE"
            result["decision_note"] = (
                "All pre-registered automatic gates pass. The auditor still does not create STAGE1_COMPLETE or STAGE1_ACCEPTED."
            )
        else:
            result["status"] = "UNSTABLE"
            result["reasons"] = [
                check["id"] for check in checks if not check["passed"]
            ]
            result["decision_note"] = (
                "Coverage is complete, but one or more frozen physical-stability gates fail."
            )
        result["recommended_next_segment_end_s"] = recommended_next_endpoint(
            gate, latest, stable=stable
        )
    except AuditInputError as exc:
        result["status"] = "INCONCLUSIVE"
        result["reasons"] = [f"Terminal-window data could not be evaluated: {exc}"]
        result["recommended_next_segment_end_s"] = recommended_next_endpoint(
            gate, latest, stable=False
        )
        result["decision_note"] = "Malformed or incomplete inputs cannot be interpreted as instability."
    return result


def load_gate() -> tuple[dict[str, Any], str]:
    raw = GATE_PATH.read_bytes()
    gate = json.loads(raw.decode("utf-8"))
    if gate.get("schema_version") != "case3_stage1_stability_gate_v1":
        raise AuditInputError("unexpected Stage-1 stability gate schema")
    if gate.get("decision_policy", {}).get("automatic_stage1_marker_creation") is not False:
        raise AuditInputError("gate must prohibit automatic Stage-1 marker creation")
    return gate, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=["all", *LEVELS],
        default="all",
        help="mesh level to audit (default: all)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print results without writing stage1_stability_audit.json",
    )
    args = parser.parse_args()
    gate, digest = load_gate()
    selected = LEVELS if args.level == "all" else {args.level: LEVELS[args.level]}
    summaries = []
    for level in selected:
        result = audit_level(level, gate, digest)
        if not args.no_write:
            _atomic_json(LEVELS[level]["case"] / "stage1_stability_audit.json", result)
        summaries.append(
            {
                "level": level,
                "status": result["status"],
                "latest_saved_time_s": result["coverage"]["latest_saved_time_s"],
                "recommended_next_segment_end_s": result.get(
                    "recommended_next_segment_end_s"
                ),
                "output": (
                    None
                    if args.no_write
                    else str(LEVELS[level]["case"] / "stage1_stability_audit.json")
                ),
            }
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
